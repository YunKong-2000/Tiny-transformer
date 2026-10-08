
#include <optional>
#include <limits>
#include <cstdint>
#include <type_traits>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAException.h>
#include "c10/cuda/CUDAStream.h"
#include "c10/cuda/CUDAGuard.h"

#include "attention.h"
#include "attention_common.h"
#include "attention_forward_kernel.cuh"
#include "attention_forward_BF16_kernel.cuh"
#include "attention_decode_BF16_kernel.cuh"

namespace {
torch::Tensor aligned_contiguous(torch::Tensor x) {
  x = x.contiguous();
  // contiguous() preserves already-contiguous views with an unaligned offset.
  if (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 != 0) {
    x = x.clone();
  }
  return x;
}
} // namespace

std::tuple<torch::Tensor, torch::Tensor>
attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                  int64_t past_len, std::optional<torch::Tensor> segment_ids)
{
  attention_detail::check_qkv(q, k, v);
  attention_detail::check_mask_inputs(q, k, v, past_len, segment_ids);
  TORCH_CHECK(!(at::GradMode::is_enabled() &&
                (q.requires_grad() || k.requires_grad() || v.requires_grad())),
              "student attention backward is not implemented; use no_grad for inference");
  const int64_t B = q.size(0);
  const int64_t Nh = q.size(1);
  const int64_t Tq = q.size(2);
  const int64_t Tk = k.size(2);
  const bool is_bf16 = q.scalar_type() == at::kBFloat16;
  const int64_t tile_rows = is_bf16 ? attention_bf16::BQ : attention_detail::BQ;
  const int64_t q_tiles = (Tq - 1) / tile_rows + 1;
  const int64_t max_grid_x = std::numeric_limits<int>::max();
  if (is_bf16) {
    TORCH_CHECK(!segment_ids.has_value(),
                "segment_ids is not supported for bf16 kernel");
    // Divide before multiplying: the BF16 kernel flattens all CTAs into grid.x.
    TORCH_CHECK(q_tiles <= max_grid_x && B <= max_grid_x / q_tiles / Nh,
                "attention shape exceeds the supported CUDA grid limits");
    // check_mask_inputs established 0 < Tq <= Tk and past_len == Tk - Tq.
    // Reject before contiguous copies/output allocation or any narrowing cast.
    TORCH_CHECK(Tk <= std::numeric_limits<int32_t>::max(),
                "bf16 attention sequence lengths must fit int32 (Tk <= INT32_MAX)");
  } else {
    TORCH_CHECK(q_tiles <= max_grid_x && B <= 65535 / Nh,
                "attention shape exceeds the supported CUDA grid limits");
  }

  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());

  // Q/K are contiguous inside the kernel. This also handles cache prefixes,
  // arbitrary input views and their storage offsets on the current stream.
  if (is_bf16) {
    // PyTorch caches properties per device; avoid a runtime query on every call.
    const auto* props = at::cuda::getDeviceProperties(q.get_device());
    TORCH_CHECK(props->major >= 8, "BF16 MMA/cp.async require SM80 or later");
    q = aligned_contiguous(q);
    k = aligned_contiguous(k);
    v = aligned_contiguous(v);
  } else {
    q = q.contiguous();
    k = k.contiguous();
  }
  auto o = torch::empty(q.sizes(), q.options());
  auto lse = torch::empty({B, Nh, Tq}, q.options().dtype(at::kFloat));

  if (q.scalar_type() == at::kFloat) {
    if (segment_ids.has_value()) {
      segment_ids = segment_ids->contiguous();
    }
    attention_detail::Stride4D V_stride;
    V_stride.b = v.stride(0);
    V_stride.h = v.stride(1);
    V_stride.t = v.stride(2);
    V_stride.d = v.stride(3);
    dim3 Blocks(static_cast<unsigned>(q_tiles), static_cast<unsigned>(B * Nh));
    attention_detail::forward<64><<<Blocks, attention_detail::THREADS, 0, stream>>>
    (
      B,
      Nh,
      Tq,
      Tk,
      V_stride,
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      segment_ids.has_value() ? segment_ids->data_ptr<int64_t>() : nullptr,
      past_len,
      o.data_ptr<float>(),
      lse.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  else {
    static_assert(sizeof(at::BFloat16) == sizeof(attention_bf16::Element));
    if (Tq == 1 && Tk <= attention_bf16::DECODE_MAX_KEYS) {
      namespace bf = attention_bf16;
      const int splits = int((Tk - 1) / bf::DECODE_KEYS + 1);
      auto qp = reinterpret_cast<const bf::Element*>(q.data_ptr<at::BFloat16>());
      auto kp = reinterpret_cast<const bf::Element*>(k.data_ptr<at::BFloat16>());
      auto vp = reinterpret_cast<const bf::Element*>(v.data_ptr<at::BFloat16>());
      auto op = reinterpret_cast<bf::Element*>(o.data_ptr<at::BFloat16>());
      // bh uses grid.x; splitting uses grid.y (at most 32), so existing grid
      // bounds remain sufficient even for many batch/head pairs.
      dim3 blocks(static_cast<unsigned>(B * Nh), static_cast<unsigned>(splits));
      if (splits == 1) {
        bf::decode_partial<true><<<blocks, bf::DECODE_THREADS, 0, stream>>>(
            qp, kp, vp, op, lse.data_ptr<float>(), nullptr, Tk, splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        auto partials = torch::empty({B, Nh, splits, bf::DECODE_PARTIAL},
                                    q.options().dtype(at::kFloat));
        bf::decode_partial<false><<<blocks, bf::DECODE_THREADS, 0, stream>>>(
            qp, kp, vp, op, lse.data_ptr<float>(), partials.data_ptr<float>(), Tk, splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        bf::decode_merge<<<static_cast<unsigned>(B * Nh), bf::DH, 0, stream>>>(
            partials.data_ptr<float>(), op, lse.data_ptr<float>(), splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      return {o, lse};
    }
    dim3 Blocks(static_cast<unsigned>(B * Nh * q_tiles));
    auto launch_forward = [&](auto even_tiles) {
      attention_bf16::forward<decltype(even_tiles)::value><<<Blocks, attention_bf16::THREADS,
                              sizeof(attention_bf16::SharedStorage), stream>>>(
          reinterpret_cast<const attention_bf16::Element*>(q.data_ptr<at::BFloat16>()),
          reinterpret_cast<const attention_bf16::Element*>(k.data_ptr<at::BFloat16>()),
          reinterpret_cast<const attention_bf16::Element*>(v.data_ptr<at::BFloat16>()),
          reinterpret_cast<attention_bf16::Element*>(o.data_ptr<at::BFloat16>()),
          lse.data_ptr<float>(),
          static_cast<int32_t>(Tq),
          static_cast<int32_t>(Tk),
          static_cast<int32_t>(past_len));
    };
    if (Tq % attention_bf16::BQ == 0 && Tk % attention_bf16::BK == 0)
      launch_forward(std::true_type{});
    else
      launch_forward(std::false_type{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {o, lse};
}
