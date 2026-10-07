
#include <optional>
#include <limits>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAException.h>
#include "c10/cuda/CUDAStream.h"
#include "c10/cuda/CUDAGuard.h"

#include "attention.h"
#include "attention_common.h"
#include "attention_forward_kernel.cuh"


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
  const int64_t q_tiles = (Tq - 1) / attention_detail::BQ + 1;
  TORCH_CHECK(q_tiles <= std::numeric_limits<int>::max() && B <= 65535 / Nh,
              "attention shape exceeds the supported CUDA grid limits");

  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = c10::cuda::getCurrentCUDAStream();

  // Q/K are contiguous inside the kernel. This also handles cache prefixes,
  // arbitrary input views and their storage offsets on the current stream.
  q = q.contiguous();
  k = k.contiguous();
  if (segment_ids.has_value()) {
    segment_ids = segment_ids->contiguous();
  }

  auto o = torch::empty(q.sizes(), q.options());
  auto lse = torch::empty({B, Nh, Tq}, q.options());

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
  return {o, lse};
}
