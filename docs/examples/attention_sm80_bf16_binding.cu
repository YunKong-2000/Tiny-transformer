// Standalone example binding; not registered as the production student operator.
#include <torch/extension.h>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>
#include <tuple>

#include "attention_sm80_bf16_pipeline.cuh"

namespace {
torch::Tensor aligned_contiguous(torch::Tensor x) {
  x = x.contiguous();
  // A contiguous view may still start at an odd BF16 storage offset.
  if (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 != 0)
    x = x.clone();
  return x;
}

std::tuple<torch::Tensor, torch::Tensor>
forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, int64_t past_len) {
  namespace ex = attention_sm80_example;
  for (const auto& x : {q, k, v}) {
    TORCH_CHECK(x.is_cuda(), "example requires CUDA inputs");
    TORCH_CHECK(x.device() == q.device(), "inputs must share a CUDA device");
    TORCH_CHECK(x.layout() == at::kStrided && x.dim() == 4,
                "inputs must be strided [B,H,T,D] tensors");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "example requires BF16 inputs");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0 && x.size(3) == ex::DH,
                "positive B/H/T and head_dim=64 are required");
    TORCH_CHECK(!(at::GradMode::is_enabled() && x.requires_grad()),
                "example has no backward; use no_grad for inference");
  }
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
              q.size(1) == k.size(1) && q.size(1) == v.size(1),
              "batch and heads must match (no GQA)");
  const int64_t tq = q.size(2), tk = k.size(2);
  TORCH_CHECK(v.size(2) == tk && past_len >= 0 && tk >= tq && past_len == tk - tq,
              "expected Tk=Tv=past_len+Tq");
  const int64_t q_tiles = (tq - 1) / ex::BQ + 1;
  const int64_t bh = q.size(0) * q.size(1);
  TORCH_CHECK(q_tiles <= std::numeric_limits<int>::max() &&
              bh <= std::numeric_limits<int>::max() / q_tiles,
              "example grid exceeds INT_MAX");

  const c10::cuda::CUDAGuard guard(q.device());
  cudaDeviceProp props;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&props, q.get_device()));
  TORCH_CHECK(props.major >= 8, "BF16 MMA/cp.async require SM80 or later");
  q = aligned_contiguous(q);
  k = aligned_contiguous(k);
  v = aligned_contiguous(v);
  auto out = torch::empty(q.sizes(), q.options());
  auto lse = torch::empty({q.size(0), q.size(1), tq}, q.options().dtype(at::kFloat));
  const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
  ex::forward<<<static_cast<unsigned>(bh * q_tiles), ex::THREADS,
                sizeof(ex::SharedStorage), stream>>>(
      reinterpret_cast<const ex::Element*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const ex::Element*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const ex::Element*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<ex::Element*>(out.data_ptr<at::BFloat16>()),
      lse.data_ptr<float>(), tq, tk, past_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "SM80 BF16 causal attention template (forward only)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("past_len") = 0);
}
