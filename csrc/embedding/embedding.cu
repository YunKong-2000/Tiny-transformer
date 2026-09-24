#include "embedding.h"

#include <algorithm>
#include <cstdint>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int kVectorWidth = 4;

__global__ void embedding_forward_kernel_vector(
    int64_t rows, int64_t vocab_size, int64_t dim,
    const int64_t* ids, const float* weight, float* output) {
  const int64_t warp_id =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpSize;
  const int64_t warp_count = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
  const int lane = threadIdx.x % kWarpSize;

  for (int64_t row = warp_id; row < rows; row += warp_count) {
    const int64_t index = ids[row];
    // Asynchronous device assertion: never dereference an invalid row.
    CUDA_KERNEL_ASSERT(index >= 0 && index < vocab_size);
    if (index < 0 || index >= vocab_size) continue;

    // The wrapper guarantees aligned row starts and dim divisible by four.
    const auto* src = reinterpret_cast<const float4*>(weight + index * dim);
    auto* dst = reinterpret_cast<float4*>(output + row * dim);
    for (int64_t j = lane; j < dim / kVectorWidth; j += kWarpSize) {
      dst[j] = src[j];
    }
  }
}

__global__ void embedding_forward_kernel_scalar(
    int64_t rows, int64_t vocab_size, int64_t dim,
    const int64_t* ids, const float* weight, float* output) {
  const int64_t warp_id =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpSize;
  const int64_t warp_count = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
  const int lane = threadIdx.x % kWarpSize;

  for (int64_t row = warp_id; row < rows; row += warp_count) {
    const int64_t index = ids[row];
    CUDA_KERNEL_ASSERT(index >= 0 && index < vocab_size);
    if (index < 0 || index >= vocab_size) continue;

    const float* src = weight + index * dim;
    float* dst = output + row * dim;
    for (int64_t j = lane; j < dim; j += kWarpSize) {
      dst[j] = src[j];
    }
  }
}

}  // namespace

torch::Tensor embedding_forward_cuda(torch::Tensor ids, torch::Tensor weight) {
  TORCH_CHECK(ids.is_cuda() && weight.is_cuda(),
              "ids and weight must be CUDA tensors");
  TORCH_CHECK(ids.device() == weight.device(),
              "ids and weight must be on the same CUDA device");
  TORCH_CHECK(ids.layout() == at::kStrided && weight.layout() == at::kStrided,
              "ids and weight must have strided layout");
  TORCH_CHECK(ids.dim() == 2 && weight.dim() == 2,
              "ids and weight must be 2D tensors");
  TORCH_CHECK(ids.scalar_type() == at::kLong, "ids must have dtype int64");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must have dtype float32");
  TORCH_CHECK(ids.is_contiguous() && weight.is_contiguous(),
              "ids and weight must be contiguous");
  TORCH_CHECK(!(at::GradMode::is_enabled() && weight.requires_grad()),
              "student embedding is forward-only; use torch.no_grad() or "
              "torch.inference_mode(), or select the reference backend for training");

  const int64_t batch = ids.size(0);
  const int64_t length = ids.size(1);
  const int64_t vocab_size = weight.size(0);
  const int64_t dim = weight.size(1);
  TORCH_CHECK(vocab_size > 0 && dim > 0,
              "weight dimensions V and H must be positive");

  const c10::cuda::CUDAGuard device_guard(weight.device());
  auto output = torch::empty({batch, length, dim}, weight.options());
  const int64_t rows = ids.numel();
  if (rows == 0) return output;

  // Each block processes eight IDs, not 256 IDs. Cap the grid and let the
  // warp-stride loop handle very large batches without overflowing grid.x.
  const int blocks = static_cast<int>(std::min<int64_t>(
      (rows - 1) / kWarpsPerBlock + 1, 65535));
  const auto stream = c10::cuda::getCurrentCUDAStream(weight.get_device());
  const auto weight_address =
      reinterpret_cast<std::uintptr_t>(weight.data_ptr<float>());
  const auto output_address =
      reinterpret_cast<std::uintptr_t>(output.data_ptr<float>());
  const bool vectorized = dim % kVectorWidth == 0 &&
      weight_address % alignof(float4) == 0 &&
      output_address % alignof(float4) == 0;

  if (vectorized) {
    embedding_forward_kernel_vector<<<blocks, kThreads, 0, stream>>>(
        rows, vocab_size, dim, ids.data_ptr<int64_t>(),
        weight.data_ptr<float>(), output.data_ptr<float>());
  } else {
    embedding_forward_kernel_scalar<<<blocks, kThreads, 0, stream>>>(
        rows, vocab_size, dim, ids.data_ptr<int64_t>(),
        weight.data_ptr<float>(), output.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
