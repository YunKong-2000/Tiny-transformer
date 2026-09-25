#include "embedding.h"

#include <algorithm>
#include <cstdint>
#include <ATen/Context.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr unsigned kFullWarp = 0xffffffffu;
constexpr int kVector = 4;

__global__ void embedding_backward_kernel_scalar(
    int64_t rows, int64_t vocab_size, int64_t dim,
    const int64_t* ids, const float* gradient, float* output) {
  const int lane = threadIdx.x % kWarpSize;
  const int64_t first_warp_base =
      static_cast<int64_t>(blockIdx.x) * blockDim.x +
      (threadIdx.x / kWarpSize) * kWarpSize;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  // All 32 lanes cooperate on channels, even when the last warp has fewer IDs.
  // Keep every collective outside the possibly divergent channel loop.
  for (int64_t warp_base = first_warp_base; warp_base < rows;
       warp_base += stride) {
    const int64_t row = warp_base + lane;
    const int64_t index = row < rows ? ids[row] : -1;
    CUDA_KERNEL_ASSERT(row >= rows || (index >= 0 && index < vocab_size));
    const bool valid = row < rows && index >= 0 && index < vocab_size;

    // Older targets lack match.any; discover each group with ballot instead.
    unsigned leaders = __ballot_sync(kFullWarp, valid);
    while (leaders) {
      const int leader_lane = __ffs(leaders) - 1;
      const int64_t token = __shfl_sync(kFullWarp, index, leader_lane);
      const unsigned group = __ballot_sync(kFullWarp, valid && index == token);
      for (int64_t channel = lane; channel < dim; channel += kWarpSize) {
        float sum = 0.0f;
        unsigned members = group;
        while (members) {
          const int member = __ffs(members) - 1;
          sum += gradient[(warp_base + member) * dim + channel];
          members &= members - 1;
        }
        // Other warps/blocks may contribute to the same token row.
        atomicAdd(output + token * dim + channel, sum);
      }
      leaders &= ~group;
    }
  }
}

__global__ void embedding_backward_baseline_kernel_scalar(
    int64_t rows, int64_t vocab_size, int64_t dim,
    const int64_t* ids, const float* gradient, float* output) {
  const int64_t warp_id =
      static_cast<int64_t>(blockIdx.x) * (blockDim.x / kWarpSize) +
      threadIdx.x / kWarpSize;
  const int64_t warp_count =
      static_cast<int64_t>(gridDim.x) * (blockDim.x / kWarpSize);
  const int lane = threadIdx.x % kWarpSize;
  for (int64_t row = warp_id; row < rows; row += warp_count) {
    const int64_t index = ids[row];
    CUDA_KERNEL_ASSERT(index >= 0 && index < vocab_size);
    if (index < 0 || index >= vocab_size) continue;
    for (int64_t channel = lane; channel < dim; channel += kWarpSize) {
      atomicAdd(output + index * dim + channel, gradient[row * dim + channel]);
    }
  }
}

__global__ void embedding_backward_baseline_kernel_vector(
    int64_t rows, int64_t vocab_size, int64_t dim,
    const int64_t* ids, const float* gradient, float* output) {
  const int64_t warp_id =
      static_cast<int64_t>(blockIdx.x) * (blockDim.x / kWarpSize) +
      threadIdx.x / kWarpSize;
  const int64_t warp_count =
      static_cast<int64_t>(gridDim.x) * (blockDim.x / kWarpSize);
  const int lane = threadIdx.x % kWarpSize;
  const int warp_stride = kWarpSize * kVector;
  for (int64_t row = warp_id; row < rows; row += warp_count) {
    const int64_t index = ids[row];
    CUDA_KERNEL_ASSERT(index >= 0 && index < vocab_size);
    if (index < 0 || index >= vocab_size) continue;
    // channel already counts float elements, not float4 vectors.
    for (int64_t channel = static_cast<int64_t>(lane) * kVector; channel < dim; channel += warp_stride) {
      const float4 grad = *reinterpret_cast<const float4*>(&gradient[row * dim + channel]);
      atomicAdd(output + index * dim + channel + 0, grad.x);
      atomicAdd(output + index * dim + channel + 1, grad.y);
      atomicAdd(output + index * dim + channel + 2, grad.z);
      atomicAdd(output + index * dim + channel + 3, grad.w);
    }
  }
 }

}  // namespace

torch::Tensor embedding_backward_cuda(
    torch::Tensor ids, torch::Tensor gradient, int64_t vocab_size,
    const std::string& implementation) {
  TORCH_CHECK(implementation == "grouped" || implementation == "baseline",
              "embedding backward implementation must be 'grouped' or 'baseline'");
  TORCH_CHECK(ids.is_cuda() && gradient.is_cuda(),
              "ids and gradient must be CUDA tensors");
  TORCH_CHECK(ids.device() == gradient.device(),
              "ids and gradient must be on the same CUDA device");
  TORCH_CHECK(ids.layout() == at::kStrided && gradient.layout() == at::kStrided,
              "ids and gradient must have strided layout");
  TORCH_CHECK(ids.dim() == 2 && gradient.dim() == 3,
              "ids must be 2D and gradient must be 3D");
  TORCH_CHECK(ids.size(0) == gradient.size(0) && ids.size(1) == gradient.size(1),
              "ids and gradient must have the same first two dimensions");
  TORCH_CHECK(ids.scalar_type() == at::kLong, "ids must have dtype int64");
  TORCH_CHECK(gradient.scalar_type() == at::kFloat,
              "gradient must have dtype float32");
  TORCH_CHECK(ids.is_contiguous() && gradient.is_contiguous(),
              "ids and gradient must be contiguous");
  const int64_t dim = gradient.size(2);
  TORCH_CHECK(vocab_size > 0 && dim > 0,
              "vocab_size and gradient dimension H must be positive");

  const c10::cuda::CUDAGuard device_guard(gradient.device());
  auto output = torch::zeros({vocab_size, dim}, gradient.options());
  const int64_t rows = ids.numel();
  if (rows == 0) return output;
  // Floating atomicAdd order across warps is not deterministic.
  at::globalContext().alertNotDeterministic("embedding_backward_cuda");

  // Baseline: eight warps => eight rows per block. Grouped: 256 rows per block.
  const bool baseline = implementation == "baseline";
  const int rows_per_block = baseline ? kThreads / kWarpSize : kThreads;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (rows - 1) / rows_per_block + 1, 65535));
  const auto stream = c10::cuda::getCurrentCUDAStream(gradient.get_device());
  if (baseline) {
    const auto grad_addr = reinterpret_cast<std::uintptr_t>(gradient.data_ptr<float>());
    bool vectorized = (dim % kVector == 0) && (grad_addr % alignof(float4) == 0);
    if (vectorized) {
      embedding_backward_baseline_kernel_vector<<<blocks, kThreads, 0, stream>>>(
        rows, vocab_size, dim, ids.data_ptr<int64_t>(),
        gradient.data_ptr<float>(), output.data_ptr<float>());
    }
    else {
      embedding_backward_baseline_kernel_scalar<<<blocks, kThreads, 0, stream>>>(
        rows, vocab_size, dim, ids.data_ptr<int64_t>(),
        gradient.data_ptr<float>(), output.data_ptr<float>());
    }
    
  } else {
    embedding_backward_kernel_scalar<<<blocks, kThreads, 0, stream>>>(
        rows, vocab_size, dim, ids.data_ptr<int64_t>(),
        gradient.data_ptr<float>(), output.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
