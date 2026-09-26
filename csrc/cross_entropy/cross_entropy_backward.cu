#include "cross_entropy.h"

#include <algorithm>
#include <cstdint>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {
constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpNum = kBlockSize / kWarpSize;

__global__ void cross_entropy_backward_kernel(
    int64_t rows, int64_t dim, const float* logits, const int64_t* targets,
    const float* lse, const float* grad_loss, float* dz) {
  const int64_t warp_id = static_cast<int64_t>(blockIdx.x) * kWarpNum + threadIdx.x / kWarpSize;
  const int64_t warp_num = static_cast<int64_t>(gridDim.x) * kWarpNum;
  const int lane_id = threadIdx.x % kWarpSize;
  for (int64_t row = warp_id; row < rows; row += warp_num) {
    const int64_t target = targets[row];
    CUDA_KERNEL_ASSERT((target >= 0 && target < dim) || target == -100);
    const int64_t base = row * dim;
    for (int64_t col = lane_id; col < dim; col += kWarpSize) {
      // Branch before reading logits/cache/upstream: ignored rows stay zero even
      // with NaN logits or infinite upstream (all-ignored mean divides by zero).
      if (target == -100) {
        dz[base + col] = 0.0f;
      } else {
        const float probability = expf((logits[base + col] - lse[2 * row]) - lse[2 * row + 1]);
        dz[base + col] = (probability - (col == target ? 1.0f : 0.0f)) * grad_loss[0];
      }
    }
  }
}
}  // namespace

torch::Tensor cross_entropy_backward_cuda(torch::Tensor logits, torch::Tensor targets,
                                          torch::Tensor lse, torch::Tensor grad_loss) {
  check_cross_entropy_inputs(logits, targets);
  TORCH_CHECK(lse.is_cuda() && grad_loss.is_cuda(), "lse and grad_loss must be CUDA tensors");
  TORCH_CHECK(logits.device() == lse.device() && logits.device() == grad_loss.device(),
              "all inputs must be on the same CUDA device");
  TORCH_CHECK(lse.layout() == at::kStrided && grad_loss.layout() == at::kStrided,
              "lse and grad_loss must have strided layout");
  TORCH_CHECK(lse.scalar_type() == at::kFloat && grad_loss.scalar_type() == at::kFloat,
              "lse and grad_loss must be float32");
  TORCH_CHECK(lse.dim() == 3 && lse.size(0) == logits.size(0)
              && lse.size(1) == logits.size(1) && lse.size(2) == 2,
              "lse shape must be [B,T,2]");
  TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");
  TORCH_CHECK(grad_loss.dim() == 0, "grad_loss must be a scalar (0D tensor)");
  TORCH_CHECK(!(at::GradMode::is_enabled()
              && (logits.requires_grad() || lse.requires_grad() || grad_loss.requires_grad())),
              "cross_entropy_backward has no autograd binding; only first-order gradients are supported");
  const c10::cuda::CUDAGuard device_guard(logits.device());
  auto dz = torch::empty(logits.sizes(), logits.options());
  const int64_t rows = targets.numel();
  if (rows == 0) return dz;
  const auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());
  const int blocks = static_cast<int>(std::min<int64_t>((rows - 1) / kWarpNum + 1, 65535));
  cross_entropy_backward_kernel<<<blocks, kBlockSize, 0, stream>>>(
      rows, logits.size(2), logits.data_ptr<float>(), targets.data_ptr<int64_t>(),
      lse.data_ptr<float>(), grad_loss.data_ptr<float>(), dz.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dz;
}
