#include "cross_entropy.h"

#include <algorithm>
#include <cstdint>
#include <math_constants.h>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {
constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpNum = kBlockSize / kWarpSize;
constexpr unsigned kFullWarp = 0xffffffffu;

__global__ void cross_entropy_forward_kernel(
    int64_t rows, int64_t dim, const float* logits, const int64_t* targets,
    float* loss, float* lse) {
  const int64_t warp_id = static_cast<int64_t>(blockIdx.x) * kWarpNum + threadIdx.x / kWarpSize;
  const int64_t warp_num = static_cast<int64_t>(gridDim.x) * kWarpNum;
  const int lane_id = threadIdx.x % kWarpSize;
  const int local_warp_id = threadIdx.x / kWarpSize;
  __shared__ float block_sum[kWarpNum];
  for (int64_t row = warp_id; row < rows; row += warp_num) {
    const int64_t target = targets[row];
    if (target == -100) {
      if (lane_id == 0) {
        loss[row] = 0;
        lse[2 * row] = lse[2 * row + 1] = 0.0f;
      }
      continue;
    }
    CUDA_KERNEL_ASSERT(target >= 0 && target < dim);
    const int64_t base = row * dim;
    float local_max = -CUDART_INF_F;
    for (int64_t col = lane_id; col < dim; col += kWarpSize) {
      local_max = fmaxf(local_max, logits[base + col]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      local_max = fmaxf(local_max, __shfl_down_sync(kFullWarp, local_max, offset));
    }
    const float row_max = __shfl_sync(kFullWarp, local_max, 0);
    float local_sum = 0.0f;
    for (int64_t col = lane_id; col < dim; col += kWarpSize) {
      local_sum += expf(logits[base + col] - row_max);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(kFullWarp, local_sum, offset);
    }
    if (lane_id == 0) {
      const float log_sum = logf(local_sum);
      lse[2 * row] = row_max;
      lse[2 * row + 1] = log_sum;
      loss[row] = (row_max - logits[base + target]) + log_sum;
    }
  }
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor>
cross_entropy_forward_cuda(torch::Tensor logits, torch::Tensor targets) {
  check_cross_entropy_inputs(logits, targets);
  TORCH_CHECK(!(at::GradMode::is_enabled() && logits.requires_grad()),
              "cross_entropy_forward has no autograd binding; use student.cross_entropy for training");
  const c10::cuda::CUDAGuard device_guard(logits.device());
  auto loss = torch::empty({logits.size(0), logits.size(1)}, logits.options());
  auto lse = torch::empty({logits.size(0), logits.size(1), 2}, logits.options());
  const int64_t rows = targets.numel();
  if (rows == 0) return {loss, lse};
  const auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());
  const int blocks = static_cast<int>(std::min<int64_t>((rows - 1) / kWarpNum + 1, 65535));
  cross_entropy_forward_kernel<<<blocks, kBlockSize, 0, stream>>>(
      rows, logits.size(2), logits.data_ptr<float>(), targets.data_ptr<int64_t>(),
      loss.data_ptr<float>(), lse.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {loss, lse};
}
