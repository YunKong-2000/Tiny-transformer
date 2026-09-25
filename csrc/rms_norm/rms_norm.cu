#include "rms_norm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullWarp = 0xffffffff;
constexpr int kThreadNum = 256;
constexpr int kWarpNum = kThreadNum / kWarpSize;

__global__ void rms_norm_kernel(
    int64_t rows, int64_t dim, float epsilon,
    const float* X, const float* weight, float* output) {
  const int64_t warp_num = static_cast<int64_t>(gridDim.x) * kWarpNum;
  const int64_t warp_id =
      static_cast<int64_t>(blockIdx.x) * kWarpNum + threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;

  // A whole warp handles one row. Even lanes with no columns participate in
  // every shuffle, so the full mask is valid for H < 32 and odd row widths.
  for (int64_t row = warp_id; row < rows; row += warp_num) {
    float local_sum = 0.0f;
    const int64_t base = row * dim;
    for (int64_t col = lane_id; col < dim; col += kWarpSize) {
      const float x_value = X[base + col];
      local_sum += x_value * x_value;
    }
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(kFullWarp, local_sum, offset);
    }
    const float warp_sum = __shfl_sync(kFullWarp, local_sum, 0);
    const float inv_rms = rsqrtf(warp_sum / static_cast<float>(dim) + epsilon);

    for (int64_t col = lane_id; col < dim; col += kWarpSize) {
      // Normalize before scaling, matching reference.py's operation order.
      output[base + col] = (X[base + col] * inv_rms) * weight[col];
    }
  }
}

}  // namespace

torch::Tensor rms_norm_forward_cuda(torch::Tensor X, torch::Tensor weight, float epsilon) {
  TORCH_CHECK(X.is_cuda() && weight.is_cuda(),
              "X and weight must be CUDA tensors");
  TORCH_CHECK(X.device() == weight.device(),
              "X and weight must be on the same CUDA device");
  TORCH_CHECK(X.layout() == at::kStrided && weight.layout() == at::kStrided,
              "X and weight must have strided layout");
  TORCH_CHECK(X.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
              "X and weight must have dtype float32");
  TORCH_CHECK(X.dim() >= 1 && weight.dim() == 1,
              "X must have at least one dimension and weight must be 1D");
  const int64_t dim = X.size(-1);
  TORCH_CHECK(dim > 0, "X last dimension H must be positive");
  TORCH_CHECK(dim == weight.size(0), "X and weight must have the same last dimension");
  TORCH_CHECK(X.is_contiguous() && weight.is_contiguous(),
              "X and weight must be contiguous");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0.0f,
              "epsilon must be finite and non-negative");
  TORCH_CHECK(!(at::GradMode::is_enabled() && (X.requires_grad() || weight.requires_grad())),
              "student rms_norm backward is not implemented; "
              "use torch.no_grad()/torch.inference_mode() for forward-only inference");

  const c10::cuda::CUDAGuard device_guard(X.device());
  auto output = torch::empty(X.sizes(), X.options());
  const int64_t rows = X.numel() / dim;
  if (rows == 0) return output;

  const int blocks = static_cast<int>(std::min<int64_t>((rows - 1) / kWarpNum + 1, 65535));
  const auto stream = c10::cuda::getCurrentCUDAStream(X.get_device());
  rms_norm_kernel<<<blocks, kThreadNum, 0, stream>>>(
      rows, dim, epsilon, X.data_ptr<float>(), weight.data_ptr<float>(),
      output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
