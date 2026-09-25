#include "rms_norm.h"

#include <algorithm>
#include <ATen/Context.h>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace{

constexpr int kWarpSize = 32;
constexpr int kShmSize = 1024;
constexpr unsigned kFullWarp = 0xffffffff;
constexpr int kThreadNum = 256;
constexpr int kWarpNum = kThreadNum / kWarpSize;


__global__ void rms_norm_backward_shmcache_kernel
(
  const int64_t rows,
  const int64_t dim,
  const float* X,
  const float* dY,
  const float* weight,
  const float* R,
  float* dX,
  float* dGamma
)
{
  const int64_t bid = blockIdx.x;
  const int64_t grid_size = gridDim.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  __shared__ float cache_x[kShmSize];
  __shared__ float cache_dy[kShmSize];
  __shared__ float cache_u[kShmSize];
  __shared__ float block_sum[kWarpNum];
  // The host wrapper checks H <= kShmSize before launching.
  for (int64_t row = bid; row < rows; row += grid_size) {
    int64_t row_base = row * dim;
    float r = R[row];
    float local_sum = 0;

    for (int64_t col = tid; col < dim; col += kThreadNum) {
      float x_value = X[row_base + col];
      float dy_value = dY[row_base + col];
      cache_x[col] = x_value;
      cache_dy[col] = dy_value;
      float gamma = weight[col];
      float u_value = dy_value * gamma;
      cache_u[col] = u_value;
      local_sum += u_value * x_value * r;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(kFullWarp, local_sum, offset);
    }
    if (lane_id == 0) block_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
      local_sum = (lane_id < kWarpNum) ? block_sum[lane_id] : 0;
      for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(kFullWarp, local_sum, offset);
      }
      if (lane_id == 0) block_sum[0] = local_sum;
    }
    __syncthreads();
    float S = block_sum[0];

    for (int64_t col = tid; col < dim; col += kThreadNum) {
      float u_value = cache_u[col];
      float x_value = cache_x[col];
      float dy_value = cache_dy[col];
      dX[row_base + col] = r * (u_value - x_value * r * S / dim);
      atomicAdd(dGamma + col, dy_value * x_value * r);
    }
    __syncthreads();
  }
}

} //namespace


std::tuple<torch::Tensor, torch::Tensor>
rms_norm_backward_cuda(torch::Tensor X, torch::Tensor gradient,
                       torch::Tensor weight, torch::Tensor R) {
  TORCH_CHECK(X.is_cuda() && weight.is_cuda() && gradient.is_cuda() && R.is_cuda(),
              "X, weight, gradient and R must be CUDA tensors");
  TORCH_CHECK(X.device() == weight.device() && X.device() == gradient.device()
              && X.device() == R.device(),
              "X, weight, gradient and R must be on the same CUDA device");
  TORCH_CHECK(X.layout() == at::kStrided && weight.layout() == at::kStrided
              && gradient.layout() == at::kStrided && R.layout() == at::kStrided,
              "X, weight, gradient and R must have strided layout");
  TORCH_CHECK(X.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat
              && gradient.scalar_type() == at::kFloat && R.scalar_type() == at::kFloat,
              "X, weight, gradient and R must have dtype float32");
  TORCH_CHECK(X.dim() >= 1 && weight.dim() == 1,
              "X must have at least one dimension and weight must be 1D");
  TORCH_CHECK(X.sizes() == gradient.sizes(), "X and gradient must have the same shape");
  const int64_t dim = X.size(-1);
  TORCH_CHECK(dim > 0 && dim <= kShmSize, "rms_norm backward requires 0 < H <= 1024");
  TORCH_CHECK(dim == weight.size(0), "X and weight must have the same last dimension");
  const int64_t rows = X.numel() / dim;
  TORCH_CHECK(R.dim() == 1 && R.numel() == rows, "R must have shape [rows]");
  TORCH_CHECK(X.is_contiguous() && weight.is_contiguous()
              && gradient.is_contiguous() && R.is_contiguous(),
              "X, weight, gradient and R must be contiguous");
  TORCH_CHECK(!(at::GradMode::is_enabled() && (X.requires_grad() || weight.requires_grad()
              || gradient.requires_grad() || R.requires_grad())),
              "rms_norm_backward has no autograd binding; only first-order gradients are supported");

  const c10::cuda::CUDAGuard device_guard(X.device());
  auto dX = torch::empty(X.sizes(), X.options());
  auto dGamma = torch::zeros(weight.sizes(), weight.options());
  if (rows == 0) return {dX, dGamma};
  // Cross-row floating-point atomic additions are not deterministic.
  at::globalContext().alertNotDeterministic("rms_norm_backward_cuda");
  const auto stream = c10::cuda::getCurrentCUDAStream(X.get_device());
  const int blocks = static_cast<int>(std::min<int64_t>(rows, 65535));
  rms_norm_backward_shmcache_kernel<<<blocks, kThreadNum, 0, stream>>>(
      rows, dim, X.data_ptr<float>(), gradient.data_ptr<float>(),
      weight.data_ptr<float>(), R.data_ptr<float>(),
      dX.data_ptr<float>(), dGamma.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dX, dGamma};
}
