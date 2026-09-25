#include "residual.h"

#include <algorithm>
#include <cstdint>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {
constexpr int kVector = 4;
constexpr int kBlockSize = 256;

__global__ void residual_vector_kernel(
    int64_t N, const float* X, const float* update, float* Y) {
  const int64_t tid = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x * kVector;
  for (int64_t i = tid * kVector; i < N; i += stride) {
    const float4 x = *reinterpret_cast<const float4*>(X + i);
    const float4 u = *reinterpret_cast<const float4*>(update + i);
    float4 sum;
    sum.x = x.x + u.x;
    sum.y = x.y + u.y;
    sum.z = x.z + u.z;
    sum.w = x.w + u.w;
    *reinterpret_cast<float4*>(Y + i) = sum;
  }
}

__global__ void residual_scalar_kernel(
    int64_t N, const float* X, const float* update, float* Y) {
  const int64_t tid = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < N; i += stride) {
    Y[i] = X[i] + update[i];
  }
}
}  // namespace

torch::Tensor residual_forward_cuda(torch::Tensor X, torch::Tensor update) {
  TORCH_CHECK(X.is_cuda() && update.is_cuda(), "X and update must be CUDA tensors");
  TORCH_CHECK(X.device() == update.device(), "X and update must be on the same CUDA device");
  TORCH_CHECK(X.layout() == at::kStrided && update.layout() == at::kStrided,
              "X and update must have strided layout");
  TORCH_CHECK(X.scalar_type() == at::kFloat && update.scalar_type() == at::kFloat,
              "X and update must have dtype float32");
  TORCH_CHECK(X.sizes() == update.sizes(), "X and update must have the same shape");
  TORCH_CHECK(X.is_contiguous() && update.is_contiguous(), "X and update must be contiguous");
  TORCH_CHECK(!(at::GradMode::is_enabled() && (X.requires_grad() || update.requires_grad())),
              "residual_forward has no autograd binding; use student.residual for training "
              "or torch.no_grad()/torch.inference_mode() for inference");

  const c10::cuda::CUDAGuard device_guard(X.device());
  auto Y = torch::empty(X.sizes(), X.options());
  const int64_t N = X.numel();
  if (N == 0) return Y;

  const auto stream = c10::cuda::getCurrentCUDAStream(X.get_device());
  const auto x_addr = reinterpret_cast<std::uintptr_t>(X.data_ptr<float>());
  const auto update_addr = reinterpret_cast<std::uintptr_t>(update.data_ptr<float>());
  // Newly allocated outputs are aligned; contiguous views of inputs may not be.
  const bool vectorized = (x_addr % alignof(float4) == 0)
                          && (update_addr % alignof(float4) == 0) && (N % kVector == 0);
  const int64_t elements_per_block = kBlockSize * (vectorized ? kVector : 1);
  const int blocks = static_cast<int>(std::min<int64_t>((N - 1) / elements_per_block + 1, 65535));
  if (vectorized) {
    residual_vector_kernel<<<blocks, kBlockSize, 0, stream>>>(
        N, X.data_ptr<float>(), update.data_ptr<float>(), Y.data_ptr<float>());
  } else {
    residual_scalar_kernel<<<blocks, kBlockSize, 0, stream>>>(
        N, X.data_ptr<float>(), update.data_ptr<float>(), Y.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Y;
}
