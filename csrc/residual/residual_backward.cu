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

__global__ void residual_backward_vector_kernel(
    int64_t N, const float* dY, float* dX, float* dUpdate) {
  const int64_t tid = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x * kVector;
  for (int64_t i = tid * kVector; i < N; i += stride) {
    const float4 dy = *reinterpret_cast<const float4*>(dY + i);
    *reinterpret_cast<float4*>(dX + i) = dy;
    *reinterpret_cast<float4*>(dUpdate + i) = dy;
  }
}

__global__ void residual_backward_scalar_kernel(
    int64_t N, const float* dY, float* dX, float* dUpdate) {
  const int64_t tid = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < N; i += stride) {
    dX[i] = dY[i];
    dUpdate[i] = dY[i];
  }
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor> residual_backward_cuda(torch::Tensor dY) {
  TORCH_CHECK(dY.is_cuda(), "dY must be a CUDA tensor");
  TORCH_CHECK(dY.layout() == at::kStrided, "dY must have strided layout");
  TORCH_CHECK(dY.scalar_type() == at::kFloat, "dY must have dtype float32");
  TORCH_CHECK(dY.is_contiguous(), "dY must be contiguous");
  TORCH_CHECK(!(at::GradMode::is_enabled() && dY.requires_grad()),
              "residual_backward has no autograd binding; only first-order gradients are supported");

  const c10::cuda::CUDAGuard device_guard(dY.device());
  const int64_t N = dY.numel();
  auto dX = torch::empty(dY.sizes(), dY.options());
  auto dUpdate = torch::empty(dY.sizes(), dY.options());
  if (N == 0) return {dX, dUpdate};

  const auto stream = c10::cuda::getCurrentCUDAStream(dY.get_device());
  const auto dy_addr = reinterpret_cast<std::uintptr_t>(dY.data_ptr<float>());
  const bool vectorized = (dy_addr % alignof(float4) == 0) && (N % kVector == 0);
  const int64_t elements_per_block = kBlockSize * (vectorized ? kVector : 1);
  const int blocks = static_cast<int>(std::min<int64_t>((N - 1) / elements_per_block + 1, 65535));
  if (vectorized) {
    residual_backward_vector_kernel<<<blocks, kBlockSize, 0, stream>>>(
        N, dY.data_ptr<float>(), dX.data_ptr<float>(), dUpdate.data_ptr<float>());
  } else {
    residual_backward_scalar_kernel<<<blocks, kBlockSize, 0, stream>>>(
        N, dY.data_ptr<float>(), dX.data_ptr<float>(), dUpdate.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dX, dUpdate};
}
