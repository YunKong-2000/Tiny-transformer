#pragma once

#include "swiglu.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cute/tensor.hpp>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace swiglu_detail {
constexpr int kWarpSize = 32;
constexpr int kBlockSize = 128;
constexpr int kWarpPerBlock = kBlockSize / kWarpSize;

struct Strides {
  int64_t b, t, i;
};

inline Strides strides_of(const torch::Tensor& x) {
  return {x.stride(0), x.stride(1), x.stride(2)};
}

inline void check_tensor(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "swiglu inputs must be CUDA tensors");
  TORCH_CHECK(x.layout() == at::kStrided, "swiglu inputs must have strided layout");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "swiglu supports only float32 inputs");
  TORCH_CHECK(x.dim() == 3, "swiglu inputs must be 3D tensors");
}

inline void check_inputs(const torch::Tensor& gate, const torch::Tensor& up) {
  check_tensor(gate);
  check_tensor(up);
  TORCH_CHECK(gate.device() == up.device(), "gate and up must be on the same CUDA device");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");
}

inline int block_count(int64_t B, int64_t T) {
  const int64_t tiles = B * ((T + kWarpPerBlock - 1) / kWarpPerBlock);
  return static_cast<int>(std::min<int64_t>(tiles, 65535));
}

// Avoid overflowing exp for large negative finite gate values.
__device__ inline float sigmoid(float x) {
  const float e = expf(-fabsf(x));
  return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}
}  // namespace swiglu_detail
