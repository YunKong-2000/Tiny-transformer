#pragma once

#include <torch/extension.h>
#include <c10/core/GradMode.h>
#include <limits>


inline void check_inputs_forward(const torch::Tensor& x, const torch::Tensor& weight) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(),
              "Tensor must be on CUDA device");
  TORCH_CHECK(x.device() == weight.device(),
              "Tensors must be on the same device");
  TORCH_CHECK(x.layout() == torch::kStrided && weight.layout() == torch::kStrided,
              "linear requires strided layout");
  TORCH_CHECK(x.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32,
              "Tensor must be of type float32");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
              "Tensor must be contiguous");
  TORCH_CHECK(x.dim() == 3 && weight.dim() == 2,
              "x must be 3D and weight must be 2D");
  TORCH_CHECK(x.size(2) == weight.size(1),
              "The last dimension of x must match the second dimension of weight");
  // GemmCoord uses signed 32-bit dimensions. Check before narrowing or multiplying.
  constexpr int64_t limit = std::numeric_limits<int>::max();
  TORCH_CHECK(x.size(0) <= limit && x.size(1) <= limit &&
              weight.size(0) <= limit && weight.size(1) <= limit &&
              (x.size(1) == 0 || x.size(0) <= limit / x.size(1)),
              "linear GEMM dimensions must fit in int32");
}

inline void check_inputs_backward(const torch::Tensor& gradient, const torch::Tensor& x, const torch::Tensor& weight) {
  check_inputs_forward(x, weight);
  TORCH_CHECK(gradient.is_cuda(), "Gradient must be a CUDA tensor");
  TORCH_CHECK(gradient.device() == x.device(), "Gradient must be on the same device as input");
  TORCH_CHECK(gradient.dtype() == torch::kFloat32, "Gradient must be of type float32");
  TORCH_CHECK(gradient.layout() == torch::kStrided, "Gradient must have strided layout");
  TORCH_CHECK(gradient.is_contiguous(), "Gradient must be contiguous");
  TORCH_CHECK(gradient.dim() == 3, "Gradient must be 3D");
  TORCH_CHECK(gradient.size(0) == x.size(0)
              && gradient.size(1) == x.size(1),
              "Gradient and x must have the same first two dimensions");
  TORCH_CHECK(gradient.size(2) == weight.size(0),
              "The last dimension of gradient must match the first dimension of weight");
}
