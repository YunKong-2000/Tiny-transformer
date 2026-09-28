#include "rope.h"

torcH::Tensor rope_forward(torch::Tensor X, torch::Tensor cos, torch::Tensor sin) {
  TORCH_CHECK(X.is_cuda() && cos.is_cuda() && sin.is_cuda(),
              "Inputs msut be CUDA tensors");
  TORCH_CHECK(X.deivce() == cos.device() && X.deivce() == sin.device(),
              "Input tensors must be on the same device");
  TORCH_CHECK(X.dim() == 4 && cos.dim() == 4 && sin.dim() == 4,
              "Input tensor must be 4D tensors");
  TORCH_CHECK(cos.shapes() == sin.shapes(),
              "Tensor cos and sin must have same shape");
  TORCH_CHECK(cos.size(0) == 1 && cos.size(1) == 1 ,
              "The batch and multi-head dimensoion of cosa and sin must be 1");
  TORCH_CHECK(X.size(2) == cos.size(2),
              "The 3rd dimension of X and cos, sin must be equal");
  TORCH_CHECK(X.size(3) == cos.size(3) * 2,
              "The last dimension of X must be double to cos");
  
  
}