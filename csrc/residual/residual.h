#pragma once

#include <torch/extension.h>
#include <tuple>

// Raw API: contiguous CUDA FP32 tensors, same shape; no broadcasting or autograd.
torch::Tensor residual_forward_cuda(torch::Tensor X, torch::Tensor update);

// Both gradients equal dY. No forward activation needs to be saved.
std::tuple<torch::Tensor, torch::Tensor> residual_backward_cuda(torch::Tensor dY);
