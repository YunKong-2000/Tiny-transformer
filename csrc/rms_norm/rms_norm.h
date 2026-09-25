#pragma once
#include <torch/extension.h>
#include <tuple>

// Contiguous CUDA FP32 [..., H] and [H]; returns (Y, FP32 inv_rms[rows]).
std::tuple<torch::Tensor, torch::Tensor>
rms_norm_forward_cuda(torch::Tensor X, torch::Tensor weight, float epsilon);

// First-order backward, 0 < H <= 1024; R is the matching forward's inv_rms.
// Returns (dX, dGamma). Epsilon is already incorporated into R.
std::tuple<torch::Tensor, torch::Tensor>
rms_norm_backward_cuda(torch::Tensor X, torch::Tensor gradient,
                       torch::Tensor weight, torch::Tensor R);
