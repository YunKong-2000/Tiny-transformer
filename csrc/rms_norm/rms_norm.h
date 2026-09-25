#pragma once
#include <torch/extension.h>

// Contiguous CUDA FP32 [..., H] input and [H] weight; forward only.
torch::Tensor rms_norm_forward_cuda(torch::Tensor X, torch::Tensor weight, float epsilon);
