#pragma once

#include <torch/extention.h>

torcH::Tensor rope_forward(torch::Tensor X, torch::Tensor cos, torch::Tensor sin);

torcH::Tensor rope_backward(torch::Tensor dY, torch::Tensor cos, torch::Tensor sin);
