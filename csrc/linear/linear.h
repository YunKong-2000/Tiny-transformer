#pragma once

#include <torch/extension.h>
#include <tuple>

// forward
torch::Tensor linear_forward(torch::Tensor x, torch::Tensor weight);

// backward
std::tuple<torch::Tensor, torch::Tensor>
linear_backward(torch::Tensor gradient, torch::Tensor x, torch::Tensor weight);
