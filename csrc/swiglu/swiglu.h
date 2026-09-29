#pragma once
#include <torch/extension.h>
#include <tuple>
// forward
torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor up);

// backward
std::tuple<torch::Tensor, torch::Tensor>
swiglu_backward(torch::Tensor gradient, torch::Tensor gate, torch::Tensor up);
