#pragma once
#include <torch/extension.h>
#include <optional>
#include <tuple>

// forward, intput Q, K, V, past_legnth, segment_ids; output O, LSE
std::tuple<torch::Tensor, torch::Tensor>
attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                  int64_t past_length, std::optional<torch::Tensor> segment_ids);

// backward preprocess, input O, dO; output D
std::tuple<torch::Tensor, torch::Tensor>
attention_backward_preprocess(torch::Tensor dO, torch::Tensor O);

// backward gradient of Q, input Q、K、V、dO、LSE、D, output dQ
torch::Tensor attention_backward_dQ(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dO, torch::Tensor LSE, torch::Tensor D, float scale);

// backward gradient of K, input Q、K、V、dO、LSE、D, output dK, dV
std::tuple<torch::Tensor, torch::Tensor>
attention_backward_dKV(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dO, torch::Tensor LSE, torch::Tensor D, float scale);
