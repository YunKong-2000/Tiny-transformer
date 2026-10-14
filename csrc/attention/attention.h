#pragma once
#include <torch/extension.h>
#include <optional>
#include <tuple>

// Raw forward returns O and FP32 natural-log LSE; autograd lives in Python.
std::tuple<torch::Tensor, torch::Tensor>
attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                  int64_t past_length, std::optional<torch::Tensor> segment_ids);
// BF16 causal backward on SM80+, head_dim=64; returns contiguous dQ/dK/dV.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
attention_backward(
  torch::Tensor q, torch::Tensor k, torch::Tensor v,
  torch::Tensor dO, torch::Tensor O, torch::Tensor lse,
  int64_t past_len, std::optional<torch::Tensor> segment_ids);
