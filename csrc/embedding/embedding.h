#pragma once

#include <torch/extension.h>
#include <string>

// CUDA FP32 forward: contiguous int64 [B,T] IDs and float32 [V,H] weights.
torch::Tensor embedding_forward_cuda(torch::Tensor ids, torch::Tensor weight);

// CUDA FP32 backward: contiguous int64 [B,T] IDs, float32 [B,T,H] gradients and size of vocabulary V.
torch::Tensor embedding_backward_cuda(
    torch::Tensor ids, torch::Tensor gradient, int64_t vocab_size,
    const std::string& implementation = "grouped");
