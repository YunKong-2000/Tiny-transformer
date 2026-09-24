#pragma once

#include <torch/extension.h>

// CUDA FP32 forward only: contiguous int64 [B,T] IDs and float32 [V,H] weights.
torch::Tensor embedding_forward_cuda(torch::Tensor ids, torch::Tensor weight);
