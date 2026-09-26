#pragma once
#include <torch/extension.h>
#include <tuple>

// Native inputs: contiguous CUDA FP32 [B,T,V], int64 [B,T].
inline void check_cross_entropy_inputs(const torch::Tensor& logits,
                                       const torch::Tensor& targets) {
  TORCH_CHECK(logits.is_cuda() && targets.is_cuda(), "logits and targets must be CUDA tensors");
  TORCH_CHECK(logits.device() == targets.device(), "logits and targets must be on the same CUDA device");
  TORCH_CHECK(logits.layout() == at::kStrided && targets.layout() == at::kStrided,
              "logits and targets must have strided layout");
  TORCH_CHECK(logits.scalar_type() == at::kFloat && targets.scalar_type() == at::kLong,
              "logits must be float32 and targets must be int64");
  TORCH_CHECK(logits.dim() == 3 && targets.dim() == 2,
              "logits must be 3D and targets must be 2D");
  TORCH_CHECK(logits.size(0) == targets.size(0) && logits.size(1) == targets.size(1),
              "targets shape must match logits leading dimensions");
  TORCH_CHECK(logits.size(2) > 0, "vocabulary size must be positive");
  TORCH_CHECK(logits.is_contiguous() && targets.is_contiguous(),
              "logits and targets must be contiguous");
}

// Returns row losses [B,T] and split LSE cache [B,T,2]: (max, log(sum(exp(z-max)))).
// Keeping the two terms separate avoids cancellation for large common offsets.
std::tuple<torch::Tensor, torch::Tensor>
cross_entropy_forward_cuda(torch::Tensor logits, torch::Tensor targets);

// grad_loss is a CUDA FP32 scalar (0D tensor), already divided by valid count.
// The kernel broadcasts this shared scale without host readback or a [B,T] buffer.
torch::Tensor cross_entropy_backward_cuda(torch::Tensor logits, torch::Tensor targets,
                                          torch::Tensor lse, torch::Tensor grad_loss);
