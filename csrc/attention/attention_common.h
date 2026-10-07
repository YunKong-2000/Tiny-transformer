#pragma once
#include <torch/extension.h>
#include <initializer_list>
#include <optional>
#include <string>

namespace attention_detail {

inline void check_tensors(const std::initializer_list<torch::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.defined(), "attention inputs must be defined");
    TORCH_CHECK(tensor.is_cuda(), "All inputs must be CUDA tensors");
  }
  auto device = tensors.begin()->device();
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.device() == device, "All inputs must be on the same device");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat
                || tensor.scalar_type() == at::kBFloat16,
                "attention supports only float32 and bf16 inputs");
    TORCH_CHECK(tensor.scalar_type() == tensors.begin()->scalar_type(),
                "q, k, v must have the same dtype");
    TORCH_CHECK(tensor.layout() == at::kStrided, "All inputs must be strided tensors");
  }
}

inline void check_qkv(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  check_tensors({q, k, v});
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "q, k, v must be a 4D tensor");
  const int64_t B = q.size(0);
  const int64_t Nh = q.size(1);
  const int64_t Dh = q.size(-1);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B,
             "q, k, v must have same the first dimension");
  TORCH_CHECK(k.size(1) == Nh && v.size(1) == Nh,
             "q, k, v must have same the second dimension");
  TORCH_CHECK(k.size(-1) == Dh && v.size(-1) == Dh,
             "q, k, v must have same the last dimension");
  TORCH_CHECK(Dh == 64, "attention currently supports only head_dim=64");
  TORCH_CHECK(B > 0 && Nh > 0 && q.size(2) > 0 && k.size(2) > 0,
              "attention batch, heads and sequence lengths must be positive");
}

inline void check_mask_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t past_len,
    const std::optional<torch::Tensor>& segment_ids) {

  const int64_t B  = q.size(0);
  const int64_t Tq = q.size(2);
  const int64_t Tk = k.size(2);

  TORCH_CHECK(past_len >= 0,
              "past_len must be non-negative");
  TORCH_CHECK(v.size(2) == Tk,
              "k and v must have the same sequence length");

  // 使用减法，避免 past_len + Tq 的整数溢出。
  TORCH_CHECK(Tk >= Tq && past_len == Tk - Tq,
              "expected Tk == past_len + Tq");

  if (!segment_ids.has_value()) {
    return;
  }

  const auto& ids = *segment_ids;

  TORCH_CHECK(ids.defined(), "segment_ids must be defined");
  TORCH_CHECK(ids.is_cuda(),
              "segment_ids must be a CUDA tensor");
  TORCH_CHECK(ids.device() == q.device(),
              "segment_ids must be on the same device as q");
  TORCH_CHECK(ids.layout() == at::kStrided,
              "segment_ids must have strided layout");
  TORCH_CHECK(ids.scalar_type() == at::kLong,
              "segment_ids must be int64");

  TORCH_CHECK(ids.dim() == 2,
              "segment_ids must be 2D");
  TORCH_CHECK(ids.size(0) == B && ids.size(1) == Tq,
              "segment_ids must have shape [B, Tq]");

  TORCH_CHECK(past_len == 0 && Tq == Tk,
              "segment_ids are supported only without past context");
}

} // namespace attention_detail
