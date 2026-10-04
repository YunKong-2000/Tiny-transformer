#include "linear.h"
#include "linear_kernel.cuh"
#include "linear_common.cuh"
#include "c10/cuda/CUDAStream.h"
#include "c10/cuda/CUDAGuard.h"

namespace {
constexpr int kSmallMThreshold = 128;
constexpr int kSplitKThreshold = 2048;
constexpr int kSplitKSlices = 2;

// Both forward kernels use the same layouts; only the GEMM type changes.
// Gemm::Arguments is a dependent type, so it requires typename here.
template <typename Gemm>
void launch_linear_forward(const torch::Tensor& x, const torch::Tensor& weight,
                           const torch::Tensor& output, int M, int N, int K,
                           cudaStream_t stream) {
  typename Gemm::Arguments args({M, N, K},
                              {x.data_ptr<float>(), K},
                              {weight.data_ptr<float>(), K},
                              {output.data_ptr<float>(), N},
                              {output.data_ptr<float>(), N},
                              {1.0f, 0.0f});
  auto status = Gemm::can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "Forward GEMM arguments: ", cutlassGetStatusString(status));

  Gemm gemm_op;
  status = gemm_op(args, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "Forward GEMM launch: ", cutlassGetStatusString(status));
}

void launch_linear_split_k(const torch::Tensor& x, const torch::Tensor& weight,
                           const torch::Tensor& output, int M, int N, int K,
                           cudaStream_t stream) {
  using Gemm = InferenceSplitKGemm;
  static_assert(kSplitKSlices > 1, "Use ordinary GEMM for a single K partition");
  // This CUTLASS version's can_implement() unconditionally returns success.
  // Ensure each partition receives at least one full K tile ourselves.
  TORCH_CHECK(K / Gemm::ThreadblockShape::kK >= kSplitKSlices,
              "Split-K requires at least one full K tile per partition");
  Gemm::Arguments args{
    {M, N, K},
    {x.data_ptr<float>(), K},
    {weight.data_ptr<float>(), K},
    {output.data_ptr<float>(), N},
    {output.data_ptr<float>(), N},
    {1.0f, 0.0f},
    kSplitKSlices
  };
  const size_t workspace_size = Gemm::get_workspace_size(args);
  TORCH_CHECK(workspace_size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
              "Split-K workspace size exceeds int64 range");
  // The caller supplies PyTorch's current stream under the input device guard.
  // Same-stream allocation/use lets the caching allocator manage async lifetime;
  // all partials are overwritten, so no workspace zeroing or synchronization.
  auto workspace = torch::empty(
      {static_cast<int64_t>(workspace_size)}, x.options().dtype(torch::kUInt8));
  Gemm gemm_op;
  auto status = gemm_op(args, workspace.data_ptr<uint8_t>(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "Split-K GEMM launch: ", cutlassGetStatusString(status));
}


} // namespace


torch::Tensor linear_forward(torch::Tensor x, torch::Tensor weight) {
  check_inputs_forward(x, weight);
  TORCH_CHECK(!c10::GradMode::is_enabled() || !(x.requires_grad() || weight.requires_grad()),
              "use student.linear autograd binding for trainable inputs");
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto output = torch::empty({x.size(0), x.size(1), weight.size(0)}, x.options());
  if (output.numel() == 0) return output;
  if (weight.size(1) == 0) return output.zero_();
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const int M = static_cast<int>(x.size(0) * x.size(1));
  const int N = static_cast<int>(weight.size(0));
  const int K = static_cast<int>(weight.size(1));

  if (M < kSmallMThreshold) {
    if (K >= kSplitKThreshold) {
      launch_linear_split_k(x, weight, output, M, N, K, stream.stream());
    } else {
      launch_linear_forward<InferenceGemm>(x, weight, output, M, N, K, stream.stream());
    }
  } else {
    launch_linear_forward<ForwardGemm>(x, weight, output, M, N, K, stream.stream());
  }
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> linear_backward(torch::Tensor gradient, torch::Tensor x, torch::Tensor weight) {
  check_inputs_backward(gradient, x, weight);
  TORCH_CHECK(!c10::GradMode::is_enabled() ||
              !(gradient.requires_grad() || x.requires_grad() || weight.requires_grad()),
              "linear backward supports first-order gradients only");
  const c10::cuda::CUDAGuard device_guard(gradient.device());
  auto dX = torch::empty(x.sizes(), x.options());
  auto dW = torch::empty(weight.sizes(), weight.options());
  // Zero-size reductions can still have nonempty outputs. Normal GEMMs overwrite
  // their complete outputs (beta=0), so do not pay for redundant zeroing there.
  if (gradient.numel() == 0 || weight.size(1) == 0) {
    dX.zero_();
    dW.zero_();
    return {dX, dW};
  }
  const auto stream = c10::cuda::getCurrentCUDAStream(gradient.get_device());

  const int M = static_cast<int>(gradient.size(0) * gradient.size(1));
  const int N = static_cast<int>(weight.size(0));
  const int K = static_cast<int>(weight.size(1));

  BackwardGemmX gemm_op_x;
  BackwardGemmX::Arguments args_x({M, K, N},
                                  {gradient.data_ptr<float>(), N},
                                  {weight.data_ptr<float>(), K},
                                  {dX.data_ptr<float>(), K},
                                  {dX.data_ptr<float>(), K},
                                  {1.0f, 0.0f});

  auto status_x = BackwardGemmX::can_implement(args_x);
  TORCH_CHECK(status_x == cutlass::Status::kSuccess,
              "Backward GEMM for dX arguments: ", cutlassGetStatusString(status_x));
  status_x = gemm_op_x(args_x, nullptr, stream.stream());
  TORCH_CHECK(status_x == cutlass::Status::kSuccess,
              "Backward GEMM for dX launch: ", cutlassGetStatusString(status_x));

  BackwardGemmW gemm_op_w;
  BackwardGemmW::Arguments args_w({N, K, M},
                                  {gradient.data_ptr<float>(), N},
                                  {x.data_ptr<float>(), K},
                                  {dW.data_ptr<float>(), K},
                                  {dW.data_ptr<float>(), K},
                                  {1.0f, 0.0f});
  auto status_w = BackwardGemmW::can_implement(args_w);
  TORCH_CHECK(status_w == cutlass::Status::kSuccess,
              "Backward GEMM for dW arguments: ", cutlassGetStatusString(status_w));
  status_w = gemm_op_w(args_w, nullptr, stream.stream());
  TORCH_CHECK(status_w == cutlass::Status::kSuccess,
              "Backward GEMM for dW launch: ", cutlassGetStatusString(status_w));

  return {dX, dW};
}
