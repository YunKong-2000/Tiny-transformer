#include "linear.h"
#include "linear_kernel.cuh"
#include "linear_common.cuh"
#include "linear_benchmark.h"
#include <c10/cuda/CUDAException.h>
#include "c10/cuda/CUDAStream.h"
#include "c10/cuda/CUDAGuard.h"

namespace {
constexpr int kSmallMThreshold = 128;
constexpr int kSplitKThreshold = 2048;
constexpr int kSplitKSlices = 2;

enum class ForwardKind { Large, Small, SplitK };
ForwardKind forward_kind(int M, int K) {
  return M >= kSmallMThreshold ? ForwardKind::Large :
      (K >= kSplitKThreshold ? ForwardKind::SplitK : ForwardKind::Small);
}

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

  switch (forward_kind(M, K)) {
    case ForwardKind::SplitK:
      launch_linear_split_k(x, weight, output, M, N, K, stream.stream());
      break;
    case ForwardKind::Small:
      launch_linear_forward<InferenceGemm>(x, weight, output, M, N, K, stream.stream());
      break;
    case ForwardKind::Large:
      launch_linear_forward<ForwardGemm>(x, weight, output, M, N, K, stream.stream());
      break;
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

namespace {
template <typename Gemm>
void benchmark_tiles(LinearBenchmark& bench, const std::string& name) {
  bench.tiles[name + "_cta"] = {Gemm::ThreadblockShape::kM, Gemm::ThreadblockShape::kN,
                                 Gemm::ThreadblockShape::kK};
  bench.tiles[name + "_warp"] = {Gemm::WarpShape::kM, Gemm::WarpShape::kN, Gemm::WarpShape::kK};
}

template <typename Gemm>
void prepare_benchmark_gemm(LinearBenchmark& bench, const std::string& name,
                            cutlass::gemm::GemmCoord problem,
                            const torch::Tensor& a, int lda,
                            const torch::Tensor& b, int ldb,
                            const torch::Tensor& output, int ldd) {
  typename Gemm::Arguments args(problem, {a.data_ptr<float>(), lda},
      {b.data_ptr<float>(), ldb}, {output.data_ptr<float>(), ldd},
      {output.data_ptr<float>(), ldd}, {1.0f, 0.0f});
  auto status = Gemm::can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, name, ": ", cutlassGetStatusString(status));
  auto gemm = std::make_shared<Gemm>();
  status = gemm->initialize(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, name, ": ", cutlassGetStatusString(status));
  bench.calls[name] = [gemm, name](cudaStream_t stream) {
    auto status = gemm->run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess, name, ": ", cutlassGetStatusString(status));
  };
  benchmark_tiles<Gemm>(bench, name);
}

void prepare_benchmark_split_k(LinearBenchmark& bench, int M, int N, int K) {
  using Gemm = InferenceSplitKGemm;
  using Partial = Gemm::GemmKernel;
  using Reduction = Gemm::ReductionKernel;
  bench.split_k_slices = kSplitKSlices;
  TORCH_CHECK(K / Gemm::ThreadblockShape::kK >= kSplitKSlices,
              "Split-K requires at least one full K tile per partition");
  Gemm::Arguments args({M, N, K}, {bench.x.data_ptr<float>(), K},
      {bench.weight.data_ptr<float>(), K}, {bench.output.data_ptr<float>(), N},
      {bench.output.data_ptr<float>(), N}, {1.0f, 0.0f}, kSplitKSlices);
  const size_t bytes = Gemm::get_workspace_size(args);
  TORCH_CHECK(bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
              "Split-K workspace exceeds int64 range");
  bench.workspace = torch::empty({static_cast<int64_t>(bytes)}, bench.x.options().dtype(torch::kUInt8));
  auto gemm = std::make_shared<Gemm>();
  auto status = gemm->initialize(args, bench.workspace.data_ptr<uint8_t>());
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
  // Keep the production device operator for the full two-kernel pipeline.
  bench.calls["forward"] = [gemm](cudaStream_t stream) {
    auto status = gemm->run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
  };

  // Device::GemmSplitKParallel keeps params private. Reproduce initialize()'s
  // public kernel Params to expose its SAME partial/reduction kernels separately.
  // The benchmark verifies these stages against both production output and FP64.
  Gemm::ThreadblockSwizzle swizzle;
  auto grid_shape = swizzle.get_tiled_shape(args.problem_size,
      {Gemm::ThreadblockShape::kM, Gemm::ThreadblockShape::kN, Gemm::ThreadblockShape::kK},
      args.split_k_slices);
  cutlass::TensorRef<float, cutlass::layout::RowMajor> workspace(
      reinterpret_cast<float*>(bench.workspace.data_ptr<uint8_t>()), N);
  const int64_t stride = int64_t(M) * N;
  Partial::Params partial_params(args.problem_size, grid_shape, args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(), workspace, args.convert, stride);
  Reduction::Params reduction_params(args.problem_size.mn(), grid_shape.k(), stride,
      workspace, args.ref_D, args.ref_C.non_const_ref(), args.epilogue);
  const dim3 partial_grid = swizzle.get_grid_shape(grid_shape);
  const int smem = sizeof(Partial::SharedStorage);
  if (smem >= (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<Partial>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  }
  bench.calls["split_k_partials"] = [partial_params, partial_grid, smem](cudaStream_t stream) {
    cutlass::Kernel<Partial><<<partial_grid, Partial::kThreadCount, smem, stream>>>(partial_params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  const dim3 reduction_grid = Reduction::grid_shape(args.problem_size.mn());
  const dim3 reduction_block = Reduction::block_shape();
  bench.calls["split_k_reduce"] = [reduction_params, reduction_grid, reduction_block](cudaStream_t stream) {
    cutlass::Kernel<Reduction><<<reduction_grid, reduction_block, 0, stream>>>(reduction_params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  benchmark_tiles<Gemm>(bench, "forward");
}
} // namespace

void LinearBenchmark::run(const std::string& name) {
  TORCH_CHECK(!c10::GradMode::is_enabled(), "Linear benchmark requires no_grad");
  auto call = calls.find(name);
  TORCH_CHECK(call != calls.end(), "Unprepared Linear benchmark kernel: ", name);
  const c10::cuda::CUDAGuard guard(x.device());
  call->second(c10::cuda::getCurrentCUDAStream(x.get_device()).stream());
}

std::shared_ptr<LinearBenchmark> prepare_linear_benchmark(
    torch::Tensor x, torch::Tensor weight, torch::Tensor gradient, bool backward) {
  TORCH_CHECK(!c10::GradMode::is_enabled(), "Linear benchmark requires no_grad");
  check_inputs_forward(x, weight);
  if (backward) check_inputs_backward(gradient, x, weight);
  TORCH_CHECK(x.numel() > 0 && weight.size(0) > 0, "Linear benchmark requires nonempty GEMMs");
  const c10::cuda::CUDAGuard guard(x.device());
  const int M = static_cast<int>(x.size(0) * x.size(1));
  const int N = static_cast<int>(weight.size(0));
  const int K = static_cast<int>(weight.size(1));
  auto bench = std::make_shared<LinearBenchmark>();
  bench->x = x;
  bench->weight = weight;
  bench->gradient = gradient;
  bench->output = torch::empty({x.size(0), x.size(1), N}, x.options());
  switch (forward_kind(M, K)) {
    case ForwardKind::SplitK:
      bench->forward_kind = "split_k";
      prepare_benchmark_split_k(*bench, M, N, K);
      break;
    case ForwardKind::Small:
      bench->forward_kind = "small_m";
      prepare_benchmark_gemm<InferenceGemm>(*bench, "forward", {M, N, K}, x, K, weight, K, bench->output, N);
      break;
    case ForwardKind::Large:
      bench->forward_kind = "large_m";
      prepare_benchmark_gemm<ForwardGemm>(*bench, "forward", {M, N, K}, x, K, weight, K, bench->output, N);
      break;
  }
  if (backward) {
    bench->dx = torch::empty_like(x);
    bench->dweight = torch::empty_like(weight);
    prepare_benchmark_gemm<BackwardGemmX>(*bench, "dx", {M, K, N}, gradient, N, weight, K, bench->dx, K);
    prepare_benchmark_gemm<BackwardGemmW>(*bench, "dweight", {N, K, M}, gradient, N, x, K, bench->dweight, K);
  }
  return bench;
}
