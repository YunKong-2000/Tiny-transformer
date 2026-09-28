// A tiny integration check: CuTe tensor indexing + CUTLASS device arithmetic.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/cutlass.h>
#include <cutlass/functional.h>
#include <cute/tensor.hpp>

namespace {
__global__ void add_one_kernel(const float* input, float* output, int64_t n) {
  auto layout = cute::make_layout(cute::make_shape(n));
  auto x = cute::make_tensor(cute::make_gmem_ptr(input), layout);
  auto y = cute::make_tensor(cute::make_gmem_ptr(output), layout);
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    y(i) = cutlass::plus<float>{}(x(i), 1.0f);
  }
}

torch::Tensor add_one(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat32,
              "expected a CUDA float32 tensor");
  TORCH_CHECK(input.is_contiguous() && input.dim() == 1,
              "expected a contiguous 1D tensor");
  c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty_like(input);
  auto n = input.numel();
  if (n > 0) {
    add_one_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &add_one);
}
