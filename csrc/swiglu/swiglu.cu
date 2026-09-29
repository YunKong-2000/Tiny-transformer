#include "swiglu_common.cuh"

namespace {
using namespace swiglu_detail;

__global__ void swiglu_forward_kernel(
    int64_t B, int64_t T, int64_t I, Strides gs, Strides us,
    const float* gate, const float* up, float* z) {
  using namespace cute;
  auto shape = make_shape(B, T, I);
  auto G = make_tensor(make_gmem_ptr(gate), make_layout(shape, make_stride(gs.b, gs.t, gs.i)));
  auto U = make_tensor(make_gmem_ptr(up), make_layout(shape, make_stride(us.b, us.t, us.i)));
  auto Z = make_tensor(make_gmem_ptr(z), make_layout(shape, make_stride(T * I, I, Int<1>{})));

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  auto cta_shape = make_shape(Int<kWarpPerBlock>{}, I);
  auto threads = make_layout(
      make_shape(Int<kWarpPerBlock>{}, Int<kWarpSize>{}),
      make_stride(Int<kWarpSize>{}, Int<1>{}));
  const int64_t t_lines = (T + kWarpPerBlock - 1) / kWarpPerBlock;
  const int64_t tiles = B * t_lines;
  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int64_t b = tile / t_lines;
    const int64_t tt = tile % t_lines;
    const int64_t t = tt * kWarpPerBlock + warp_id;
    if (t >= T) continue;

    auto gG = local_tile(G(b, _, _), cta_shape, make_coord(tt, 0));
    auto gU = local_tile(U(b, _, _), cta_shape, make_coord(tt, 0));
    auto gZ = local_tile(Z(b, _, _), cta_shape, make_coord(tt, 0));
    auto tG = local_partition(gG, threads, threadIdx.x);
    auto tU = local_partition(gU, threads, threadIdx.x);
    auto tZ = local_partition(gZ, threads, threadIdx.x);
    // Thread views have shape (1, ceil_div(I, 32)); scalar i is equivalent to (0, i).
    // CuTe does not mask partial tiles: guard both token and channel before access.
    for (int64_t i = 0; lane_id + i * kWarpSize < I; ++i) {
      const float g = tG(i);
      tZ(i) = (g * sigmoid(g)) * tU(i);
    }
  }
}
}  // namespace

torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor up) {
  using namespace swiglu_detail;
  check_inputs(gate, up);
  TORCH_CHECK(!(at::GradMode::is_enabled() && (gate.requires_grad() || up.requires_grad())),
              "swiglu_forward has no autograd binding; use student.swiglu for training");
  const c10::cuda::CUDAGuard device_guard(gate.device());
  auto z = torch::empty(gate.sizes(), gate.options());
  if (gate.numel() == 0) return z;

  const int64_t B = gate.size(0), T = gate.size(1), I = gate.size(2);
  const auto stream = c10::cuda::getCurrentCUDAStream(gate.get_device());
  // data_ptr() includes each view's storage offset, including up's chunk offset.
  swiglu_forward_kernel<<<block_count(B, T), kBlockSize, 0, stream>>>(
      B, T, I, strides_of(gate), strides_of(up),
      gate.data_ptr<float>(), up.data_ptr<float>(), z.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return z;
}
