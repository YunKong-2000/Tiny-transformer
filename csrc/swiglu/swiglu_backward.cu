#include "swiglu_common.cuh"

namespace {
using namespace swiglu_detail;

__global__ void swiglu_backward_kernel(
    int64_t B, int64_t T, int64_t I, Strides gs, Strides us, Strides zs,
    const float* gate, const float* up, const float* dz, float* dg, float* du) {
  using namespace cute;
  auto shape = make_shape(B, T, I);
  auto output_layout = make_layout(shape, make_stride(T * I, I, Int<1>{}));
  auto G = make_tensor(make_gmem_ptr(gate), make_layout(shape, make_stride(gs.b, gs.t, gs.i)));
  auto U = make_tensor(make_gmem_ptr(up), make_layout(shape, make_stride(us.b, us.t, us.i)));
  auto dZ = make_tensor(make_gmem_ptr(dz), make_layout(shape, make_stride(zs.b, zs.t, zs.i)));
  auto dGate = make_tensor(make_gmem_ptr(dg), output_layout);
  auto dUp = make_tensor(make_gmem_ptr(du), output_layout);

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
    auto gdZ = local_tile(dZ(b, _, _), cta_shape, make_coord(tt, 0));
    auto gdGate = local_tile(dGate(b, _, _), cta_shape, make_coord(tt, 0));
    auto gdUp = local_tile(dUp(b, _, _), cta_shape, make_coord(tt, 0));
    auto tG = local_partition(gG, threads, threadIdx.x);
    auto tU = local_partition(gU, threads, threadIdx.x);
    auto tdZ = local_partition(gdZ, threads, threadIdx.x);
    auto tdGate = local_partition(gdGate, threads, threadIdx.x);
    auto tdUp = local_partition(gdUp, threads, threadIdx.x);
    for (int64_t i = 0; lane_id + i * kWarpSize < I; ++i) {
      const float g = tG(i);
      const float u = tU(i);
      const float dz_value = tdZ(i);
      const float s = sigmoid(g);
      const float silu = g * s;
      const float dsilu = s + silu * (1.0f - s);
      tdGate(i) = (dz_value * u) * dsilu;
      tdUp(i) = dz_value * silu;
    }
  }
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor>
swiglu_backward(torch::Tensor gradient, torch::Tensor gate, torch::Tensor up) {
  using namespace swiglu_detail;
  check_inputs(gate, up);
  check_tensor(gradient);
  TORCH_CHECK(gradient.device() == gate.device(),
              "gradient and inputs must be on the same CUDA device");
  TORCH_CHECK(gradient.sizes() == gate.sizes(),
              "gradient and inputs must have the same shape");
  TORCH_CHECK(!(at::GradMode::is_enabled() &&
              (gradient.requires_grad() || gate.requires_grad() || up.requires_grad())),
              "swiglu_backward supports only first-order gradients");
  const c10::cuda::CUDAGuard device_guard(gate.device());
  auto dGate = torch::empty(gate.sizes(), gate.options());
  auto dUp = torch::empty(up.sizes(), up.options());
  if (gate.numel() == 0) return {dGate, dUp};

  const int64_t B = gate.size(0), T = gate.size(1), I = gate.size(2);
  const auto stream = c10::cuda::getCurrentCUDAStream(gate.get_device());
  swiglu_backward_kernel<<<block_count(B, T), kBlockSize, 0, stream>>>(
      B, T, I, strides_of(gate), strides_of(up), strides_of(gradient),
      gate.data_ptr<float>(), up.data_ptr<float>(), gradient.data_ptr<float>(),
      dGate.data_ptr<float>(), dUp.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dGate, dUp};
}
