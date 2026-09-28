#include "rope.h"

#include <algorithm>
#include <cstdint>
#include <cute/tensor.hpp>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {
constexpr int kWarpSize = 32;
constexpr int kWarpNumPerCTA = 4;
constexpr int kBlockSize = kWarpSize * kWarpNumPerCTA;

struct Strides {
  int64_t b, h, t, d;
};

Strides strides_of(const torch::Tensor& x) {
  return {x.stride(0), x.stride(1), x.stride(2), x.stride(3)};
}

// Backward uses the transpose rotation: replace sin by -sin. The same
// partition reads strided dY, including the zero strides produced by sum().
template <bool Backward>
__global__ void rope_kernel(
    int64_t B, int64_t Nh, int64_t T, int64_t Dh,
    Strides xs, Strides cs, Strides ss,
    const float* input, const float* cos, const float* sin, float* output) {
  using namespace cute;
  auto X = make_tensor(make_gmem_ptr(input), make_layout(
      make_shape(B, Nh, T, Dh), make_stride(xs.b, xs.h, xs.t, xs.d)));
  auto Y = make_tensor(make_gmem_ptr(output), make_layout(
      make_shape(B, Nh, T, Dh),
      make_stride(Nh * T * Dh, T * Dh, Dh, Int<1>{})));
  // Host sets coefficient batch stride to zero for shared positions.
  auto Cos = make_tensor(make_gmem_ptr(cos), make_layout(
      make_shape(B, T, Dh / 2), make_stride(cs.b, cs.t, cs.d)));
  auto Sin = make_tensor(make_gmem_ptr(sin), make_layout(
      make_shape(B, T, Dh / 2), make_stride(ss.b, ss.t, ss.d)));

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int64_t head_tiles = (Nh + kWarpNumPerCTA - 1) / kWarpNumPerCTA;
  const int64_t tiles = B * T * head_tiles;
  // Flatten (b, head_tile, t) to avoid CUDA grid.y/grid.z size limits.
  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int64_t t = tile % T;
    const int64_t ht = (tile / T) % head_tiles;
    const int64_t b = tile / (T * head_tiles);
    const int64_t h = ht * kWarpNumPerCTA + warp_id;
    if (h >= Nh) continue;

    auto X_bt = X(b, _, t, _);
    auto Y_bt = Y(b, _, t, _);
    auto cos_bt = Cos(b, t, _);
    auto sin_bt = Sin(b, t, _);
    auto cta_tile = make_shape(Int<kWarpNumPerCTA>{}, Dh);
    auto gX = local_tile(X_bt, cta_tile, make_coord(ht, 0));
    auto gY = local_tile(Y_bt, cta_tile, make_coord(ht, 0));

    // Coordinates are (head_in_tile, pair), not (head, channel).
    auto pair_shape = make_shape(Int<kWarpNumPerCTA>{}, Dh / 2);
    auto x_pair_layout = make_layout(pair_shape, make_stride(xs.h, 2 * xs.d));
    auto y_pair_layout = make_layout(pair_shape, make_stride(T * Dh, Int<2>{}));
    auto Xe = make_tensor(gX.data(),        x_pair_layout);
    auto Xo = make_tensor(gX.data() + xs.d, x_pair_layout);
    auto Ye = make_tensor(gY.data(),        y_pair_layout);
    auto Yo = make_tensor(gY.data() + 1,    y_pair_layout);
    auto C = make_tensor(cos_bt.data(), make_layout(
        pair_shape, make_stride(Int<0>{}, cs.d)));
    auto S = make_tensor(sin_bt.data(), make_layout(
        pair_shape, make_stride(Int<0>{}, ss.d)));

    auto threads = make_layout(
        make_shape(Int<kWarpNumPerCTA>{}, Int<kWarpSize>{}),
        make_stride(Int<kWarpSize>{}, Int<1>{}));
    auto tXe = local_partition(Xe, threads, threadIdx.x);
    auto tXo = local_partition(Xo, threads, threadIdx.x);
    auto tYe = local_partition(Ye, threads, threadIdx.x);
    auto tYo = local_partition(Yo, threads, threadIdx.x);
    auto tC = local_partition(C, threads, threadIdx.x);
    auto tS = local_partition(S, threads, threadIdx.x);

    // CuTe partitions do not mask partial tiles. Guard every load and store.
    for (int64_t i = 0; lane_id + i * kWarpSize < Dh / 2; ++i) {
      const float a = tXe(0, i);
      const float b_value = tXo(0, i);
      const float c = tC(0, i);
      const float s = Backward ? -tS(0, i) : tS(0, i);
      tYe(0, i) = a * c - b_value * s;
      tYo(0, i) = a * s + b_value * c;
    }
  }
}

void check_inputs(const torch::Tensor& x, const torch::Tensor& cos,
                  const torch::Tensor& sin) {
  TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda(),
              "rope inputs must be CUDA tensors");
  TORCH_CHECK(x.device() == cos.device() && x.device() == sin.device(),
              "rope inputs must be on the same CUDA device");
  TORCH_CHECK(x.layout() == at::kStrided && cos.layout() == at::kStrided &&
              sin.layout() == at::kStrided, "rope inputs must have strided layout");
  TORCH_CHECK(x.scalar_type() == at::kFloat && cos.scalar_type() == at::kFloat &&
              sin.scalar_type() == at::kFloat, "rope supports only float32 inputs");
  TORCH_CHECK(x.dim() == 4 && cos.dim() == 4 && sin.dim() == 4,
              "rope inputs must be 4D tensors");
  TORCH_CHECK(x.size(3) > 0 && x.size(3) % 2 == 0,
              "rope head dimension must be positive and even");
  TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin must have the same shape");
  TORCH_CHECK((cos.size(0) == 1 || cos.size(0) == x.size(0)) && cos.size(1) == 1 &&
              cos.size(2) == x.size(2) && cos.size(3) == x.size(3) / 2,
              "cos/sin must have shape [1 or B, 1, T, Dh/2]");
  TORCH_CHECK(!cos.requires_grad() && !sin.requires_grad(),
              "rope requires constant cos/sin; trainable coefficients are not supported");
}

template <bool Backward>
torch::Tensor launch_rope(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
  check_inputs(x, cos, sin);
  TORCH_CHECK(!(at::GradMode::is_enabled() && x.requires_grad()),
              Backward ? "rope_backward supports only first-order gradients"
                       : "rope_forward has no autograd binding; use student.rope for training");
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto output = torch::empty(x.sizes(), x.options());
  if (x.numel() == 0) return output;

  const int64_t B = x.size(0), Nh = x.size(1), T = x.size(2), Dh = x.size(3);
  auto cs = strides_of(cos);
  auto ss = strides_of(sin);
  if (cos.size(0) == 1) cs.b = 0;
  if (sin.size(0) == 1) ss.b = 0;
  const int64_t tiles = B * T * ((Nh + kWarpNumPerCTA - 1) / kWarpNumPerCTA);
  const int blocks = static_cast<int>(std::min<int64_t>(tiles, 65535));
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  // data_ptr() already includes the tensor's storage offset.
  rope_kernel<Backward><<<blocks, kBlockSize, 0, stream>>>(
      B, Nh, T, Dh, strides_of(x), cs, ss,
      x.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
      output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
}  // namespace

torch::Tensor rope_forward(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
  return launch_rope<false>(x, cos, sin);
}

torch::Tensor rope_backward(torch::Tensor gradient, torch::Tensor cos, torch::Tensor sin) {
  return launch_rope<true>(gradient, cos, sin);
}
