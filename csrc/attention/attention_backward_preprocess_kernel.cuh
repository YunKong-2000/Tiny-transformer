#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy.hpp>
#include <cstdint>

namespace attention_backward_preprocess {
using namespace cute;
using Element = cute::bfloat16_t;
constexpr int DH = 64;
constexpr int THREADS = 256;
constexpr int WARPSIZE = 32;
constexpr int WARP_M = THREADS / WARPSIZE;
constexpr int BQ = WARP_M;

using ThreadLayout = decltype(
  Layout<Shape<Int<WARP_M>, Int<WARPSIZE>>,
         Stride<Int<WARPSIZE>, Int<1>>
  >{}
);

using ValueLayout = decltype(
  Layout<Shape<Int<1>, Int<2>>,
         Stride<Int<2>, Int<1>>
  >{}
);

using TiledCopy = decltype(make_tiled_copy(
  // Each lane copies two adjacent BF16 values (32 bits).
  Copy_Atom<UniversalCopy<uint32_t>, Element>{},
  ThreadLayout{},
  ValueLayout{})
);

__device__ __forceinline__ float warp_reduce_sum(float local_sum) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);
  }
  return local_sum;
}

// Contract: contiguous BF16 inputs [B, Nh, Tq, DH], aligned to 4 bytes;
// contiguous FP32 output [B, Nh, Tq], with no overlap with the inputs.
// Launch with block=(THREADS, 1, 1), grid=(ceil(Tq / BQ), Nh, B).
// The host must validate positive sizes and device grid limits before launch.
__global__ void kernel
(
  const int64_t B,
  const int64_t Nh,
  const int64_t Tq,
  const Element* dO,
  const Element* O,
  float* D
)
{
  const int warp = threadIdx.x / WARPSIZE;
  const int lane = threadIdx.x % WARPSIZE;
  const int64_t b = blockIdx.z;
  const int64_t h = blockIdx.y;
  const int64_t q = int64_t(blockIdx.x) * BQ + warp;

  // One entire warp owns a row. Invalid rows exit before either global load;
  // every remaining warp has all 32 lanes active for the full-mask shuffle.
  // There is no block-wide barrier after this return.
  if (b >= B || h >= Nh || q >= Tq) {
    return;
  }

  auto OLayout = make_layout(make_shape(B, Nh, Tq, Int<DH>{}), LayoutRight{});
  auto DLayout = make_layout(make_shape(B, Nh, Tq), LayoutRight{});
  auto gdO = make_tensor(make_gmem_ptr(dO), OLayout);
  auto gO = make_tensor(make_gmem_ptr(O), OLayout);
  auto gD = make_tensor(make_gmem_ptr(D), DLayout);

  auto gdO_tile = local_tile(
    gdO,
    Shape<Int<1>, Int<1>, Int<BQ>, Int<DH>>{},
    make_coord(blockIdx.z, blockIdx.y, blockIdx.x, 0)
  );
  auto gO_tile = local_tile(
    gO,
    Shape<Int<1>, Int<1>, Int<BQ>, Int<DH>>{},
    make_coord(blockIdx.z, blockIdx.y, blockIdx.x, 0)
  );
  TiledCopy tiled_copy;
  auto thr_copy = tiled_copy.get_slice(threadIdx.x);
  // Remove the singleton batch/head modes before partitioning (BQ, DH).
  // Thread tid owns row tid / 32 and columns 2 * (tid % 32) + {0, 1}.
  auto tdO = thr_copy.partition_S(gdO_tile(0, 0, _, _));
  auto tO = thr_copy.partition_S(gO_tile(0, 0, _, _));
  CUTE_STATIC_ASSERT_V(size(tdO) == Int<2>{});
  CUTE_STATIC_ASSERT_V(size(tO) == Int<2>{});

  auto rdO = make_fragment_like(tdO);
  auto rO = make_fragment_like(tO);

  copy(tiled_copy, tdO, rdO);
  copy(tiled_copy, tO, rO);

  float local_sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < size(rdO); i++) {
    local_sum += float(rdO(i)) * float(rO(i));
  }
  const float row_sum = warp_reduce_sum(local_sum);
  if (lane == 0) {
    // gD is a global-memory CuTe tensor; this performs the actual store.
    gD(b, h, q) = row_sum;
  }
}

} // namespace attention_backward_preprocess
