#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cmath>
#include <cstdint>

// Educational SM80 forward template. Compile/validate on an A100 before use.
// BF16 Q/K/V/O, FP32 score/softmax/output accumulation and natural-log LSE.
// Contiguous [B,H,T,64], causal Tk=past_len+Tq, positive sizes, finite inputs.
// No dropout, segments, GQA, backward or optimized single-query decode path.
// Inputs must be 16-byte aligned; launch exactly 128 threads, dynamic smem 48 KiB.
namespace attention_sm80_example {

using namespace cute;
using Element = cute::bfloat16_t;
constexpr int DH = 64, BQ = 64, BK = 64, THREADS = 128, STAGES = 2;

using TiledMma = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{}));

// Preserve the lowest 3 BF16 address bits: each cp.async moves 8 contiguous
// elements (16 bytes). XOR the 8-element groups using the low row bits.
using RowLayout = decltype(composition(
    Swizzle<3, 3, 3>{}, Layout<Shape<_64, _64>, Stride<_64, _1>>{}));
// Same V bytes; view B as (output feature, key), not (key, output feature).
using VTransposedLayout = decltype(composition(
    Swizzle<3, 3, 3>{}, Layout<Shape<_64, _64>, Stride<_1, _64>>{}));

struct alignas(16) SharedStorage {
  Element q[BQ * DH];
  Element k[STAGES][BK * DH];
  Element v[STAGES][BK * DH];
  Element p[BQ * BK];
};
static_assert(sizeof(SharedStorage) == 48 * 1024);
static_assert(cosize_v<RowLayout> == 64 * 64);
static_assert(cosize_v<VTransposedLayout> == 64 * 64);

// All 128 threads call this. A complete row has 8 aligned vectors, and each
// thread copies 4 vectors. Invalid token rows use cp.async zero-fill. Keep the
// source pointer in bounds even when src_size=0 (no out-of-range C++ reference).
__device__ __forceinline__ void
load_tile_async(const Element* base, int64_t row0, int64_t rows, Element* dst) {
  RowLayout layout;
  CUTE_UNROLL
  for (int vector = int(threadIdx.x); vector < 64 * 8; vector += THREADS) {
    const int r = vector / 8;
    const int d = (vector % 8) * 8;
    const bool valid = row0 + r < rows;
    const Element* src = valid ? base + (row0 + r) * DH + d : base;
    Element* target = dst + layout(make_coord(r, d));
    SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
        *reinterpret_cast<const uint128_t*>(src),
        *reinterpret_cast<uint128_t*>(target), valid);
  }
}

__device__ __forceinline__ void
load_kv_async(const Element* k, const Element* v, int64_t key0, int64_t tk,
              SharedStorage& storage, int stage) {
  load_tile_async(k, key0, tk, storage.k[stage]);
  load_tile_async(v, key0, tk, storage.v[stage]);
  cp_async_fence(); // One committed group contains both K and V for this block.
}

// sA:(64,64), sB:(64,64), C:(4,1,8) per thread. Move just one K=16
// fragment at a time into registers. The B copy atom distinguishes K from V^T.
template <class ATensor, class BTensor, class CTensor, class BCopyAtom>
__device__ __forceinline__ void
gemm_smem(const ATensor& sA, const BTensor& sB, CTensor& acc, BCopyAtom b_atom) {
  TiledMma mma;
  auto thr_mma = mma.get_slice(threadIdx.x);
  auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, mma);
  auto copy_b = make_tiled_copy_B(b_atom, mma);
  auto thr_a = copy_a.get_slice(threadIdx.x);
  auto thr_b = copy_b.get_slice(threadIdx.x);

  CUTE_UNROLL
  for (int ki = 0; ki < 4; ++ki) {
    auto a = local_tile(sA, Shape<_64, _16>{}, make_coord(0, ki));
    auto b = local_tile(sB, Shape<_64, _16>{}, make_coord(0, ki));
    auto rA = thr_mma.partition_fragment_A(a);
    auto rB = thr_mma.partition_fragment_B(b);
    auto tAs = thr_a.partition_S(a);
    auto tBs = thr_b.partition_S(b);
    auto tAr = thr_a.retile_D(rA);
    auto tBr = thr_b.retile_D(rB);
    copy(copy_a, tAs, tAr); // ldmatrix.x4
    copy(copy_b, tBs, tBr); // ldmatrix.x2 or ldmatrix.x2.trans
    gemm(mma, rA, rB, acc); // mma.sync.aligned.m16n8k16 ... f32.bf16.bf16.f32
  }
}

// SM80 C fragment: v=0,1 share a row; v=2,3 share the row eight below.
// A row spans the 4 contiguous lanes having the same lane/4.
__device__ __forceinline__ float row_max(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
  return fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
}
__device__ __forceinline__ float row_sum(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  return x + __shfl_xor_sync(0xffffffffu, x, 2);
}

__global__ __launch_bounds__(THREADS)
void forward(const Element* q, const Element* k, const Element* v,
             Element* o, float* lse, int64_t tq, int64_t tk, int64_t past_len) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto& storage = *reinterpret_cast<SharedStorage*>(shared_bytes);
  const int64_t q_tiles = (tq - 1) / BQ + 1;
  const int64_t bh = int64_t(blockIdx.x) / q_tiles;
  const int64_t q0 = (int64_t(blockIdx.x) % q_tiles) * BQ;
  q += bh * tq * DH;
  k += bh * tk * DH;
  v += bh * tk * DH;
  o += bh * tq * DH;
  lse += bh * tq;
  // Skip KV blocks that are entirely in the future of this query tile.
  const int64_t q_end = q0 + BQ < tq ? q0 + BQ : tq;
  const int64_t key_end = past_len + q_end;
  const int64_t kv_tiles = (key_end - 1) / BK + 1;

  auto sQ = make_tensor(make_smem_ptr(storage.q), RowLayout{});
  auto sP = make_tensor(make_smem_ptr(storage.p), RowLayout{});
  TiledMma mma;
  auto thr = mma.get_slice(threadIdx.x);
  auto coords = make_identity_tensor(Shape<_64, _64>{});
  auto tCoords = thr.partition_C(coords);
  auto rS = thr.make_fragment_C(tCoords);
  auto rO = thr.make_fragment_C(tCoords);
  CUTE_STATIC_ASSERT_V(size<0>(rS) == Int<4>{});
  CUTE_STATIC_ASSERT_V(size<1>(rS) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<2>(rS) == Int<8>{});
  clear(rO);
  float m[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float l[2] = {0.f, 0.f};

  // Prologue: one group contains Q and KV[0]; all lanes commit and wait.
  load_tile_async(q, q0, tq, storage.q);
  load_kv_async(k, v, 0, tk, storage, 0);
  cp_async_wait<0>();
  __syncthreads(); // A per-thread async wait alone is not a CTA rendezvous.

  for (int64_t tile = 0; tile < kv_tiles; ++tile) {
    const int stage = int(tile % STAGES);
    const bool has_next = tile + 1 < kv_tiles;
    if (has_next) {
      // Genuine overlap: load the next KV block while QK/softmax/PV use this one.
      load_kv_async(k, v, (tile + 1) * BK, tk, storage, stage ^ 1);
    }
    auto sK = make_tensor(make_smem_ptr(storage.k[stage]), RowLayout{});
    auto sVt = make_tensor(make_smem_ptr(storage.v[stage]), VTransposedLayout{});
    clear(rS);
    gemm_smem(sQ, sK, rS, Copy_Atom<SM75_U32x2_LDSM_N, Element>{});

    CUTE_UNROLL
    for (int a = 0; a < 2; ++a) {
      const int row = get<0>(tCoords(2 * a, 0, 0));
      const int64_t qi = q0 + row;
      float local_max = -CUDART_INF_F;
      CUTE_UNROLL
      for (int n = 0; n < 8; ++n) {
        CUTE_UNROLL
        for (int b = 0; b < 2; ++b) {
          const int vi = 2 * a + b;
          const int64_t kj = tile * BK + get<1>(tCoords(vi, 0, n));
          const bool valid = qi < tq && kj < tk && kj <= past_len + qi;
          rS(vi, 0, n) = valid ? rS(vi, 0, n) * 0.125f : -CUDART_INF_F;
          local_max = fmaxf(local_max, rS(vi, 0, n));
        }
      }
      const float next_m = fmaxf(m[a], row_max(local_max));
      const float alpha = m[a] == -CUDART_INF_F ? 0.f : expf(m[a] - next_m);
      float local_sum = 0.f;
      CUTE_UNROLL
      for (int n = 0; n < 8; ++n) {
        CUTE_UNROLL
        for (int b = 0; b < 2; ++b) {
          const int vi = 2 * a + b;
          const float score = rS(vi, 0, n);
          const float p = score == -CUDART_INF_F ? 0.f : expf(score - next_m);
          local_sum += p; // Statistics stay FP32, before BF16 conversion.
          const int col = get<1>(tCoords(vi, 0, n));
          sP(row, col) = Element(p);
          rO(vi, 0, n) *= alpha; // Once per KV block, for every output element.
        }
      }
      l[a] = alpha * l[a] + row_sum(local_sum);
      m[a] = next_m;
    }
    __syncthreads(); // All P stores must be visible before ldmatrix reads.
    gemm_smem(sP, sVt, rO, Copy_Atom<SM75_U16x4_LDSM_T, Element>{});
    __syncthreads(); // Release current KV stage and P before either is overwritten.

    if (has_next) {
      cp_async_wait<0>(); // Only the prefetched next block is outstanding.
      __syncthreads();   // Make that stage consumable by the whole CTA.
    }
  }

  CUTE_UNROLL
  for (int a = 0; a < 2; ++a) {
    const int64_t qi = q0 + get<0>(tCoords(2 * a, 0, 0));
    if (qi < tq) {
      CUTE_UNROLL
      for (int n = 0; n < 8; ++n) {
        CUTE_UNROLL
        for (int b = 0; b < 2; ++b) {
          const int vi = 2 * a + b;
          const int d = get<1>(tCoords(vi, 0, n));
          o[qi * DH + d] = Element(l[a] > 0.f ? rO(vi, 0, n) / l[a] : 0.f);
        }
      }
      if (threadIdx.x % 4 == 0)
        lse[qi] = l[a] > 0.f ? m[a] + logf(l[a]) : -CUDART_INF_F;
    }
  }
}

} // namespace attention_sm80_example
