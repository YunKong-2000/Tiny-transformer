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


namespace attention_bf16{
using namespace cute;
using Element = cute::bfloat16_t;
constexpr int DH = 64, BQ = 64, BK = 64, THREADS = 128, STAGES = 2;
constexpr int VEC_LEN = 8;
constexpr int VEC_NUM_ROW = DH / VEC_LEN;// 8
constexpr int TM = 2;
constexpr int TN = 2;
constexpr int MMA_N_ITER = 8;
constexpr float SCALE = 0.125f; // 1 / sqrt(DH), with DH fixed to 64 below.
constexpr int LANES_PER_ROW = 4;

// These constants describe one fixed MMA layout, not independent tuning knobs.
static_assert(DH == 64 && BQ == 64 && BK == 64);
static_assert(THREADS == 128 && STAGES == 2);
static_assert(TM == 2 && TN == 2 && MMA_N_ITER == 8 && LANES_PER_ROW == 4);
static_assert(VEC_LEN * sizeof(Element) == 16 && DH % VEC_LEN == 0);

// BF16 input/output, FP32 accumulation. Four warps along the MMA M dimension.
using TiledMma = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{}));

// use swizzle to avoid bank conflict for shared memory loading.
using RowLayout = decltype(composition(
    Swizzle<3, 3, 3>{}, Layout<Shape<_64, _64>, Stride<_64, _1>>{}));
using VTransposedLayout = decltype(composition(
    Swizzle<3, 3, 3>{}, Layout<Shape<_64, _64>, Stride<_1, _64>>{}));

// Q_i use a whole tile on shared memory, 
// struct alignas(16) SharedStorage {
//   Element q[BQ * DH];
//   union {
//     Element k[BK * DH];
//     Element v[BK * DH];
//   } kv[STAGES];
//   Element p[BQ * BK];
// };

// Q_i use a whole tile on shared memory, subtile of K, V, P also on the smem
struct alignas(16) SharedStorage {
  Element q[BQ * DH];
  Element k[STAGES][BK * DH];
  Element v[STAGES][BK * DH];
  Element p[BQ * BK];
};
static_assert(sizeof(SharedStorage) == 48 * 1024);
static_assert(cosize_v<RowLayout> == 64 * 64);
static_assert(cosize_v<VTransposedLayout> == 64 * 64);

// copy data from gmem to smem using layout RowLayout.
template<int TILE_ROWS>
__device__ __forceinline__ void
load_tile_async(const Element* base, int64_t row0, int64_t rows, Element* dst) {
  RowLayout layout;
  CUTE_UNROLL
  for (int vector = int(threadIdx.x); vector < TILE_ROWS * VEC_NUM_ROW; vector += THREADS) {
    const int r = vector / VEC_NUM_ROW;
    const int d = (vector % VEC_NUM_ROW) * VEC_LEN;
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
  load_tile_async<BK>(k, key0, tk, storage.k[stage]);
  load_tile_async<BK>(v, key0, tk, storage.v[stage]);
  cp_async_fence(); // One committed group contains both K and V for this block.
}

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
  CUTE_STATIC_ASSERT_V(size<0>(rS) == Int<TM * TN>{});
  CUTE_STATIC_ASSERT_V(size<1>(rS) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<2>(rS) == Int<MMA_N_ITER>{});
  clear(rO);
  float m[TM] = {-CUDART_INF_F, -CUDART_INF_F};
  float l[TM] = {0.f, 0.f};

  // Prologue: one group contains Q and KV[0]; all lanes commit and wait.
  load_tile_async<BQ>(q, q0, tq, storage.q);
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
    for (int a = 0; a < TM; ++a) {
      
      const int row = get<0>(tCoords(TN * a, 0, 0));// get row index in S tile for a-th row in S fragment
      const int64_t qi = q0 + row;                  // get global index of this row in S fragment
      float local_max = -CUDART_INF_F;
      // reduce max for TN * MMA_N_ITER = 16 elements in a same row in S tile
      CUTE_UNROLL
      for (int n = 0; n < MMA_N_ITER; ++n) {
        CUTE_UNROLL
        for (int b = 0; b < TN; ++b) {
          const int vi = TN * a + b;
          // get global index of vi-th value in S fragment
          const int64_t kj = tile * BK + get<1>(tCoords(vi, 0, n));
          const bool valid = qi < tq && kj < tk && kj <= past_len + qi;
          rS(vi, 0, n) = valid ? rS(vi, 0, n) * SCALE : -CUDART_INF_F;
          local_max = fmaxf(local_max, rS(vi, 0, n));
        }
      }
      const float m_new = fmaxf(m[a], row_max(local_max));
      const float alpha = m[a] == -CUDART_INF_F ? 0.f : expf(m[a] - m_new);
      float local_sum = 0.f;
      // reduce sum for TN * MMA_N_ITER = 16 elements in a same row in S tile
      CUTE_UNROLL
      for (int n = 0; n < MMA_N_ITER; ++n) {
        CUTE_UNROLL
        for (int b = 0; b < TN; ++b) {
          const int vi = TN * a + b;
          const float score = rS(vi, 0, n);
          const float p = score == -CUDART_INF_F ? 0.f : expf(score - m_new);
          local_sum += p; // Statistics stay FP32, before BF16 conversion.
          const int col = get<1>(tCoords(vi, 0, n));
          sP(row, col) = Element(p);
          rO(vi, 0, n) *= alpha; // Once per KV block, for every output element.
        }
      }
      l[a] = alpha * l[a] + row_sum(local_sum);
      m[a] = m_new;
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
  for (int a = 0; a < TM; ++a) {
    const int64_t qi = q0 + get<0>(tCoords(TN * a, 0, 0));
    if (qi < tq) {
      CUTE_UNROLL
      for (int n = 0; n < MMA_N_ITER; ++n) {
        CUTE_UNROLL
        for (int b = 0; b < TN; ++b) {
          const int vi = TN * a + b;
          const int d = get<1>(tCoords(vi, 0, n));
          o[qi * DH + d] = Element(l[a] > 0.f ? rO(vi, 0, n) / l[a] : 0.f);
        }
      }
      if (threadIdx.x % LANES_PER_ROW == 0)
        lse[qi] = l[a] > 0.f ? m[a] + logf(l[a]) : -CUDART_INF_F;
    }
  }
}

} // namespace attention_bf16
