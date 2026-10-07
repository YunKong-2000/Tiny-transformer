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

namespace attention_bf16 {
using namespace cute;
using Element = cute::bfloat16_t;
constexpr int DH = 64, BQ = 64, BK = 32, BH = 32;
constexpr int THREADS = 128, STAGES = 2;
constexpr int VEC_LEN = 8, MMA_N = 8, MMA_K = 16;
constexpr int TM = 2, TN = 2, LANES_PER_ROW = 4;
constexpr int FEATURE_TILES = DH / BH;
constexpr int SCORE_N_ITER = BK / MMA_N;
constexpr int OUTPUT_N_ITER = DH / MMA_N;
constexpr int PV_N_ITER = BH / MMA_N;
constexpr float SCALE = 0.125f; // 1 / sqrt(DH).

// The fragment conversion and swizzles below are for this fixed tile family.
static_assert(DH == 64 && BQ == 64 && BK == 32 && BH == 32);
static_assert(THREADS == 128 && STAGES == 2);
static_assert(DH % BH == 0 && BH % MMA_K == 0 && BK % MMA_K == 0);
static_assert(VEC_LEN * sizeof(Element) == 16);

// BF16 operands, FP32 accumulators, four warps along the query dimension.
using TiledMma = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{}));

using QLayout = decltype(composition(
    Swizzle<3, 3, 3>{}, Layout<Shape<Int<BQ>, Int<DH>>, Stride<Int<DH>, _1>>{}));
// A 32-element row has four 16-byte vectors: XOR its two vector-index bits.
using KVLayout = decltype(composition(
    Swizzle<2, 3, 2>{}, Layout<Shape<Int<BK>, Int<BH>>, Stride<Int<BH>, _1>>{}));
using VTransposedLayout = decltype(composition(
    Swizzle<2, 3, 2>{}, Layout<Shape<Int<BH>, Int<BK>>, Stride<_1, Int<BH>>>{}));

struct alignas(16) SharedStorage {
  Element q[BQ * DH];
  // K and V are consumed in separate phases. Each stage changes ownership
  // only after every thread has finished reading the previous subtile.
  union {
    Element k[BK * BH];
    Element v[BK * BH];
  } kv[STAGES];
};
static_assert(sizeof(SharedStorage) == 12 * 1024);
static_assert(cosize_v<QLayout> == BQ * DH);
static_assert(cosize_v<KVLayout> == BK * BH);
static_assert(cosize_v<VTransposedLayout> == BK * BH);

// Atom layouts encode row + 16*column. Two adjacent C N=8 fragments form
// one A K=16 fragment in the same thread; no shuffle or shared-memory P.
constexpr bool compatible_probability_fragment() {
  using Traits = MMA_Traits<SM80_16x8x16_F32BF16BF16F32_TN>;
  typename Traits::ALayout a{};
  typename Traits::CLayout c{};
  for (int lane = 0; lane < 32; ++lane) {
    for (int value = 0; value < 8; ++value) {
      if (a(make_coord(lane, value)) !=
          c(make_coord(lane, value % 4)) + (value / 4) * 16 * MMA_N)
        return false;
    }
  }
  return true;
}
static_assert(compatible_probability_fragment());

// Source rows always have global stride DH; COLS is only the copied width.
// Invalid token rows are zero-filled. All valid source/target vectors are 16B aligned.
template <int ROWS, int COLS, class SmemLayout>
__device__ __forceinline__ void
load_subtile_async(const Element* base, int64_t row0, int64_t rows,
                   int feature0, Element* dst, SmemLayout layout) {
  static_assert(COLS % VEC_LEN == 0);
  constexpr int vectors_per_row = COLS / VEC_LEN;
  CUTE_UNROLL
  for (int vector = int(threadIdx.x); vector < ROWS * vectors_per_row; vector += THREADS) {
    const int r = vector / vectors_per_row;
    const int d = (vector % vectors_per_row) * VEC_LEN;
    const bool valid = row0 + r < rows;
    const Element* src = valid ? base + (row0 + r) * DH + feature0 + d : base;
    Element* target = dst + layout(make_coord(r, d));
    SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
        *reinterpret_cast<const uint128_t*>(src),
        *reinterpret_cast<uint128_t*>(target), valid);
  }
}

__device__ __forceinline__ void
load_k_async(const Element* k, int64_t key0, int64_t tk,
             int feature0, SharedStorage& storage, int stage) {
  load_subtile_async<BK, BH>(k, key0, tk, feature0, storage.kv[stage].k, KVLayout{});
  cp_async_fence();
}

__device__ __forceinline__ void
load_v_async(const Element* v, int64_t key0, int64_t tk,
             int feature0, SharedStorage& storage, int stage) {
  load_subtile_async<BK, BH>(v, key0, tk, feature0, storage.kv[stage].v, KVLayout{});
  cp_async_fence();
}

// QK: consume Q[:, feature0:feature0+BH] and K[:, feature0:feature0+BH].
// Only a K=16 register fragment of Q is live at a time, not the full Q tile.
template <class QTensor, class KTensor, class STensor>
__device__ __forceinline__ void
gemm_qk_subtile(const QTensor& sQ, const KTensor& sK, STensor& acc) {
  TiledMma mma;
  auto thr = mma.get_slice(threadIdx.x);
  auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, mma);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, Element>{}, mma);
  auto thr_a = copy_a.get_slice(threadIdx.x);
  auto thr_b = copy_b.get_slice(threadIdx.x);
  CUTE_UNROLL
  for (int ki = 0; ki < BH / MMA_K; ++ki) {
    auto a = local_tile(sQ, Shape<Int<BQ>, Int<MMA_K>>{}, make_coord(0, ki));
    auto b = local_tile(sK, Shape<Int<BK>, Int<MMA_K>>{}, make_coord(0, ki));
    auto rA = thr.partition_fragment_A(a);
    auto rB = thr.partition_fragment_B(b);
    auto tAs = thr_a.partition_S(a);
    auto tBs = thr_b.partition_S(b);
    auto tAr = thr_a.retile_D(rA);
    auto tBr = thr_b.retile_D(rB);
    copy(copy_a, tAs, tAr);
    copy(copy_b, tBs, tBr);
    gemm(mma, rA, rB, acc);
  }
}

// PV: output width is BH, while the reduction dimension is BK (key tokens).
template <class PTensor, class VTensor, class OTensor>
__device__ __forceinline__ void
gemm_pv_subtile(const PTensor& rP, const VTensor& sVt, OTensor& acc) {
  TiledMma mma;
  auto thr = mma.get_slice(threadIdx.x);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, Element>{}, mma);
  auto thr_b = copy_b.get_slice(threadIdx.x);
  CUTE_STATIC_ASSERT_V(size<0>(rP) == Int<8>{});
  CUTE_STATIC_ASSERT_V(size<2>(rP) == Int<BK / MMA_K>{});
  CUTE_STATIC_ASSERT_V(size<2>(acc) == Int<PV_N_ITER>{});
  CUTE_UNROLL
  for (int ki = 0; ki < BK / MMA_K; ++ki) {
    auto b = local_tile(sVt, Shape<Int<BH>, Int<MMA_K>>{}, make_coord(0, ki));
    auto rB = thr.partition_fragment_B(b);
    auto tBs = thr_b.partition_S(b);
    auto tBr = thr_b.retile_D(rB);
    copy(copy_b, tBs, tBr);
    gemm(mma, rP(_, _, ki), rB(_, _, 0), acc);
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
  const int64_t q_end = q0 + BQ < tq ? q0 + BQ : tq;
  const int64_t kv_tiles = (past_len + q_end - 1) / BK + 1;

  auto sQ = make_tensor(make_smem_ptr(storage.q), QLayout{});
  TiledMma mma;
  auto thr = mma.get_slice(threadIdx.x);
  auto score_coords = make_identity_tensor(Shape<Int<BQ>, Int<BK>>{});
  auto output_coords = make_identity_tensor(Shape<Int<BQ>, Int<DH>>{});
  auto tS = thr.partition_C(score_coords);
  auto tO = thr.partition_C(output_coords);
  auto rO = thr.make_fragment_C(tO);
  CUTE_STATIC_ASSERT_V(size<0>(tS) == Int<TM * TN>{});
  CUTE_STATIC_ASSERT_V(size<1>(tS) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<2>(tS) == Int<SCORE_N_ITER>{});
  CUTE_STATIC_ASSERT_V(size<2>(rO) == Int<OUTPUT_N_ITER>{});
  clear(rO);
  float m[TM] = {-CUDART_INF_F, -CUDART_INF_F};
  float l[TM] = {0.f, 0.f};

  load_subtile_async<BQ, DH>(q, q0, tq, 0, storage.q, QLayout{});
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads(); // Q remains read-only in shared memory for the whole CTA.

  for (int64_t tile = 0; tile < kv_tiles; ++tile) {
    const int64_t key0 = tile * BK;
    // An A-shaped register fragment of P; no memory is read from this identity tensor.
    auto rP = thr.partition_fragment_A(score_coords);
    {
      auto rS = thr.make_fragment_C(tS);
      clear(rS); // Clear once per KV tile, not once per feature subtile.
      load_k_async(k, key0, tk, 0, storage, 0);
      cp_async_wait<0>();
      __syncthreads();

      for_each(make_seq<FEATURE_TILES>{}, [&](auto feature) {
        constexpr int f = decltype(feature)::value;
        constexpr int stage = f % STAGES;
        if constexpr (f + 1 < FEATURE_TILES)
          load_k_async(k, key0, tk, (f + 1) * BH, storage, (f + 1) % STAGES);
        auto q_part = local_tile(sQ, Shape<Int<BQ>, Int<BH>>{}, make_coord(0, feature));
        auto sK = make_tensor(make_smem_ptr(storage.kv[stage].k), KVLayout{});
        gemm_qk_subtile(q_part, sK, rS);
        if constexpr (f + 1 < FEATURE_TILES) cp_async_wait<0>();
        __syncthreads(); // Publish next K; the last iteration releases both union stages.
      });

      // K is fully consumed. Overlap the first V subtile load with softmax.
      load_v_async(v, key0, tk, 0, storage, 0);
      CUTE_UNROLL
      for (int a = 0; a < TM; ++a) {
        const int row = get<0>(tS(TN * a, 0, 0));
        const int64_t qi = q0 + row;
        float local_max = -CUDART_INF_F;
        CUTE_UNROLL
        for (int n = 0; n < SCORE_N_ITER; ++n) {
          CUTE_UNROLL
          for (int b = 0; b < TN; ++b) {
            const int vi = TN * a + b;
            const int64_t kj = key0 + get<1>(tS(vi, 0, n));
            const bool valid = qi < tq && kj < tk && kj <= past_len + qi;
            rS(vi, 0, n) = valid ? rS(vi, 0, n) * SCALE : -CUDART_INF_F;
            local_max = fmaxf(local_max, rS(vi, 0, n));
          }
        }
        const float m_new = fmaxf(m[a], row_max(local_max));
        const float alpha = m[a] == -CUDART_INF_F ? 0.f : __expf(m[a] - m_new);
        float local_sum = 0.f;
        CUTE_UNROLL
        for (int n = 0; n < SCORE_N_ITER; ++n) {
          CUTE_UNROLL
          for (int b = 0; b < TN; ++b) {
            const int vi = TN * a + b;
            const float score = rS(vi, 0, n);
            const float p = score == -CUDART_INF_F ? 0.f : __expf(score - m_new);
            local_sum += p;
            rP(vi + (n % 2) * (TM * TN), 0, n / 2) = Element(p);
          }
        }
        // O has DH columns, not BK columns. Rescale every column exactly once
        // before the separate output-feature loop accumulates P @ V_subtile.
        CUTE_UNROLL
        for (int n = 0; n < OUTPUT_N_ITER; ++n) {
          CUTE_UNROLL
          for (int b = 0; b < TN; ++b) rO(TN * a + b, 0, n) *= alpha;
        }
        l[a] = alpha * l[a] + row_sum(local_sum);
        m[a] = m_new;
      }
    } // rS is no longer needed while PV holds rP and the full rO.

    cp_async_wait<0>();
    __syncthreads(); // The prefetched V[0] is now ready for all warps.
    for_each(make_seq<FEATURE_TILES>{}, [&](auto feature) {
      constexpr int f = decltype(feature)::value;
      constexpr int stage = f % STAGES;
      if constexpr (f + 1 < FEATURE_TILES)
        load_v_async(v, key0, tk, (f + 1) * BH, storage, (f + 1) % STAGES);
      auto sVt = make_tensor(make_smem_ptr(storage.kv[stage].v), VTransposedLayout{});
      // Compile-time feature index keeps this a register view, not a dynamically indexed array.
      auto out_part = local_tile(rO, Shape<Int<TM * TN>, _1, Int<PV_N_ITER>>{},
                                 make_coord(_0{}, _0{}, feature));
      gemm_pv_subtile(rP, sVt, out_part);
      if constexpr (f + 1 < FEATURE_TILES) cp_async_wait<0>();
      __syncthreads(); // Release V before the next KV tile can write K into the union.
    });
  }

  CUTE_UNROLL
  for (int a = 0; a < TM; ++a) {
    const int64_t qi = q0 + get<0>(tO(TN * a, 0, 0));
    if (qi < tq) {
      CUTE_UNROLL
      for (int n = 0; n < OUTPUT_N_ITER; ++n) {
        CUTE_UNROLL
        for (int b = 0; b < TN; ++b) {
          const int vi = TN * a + b;
          const int d = get<1>(tO(vi, 0, n));
          o[qi * DH + d] = Element(l[a] > 0.f ? rO(vi, 0, n) / l[a] : 0.f);
        }
      }
      if (threadIdx.x % LANES_PER_ROW == 0)
        lse[qi] = l[a] > 0.f ? m[a] + logf(l[a]) : -CUDART_INF_F;
    }
  }
}

} // namespace attention_bf16
