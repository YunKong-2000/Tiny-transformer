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
// A 32-BF16 row advances 16 banks. Its low row bit already selects a bank
// half, so XOR row bits 1/2 (element-address bits 6/7) into vector bits 3/4.
// Shift=2 would reuse the low row bit and collide again after four rows.
using KVLayout = decltype(composition(
    Swizzle<2, 3, 3>{}, Layout<Shape<Int<BK>, Int<BH>>, Stride<Int<BH>, _1>>{}));
using VTransposedLayout = decltype(composition(
    Swizzle<2, 3, 3>{}, Layout<Shape<Int<BH>, Int<BK>>, Stride<_1, Int<BH>>>{}));

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
// Keep B's N fragments in one CuTe GEMM. Manually expanding one N=8 MMA at
// a time did not reduce compiled register usage in the measured build.
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

// The full-tile specialization has no per-element coordinate/mask operations.
// Keep only this small loop specialized; share the online-softmax/PV code so
// that the fast path does not duplicate the entire kernel body.
template <bool FULL_TILE, class STensor, class CoordTensor>
__device__ __forceinline__ void
scale_mask_scores(STensor& scores, const CoordTensor& coords,
                  int64_t q0, int64_t key0, int64_t tq, int64_t tk, int64_t past_len) {
  CUTE_UNROLL
  for (int a = 0; a < TM; ++a) {
    CUTE_UNROLL
    for (int n = 0; n < SCORE_N_ITER; ++n) {
      CUTE_UNROLL
      for (int b = 0; b < TN; ++b) {
        const int vi = TN * a + b;
        if constexpr (FULL_TILE) {
          scores(vi, 0, n) *= SCALE;
        } else {
          const int64_t qi = q0 + get<0>(coords(vi, 0, n));
          const int64_t kj = key0 + get<1>(coords(vi, 0, n));
          const bool valid = qi < tq && kj < tk && kj <= past_len + qi;
          scores(vi, 0, n) = valid ? scores(vi, 0, n) * SCALE : -CUDART_INF_F;
        }
      }
    }
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
  // Enumerate long causal tiles first, across all heads. This is a scheduling
  // hint via block numbering; CUDA does not guarantee block execution order.
  const int64_t batch_heads = int64_t(gridDim.x) / q_tiles;
  const int64_t bh = int64_t(blockIdx.x) % batch_heads;
  const int64_t q0 = (q_tiles - 1 - int64_t(blockIdx.x) / batch_heads) * BQ;
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
  load_k_async(k, 0, tk, 0, storage, 0); // Commit Q and first K in one group.
  cp_async_wait<0>();
  __syncthreads(); // Q and K[0] ready; Q stays read-only for the whole CTA.

  for (int64_t tile = 0; tile < kv_tiles; ++tile) {
    const int64_t key0 = tile * BK;
    // Identity tensors have ScaledBasis strides, which make_fragment_like cannot
    // order. Keep only the partition's shape and allocate ordinary compact
    // register storage, with the atom value dimension contiguous.
    auto rP = make_tensor<Element>(shape(thr.partition_A(score_coords)));
    {
      auto rS = thr.make_fragment_C(tS);
      clear(rS); // Clear once per KV tile, not once per feature subtile.
      // K[0] is ready from the prologue or the preceding tile's last PV step.

      for_each(make_seq<FEATURE_TILES>{}, [&](auto feature) {
        constexpr int f = decltype(feature)::value;
        constexpr int stage = f % STAGES;
        if constexpr (f + 1 < FEATURE_TILES)
          load_k_async(k, key0, tk, (f + 1) * BH, storage, (f + 1) % STAGES);
        else {
          static_assert(FEATURE_TILES == 2);
          // The preceding barrier released stage 0. Load V[0] while K[1]
          // is consumed from stage 1, then continue overlapping with softmax.
          load_v_async(v, key0, tk, 0, storage, 0);
        }
        auto q_part = local_tile(sQ, Shape<Int<BQ>, Int<BH>>{}, make_coord(0, feature));
        auto sK = make_tensor(make_smem_ptr(storage.kv[stage].k), KVLayout{});
        gemm_qk_subtile(q_part, sK, rS);
        if constexpr (f + 1 < FEATURE_TILES) {
          cp_async_wait<0>();
          __syncthreads(); // Publish K[1] and release K[0] before V[0] overwrites it.
        }
        // The V-ready barrier after softmax also releases the last K stage.
      });

      // All rows must exist, and even the first query must see the last key.
      // Subtractions avoid constructing a padded key endpoint beyond int64_t.
      // This branch is uniform across the CTA; padded query rows take the mask path.
      const bool fully_valid = q_end - q0 == BQ && tk - key0 >= BK &&
                               past_len + q0 - key0 >= BK - 1;
      if (fully_valid)
        scale_mask_scores<true>(rS, tS, q0, key0, tq, tk, past_len);
      else
        scale_mask_scores<false>(rS, tS, q0, key0, tq, tk, past_len);

      // V[0] is already in flight; softmax needs no shared-memory operands.
      CUTE_UNROLL
      for (int a = 0; a < TM; ++a) {
        float local_max = -CUDART_INF_F;
        CUTE_UNROLL
        for (int n = 0; n < SCORE_N_ITER; ++n) {
          CUTE_UNROLL
          for (int b = 0; b < TN; ++b) {
            const int vi = TN * a + b;
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
      else if (tile + 1 < kv_tiles) {
        // V[0] has been released. Prefetch next tile's K[0] into that stage
        // while current V[1] is consumed; no extra shared-memory buffer.
        load_k_async(k, key0 + BK, tk, 0, storage, 0);
      }
      auto sVt = make_tensor(make_smem_ptr(storage.kv[stage].v), VTransposedLayout{});
      // Compile-time feature index keeps this a register view, not a dynamically indexed array.
      auto out_part = local_tile(rO, Shape<Int<TM * TN>, _1, Int<PV_N_ITER>>{},
                                 make_coord(_0{}, _0{}, feature));
      gemm_pv_subtile(rP, sVt, out_part);
      if constexpr (f + 1 < FEATURE_TILES) {
        cp_async_wait<0>();
        __syncthreads(); // Publish V[1] and release V[0] before next K[0].
      } else if (tile + 1 < kv_tiles) {
        cp_async_wait<0>();
        __syncthreads(); // Publish next K[0], release V[1] before next K[1].
      }
      // Last tile: no further shared-memory reads/writes need a rendezvous.
    });
  }

  CUTE_UNROLL
  for (int a = 0; a < TM; ++a) {
    const int64_t qi = q0 + get<0>(tO(TN * a, 0, 0));
    if (qi < tq) {
      CUTE_UNROLL
      for (int n = 0; n < OUTPUT_N_ITER; ++n) {
        static_assert(TN == 2);
        const int vi = TN * a;
        const int d = get<1>(tO(vi, 0, n));
        const Element lo(l[a] > 0.f ? rO(vi, 0, n) / l[a] : 0.f);
        const Element hi(l[a] > 0.f ? rO(vi + 1, 0, n) / l[a] : 0.f);
        // The two atom values are consecutive columns with an even first d.
        // Pack their exact BF16 bits into one aligned store instead of two STG.U16.
        const uint32_t packed = uint32_t(lo.raw()) | (uint32_t(hi.raw()) << 16);
        *reinterpret_cast<uint32_t*>(o + qi * DH + d) = packed;
      }
      if (threadIdx.x % LANES_PER_ROW == 0)
        lse[qi] = l[a] > 0.f ? m[a] + logf(l[a]) : -CUDART_INF_F;
    }
  }
}

} // namespace attention_bf16
