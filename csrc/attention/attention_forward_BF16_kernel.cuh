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
constexpr int DH = 64, BQ = 64, BK = 64, BH = 64;
constexpr int THREADS = 128, STAGES = 2;
constexpr int VEC_LEN = 8, MMA_N = 8, MMA_K = 16;
constexpr int TM = 2, TN = 2, LANES_PER_ROW = 4;
constexpr int FEATURE_TILES = DH / BH;
constexpr int SCORE_N_ITER = BK / MMA_N;
constexpr int OUTPUT_N_ITER = DH / MMA_N;
constexpr int PV_N_ITER = BH / MMA_N;
constexpr float SCALE = 0.125f; // 1 / sqrt(DH).
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;
constexpr float SCALE_LOG2 = SCALE * LOG2E;

// The fragment conversion and swizzles below are for this fixed tile family.
static_assert(DH == 64 && BQ == 64 && BK == 64 && BH == 64);
static_assert(FEATURE_TILES == 1);
static_assert(THREADS == 128 && STAGES == 2);
static_assert(DH % BH == 0 && BH % MMA_K == 0 && BK % MMA_K == 0);
static_assert(VEC_LEN * sizeof(Element) == 16);

// BF16 operands, FP32 accumulators, four warps along the query dimension.
using TiledMma = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{}));

using QLayout = decltype(composition(
    Swizzle<3, 3, 3>{}, Layout<Shape<Int<BQ>, Int<DH>>, Stride<Int<DH>, _1>>{}));
// A full 64-BF16 row advances 32 banks. XOR its three low row bits
// (element-address bits 6/7/8) into the 16-byte vector-index bits 3/4/5.
using KVLayout = decltype(composition(
    Swizzle<3, 3, 3>{}, Layout<Shape<Int<BK>, Int<BH>>, Stride<Int<BH>, _1>>{}));
using VTransposedLayout = decltype(composition(
    Swizzle<3, 3, 3>{}, Layout<Shape<Int<BH>, Int<BK>>, Stride<_1, Int<BH>>>{}));

struct alignas(16) SharedStorage {
  Element q[BQ * DH];
  // Two full-width slots. The pipeline keeps K in slot 0 and V in slot 1
  // so loading one can overlap consuming the other. A slot is overwritten
  // only after a CTA barrier releases all its readers.
  union {
    Element k[BK * BH];
    Element v[BK * BH];
  } kv[STAGES];
};
static_assert(sizeof(SharedStorage) == 24 * 1024);
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
// The full-tile specialization needs no row predicate or zero-fill operand.
// Otherwise invalid token rows are zero-filled. All vectors are 16B aligned.
template <int ROWS, int COLS, bool EVEN_TILES, class SmemLayout>
__device__ __forceinline__ void
load_subtile_async(const Element* base, int32_t row0, int32_t rows,
                   int feature0, Element* dst, SmemLayout layout) {
  static_assert(COLS % VEC_LEN == 0);
  constexpr int vectors_per_row = COLS / VEC_LEN;
  CUTE_UNROLL
  for (int vector = int(threadIdx.x); vector < ROWS * vectors_per_row; vector += THREADS) {
    const int r = vector / vectors_per_row;
    const int d = (vector % vectors_per_row) * VEC_LEN;
    Element* target = dst + layout(make_coord(r, d));
    // Row indices are int32; a row's element offset need not fit int32.
    if constexpr (EVEN_TILES) {
      const Element* src = base + int64_t(row0 + r) * DH + feature0 + d;
      SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>::copy(
          *reinterpret_cast<const uint128_t*>(src),
          *reinterpret_cast<uint128_t*>(target));
    } else {
      const bool valid = row0 + r < rows;
      const Element* src = valid ? base + int64_t(row0 + r) * DH + feature0 + d : base;
      SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
          *reinterpret_cast<const uint128_t*>(src),
          *reinterpret_cast<uint128_t*>(target), valid);
    }
  }
}

template <bool EVEN_TILES>
__device__ __forceinline__ void
load_k_async(const Element* k, int32_t key0, int32_t tk,
             int feature0, SharedStorage& storage, int stage) {
  load_subtile_async<BK, BH, EVEN_TILES>(k, key0, tk, feature0, storage.kv[stage].k, KVLayout{});
  cp_async_fence();
}

template <bool EVEN_TILES>
__device__ __forceinline__ void
load_v_async(const Element* v, int32_t key0, int32_t tk,
             int feature0, SharedStorage& storage, int stage) {
  load_subtile_async<BK, BH, EVEN_TILES>(v, key0, tk, feature0, storage.kv[stage].v, KVLayout{});
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

  // Two explicit register fragments, not a dynamically indexed tensor array.
  // The next ldmatrix load is issued before MMA consumes the current fragment.
  auto first_b = local_tile(sVt, Shape<Int<BH>, Int<MMA_K>>{}, make_coord(_0{}, _0{}));
  auto rB0 = thr.partition_fragment_B(first_b);
  auto rB1 = thr.partition_fragment_B(first_b);
  CUTE_STATIC_ASSERT_V(size(rB0) == Int<4 * PV_N_ITER>{});
  auto load_fragment = [&](auto ki, auto& rB) {
    auto b = local_tile(sVt, Shape<Int<BH>, Int<MMA_K>>{}, make_coord(_0{}, ki));
    auto tBs = thr_b.partition_S(b);
    auto tBr = thr_b.retile_D(rB);
    copy(copy_b, tBs, tBr);
  };

  load_fragment(_0{}, rB0);
  for_each(make_seq<BK / MMA_K>{}, [&](auto ki) {
    constexpr int k = decltype(ki)::value;
    if constexpr (k % 2 == 0) {
      if constexpr (k + 1 < BK / MMA_K) load_fragment(Int<k + 1>{}, rB1);
      gemm(mma, rP(_, _, ki), rB0(_, _, _0{}), acc);
    } else {
      // MMA k-1 has consumed rB0 before it is reused for k+1.
      if constexpr (k + 1 < BK / MMA_K) load_fragment(Int<k + 1>{}, rB0);
      gemm(mma, rP(_, _, ki), rB1(_, _, _0{}), acc);
    }
  }); // Last MMA drains the pipeline without loading beyond the BK tile.
}

// Keep valid scores unscaled: max(raw_score) * SCALE_LOG2 needs one scale
// per row, and each probability fuses its scale/subtraction into one FMA.
// Full tiles skip this helper entirely; boundary tiles only mask invalid scores.
template <bool EVEN_TILES, class STensor, class CoordTensor>
__device__ __forceinline__ void
mask_scores(STensor& scores, const CoordTensor& coords,
                  int32_t q0, int32_t key0, int32_t tq, int32_t tk, int32_t past_len) {
  CUTE_UNROLL
  for (int a = 0; a < TM; ++a) {
    CUTE_UNROLL
    for (int n = 0; n < SCORE_N_ITER; ++n) {
      CUTE_UNROLL
      for (int b = 0; b < TN; ++b) {
        const int vi = TN * a + b;
        const int32_t qi = q0 + get<0>(coords(vi, 0, n));
        const int32_t kj = key0 + get<1>(coords(vi, 0, n));
        // Subtract past instead of adding it to a potentially padded query
        // index. kj - past_len fits int32 even at Tk == INT32_MAX.
        bool valid = kj - past_len <= qi;
        if constexpr (!EVEN_TILES) valid = valid && qi < tq && kj < tk;
        if (!valid) scores(vi, 0, n) = -CUDART_INF_F;
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

// Host selects EVEN_TILES only when Tq % BQ == 0 && Tk % BK == 0.
// This removes sequence-tail guards, not the causal diagonal mask.
template <bool EVEN_TILES>
__global__ __launch_bounds__(THREADS, 4)
void forward(const Element* q, const Element* k, const Element* v,
             Element* o, float* lse, int32_t tq, int32_t tk, int32_t past_len) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto& storage = *reinterpret_cast<SharedStorage*>(shared_bytes);
  // Host bounds all lengths and grid.x by INT32_MAX. Fixed 64-row tiles
  // also keep padded indices <= INT32_MAX (the largest is 2^31 - 1).
  static_assert(BQ == 64 && BK == 64);
  const int32_t q_tiles = EVEN_TILES ? tq / BQ : (tq - 1) / BQ + 1;
  // Enumerate long causal tiles first, across all heads. This is a scheduling
  // hint via block numbering; CUDA does not guarantee block execution order.
  const int32_t batch_heads = int32_t(gridDim.x) / q_tiles;
  const int32_t bh = int32_t(blockIdx.x) % batch_heads;
  const int32_t q0 = (q_tiles - 1 - int32_t(blockIdx.x) / batch_heads) * BQ;
  q += int64_t(bh) * tq * DH;
  k += int64_t(bh) * tk * DH;
  v += int64_t(bh) * tk * DH;
  o += int64_t(bh) * tq * DH;
  lse += int64_t(bh) * tq;
  // q0+BQ can be 2^31 on a tail tile, so bound the addend first.
  const int32_t q_end = q0 + (EVEN_TILES ? BQ : (tq - q0 < BQ ? tq - q0 : BQ));
  const int32_t kv_tiles = (past_len + q_end - 1) / BK + 1;

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
  float m2[TM] = {-CUDART_INF_F, -CUDART_INF_F}; // Row maxima in base-2 score units.
  // Each lane keeps its own running contribution for its two query rows.
  // Row maxima/alpha remain shared across the four lanes, but the denominator
  // is not needed by QK, softmax weights or PV until the final normalization.
  float l_partial[TM] = {0.f, 0.f};

  load_subtile_async<BQ, DH, EVEN_TILES>(q, q0, tq, 0, storage.q, QLayout{});
  load_k_async<EVEN_TILES>(k, 0, tk, 0, storage, 0); // Commit Q and first K in one group.
  cp_async_wait<0>();
  __syncthreads(); // Q and K[0] ready; Q stays read-only for the whole CTA.

  for (int32_t tile = 0; tile < kv_tiles; ++tile) {
    const int32_t key0 = tile * BK;
    // Identity tensors have ScaledBasis strides, which make_fragment_like cannot
    // order. Keep only the partition's shape and allocate ordinary compact
    // register storage, with the atom value dimension contiguous.
    auto rP = make_tensor<Element>(shape(thr.partition_A(score_coords)));
    CUTE_STATIC_ASSERT_V(size<0>(rP) == Int<8>{});
    CUTE_STATIC_ASSERT_V(size<1>(rP) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size<2>(rP) == Int<BK / MMA_K>{});
    {
      auto rS = thr.make_fragment_C(tS);
      clear(rS); // Clear once per KV tile, not once per feature subtile.
      // The full K tile is ready from the prologue or previous PV step.
      // Prefetch full V in the other slot while QK and softmax run. The
      // preceding CTA barrier guarantees that all old V readers have finished.
      load_v_async<EVEN_TILES>(v, key0, tk, 0, storage, 1);
      auto sK = make_tensor(make_smem_ptr(storage.kv[0].k), KVLayout{});
      gemm_qk_subtile(sQ, sK, rS);

      // All rows must exist, and even the first query must see the last key.
      // Subtractions avoid constructing a padded key endpoint beyond int32.
      // This branch is uniform across the CTA; padded query rows take the mask path.
      bool fully_valid = past_len + q0 - key0 >= BK - 1;
      if constexpr (!EVEN_TILES)
        fully_valid = fully_valid && q_end - q0 == BQ && tk - key0 >= BK;
      if (!fully_valid)
        mask_scores<EVEN_TILES>(rS, tS, q0, key0, tq, tk, past_len);

      // V is already in flight; softmax needs no shared-memory operands.
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
        const float m2_new = fmaxf(m2[a], row_max(local_max) * SCALE_LOG2);
        const float alpha = m2[a] == -CUDART_INF_F ? 0.f : exp2f(m2[a] - m2_new);
        float local_sum = 0.f;
        CUTE_UNROLL
        for (int n = 0; n < SCORE_N_ITER; ++n) {
          CUTE_UNROLL
          for (int b = 0; b < TN; ++b) {
            const int vi = TN * a + b;
            const float score = rS(vi, 0, n);
            // Guard fully masked padded rows before evaluating -inf - (-inf).
            const float p = score == -CUDART_INF_F ? 0.f :
                exp2f(fmaf(score, SCALE_LOG2, -m2_new));
            local_sum += p;
            rP(vi + (n % 2) * (TM * TN), 0, n / 2) = Element(p);
          }
        }
        // O has DH columns, not BK columns. Rescale every column exactly once
        // before PV accumulates all output features from the full V tile.
        CUTE_UNROLL
        for (int n = 0; n < OUTPUT_N_ITER; ++n) {
          CUTE_UNROLL
          for (int b = 0; b < TN; ++b) rO(TN * a + b, 0, n) *= alpha;
        }
        l_partial[a] = alpha * l_partial[a] + local_sum;
        m2[a] = m2_new;
      }
    } // rS is no longer needed while PV holds rP and the full rO.

    cp_async_wait<0>();
    __syncthreads(); // Publish V[1] and release K[0] after all QK reads.
    if (tile + 1 < kv_tiles) {
      // Next K can overwrite slot 0 while current PV only reads slot 1.
      load_k_async<EVEN_TILES>(k, key0 + BK, tk, 0, storage, 0);
    }
    auto sVt = make_tensor(make_smem_ptr(storage.kv[1].v), VTransposedLayout{});
    gemm_pv_subtile(rP, sVt, rO);
    if (tile + 1 < kv_tiles) {
      cp_async_wait<0>();
      __syncthreads(); // Publish next K[0] and release V[1] before next V load.
    }
    // Last iteration has no outstanding copies and only stores register results.
  }

  CUTE_UNROLL
  for (int a = 0; a < TM; ++a) {
    // All lanes, including padded rows, must execute the full-mask shuffle.
    // Placing this inside qi<tq would make a tail warp's participation invalid.
    const float denominator = row_sum(l_partial[a]);
    const int32_t qi = q0 + get<0>(tO(TN * a, 0, 0));
    if (EVEN_TILES || qi < tq) {
      const bool has_sum = denominator > 0.f;
      const float inv_l = has_sum ? 1.0f / denominator : 0.f;
      CUTE_UNROLL
      for (int n = 0; n < OUTPUT_N_ITER; ++n) {
        static_assert(TN == 2);
        const int vi = TN * a;
        const int d = get<1>(tO(vi, 0, n));
        const Element lo(has_sum ? rO(vi, 0, n) * inv_l : 0.f);
        const Element hi(has_sum ? rO(vi + 1, 0, n) * inv_l : 0.f);
        // The two atom values are consecutive columns with an even first d.
        // Pack their exact BF16 bits into one aligned store instead of two STG.U16.
        const uint32_t packed = uint32_t(lo.raw()) | (uint32_t(hi.raw()) << 16);
        *reinterpret_cast<uint32_t*>(o + int64_t(qi) * DH + d) = packed;
      }
      if (threadIdx.x % LANES_PER_ROW == 0)
        // Public LSE is a natural logarithm, even though internal maxima use base 2.
        lse[qi] = has_sum ? fmaf(m2[a], LN2, logf(denominator)) : -CUDART_INF_F;
    }
  }
}

} // namespace attention_bf16
