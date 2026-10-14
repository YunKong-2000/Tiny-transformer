#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>

// scale = 1 / sqrt(head_dim)

// S  = scale * Q @ K.T
// P  = exp(S - LSE[:, None])        # 非法位置设为 0
// dP = dO @ V.T
// dS = P * (dP - D[:, None])
// dQ = scale * dS @ K

namespace attention_backward_dQ{
using namespace cute;
using Element = cute::bfloat16_t;
constexpr int DH = 64;
constexpr int THREADS = 128;
constexpr int WARPSIZE = 32;
constexpr int WARP_M = THREADS / WARPSIZE;
constexpr int BQ = 64;
constexpr int BK = 64;
constexpr int STAGES = 2;
constexpr float scale = 0.125f;
constexpr int MMA_K = 16;
constexpr int MMA_N = 8;
constexpr int TM = 2;
constexpr int TN = 2;
constexpr int VEC_LEN = 8;
constexpr int SCORE_N_ITER = BK / MMA_N;
constexpr int OUTPUT_N_ITER = DH / MMA_N;

using TiledMMA = decltype(make_tiled_mma(
  SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_4, _1, _1>>{}
));

using QLayout = decltype(composition(
  Swizzle<3,3,3>{},
  Layout<
  Shape<Int<BQ>, Int<DH>>,
  Stride<Int<DH>, Int<1>>
  >{})
);

using KVLayout = decltype(composition(
  Swizzle<3,3,3>{},
  Layout<
  Shape<Int<BK>, Int<DH>>,
  Stride<Int<DH>, Int<1>>
  >{})
);

using KVTransposedLayout = decltype(composition(
  Swizzle<3, 3, 3>{},
  Layout<
      Shape<Int<DH>, Int<BK>>,
      Stride<_1, Int<DH>>
  >{}
));

struct alignas(16) SharedMemoryStorage{
  Element Q[BQ * DH];
  Element dO[BQ * DH];
  Element K[STAGES][BK * DH];
  Element V[STAGES][BK * DH];
};

static_assert(DH == 64 && BQ == 64 && BK == 64);
static_assert(THREADS == 128 && STAGES == 2);
static_assert(sizeof(SharedMemoryStorage) == 48 * 1024);
static_assert(cosize_v<QLayout> == BQ * DH);
static_assert(cosize_v<KVLayout> == BK * DH);
static_assert(cosize_v<KVTransposedLayout> == BK * DH);
static_assert(VEC_LEN * sizeof(Element) == 16);

// Two adjacent C N=8 fragments form one A K=16 fragment in the same lane.
constexpr bool compatible_dS_fragment() {
  using Traits = MMA_Traits<SM80_16x8x16_F32BF16BF16F32_TN>;
  typename Traits::ALayout a{};
  typename Traits::CLayout c{};
  for (int lane = 0; lane < WARPSIZE; ++lane) {
    for (int value = 0; value < 8; ++value) {
      if (a(make_coord(lane, value)) !=
          c(make_coord(lane, value % 4)) + (value / 4) * 16 * MMA_N) {
        return false;
      }
    }
  }
  return true;
}
static_assert(compatible_dS_fragment());

// Load a complete ROWS x DH tile from one batch/head, 16 bytes per vector.
// Invalid token rows are zero-filled without forming out-of-bounds pointers.
template<int ROWS, class SmemLayout>
__device__ __forceinline__ void load_smem_tile(
    const Element* base, int64_t row0, int64_t rows,
    Element* dst, SmemLayout layout) {
  constexpr int vectors_per_row = DH / VEC_LEN;
  CUTE_UNROLL
  for (int vector = int(threadIdx.x); vector < ROWS * vectors_per_row;
       vector += THREADS) {
    const int r = vector / vectors_per_row;
    const int d = (vector % vectors_per_row) * VEC_LEN;
    const bool valid = row0 + r < rows;
    const Element* src = valid ? base + (row0 + r) * DH + d : base;
    Element* target = dst + layout(make_coord(r, d));
    SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
        *reinterpret_cast<const uint128_t*>(src),
        *reinterpret_cast<uint128_t*>(target), valid);
  }
}

template<class ATensor, class BTensor, class CTensor>
__device__ __forceinline__ void gemm_qk(const ATensor& sA, const BTensor& sB, CTensor& acc)
{
  TiledMMA mma;
  auto thr_mma = mma.get_slice(threadIdx.x);
  auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, mma);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, Element>{}, mma);
  auto thr_copy_a = copy_a.get_slice(threadIdx.x);
  auto thr_copy_b = copy_b.get_slice(threadIdx.x);
  CUTE_UNROLL
  for (int k_iter = 0; k_iter < DH / MMA_K ; k_iter++) {
    auto a = local_tile(sA, make_shape(Int<BQ>{}, Int<MMA_K>{}), make_coord(0, k_iter));
    auto b = local_tile(sB, make_shape(Int<BK>{}, Int<MMA_K>{}), make_coord(0, k_iter));
    auto rA = thr_mma.partition_fragment_A(a);
    auto rB = thr_mma.partition_fragment_B(b);
    auto tAs = thr_copy_a.partition_S(a);
    auto tBs = thr_copy_b.partition_S(b);
    auto tAr = thr_copy_a.retile_D(rA);
    auto tBr = thr_copy_b.retile_D(rB);
    copy(copy_a, tAs, tAr);
    copy(copy_b, tBs, tBr);
    gemm(mma, rA, rB, acc);
  }
}

template<class ATensor, class BTensor, class CTensor>
__device__ __forceinline__ void gemm_dSk(const ATensor& rA, const BTensor& sB, CTensor& acc)
{
  TiledMMA mma;
  auto thr_mma = mma.get_slice(threadIdx.x);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, Element>{}, mma);
  auto thr_copy_b = copy_b.get_slice(threadIdx.x);
  for_each(make_seq<BK / MMA_K>{}, [&](auto ki) {
    auto b = local_tile(sB, Shape<Int<DH>, Int<MMA_K>>{}, make_coord(_0{}, ki));
    auto rB = thr_mma.partition_fragment_B(b);
    auto tBs = thr_copy_b.partition_S(b);
    auto tBr = thr_copy_b.retile_D(rB);
    copy(copy_b, tBs, tBr);
    gemm(mma, rA(_, _, ki), rB(_, _, _0{}), acc);
  });
}

// Contract: contiguous BF16 Q/dO/dQ [B, Nh, Tq, 64], K/V [B, Nh, Tk, 64];
// contiguous FP32 D/LSE [B, Nh, Tq]. Q/K/V/dO bases must be 16-byte aligned.
// D = sum(dO * O, dim=-1); LSE is the natural-log normalization of scaled QK.
// Launch on SM80+ with block=(THREADS,1,1), grid=(ceil(Tq/BQ),Nh,B), no dynamic
// shared memory. The host must validate positive sizes, Tk=Tq+past_len,
// nonnegative past_len, device/grid limits, and no input/output aliasing.
// Supports causal attention without segments or dropout.
__global__ void kernel
(
  const int B,
  const int Nh,
  const int Tq,
  const int Tk,
  const int past_len,
  const Element* Q,
  const Element* K,
  const Element* V,
  const Element* dO,
  const float* D,
  const float* LSE,
  Element* dQ
)
{
  const int64_t batch = blockIdx.z;
  const int64_t head = blockIdx.y;
  const int64_t q0 = int64_t(blockIdx.x) * BQ;
  // Uniform across the CTA: no active thread skips a later barrier.
  if (batch >= B || head >= Nh || q0 >= Tq) return;

  // Apply batch/head offsets once. Row indices below are relative to this head.
  // Keep pointer arithmetic in int64_t even though sequence sizes fit int.
  const int64_t bh = batch * Nh + head;
  Q += bh * Tq * DH;
  dO += bh * Tq * DH;
  dQ += bh * Tq * DH;
  K += bh * Tk * DH;
  V += bh * Tk * DH;
  D += bh * Tq;
  LSE += bh * Tq;

  __shared__ SharedMemoryStorage smem;
  auto sQ = make_tensor(make_smem_ptr(smem.Q), QLayout{});
  auto sdO = make_tensor(make_smem_ptr(smem.dO), QLayout{});

  TiledMMA mma;
  auto thr_mma = mma.get_slice(threadIdx.x);
  auto score_coords = make_identity_tensor(Shape<Int<BQ>, Int<BK>>{});
  auto output_coords = make_identity_tensor(Shape<Int<BQ>, Int<DH>>{});
  auto tS = thr_mma.partition_C(score_coords);
  auto rS = thr_mma.make_fragment_C(tS);
  auto tdP = thr_mma.partition_C(score_coords);
  auto rdP = thr_mma.make_fragment_C(tdP);
  auto tdQ_acc = thr_mma.partition_C(output_coords);
  auto rdQ_acc = thr_mma.make_fragment_C(tdQ_acc);

  CUTE_STATIC_ASSERT_V(size<0>(tS) == Int<TM * TN>{});
  CUTE_STATIC_ASSERT_V(size<1>(tS) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<2>(tS) == Int<SCORE_N_ITER>{});
  CUTE_STATIC_ASSERT_V(size<2>(tdQ_acc) == Int<OUTPUT_N_ITER>{});
  clear(rdQ_acc);
  load_smem_tile<BQ>(Q, q0, Tq, smem.Q, QLayout{});
  load_smem_tile<BQ>(dO, q0, Tq, smem.dO, QLayout{});
  int stage = 0;
  load_smem_tile<BK>(K, 0, Tk, smem.K[stage], KVLayout{});
  load_smem_tile<BK>(V, 0, Tk, smem.V[stage], KVLayout{});
  // Publish Q/dO and the first K/V tile before any ldmatrix load.
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  for (int64_t k_tile = 0; k_tile < Tk; k_tile += BK) {
    clear(rS);
    clear(rdP);
    if (k_tile + BK < Tk) {
      load_smem_tile<BK>(K, k_tile + BK, Tk, smem.K[stage ^ 1], KVLayout{});
      load_smem_tile<BK>(V, k_tile + BK, Tk, smem.V[stage ^ 1], KVLayout{});
      cp_async_fence();
    }
    //S = Q @ K^T
    auto sK = make_tensor(make_smem_ptr(smem.K[stage]), KVLayout{});
    gemm_qk(sQ, sK, rS);
    // mask softmax, P = exp(S - LSE)
    CUTE_UNROLL
    for (int a = 0; a < TM; a++) {
      CUTE_UNROLL
      for (int n = 0; n < SCORE_N_ITER; n++) {
        CUTE_UNROLL
        for (int bb = 0; bb < TN; ++bb) {
          const int vi = a * TN + bb;
          const int64_t qi = q0 + get<0>(tS(vi, 0, n));
          const int64_t kj = k_tile + get<1>(tS(vi, 0, n));
          const bool valid = qi < Tq && kj < Tk && kj - past_len <= qi;
          rS(vi, 0, n) = valid ? expf(scale * rS(vi, 0, n) - LSE[qi]) : 0.0f;
        }
      }
    }
    //dP = dO @ V^T
    auto sV = make_tensor(make_smem_ptr(smem.V[stage]), KVLayout{});
    gemm_qk(sdO, sV, rdP);
    //dS = P * (dP - D)
    CUTE_UNROLL
    for (int a = 0; a < TM; a++) {
      CUTE_UNROLL
      for (int n = 0; n < SCORE_N_ITER; n++) {
        CUTE_UNROLL
        for (int bb = 0; bb < TN; ++bb) {
          const int vi = a * TN + bb;
          const int64_t qi = q0 + get<0>(tdP(vi, 0, n));
          const int64_t kj = k_tile + get<1>(tdP(vi, 0, n));
          const bool valid = qi < Tq && kj < Tk && kj - past_len <= qi;
          rdP(vi, 0, n) = valid ? rS(vi, 0, n) * (rdP(vi, 0, n) - D[qi]) : 0.0f;
        }
      }
    }
    auto rDS_A = make_tensor<Element>(shape(thr_mma.partition_A(score_coords)));
    CUTE_STATIC_ASSERT_V(size<0>(rDS_A) == Int<8>{});
    CUTE_STATIC_ASSERT_V(size<1>(rDS_A) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size<2>(rDS_A) == Int<BK / MMA_K>{});
    CUTE_UNROLL
    for (int n = 0; n < BK / MMA_N; ++n) {
        CUTE_UNROLL
        for (int vi = 0; vi < 4; ++vi) {
            rDS_A(vi + 4 * (n % 2), 0, n / 2) =
                Element(rdP(vi, 0, n));
        }
    }
    auto sKT = make_tensor(make_smem_ptr(smem.K[stage]), KVTransposedLayout{});
    gemm_dSk(rDS_A, sKT, rdQ_acc);
    if (k_tile + BK < Tk) {
      cp_async_wait<0>();
      // Publish next K/V and release current readers before reusing their slot.
      __syncthreads();
    }
    stage ^= 1;
  }

  CUTE_UNROLL
  for (int a = 0; a < TM; a++) {
    CUTE_UNROLL
    for (int n = 0; n < OUTPUT_N_ITER; n++) {
      CUTE_UNROLL
      for (int bb = 0; bb < TN; ++bb) {
        const int vi = a * TN + bb;
        const int64_t qi = q0 + get<0>(tdQ_acc(vi, 0, n));
        const int64_t hi = get<1>(tdQ_acc(vi, 0, n));
        if (qi < Tq && hi < DH) {
          dQ[qi * DH + hi] = Element(scale * rdQ_acc(vi, 0, n));
        }
      }
    }
  }
} // dQ kernel

} // namespace attention_backward_dQ
