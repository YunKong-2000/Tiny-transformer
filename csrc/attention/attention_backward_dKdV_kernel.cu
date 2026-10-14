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

namespace attention_backward_dKdV{
using namespace cute;
using Element = cute::bfloat16_t;
const int BQ = 64;
const int BK = 64;
const int DH = 64;
const int THREADS = 128;
const int VEC_LEN = 8;
const int MMA_K = 16;
const int MMA_N = 8;
const int VEC_PER_ROW = DH / VEC_LEN;
const int STAGES = 2;
constexpr float SCALE = 0.125f;
const int SCORE_N_ITER = BQ / MMA_N;
const int OUTPUT_N_ITER = DH / MMA_N;
const int TM = 2;
const int TN = 2;

using Tiled_MMA = decltype(make_tiled_mma(
  SM80_16x8x16_F32BF16BF16F32_TN{},
  Layout<Shape<_4, _1, _1>>{}
));

using QLayout = decltype(composition(
  Swizzle<3, 3, 3>{},
  Layout<Shape<Int<BQ>, Int<DH>>,
         Stride<Int<DH>, Int<1>>
  >{})
);

using KVLayout = decltype(composition(
  Swizzle<3, 3, 3>{},
  Layout<Shape<Int<BK>, Int<DH>>,
         Stride<Int<DH>, Int<1>>
  >{})
);

using QTransposedLayout = decltype(composition(
  Swizzle<3, 3, 3>{},
  Layout<Shape<Int<DH>, Int<BQ>>,
         Stride<Int<1>, Int<DH>>
  >{})
);

struct alignas(16) SmemStorage{
  Element K[BK * DH];
  Element V[BK * DH];
  Element Q[STAGES][BQ * DH];
  Element dO[STAGES][BQ * DH];
};

static_assert(BQ == 64 && BK == 64 && DH == 64 && THREADS == 128);
static_assert(STAGES == 2 && sizeof(SmemStorage) == 48 * 1024);
static_assert(VEC_LEN * sizeof(Element) == 16);
static_assert(cosize_v<QLayout> == BQ * DH);
static_assert(cosize_v<KVLayout> == BK * DH);
static_assert(cosize_v<QTransposedLayout> == BQ * DH);

// Adjacent N=8 C fragments form a K=16 A fragment in the same lane.
constexpr bool compatible_score_fragment() {
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
static_assert(compatible_score_fragment());

template<int ROWS, class SmemLayout>
__device__ __forceinline__ void load_smem(
const Element* g_ptr, 
Element* s_ptr, 
const int64_t row_base,
const int64_t row_limit,
const SmemLayout& layout) {
  for (int vi = threadIdx.x; vi < ROWS * VEC_PER_ROW; vi += THREADS) {
    int row = vi / VEC_PER_ROW;
    int col = (vi % VEC_PER_ROW) * VEC_LEN;
    const bool valid = row_base + row < row_limit;
    const Element* src = valid ? g_ptr + (row_base + row) * DH + col : g_ptr;
    Element* dst = s_ptr + layout(make_coord(row, col));
    SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
      *reinterpret_cast<const uint128_t*>(src),
      *reinterpret_cast<uint128_t*>(dst),
      valid
    );
  }
}

template<class ATensor, class BTensor, class CTensor>
__device__ __forceinline__ void gemm_KQ(const ATensor& sK, const BTensor& sQ, CTensor& rAcc) {
  Tiled_MMA mma;
  auto thr_mma = mma.get_slice(threadIdx.x);
  auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, mma);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, Element>{}, mma); 
  auto thr_copy_a = copy_a.get_slice(threadIdx.x);
  auto thr_copy_b = copy_b.get_slice(threadIdx.x);
  CUTE_UNROLL
  for (int k_iter = 0; k_iter < DH / MMA_K; k_iter++) {
    auto a = local_tile(sK, make_shape(Int<BK>{}, Int<MMA_K>{}), make_coord(0, k_iter));
    auto b = local_tile(sQ, make_shape(Int<BQ>{}, Int<MMA_K>{}), make_coord(0, k_iter));
    auto rA = thr_mma.partition_fragment_A(a);
    auto rB = thr_mma.partition_fragment_B(b);
    auto tAs = thr_copy_a.partition_S(a);
    auto tBs = thr_copy_b.partition_S(b);
    auto tAr = thr_copy_a.retile_D(rA);
    auto tBr = thr_copy_b.retile_D(rB);
    copy(copy_a, tAs, tAr);
    copy(copy_b, tBs, tBr);
    gemm(mma, rA, rB, rAcc);
  }
}

template<class ATensor, class BTensor, class CTensor>
__device__ __forceinline__ void gemm_PdO(const ATensor& rP, const BTensor& sdO, CTensor& rdVAcc) {
  Tiled_MMA mma;
  auto thr_mma = mma.get_slice(threadIdx.x);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, Element>{}, mma);
  auto thr_copy_b = copy_b.get_slice(threadIdx.x);
  for_each(make_seq<BQ / MMA_K>{}, [&](auto ki) {
    auto b = local_tile(sdO, make_shape(Int<DH>{}, Int<MMA_K>{}), make_coord(0, ki));
    auto rB = thr_mma.partition_fragment_B(b);
    auto tBs = thr_copy_b.partition_S(b);
    auto tBr = thr_copy_b.retile_D(rB);
    copy(copy_b, tBs, tBr);
    gemm(mma, rP(_,_,ki), rB(_,_,_0{}), rdVAcc);
  });
}

// Contract: contiguous BF16 Q/dO [B,Nh,Tq,64], K/V/dK/dV [B,Nh,Tk,64];
// contiguous FP32 D/LSE [B,Nh,Tq]. BF16 bases must be 16-byte aligned.
// D=sum(dO*O,dim=-1); LSE is the natural-log normalization of scaled QK.
// Launch on SM80+ with block=(128,1,1), grid=(ceil(Tk/BK),Nh,B).
// Host validates positive sizes, Tk=Tq+past_len, nonnegative past_len,
// int32 sequence sizes, device/grid limits and non-aliasing outputs.
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
  Element* dK,
  Element* dV
)
{
  const int64_t b = blockIdx.z;
  const int64_t h = blockIdx.y;
  const int64_t k0 = int64_t(blockIdx.x) * BK;
  // Uniform CTA exit before any asynchronous copy or barrier.
  if (b >= B || h >= Nh || k0 >= Tk) return;
  const int64_t bh = b * Nh + h;
  Q  += bh * Tq * DH;
  K  += bh * Tk * DH;
  V  += bh * Tk * DH;
  dO += bh * Tq * DH;
  D   += bh * Tq;
  LSE += bh * Tq;
  dK  += bh * Tk * DH;
  dV  += bh * Tk * DH;

  __shared__ SmemStorage smem;
  

  Tiled_MMA mma;
  auto thr_mma = mma.get_slice(threadIdx.x);
  auto scores = make_identity_tensor(Shape<Int<BK>, Int<BQ>>{});
  auto outputs = make_identity_tensor(Shape<Int<BK>, Int<DH>>{});
  
  auto tdV = thr_mma.partition_C(outputs);
  auto rdV = thr_mma.make_fragment_C(tdV);
  auto tdK = thr_mma.partition_C(outputs);
  auto rdK = thr_mma.make_fragment_C(tdK);

  clear(rdV);
  clear(rdK);
  load_smem<BK>(K, smem.K, k0, Tk, KVLayout{});
  load_smem<BK>(V, smem.V, k0, Tk, KVLayout{});
  int stage = 0;
  load_smem<BQ>(Q, smem.Q[0], 0, Tq, QLayout{});
  load_smem<BQ>(dO, smem.dO[0], 0, Tq, QLayout{});
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();
  for (int64_t q_iter = 0; q_iter < Tq; q_iter += BQ) {
    
    if (q_iter + BQ < Tq) {
      const int64_t next_q = q_iter + BQ;
      load_smem<BQ>(Q,   smem.Q[stage ^ 1], next_q, Tq, QLayout{});
      load_smem<BQ>(dO, smem.dO[stage ^ 1], next_q, Tq, QLayout{});
      cp_async_fence();
    }
    // R = K @ Q.T
    auto tR = thr_mma.partition_C(scores);
    auto rR = thr_mma.make_fragment_C(tR);
    clear(rR);
    auto sK = make_tensor(make_smem_ptr(smem.K), KVLayout{});
    auto sQ = make_tensor(make_smem_ptr(smem.Q[stage]), QLayout{});
    gemm_KQ(sK, sQ, rR);
    // G = V @ dO.T
    auto tG = thr_mma.partition_C(scores);
    auto rG = thr_mma.make_fragment_C(tG);
    clear(rG);
    auto sV = make_tensor(make_smem_ptr(smem.V), KVLayout{});
    auto sdO = make_tensor(make_smem_ptr(smem.dO[stage]), QLayout{});
    gemm_KQ(sV, sdO, rG);
    // P.T = exp(scale * R - LSE), P.T -> rR
    CUTE_UNROLL
    for (int a = 0; a < TM; a++) {
      CUTE_UNROLL
      for (int n = 0; n < SCORE_N_ITER; n++) {
        CUTE_UNROLL
        for (int b = 0; b < TN; b++) {
          const int vi = a * TN + b;
          const int64_t row = k0 + get<0>(tR(vi, 0, n));
          const int64_t col = q_iter + get<1>(tR(vi, 0, n));
          bool valid = (row < Tk) && (col < Tq) && (col + past_len >= row);
          rR(vi, 0, n) = valid ? expf(SCALE * rR(vi, 0, n) - LSE[col]) : 0.0f;
        }
      }
    }
    //dS.T = P.T * (G - D), keep P.T in rR, dS.T -> rG
    CUTE_UNROLL
    for (int a = 0; a < TM; a++) {
      CUTE_UNROLL
      for (int n = 0; n < SCORE_N_ITER; n++) {
        CUTE_UNROLL
        for (int b = 0; b < TN; b++) {
          const int vi = a * TN + b;
          const int64_t row = k0 + get<0>(tG(vi, 0, n));
          const int64_t col = q_iter + get<1>(tG(vi, 0, n));
          bool valid = (row < Tk) && (col < Tq) && (col + past_len >= row);
          rG(vi, 0, n) = valid ? rR(vi, 0, n) * (rG(vi, 0, n) - D[col]) : 0.0f;
        }
      }
    }
    // Convert P.T from C layout to A layout; dV_acc += P.T @ dO.
    auto rP = make_tensor<Element>(shape(thr_mma.partition_A(scores)));
    CUTE_STATIC_ASSERT_V(size<0>(rP) == Int<8>{});
    CUTE_STATIC_ASSERT_V(size<1>(rP) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size<2>(rP) == Int<BQ / MMA_K>{});
    CUTE_UNROLL
    for (int n = 0; n < SCORE_N_ITER; n++) {
      CUTE_UNROLL
      for (int vi = 0; vi < TM * TN; vi++) {
        const int m = n / 2;
        const int k = n % 2;
        rP(vi + k * 4, 0, m) = Element(rR(vi, 0, n));
      }
    }
    auto sdOT = make_tensor(make_smem_ptr(smem.dO[stage]), QTransposedLayout{});
    gemm_PdO(rP, sdOT, rdV);
    //dK_acc += dS.T @ Q_i
    auto rS = make_tensor<Element>(shape(thr_mma.partition_A(scores)));
    CUTE_UNROLL
    for (int n = 0; n < SCORE_N_ITER; n++) {
      CUTE_UNROLL
      for (int vi = 0; vi < TM * TN; vi++) {
        const int m = n / 2;
        const int k = n % 2;
        rS(vi + k * 4, 0, m) = Element(rG(vi, 0, n));
      }
    }
    auto sQT = make_tensor(make_smem_ptr(smem.Q[stage]), QTransposedLayout{});
    gemm_PdO(rS, sQT, rdK);
    if (q_iter + BQ < Tq) {
      cp_async_wait<0>();
      // Publish the next tile and release readers before reusing the old slot.
      __syncthreads();
    }
    stage ^= 1;
  }
  CUTE_UNROLL
  for (int a = 0; a < TM; a++) {
    CUTE_UNROLL
    for (int n = 0; n < OUTPUT_N_ITER; n++) {
      CUTE_UNROLL
      for (int b = 0; b < TN; b++) {
        const int vi = a * TN + b;
        const int64_t row = k0 + get<0>(tdV(vi, 0, n));
        const int col = get<1>(tdV(vi, 0, n));
        if (row < Tk && col < DH) {
          dK[row * DH + col] = Element(SCALE * rdK(vi, 0, n));
          dV[row * DH + col] = Element(rdV(vi, 0, n));
        }
      }
    }
  }
}



} //namespace dKdV
