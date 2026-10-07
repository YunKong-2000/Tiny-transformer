#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cmath>
#include <cstdint>

// Educational FP32 SIMT forward template; no pipeline or Tensor Core MMA.
// Not compiled/validated on CUDA in the authoring environment.
// Matches csrc/attention/development.md: full Q tile persists in shared memory;
// K is tiled along the reduction/head dimension, V along output features.
// Launch exactly 128 threads. BQ=32, BK=32, BD=16, BN=32.
// SIMT layout: thread tile 2x4, lanes 4x8, warps 4x1, warp tile 8x32.
// DH is a positive multiple of 32. QK reduces DH in BD-wide subtiles;
// PV visits BN output columns at a time, with RK=8 register reduction chunks.
// E stays in registers and is shuffled
// across each group of eight same-row lanes to construct PV A fragments.
// Q/K: contiguous [B,H,Tq/Tk,DH]. V: arbitrary nonnegative element strides.
// O: contiguous [B,H,Tq,DH]. LSE: contiguous FP32 [B,H,Tq], natural log.
// Contract: Tk == past_len + Tq, positive dimensions, finite Q/K/V.
// Optional int64 segments: [B,Tq], only when past_len=0 and Tq=Tk.
// Use modest DH (e.g. 64); shared-memory/register limits must be checked
// before adding other specializations. This header supplies no host checks.
namespace attention_simt_example {

struct Stride4 {
  int64_t b, h, t, d;
};

// lane = lane_m + 4*lane_n: XOR 4/8/16 changes only lane_n.
// All 32 lanes execute; four independent groups each reduce one row.
__device__ __forceinline__ float row_group_max(float x) {
  #pragma unroll
  for (int delta = 4; delta <= 16; delta <<= 1)
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, delta));
  return x;
}

__device__ __forceinline__ float row_group_sum(float x) {
  #pragma unroll
  for (int delta = 4; delta <= 16; delta <<= 1)
    x += __shfl_xor_sync(0xffffffffu, x, delta);
  return x;
}

template <int DH>
__global__ void forward(
    const float* q, const float* k, const float* v,
    float* o, float* lse,
    int64_t H, int64_t Tq, int64_t Tk, Stride4 vs,
    int64_t past_len, const int64_t* segments,
    int64_t seg_stride_b, int64_t seg_stride_t, float scale) {
  using namespace cute;
  constexpr int BQ = 32, BK = 32, BD = 16, BN = 32, RK = 8, THREADS = 128;
  constexpr int TM = 2, TN = 4, ON = DH / 8, VN = BN / 8;
  static_assert(DH > 0 && DH % 32 == 0);

  // Flat grid: [B,H,ceil(Tq/BQ)] -> grid.x. Host validates grid limits.
  const int64_t q_tiles = (Tq + BQ - 1) / BQ;
  const int64_t bh = int64_t(blockIdx.x) / q_tiles;
  const int64_t q0 = (int64_t(blockIdx.x) % q_tiles) * BQ;
  const int64_t b = bh / H, h = bh % H;
  const int tid = threadIdx.x, lane = tid % 32;
  const int lane_m = lane % 4, lane_n = lane / 4;

  // Q is loaded once and stays live for the entire streaming loop.
  // Only K and V share storage: they are consumed sequentially (no pipeline).
  // DH=64: [32*65 + max(32*17, 32*32)]*4 = 12416 bytes.
  struct Storage {
    float q[BQ * (DH + 1)];
    union {
      float k[BK * (BD + 1)];
      float v[BK * BN];
    } kv;
  };
  __shared__ Storage storage;

  auto sQ = make_tensor(make_smem_ptr(storage.q),
      make_layout(make_shape(Int<BQ>{}, Int<DH>{}),
                  make_stride(Int<DH + 1>{}, Int<1>{})));
  // Padding avoids bank conflicts when lanes read different K rows.
  auto sK = make_tensor(make_smem_ptr(storage.kv.k),
      make_layout(make_shape(Int<BK>{}, Int<BD>{}),
                  make_stride(Int<BD + 1>{}, Int<1>{})));
  auto sV = make_tensor(make_smem_ptr(storage.kv.v),
      make_layout(make_shape(Int<BK>{}, Int<BN>{}),
                  make_stride(Int<BN>{}, Int<1>{})));

  // Thread atom coordinates: M=(lane_m,warp_m), N=lane_n, K=0.
  // tid = lane_m + 32*warp_m + 4*lane_n.
  // Permutations make each thread's 2x4 C fragment contiguous in M/N:
  // row = 8*warp_m + 2*lane_m + a; col = 4*lane_n + c.
  // In PV, additional output-column groups repeat this N tile every 32.
  auto mma = make_tiled_mma(UniversalFMA<float, float, float>{},
      Layout<Shape<Shape<Int<4>, Int<4>>, Int<8>, Int<1>>,
             Stride<Stride<Int<1>, Int<32>>, Int<4>, Int<128>>>{},
      make_tile(
          Layout<Shape<Int<4>, Int<4>, Int<2>>,
                 Stride<Int<2>, Int<8>, Int<1>>>{},
          Layout<Shape<Int<8>, Int<4>>,
                 Stride<Int<4>, Int<1>>>{},
          _));
  auto thr = mma.get_slice(tid);

  auto tK = thr.partition_B(sK);

  // Coordinate tensors let us map fragments back to logical coordinates.
  auto cS = make_identity_tensor(make_shape(Int<BQ>{}, Int<BK>{}));
  auto cO = make_identity_tensor(make_shape(Int<BQ>{}, Int<DH>{}));
  auto tSc = thr.partition_C(cS);
  auto tOc = thr.partition_C(cO);
  auto rS = thr.make_fragment_C(tSc);
  auto rO = thr.make_fragment_C(tOc);
  CUTE_STATIC_ASSERT_V(size<0>(rS) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<1>(rS) == Int<TM>{});
  CUTE_STATIC_ASSERT_V(size<2>(rS) == Int<TN>{});
  CUTE_STATIC_ASSERT_V(size<0>(rO) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<1>(rO) == Int<TM>{});
  CUTE_STATIC_ASSERT_V(size<2>(rO) == Int<ON>{});
  clear(rO);

  // The ONLY global-memory Q load. Tail query rows are zero-filled.
  for (int x = tid; x < BQ * DH; x += THREADS) {
    const int r = x / DH, d = x % DH;
    sQ(r, d) = q0 + r < Tq ? q[(bh * Tq + q0 + r) * DH + d] : 0.f;
  }
  __syncthreads();

  float m[TM] = {-CUDART_INF_F, -CUDART_INF_F};
  float l[TM] = {0.f, 0.f};
  // No early return for padded rows: barriers and full-warp shuffles follow.
  for (int64_t k0 = 0; k0 < Tk; k0 += BK) {
    // 1. Stream K subtiles; Q subtiles are views of the persistent sQ.
    clear(rS);
    for (int d0 = 0; d0 < DH; d0 += BD) {
      for (int x = tid; x < BK * BD; x += THREADS) {
        const int n = x / BD, d = x % BD;
        sK(n, d) = k0 + n < Tk
            ? k[(bh * Tk + k0 + n) * DH + d0 + d] : 0.f;
      }
      __syncthreads();
      auto sQd = local_tile(sQ, make_shape(Int<BQ>{}, Int<BD>{}),
                           make_coord(0, d0 / BD));
      auto tQ = thr.partition_A(sQd);
      gemm(mma, tQ, tK, rS);
      __syncthreads(); // K reads finish before K/V reuse; Q is never overwritten.
    }

    // 2. Two rows/thread; reduce four local columns, then eight same-row
    // lanes. No shared-memory or cross-warp reduction is needed.
    CUTE_UNROLL
    for (int a = 0; a < TM; ++a) {
      const int64_t qi = q0 + get<0>(tSc(0, a, 0));
      float local_max = -CUDART_INF_F;
      CUTE_UNROLL
      for (int c = 0; c < TN; ++c) {
        const int64_t kj = k0 + get<1>(tSc(0, a, c));
        bool allowed = qi < Tq && kj < Tk && kj <= past_len + qi;
        if (allowed && segments != nullptr) {
          allowed = segments[b * seg_stride_b + qi * seg_stride_t] ==
                    segments[b * seg_stride_b + kj * seg_stride_t];
        }
        rS(0, a, c) = allowed ? scale * rS(0, a, c) : -CUDART_INF_F;
        local_max = fmaxf(local_max, rS(0, a, c));
      }
      const float m_new = fmaxf(m[a], row_group_max(local_max));
      const float alpha = m[a] == -CUDART_INF_F ? 0.f : expf(m[a] - m_new);
      float local_sum = 0.f;
      CUTE_UNROLL
      for (int c = 0; c < TN; ++c) {
        const float s = rS(0, a, c);
        // Handle masked rows/blocks without exp(-inf - -inf).
        rS(0, a, c) = s == -CUDART_INF_F ? 0.f : expf(s - m_new);
        local_sum += rS(0, a, c);
      }
      l[a] = alpha * l[a] + row_group_sum(local_sum);
      m[a] = m_new;
      CUTE_UNROLL
      for (int x = 0; x < ON; ++x)
        rO(0, a, x) *= alpha;
    }

    // 3. Visit output-feature tiles V_jn[BK,BN]. The loop over d0 is NOT a
    // reduction: each iteration updates different output columns. The actual
    // PV reduction is over all BK keys, processed in RK-sized register chunks.
    // rO was scaled once above, so do not apply alpha again in this loop.
    for (int d0 = 0; d0 < DH; d0 += BN) {
      for (int x = tid; x < BK * BN; x += THREADS) {
        const int n = x / BN, d = x % BN;
        const int64_t key = k0 + n;
        sV(n, d) = key < Tk
            ? v[b * vs.b + h * vs.h + key * vs.t + (d0 + d) * vs.d] : 0.f;
      }
      __syncthreads();

      auto rOn = make_tensor<float>(
          make_shape(Int<1>{}, Int<TM>{}, Int<VN>{}));
      const int out_fragment_base = (d0 / BN) * VN;
      CUTE_UNROLL
      for (int a = 0; a < TM; ++a) {
        CUTE_UNROLL
        for (int x = 0; x < VN; ++x)
          rOn(0, a, x) = rO(0, a, out_fragment_base + x);
      }

      // Register GEMM expects A(V,M,K), B(V,N,K), C(V,M,N).
      // Per thread: M=2, N=BN/8, K=RK, V=1 for UniversalFMA.
      for (int n0 = 0; n0 < BK; n0 += RK) {
        auto rE = make_tensor<float>(
            make_shape(Int<1>{}, Int<TM>{}, Int<RK>{}));
        auto rV = make_tensor<float>(
            make_shape(Int<1>{}, Int<VN>{}, Int<RK>{}));
        CUTE_UNROLL
        for (int n = 0; n < RK; ++n) {
          // E column c_global belongs to lane_n=c_global/4, fragment c%4.
          // Keep lane_m fixed: never mix the other query-row groups.
          const int c_global = n0 + n;
          const int src_lane = lane_m + 4 * (c_global / TN);
          CUTE_UNROLL
          for (int a = 0; a < TM; ++a)
            rE(0, a, n) = __shfl_sync(
                0xffffffffu, rS(0, a, c_global % TN), src_lane);
          CUTE_UNROLL
          for (int x = 0; x < VN; ++x) {
            const int d = get<1>(tOc(0, 0, x));
            rV(0, x, n) = sV(n0 + n, d);
          }
        }
        gemm(mma, rE, rV, rOn);
      }
      CUTE_UNROLL
      for (int a = 0; a < TM; ++a) {
        CUTE_UNROLL
        for (int x = 0; x < VN; ++x)
          rO(0, a, out_fragment_base + x) = rOn(0, a, x);
      }
      __syncthreads(); // V reads finish before the next V/K overwrite.
    }
  }

  CUTE_UNROLL
  for (int a = 0; a < TM; ++a) {
    const int64_t qi = q0 + get<0>(tOc(0, a, 0));
    if (qi < Tq) {
      CUTE_UNROLL
      for (int x = 0; x < ON; ++x) {
        const int d = get<1>(tOc(0, a, x));
        o[(bh * Tq + qi) * DH + d] = l[a] > 0.f ? rO(0, a, x) / l[a] : 0.f;
      }
      // One writer per row, not one writer per entire warp.
      if (lane_n == 0)
        lse[bh * Tq + qi] = l[a] > 0.f ? m[a] + logf(l[a]) : -CUDART_INF_F;
    }
  }
}

} // namespace attention_simt_example
