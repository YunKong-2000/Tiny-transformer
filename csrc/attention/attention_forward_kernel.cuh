#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cmath>
#include <cstdint>

namespace attention_detail {

using namespace cute;
constexpr int THREADS = 128;
constexpr int WM = 4;
constexpr int WN = 8;
constexpr int BQ = 32;
constexpr int BK = 32;
constexpr int RK = 8;
constexpr int BD = 16;
constexpr int BN = 16;
constexpr int TM = 2;
constexpr int TN = 4;
constexpr int VN = 2;


struct Stride4D{
  int64_t b, h, t, d;
};

__device__ __forceinline__ float warp_reduce_sum(float x) {
  #pragma unroll
  for (int offset = 4; offset <= 16; offset <<= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset);
  }
  return x;
}

__device__ __forceinline__ float warp_reduce_max(float x) {
  #pragma unroll
  for (int offset = 4; offset <= 16; offset <<= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset));
  }
  return x;
}

template<int DH>
__global__ void forward
(
  const int64_t B,
  const int64_t H,
  const int64_t T_q,
  const int64_t T_k,
  Stride4D V_stride,
  const float* Q,
  const float* K,
  const float* V,
  const int64_t* segment_ids,
  const int64_t past_len,
  float* O,
  float* LSE
)
{
  static_assert(DH == 64, "Only the DH=64 layout is supported");
  const int64_t batch = blockIdx.y / H;
  const int64_t head = blockIdx.y % H;
  const int64_t Q_offset = blockIdx.y * T_q * DH;
  const int64_t KV_offset = blockIdx.y * T_k * DH;
  const int64_t q0 = int64_t(blockIdx.x) * BQ;
  const int64_t Q_CTA_offset = q0 * DH;
  const float scale = 1.0f / sqrtf(float(DH));
  Q = Q + Q_offset;
  K = K + KV_offset;
  const int tid = threadIdx.x;
  const int lane_tid = tid & 31;
  const int lane_m = lane_tid % WM;
  const int lane_n = lane_tid / WM;
  struct Storage {
    float q[BQ * DH];
    union {
      float k[BK * BD];
      float v[BK * BN];
    } kv;
  };
  __shared__ Storage smem_storage;
  auto q_layout = composition(
    Swizzle<2,1,6>{},
    Layout<Shape<Int<BQ>, Int<DH>>,
           Stride<Int<DH>, Int<1>>
    >{}
  );
  auto k_layout = composition(
    Swizzle<3,2,4>{},
    Layout<Shape<Int<BK>, Int<BD>>,
           Stride<Int<BD>, Int<1>>
    >{}
  );
  auto v_layout = make_layout(
    make_shape(Int<BK>{}, Int<BN>{}),
    make_stride(Int<BN>{}, Int<1>{})
  );
  auto sQ = make_tensor(make_smem_ptr(smem_storage.q), q_layout);
  auto sK = make_tensor(make_smem_ptr(smem_storage.kv.k), k_layout);
  auto sV = make_tensor(make_smem_ptr(smem_storage.kv.v), v_layout);
  // ((lane_m, warp_m), lane_n, reduction_thread): no split-K threads.
  using AtomThreads = Layout<
    Shape<Shape<_4, _4>, _8, _1>,
    Stride<Stride<_1, _32>, _4, _128>
  >;
  // row = lane_m * 2 + warp_m * 8 + a
  using PermM = Layout<
    Shape<_4, _4, _2>,
    Stride<_2, _8, _1>
  >;
  // col = lane_n * 4 + 0 * 32 + b
  using PermNQK = Layout<
    Shape<_8, _4>,
    Stride<_4, _1>
  >;
  // col = lane_n * 2 + 0 * 16 + b
  using PermNPV = Layout<
    Shape<_8, _2>,
    Stride<_2, _1>
  >;

  auto mma_QK = make_tiled_mma(
    UniversalFMA<float, float, float>{},
    AtomThreads{},
    make_tile(PermM{}, PermNQK{}, _)
  );
  auto mma_PV = make_tiled_mma(
    UniversalFMA<float, float, float>{},
    AtomThreads{},
    make_tile(PermM{}, PermNPV{}, _)
  );
  auto thrQK = mma_QK.get_slice(tid);
  auto thrPV = mma_PV.get_slice(tid);
  auto tK = thrQK.partition_B(sK);
  auto cS = make_identity_tensor(make_shape(Int<BQ>{}, Int<BK>{}));
  auto cO = make_identity_tensor(make_shape(Int<BQ>{}, Int<DH>{}));
  auto tSc = thrQK.partition_C(cS);
  auto tOc = thrPV.partition_C(cO);
  auto rS = thrQK.make_fragment_C(tSc);
  auto rO = thrPV.make_fragment_C(tOc);
  CUTE_STATIC_ASSERT_V(size<0>(rS) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<1>(rS) == Int<TM>{});
  CUTE_STATIC_ASSERT_V(size<2>(rS) == Int<TN>{});
  CUTE_STATIC_ASSERT_V(size<0>(rO) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<1>(rO) == Int<TM>{});
  CUTE_STATIC_ASSERT_V(size<2>(rO) == Int<DH / WN>{});
  clear(rO);
  // load Q_i from Gmem to Smem
  Q += Q_CTA_offset;
  for (int x = tid; x < BQ * DH; x += THREADS) {
    int row = x / DH;
    int col = x % DH;
    sQ(row, col) = q0 + row < T_q ? Q[x] : 0.f;
  }
  __syncthreads();
  float m[TM] = {-CUDART_INF_F, -CUDART_INF_F};
  float l[TM] = {0.f, 0.f};
  // Padded query rows still participate in every barrier and full-warp shuffle.
  for (int64_t k = 0; k < T_k; k += BK) {
    clear(rS);
    // Q @ K^T
    const float* K_tile = K + k * DH;
    for (int d = 0; d < DH; d += BD) {
      //load subtile K_kd
      for (int x = tid; x < BK * BD; x += THREADS) {
        int row = x / BD;
        int col = x % BD;
        sK(row, col) = k + row < T_k ? K_tile[row * DH + col + d] : 0.f;
      }
      __syncthreads();
      auto sQd = local_tile(sQ,
        make_shape(Int<BQ>{}, Int<BD>{}),
        make_coord(0, d / BD)
      );
      auto tQ = thrQK.partition_A(sQd);
      gemm(mma_QK, tQ, tK, rS);
      __syncthreads(); // Finish K reads before reusing the K/V storage.
    } // loop DH

    // mask and softmax
    CUTE_UNROLL
    for (int a = 0; a < TM; a++) {
      const int64_t qi = q0 + get<0>(tSc(0, a, 0));
      float local_max = -CUDART_INF_F;
      CUTE_UNROLL
      for (int b = 0; b < TN; b++) {
        const int64_t kj = k + get<1>(tSc(0, a, b));
        bool valid = qi < T_q && kj < T_k && kj <= past_len + qi;
        if (valid && segment_ids != nullptr) {
          valid = segment_ids[batch * T_q + qi] ==
                  segment_ids[batch * T_q + kj];
        }
        rS(0, a, b) = valid ? scale * rS(0, a, b) : -CUDART_INF_F;
        local_max = fmaxf(local_max, rS(0, a, b));
      }
      const float m_new = fmaxf(m[a], warp_reduce_max(local_max));
      const float alpha = m[a] == -CUDART_INF_F ? 0.f : expf(m[a] - m_new);
      float local_sum = 0.f;
      CUTE_UNROLL
      for (int b = 0; b < TN; b++) {
        const float s_value = rS(0, a, b);
        rS(0, a, b) = s_value == -CUDART_INF_F ? 0.0f : expf(s_value - m_new);
        local_sum += rS(0, a, b);
      }
      l[a] = alpha * l[a] + warp_reduce_sum(local_sum);
      m[a] = m_new;
      CUTE_UNROLL
      for (int x = 0; x < DH / WN; x++) {
        rO(0, a, x) = rO(0, a, x) * alpha;
      }
    }

    // P @ V. B fragments use (output feature, key) coordinates.
    const float* V_tile = V + batch * V_stride.b + head * V_stride.h + k * V_stride.t;
    for (int n = 0; n < DH; n += BN) {
      for (int x = tid; x < BK * BN; x += THREADS) {
        int row = x / BN;
        int col = x % BN;
        sV(row, col) = k + row < T_k ?
                        V_tile[row * V_stride.t + (col + n) * V_stride.d] : 0.f;
      }
      __syncthreads();
      auto rOn = make_tensor<float>(
        make_shape(Int<1>{}, Int<TM>{}, Int<VN>{})
      );
      const int segment_offset = (n / BN) * VN;
      CUTE_UNROLL
      for (int a = 0; a < TM; a++) {
        CUTE_UNROLL
        for (int b = 0; b < VN; b++) {
          rOn(0, a, b) = rO(0, a, segment_offset + b);
        }
      }

      CUTE_UNROLL
      for (int k_iter = 0; k_iter < BK; k_iter += RK) {
        auto rE = make_tensor<float>(
          make_shape(Int<1>{}, Int<TM>{}, Int<RK>{})
        );
        auto rV = make_tensor<float>(
          make_shape(Int<1>{}, Int<VN>{}, Int<RK>{})
        );
        CUTE_UNROLL
        for (int t = 0; t < RK; t++) {
          const int src_lane = lane_m + (t + k_iter) / TN * WM;
          CUTE_UNROLL
          for (int a = 0; a < TM; a++) {
            rE(0, a, t) = __shfl_sync(0xffffffff, rS(0, a, (t + k_iter) % TN), src_lane);
          }
          CUTE_UNROLL
          for (int x = 0; x < VN; ++x) {
            const int d = get<1>(tOc(0, 0, x));
            rV(0, x, t) = sV(k_iter + t, d);
          }
        }
        gemm(mma_PV, rE, rV, rOn);
      }

      CUTE_UNROLL
      for (int a = 0; a < TM; a++) {
        CUTE_UNROLL
        for (int b = 0; b < VN; b++) {
          rO(0, a, segment_offset + b) = rOn(0, a, b);
        }
      }
      __syncthreads(); // All V reads finish before the next V/K load.
    } // loop DH
  } // loop k

  CUTE_UNROLL
  for (int a = 0; a < TM; ++a) {
    const int64_t qi = q0 + get<0>(tOc(0, a, 0));
    if (qi < T_q) {
      CUTE_UNROLL
      for (int x = 0; x < DH / WN; ++x) {
        const int d = get<1>(tOc(0, a, x));
        if (blockIdx.y < B * H && d < DH){
          O[(blockIdx.y * T_q + qi) * DH + d] = l[a] > 0.f ? rO(0, a, x) / l[a] : 0.f;
        }
      }
      // One writer per row, not one writer per entire warp.
      if (lane_n == 0) {
        if(blockIdx.y < B * H){
          LSE[blockIdx.y * T_q + qi] = l[a] > 0.f ? m[a] + logf(l[a]) : -CUDART_INF_F;
        }
      }
    }
  }

} //kernel


} // namespace attention_detail
