#pragma once

#include "attention_forward_BF16_kernel.cuh"

namespace attention_bf16 {

constexpr int DECODE_KEYS = 128;
constexpr int DECODE_THREADS = 256;
constexpr int DECODE_WARPS = DECODE_THREADS / 32;
constexpr int DECODE_GROUPS = DECODE_THREADS / DH;
constexpr int DECODE_PARTIAL = DH + 2; // FP32 numerator[64], row max, exp sum.
constexpr int DECODE_MAX_KEYS = 4096;

template <bool MAXIMUM>
__device__ __forceinline__ float decode_warp_reduce(float x) {
  CUTE_UNROLL
  for (int offset = 16; offset > 0; offset /= 2) {
    const float other = __shfl_xor_sync(0xffffffffu, x, offset);
    if constexpr (MAXIMUM) x = fmaxf(x, other);
    else x += other;
  }
  return x;
}

// All threads participate. A caller reusing scratch must rendezvous after all
// threads have read scratch[0], before invoking another reduction.
template <bool MAXIMUM>
__device__ __forceinline__ float decode_block_reduce(float x, float* scratch) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  x = decode_warp_reduce<MAXIMUM>(x);
  if (lane == 0) scratch[warp] = x;
  __syncthreads();
  if (warp == 0) {
    x = lane < DECODE_WARPS ? scratch[lane] : (MAXIMUM ? -CUDART_INF_F : 0.f);
    x = decode_warp_reduce<MAXIMUM>(x);
    if (lane == 0) scratch[0] = x;
  }
  __syncthreads();
  return scratch[0];
}

// Only Tq=1: all keys are causal because the host enforces Tk=past_len+1.
// Each CTA processes one head and 128 keys, without padding the query to 64 rows.
template <bool FINAL>
__global__ __launch_bounds__(DECODE_THREADS)
void decode_partial(const Element* q, const Element* k, const Element* v,
                    Element* o, float* lse, float* partials, int64_t tk, int splits) {
  __shared__ float weights[DECODE_KEYS];
  __shared__ float scratch[DECODE_WARPS];
  __shared__ float numerators[DECODE_GROUPS][DH];
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int64_t bh = blockIdx.x;
  const int split = blockIdx.y;
  const int64_t key0 = int64_t(split) * DECODE_KEYS;
  const int keys = int(tk - key0 < DECODE_KEYS ? tk - key0 : DECODE_KEYS);
  q += bh * DH;
  k += (bh * tk + key0) * DH;
  v += (bh * tk + key0) * DH;
  const float q_lo = float(q[lane]);
  const float q_hi = float(q[lane + 32]);

  CUTE_UNROLL
  for (int j = warp; j < DECODE_KEYS; j += DECODE_WARPS) {
    float score = 0.f;
    if (j < keys) {
      score = q_lo * float(k[j * DH + lane]);
      score = fmaf(q_hi, float(k[j * DH + lane + 32]), score);
    }
    score = decode_warp_reduce<false>(score);
    if (lane == 0) weights[j] = j < keys ? score * SCALE : -CUDART_INF_F;
  }
  __syncthreads();

  const float score = tid < DECODE_KEYS ? weights[tid] : -CUDART_INF_F;
  const float row_max = decode_block_reduce<true>(score, scratch);
  const float p = tid < keys ? __expf(score - row_max) : 0.f;
  if (tid < DECODE_KEYS) weights[tid] = p;
  __syncthreads(); // Publish P and finish reading max before scratch is reused.
  const float row_sum = decode_block_reduce<false>(p, scratch);

  const int d = tid % DH;
  const int group = tid / DH;
  float numerator = 0.f;
  for (int j = group; j < keys; j += DECODE_GROUPS) {
    numerator = fmaf(weights[j], float(v[j * DH + d]), numerator);
  }
  numerators[group][d] = numerator;
  __syncthreads();
  if (tid < DH) {
    numerator = 0.f;
    CUTE_UNROLL
    for (int g = 0; g < DECODE_GROUPS; ++g) numerator += numerators[g][tid];
    if constexpr (FINAL) {
      o[bh * DH + tid] = Element(numerator / row_sum);
      if (tid == 0) lse[bh] = row_max + logf(row_sum);
    } else {
      float* dst = partials + (bh * splits + split) * DECODE_PARTIAL;
      dst[tid] = numerator;
      if (tid == 0) {
        dst[DH] = row_max;
        dst[DH + 1] = row_sum;
      }
    }
  }
}

// Merge unnormalized numerators with max-rescaled sums, not averaged outputs.
__global__ __launch_bounds__(DH)
void decode_merge(const float* partials, Element* o, float* lse, int splits) {
  const int64_t bh = blockIdx.x;
  const int d = threadIdx.x;
  const float* src = partials + bh * splits * DECODE_PARTIAL;
  float row_max = -CUDART_INF_F;
  for (int s = 0; s < splits; ++s)
    row_max = fmaxf(row_max, src[s * DECODE_PARTIAL + DH]);
  float numerator = 0.f, denominator = 0.f;
  for (int s = 0; s < splits; ++s) {
    const float* part = src + s * DECODE_PARTIAL;
    const float factor = __expf(part[DH] - row_max);
    numerator = fmaf(factor, part[d], numerator);
    denominator = fmaf(factor, part[DH + 1], denominator);
  }
  o[bh * DH + d] = Element(numerator / denominator);
  if (d == 0) lse[bh] = row_max + logf(denominator);
}

} // namespace attention_bf16
