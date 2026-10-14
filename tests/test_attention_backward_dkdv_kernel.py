"""Standalone dK/dV CUDA tests; do not depend on the unfinished attention binding.

Run on SM80+ with CUDA/nvcc:
    python -m unittest discover -s tests -p 'test_attention_backward_dkdv_kernel.py' -v
For memory and synchronization checks, prefix that command with
    compute-sanitizer --tool memcheck
or  compute-sanitizer --tool racecheck
"""
from pathlib import Path
import unittest

import torch

from tiny_transformer._cutlass import cutlass_include_paths


CPP_SOURCE = r"""
#include <torch/extension.h>
void dkdv_into(torch::Tensor q, torch::Tensor k, torch::Tensor v,
             torch::Tensor dout, torch::Tensor d, torch::Tensor lse,
             torch::Tensor dk, torch::Tensor dv, int64_t past_len);
"""

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <limits>
#include "attention_backward_dKdV_kernel.cu"

void dkdv_into(torch::Tensor q, torch::Tensor k, torch::Tensor v,
             torch::Tensor dout, torch::Tensor d, torch::Tensor lse,
             torch::Tensor dk, torch::Tensor dv, int64_t past_len) {
  namespace bf = attention_backward_dKdV;
  for (const auto& x : {q, k, v, dout, dk, dv}) {
    TORCH_CHECK(x.is_cuda() && x.device() == q.device());
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous());
    TORCH_CHECK(x.dim() == 4 && x.size(3) == bf::DH);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
  }
  const auto b = q.size(0), nh = q.size(1), tq = q.size(2), tk = k.size(2);
  TORCH_CHECK(b > 0 && b <= 65535 && nh > 0 && nh <= 65535);
  TORCH_CHECK(tq > 0 && tk >= tq && tk <= std::numeric_limits<int>::max());
  TORCH_CHECK(past_len >= 0 && past_len == tk - tq);
  TORCH_CHECK(k.size(0) == b && k.size(1) == nh && v.sizes() == k.sizes());
  TORCH_CHECK(dout.sizes() == q.sizes() && dk.sizes() == k.sizes() && dv.sizes() == v.sizes());
  for (const auto& x : {d, lse}) {
    TORCH_CHECK(x.is_cuda() && x.device() == q.device() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == at::kFloat && x.dim() == 3);
    TORCH_CHECK(x.size(0) == b && x.size(1) == nh && x.size(2) == tq);
  }
  const c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(at::cuda::getDeviceProperties(q.get_device())->major >= 8);
  const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
  dim3 grid((tk - 1) / bf::BK + 1, nh, b);
  bf::kernel<<<grid, bf::THREADS, 0, stream>>>(
      int(b), int(nh), int(tq), int(tk), int(past_len),
      reinterpret_cast<const bf::Element*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf::Element*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf::Element*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf::Element*>(dout.data_ptr<at::BFloat16>()),
      d.data_ptr<float>(), lse.data_ptr<float>(),
      reinterpret_cast<bf::Element*>(dk.data_ptr<at::BFloat16>()),
      reinterpret_cast<bf::Element*>(dv.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA, nvcc and SM80+')
class AttentionBackwardDKDVKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] < 8:
            raise unittest.SkipTest('BF16 MMA/cp.async require SM80+')
        from torch.utils.cpp_extension import CUDA_HOME, load_inline
        if CUDA_HOME is None:
            raise unittest.SkipTest('requires nvcc')
        source_dir = Path(__file__).resolve().parents[1] / 'csrc' / 'attention'
        cls.extension = load_inline(
            name='tiny_transformer_attention_dkdv_test',
            cpp_sources=CPP_SOURCE, cuda_sources=CUDA_SOURCE,
            functions=['dkdv_into'],
            extra_include_paths=[str(source_dir), *cutlass_include_paths()],
            extra_cflags=['-O2', '-std=c++17'],
            extra_cuda_cflags=['-O2', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo'],
        )

    def setUp(self):
        torch.manual_seed(123)

    def check_case(self, batch, heads, tq, tk, *, pattern='random', repeats=1):
        q = torch.randn(batch, heads, tq, 64, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(batch, heads, tk, 64, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        dout = torch.randn_like(q)
        if pattern == 'zero_upstream':
            dout.zero_()
        elif pattern == 'uniform':
            q.zero_()
        elif pattern == 'late_queries':
            # Only later query tiles contribute: exercise alternating slots.
            dout[:, :, :128].zero_()

        # FP64 autograd is independent of the kernel's tiled backward formula.
        rq, upstream = q.double(), dout.double()
        rk = k.double().requires_grad_()
        rv = v.double().requires_grad_()
        past_len = tk - tq
        allowed = (torch.arange(tk, device='cuda')[None, :] <=
                   torch.arange(tq, device='cuda')[:, None] + past_len)
        scores = (rq @ rk.transpose(-1, -2) * 0.125).masked_fill(~allowed, -torch.inf)
        output = scores.softmax(-1) @ rv
        expected = torch.autograd.grad(output, (rk, rv), upstream)
        # Isolate dK/dV from forward/preprocess rounding: use exact reference O,
        # rounded only when saving the FP32 D/LSE inputs required by this kernel.
        d = (output.detach() * upstream).sum(-1).float().contiguous()
        lse = scores.detach().logsumexp(-1).float().contiguous()
        inputs = (q, k, v, dout, d, lse)
        originals = tuple(x.clone() for x in inputs)

        # NaNs catch missing stores; aligned guards catch output tail writes.
        storages = tuple(torch.full((k.numel() + 128,), 17, device='cuda', dtype=k.dtype)
                         for _ in range(2))
        actual = tuple(storage[64:-64].view_as(k) for storage in storages)
        previous = None
        for _ in range(repeats):
            for result in actual:
                result.fill_(float('nan'))
            self.extension.dkdv_into(*inputs, *actual, past_len)
            torch.cuda.current_stream().synchronize()
            for index, (result, reference) in enumerate(zip(actual, expected)):
                self.assertTrue(torch.isfinite(result).all().item())
                # BF16 MMA operands and BF16 output introduce rounding.
                torch.testing.assert_close(result.double(), reference, atol=1e-2, rtol=3e-2)
                if pattern == 'zero_upstream':
                    self.assertEqual(torch.count_nonzero(result).item(), 0)
                if previous is not None:
                    torch.testing.assert_close(result, previous[index], atol=0, rtol=0)
            previous = tuple(result.clone() for result in actual)
        for storage in storages:
            self.assertTrue((storage[:64] == 17).all().item())
            self.assertTrue((storage[-64:] == 17).all().item())
        if tq == tk == 1:
            # One allowed key: P=1, so dV=dO and dK=0.
            torch.testing.assert_close(actual[1], dout, atol=0, rtol=0)
            torch.testing.assert_close(actual[0], torch.zeros_like(k), atol=1e-5, rtol=0)
        for x, original in zip(inputs, originals):
            torch.testing.assert_close(x, original, atol=0, rtol=0)

    def test_causal_single_multiple_and_tail_tiles(self):
        for length in (1, 63, 64, 65, 128, 129):
            with self.subTest(length=length):
                self.check_case(2, 3, length, length)

    def test_past_context_and_independent_query_key_tails(self):
        for tq, tk in ((1, 129), (17, 64), (63, 129), (64, 65), (65, 193)):
            with self.subTest(tq=tq, tk=tk):
                self.check_case(2, 3, tq, tk)

    def test_zero_upstream_and_uniform_probabilities(self):
        for pattern in ('zero_upstream', 'uniform'):
            with self.subTest(pattern=pattern):
                self.check_case(2, 2, 65, 129, pattern=pattern)

    def test_stage_reuse_on_current_nondefault_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.check_case(2, 3, 257, 321, pattern='late_queries', repeats=5)


if __name__ == '__main__':
    unittest.main()
