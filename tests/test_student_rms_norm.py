"""RMSNorm integration tests; CUDA cases JIT-build and execute the real kernel."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from tiny_transformer.config import ModelConfig
from tiny_transformer.model import Transformer
from tiny_transformer.operators import Operators, reference, student
from tiny_transformer.operators._extension import load_embedding_extension, load_rms_norm_extension

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def full_precision_matmul():
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


class StudentRMSNormHostTests(unittest.TestCase):
    def test_import_and_cpu_rejection_do_not_load_compiler(self):
        result = subprocess.run([sys.executable, "-c", """
import sys
import torch
from tiny_transformer.operators import Operators
assert 'torch.utils.cpp_extension' not in sys.modules
try:
    Operators({'rms_norm': 'student'}).rms_norm(torch.ones(2, 3, 65), torch.ones(65), 1e-6)
except RuntimeError as error:
    assert 'CUDA' in str(error)
else:
    raise AssertionError('CPU input was accepted')
assert 'torch.utils.cpp_extension' not in sys.modules
"""], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_jit_modules_have_independent_sources_and_cache(self):
        # Mock compilation only: this checks build configuration, not GPU correctness.
        import torch.utils.cpp_extension as cpp_extension

        loaders = (load_rms_norm_extension, load_embedding_extension)
        for loader in loaders:
            loader.cache_clear()
        try:
            with patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(cpp_extension, "CUDA_HOME", "/test/cuda"), \
                    patch.object(cpp_extension, "load", side_effect=[object(), object()]) as build:
                rms_module = load_rms_norm_extension()
                embedding_module = load_embedding_extension()
                self.assertIs(load_rms_norm_extension(), rms_module)
                self.assertIs(load_embedding_extension(), embedding_module)
                self.assertIsNot(rms_module, embedding_module)
            self.assertEqual(build.call_count, 2)
            rms_args, embedding_args = [call.kwargs for call in build.call_args_list]
            self.assertNotEqual(rms_args['name'], embedding_args['name'])
            self.assertTrue(rms_args['with_cuda'])
            self.assertEqual({Path(p).name for p in rms_args['sources']},
                             {'bindings.cpp', 'rms_norm.cu'})
            self.assertTrue(all(Path(p).is_file() for p in rms_args['sources']))
            self.assertTrue(all(Path(p).parent.name == 'rms_norm' for p in rms_args['sources']))
            self.assertTrue(set(rms_args['sources']).isdisjoint(embedding_args['sources']))
        finally:
            for loader in loaders:
                loader.cache_clear()

    def test_loader_reports_missing_cuda_and_toolkit(self):
        import torch.utils.cpp_extension as cpp_extension

        load_rms_norm_extension.cache_clear()
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'rms_norm.*CUDA'):
                load_rms_norm_extension()
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(cpp_extension, 'CUDA_HOME', None):
            with self.assertRaisesRegex(RuntimeError, 'nvcc'):
                load_rms_norm_extension()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and nvcc")
class StudentRMSNormCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = load_rms_norm_extension()

    def setUp(self):
        torch.manual_seed(42)

    def assert_norm(self, x, weight, eps=1e-6):
        old_x, old_weight = x.clone(), weight.clone()
        expected = reference.rms_norm(x, weight, eps)
        actual = Operators({'rms_norm': 'student'}).rms_norm(x, weight, eps)
        native = self.extension.rms_norm_forward(x.contiguous(), weight.contiguous(), eps)
        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, x.dtype)
        self.assertEqual(actual.device, x.device)
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4, equal_nan=True)
        torch.testing.assert_close(native, actual, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(x, old_x, atol=0, rtol=0)
        torch.testing.assert_close(weight, old_weight, atol=0, rtol=0)
        actual.fill_(123)
        torch.testing.assert_close(x, old_x, atol=0, rtol=0)
        torch.testing.assert_close(weight, old_weight, atol=0, rtol=0)

    @torch.no_grad()
    def test_warp_tails_real_shapes_and_capped_grid(self):
        shapes = [(2, 17, h) for h in (1, 3, 31, 32, 33, 64, 65, 768, 1025)]
        shapes += [(8, 512, 768), (4, 1, 768), (65,), (7, 65), (2, 3, 4, 65),
                   (1, 65535 * 8 + 1, 3)]
        for shape in shapes:
            with self.subTest(shape=shape):
                self.assert_norm(torch.randn(shape, device='cuda'),
                                 torch.randn(shape[-1], device='cuda'))

    @torch.no_grad()
    def test_zero_input_scales_weights_and_epsilon(self):
        weight = torch.linspace(-3, 3, 65, device='cuda')
        for scale in (0, 1e-8, 1, 1e6):
            for eps in (1e-12, 1e-6, 0.1):
                with self.subTest(scale=scale, eps=eps):
                    self.assert_norm(torch.randn(2, 17, 65, device='cuda') * scale, weight, eps)
        self.assert_norm(torch.ones(2, 3, 65, device='cuda'), weight, 0.0)
        # eps=0 and a zero row follow reference's NaN semantics, without clamping.
        self.assert_norm(torch.zeros(2, 3, 65, device='cuda'), weight, 0.0)
        # Multiplying X * weight before normalization would overflow here.
        self.assert_norm(torch.full((2, 3, 65), 1e10, device='cuda'),
                         torch.full((65,), 1e30, device='cuda'))

    @torch.no_grad()
    def test_strided_inputs_and_storage_offsets(self):
        base = torch.randn(2, 5, 130, device='cuda')
        weight = torch.randn(130, device='cuda')[::2]
        inputs = [base[..., ::2], base[..., :65][:, -1:, :],
                  base[..., :65].transpose(0, 1),
                  torch.randn(65, device='cuda').expand(2, 3, 65)]
        for x in inputs:
            self.assertFalse(x.is_contiguous())
            self.assert_norm(x, weight)
        x = torch.randn(2 * 3 * 65 + 1, device='cuda')[1:].view(2, 3, 65)
        weight = torch.randn(66, device='cuda')[1:]
        self.assertTrue(x.is_contiguous() and weight.is_contiguous())
        self.assertGreater(x.storage_offset(), 0)
        self.assert_norm(x, weight)

    @torch.no_grad()
    def test_empty_leading_dimensions(self):
        for shape in ((0, 17, 65), (2, 0, 65), (0, 65)):
            self.assert_norm(torch.empty(shape, device='cuda'), torch.randn(65, device='cuda'))

    def test_native_rejects_unsupported_inputs(self):
        x, weight = torch.randn(2, 3, 65, device='cuda'), torch.randn(65, device='cuda')
        cases = [(x.cpu(), weight, 'CUDA tensors'), (x, weight.cpu(), 'CUDA tensors'),
                 (x.half(), weight, 'float32'), (x, weight.bfloat16(), 'float32'),
                 (x.double(), weight.double(), 'float32'),
                 (x[0, 0, 0], weight, 'at least one dimension'),
                 (x, weight[None, :], '1D'), (x, weight[:64], 'same last dimension'),
                 (x.transpose(0, 1), weight, 'contiguous'),
                 (x, torch.randn(130, device='cuda')[::2], 'contiguous'),
                 (x.to_sparse(), weight, 'strided layout'),
                 (torch.empty(2, 3, 0, device='cuda'), weight[:0], 'positive')]
        for value, scale, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.extension.rms_norm_forward(value, scale, 1e-6)
        for eps in (-1e-6, float('nan'), float('inf'), -float('inf')):
            with self.subTest(eps=eps), self.assertRaisesRegex(RuntimeError, 'epsilon'):
                self.extension.rms_norm_forward(x, weight, eps)

    def test_forward_only_gradient_guard(self):
        for x_grad, weight_grad in ((True, False), (False, True), (True, True)):
            x = torch.randn(2, 3, 65, device='cuda', requires_grad=x_grad)
            weight = torch.randn(65, device='cuda', requires_grad=weight_grad)
            for function in (student.rms_norm, self.extension.rms_norm_forward):
                with self.assertRaisesRegex(RuntimeError, 'backward is not implemented'):
                    function(x, weight, 1e-6)
                for mode in (torch.no_grad, torch.inference_mode):
                    with mode():
                        actual = function(x, weight, 1e-6)
                        self.assertFalse(actual.requires_grad)
                        torch.testing.assert_close(actual, reference.rms_norm(x, weight, 1e-6),
                                                   atol=1e-5, rtol=1e-4)
        # Grad mode itself is fine when neither input requires gradients.
        self.assert_norm(x.detach(), weight.detach())

    @torch.no_grad()
    def test_autocast_preserves_fp32(self):
        with torch.autocast('cuda', dtype=torch.float16):
            self.assert_norm(torch.randn(2, 17, 65, device='cuda'),
                             torch.randn(65, device='cuda'))

    @torch.no_grad()
    def test_current_stream(self):
        x, weight = torch.zeros(2, 17, 65, device='cuda'), torch.zeros(65, device='cuda')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            x.fill_(2)
            weight.fill_(3)
            actual = student.rms_norm(x, weight, 1e-6).clone()
            expected = reference.rms_norm(x, weight, 1e-6)
        stream.synchronize()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'requires two CUDA devices')
    @torch.no_grad()
    def test_device_guard_and_mismatched_devices(self):
        with torch.cuda.device(0):
            x = torch.randn(2, 3, 65, device='cuda:1')
            weight = torch.randn(65, device='cuda:1')
            self.assert_norm(x, weight)
            self.assertEqual(torch.cuda.current_device(), 0)
            with self.assertRaisesRegex(RuntimeError, 'same CUDA device'):
                self.extension.rms_norm_forward(x.to('cuda:0'), weight, 1e-6)

    @torch.inference_mode()
    def test_model_full_prefill_last_only_and_cached_decode(self):
        with full_precision_matmul():
            model = Transformer(ModelConfig(vocab_size=31, dim=64, n_layers=2,
                                            n_heads=4, hidden_dim=96, max_seq_len=32)).cuda().eval()
            ids = torch.randint(31, (2, 11), device='cuda')
            chunks = [ids[:, :4]] + [ids[:, i:i + 1] for i in range(4, 11)]
            expected_full = model(ids)
            expected_last = model(ids, last_only=True)
            reference_cache = model.new_cache(2, 16)
            expected_parts = [model(chunk, cache=reference_cache, last_only=True) for chunk in chunks]
            model.ops = Operators({'rms_norm': 'student'})
            torch.testing.assert_close(model(ids), expected_full, atol=2e-5, rtol=1e-4)
            torch.testing.assert_close(model(ids, last_only=True), expected_last, atol=2e-5, rtol=1e-4)
            cache = model.new_cache(2, 16)
            end = 0
            for chunk, expected in zip(chunks, expected_parts):
                end += chunk.shape[1]
                actual = model(chunk, cache=cache, last_only=True)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
                torch.testing.assert_close(actual, expected_full[:, end - 1:end], atol=2e-5, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
