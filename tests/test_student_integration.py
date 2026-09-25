"""Shared student integration: runtime contract, model training and KV cache."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import unittest

import torch

from tiny_transformer.config import ModelConfig
from tiny_transformer.model import Transformer
from tiny_transformer.operators import Operators
from tiny_transformer.operators._extension import (
    load_embedding_extension, load_residual_extension, load_rms_norm_extension,
)


@contextmanager
def full_precision_matmul():
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


class StudentRuntimeHostTests(unittest.TestCase):
    def test_cpu_rejection_does_not_load_compiler(self):
        # One fresh process checks all public entries and import-time laziness.
        code = """
import sys
import torch
from tiny_transformer.operators import student
assert 'torch.utils.cpp_extension' not in sys.modules
for function, args in (
    (student.embedding, (torch.zeros(1, 1, dtype=torch.long), torch.ones(7, 4))),
    (student.rms_norm, (torch.ones(2, 3, 4), torch.ones(4), 1e-6)),
    (student.residual, (torch.ones(2, 3, 4), torch.ones(2, 3, 4))),
):
    try:
        function(*args)
    except RuntimeError as error:
        assert 'CUDA' in str(error)
    else:
        raise AssertionError('CPU input was accepted')
assert 'torch.utils.cpp_extension' not in sys.modules
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA and nvcc')
class StudentIntegrationCudaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = ModelConfig(vocab_size=31, dim=64, n_layers=1, n_heads=4,
                                  hidden_dim=96, max_seq_len=16)

    def test_model_training_and_cached_inference(self):
        for names in (('embedding',), ('rms_norm',), ('residual',),
                      ('embedding', 'rms_norm'), ('embedding', 'rms_norm', 'residual')):
            with self.subTest(operators=names), full_precision_matmul():
                expected, actual = Transformer(self.config).cuda(), Transformer(self.config).cuda()
                actual.load_state_dict(expected.state_dict())
                actual.ops = Operators(dict.fromkeys(names, 'student'))
                self.assertIs(actual.embedding, actual.output_weight)
                # Accumulated parameter gradients exercise tied embedding weights
                # and norm/residual calls, in FP32 and in the model's AMP path.
                for amp in (False, True):
                    expected.zero_grad(set_to_none=True)
                    actual.zero_grad(set_to_none=True)
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    for _ in range(2):
                        ids, targets = [torch.randint(31, (2, 5), device='cuda') for _ in range(2)]
                        for model in (expected, actual):
                            with torch.autocast('cuda', dtype=dtype, enabled=amp):
                                loss = model.loss(ids, targets)
                            (loss / 2).backward()
                    atol, rtol = (3e-3, 3e-2) if amp else (3e-5, 3e-4)
                    for (name, value), (_, target) in zip(actual.named_parameters(), expected.named_parameters()):
                        with self.subTest(amp=amp, parameter=name):
                            torch.testing.assert_close(value.grad, target.grad, atol=atol, rtol=rtol)
                expected.eval()
                actual.eval()
                with torch.inference_mode():
                    ids = torch.randint(31, (2, 7), device='cuda')
                    atol, rtol = (0, 0) if names == ('embedding',) else (2e-5, 1e-4)
                    torch.testing.assert_close(actual(ids), expected(ids), atol=atol, rtol=rtol)
                    caches = [model.new_cache(2, 12) for model in (expected, actual)]
                    # last_only with B>1 exposes the final norm's strided prefill input.
                    chunks = [ids[:, :4].contiguous(), ids[:, 4:5].contiguous(), ids[:, 5:].contiguous()]
                    for chunk in chunks:
                        left = actual(chunk, cache=caches[1], last_only=True)
                        right = expected(chunk, cache=caches[0], last_only=True)
                        torch.testing.assert_close(left, right, atol=2e-5, rtol=1e-4)

    def test_native_grad_guards_and_deterministic_backward(self):
        embedding, norm = load_embedding_extension(), load_rms_norm_extension()
        residual = load_residual_extension()
        ids = torch.zeros(2, 3, device='cuda', dtype=torch.long)
        x = torch.ones(2, 3, 65, device='cuda')
        ew, nw = torch.ones(7, 65, device='cuda', requires_grad=True), torch.ones(65, device='cuda', requires_grad=True)
        residual_grad = x.clone().requires_grad_()
        for call in (lambda: embedding.embedding_forward(ids, ew),
                     lambda: norm.rms_norm_forward(x, nw, 1e-6),
                     lambda: residual.residual_forward(residual_grad, x),
                     lambda: residual.residual_forward(x, residual_grad)):
            with self.assertRaisesRegex(RuntimeError, 'autograd binding'):
                call()
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            # Residual uses no atomics and remains valid in strict deterministic mode.
            from tiny_transformer.operators import student
            student.residual(residual_grad, residual_grad).sum().backward()
            torch.testing.assert_close(residual_grad.grad, torch.full_like(x, 2), atol=0, rtol=0)
            with torch.no_grad():
                _, r = norm.rms_norm_forward(x, nw, 1e-6)
                for call in (lambda: embedding.embedding_backward(ids, x, 7),
                             lambda: norm.rms_norm_backward(x, x, nw, r)):
                    with self.assertRaisesRegex(RuntimeError, 'deterministic'):
                        call()
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


if __name__ == '__main__':
    unittest.main()
