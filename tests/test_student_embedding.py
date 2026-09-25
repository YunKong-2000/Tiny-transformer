"""Embedding kernel correctness. Model integration lives in test_student_integration.py."""
from pathlib import Path
from unittest.mock import Mock, patch
import subprocess
import sys
import unittest

import torch

from tiny_transformer.operators import Operators, reference, student
from tiny_transformer.operators._extension import load_embedding_extension

ROOT = Path(__file__).resolve().parents[1]


class StudentEmbeddingHostTests(unittest.TestCase):
    def test_autograd_bridge_with_cpu_test_double(self):
        # Verify Python graph wiring without pretending to execute a CUDA kernel.
        ids = torch.tensor([[2, 0, 2], [1, 2, 0]])
        weight = torch.randn(7, 5, requires_grad=True)
        expected_weight = weight.detach().clone().requires_grad_()

        def forward(saved_ids, saved_weight):
            self.assertFalse(torch.is_grad_enabled())
            return reference.embedding(saved_ids, saved_weight)

        def backward(saved_ids, gradient, vocab_size, implementation):
            self.assertTrue(gradient.is_contiguous())
            self.assertIn(implementation, ("grouped", "baseline"))
            return gradient.new_zeros(vocab_size, gradient.shape[-1]).index_add_(
                0, saved_ids.reshape(-1), gradient.reshape(-1, gradient.shape[-1]))

        extension = Mock(embedding_forward=Mock(side_effect=forward),
                         embedding_backward=Mock(side_effect=backward))
        with patch.object(student, "load_embedding_extension", return_value=extension):
            for use_sum, implementation in ((False, "grouped"), (True, "baseline")):
                actual = student._Embedding.apply(ids, weight, implementation)
                expected = reference.embedding(ids, expected_weight)
                self.assertIsNotNone(actual.grad_fn)
                torch.testing.assert_close(actual, expected)
                # Exercise a strided upstream, an expanded upstream from sum(),
                # tied-weight contributions, and accumulation across two calls.
                upstream = torch.randn(2, 3, 10)[..., ::2]
                actual_loss = actual.sum() if use_sum else actual
                expected_loss = expected.sum() if use_sum else expected
                gradients = (None if use_sum else upstream, None)
                torch.autograd.backward((actual_loss, weight.square().sum()), gradients)
                torch.autograd.backward((expected_loss, expected_weight.square().sum()), gradients)
                torch.testing.assert_close(weight.grad, expected_weight.grad)
            self.assertEqual(extension.embedding_backward.call_count, 2)
            self.assertEqual(extension.embedding_backward.call_args.args[2], 7)
            self.assertEqual([call.args[3] for call in extension.embedding_backward.call_args_list],
                             ["grouped", "baseline"])
            self.assertIsNone(ids.grad)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and nvcc")
class StudentEmbeddingCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = load_embedding_extension()

    def setUp(self):
        torch.manual_seed(42)

    def assert_gather(self, ids, weight):
        old_ids, old_weight = ids.clone(), weight.clone()
        actual = Operators({"embedding": "student"}).embedding(ids, weight)
        expected = reference.embedding(ids, weight)
        self.assertEqual(actual.shape, (*ids.shape, weight.shape[1]))
        self.assertEqual(actual.dtype, weight.dtype)
        self.assertEqual(actual.device, weight.device)
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(ids, old_ids, atol=0, rtol=0)
        torch.testing.assert_close(weight, old_weight, atol=0, rtol=0)
        if actual.numel():
            actual.fill_(123)
            torch.testing.assert_close(weight, old_weight, atol=0, rtol=0)

    @torch.no_grad()
    def test_scalar_vector_boundaries_and_real_shapes(self):
        cases = [
            (1, 1, 7, 1), (2, 17, 259, 3), (2, 17, 259, 4),
            (2, 17, 259, 31), (2, 17, 259, 32), (2, 17, 259, 33),
            (2, 17, 259, 64), (2, 17, 259, 65), (2, 17, 259, 132),
            (8, 512, 8192, 768), (4, 1, 8192, 768), (2, 0, 7, 65),
            # More rows than the capped grid's warps: exercise grid-stride loops.
            (1, 65535 * 8 + 1, 7, 4), (1, 65535 * 8 + 1, 7, 1),
        ]
        for batch, length, vocab, dim in cases:
            with self.subTest(shape=(batch, length, vocab, dim)):
                ids = torch.randint(vocab, (batch, length), device="cuda")
                if ids.numel():
                    ids[0, 0] = 0
                    ids[-1, -1] = vocab - 1
                weight = torch.randn(vocab, dim, device="cuda")
                self.assert_gather(ids, weight)

    def test_backward_noncontiguous_upstream_and_storage_offsets(self):
        ids = torch.tensor([99, 0, 6, 2, 2, 1, 0], device="cuda")[1:].view(2, 3)
        weight = torch.randn(7 * 65 + 1, device="cuda")[1:].view(7, 65).requires_grad_()
        upstreams = [torch.randn(2, 3, 130, device="cuda")[..., ::2],
                     torch.ones(1, device="cuda").expand(2, 3, 65),
                     torch.randn(2 * 3 * 65 + 1, device="cuda")[1:].view(2, 3, 65)]
        for upstream in upstreams:
            expected = torch.autograd.grad(reference.embedding(ids, weight), weight, upstream)[0]
            for implementation in ("grouped", "baseline"):
                actual = torch.autograd.grad(
                    student.embedding(ids, weight, backward_impl=implementation), weight, upstream)[0]
                torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)

    def test_vector_alignment_forward_and_backward(self):
        # A nonzero ID storage offset must not prevent valid float4 gradient loads.
        ids = torch.tensor([99, 0, 6, 2, 2, 1, 0], device="cuda")[1:].view(2, 3)
        for offset in (1, 4):
            weight = torch.randn(7 * 768 + offset, device="cuda")[offset:].view(7, 768)
            with torch.no_grad():
                self.assert_gather(ids, weight)
        for dim in (4, 65, 132, 768):
            for offset in (0, 1, 4):
                with self.subTest(dim=dim, gradient_offset=offset):
                    storage = torch.empty(6 * dim + offset, device="cuda")
                    upstream = storage[offset:].view(2, 3, dim)
                    # Nonzero exact integers expose every missing/misrouted channel.
                    upstream.copy_(torch.arange(1, 6 * dim + 1, device="cuda").view(2, 3, dim))
                    self.assertTrue(upstream.is_contiguous())
                    self.assertEqual(upstream.data_ptr() % 16, 4 if offset == 1 else 0)
                    weight = torch.randn(7, dim, device="cuda", requires_grad=True)
                    expected = torch.autograd.grad(reference.embedding(ids, weight), weight, upstream)[0]
                    actual = torch.autograd.grad(
                        student.embedding(ids, weight, backward_impl="baseline"), weight, upstream)[0]
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_native_wrapper_rejects_unsupported_inputs(self):
        ids = torch.zeros(2, 3, device="cuda", dtype=torch.long)
        weight = torch.randn(7, 64, device="cuda")
        cases = [
            (ids.cpu(), weight, "CUDA tensors"),
            (ids, weight.cpu(), "CUDA tensors"),
            (ids.int(), weight, "int64"),
            (ids, weight.bfloat16(), "float32"),
            (ids.flatten(), weight, "2D"),
            (ids, weight.flatten(), "2D"),
            (ids.t(), weight, "contiguous"),
            (ids, weight.t(), "contiguous"),
            (ids, torch.empty(0, 64, device="cuda"), "positive"),
            (ids, torch.empty(7, 0, device="cuda"), "positive"),
        ]
        for index, (bad_ids, bad_weight, message) in enumerate(cases):
            with self.subTest(case=index), self.assertRaisesRegex(RuntimeError, message):
                self.extension.embedding_forward(bad_ids, bad_weight)

    def test_backward_native_rejects_unsupported_inputs(self):
        ids = torch.zeros(2, 3, device="cuda", dtype=torch.long)
        gradient = torch.randn(2, 3, 65, device="cuda")
        cases = [
            (ids.cpu(), gradient, 7, "CUDA tensors"),
            (ids, gradient.cpu(), 7, "CUDA tensors"),
            (ids.int(), gradient, 7, "int64"),
            (ids, gradient.half(), 7, "float32"),
            (ids.flatten(), gradient, 7, "2D"),
            (ids, gradient.flatten(), 7, "3D"),
            (ids, gradient[:1], 7, "first two dimensions"),
            (ids.t(), gradient.transpose(0, 1), 7, "contiguous"),
            (ids, gradient[..., ::2], 7, "contiguous"),
            (ids, gradient[..., :0], 7, "positive"),
            (ids, gradient, 0, "positive"),
        ]
        for bad_ids, bad_gradient, vocab, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.extension.embedding_backward(bad_ids, bad_gradient, vocab)
        with self.assertRaisesRegex(RuntimeError, "implementation"):
            self.extension.embedding_backward(ids, gradient, 7, "typo")

    def test_backward_capped_grid(self):
        for implementation, rows_per_block in (("grouped", 256), ("baseline", 8)):
            with self.subTest(implementation=implementation):
                rows = 65535 * rows_per_block + 1
                ids = torch.zeros(1, rows, device="cuda", dtype=torch.long)
                gradient = torch.ones(1, rows, 1, device="cuda")
                output = self.extension.embedding_backward(ids, gradient, 2, implementation)
                expected = torch.tensor([[rows], [0]], device="cuda", dtype=torch.float32)
                torch.testing.assert_close(output, expected, atol=0, rtol=0)

    @torch.no_grad()
    def test_current_stream(self):
        for dim in (65, 768):
            ids = torch.zeros(2, 17, device="cuda", dtype=torch.long)
            weight = torch.zeros(7, dim, device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # Delay the producer to expose accidental launches on the default stream.
                torch.cuda._sleep(10_000_000)
                weight.fill_(7)
                actual = student.embedding(ids, weight)
                consumed = actual.clone()
            stream.synchronize()
            torch.testing.assert_close(consumed, torch.full_like(consumed, 7), atol=0, rtol=0)

    def test_backward_current_stream(self):
        ids = torch.zeros(1, 33, device="cuda", dtype=torch.long)
        gradient = torch.zeros(1, 33, 65, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            ids.fill_(3)
            gradient.fill_(2)
            output = self.extension.embedding_backward(ids, gradient, 7)
            consumed = output.clone()
        stream.synchronize()
        expected = torch.zeros_like(output)
        expected[3] = 66
        torch.testing.assert_close(consumed, expected, atol=0, rtol=0)

    def test_backward_patterns_and_row_boundaries(self):
        # Vary one boundary at a time instead of taking a large Cartesian product.
        cases = [(1, rows, 65) for rows in (1, 31, 32, 33, 257)]
        cases += [(2, 17, h) for h in (1, 4, 31, 32, 33, 768)] + [(2, 0, 65)]
        for batch, length, dim in cases:
            for pattern in ("same", "unique", "mixed"):
                with self.subTest(shape=(batch, length, dim), pattern=pattern):
                    vocab = max(67, batch * length)
                    ids = torch.arange(batch * length, device="cuda").reshape(batch, length)
                    if pattern == "same":
                        ids.fill_(vocab - 1)
                    elif pattern == "mixed":
                        ids = torch.randint(7, ids.shape, device="cuda")
                    weight = torch.randn(vocab, dim, device="cuda", requires_grad=True)
                    upstream = torch.randn(batch, length, dim, device="cuda")
                    expected = torch.autograd.grad(reference.embedding(ids, weight), weight, upstream)[0]
                    for implementation in ("grouped", "baseline"):
                        actual = torch.autograd.grad(student.embedding(ids, weight, backward_impl=implementation),
                                                     weight, upstream)[0]
                        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)

    def test_invalid_ids_fail_in_isolated_processes(self):
        # Each distinct kernel needs one lower/upper-bound check. Device assertions
        # poison the CUDA context, so these must stay in separate processes.
        for phase, variant in (("forward", 65), ("forward", 64),
                               ("backward", "grouped"), ("backward", "baseline")):
            for invalid in (-1, 7):
                with self.subTest(phase=phase, variant=variant, invalid=invalid):
                    call = (f"ext.embedding_forward(ids, torch.ones(7, {variant}, device='cuda'))"
                            if phase == "forward" else
                            f"ext.embedding_backward(ids, torch.ones(1, 1, 33, device='cuda'), 7, {variant!r})")
                    code = f"""
import torch
from tiny_transformer.operators._extension import load_embedding_extension
ext = load_embedding_extension()
ids = torch.tensor([[{invalid}]], device='cuda')
{call}
torch.cuda.synchronize()
"""
                    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                            capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("device-side assert", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
