"""Student embedding integration tests; CUDA cases JIT-build the real extension."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch
import subprocess
import sys
import unittest

import torch

from tiny_transformer.config import ModelConfig
from tiny_transformer.model import Transformer
from tiny_transformer.operators import Operators, reference, student
from tiny_transformer.operators._extension import load_embedding_extension


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def full_precision_matmul():
    # Full-sequence and cached GEMMs have different shapes. Under TF32 their
    # numerical differences can exceed this FP32 correctness test's tolerance.
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


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

    def test_autograd_bridge_detects_modified_saved_ids(self):
        ids = torch.tensor([[0, 1]])
        weight = torch.randn(7, 5, requires_grad=True)
        extension = Mock(embedding_forward=Mock(side_effect=reference.embedding))
        with patch.object(student, "load_embedding_extension", return_value=extension):
            output = student._Embedding.apply(ids, weight, "grouped")
            ids[0, 0] = 2
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                output.sum().backward()
        extension.embedding_backward.assert_not_called()

    def test_backward_implementation_is_saved_per_graph(self):
        ids = torch.tensor([[0, 1]])
        weight = torch.randn(7, 5, requires_grad=True)
        extension = Mock(embedding_forward=Mock(side_effect=reference.embedding),
                         embedding_backward=Mock(return_value=torch.ones_like(weight)))
        with patch.object(student, "load_embedding_extension", return_value=extension):
            grouped = student._Embedding.apply(ids, weight, "grouped")
            baseline = student._Embedding.apply(ids, weight, "baseline")
            torch.autograd.grad(baseline.sum(), weight)
            torch.autograd.grad(grouped.sum(), weight)
        self.assertEqual([call.args[3] for call in extension.embedding_backward.call_args_list],
                         ["baseline", "grouped"])

    def test_unknown_backward_implementation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "backward_impl"):
            student.embedding(torch.zeros(1, 1, dtype=torch.long), torch.ones(2, 4),
                              backward_impl="typo")

    def test_import_and_cpu_rejection_do_not_load_compiler(self):
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
import torch
from tiny_transformer.operators import Operators
assert 'torch.utils.cpp_extension' not in sys.modules
try:
    Operators({'embedding': 'student'}).embedding(
        torch.zeros(1, 1, dtype=torch.long), torch.ones(2, 4))
except RuntimeError as error:
    assert 'CUDA' in str(error)
else:
    raise AssertionError('CPU input was accepted')
assert 'torch.utils.cpp_extension' not in sys.modules
"""], cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


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
            (8, 512, 8192, 768), (1, 512, 8192, 768), (4, 1, 8192, 768),
            # More rows than the capped grid's warps: exercise grid-stride loops.
            (1, 65535 * 8 + 1, 7, 4), (1, 65535 * 8 + 1, 7, 1),
        ]
        for batch, length, vocab, dim in cases:
            with self.subTest(shape=(batch, length, vocab, dim)):
                ids = torch.randint(vocab, (batch, length), device="cuda")
                ids[0, 0] = 0
                ids[-1, -1] = vocab - 1
                weight = torch.randn(vocab, dim, device="cuda")
                self.assert_gather(ids, weight)
        for dim in (65, 768):
            self.assert_gather(torch.full((2, 17), 3, device="cuda", dtype=torch.long),
                               torch.randn(7, dim, device="cuda"))

    @torch.no_grad()
    def test_contiguous_views_with_storage_offsets(self):
        ids = torch.tensor([99, 0, 6, 2, 2, 1, 0], device="cuda")[1:].view(2, 3)
        for offset in (1, 4):
            with self.subTest(offset=offset):
                weight = torch.randn(7 * 768 + offset, device="cuda")[offset:].view(7, 768)
                self.assertTrue(weight.is_contiguous())
                self.assertEqual(weight.data_ptr() % 16, 4 if offset == 1 else 0)
                self.assert_gather(ids, weight)

    @torch.no_grad()
    def test_empty_ids(self):
        for shape in ((0, 17), (2, 0), (0, 0)):
            self.assert_gather(torch.empty(shape, device="cuda", dtype=torch.long),
                               torch.randn(7, 64, device="cuda"))

    def test_native_wrapper_rejects_unsupported_inputs(self):
        ids = torch.zeros(2, 3, device="cuda", dtype=torch.long)
        weight = torch.randn(7, 64, device="cuda")
        cases = [
            (ids.cpu(), weight, "CUDA tensors"),
            (ids, weight.cpu(), "CUDA tensors"),
            (ids.int(), weight, "int64"),
            (ids, weight.bfloat16(), "float32"),
            (ids, weight.half(), "float32"),
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

    def test_native_grad_guard_and_amp_forward_backward(self):
        ids = torch.zeros(2, 3, device="cuda", dtype=torch.long)
        weight = torch.randn(7, 64, device="cuda", requires_grad=True)
        # Calling pybind directly does not attach the Python autograd node.
        with self.assertRaisesRegex(RuntimeError, "use student.embedding"):
            self.extension.embedding_forward(ids, weight)
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                actual = student.embedding(ids, weight)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertFalse(actual.requires_grad)
                torch.testing.assert_close(actual, reference.embedding(ids, weight), atol=0, rtol=0)
            weight.grad = None
            expected_weight = weight.detach().clone().requires_grad_()
            with torch.autocast("cuda", dtype=dtype):
                actual = student.embedding(ids, weight)
                expected = reference.embedding(ids, expected_weight)
                self.assertEqual(actual.dtype, torch.float32)
                upstream = torch.randn_like(actual)
                actual_loss = (actual * upstream).sum()
                expected_loss = (expected * upstream).sum()
            actual_loss.backward()
            expected_loss.backward()
            self.assertEqual(weight.grad.dtype, torch.float32)
            torch.testing.assert_close(weight.grad, expected_weight.grad, atol=3e-5, rtol=3e-4)

    def test_backward_grouping_tails_and_empty_inputs(self):
        cases = [(1, rows, 67, dim)
                 for rows in (1, 17, 31, 32, 33, 257)
                 for dim in (1, 31, 32, 33, 65)]
        cases += [(8, 512, 8192, 768), (0, 17, 7, 65), (2, 0, 7, 65)]
        for batch, length, vocab, dim in cases:
            for pattern in ("same", "distinct", "mixed"):
                with self.subTest(shape=(batch, length, vocab, dim), pattern=pattern):
                    ids = torch.arange(batch * length, device="cuda").reshape(batch, length)
                    if pattern == "same":
                        ids.fill_(vocab - 1)
                    elif pattern == "distinct":
                        ids.remainder_(vocab)
                    else:
                        ids = torch.randint(vocab, ids.shape, device="cuda")
                    weight = torch.randn(vocab, dim, device="cuda", requires_grad=True)
                    upstream = torch.randn(batch, length, dim, device="cuda")
                    if pattern == "same":
                        # Exact binary fractions make even 4096 repeated IDs an
                        # exact routing/counting check, independent of sum order.
                        upstream = torch.randint(-8, 9, upstream.shape, device="cuda").float() / 8
                    expected = torch.autograd.grad(reference.embedding(ids, weight), weight, upstream)[0]
                    native = self.extension.embedding_backward(ids, upstream, vocab)
                    actual = torch.autograd.grad(student.embedding(ids, weight), weight, upstream)[0]
                    baseline_native = self.extension.embedding_backward(ids, upstream, vocab, "baseline")
                    baseline = torch.autograd.grad(
                        student.embedding(ids, weight, backward_impl="baseline"), weight, upstream)[0]
                    for result in (native, actual, baseline_native, baseline):
                        self.assertEqual(result.shape, weight.shape)
                        self.assertEqual(result.dtype, weight.dtype)
                        self.assertEqual(result.device, weight.device)
                        atol, rtol = (0, 0) if pattern == "same" else (3e-5, 3e-4)
                        torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)

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

    def test_backward_native_rejects_unsupported_inputs(self):
        ids = torch.zeros(2, 3, device="cuda", dtype=torch.long)
        gradient = torch.randn(2, 3, 65, device="cuda")
        cases = [
            (ids.cpu(), gradient, 7, "CUDA tensors"),
            (ids, gradient.cpu(), 7, "CUDA tensors"),
            (ids.int(), gradient, 7, "int64"),
            (ids, gradient.half(), 7, "float32"),
            (ids, gradient.bfloat16(), 7, "float32"),
            (ids.flatten(), gradient, 7, "2D"),
            (ids, gradient.flatten(), 7, "3D"),
            (ids, gradient[:1], 7, "first two dimensions"),
            (ids.t(), gradient.transpose(0, 1), 7, "contiguous"),
            (ids, gradient[..., ::2], 7, "contiguous"),
            (ids, gradient[..., :0], 7, "positive"),
            (ids, gradient, 0, "positive"),
            (ids, gradient, -1, "positive"),
        ]
        for bad_ids, bad_gradient, vocab, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.extension.embedding_backward(bad_ids, bad_gradient, vocab)
        with self.assertRaisesRegex(RuntimeError, "implementation"):
            self.extension.embedding_backward(ids, gradient, 7, "typo")

    def test_backward_respects_deterministic_mode(self):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            ids = torch.zeros(1, 33, device="cuda", dtype=torch.long)
            gradient = torch.ones(1, 33, 65, device="cuda")
            with self.assertRaisesRegex(RuntimeError, "embedding_backward_cuda"):
                self.extension.embedding_backward(ids, gradient, 7)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    def test_model_training_with_tied_weight_and_accumulation(self):
        with full_precision_matmul():
            expected_model = Transformer(ModelConfig(vocab_size=31, dim=32, n_layers=1,
                                                    n_heads=4, hidden_dim=48, max_seq_len=32)).cuda()
            actual_model = Transformer(expected_model.config).cuda()
            actual_model.load_state_dict(expected_model.state_dict())
            actual_model.ops = Operators({"embedding": "student"})
            self.assertIs(actual_model.embedding, actual_model.output_weight)
            for _ in range(2):
                ids = torch.randint(4, (2, 17), device="cuda")
                expected, actual = expected_model(ids), actual_model(ids)
                upstream = torch.randn_like(actual)
                (expected * upstream).sum().backward()
                (actual * upstream).sum().backward()
                for (name, left), (_, right) in zip(actual_model.named_parameters(),
                                                    expected_model.named_parameters()):
                    with self.subTest(parameter=name):
                        torch.testing.assert_close(left.grad, right.grad, atol=3e-5, rtol=3e-4)

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

    def test_backward_capped_grid(self):
        for implementation, rows_per_block in (("grouped", 256), ("baseline", 8)):
            with self.subTest(implementation=implementation):
                rows = 65535 * rows_per_block + 1
                ids = torch.zeros(1, rows, device="cuda", dtype=torch.long)
                gradient = torch.ones(1, rows, 1, device="cuda")
                output = self.extension.embedding_backward(ids, gradient, 2, implementation)
                expected = torch.tensor([[rows], [0]], device="cuda", dtype=torch.float32)
                torch.testing.assert_close(output, expected, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    @torch.no_grad()
    def test_device_guard_and_device_mismatch(self):
        with torch.cuda.device(0):
            ids = torch.zeros(2, 3, device="cuda:1", dtype=torch.long)
            weight = torch.randn(7, 64, device="cuda:1")
            self.assert_gather(ids, weight)
            self.assertEqual(torch.cuda.current_device(), 0)
            with self.assertRaisesRegex(RuntimeError, "same CUDA device"):
                self.extension.embedding_forward(ids.to("cuda:0"), weight)
            gradient = torch.ones(2, 3, 64, device="cuda:1")
            output = self.extension.embedding_backward(ids, gradient, 7)
            expected = torch.zeros_like(weight)
            expected[0] = 6
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            self.assertEqual(torch.cuda.current_device(), 0)
            with self.assertRaisesRegex(RuntimeError, "same CUDA device"):
                self.extension.embedding_backward(ids.to("cuda:0"), gradient, 7)

    @torch.inference_mode()
    def test_model_prefill_and_cached_decode(self):
        with full_precision_matmul():
            model = Transformer(ModelConfig(vocab_size=31, dim=32, n_layers=1,
                                            n_heads=4, hidden_dim=48, max_seq_len=32)).cuda().eval()
            ids = torch.randint(31, (2, 11), device="cuda")
            chunks = [ids[:, :4].contiguous()]
            chunks.extend(ids[:, index:index + 1].contiguous()
                          for index in range(4, ids.size(1)))

            expected = model(ids)
            reference_cache = model.new_cache(2, 16)
            reference_parts = [model(chunk, cache=reference_cache) for chunk in chunks]

            # Check the reference's cache path independently of the student op.
            # Retain the original tolerance; do not hide errors by widening it.
            with self.subTest(stage="reference cached vs full"):
                torch.testing.assert_close(torch.cat(reference_parts, dim=1), expected,
                                           atol=2e-5, rtol=1e-4)

            model.ops = Operators({"embedding": "student"})
            # Pure gather must remain exact on full, prefill and decode inputs.
            for index, chunk in enumerate([ids] + chunks):
                with self.subTest(stage="embedding", input_index=index):
                    torch.testing.assert_close(student.embedding(chunk, model.embedding),
                                               reference.embedding(chunk, model.embedding),
                                               atol=0, rtol=0)

            with self.subTest(stage="student full vs reference full"):
                torch.testing.assert_close(model(ids), expected, atol=0, rtol=0)

            cache = model.new_cache(2, 16)
            parts = []
            for index, (chunk, reference_part) in enumerate(zip(chunks, reference_parts)):
                actual = model(chunk, cache=cache)
                parts.append(actual)
                with self.subTest(stage="student cached vs reference cached", chunk=index):
                    torch.testing.assert_close(actual, reference_part, atol=2e-5, rtol=1e-4)
            with self.subTest(stage="student cached vs reference full"):
                torch.testing.assert_close(torch.cat(parts, dim=1), expected, atol=2e-5, rtol=1e-4)

    def test_invalid_ids_fail_in_isolated_processes(self):
        # A device-side assertion poisons its CUDA context; never trigger it in
        # the main test process. Synchronization belongs in this test, not the op.
        for dim in (65, 64):
            for invalid in (-1, -100, 7):
                with self.subTest(dim=dim, invalid=invalid):
                    code = f"""
import torch
from tiny_transformer.operators import student
ids = torch.tensor([[{invalid}]], device='cuda')
weight = torch.randn(7, {dim}, device='cuda')
student.embedding(ids, weight)
torch.cuda.synchronize()
"""
                    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                            capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("device-side assert", result.stdout + result.stderr)

    def test_backward_invalid_ids_fail_in_isolated_processes(self):
        # Call backward directly: a failing forward would hide its missing check.
        cases = [(implementation, invalid) for implementation in ("grouped", "baseline")
                 for invalid in (-1, -100, 7)]
        for implementation, invalid in cases:
            with self.subTest(implementation=implementation, invalid=invalid):
                code = f"""
import torch
from tiny_transformer.operators._extension import load_embedding_extension
ids = torch.tensor([[0, {invalid}, 2]], device='cuda')
gradient = torch.ones(1, 3, 33, device='cuda')
load_embedding_extension().embedding_backward(ids, gradient, 7, {implementation!r})
torch.cuda.synchronize()
"""
                result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("device-side assert", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
