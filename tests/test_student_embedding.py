"""Student embedding integration tests; CUDA cases JIT-build the real extension."""
from contextlib import contextmanager
from pathlib import Path
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

    def test_grad_guard_and_amp_fp32_output(self):
        ids = torch.zeros(2, 3, device="cuda", dtype=torch.long)
        weight = torch.randn(7, 64, device="cuda", requires_grad=True)
        for function in (student.embedding, self.extension.embedding_forward):
            with self.assertRaisesRegex(RuntimeError, "forward-only"):
                function(ids, weight)
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                actual = student.embedding(ids, weight)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertFalse(actual.requires_grad)
                torch.testing.assert_close(actual, reference.embedding(ids, weight), atol=0, rtol=0)

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


if __name__ == "__main__":
    unittest.main()
