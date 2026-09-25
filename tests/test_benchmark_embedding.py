"""Input distribution and measurement-boundary checks for embedding benchmarks."""
import unittest
from unittest.mock import Mock

import torch

from tiny_transformer.benchmark_embedding import PATTERNS, make_ids, measure_pair, prepare_calls
from tiny_transformer.operators import reference


class EmbeddingBenchmarkHostTests(unittest.TestCase):
    def test_distribution_contracts(self):
        torch.manual_seed(42)
        for pattern in PATTERNS:
            with self.subTest(pattern=pattern):
                ids = make_ids(pattern, 2, 17, 67, 4, "cpu")
                self.assertEqual(ids.shape, (2, 17))
                self.assertEqual(ids.dtype, torch.int64)
                self.assertTrue(ids.is_contiguous())
                self.assertGreaterEqual(ids.min().item(), 0)
                self.assertLess(ids.max().item(), 67)
                if pattern == "same":
                    self.assertEqual(ids.unique().numel(), 1)
                elif pattern == "unique":
                    self.assertEqual(ids.unique().numel(), ids.numel())
                elif pattern == "hot":
                    self.assertLess(ids.max().item(), 4)

    def test_unique_does_not_silently_wrap_and_hot_ids_stay_in_vocab(self):
        with self.assertRaisesRegex(ValueError, "unique IDs require"):
            make_ids("unique", 2, 17, 7, 16, "cpu")
        ids = make_ids("hot", 2, 17, 7, 16, "cpu")
        self.assertLess(ids.max().item(), 7)

    def test_backward_reuses_graph_without_forward_or_grad_accumulation(self):
        ids = torch.tensor([[2, 0, 2], [1, 2, 0]])
        weight = torch.randn(7, 5, requires_grad=True)
        upstream = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)
        candidate = Mock(side_effect=reference.embedding)
        calls, errors = prepare_calls(ids, weight, upstream, candidate=candidate)
        self.assertEqual(errors["backward_max_abs_error"], 0)
        candidate.reset_mock()
        expected = torch.zeros_like(weight).index_add_(0, ids.flatten(), upstream.reshape(-1, 5))
        for _ in range(3):
            for function in calls["backward"]:
                torch.testing.assert_close(function(), expected)
        candidate.assert_not_called()
        self.assertIsNone(weight.grad)
        for function in calls["forward"]:
            self.assertFalse(function().requires_grad)

    def test_incorrect_candidate_fails_before_timing(self):
        ids = torch.tensor([[0, 1]])
        weight = torch.randn(7, 5, requires_grad=True)
        with self.assertRaises(AssertionError):
            prepare_calls(ids, weight, torch.ones(1, 2, 5),
                          candidate=lambda i, w: reference.embedding(i, w) + 1)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and nvcc")
class EmbeddingBenchmarkCudaTests(unittest.TestCase):
    def test_all_patterns_forward_backward_measurements(self):
        torch.manual_seed(42)
        with torch.cuda.device(0):
            weight = torch.randn(67, 33, device="cuda", requires_grad=True)
            upstream = torch.randn(2, 17, 33, device="cuda") / 34
            for pattern in PATTERNS:
                ids = make_ids(pattern, 2, 17, 67, 4, "cuda")
                calls, _ = prepare_calls(ids, weight, upstream)
                for phase, functions in calls.items():
                    with self.subTest(pattern=pattern, phase=phase):
                        result = measure_pair(functions, warmup=2, repeats=2, trials=3)
                        self.assertEqual(len(result["reference_trials_us"]), 3)
                        self.assertEqual(len(result["student_trials_us"]), 3)
                        self.assertGreater(result["reference_us"], 0)
                        self.assertGreater(result["student_us"], 0)
                        self.assertGreater(result["speedup"], 0)
            self.assertIsNone(weight.grad)


if __name__ == "__main__":
    unittest.main()
