import unittest

import torch

from tiny_transformer.operators import Operators, reference, student
from tiny_transformer.check_ops import cases, differentiable_args


class OperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_rms_norm_matches_equation_and_gradcheck(self):
        x = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
        weight = torch.randn(5, dtype=torch.double, requires_grad=True)
        expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight
        torch.testing.assert_close(reference.rms_norm(x, weight, 1e-6), expected)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: reference.rms_norm(a, b, 1e-6), (x, weight)))

    def test_rope_preserves_pair_norm_and_position_zero(self):
        x = torch.randn(2, 3, 7, 8)
        angle = torch.randn(1, 1, 7, 4)
        rotated = reference.rope(x, angle.cos(), angle.sin())
        torch.testing.assert_close(rotated.square().reshape(2, 3, 7, 4, 2).sum(-1), x.square().reshape(2, 3, 7, 4, 2).sum(-1))
        torch.testing.assert_close(reference.rope(x, torch.ones_like(angle), torch.zeros_like(angle)), x)

    def test_rope_gradcheck(self):
        x = torch.randn(1, 2, 3, 4, dtype=torch.double, requires_grad=True)
        angle = torch.randn(1, 1, 3, 2, dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(lambda value: reference.rope(value, angle.cos(), angle.sin()), (x,)))

    def test_swiglu_gradcheck(self):
        values = [torch.randn(2, 3, dtype=torch.double, requires_grad=True) for _ in range(2)]
        self.assertTrue(torch.autograd.gradcheck(reference.swiglu, tuple(values)))

    def test_sdpa_forward_backward_prefill_and_decode(self):
        for decode in (False, True):
            inputs = cases("attention", torch.device("cpu"), torch.float32, decode)
            left, right = differentiable_args(inputs), differentiable_args(inputs)
            actual, expected = reference.sdpa_attention(*left), reference.attention(*right)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            grad = torch.randn_like(actual)
            a = torch.autograd.grad(actual, left[:3], grad)
            b = torch.autograd.grad(expected, right[:3], grad)
            for actual_grad, expected_grad in zip(a, b):
                torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=1e-4)

    def test_sdpa_chunked_cached_mask(self):
        q, k, v = torch.randn(1, 2, 3, 8), torch.randn(1, 2, 7, 8), torch.randn(1, 2, 7, 8)
        expected = reference.attention(q, k, v, past_len=4)
        torch.testing.assert_close(reference.sdpa_attention(q, k, v, past_len=4), expected)

    def test_sdpa_isolated_documents(self):
        q, k, v = [torch.randn(2, 2, 4, 8) for _ in range(3)]
        segments = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        torch.testing.assert_close(reference.sdpa_attention(q, k, v, segment_ids=segments), reference.attention(q, k, v, segment_ids=segments))

    def test_loss_ignores_masked_targets(self):
        logits = torch.randn(1, 3, 7)
        targets = torch.tensor([[2, -100, 4]])
        expected = torch.nn.functional.cross_entropy(logits[:, [0, 2], :].reshape(2, 7), torch.tensor([2, 4]))
        torch.testing.assert_close(reference.cross_entropy(logits, targets), expected)

    def test_student_does_not_silently_fallback(self):
        ops = Operators({"rms_norm": "student"})
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            ops.rms_norm(torch.ones(1, 4), torch.ones(4), 1e-6)
        with self.assertRaisesRegex(NotImplementedError, "linear"):
            Operators({"linear": "student"}).linear(torch.ones(1, 4), torch.ones(4, 4))
        self.assertIs(ops.linear, reference.linear)

    def test_invalid_dispatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "backward_impl"):
            student.embedding(torch.zeros(1, 1, dtype=torch.long), torch.ones(2, 4), backward_impl="typo")
        for options in ({"typo": "student"}, {"linear": "sdpa"}, {"rope": "unknown"}):
            with self.assertRaises(ValueError):
                Operators(options)


if __name__ == "__main__":
    unittest.main()
