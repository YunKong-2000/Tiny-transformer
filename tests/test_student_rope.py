"""RoPE CuTe partitions, strided storage, constant coefficients and autograd."""
import unittest
from unittest.mock import Mock, patch

import torch

from tiny_transformer.operators import reference, student
from tiny_transformer.operators._extension import load_rope_extension


class StudentRopeHostTests(unittest.TestCase):
    def test_autograd_bridge_preserves_views_and_saves_coefficients(self):
        # CPU double checks Python wiring only, not the CUDA implementation.
        def forward(x, cos, sin):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(x.is_contiguous())
            self.assertGreater(x.storage_offset(), 0)
            return reference.rope(x, cos, sin)

        seen_strides = []

        def backward(gradient, cos, sin):
            self.assertFalse(torch.is_grad_enabled())
            seen_strides.append(gradient.stride())
            return reference.rope(gradient, cos, -sin)

        extension = Mock(rope_forward=Mock(side_effect=forward),
                         rope_backward=Mock(side_effect=backward))
        with patch.object(student, 'load_rope_extension', return_value=extension):
            base = torch.randn(2, 5, 3, 5, 16, requires_grad=True)
            expected_base = base.detach().clone().requires_grad_()
            for coefficient_batch in (1, 2):
                angle = torch.randn(coefficient_batch, 1, 5, 8)
                cos, sin = angle.cos(), angle.sin()
                x = base[:, :, 1].transpose(1, 2)
                rx = expected_base[:, :, 1].transpose(1, 2)
                actual = student._Rope.apply(x, cos, sin)
                expected = reference.rope(rx, cos, sin)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(extension.rope_forward.call_args.args[0].data_ptr(), x.data_ptr())
                if coefficient_batch == 1:
                    upstream = torch.randn(2, 5, 5, 32)[..., ::2]
                    actual.backward(upstream)
                    expected.backward(upstream)
                else:
                    actual.sum().backward()
                    expected.sum().backward()
                self.assertIs(extension.rope_backward.call_args.args[1], cos)
                self.assertIs(extension.rope_backward.call_args.args[2], sin)
            torch.testing.assert_close(base.grad, expected_base.grad)
            self.assertEqual(seen_strides[0][-1], 2)
            self.assertEqual(seen_strides[1], (0, 0, 0, 0))


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA and nvcc')
class StudentRopeCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = load_rope_extension()

    def setUp(self):
        torch.manual_seed(42)

    def coefficients(self, batch, time, dim):
        # Include position zero, distinct tokens/batches and large positions.
        positions = torch.arange(batch * time, device='cuda').reshape(batch, time) * 997
        frequency = 1 / 10000 ** (torch.arange(0, dim, 2, device='cuda').float() / dim)
        angles = (positions[..., None] * frequency)[:, None]
        return angles.cos(), angles.sin()

    def check_values_and_gradients(self, x, cos, sin, use_sum=False):
        x = x.detach().requires_grad_()  # Retain the original stride and offset.
        rx = x.detach().clone().requires_grad_()
        original = x.detach().clone()
        before_c, before_s = cos.clone(), sin.clone()
        actual = student.rope(x, cos, sin)
        expected = reference.rope(rx, cos, sin)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(actual.dtype, x.dtype)
        if actual.numel():
            self.assertNotEqual(actual.data_ptr(), x.data_ptr())
        if use_sum:
            actual.sum().backward()
            expected.sum().backward()
        else:
            upstream = torch.randn(*x.shape[:-1], x.shape[-1] * 2, device='cuda')[..., ::2]
            actual.backward(upstream)
            expected.backward(upstream)
        torch.testing.assert_close(x.grad, rx.grad, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(x, original, atol=0, rtol=0)
        torch.testing.assert_close(cos, before_c, atol=0, rtol=0)
        torch.testing.assert_close(sin, before_s, atol=0, rtol=0)
        # Grad-enabled inputs still work in both inference contexts.
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                value = student.rope(x, cos, sin)
                self.assertFalse(value.requires_grad)
                torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-6)

    def test_real_qkv_views_head_pair_tails_and_offsets(self):
        for batch, heads, time, dim in (
                (2, 12, 5, 64), (3, 5, 2, 80), (2, 4, 1, 16),
                (1, 1, 3, 2), (2, 3, 7, 32), (2, 8, 3, 128), (2, 5, 3, 130)):
            with self.subTest(shape=(batch, heads, time, dim)):
                # Nonzero base offset also catches double-counting storage_offset().
                count = batch * time * 3 * heads * dim
                storage = torch.randn(count + 7, device='cuda')
                qkv = storage[7:].view(batch, time, 3, heads, dim)
                before = storage.clone()
                q, k, _ = [v.transpose(1, 2) for v in qkv.unbind(2)]
                self.assertEqual(q.stride(), (3*time*heads*dim, dim, 3*heads*dim, 1))
                self.assertEqual(k.storage_offset() - q.storage_offset(), heads*dim)
                for x in (q, k):
                    for cb in (1, batch):
                        self.check_values_and_gradients(x, *self.coefficients(cb, time, dim),
                                                       use_sum=(cb == batch))
                torch.testing.assert_close(storage, before, atol=0, rtol=0)

    def test_general_strides_and_independent_coefficient_layouts(self):
        shape = (2, 5, 3, 80)
        base = torch.randn(2, 3, 5, 160, device='cuda')
        for x in (base[..., ::2].transpose(1, 2), torch.randn(shape, device='cuda'),
                  torch.randn(1, 1, 3, 80, device='cuda').expand(shape)):
            cos, sin = self.coefficients(2, 3, 80)
            cstore = torch.empty(2, 1, 3, 81, device='cuda')
            cview = cstore[..., 1::2]
            cview.copy_(cos)
            # sin uses a different noncontiguous layout from cos.
            sstore = torch.empty(2, 1, 40, 3, device='cuda')
            sview = sstore.transpose(2, 3)
            sview.copy_(sin)
            self.check_values_and_gradients(x, cview, sview)
        # Broadcast across batch/token/pair via actual zero strides as well.
        cos = torch.ones(1, 1, 1, 1, device='cuda').expand(2, 1, 3, 40)
        sin = torch.zeros_like(cos)
        self.check_values_and_gradients(torch.randn(shape, device='cuda'), cos, sin, True)

    @torch.no_grad()
    def test_adjacent_pair_signs_identity_and_norm(self):
        x = torch.arange(2*5*3*8, device='cuda', dtype=torch.float32).reshape(2, 5, 3, 8)
        zeros = torch.zeros(1, 1, 3, 4, device='cuda')
        ones = torch.ones_like(zeros)
        rotated = student.rope(x, zeros, ones)
        torch.testing.assert_close(rotated[..., ::2], -x[..., 1::2], atol=0, rtol=0)
        torch.testing.assert_close(rotated[..., 1::2], x[..., ::2], atol=0, rtol=0)
        torch.testing.assert_close(student.rope(x, ones, zeros), x, atol=0, rtol=0)
        cos, sin = self.coefficients(2, 3, 8)
        y = student.rope(x, cos, sin)
        torch.testing.assert_close(y.reshape(2, 5, 3, 4, 2).square().sum(-1),
                                   x.reshape(2, 5, 3, 4, 2).square().sum(-1),
                                   atol=1e-4, rtol=5e-6)

    def test_empty_inputs_and_capped_grid(self):
        for shape in ((0, 5, 3, 16), (2, 0, 3, 16), (2, 5, 0, 16)):
            self.check_values_and_gradients(torch.empty(shape, device='cuda'),
                                           *self.coefficients(1, shape[2], shape[3]), use_sum=True)
        # 65538 CTA tiles force the capped grid to take a second iteration.
        x = torch.randn(2, 1, 32769, 2, device='cuda')
        self.check_values_and_gradients(x, *self.coefficients(1, 32769, 2), use_sum=True)

    def test_native_and_public_validation_and_grad_guards(self):
        x = torch.randn(2, 5, 3, 16, device='cuda')
        cos, sin = self.coefficients(1, 3, 16)
        cases = [
            ((x.cpu(), cos, sin), 'CUDA'),
            ((x.double(), cos, sin), 'float32'),
            ((x.half(), cos, sin), 'float32'),
            ((x, cos.bfloat16(), sin), 'float32'),
            ((x.to_sparse(), cos, sin), 'strided layout'),
            ((x[0], cos, sin), '4D'),
            ((x[..., :15], cos, sin), 'positive and even'),
            ((x[..., :0], cos[..., :0], sin[..., :0]), 'positive and even'),
            ((x, cos, sin[..., :7]), 'same shape'),
            ((x, cos.expand(3, -1, -1, -1), sin.expand(3, -1, -1, -1)), 'shape'),
            ((x, cos.expand(-1, 5, -1, -1), sin.expand(-1, 5, -1, -1)), 'shape'),
            ((x, cos[:, :, :1], sin[:, :, :1]), 'shape'),
            ((x, cos[..., :7], sin[..., :7]), 'shape'),
            ((x, cos.detach().requires_grad_(), sin), 'constant cos/sin'),
            ((x, cos, sin.detach().requires_grad_()), 'constant cos/sin'),
        ]
        for args, message in cases:
            for call in (student.rope, self.extension.rope_forward, self.extension.rope_backward):
                with self.subTest(message=message, call=call), self.assertRaisesRegex(RuntimeError, message):
                    call(*args)
        with self.assertRaisesRegex(RuntimeError, 'autograd binding'):
            self.extension.rope_forward(x.requires_grad_(), cos, sin)
        with self.assertRaisesRegex(RuntimeError, 'first-order'):
            self.extension.rope_backward(x, cos, sin)
        # Coefficient gradients are explicitly unsupported even during inference.
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, 'constant cos/sin'):
            student.rope(x, cos.requires_grad_(), sin)

    @torch.no_grad()
    def test_current_stream_forward_and_backward(self):
        x = torch.zeros(2, 5, 3, 80, device='cuda')
        cos = torch.zeros(1, 1, 3, 40, device='cuda')
        sin = torch.zeros_like(cos)
        gradient = torch.zeros_like(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            x.fill_(2)
            cos.fill_(0.75)
            sin.fill_(0.25)
            gradient.fill_(3)
            y = self.extension.rope_forward(x, cos, sin).clone()
            dx = self.extension.rope_backward(gradient, cos, sin).clone()
        stream.synchronize()
        torch.testing.assert_close(y, reference.rope(x, cos, sin), atol=0, rtol=0)
        torch.testing.assert_close(dx, reference.rope(gradient, cos, -sin), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
