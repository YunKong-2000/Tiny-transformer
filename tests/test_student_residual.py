"""Residual CUDA values, promotion, layouts and gradients; shared model tests are separate."""
from itertools import product
import unittest
from unittest.mock import Mock, patch

import torch

from tiny_transformer.operators import reference, student
from tiny_transformer.operators._extension import load_residual_extension


class StudentResidualHostTests(unittest.TestCase):
    def test_autograd_bridge_casts_views_aliases_and_accumulation(self):
        # A CPU double checks graph wiring only, never CUDA kernel correctness.
        def forward(x, update):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(x.is_contiguous() and update.is_contiguous())
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(update.dtype, torch.float32)
            return x + update

        def backward(gradient):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(gradient.is_contiguous())
            return gradient.clone(), gradient.clone()

        extension = Mock(residual_forward=Mock(side_effect=forward),
                         residual_backward=Mock(side_effect=backward))
        with patch.object(student, 'load_residual_extension', return_value=extension):
            for dtype, needs_x, needs_update in product(
                    (torch.float32, torch.float16, torch.bfloat16), (False, True), (False, True)):
                if not (needs_x or needs_update):
                    continue
                with self.subTest(dtype=dtype, needs_x=needs_x, needs_update=needs_update):
                    x = torch.randn(2, 3, 10, requires_grad=needs_x)
                    update = torch.randn(2, 3, 10, dtype=dtype, requires_grad=needs_update)
                    rx = x.detach().clone().requires_grad_(needs_x)
                    ru = update.detach().clone().requires_grad_(needs_update)
                    for use_sum in (False, True):
                        # Match the public wrapper's casts/copies outside Function.apply.
                        actual = student._Residual.apply(x[..., ::2].float().contiguous(),
                                                         update[..., ::2].float().contiguous())
                        expected = reference.residual(rx[..., ::2], ru[..., ::2])
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        upstream = torch.randn(2, 3, 10)[..., ::2]
                        if use_sum:
                            actual.sum().backward()
                            expected.sum().backward()
                        else:
                            actual.backward(upstream)
                            expected.backward(upstream)
                    for value, target in ((x, rx), (update, ru)):
                        if value.requires_grad:
                            torch.testing.assert_close(value.grad, target.grad, atol=0, rtol=0)
                        else:
                            self.assertIsNone(value.grad)
            x = torch.randn(7, requires_grad=True)
            student._Residual.apply(x, x).sum().backward()
            torch.testing.assert_close(x.grad, torch.full_like(x, 2), atol=0, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA and nvcc')
class StudentResidualCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = load_residual_extension()

    def setUp(self):
        torch.manual_seed(42)

    def assert_forward_backward(self, x, update, upstream=None):
        x, update = x.detach().requires_grad_(), update.detach().requires_grad_()
        rx = x.detach().clone().requires_grad_()
        ru = update.detach().clone().requires_grad_()
        actual, expected = student.residual(x, update), reference.residual(rx, ru)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(x, rx, atol=0, rtol=0)
        torch.testing.assert_close(update, ru, atol=0, rtol=0)
        if actual.numel():
            self.assertNotEqual(actual.data_ptr(), x.data_ptr())
            self.assertNotEqual(actual.data_ptr(), update.data_ptr())
        if upstream is None:
            upstream = torch.randn_like(expected)
        actual_grads = torch.autograd.grad(actual, (x, update), upstream)
        expected_grads = torch.autograd.grad(expected, (rx, ru), upstream)
        for value, target in zip(actual_grads, expected_grads):
            torch.testing.assert_close(value, target, atol=0, rtol=0)

    def test_shapes_tails_and_empty_inputs(self):
        shapes = ((), (0,), (2, 0, 4), (2, 3, 0), (1,), (3,), (4,), (7,),
                  (255,), (256,), (257,), (1023,), (1024,), (1025,), (2, 3, 65))
        for shape in shapes:
            with self.subTest(shape=shape):
                self.assert_forward_backward(torch.randn(shape, device='cuda'),
                                             torch.randn(shape, device='cuda'))

    def test_contiguous_storage_offsets_for_each_pointer(self):
        for offsets in ((0, 0, 0), (4, 8, 4), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
            with self.subTest(offsets=offsets):
                x, u, g = [torch.randn(1028 + offset, device='cuda')[offset:]
                           for offset in offsets]
                self.assert_forward_backward(x, u, g)

    def test_strided_inputs_expanded_gradients_and_shared_input(self):
        for view in (lambda t: t[..., ::2], lambda t: t.transpose(0, 1),
                     lambda t: t[:1].expand(2, -1, -1)):
            with self.subTest(view=view):
                x, u = [torch.randn(2, 3, 10, device='cuda', requires_grad=True) for _ in range(2)]
                rx, ru = [v.detach().clone().requires_grad_() for v in (x, u)]
                actual = student.residual(view(x), view(u))
                expected = reference.residual(view(rx), view(ru))
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                # sum() supplies a zero-stride upstream; accumulate with another branch.
                (actual.sum() + x.square().sum()).backward()
                (expected.sum() + rx.square().sum()).backward()
                torch.testing.assert_close(x.grad, rx.grad, atol=0, rtol=0)
                torch.testing.assert_close(u.grad, ru.grad, atol=0, rtol=0)
        x = torch.randn(2, 3, 4, device='cuda', requires_grad=True)
        student.residual(x, x).sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2), atol=0, rtol=0)
        for needs_x, needs_u in ((True, False), (False, True)):
            x = torch.randn(7, device='cuda', requires_grad=needs_x)
            u = torch.randn(7, device='cuda', requires_grad=needs_u)
            student.residual(x, u).sum().backward()
            for value in (x, u):
                if value.requires_grad:
                    torch.testing.assert_close(value.grad, torch.ones_like(value), atol=0, rtol=0)
                else:
                    self.assertIsNone(value.grad)

    def test_dtype_promotion_and_amp_preserve_fp32_residual(self):
        for xd, ud in product((torch.float32, torch.float16, torch.bfloat16), repeat=2):
            with self.subTest(x_dtype=xd, update_dtype=ud):
                x = torch.randn(2, 3, 7, device='cuda', dtype=xd)
                u = torch.randn(2, 3, 7, device='cuda', dtype=ud)
                self.assert_forward_backward(x, u)
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            # Rounding x to FP16/BF16 before the add would discard this increment.
            x = torch.full((2, 3, 7), 1.0001, device='cuda')
            u = torch.full_like(x, -1, dtype=dtype)
            with torch.autocast('cuda', dtype=dtype):
                self.assert_forward_backward(x, u)
                self.assert_forward_backward(u, x)

    @torch.no_grad()
    def test_raw_backward_outputs_are_independent(self):
        g = torch.randn(1024, device='cuda')
        before = g.clone()
        dx, du = self.extension.residual_backward(g)
        torch.testing.assert_close(dx, before, atol=0, rtol=0)
        torch.testing.assert_close(du, before, atol=0, rtol=0)
        dx.zero_()
        torch.testing.assert_close(g, before, atol=0, rtol=0)
        torch.testing.assert_close(du, before, atol=0, rtol=0)

    def test_native_and_public_input_validation(self):
        x = torch.ones(2, 4, device='cuda')
        noncontiguous = torch.ones(2, 8, device='cuda')[:, ::2]
        for bad, message in ((x.cpu(), 'CUDA'), (x.double(), 'float32'),
                             (x.half(), 'float32'), (x.reshape(4, 2), 'same shape'),
                             (noncontiguous, 'contiguous')):
            for args in ((x, bad), (bad, x)):
                with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                    self.extension.residual_forward(*args)
        for bad, message in ((x.cpu(), 'CUDA'), (x.double(), 'float32'),
                             (noncontiguous, 'contiguous')):
            with self.assertRaisesRegex(RuntimeError, message):
                self.extension.residual_backward(bad)
        for bad, message in ((x.double(), 'supports only'), (x.long(), 'supports only'),
                             (x[:1], 'same shape'), (x.reshape(4, 2), 'same shape'),
                             (x.to_sparse(), 'strided layout')):
            for args in ((x, bad), (bad, x)):
                with self.assertRaisesRegex(RuntimeError, message):
                    student.residual(*args)
        with self.assertRaisesRegex(RuntimeError, 'strided layout'):
            self.extension.residual_forward(x.to_sparse(), x)
        with self.assertRaisesRegex(RuntimeError, 'strided layout'):
            self.extension.residual_backward(x.to_sparse())
        with self.assertRaisesRegex(RuntimeError, 'first-order'):
            self.extension.residual_backward(x.requires_grad_())

    @torch.no_grad()
    def test_current_stream_forward_and_backward(self):
        for n in (1024, 1025):
            x, u, g = [torch.zeros(n, device='cuda') for _ in range(3)]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda._sleep(10_000_000)
                x.fill_(2)
                u.fill_(3)
                g.fill_(7)
                y = self.extension.residual_forward(x, u).clone()
                dx, du = [v.clone() for v in self.extension.residual_backward(g)]
            stream.synchronize()
            torch.testing.assert_close(y, torch.full_like(y, 5), atol=0, rtol=0)
            for value in (dx, du):
                torch.testing.assert_close(value, torch.full_like(value, 7), atol=0, rtol=0)

    @torch.no_grad()
    def test_capped_grid_scalar_and_vector_loops(self):
        # Force a second grid-stride iteration in each implementation (~1.4 GB peak).
        for n in (65535 * 256 + 1, 65535 * 256 * 4 + 4):
            with self.subTest(n=n):
                x, u = torch.ones(n, device='cuda'), torch.full((n,), 2., device='cuda')
                y = self.extension.residual_forward(x, u)
                self.assertTrue(torch.equal(y, torch.full_like(y, 3)))
                del x, u
                dx, du = self.extension.residual_backward(y)
                self.assertTrue(torch.equal(dx, y))
                self.assertTrue(torch.equal(du, y))
                del y, dx, du


if __name__ == '__main__':
    unittest.main()
