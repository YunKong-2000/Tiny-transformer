"""RMSNorm Y/R and gradient correctness. Shared model tests live in test_student_integration.py."""
import unittest
from unittest.mock import Mock, patch

import torch

from tiny_transformer.operators import reference, student
from tiny_transformer.operators._extension import load_rms_norm_extension


def cpu_extension_double():
    """Mimic the native API for host autograd tests; does not execute CUDA."""
    def forward(x, weight, eps):
        assert not torch.is_grad_enabled()
        inv_rms = torch.rsqrt(x.square().mean(-1) + eps).reshape(-1)
        return reference.rms_norm(x, weight, eps), inv_rms

    def backward(x, gradient, weight, inv_rms):
        assert not torch.is_grad_enabled()
        assert gradient.is_contiguous()
        r = inv_rms.reshape(*x.shape[:-1], 1)
        u = gradient * weight
        dx = r * (u - x * r.square() * (u * x).mean(-1, keepdim=True))
        dw = (gradient * x * r).reshape(-1, x.shape[-1]).sum(0)
        return dx, dw

    return Mock(rms_norm_forward=Mock(side_effect=forward),
                rms_norm_backward=Mock(side_effect=backward))


class StudentRMSNormHostTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_saved_r_per_graph_strided_inputs_and_accumulation(self):
        x = torch.randn(2, 3, 10, requires_grad=True)
        weight = torch.randn(10, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_()
        expected_weight = weight.detach().clone().requires_grad_()
        extension = cpu_extension_double()
        with patch.object(student, 'load_rms_norm_extension', return_value=extension):
            # Match the public wrapper's copies; their backward must map to the base.
            actual = [student._RMSNorm.apply(x[..., ::2].contiguous(), weight[::2].contiguous(), eps)
                      for eps in (1e-6, 0.5)]
            expected = [reference.rms_norm(expected_x[..., ::2], expected_weight[::2], eps)
                        for eps in (1e-6, 0.5)]
            saved_r = [value.grad_fn.saved_tensors[2] for value in actual]
            self.assertNotEqual(saved_r[0].data_ptr(), saved_r[1].data_ptr())
            for index in (1, 0):
                if index:
                    # sum() produces an expanded upstream gradient.
                    actual[index].sum().backward()
                    expected[index].sum().backward()
                else:
                    upstream = torch.randn(2, 3, 10)[..., ::2] * 0.3
                    actual[index].backward(upstream)
                    expected[index].backward(upstream)
                self.assertIs(extension.rms_norm_backward.call_args.args[3], saved_r[index])
            torch.testing.assert_close(x.grad, expected_x.grad)
            torch.testing.assert_close(weight.grad, expected_weight.grad)
            self.assertEqual(extension.rms_norm_forward.call_count, 2)  # No recomputation.


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and nvcc")
class StudentRMSNormCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = load_rms_norm_extension()

    def setUp(self):
        torch.manual_seed(42)

    def assert_forward_backward(self, x, weight, eps=1e-6, upstream=None):
        x = x.detach().requires_grad_()
        weight = weight.detach().requires_grad_()
        expected_x = x.detach().clone().requires_grad_()
        expected_weight = weight.detach().clone().requires_grad_()
        actual = student.rms_norm(x, weight, eps)
        # Inspect the cache from this forward instead of launching forward twice.
        r = actual.grad_fn.saved_tensors[2]
        torch.testing.assert_close(r, torch.rsqrt(x.square().mean(-1) + eps).reshape(-1),
                                   atol=1e-5, rtol=1e-4)
        expected = reference.rms_norm(expected_x, expected_weight, eps)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        if upstream is None:
            upstream = torch.randn_like(actual) / max(1, x.numel() // x.shape[-1])
        gradients = torch.autograd.grad(actual, (x, weight), upstream)
        torch.testing.assert_close(x, expected_x, atol=0, rtol=0)
        torch.testing.assert_close(weight, expected_weight, atol=0, rtol=0)
        expected_gradients = torch.autograd.grad(expected, (expected_x, expected_weight), upstream)
        for left, right in zip(gradients, expected_gradients):
            torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-4)
        self.assertEqual(gradients[0].shape, x.shape)
        self.assertEqual(gradients[1].shape, weight.shape)

    def test_shapes_scales_and_shared_memory_boundary(self):
        for shape in [(2, 3, h) for h in (1, 4, 31, 32, 33, 64, 65, 124, 128, 132, 768, 1024)] + [
                (65,), (3, 65), (2, 3, 4, 65), (8, 512, 768), (4, 1, 768), (0, 3, 65), (2, 0, 65)]:
            with self.subTest(shape=shape):
                self.assert_forward_backward(torch.randn(shape, device='cuda'),
                                     torch.randn(shape[-1], device='cuda'))
        for dim in (65, 768):
            for scale, eps in ((0, 1e-6), (1e-4, 0.1), (1e3, 1e-4)):
                with self.subTest(dim=dim, scale=scale, eps=eps):
                    self.assert_forward_backward(torch.randn(2, 3, dim, device='cuda') * scale,
                                         torch.randn(dim, device='cuda'), eps)

    def test_strides_alignment_and_expanded_upstream(self):
        base = torch.randn(2, 5, 130, device='cuda')
        for x in (base[..., ::2], base[..., :65][:, -1:, :], base[..., :65].transpose(0, 1)):
            gradient = torch.ones((), device='cuda').expand(x.shape)
            self.assert_forward_backward(x, torch.randn(130, device='cuda')[::2], upstream=gradient)
        for x_offset, w_offset in ((0, 0), (1, 0), (0, 1), (1, 1), (4, 4)):
            x = torch.randn(6 * 768 + x_offset, device='cuda')[x_offset:].view(2, 3, 768)
            w = torch.randn(768 + w_offset, device='cuda')[w_offset:]
            gradient = torch.randn(2, 3, 1536, device='cuda')[..., ::2]
            self.assert_forward_backward(x, w, upstream=gradient)

    @torch.no_grad()
    def test_capped_grid_and_zeroing_on_each_call(self):
        rows = 65535 + 1
        x = torch.ones(rows, 1, device='cuda')
        w = torch.ones(1, device='cuda')
        _, r = self.extension.rms_norm_forward(x, w, 0.0)
        for value in (1, 2):
            dx, dw = self.extension.rms_norm_backward(x, torch.full_like(x, value), w, r)
            torch.testing.assert_close(dx, torch.zeros_like(x), atol=0, rtol=0)
            torch.testing.assert_close(dw, torch.full_like(w, rows * value), atol=0, rtol=0)

    def test_native_rejects_unsupported_inputs(self):
        x, weight = torch.randn(2, 3, 65, device='cuda'), torch.randn(65, device='cuda')
        cases = [(x.cpu(), weight, 'CUDA tensors'), (x, weight.cpu(), 'CUDA tensors'),
                 (x.half(), weight, 'float32'), (x, weight.bfloat16(), 'float32'),
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

    @torch.no_grad()
    def test_backward_rejects_unsupported_inputs(self):
        x = torch.randn(2, 3, 65, device='cuda')
        g, w = torch.randn_like(x), torch.randn(65, device='cuda')
        _, r = self.extension.rms_norm_forward(x, w, 1e-6)
        args = [x, g, w, r]
        cases = [(i, tensor.cpu(), 'CUDA tensors') for i, tensor in enumerate(args)]
        cases += [(i, tensor.half(), 'float32') for i, tensor in enumerate(args)]
        cases += [(1, g[:, :1], 'same shape'), (3, r[:-1], 'shape'), (3, r.view(2, 3), 'shape'),
                  (3, torch.ones(12, device='cuda')[::2], 'contiguous'),
                  (2, w[None, :], '1D'), (2, w[:-1], 'same last dimension'),
                  (1, torch.randn(2, 3, 130, device='cuda')[..., ::2], 'contiguous')]
        for index, value, message in cases:
            with self.subTest(index=index, message=message):
                call_args = args.copy()
                call_args[index] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    self.extension.rms_norm_backward(*call_args)
        for h in (0, 1025):
            with self.assertRaisesRegex(RuntimeError, 'H <= 1024'):
                self.extension.rms_norm_backward(torch.empty(2, h, device='cuda'),
                                                torch.empty(2, h, device='cuda'),
                                                torch.empty(h, device='cuda'),
                                                torch.empty(2, device='cuda'))

    def test_training_width_limit_and_native_double_backward_guard(self):
        x = torch.randn(2, 1025, device='cuda', requires_grad=True)
        w = torch.ones(1025, device='cuda', requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, 'H <= 1024'):
            student.rms_norm(x, w, 1e-6)
        with torch.no_grad():
            torch.testing.assert_close(student.rms_norm(x, w, 1e-6), reference.rms_norm(x, w, 1e-6))
        x, w = x[:, :65].contiguous().detach(), w[:65].detach()
        with torch.no_grad():
            _, r = self.extension.rms_norm_forward(x, w, 1e-6)
        with self.assertRaisesRegex(RuntimeError, 'first-order'):
            self.extension.rms_norm_backward(x, torch.ones_like(x, requires_grad=True), w, r)

    @torch.no_grad()
    def test_backward_reads_cached_r_on_current_stream(self):
        x, g = torch.zeros(2, 3, 65, device='cuda'), torch.zeros(2, 3, 65, device='cuda')
        w, r = torch.zeros(65, device='cuda'), torch.zeros(6, device='cuda')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            x.fill_(2)
            g.fill_(3)
            w.fill_(4)
            r.fill_(0.25)  # Deliberately supplied cache; backward must consume it.
            dx, dw = self.extension.rms_norm_backward(x, g, w, r)
            dx, dw = dx.clone(), dw.clone()
        stream.synchronize()
        torch.testing.assert_close(dx, torch.full_like(x, 2.25), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(dw, torch.full_like(w, 9), atol=1e-6, rtol=1e-5)

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

    @torch.no_grad()
    def test_forward_only_extremes_and_capped_grid(self):
        for rows, dim, value, scale, eps in ((6, 65, 0., 1., 0.),
                                           (6, 65, 1e10, 1e30, 1e-6),
                                           (65535 * 8 + 1, 4, 1., 2., 1e-6)):
            x, w = torch.full((rows, dim), value, device="cuda"), torch.full((dim,), scale, device="cuda")
            y, r = self.extension.rms_norm_forward(x, w, eps)
            torch.testing.assert_close(y, reference.rms_norm(x, w, eps), atol=1e-5, rtol=1e-4, equal_nan=True)
            torch.testing.assert_close(r, torch.rsqrt(x.square().mean(-1) + eps), atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
