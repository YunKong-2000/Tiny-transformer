"""SwiGLU CuTe partitions, independent strides, input gradients and autograd."""
import unittest
from unittest.mock import Mock, patch

import torch

from tiny_transformer.operators import reference, student
from tiny_transformer.operators._extension import load_swiglu_extension


class StudentSwiGLUHostTests(unittest.TestCase):
    def extension_double(self):
        # Only checks Python graph wiring on CPU; this is not a kernel test.
        def forward(gate, up):
            self.assertFalse(torch.is_grad_enabled())
            return reference.swiglu(gate, up)

        def backward(gradient, gate, up):
            self.assertFalse(torch.is_grad_enabled())
            with torch.enable_grad():
                g = gate.detach().requires_grad_()
                u = up.detach().requires_grad_()
                return torch.autograd.grad(reference.swiglu(g, u), (g, u), gradient)

        return Mock(swiglu_forward=Mock(side_effect=forward),
                    swiglu_backward=Mock(side_effect=backward))

    def test_chunk_views_saved_without_copies_and_gradient_accumulation(self):
        extension = self.extension_double()
        with patch.object(student, 'load_swiglu_extension', return_value=extension):
            base = torch.randn(2, 5, 130, requires_grad=True)
            expected_base = base.detach().clone().requires_grad_()
            for use_sum in (False, True):
                gate, up = base.chunk(2, -1)
                rg, ru = expected_base.chunk(2, -1)
                actual = student._SwiGLU.apply(gate, up)
                expected = reference.swiglu(rg, ru)
                torch.testing.assert_close(actual, expected)
                args = extension.swiglu_forward.call_args.args
                self.assertIs(args[0], gate)
                self.assertIs(args[1], up)
                self.assertEqual(args[1].storage_offset(), 65)
                self.assertEqual(args[0].stride(), (650, 130, 1))
                upstream = torch.randn(2, 5, 130)[..., ::2]
                if use_sum:
                    actual.sum().backward()
                    expected.sum().backward()
                else:
                    actual.backward(upstream)
                    expected.backward(upstream)
                grad, saved_gate, saved_up = extension.swiglu_backward.call_args.args
                self.assertIs(saved_gate, gate)
                self.assertIs(saved_up, up)
                self.assertEqual(grad.stride(), (0, 0, 0) if use_sum else upstream.stride())
            torch.testing.assert_close(base.grad, expected_base.grad)

    def test_one_trainable_input_and_aliases(self):
        with patch.object(student, 'load_swiglu_extension', return_value=self.extension_double()):
            for needs_gate, needs_up in ((True, False), (False, True), (True, True)):
                gate = torch.randn(2, 3, 7, requires_grad=needs_gate)
                up = torch.randn(2, 3, 7, requires_grad=needs_up)
                rg = gate.detach().clone().requires_grad_(needs_gate)
                ru = up.detach().clone().requires_grad_(needs_up)
                student._SwiGLU.apply(gate, up).sum().backward()
                reference.swiglu(rg, ru).sum().backward()
                for value, expected in ((gate, rg), (up, ru)):
                    if value.requires_grad:
                        torch.testing.assert_close(value.grad, expected.grad)
                    else:
                        self.assertIsNone(value.grad)
            gate = torch.randn(2, 3, 7, requires_grad=True)
            rg = gate.detach().clone().requires_grad_()
            student._SwiGLU.apply(gate, gate).sum().backward()
            reference.swiglu(rg, rg).sum().backward()
            torch.testing.assert_close(gate.grad, rg.grad)

    def test_higher_order_is_not_silently_supported(self):
        with patch.object(student, 'load_swiglu_extension', return_value=self.extension_double()):
            gate = torch.randn(1, 1, 7, requires_grad=True)
            up = torch.randn_like(gate, requires_grad=True)
            output = student._SwiGLU.apply(gate, up)
            dg, = torch.autograd.grad(output.sum(), gate, create_graph=True)
            with self.assertRaises(RuntimeError):
                torch.autograd.grad(dg.sum(), gate)


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA and nvcc')
class StudentSwiGLUCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = load_swiglu_extension()

    def setUp(self):
        torch.manual_seed(42)

    def check_values_and_gradients(self, gate, up, use_sum=False, needs=(True, True)):
        gate, up = gate.detach().requires_grad_(needs[0]), up.detach().requires_grad_(needs[1])
        rg = gate.detach().clone().requires_grad_(needs[0])
        ru = up.detach().clone().requires_grad_(needs[1])
        before = (gate.detach().clone(), up.detach().clone())
        actual, expected = student.swiglu(gate, up), reference.swiglu(rg, ru)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(actual.dtype, gate.dtype)
        self.assertEqual(actual.device, gate.device)
        if actual.numel():
            self.assertNotIn(actual.data_ptr(), (gate.data_ptr(), up.data_ptr()))
        if use_sum:
            actual.sum().backward()
            expected.sum().backward()
        else:
            upstream = torch.randn(*gate.shape[:-1], gate.shape[-1] * 2 + 1, device='cuda')[..., 1::2]
            actual.backward(upstream)
            expected.backward(upstream)
        for value, target, original in zip((gate, up), (rg, ru), before):
            if value.requires_grad:
                torch.testing.assert_close(value.grad, target.grad, atol=3e-6, rtol=3e-5)
            else:
                self.assertIsNone(value.grad)
            torch.testing.assert_close(value, original, atol=0, rtol=0)
        # In particular, up.requires_grad must not override disabled grad mode.
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                value = student.swiglu(gate, up)
                self.assertFalse(value.requires_grad)
                torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-5)

    def test_chunk_views_token_channel_tails_and_offsets(self):
        for B, T, I in ((1, 1, 1), (2, 1, 65), (2, 3, 31), (2, 4, 32),
                        (2, 5, 33), (2, 17, 65), (2, 5, 2048)):
            with self.subTest(shape=(B, T, I)):
                storage = torch.randn(B * T * 2 * I + 14, device='cuda')
                packed = storage[7:-7].view(B, T, 2 * I)
                before = storage.clone()
                gate, up = packed.chunk(2, -1)
                self.assertEqual(gate.stride(), (2*T*I, 2*I, 1))
                self.assertEqual(up.storage_offset() - gate.storage_offset(), I)
                for use_sum in (False, True):
                    self.check_values_and_gradients(gate, up, use_sum)
                torch.testing.assert_close(storage, before, atol=0, rtol=0)

    def test_independent_strides_contiguous_transposed_and_expanded(self):
        shape = (2, 5, 65)
        pairs = [
            (torch.randn(shape, device='cuda'), torch.randn(shape, device='cuda')),
            (torch.randn(2, 5, 131, device='cuda')[..., 1::2],
             torch.randn(2, 65, 5, device='cuda').transpose(1, 2)),
            (torch.randn(1, 5, 1, device='cuda').expand(shape),
             torch.randn(2, 1, 65, device='cuda').expand(shape)),
            torch.randn(2, 5, 260, device='cuda')[..., ::2].chunk(2, -1),
        ]
        for gate, up in pairs:
            for needs in ((True, True), (True, False), (False, True)):
                with self.subTest(strides=(gate.stride(), up.stride()), needs=needs):
                    self.check_values_and_gradients(gate, up, needs=needs)

    def test_raw_backward_values_zero_gate_and_large_magnitudes(self):
        gate = torch.tensor([-1000, -100, -20, -2, 0, 2, 20, 100, 1000],
                            dtype=torch.float32, device='cuda').view(1, 1, -1).expand(2, 5, -1)
        up = torch.randn_like(gate)
        gradient = torch.randn(1, 5, 1, device='cuda').expand_as(gate)
        rg, ru = gate.double().requires_grad_(), up.double().requires_grad_()
        expected = reference.swiglu(rg, ru)
        expected_grads = torch.autograd.grad(expected, (rg, ru), gradient.double())
        with torch.no_grad():
            actual = self.extension.swiglu_forward(gate, up)
            grads = self.extension.swiglu_backward(gradient, gate, up)
        torch.testing.assert_close(actual, expected.float(), atol=2e-6, rtol=2e-5)
        for value, target in zip(grads, expected_grads):
            self.assertTrue(value.is_contiguous())
            self.assertTrue(torch.isfinite(value).all())
            torch.testing.assert_close(value, target.float(), atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(grads[0][..., 4], gradient[..., 4] * up[..., 4] / 2)
        torch.testing.assert_close(grads[1][..., 4], torch.zeros_like(up[..., 4]))

    def test_empty_inputs_and_capped_grid(self):
        for shape in ((0, 5, 65), (2, 0, 65), (2, 5, 0), (65536, 1, 1)):
            with self.subTest(shape=shape):
                self.check_values_and_gradients(torch.randn(shape, device='cuda'),
                                                torch.randn(shape, device='cuda'), use_sum=True)

    def test_chunk_projection_chain_and_aliased_inputs(self):
        # Verify the actual Linear -> chunk -> SwiGLU -> Linear training path.
        x = torch.randn(2, 5, 8, device='cuda', requires_grad=True)
        weight = (torch.randn(130, 8, device='cuda') / 8).requires_grad_()
        down = (torch.randn(8, 65, device='cuda') / 65).requires_grad_()
        rx, rw, rd = [v.detach().clone().requires_grad_() for v in (x, weight, down)]
        gate, up = torch.nn.functional.linear(x, weight).chunk(2, -1)
        rg, ru = torch.nn.functional.linear(rx, rw).chunk(2, -1)
        actual = torch.nn.functional.linear(student.swiglu(gate, up), down)
        expected = torch.nn.functional.linear(reference.swiglu(rg, ru), rd)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        for value, target in zip((x, weight, down), (rx, rw, rd)):
            torch.testing.assert_close(value.grad, target.grad, atol=5e-6, rtol=5e-5)
        g = torch.randn(2, 5, 65, device='cuda', requires_grad=True)
        rg = g.detach().clone().requires_grad_()
        student.swiglu(g, g).sum().backward()
        reference.swiglu(rg, rg).sum().backward()
        torch.testing.assert_close(g.grad, rg.grad, atol=3e-6, rtol=3e-5)

    def test_validation_and_native_grad_guards(self):
        gate = torch.randn(2, 5, 65, device='cuda')
        up = torch.randn_like(gate)
        invalid = [(gate.cpu(), up, 'CUDA'), (gate, up.cpu(), 'CUDA'),
                   (gate.half(), up, 'float32'), (gate, up.bfloat16(), 'float32'),
                   (gate.double(), up.double(), 'float32'),
                   (gate.to_sparse(), up, 'strided layout'),
                   (gate[0], up[0], '3D'), (gate, up[..., :-1], 'same shape')]
        for g, u, message in invalid:
            for call in (student.swiglu, self.extension.swiglu_forward):
                with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                    call(g, u)
            with self.assertRaisesRegex(RuntimeError, message):
                self.extension.swiglu_backward(torch.ones_like(gate), g, u)
        for dz, message in ((gate.cpu(), 'CUDA'), (gate.half(), 'float32'),
                            (gate[0], '3D'), (gate[..., :-1], 'same shape'),
                            (gate.to_sparse(), 'strided layout')):
            with self.assertRaisesRegex(RuntimeError, message):
                self.extension.swiglu_backward(dz, gate, up)
        for needs in ((True, False), (False, True), (True, True)):
            g, u = gate.detach().requires_grad_(needs[0]), up.detach().requires_grad_(needs[1])
            with self.assertRaisesRegex(RuntimeError, 'autograd binding'):
                self.extension.swiglu_forward(g, u)
            with self.assertRaisesRegex(RuntimeError, 'first-order'):
                self.extension.swiglu_backward(torch.ones_like(gate), g, u)
            with torch.no_grad():
                self.extension.swiglu_forward(g, u)
                self.extension.swiglu_backward(torch.ones_like(gate), g, u)
        with self.assertRaisesRegex(RuntimeError, 'first-order'):
            self.extension.swiglu_backward(torch.ones_like(gate, requires_grad=True), gate, up)

    @torch.no_grad()
    def test_current_stream_forward_and_backward(self):
        gate, up = torch.zeros(2, 5, 130, device='cuda').chunk(2, -1)
        gradient = torch.zeros_like(gate)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            gate.fill_(2)
            up.fill_(3)
            gradient.fill_(4)
            output = self.extension.swiglu_forward(gate, up).clone()
            dg, du = [v.clone() for v in self.extension.swiglu_backward(gradient, gate, up)]
        stream.synchronize()
        with torch.enable_grad():
            g, u = gate.detach().requires_grad_(), up.detach().requires_grad_()
            expected = reference.swiglu(g, u)
            eg, eu = torch.autograd.grad(expected, (g, u), gradient)
        for value, target in ((output, expected), (dg, eg), (du, eu)):
            torch.testing.assert_close(value, target, atol=3e-6, rtol=3e-5)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'requires two CUDA devices')
    def test_device_guard_and_mismatched_devices(self):
        with torch.cuda.device(0):
            gate = torch.randn(2, 5, 65, device='cuda:1')
            up = torch.randn_like(gate)
            with torch.no_grad():
                out = self.extension.swiglu_forward(gate, up)
                dg, du = self.extension.swiglu_backward(torch.ones_like(gate), gate, up)
            self.assertEqual(out.device, gate.device)
            self.assertEqual(dg.device, gate.device)
            self.assertEqual(du.device, gate.device)
            self.assertEqual(torch.cuda.current_device(), 0)
            torch.testing.assert_close(out, reference.swiglu(gate, up), atol=2e-6, rtol=2e-5)
            with self.assertRaisesRegex(RuntimeError, 'same CUDA device'):
                self.extension.swiglu_forward(gate, up.to('cuda:0'))
            with self.assertRaisesRegex(RuntimeError, 'same CUDA device'):
                self.extension.swiglu_backward(gate.to('cuda:0'), gate, up)


if __name__ == '__main__':
    unittest.main()
