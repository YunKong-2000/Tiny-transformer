"""Cross entropy values, first-order gradients and CUDA boundary contracts."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from tiny_transformer.operators import reference, student
from tiny_transformer.operators._extension import load_cross_entropy_extension


class StudentCrossEntropyHostTests(unittest.TestCase):
    def test_autograd_bridge_with_cpu_test_double(self):
        # Only tests graph wiring; this explicitly does NOT validate the CUDA kernels.
        def forward(logits, targets):
            self.assertFalse(torch.is_grad_enabled())
            maximum = logits.amax(-1)
            log_sum = (logits - maximum[..., None]).exp().sum(-1).log()
            cache = torch.stack((maximum, log_sum), -1)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                   reduction='none').view_as(targets)
            return loss, cache

        def backward(logits, targets, lse, grad_loss):
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(grad_loss.shape, torch.Size([]))
            self.assertEqual(grad_loss.dtype, torch.float32)
            probability = ((logits - lse[..., :1]) - lse[..., 1:]).exp()
            probability.scatter_add_(-1, targets.clamp_min(0)[..., None],
                                     -torch.ones_like(targets[..., None], dtype=logits.dtype))
            return (probability * grad_loss).masked_fill((targets == -100)[..., None], 0)

        extension = Mock(cross_entropy_forward=Mock(side_effect=forward),
                         cross_entropy_backward=Mock(side_effect=backward))
        torch.manual_seed(42)
        with patch.object(student, 'load_cross_entropy_extension', return_value=extension):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                x = torch.randn(2, 3, 14, dtype=dtype, requires_grad=True)
                expected_x = x.detach().clone().requires_grad_()
                targets = torch.tensor([[0, -100, 6], [3, 1, -100]])
                # Two live graphs, different valid counts, accumulation and scaled upstream.
                actual, expected = [], []
                for labels in (targets, targets.masked_fill(targets != 0, -100)):
                    actual.append(student._CrossEntropy.apply(x[..., ::2].float().contiguous(), labels))
                    expected.append(reference.cross_entropy(expected_x[..., ::2], labels))
                for i in (1, 0):
                    self.assertEqual(actual[i].shape, torch.Size([]))
                    self.assertEqual(actual[i].dtype, torch.float32)
                    torch.testing.assert_close(actual[i], expected[i])
                    (actual[i] * -2.75).backward()
                    (expected[i] * -2.75).backward()
                torch.testing.assert_close(x.grad, expected_x.grad)
            x = torch.randn(2, 3, 7, requires_grad=True)
            loss = student._CrossEntropy.apply(x, torch.full((2, 3), -100))
            self.assertTrue(torch.isnan(loss))
            loss.backward()
            torch.testing.assert_close(x.grad, torch.zeros_like(x))


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA and nvcc')
class StudentCrossEntropyCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = load_cross_entropy_extension()

    def setUp(self):
        torch.manual_seed(42)

    def assert_matches(self, values, targets, scale=-2.75):
        x = values.detach().requires_grad_()
        ref = values.detach().clone().requires_grad_()
        old_x, old_targets = x.detach().clone(), targets.clone()
        actual, expected = student.cross_entropy(x, targets), reference.cross_entropy(ref, targets)
        self.assertEqual(actual.shape, torch.Size([]))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device, x.device)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5, equal_nan=True)
        (actual * scale).backward()
        (expected * scale).backward()
        atol, rtol = (2e-6, 2e-5) if x.dtype == torch.float32 else (2e-3, 2e-2)
        torch.testing.assert_close(x.grad, ref.grad, atol=atol, rtol=rtol, equal_nan=True)
        torch.testing.assert_close(x.grad[targets == -100], torch.zeros_like(x.grad[targets == -100]), atol=0, rtol=0)
        torch.testing.assert_close(x, old_x, equal_nan=True)
        torch.testing.assert_close(targets, old_targets)
        with torch.no_grad():
            torch.testing.assert_close(student.cross_entropy(x, targets), expected, atol=2e-5, rtol=2e-5, equal_nan=True)

    def test_shapes_dtypes_and_strided_inputs(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            for vocab in (1, 3, 31, 32, 33, 65, 8192):
                with self.subTest(dtype=dtype, vocab=vocab):
                    x = torch.randn(2, 17, vocab * 2, device='cuda', dtype=dtype)[..., ::2]
                    targets = torch.randint(vocab, (2, 34), device='cuda')[:, ::2]
                    targets[0, ::3] = -100
                    self.assert_matches(x, targets)
        # Copies/casts in the public wrapper must propagate back into the base view.
        base = torch.randn(3, 2, 14, device='cuda', requires_grad=True)
        expected = base.detach().clone().requires_grad_()
        targets = torch.randint(7, (2, 3), device='cuda')
        student.cross_entropy(base.transpose(0, 1)[..., ::2], targets).backward()
        reference.cross_entropy(expected.transpose(0, 1)[..., ::2], targets).backward()
        torch.testing.assert_close(base.grad, expected.grad, atol=2e-6, rtol=2e-5)

    def test_stability_ignore_and_empty(self):
        targets = torch.tensor([[0, -100, 6], [3, 1, -100]], device='cuda')
        for offset in (0., -1e4, 1e8):
            with self.subTest(offset=offset):
                self.assert_matches(torch.full((2, 3, 7), offset, device='cuda'), targets)
        values = torch.randn(2, 3, 7, device='cuda') * 1000
        self.assert_matches(values, targets)
        one_valid = torch.full_like(targets, -100)
        one_valid[0, 0] = 0
        self.assert_matches(values, one_valid)
        self.assert_matches(values, torch.full_like(targets, -100))
        for shape in ((0, 3, 7), (2, 0, 7)):
            self.assert_matches(torch.empty(shape, device='cuda'),
                                torch.empty(shape[:2], device='cuda', dtype=torch.long))

    def test_raw_row_loss_cache_and_upstream(self):
        x = torch.randn(2, 3, 65, device='cuda')
        targets = torch.tensor([[0, -100, 64], [4, 2, 1]], device='cuda')
        with torch.no_grad():
            loss, lse = self.extension.cross_entropy_forward(x, targets)
            expected = F.cross_entropy(x.flatten(0, 1), targets.flatten(), reduction='none').view_as(targets)
            torch.testing.assert_close(loss, expected)
            valid = targets != -100
            torch.testing.assert_close(lse.sum(-1)[valid], x.logsumexp(-1)[valid])
            gradient = torch.tensor(-2.75, device='cuda') / valid.sum()
            actual = self.extension.cross_entropy_backward(x, targets, lse, gradient)
        ref = x.detach().requires_grad_()
        F.cross_entropy(ref.flatten(0, 1), targets.flatten(), reduction='sum').backward(gradient)
        torch.testing.assert_close(actual, ref.grad, atol=2e-6, rtol=2e-5)
        # Backward must read the supplied cache, not silently recompute it.
        with torch.no_grad():
            zero = torch.zeros(1, 1, 3, device='cuda')
            cache = torch.tensor([[[0., 0.]]], device='cuda')
            dz = self.extension.cross_entropy_backward(zero, targets[:1, :1], cache, torch.ones((), device='cuda'))
            torch.testing.assert_close(dz, torch.tensor([[[0., 1., 1.]]], device='cuda'))

    def test_capped_grid(self):
        shape = (1, 65535 * 8 + 1, 3)
        x = torch.randn(shape, device='cuda')
        targets = torch.randint(3, shape[:2], device='cuda')
        targets[:, ::13] = -100
        self.assert_matches(x, targets, scale=1000.)

    def test_current_stream(self):
        x = torch.zeros(2, 3, 65, device='cuda')
        targets = torch.zeros(2, 3, device='cuda', dtype=torch.long)
        gradient = torch.zeros((), device='cuda')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            x.normal_()
            targets.fill_(7)
            loss, cache = self.extension.cross_entropy_forward(x, targets)
            # The kernel must consume the scalar written on this stream.
            gradient.fill_(-2.75)
            dz = self.extension.cross_entropy_backward(x, targets, cache, gradient)
            loss, dz = loss.clone(), dz.clone()
        stream.synchronize()
        ref = x.detach().requires_grad_()
        expected = F.cross_entropy(ref.flatten(0, 1), targets.flatten(), reduction='none').view_as(targets)
        expected.sum().backward(gradient)
        torch.testing.assert_close(loss, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(dz, ref.grad, atol=2e-6, rtol=2e-5)

    def test_amp_scaler_and_accumulation(self):
        x = torch.randn(2, 3, 33, device='cuda', requires_grad=True)
        ref = x.detach().clone().requires_grad_()
        scaler = torch.amp.GradScaler('cuda', init_scale=128.)
        for index in range(2):
            targets = torch.randint(33, (2, 3), device='cuda')
            targets[0, :index + 1] = -100
            with torch.autocast('cuda', dtype=torch.float16):
                actual = student.cross_entropy(x.half(), targets)
                expected = reference.cross_entropy(ref.half(), targets)
            scaler.scale(actual / 2).backward()
            scaler.scale(expected / 2).backward()
        torch.testing.assert_close(x.grad, ref.grad, atol=2e-3, rtol=2e-3)

    def test_native_and_public_validation(self):
        x = torch.randn(2, 3, 7, device='cuda')
        targets = torch.zeros(2, 3, device='cuda', dtype=torch.long)
        cases = [(x.cpu(), targets, 'CUDA'), (x, targets.cpu(), 'CUDA'),
                 (x.double(), targets, 'float32'), (x, targets.int(), 'int64'),
                 (x[0], targets, '3D'), (x, targets.flatten(), '2D'),
                 (x, targets[:, :1], 'shape'), (x.transpose(0, 1), targets.T, 'contiguous'),
                 (x.to_sparse(), targets, 'strided'),
                 (x[..., :0], targets, 'positive')]
        for logits, labels, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.extension.cross_entropy_forward(logits, labels)
        for logits, labels, message in ((x.double(), targets, 'float32'), (x, targets.int(), 'int64'),
                                        (x, targets.flatten(), 'shape'), (x[..., :0], targets, 'positive')):
            with self.assertRaisesRegex(RuntimeError, message):
                student.cross_entropy(logits, labels)
        with torch.no_grad():
            _, lse = self.extension.cross_entropy_forward(x, targets)
            args = [x, targets, lse, torch.ones((), device='cuda')]
            for index, value, message in (
                (2, lse.cpu(), 'CUDA'), (3, args[3].cpu(), 'CUDA'),
                (2, lse.half(), 'float32'), (3, args[3].half(), 'float32'),
                (2, lse[..., 0], 'shape'), (3, args[3].flatten(), 'scalar'),
                (2, torch.zeros(2, 3, 4, device='cuda')[..., ::2], 'contiguous'),
                (3, torch.ones_like(targets, dtype=torch.float32), 'scalar'),
                (3, torch.empty(0, device='cuda'), 'scalar')):
                changed = args.copy()
                changed[index] = value
                with self.subTest(index=index, message=message), self.assertRaisesRegex(RuntimeError, message):
                    self.extension.cross_entropy_backward(*changed)
        with self.assertRaisesRegex(RuntimeError, 'autograd binding'):
            self.extension.cross_entropy_forward(x.requires_grad_(), targets)
        with self.assertRaisesRegex(RuntimeError, 'first-order'):
            self.extension.cross_entropy_backward(x, targets, lse, args[3])
        with self.assertRaisesRegex(RuntimeError, 'first-order'):
            self.extension.cross_entropy_backward(x.detach(), targets, lse, args[3].clone().requires_grad_())
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            student.cross_entropy(x, targets).backward()
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    def test_invalid_targets_in_separate_processes(self):
        # Device assertions poison their CUDA context, so isolate each case.
        for entry in ('forward', 'backward'):
            for target in (-1, 7):
                code = f'''
import torch
from tiny_transformer.operators._extension import load_cross_entropy_extension
extension = load_cross_entropy_extension()
x = torch.zeros(1, 1, 7, device='cuda')
y = torch.tensor([[{target}]], device='cuda')
if {entry!r} == 'forward':
    extension.cross_entropy_forward(x, y)
else:
    extension.cross_entropy_backward(x, y, torch.zeros(1, 1, 2, device='cuda'), torch.ones((), device='cuda'))
torch.cuda.synchronize()
'''
                result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                        cwd=Path(__file__).resolve().parents[1], timeout=120)
                with self.subTest(entry=entry, target=target):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('device-side assert', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
