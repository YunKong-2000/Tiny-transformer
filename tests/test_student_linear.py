"""FP32 CUTLASS Linear: rectangular GEMMs, view copies, gradients and streams."""
import unittest
from unittest.mock import Mock, patch

import torch

from tiny_transformer.operators import reference, student
from tiny_transformer.operators._extension import load_linear_extension
from tiny_transformer.benchmarks.linear import LinearFp32Validator


class StudentLinearHostTests(unittest.TestCase):
    def test_autograd_bridge_copies_upstream_and_accumulates_view_gradients(self):
        # CPU test double checks graph wiring only, not CUTLASS numerics.
        def forward(x, weight):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(x.is_contiguous() and weight.is_contiguous())
            return reference.linear(x, weight)

        def backward(gradient, x, weight):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(gradient.is_contiguous())
            return gradient @ weight, gradient.flatten(0, 1).T @ x.flatten(0, 1)

        extension = Mock(linear_forward=Mock(side_effect=forward),
                         linear_backward=Mock(side_effect=backward))
        with patch.object(student, 'load_linear_extension', return_value=extension):
            for needs in ((True, True), (True, False), (False, True)):
                xbase = torch.randn(2, 3, 10, requires_grad=needs[0])
                wbase = torch.randn(7, 10, requires_grad=needs[1])
                rx = xbase.detach().clone().requires_grad_(needs[0])
                rw = wbase.detach().clone().requires_grad_(needs[1])
                for use_sum in (False, True):
                    actual = student._Linear.apply(xbase[..., ::2].contiguous(),
                                                   wbase[..., ::2].contiguous())
                    expected = reference.linear(rx[..., ::2], rw[..., ::2])
                    torch.testing.assert_close(actual, expected)
                    if use_sum:
                        actual.sum().backward()
                        expected.sum().backward()
                    else:
                        upstream = torch.randn(2, 3, 14)[..., ::2]
                        actual.backward(upstream)
                        expected.backward(upstream)
                for value, target in ((xbase, rx), (wbase, rw)):
                    if value.requires_grad:
                        torch.testing.assert_close(value.grad, target.grad)
                    else:
                        self.assertIsNone(value.grad)


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA and nvcc')
class StudentLinearCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A GPU without build tools must fail here, not silently skip compilation.
        cls.extension = load_linear_extension()

    def setUp(self):
        torch.manual_seed(42)
        previous = torch.get_float32_matmul_precision()
        self.addCleanup(torch.set_float32_matmul_precision, previous)
        torch.set_float32_matmul_precision('highest')

    def check_values_and_gradients(self, x, weight, *, use_sum=False,
                                   needs=(True, True), check_inference=False):
        x, weight = x.detach().requires_grad_(needs[0]), weight.detach().requires_grad_(needs[1])
        rx = x.detach().clone().requires_grad_(needs[0])
        rw = weight.detach().clone().requires_grad_(needs[1])
        actual, expected = student.linear(x, weight), reference.linear(rx, rw)
        validator = LinearFp32Validator(x, weight)
        self.assertEqual(actual.shape, (*x.shape[:-1], weight.shape[0]))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(actual.is_contiguous())
        validator.forward(actual, expected)
        if use_sum:
            upstream = torch.ones_like(actual)
            actual.sum().backward()
            expected.sum().backward()
        else:
            # A noncontiguous upstream also exercises the public autograd bridge.
            upstream = (torch.randn(*actual.shape[:-1], actual.shape[-1] * 2,
                                    device='cuda') * 0.1)[..., ::2]
            actual.backward(upstream)
            expected.backward(upstream)
        indices = tuple(i for i, needed in enumerate(needs) if needed)
        validator.backward(tuple(v.grad for v in (x, weight) if v.requires_grad),
                           tuple(v.grad for v in (rx, rw) if v.requires_grad), upstream, indices)
        for value, target in ((x, rx), (weight, rw)):
            torch.testing.assert_close(value, target, atol=0, rtol=0)
            if not value.requires_grad:
                self.assertIsNone(value.grad)
        if check_inference:
            for context in (torch.no_grad, torch.inference_mode):
                with context():
                    output = student.linear(x, weight)
                    self.assertFalse(output.requires_grad)
                    validator.forward(output, expected)

    def test_rectangular_tail_shapes_and_all_input_ranks(self):
        for shape, n in (((5,), 7), ((3, 17), 9), ((2, 17, 65), 97),
                         ((2, 5, 97), 31), ((1, 1, 1), 1), ((2, 3, 4, 17), 33)):
            with self.subTest(shape=shape, n=n):
                self.check_values_and_gradients(
                    torch.randn(shape, device='cuda') * 0.1,
                    torch.randn(n, shape[-1], device='cuda') * 0.1,
                    use_sum=True, check_inference=True)

    def test_forward_dispatch_boundaries_and_tail_tiles(self):
        # Cross both the small CTA's M tile and the runtime dispatch threshold.
        # Odd N/K exercise masked columns and the final reduction tile.
        for m in (7, 8, 9, 127, 128, 129):
            with self.subTest(m=m):
                self.check_values_and_gradients(
                    torch.randn(1, m, 65, device='cuda'),
                    torch.randn(97, 65, device='cuda'), check_inference=True)

    @torch.no_grad()
    def test_small_m_shared_memory_mapping_with_exact_integer_products(self):
        # These small integers make all FP32 partial sums exact, so a mismatch
        # cannot be explained by accumulation order. Exercise K-stage reuse,
        # M/N tails and the reported LM-head failure without tolerance relaxation.
        for m, n, k in ((8, 33, 8), (9, 65, 17), (8, 8192, 768)):
            with self.subTest(m=m, n=n, k=k):
                x = ((torch.arange(m * k, device='cuda') % 7) - 3).float().view(1, m, k)
                weight = ((torch.arange(n * k, device='cuda') % 5) - 2).float().view(n, k)
                expected = reference.linear(x.double(), weight.double()).float()
                for _ in range(3):
                    torch.testing.assert_close(student.linear(x, weight), expected, atol=0, rtol=0)

    def test_five_model_projections_training_and_decode(self):
        for n, k in ((2304, 768), (768, 768), (4096, 768), (768, 2048), (8192, 768)):
            for batch, time in ((8, 512), (8, 1), (1, 1)):
                with self.subTest(n=n, k=k, batch=batch, time=time):
                    self.check_values_and_gradients(
                        torch.randn(batch, time, k, device='cuda'),
                        torch.randn(n, k, device='cuda'))

    @torch.no_grad()
    def test_prepared_kernel_outputs_match_production_and_graph_replay(self):
        for m, n, k, kind in ((8, 33, 17, 'small_m'), (128, 33, 17, 'large_m'),
                              (8, 65, 2049, 'split_k')):
            with self.subTest(kind=kind):
                x = torch.randn(1, m, k, device='cuda')
                weight = torch.randn(n, k, device='cuda')
                dy = torch.randn(1, m, n, device='cuda')
                bench = self.extension.prepare_linear_benchmark(x, weight, dy, True)
                self.assertEqual(bench.forward_kind, kind)
                production = student.linear(x, weight)
                dx, dw = self.extension.linear_backward(dy, x, weight)
                for name in ('forward', 'dx', 'dweight'):
                    bench.run(name)
                for actual, expected in ((bench.output, production), (bench.dx, dx), (bench.dweight, dw)):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                with self.assertRaisesRegex(RuntimeError, 'Unprepared'):
                    bench.run('unknown')
                if kind == 'split_k':
                    self.assertEqual(bench.workspace.numel(), bench.split_k_slices * m * n * 4)
                    bench.run('split_k_partials')
                    bench.output.fill_(float('nan'))
                    bench.run('split_k_reduce')
                    torch.testing.assert_close(bench.output, production, atol=0, rtol=0)
                from tiny_transformer.benchmarks.linear_kernels import capture_repeated
                for name, output, expected in (('forward', bench.output, production),
                                               ('dx', bench.dx, dx), ('dweight', bench.dweight, dw)):
                    graph = capture_repeated(lambda: bench.run(name), x.device, 2, 3)
                    output.fill_(float('nan'))
                    graph.replay()
                    torch.cuda.current_stream().synchronize()
                    torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_kernel_benchmark_validates_all_components_before_reporting_flops(self):
        from tiny_transformer.benchmarks.cases import make_case
        from tiny_transformer.benchmarks.operators import build_parser
        from tiny_transformer.benchmarks.linear_kernels import run_kernel_case
        args = build_parser().parse_args([
            '--operator', 'linear', '--linear-projections', 'custom',
            '--batch-size', '1', '--seq-length', '8', '--dim', '2049', '--out-features', '65',
            '--warmup', '2', '--repeats', '2', '--trials', '2'])
        case = make_case('linear', args, 'cuda', torch.float32, 'train', projection='custom')
        result = run_kernel_case(args, case, ('forward', 'backward'))
        self.assertTrue(result['validation']['split_k_stages_match_production'])
        self.assertEqual(result['kernel_config']['forward_kind'], 'split_k')
        for name in ('forward', 'dx', 'dweight'):
            self.assertEqual(result[name]['flops'], 2 * 8 * 65 * 2049)
            self.assertEqual(len(result[name]['candidate_trials_us']), 2)
            self.assertAlmostEqual(result[name]['candidate_flops_per_second'],
                                   result[name]['flops'] * 1e6 / result[name]['candidate_us'])
        self.assertEqual(result['split_k_reduce']['flops'],
                         result['kernel_config']['split_k_slices'] * 8 * 65)
        # Reference-only mode must use its own preallocated outputs, without JIT.
        args.backend = 'reference'
        reference_result = run_kernel_case(args, case, ('backward',))
        self.assertEqual(reference_result['kernel_config']['forward_kind'], 'reference')
        self.assertNotIn('forward', reference_result)
        self.assertIn('dx', reference_result)
        self.assertIn('dweight', reference_result)

    def test_split_k_dispatch_boundaries_and_reduction_tail(self):
        # K=2047 selects ordinary small-M GEMM; 2048/2049 select split-K.
        # M=128 switches back to the large-M kernel. Odd N tests reduction tails.
        for m in (1, 8, 127, 128):
            for k in (2047, 2048, 2049):
                with self.subTest(m=m, k=k):
                    self.check_values_and_gradients(
                        torch.randn(1, m, k, device='cuda'),
                        torch.randn(65, k, device='cuda'), check_inference=True)

    @torch.no_grad()
    def test_split_k_exact_partitions_repeated_calls_and_last_k_element(self):
        m, n, k = 8, 65, 2049
        x = torch.zeros(1, m, k, device='cuda')
        weight = torch.zeros(n, k, device='cuda')
        # Exercise the first partition, the start of the second, and the final
        # non-tile-aligned element. Small integer products sum exactly in FP32.
        rows = torch.arange(1, m + 1, device='cuda').float()
        columns = torch.arange(1, n + 1, device='cuda').float()
        for coordinate, coefficient in ((0, 1), (1024, 2), (2048, 4)):
            x[0, :, coordinate] = rows
            weight[:, coordinate] = coefficient * columns
        expected = (7 * rows[:, None] * columns[None, :]).unsqueeze(0)
        for _ in range(3):
            torch.testing.assert_close(student.linear(x, weight), expected, atol=0, rtol=0)
        # All partial sums must be overwritten on subsequent calls, not reused.
        x.zero_()
        torch.testing.assert_close(student.linear(x, weight), torch.zeros_like(expected), atol=0, rtol=0)

    @torch.no_grad()
    def test_split_k_workspace_on_nondefault_stream(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            m, n, k = 8, 65, 2049
            x = torch.empty(1, m, k, device='cuda')
            weight = torch.empty(n, k, device='cuda')
            torch.cuda._sleep(1_000_000)
            weight.fill_(0.5)
            outputs = []
            for fill in (0.25, -0.5, 0.0):
                x.fill_(fill)
                outputs.append(student.linear(x, weight))
                # Reuse-sized allocator activity after the local workspace is
                # released must remain ordered after both split-K kernels.
                torch.empty(2 * m * n, device='cuda').fill_(123.)
            consumed = [output.clone() for output in outputs]
        stream.synchronize()
        for output, fill in zip(consumed, (0.25, -0.5, 0.0)):
            torch.testing.assert_close(output, torch.full_like(output, k * fill * 0.5), atol=0, rtol=0)

    def test_strides_offsets_last_only_and_single_trainable_input(self):
        xstore = torch.randn(2 * 5 * 17 + 3, device='cuda') * 0.1
        wstore = torch.randn(23 * 17 + 5, device='cuda') * 0.1
        cases = [
            (xstore[3:].view(2, 5, 17), wstore[5:].view(23, 17)),
            ((torch.randn(2, 5, 34, device='cuda') * 0.1)[..., ::2],
             torch.randn(17, 23, device='cuda').T * 0.1),
            (torch.randn(2, 5, 17, device='cuda')[:, -1:, :],
             torch.randn(23, 17, device='cuda') * 0.1),
            (torch.randn(1, 1, 17, device='cuda').expand(2, 5, 17),
             torch.randn(23, 17, device='cuda') * 0.1),
        ]
        for x, weight in cases:
            for needs in ((True, True), (True, False), (False, True)):
                with self.subTest(strides=x.stride(), needs=needs):
                    self.check_values_and_gradients(x, weight, needs=needs)

    def test_zero_m_n_k_and_empty_batch(self):
        for shape, n in (((0, 3, 5), 7), ((2, 0, 5), 7), ((2, 3, 0), 7),
                         ((2, 3, 5), 0), ((0,), 7), ((2, 0, 0), 0)):
            with self.subTest(shape=shape, n=n):
                self.check_values_and_gradients(torch.zeros(shape, device='cuda'),
                                                torch.zeros(n, shape[-1], device='cuda'),
                                                use_sum=True, check_inference=True)

    def test_validation_raw_grad_guards_and_unsupported_autocast(self):
        x = torch.randn(2, 3, 5, device='cuda')
        weight = torch.randn(7, 5, device='cuda')
        for args, message in (
                ((x.cpu(), weight), 'CUDA'), ((x.half(), weight), 'float32'),
                ((x, weight.bfloat16()), 'float32'), ((x.double(), weight.double()), 'float32'),
                ((x.to_sparse(), weight), 'strided'), ((x, weight[0]), '2D'),
                ((x, weight[:, :4]), 'last dimension')):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                student.linear(*args)
        with torch.autocast('cuda', dtype=torch.float16):
            with self.assertRaisesRegex(RuntimeError, 'autocast'):
                student.linear(x, weight)
        with self.assertRaisesRegex(RuntimeError, 'contiguous'):
            self.extension.linear_forward(x.transpose(0, 1), weight)
        with self.assertRaisesRegex(RuntimeError, '3D'):
            self.extension.linear_forward(x[0], weight)
        # Empty storage tests the int32 limit without allocating a huge tensor.
        with self.assertRaisesRegex(RuntimeError, 'int32'):
            self.extension.linear_forward(torch.empty(1, 2**31, 0, device='cuda'),
                                          torch.empty(7, 0, device='cuda'))
        with self.assertRaisesRegex(RuntimeError, 'int32'):
            student.linear(torch.empty(2**31, 0, device='cuda'),
                           torch.empty(7, 0, device='cuda'))
        with self.assertRaisesRegex(RuntimeError, 'autograd binding'):
            self.extension.linear_forward(x.requires_grad_(), weight)
        gradient = torch.ones(2, 3, 7, device='cuda')
        with self.assertRaisesRegex(RuntimeError, 'first-order'):
            self.extension.linear_backward(gradient, x, weight)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, 'first two dimensions'):
                self.extension.linear_backward(gradient[:, :2].contiguous(), x, weight)
            with self.assertRaisesRegex(RuntimeError, 'last dimension'):
                self.extension.linear_backward(gradient[..., :6].contiguous(), x, weight)

    def test_current_stream_orders_producer_gemm_and_backward(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x = torch.empty(2, 3, 17, device='cuda')
            weight = torch.empty(23, 17, device='cuda')
            # Delay the producer so a GEMM accidentally using the default stream
            # cannot safely consume the input before these fills finish.
            torch.cuda._sleep(1_000_000)
            x.fill_(0.25)
            weight.fill_(0.5)
            x.requires_grad_()
            weight.requires_grad_()
            output = student.linear(x, weight)
            output.sum().backward()
            result = (output.clone(), x.grad.clone(), weight.grad.clone())
        stream.synchronize()
        for actual, value in zip(result, (17 * 0.125, 23 * 0.5, 6 * 0.25)):
            torch.testing.assert_close(actual, torch.full_like(actual, value), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
