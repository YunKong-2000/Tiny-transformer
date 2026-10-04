"""Shared performance methodology, all operator cases, and phase support tests."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

import torch

from tiny_transformer.benchmarks import common
from tiny_transformer.benchmarks.cases import LINEAR_PROJECTIONS, linear_spec, make_case
from tiny_transformer.benchmarks.linear import LinearFp32Validator
from tiny_transformer.benchmarks.embedding import PATTERNS, make_ids
from tiny_transformer.benchmarks.operators import build_parser, run, run_case, unsupported_reason, validate_args
from tiny_transformer.operators import reference, student
from tiny_transformer.operators.dispatch import NAMES


def small_args(*extra):
    return build_parser().parse_args([
        '--operator', 'all', '--batch-size', '2', '--seq-length', '3',
        '--dim', '8', '--heads', '2', '--hidden-dim', '12', '--out-features', '10',
        '--vocab-size', '17', '--patterns', 'random', '--warmup', '2',
        '--repeats', '2', '--trials', '3', *extra])


class BenchmarkHostTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

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

    def test_every_operator_prefill_decode_and_strides(self):
        args = small_args()
        for operator in NAMES:
            for workload in ('prefill', 'decode'):
                for layout in ('contiguous', 'strided'):
                    with self.subTest(operator=operator, workload=workload, layout=layout):
                        case = make_case(operator, args, 'cpu', torch.float32, workload, layout)
                        function = getattr(reference, operator)
                        calls, errors = common.prepare_calls(function, function, case.inputs,
                                                             case.grad_indices,
                                                             upstream_scale=case.upstream_scale)
                        self.assertEqual(errors['forward_max_abs_error'], 0)
                        self.assertEqual(errors['backward_max_abs_error'], 0)
                        self.assertEqual(len(calls['backward'][0]()), len(case.grad_indices))
                        for value in case.inputs:
                            if torch.is_tensor(value):
                                self.assertIsNone(value.grad)
                                if layout == 'contiguous':
                                    self.assertTrue(value.is_contiguous())
                        if operator == 'rope':
                            self.assertEqual(case.grad_indices, (0,))
                        if operator == 'attention' and workload == 'decode':
                            self.assertEqual(case.inputs[0].shape[-2], 1)
                            self.assertEqual(case.inputs[1].shape[-2], args.seq_length)
                            self.assertEqual(case.inputs[3], args.seq_length - 1)

    def test_rms_norm_last_only_and_no_gradient_forward(self):
        args = small_args()
        case = make_case('rms_norm', args, 'cpu', torch.float32, 'prefill', 'last-only')
        self.assertEqual(case.inputs[0].shape, (2, 1, 8))
        self.assertFalse(case.inputs[0].is_contiguous())
        self.assertGreater(case.inputs[0].storage_offset(), 0)

        def forward_only(x, weight, eps):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(x.requires_grad or weight.requires_grad)
            self.assertEqual(x.stride(), case.inputs[0].stride())
            return reference.rms_norm(x, weight, eps)

        calls, _ = common.prepare_calls(reference.rms_norm, forward_only, case.inputs,
                                        case.grad_indices, phases=('forward',))
        self.assertEqual(set(calls), {'forward'})
        calls['forward'][1]()

    def test_backward_has_no_forward_or_leaf_accumulation(self):
        x = torch.randn(2, 3, 10)[..., ::2].requires_grad_()
        weight = torch.randn(5, requires_grad=True)
        upstream = torch.randn(2, 3, 10)[..., ::2]
        candidate = Mock(side_effect=reference.rms_norm)
        calls, _ = common.prepare_calls(reference.rms_norm, candidate, (x, weight, 1e-6),
                                        (0, 1), upstream=upstream)
        expected = torch.autograd.grad(reference.rms_norm(x, weight, 1e-6), (x, weight), upstream)
        candidate.reset_mock()
        for _ in range(3):
            for call in calls['backward']:
                for actual, target in zip(call(), expected):
                    torch.testing.assert_close(actual, target)
        candidate.assert_not_called()
        self.assertIsNone(x.grad)
        self.assertIsNone(weight.grad)

    def test_incorrect_forward_and_individual_gradient_are_rejected(self):
        class WrongSecondGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return x + y

            @staticmethod
            def backward(ctx, gradient):
                return gradient, gradient * 2

        inputs = (torch.ones(2, 3, 5), torch.ones(2, 3, 5))
        with self.assertRaises(AssertionError):
            common.prepare_calls(reference.residual, lambda x, y: x + y + 1, inputs, (0, 1))
        with self.assertRaises(AssertionError):
            common.prepare_calls(reference.residual, WrongSecondGradient.apply, inputs, (0, 1),
                                 upstream=torch.ones_like(inputs[0]))

    def test_linear_fp64_validator_accepts_fp32_accumulation_order_differences(self):
        # A deterministic cancellation example: the tiny terms are lost when
        # accumulated after 1.0, but retained by FP64. A second output remains 1.
        # This checks validation arithmetic, not a replacement CUDA kernel.
        x = torch.ones(1, 1, 768)
        weight = torch.zeros(2, 768)
        weight[0, 0], weight[0, -1] = 1, -1
        weight[0, 1:-1] = 2**-25
        weight[1, 0] = 1
        sequential = torch.zeros(1, 1, 2)
        for k in range(768):
            sequential.addcmul_(x[..., k:k+1], weight[:, k])
        oracle = reference.linear(x.double(), weight.double()).float()
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(sequential, oracle, atol=1e-5, rtol=1e-4)
        report = LinearFp32Validator(x, weight).forward(sequential, oracle)
        self.assertGreater(report['candidate']['max_abs_error'], 1e-5)
        self.assertLess(report['candidate']['max_roundoff_ratio'], 1)

    def test_linear_fp64_validator_rejects_layout_errors_nan_and_bad_gradients(self):
        x, weight = torch.randn(2, 3, 65), torch.randn(17, 65)
        expected = reference.linear(x.double(), weight.double()).float()
        validator = LinearFp32Validator(x, weight)
        wrong_layout = expected.flip(-1)
        corrupted = expected.clone()
        corrupted[0, 0, 0] += 0.1
        nonfinite = expected.clone()
        nonfinite[0, 0, 0] = float('nan')
        for actual in (wrong_layout, corrupted, nonfinite, expected.double()):
            with self.subTest(kind=actual.dtype), self.assertRaises(AssertionError):
                validator.forward(actual, expected)
        # The reference is independently checked, too.
        with self.assertRaises(AssertionError):
            validator.forward(expected, corrupted)
        upstream = torch.randn_like(expected)
        dx = (upstream.double() @ weight.double()).float()
        dw = (upstream.flatten(0, 1).double().T @ x.flatten(0, 1).double()).float()
        report = validator.backward((dx, dw), (dx, dw), upstream, (0, 1))
        self.assertEqual(report['0']['reduction_length'], 17)
        self.assertEqual(report['1']['reduction_length'], 6)
        for grads in ((dx + 1, dw), (dx, dw + 1)):
            with self.assertRaises(AssertionError):
                validator.backward(grads, (dx, dw), upstream, (0, 1))

    def test_linear_fp64_rms_gate_rejects_errors_within_loose_roundoff_bound(self):
        # For a long reduction the worst-case bound alone is too permissive.
        x, weight = torch.ones(1, 1, 4096), torch.ones(1, 4096)
        expected = torch.full((1, 1, 1), 4096.)
        with self.assertRaisesRegex(AssertionError, 'RMS limit'):
            LinearFp32Validator(x, weight).forward(expected + 0.75, expected)

    def test_linear_fp64_validation_is_outside_retained_timing_calls(self):
        x, weight = torch.randn(2, 3, 17), torch.randn(9, 17)
        validator = LinearFp32Validator(x, weight)
        with patch.object(validator, 'forward', wraps=validator.forward) as forward, \
                patch.object(validator, 'backward', wraps=validator.backward) as backward:
            calls, errors = common.prepare_calls(reference.linear, reference.linear, (x, weight),
                                                 (0, 1), validator=validator)
            self.assertEqual(forward.call_count, 2)
            self.assertEqual(backward.call_count, 1)
            forward.reset_mock()
            backward.reset_mock()
            for phase in ('forward', 'backward'):
                for function in calls[phase]:
                    function()
            forward.assert_not_called()
            backward.assert_not_called()
        self.assertEqual(errors['validation_policy'], validator.policy)
        self.assertIn('candidate', errors['forward_fp64'])

    def test_event_measurement_alternates_order_and_reports_trials(self):
        order = []
        reference_call = lambda: order.append('reference')
        candidate_call = lambda: order.append('candidate')
        start, end = Mock(), Mock()
        start.elapsed_time.side_effect = [4, 2, 6, 8, 12, 10]
        with patch.object(torch.cuda, 'Event', side_effect=(start, end)):
            measured = common.measure_pair((reference_call, candidate_call), 2, 2, 3)
        self.assertEqual(order, ['reference'] * 2 + ['candidate'] * 2 +
                         ['reference'] * 2 + ['candidate'] * 2 +
                         ['candidate'] * 2 + ['reference'] * 2 +
                         ['reference'] * 2 + ['candidate'] * 2)
        self.assertEqual(measured['reference_trials_us'], [2000, 4000, 6000])
        self.assertEqual(measured['candidate_trials_us'], [1000, 3000, 5000])
        self.assertEqual(measured['reference_us'], 4000)
        self.assertEqual(measured['candidate_us'], 3000)
        self.assertEqual(measured['speedup'], 4 / 3)
        self.assertEqual(end.synchronize.call_count, 7)  # event initialization + six intervals

    def test_all_skips_unimplemented_phases_without_fallback(self):
        args = small_args()
        timing = {'reference_us': 2, 'candidate_us': 1, 'speedup': 2,
                  'reference_trials_us': [2] * 3, 'candidate_trials_us': [1] * 3}
        def embedding(ids, weight, backward_impl):
            return reference.embedding(ids, weight)
        with patch.object(student, 'embedding', side_effect=embedding), \
                patch.object(student, 'rms_norm', side_effect=reference.rms_norm), \
                patch.object(student, 'residual', side_effect=reference.residual), \
                patch.object(student, 'cross_entropy', side_effect=reference.cross_entropy), \
                patch.object(student, 'rope', side_effect=reference.rope), \
                patch.object(student, 'swiglu', side_effect=reference.swiglu), \
                patch.object(student, 'linear', side_effect=reference.linear), \
                patch('tiny_transformer.benchmarks.operators.measure_pair', return_value=timing) as measure, \
                patch.object(student, 'attention') as missing, redirect_stdout(io.StringIO()):
            results = run(args, torch.device('cpu'))
        missing.assert_not_called()
        self.assertEqual(measure.call_count, 44)  # Other operators 24; five Linear projections 20.
        self.assertEqual(len(results), 29)  # Other operators 14; Linear 15.
        for row in results:
            if row['operator'] in ('linear', 'rms_norm', 'residual', 'cross_entropy', 'rope', 'swiglu'):
                self.assertEqual(row['forward']['status'], 'passed')
                if row['operator'] == 'linear' and row['workload'] != 'train':
                    self.assertNotIn('backward', row)
                    continue
                self.assertEqual(row['backward']['status'], 'passed')
                self.assertGreater(row['backward']['candidate_us'], 0)
            elif row['operator'] != 'embedding':
                self.assertEqual(row['status'], 'skipped')
                self.assertNotIn('validation', row)

    def test_build_and_correctness_failures_are_not_skips(self):
        for operator in ('linear', 'rms_norm', 'cross_entropy', 'rope', 'swiglu'):
            args = small_args('--operator', operator, '--workloads', 'prefill')
            for error in (RuntimeError('compiler failed'), AssertionError('wrong result')):
                with self.subTest(operator=operator, error=error), \
                        patch.object(student, operator, side_effect=error), \
                        patch('tiny_transformer.benchmarks.operators.measure_pair') as measure:
                    with self.assertRaises(type(error)):
                        run(args, torch.device('cpu'))
                measure.assert_not_called()

    def test_linear_default_suite_matches_all_model_gemm_shapes(self):
        args = build_parser().parse_args(['--operator', 'linear'])
        dimensions = {'qkv': (2304, 768), 'o': (768, 768),
                      'gate_up': (4096, 768), 'down': (768, 2048), 'lm_head': (8192, 768)}
        self.assertEqual(tuple(args.linear_projections), LINEAR_PROJECTIONS)
        for projection, (n, k) in dimensions.items():
            for workload in ('train', 'prefill', 'decode'):
                with self.subTest(projection=projection, workload=workload):
                    spec = linear_spec(args, workload, projection)
                    m = 8 if workload == 'decode' or (workload == 'prefill' and projection == 'lm_head') else 4096
                    self.assertEqual(spec['gemm_shapes']['forward'], [m, n, k])
                    self.assertEqual(spec['x_shape'], [8, m // 8, k])
                    self.assertEqual(spec['weight_shape'], [n, k])
                    self.assertEqual(spec['output_shape'], [8, m // 8, n])
                    if workload == 'train':
                        self.assertEqual(spec['gemm_shapes']['dx'], [m, k, n])
                        self.assertEqual(spec['gemm_shapes']['dweight'], [n, k, m])
                    else:
                        self.assertEqual(set(spec['gemm_shapes']), {'forward'})

    def test_linear_cases_preserve_strides_and_use_actual_row_count(self):
        args = small_args('--operator', 'linear', '--inference-batch-size', '1')
        for workload in ('train', 'prefill', 'decode'):
            for projection in LINEAR_PROJECTIONS:
                case = make_case('linear', args, 'cpu', torch.float32, workload, 'strided',
                                 projection=projection)
                spec = linear_spec(args, workload, projection)
                self.assertEqual(list(case.inputs[0].shape), spec['x_shape'])
                self.assertEqual(list(case.inputs[1].shape), spec['weight_shape'])
                self.assertEqual(case.inputs[0].stride()[-1], 2)
                self.assertEqual(case.inputs[1].stride()[-1], 2)
                self.assertEqual(case.upstream_scale, 1 / spec['gemm_shapes']['forward'][0])
                expected_batch = 2 if workload == 'train' else 1
                self.assertEqual(spec['x_shape'][0], expected_batch)

    def test_linear_suite_phase_and_projection_filters(self):
        timing = {'reference_us': 2, 'candidate_us': 1, 'speedup': 2}
        for workloads, phases, expected_rows, expected_timings in (
                (['train'], ['backward'], 2, 2),
                (['prefill', 'decode'], ['forward', 'backward'], 4, 4),
                (['decode'], ['backward'], 2, 0)):
            args = small_args('--operator', 'linear', '--backend', 'reference',
                              '--linear-projections', 'down', 'lm_head',
                              '--workloads', *workloads, '--phases', *phases)
            with patch('tiny_transformer.benchmarks.operators.measure_pair', return_value=timing) as measure, \
                    redirect_stdout(io.StringIO()):
                results = run(args, torch.device('cpu'))
            self.assertEqual(len(results), expected_rows)
            self.assertEqual(measure.call_count, expected_timings)
            self.assertEqual({row['projection'] for row in results}, {'down', 'lm_head'})
            for row in results:
                if workloads == ['train']:
                    self.assertNotIn('forward', row)
                    self.assertIn('combined', row['backward_scope'])
                elif phases == ['backward']:
                    self.assertEqual(row['backward']['reason'], 'inference has no backward')
                else:
                    self.assertNotIn('backward', row)

    def test_linear_skips_keep_shapes_without_allocating_and_custom_shape_is_explicit(self):
        args = small_args('--operator', 'linear', '--precision', 'bf16')
        with patch('tiny_transformer.benchmarks.operators.make_case') as allocate, \
                redirect_stdout(io.StringIO()):
            results = run(args, torch.device('cpu'))
        allocate.assert_not_called()
        self.assertEqual(len(results), 15)
        self.assertTrue(all(row['status'] == 'skipped' and 'gemm_shapes' in row for row in results))
        args = small_args('--operator', 'linear', '--linear-projections', 'custom', '--out-features', '11')
        self.assertEqual(linear_spec(args, 'train', 'custom')['gemm_shapes']['forward'], [6, 11, 8])

    def test_variants_share_upstream_and_restore_rng(self):
        args = small_args('--operator', 'embedding', '--backward-impl', 'all')
        case = make_case('embedding', args, 'cpu', torch.float32, 'prefill')
        rng = torch.random.get_rng_state()
        gradients = []

        def measure(functions, *unused):
            gradients.append(functions[1]()[0])
            return {}

        with patch.object(student, 'embedding', side_effect=lambda i, w, **kw: reference.embedding(i, w)), \
                patch('tiny_transformer.benchmarks.operators.measure_pair', side_effect=measure):
            for variant in ('grouped', 'baseline'):
                run_case('embedding', args, case, ('backward',), variant)
        torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_unsupported_precision_and_layout_are_explicit(self):
        self.assertIn('fp32', unsupported_reason('rms_norm', 'student', 'bf16', 'forward', 'contiguous'))
        self.assertIn('contiguous', unsupported_reason('embedding', 'student', 'fp32', 'forward', 'strided'))
        for phase in ('forward', 'backward'):
            for layout in ('contiguous', 'strided'):
                self.assertIsNone(unsupported_reason('linear', 'student', 'fp32', phase, layout))
                self.assertIsNone(unsupported_reason('rope', 'student', 'fp32', phase, layout))
                self.assertIsNone(unsupported_reason('swiglu', 'student', 'fp32', phase, layout))
            for precision in ('fp16', 'bf16'):
                self.assertIn('fp32', unsupported_reason('linear', 'student', precision, phase, 'strided'))
                self.assertIn('fp32', unsupported_reason('rope', 'student', precision, phase, 'strided'))
                self.assertIn('fp32', unsupported_reason('swiglu', 'student', precision, phase, 'strided'))
        self.assertIsNone(unsupported_reason('rms_norm', 'reference', 'bf16', 'backward', 'last-only'))
        self.assertIsNone(unsupported_reason('attention', 'sdpa', 'fp32', 'backward', 'contiguous'))
        for precision in ('fp32', 'fp16', 'bf16'):
            for phase in ('forward', 'backward'):
                self.assertIsNone(unsupported_reason('residual', 'student', precision, phase, 'strided'))
                for layout in ('contiguous', 'strided'):
                    self.assertIsNone(unsupported_reason('cross_entropy', 'student', precision, phase, layout))

    def test_rms_norm_large_width_skips_only_backward(self):
        self.assertIn('1024', unsupported_reason('rms_norm', 'student', 'fp32', 'backward', 'contiguous', 1025))
        self.assertIsNone(unsupported_reason('rms_norm', 'student', 'fp32', 'forward', 'contiguous', 1025))
        self.assertIsNone(unsupported_reason('rms_norm', 'student', 'fp32', 'backward', 'contiguous', 1024))

    def test_invalid_options(self):
        for options in (('--dim', '7'), ('--dim', '6'), ('--repeats', '0'),
                        ('--device', 'cpu'), ('--eps', 'nan'), ('--inference-batch-size', '0'),
                        ('--operator', 'rope', '--workloads', 'train')):
            with self.subTest(options=options), redirect_stdout(io.StringIO()), \
                    patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    validate_args(build_parser(), small_args(*options))


if __name__ == '__main__':
    unittest.main()
