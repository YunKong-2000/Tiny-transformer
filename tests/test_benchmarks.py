"""Shared performance methodology, all operator cases, and phase support tests."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

import torch

from tiny_transformer.benchmarks import common
from tiny_transformer.benchmarks.cases import make_case
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
                patch('tiny_transformer.benchmarks.operators.measure_pair', return_value=timing) as measure, \
                patch.object(student, 'linear') as missing, redirect_stdout(io.StringIO()):
            results = run(args, torch.device('cpu'))
        missing.assert_not_called()
        self.assertEqual(measure.call_count, 8)  # embedding and RMSNorm fwd/bwd, each workload
        self.assertEqual(len(results), 16)
        for row in results:
            if row['operator'] == 'rms_norm':
                self.assertEqual(row['forward']['status'], 'passed')
                self.assertEqual(row['backward']['status'], 'passed')
                self.assertGreater(row['backward']['candidate_us'], 0)
            elif row['operator'] != 'embedding':
                self.assertEqual(row['status'], 'skipped')
                self.assertNotIn('validation', row)

    def test_build_and_correctness_failures_are_not_skips(self):
        args = small_args('--operator', 'rms_norm', '--workloads', 'prefill')
        for error in (RuntimeError('compiler failed'), AssertionError('wrong result')):
            with patch.object(student, 'rms_norm', side_effect=error), \
                    patch('tiny_transformer.benchmarks.operators.measure_pair') as measure:
                with self.assertRaises(type(error)):
                    run(args, torch.device('cpu'))
            measure.assert_not_called()

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
        self.assertIsNone(unsupported_reason('rms_norm', 'reference', 'bf16', 'backward', 'last-only'))
        self.assertIsNone(unsupported_reason('attention', 'sdpa', 'fp32', 'backward', 'contiguous'))

    def test_rms_norm_large_width_skips_only_backward(self):
        self.assertIn('1024', unsupported_reason('rms_norm', 'student', 'fp32', 'backward', 'contiguous', 1025))
        self.assertIsNone(unsupported_reason('rms_norm', 'student', 'fp32', 'forward', 'contiguous', 1025))
        self.assertIsNone(unsupported_reason('rms_norm', 'student', 'fp32', 'backward', 'contiguous', 1024))

    def test_invalid_options(self):
        for options in (('--dim', '7'), ('--dim', '6'), ('--repeats', '0'),
                        ('--device', 'cpu'), ('--eps', 'nan')):
            with self.subTest(options=options), redirect_stdout(io.StringIO()), \
                    patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    validate_args(build_parser(), small_args(*options))


if __name__ == '__main__':
    unittest.main()
