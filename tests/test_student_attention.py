"""Attention forward: online softmax, tails, cache strides and API boundaries."""
import math
import unittest
from unittest.mock import Mock, patch

import torch

from tiny_transformer.operators import reference, student
from tiny_transformer.operators._extension import load_attention_extension


class StudentAttentionHostTests(unittest.TestCase):
    def test_forward_bridge_and_backward_guard(self):
        # Test doubles check only Python wiring; no CUDA arithmetic is simulated.
        q, k, v = [Mock(is_cuda=True, requires_grad=False) for _ in range(3)]
        output, lse, segments = object(), object(), object()
        extension = Mock()
        extension.attention_forward.return_value = (output, lse)
        with patch.object(student, 'load_attention_extension', return_value=extension) as load:
            self.assertIs(student.attention(q, k, v, 0, segments), output)
            extension.attention_forward.assert_called_once_with(q, k, v, 0, segments)
            for tensor in (q, k, v):
                tensor.requires_grad = True
                load.reset_mock()
                with self.assertRaisesRegex(NotImplementedError, 'backward'):
                    student.attention(q, k, v)
                load.assert_not_called()
                with torch.no_grad():
                    self.assertIs(student.attention(q, k, v, 9), output)
                extension.attention_forward.assert_called_with(q, k, v, 9, None)
                tensor.requires_grad = False


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA and nvcc')
class StudentAttentionCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = load_attention_extension()

    def setUp(self):
        torch.manual_seed(42)

    @torch.no_grad()
    def check_forward(self, q, k, v, past_len=0, segments=None):
        # SDPA backends may assume aligned bases even for contiguous tensors.
        # clone (not just contiguous) gives the oracle fresh aligned storage;
        # the candidate still receives the original views below.
        originals = [x.clone(memory_format=torch.contiguous_format) for x in (q, k, v)]
        actual, lse = self.extension.attention_forward(q, k, v, past_len, segments)
        torch.cuda.current_stream(q.device).synchronize()
        # FP64 oracle avoids TF32 settings affecting the reference.
        scores = q.double() @ k.double().transpose(-1, -2) / math.sqrt(64)
        allowed = reference.causal_mask(q, k, past_len, segments)
        scores = scores.masked_fill(~allowed, -float('inf'))
        expected = scores.softmax(-1) @ v.double()
        self.assertEqual(actual.dtype, q.dtype)
        self.assertEqual(lse.dtype, torch.float32)
        if q.dtype == torch.bfloat16:
            # P is rounded to BF16 before PV, while LSE stays in FP32.
            torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=3e-2)
            torch.testing.assert_close(lse, scores.logsumexp(-1).float(), atol=2e-4, rtol=2e-5)
            sdpa = reference.sdpa_attention(*originals, past_len=past_len)
            torch.cuda.current_stream(q.device).synchronize()
            torch.testing.assert_close(actual, sdpa, atol=2e-2, rtol=3e-2)
        else:
            torch.testing.assert_close(actual, expected.float(), atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(lse, scores.logsumexp(-1).float(), atol=3e-5, rtol=3e-5)
        self.assertEqual(lse.shape, q.shape[:-1])
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.isfinite(actual).all())
        for x, original in zip((q, k, v), originals):
            torch.testing.assert_close(x, original, atol=0, rtol=0)
        public = student.attention(q, k, v, past_len, segments)
        torch.testing.assert_close(public, actual, atol=0, rtol=0)
        return actual

    def require_bf16(self):
        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest('BF16 MMA/cp.async require SM80 or later')

    def test_bf16_tiles_tails_and_cache(self):
        self.require_bf16()
        for tq, tk in ((1, 1), (17, 17), (31, 31), (32, 32), (33, 33),
                       (63, 63), (64, 64), (65, 65), (127, 127), (128, 128), (129, 129),
                       (193, 193), (257, 257), (512, 512), (1, 513),
                       (7, 127), (7, 128), (7, 129), (7, 193), (65, 193)):
            with self.subTest(tq=tq, tk=tk):
                q = torch.randn(2, 3, tq, 64, device='cuda', dtype=torch.bfloat16)
                stores = [torch.full((2, 3, tk + 64, 64), float('nan'),
                                     device='cuda', dtype=torch.bfloat16) for _ in range(2)]
                for store in stores:
                    store[:, :, :tk].normal_()
                self.check_forward(q, stores[0][:, :, :tk], stores[1][:, :, :tk], tk - tq)

    def test_bf16_feature_subtiles_and_full_output_rescale(self):
        self.require_bf16()
        # Each probe isolates a QK feature on either side of a BH/MMA boundary.
        # Increasing key scores force alpha != 1 across BK=64 tiles, while V
        # varies in both token and output-feature dimensions (including d>=32).
        key_codes = torch.arange(65, device='cuda', dtype=torch.bfloat16) / 32
        feature_codes = (torch.arange(64, device='cuda', dtype=torch.bfloat16) - 32) / 16
        v = (key_codes[:, None] * feature_codes[None, :]).view(1, 1, 65, 64)
        for feature in (0, 15, 16, 31, 32, 47, 48, 63):
            with self.subTest(feature=feature):
                q = torch.zeros_like(v)
                k = torch.zeros_like(v)
                q[..., feature] = 4
                k[..., feature] = key_codes * 8
                self.check_forward(q, k, v)

    def test_bf16_full_tile_mask_boundary(self):
        self.require_bf16()
        # At past=62, key 63 is future for query 0; at past=63 it becomes
        # valid for every row of the first full query tile (inclusive boundary).
        # Tq=63/65 also forces the padded-query path; Tk crosses BK boundaries.
        for tq in (63, 64, 65, 128):
            for past in (0, 30, 31, 32, 33, 62, 63, 64, 65):
                with self.subTest(tq=tq, past=past):
                    tk = tq + past
                    q, k = [torch.zeros(1, 1, t, 64, device='cuda', dtype=torch.bfloat16)
                            for t in (tq, tk)]
                    v = torch.zeros_like(k)
                    # Spikes expose a one-key leak that random inputs may hide
                    # behind the BF16 output tolerance.
                    v[:, :, 31].fill_(64)
                    if tk > 63:
                        v[:, :, 63].fill_(-64)
                    v[:, :, -1].add_(32)
                    before = self.check_forward(q, k, v, past)
                    changed = v.clone()
                    changed[:, :, past + 1:].add_(16)
                    after = self.check_forward(q, k, changed, past)
                    torch.testing.assert_close(before[:, :, :1], after[:, :, :1], atol=0, rtol=0)

    def test_bf16_key_reduction_spans_both_halves_of_tile(self):
        self.require_bf16()
        # Probe both halves of the key tile and both halves of the output.
        q = torch.zeros(1, 1, 64, 64, device='cuda', dtype=torch.bfloat16)
        k = torch.zeros(1, 1, 129, 64, device='cuda', dtype=torch.bfloat16)
        for key_index in (0, 15, 16, 31, 32, 47, 48, 63, 64, 95, 96, 127, 128):
            with self.subTest(key_index=key_index):
                v = torch.zeros_like(k)
                v[:, :, key_index, :32] = 64
                v[:, :, key_index, 32:] = -64
                self.check_forward(q, k, v, 65)

    def test_bf16_full_width_kv_pipeline_poisoned_tails(self):
        self.require_bf16()
        for tq, tk in ((2, 63), (2, 64), (2, 65), (64, 64), (65, 129), (129, 257)):
            with self.subTest(tq=tq, tk=tk):
                q = torch.randn(1, 1, tq, 64, device='cuda', dtype=torch.bfloat16)
                stores = [torch.full((1, 1, tk + 64, 64), float('nan'),
                                     device='cuda', dtype=torch.bfloat16) for _ in range(2)]
                for storage in stores:
                    storage[:, :, :tk].normal_()
                k, v = [storage[:, :, :tk] for storage in stores]
                # Singleton batch/head keeps these prefixes contiguous: the
                # host uses the original allocation with NaNs directly after Tk.
                self.assertTrue(k.is_contiguous() and v.is_contiguous())
                self.check_forward(q, k, v, tk - tq)

    def test_bf16_causal_tile_order_and_pipeline_transitions(self):
        self.require_bf16()
        # Distinct head offsets expose remapped blocks writing to a wrong head.
        # V=token_index also makes every causal prefix produce a different mean.
        for tq, tk in ((2, 2), (33, 33), (64, 64), (65, 65),
                       (129, 161), (193, 225)):
            with self.subTest(tq=tq, tk=tk):
                q = torch.zeros(2, 3, tq, 64, device='cuda', dtype=torch.bfloat16)
                k = torch.zeros(2, 3, tk, 64, device='cuda', dtype=torch.bfloat16)
                offsets = torch.arange(6, device='cuda').view(2, 3, 1, 1) / 8
                tokens = torch.arange(tk, device='cuda').view(1, 1, tk, 1) / 256
                v = (offsets + tokens).expand(2, 3, tk, 64).to(torch.bfloat16).contiguous()
                self.check_forward(q, k, v, tk - tq)

    def test_bf16_unaligned_and_strided_views(self):
        self.require_bf16()
        count = 2 * 3 * 65 * 64
        views = [torch.randn(count + 1, device='cuda', dtype=torch.bfloat16)[1:]
                 .view(2, 3, 65, 64) for _ in range(3)]
        for view in views:
            self.assertTrue(view.is_contiguous())
            self.assertNotEqual(view.data_ptr() % 16, 0)
        self.check_forward(*views)
        views = [torch.randn(2, 3, 65, 128, device='cuda', dtype=torch.bfloat16)[..., ::2]
                 for _ in range(3)]
        self.check_forward(*views)
        views = [torch.randn(1, 1, 65, 64, device='cuda', dtype=torch.bfloat16)
                 .expand(2, 3, -1, -1) for _ in range(3)]
        self.check_forward(*views)

    def test_bf16_decode_split_boundaries_and_rescaling(self):
        self.require_bf16()
        # One-pass, split/merge, tail partitions, and the generic-kernel fallback.
        for tk in (1, 2, 7, 127, 128, 129, 255, 256, 257, 511, 512, 513,
                   4095, 4096, 4097):
            with self.subTest(tk=tk):
                q = torch.randn(2, 3, 1, 64, device='cuda', dtype=torch.bfloat16)
                k, v = [torch.randn(2, 3, tk, 64, device='cuda', dtype=torch.bfloat16)
                        for _ in range(2)]
                self.check_forward(q, k, v, tk - 1)
        # Different partition maxima must be rescaled before combining sums and
        # numerators; averaging partition outputs or exponentiating raw scores fails.
        q = torch.ones(2, 3, 1, 64, device='cuda', dtype=torch.bfloat16)
        k, v = [torch.randn(2, 3, 257, 64, device='cuda', dtype=torch.bfloat16)
                for _ in range(2)]
        k[:, :, :128].fill_(-10)
        k[:, :, 128:256].fill_(10)
        self.check_forward(q, k, v, 256)
        q.zero_()
        v.fill_(1)
        actual = self.check_forward(q, k, v, 256)
        torch.testing.assert_close(actual, torch.ones_like(actual), atol=0, rtol=0)

    def test_bf16_exact_output_large_logits_and_causality(self):
        self.require_bf16()
        q = torch.zeros(2, 3, 193, 64, device='cuda', dtype=torch.bfloat16)
        k, v = torch.zeros_like(q), torch.ones_like(q)
        actual = self.check_forward(q, k, v)
        torch.testing.assert_close(actual, v, atol=0, rtol=0)
        q.fill_(10)
        k.fill_(10)
        v.normal_()
        self.check_forward(q, k, v)
        q.normal_()
        k.normal_()
        before = self.check_forward(q, k, v)
        k[:, :, 65:].add_(4)
        v[:, :, 65:].add_(4)
        after = self.check_forward(q, k, v)
        torch.testing.assert_close(before[:, :, :65], after[:, :, :65], atol=0, rtol=0)

    @torch.no_grad()
    def test_bf16_softmax_natural_lse(self):
        self.require_bf16()
        # Nonzero positive/negative logits verify natural-log LSE for either
        # exponential implementation (and catch an incorrect base conversion).
        # Chunk past=63 crosses full-tile/masked paths; Tq=65 has padded rows.
        for tq, tk in ((64, 127), (65, 193)):
            for key_value in (-10., -0.5, 0.5, 10.):
                with self.subTest(tq=tq, tk=tk, key_value=key_value):
                    q = torch.full((1, 2, tq, 64), 0.25,
                                   device='cuda', dtype=torch.bfloat16)
                    k = torch.full((1, 2, tk, 64), key_value,
                                   device='cuda', dtype=torch.bfloat16)
                    v = torch.ones_like(k)
                    self.check_forward(q, k, v, tk - tq)
                    out, lse = self.extension.attention_forward(q, k, v, tk - tq)
                    torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)
                    expected = 2 * key_value + torch.arange(
                        tk - tq + 1, tk + 1, device='cuda').float().log()
                    torch.testing.assert_close(lse, expected.expand_as(lse), atol=2e-4, rtol=2e-5)

    @torch.no_grad()
    def test_bf16_current_stream_with_alignment_copies(self):
        self.require_bf16()
        for tq, tk in ((65, 65), (1, 257)):
            with self.subTest(tq=tq, tk=tk):
                inputs = [torch.zeros(2 * 3 * t * 64 + 1, device='cuda', dtype=torch.bfloat16)[1:]
                          .view(2, 3, t, 64) for t in (tq, tk, tk)]
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(10_000_000)
                    for x, value in zip(inputs, (0.25, 0.5, 2.0)):
                        x.fill_(value)
                    actual, lse = self.extension.attention_forward(*inputs, tk - tq)
                    actual, lse = actual.clone(), lse.clone()
                stream.synchronize()
                self.assertEqual(lse.dtype, torch.float32)
                torch.testing.assert_close(actual, torch.full_like(actual, 2.0), atol=0, rtol=0)
                expected_lse = 1.0 + torch.arange(tk - tq + 1, tk + 1, device='cuda').float().log()
                torch.testing.assert_close(lse, expected_lse.expand_as(lse), atol=2e-4, rtol=2e-5)

    def test_bf16_validation(self):
        inputs = [torch.randn(2, 3, 7, 64, device='cuda', dtype=torch.bfloat16)
                  for _ in range(3)]
        ids = torch.zeros(2, 7, dtype=torch.long, device='cuda')
        with self.assertRaisesRegex(RuntimeError, 'segment_ids is not supported'):
            self.extension.attention_forward(*inputs, 0, ids)
        for dtype in (torch.float32, torch.bfloat16):
            for index in range(3):
                with self.subTest(dtype=dtype, index=index):
                    mixed = [x.to(dtype) for x in inputs]
                    other = torch.bfloat16 if dtype == torch.float32 else torch.float32
                    mixed[index] = mixed[index].to(other)
                    with self.assertRaisesRegex(RuntimeError, 'same dtype'):
                        self.extension.attention_forward(*mixed)
        # Expanded views exercise overflow checks without allocating a huge tensor.
        for shape in ((2**31, 1, 1, 64), (65536, 1, 64 * 32768, 64)):
            big = torch.zeros(1, 1, 1, 64, device='cuda', dtype=torch.bfloat16).expand(shape)
            with self.assertRaisesRegex(RuntimeError, 'grid limits'):
                self.extension.attention_forward(big, big, big)
        if torch.cuda.get_device_capability()[0] < 8:
            with self.assertRaisesRegex(RuntimeError, 'SM80'):
                self.extension.attention_forward(*inputs)

    @torch.no_grad()
    def test_bf16_cuda_graph_replay(self):
        self.require_bf16()
        for tq, tk in ((65, 193), (1, 193)):
            with self.subTest(tq=tq, tk=tk):
                # Capture also covers alignment copies and output allocations.
                inputs = [torch.randn(2 * 3 * t * 64 + 1, device='cuda', dtype=torch.bfloat16)[1:]
                          .view(2, 3, t, 64) for t in (tq, tk, tk)]
                expected = self.check_forward(*inputs, past_len=tk - tq)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        _, expected_lse = self.extension.attention_forward(*inputs, tk - tq)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual, lse = self.extension.attention_forward(*inputs, tk - tq)
                for _ in range(2):
                    actual.fill_(float('nan'))
                    lse.fill_(float('nan'))
                    graph.replay()
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)

    def test_bf16_rejects_sequence_index_overflow_before_copy(self):
        # Zero-stride views require only 64 BF16 values. The host must reject
        # lengths before casting to int32 or materializing these huge views.
        seed = torch.zeros(1, 1, 1, 64, device='cuda', dtype=torch.bfloat16)
        for tq, tk in ((2**31, 2**31), (2, 2**31), (65, 2**31 + 64)):
            with self.subTest(tq=tq, tk=tk):
                q = seed.expand(1, 1, tq, 64)
                kv = seed.expand(1, 1, tk, 64)
                with self.assertRaisesRegex(RuntimeError, 'sequence lengths must fit int32'):
                    self.extension.attention_forward(q, kv, kv, tk - tq)

    def test_prefill_tiles_and_sequence_tails(self):
        for time in (1, 7, 31, 32, 33, 65, 96):
            with self.subTest(time=time):
                q, k, v = [torch.randn(2, 3, time, 64, device='cuda') for _ in range(3)]
                self.check_forward(q, k, v)

    def test_qkv_views_and_feature_strides(self):
        storage = torch.randn(2 * 65 * 3 * 3 * 64 + 7, device='cuda')
        qkv = storage[7:].view(2, 65, 3, 3, 64)
        q, k, v = [x.transpose(1, 2) for x in qkv.unbind(2)]
        self.check_forward(q, k, v)
        views = [torch.randn(2, 3, 33, 129, device='cuda')[..., 1::2] for _ in range(3)]
        self.check_forward(*views)
        # Zero strides are valid too; in particular V must not assume contiguous rows.
        views = [torch.randn(1, 1, 33, 64, device='cuda').expand(2, 3, -1, -1)
                 for _ in range(3)]
        self.check_forward(*views)

    def test_decode_chunk_and_cache_capacity_strides(self):
        for tq, tk in ((1, 1), (1, 65), (7, 65), (33, 96)):
            with self.subTest(tq=tq, tk=tk):
                q = torch.randn(2, 3, tq, 64, device='cuda')
                stores = [torch.full((2, 3, 128, 64), float('nan'), device='cuda')
                          for _ in range(2)]
                for store in stores:
                    store[:, :, :tk].normal_()
                k, v = [store[:, :, :tk] for store in stores]
                self.check_forward(q, k, v, tk - tq)

    def test_segments_and_causal_isolation(self):
        q, k, v = [torch.randn(2, 3, 65, 64, device='cuda') for _ in range(3)]
        backing = torch.empty(2, 130, dtype=torch.long, device='cuda')
        segments = backing[:, 1::2]
        segments.copy_(torch.arange(65, device='cuda')[None, :] // 17)
        before_ids = segments.clone()
        before = self.check_forward(q, k, v, segments=segments)
        k2, v2 = k.clone(), v.clone()
        k2[:, :, :17].add_(10)
        v2[:, :, :17].add_(10)
        after = self.check_forward(q, k2, v2, segments=segments)
        torch.testing.assert_close(before[:, :, 17:], after[:, :, 17:], atol=0, rtol=0)
        k2, v2 = k.clone(), v.clone()
        k2[:, :, 33:].add_(10)
        v2[:, :, 33:].add_(10)
        after = self.check_forward(q, k2, v2, segments=segments)
        torch.testing.assert_close(before[:, :, :33], after[:, :, :33], atol=0, rtol=0)
        torch.testing.assert_close(segments, before_ids, atol=0, rtol=0)

    def test_large_logits_remain_finite(self):
        q, k, v = [torch.randn(2, 2, 65, 64, device='cuda') for _ in range(3)]
        # Identical keys keep the large-score case well conditioned while still
        # overflowing exp(score) if the row maximum is not subtracted.
        q.fill_(10)
        k.fill_(10)
        self.check_forward(q, k, v)

    def test_native_validation_and_grad_guards(self):
        q, k, v = [torch.randn(2, 3, 7, 64, device='cuda') for _ in range(3)]
        ids = torch.zeros(2, 7, dtype=torch.long, device='cuda')
        cases = [
            ((q.cpu(), k, v), 'CUDA'),
            ((q.half(), k, v), 'float32'),
            ((q.to_sparse(), k, v), 'strided'),
            ((q[0], k, v), '4D'),
            ((q[..., :32], k[..., :32], v[..., :32]), 'head_dim=64'),
            ((q[:, :, :0], k, v), 'positive'),
            ((q, k[:1], v), 'first dimension'),
            ((q, k[:, :2], v), 'second dimension'),
            ((q, k, v[:, :, :6]), 'sequence length'),
            ((q, k, v, -1), 'non-negative'),
            ((q, k, v, 1), 'Tk == past_len'),
            ((q, k, v, 0, ids.float()), 'int64'),
            ((q, k, v, 0, ids.cpu()), 'CUDA'),
            ((q, k, v, 0, ids[:, :6]), 'shape'),
            ((q[:, :, :1], k, v, 6, ids[:, :1]), 'without past'),
        ]
        for args, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.extension.attention_forward(*args)
        for i in range(3):
            inputs = [q.detach(), k.detach(), v.detach()]
            inputs[i].requires_grad_(True)
            with self.assertRaisesRegex(RuntimeError, 'backward'):
                self.extension.attention_forward(*inputs)
            with self.assertRaisesRegex(NotImplementedError, 'backward'):
                student.attention(*inputs)
            with torch.no_grad():
                self.assertFalse(student.attention(*inputs).requires_grad)
        with torch.autocast('cuda'), self.assertRaisesRegex(RuntimeError, 'autocast'):
            student.attention(q, k, v)

    @torch.no_grad()
    def test_current_stream_including_contiguous_copies(self):
        inputs = [torch.zeros(2, 3, 33, 128, device='cuda')[..., ::2] for _ in range(3)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            for x, value in zip(inputs, (0.25, 0.5, 2.0)):
                x.fill_(value)
            actual, lse = self.extension.attention_forward(*inputs)
            actual, lse = actual.clone(), lse.clone()
        stream.synchronize()
        torch.testing.assert_close(actual, torch.full_like(actual, 2.0), atol=1e-6, rtol=1e-6)
        expected_lse = 1.0 + torch.arange(1, 34, device='cuda').float().log()
        torch.testing.assert_close(lse, expected_lse.expand_as(lse), atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
