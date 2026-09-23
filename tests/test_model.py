import unittest

import torch

from tiny_transformer.config import ModelConfig
from tiny_transformer.model import Transformer


def small_model(backend="reference"):
    return Transformer(ModelConfig(vocab_size=31, dim=32, n_layers=2, n_heads=4, hidden_dim=48, max_seq_len=32, operators={"attention": backend}))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_shapes_and_tied_weights(self):
        model = small_model()
        self.assertIs(model.embedding, model.output_weight)
        ids = torch.randint(0, 31, (2, 7))
        self.assertEqual(model(ids).shape, (2, 7, 31))
        torch.testing.assert_close(model(ids, last_only=True), model(ids)[:, -1:, :])

    def test_default_parameter_count(self):
        with torch.device("meta"):
            model = Transformer(ModelConfig())
        self.assertEqual(model.parameter_count(), 62927616)

    def test_future_tokens_do_not_change_past_logits(self):
        model = small_model().eval()
        ids = torch.randint(0, 31, (2, 9))
        changed = ids.clone()
        changed[:, 5:] = torch.randint(0, 31, (2, 4))
        torch.testing.assert_close(model(ids)[:, :5], model(changed)[:, :5])

    def test_isolated_documents_do_not_leak(self):
        for backend in ("reference", "sdpa"):
            model = small_model(backend).eval()
            ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
            changed = torch.tensor([[8, 9, 10, 4, 5, 6]])
            segments = torch.tensor([[0, 0, 0, 1, 1, 1]])
            positions = torch.tensor([[0, 1, 2, 0, 1, 2]])
            a = model(ids, segment_ids=segments, position_ids=positions)
            b = model(changed, segment_ids=segments, position_ids=positions)
            torch.testing.assert_close(a[:, 3:], b[:, 3:])

    @torch.inference_mode()
    def test_cache_matches_full_forward_with_chunks_and_reset(self):
        for backend in ("reference", "sdpa"):
            model = small_model(backend).eval()
            ids = torch.randint(0, 31, (2, 11))
            full = model(ids)
            cache = model.new_cache(2, 16)
            parts = [model(ids[:, :4], cache=cache), model(ids[:, 4:7], cache=cache)]
            parts.extend(model(ids[:, index:index + 1], cache=cache) for index in range(7, 11))
            torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=2e-6, rtol=1e-5)
            self.assertEqual(cache.length, 11)
            self.assertEqual(cache.allocated_bytes(), 2 * 2 * 2 * 4 * 16 * 8 * 4)
            pointer = cache.keys[0].data_ptr()
            cache.reset()
            torch.testing.assert_close(model(ids, cache=cache), full)
            self.assertEqual(pointer, cache.keys[0].data_ptr())

    def test_cache_requires_inference(self):
        model = small_model()
        ids = torch.ones(1, 2, dtype=torch.long)
        with self.assertRaises(ValueError):
            model(ids, cache=model.new_cache(1))
        model.eval()
        with self.assertRaises(ValueError):
            model(ids, cache=model.new_cache(1))

    @torch.inference_mode()
    def test_cache_rejects_overflow_before_writing(self):
        model = small_model().eval()
        cache = model.new_cache(1, 2)
        model(torch.ones(1, 2, dtype=torch.long), cache=cache)
        with self.assertRaises(ValueError):
            model(torch.ones(1, 1, dtype=torch.long), cache=cache)
        self.assertEqual(cache.length, 2)

    def test_training_reduces_fixed_batch_loss(self):
        model = small_model()
        ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        targets = torch.tensor([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        initial = float(model.loss(ids, targets).detach())
        for _ in range(15):
            optimizer.zero_grad()
            loss = model.loss(ids, targets)
            loss.backward()
            for parameter in model.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        self.assertLess(float(model.loss(ids, targets).detach()), initial * 0.6)

    def test_compile_eager_backend_preserves_results(self):
        model = small_model().eval()
        ids = torch.randint(0, 31, (1, 4))
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(ids), model(ids))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA; run scripts/a100_validate.sh")
    @torch.inference_mode()
    def test_cuda_bf16_cache_and_sdpa(self):
        model = small_model("sdpa").cuda().bfloat16().eval()
        ids = torch.randint(0, 31, (2, 9), device="cuda")
        expected = model(ids)
        cache = model.new_cache(2, 16)
        actual = torch.cat([model(ids[:, :5], cache=cache), model(ids[:, 5:], cache=cache)], dim=1)
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.03)


if __name__ == "__main__":
    unittest.main()
