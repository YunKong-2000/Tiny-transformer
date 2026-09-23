import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tiny_transformer.data import PackedDataset
from tiny_transformer.prepare import prepare
from tiny_transformer.runtime import load_checkpoint
from tiny_transformer.tokenizer import Tokenizer
from tiny_transformer.train import run


ROOT = Path(__file__).resolve().parents[1]


def prepare_smoke(path):
    prepare(argparse.Namespace(output=str(path), source="smoke", train_docs=30, val_docs=10, tokenizer_docs=20,
                               tokenizer="byte", vocab_size=8192, seed=42, keep_text=False))


class DataTests(unittest.TestCase):
    def test_unicode_tokenizer_roundtrip(self):
        tokenizer = Tokenizer({"kind": "byte"})
        text = "你好，CuTe! 🐈"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text, bos=True, eos=True)), text)

    def test_packing_shift_and_isolated_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Tokenizer({"kind": "byte"}).save(root / "tokenizer.json")
            (root / "metadata.json").write_text(json.dumps({"dtype": "<u2"}))
            tokens = np.array([1, 10, 2, 1, 11, 2, 1], dtype="<u2")
            tokens.tofile(root / "train.bin")
            data = PackedDataset(root, "train", 6, packing="isolated")
            batch = data.batch(1, "cpu")
            self.assertEqual(batch["ids"].tolist(), [[1, 10, 2, 1, 11, 2]])
            self.assertEqual(batch["targets"].tolist(), [[10, 2, -100, 11, 2, -100]])
            self.assertEqual(batch["segment_ids"].tolist(), [[0, 0, 0, 1, 1, 1]])
            self.assertEqual(batch["position_ids"].tolist(), [[0, 1, 2, 0, 1, 2]])
            continuous = PackedDataset(root, "train", 6).batch(1, "cpu")
            self.assertEqual(continuous["targets"].tolist(), [[10, 2, 1, 11, 2, 1]])
            self.assertIsNone(continuous["segment_ids"])

    def test_preparation_is_reproducible_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            prepare_smoke(root / "a")
            prepare_smoke(root / "b")
            for file in ("train.bin", "val.bin", "tokenizer.json", "metadata.json"):
                self.assertEqual((root / "a" / file).read_bytes(), (root / "b" / file).read_bytes())
            with self.assertRaises(FileExistsError):
                prepare_smoke(root / "a")

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            prepare_smoke(root / "data")

            def args(output, stop_after=None, resume=None):
                return argparse.Namespace(config=str(ROOT / "configs/smoke.json"), data=str(root / "data"), output=str(output),
                                          steps=4, batch_size=None, grad_accum=None, seq_len=None, precision=None, packing=None,
                                          op=[], device="cpu", tf32=False, compile=False, stop_after=stop_after, resume=resume)

            run(args(root / "full"))
            run(args(root / "resumed", stop_after=2))
            run(args(root / "resumed", resume=str(root / "resumed/last.pt")))
            full, resumed = load_checkpoint(root / "full/last.pt"), load_checkpoint(root / "resumed/last.pt")
            self.assertEqual(full["step"], resumed["step"])
            for name in full["model"]:
                torch.testing.assert_close(full["model"][name], resumed["model"][name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
