import json
from pathlib import Path

import numpy as np
import torch

from .tokenizer import Tokenizer


class PackedDataset:
    """Random contiguous windows in a token stream; exactly one next-token shift.

    Continuous mode allows cross-document context. Isolated mode resets positions,
    supplies segment ids, and masks targets crossing EOS into a new document.
    """
    def __init__(self, directory, split, seq_len, seed=42, packing="continuous"):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        self.metadata = metadata
        if packing not in ("continuous", "isolated") or seq_len <= 0:
            raise ValueError("invalid packing or seq_len")
        self.tokens = np.memmap(directory / f"{split}.bin", dtype=metadata["dtype"], mode="r")
        if len(self.tokens) <= seq_len:
            raise ValueError(f"{split} needs at least seq_len + 1 tokens")
        self.tokenizer = Tokenizer.load(directory / "tokenizer.json")
        self.seq_len = seq_len
        self.packing = packing
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size, device):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        starts = self.rng.integers(0, len(self.tokens) - self.seq_len, size=batch_size)
        windows = np.stack([self.tokens[start:start + self.seq_len + 1] for start in starts]).astype(np.int64)
        ids = torch.from_numpy(windows[:, :-1].copy())
        targets = torch.from_numpy(windows[:, 1:].copy())
        segments, positions = None, None
        if self.packing == "isolated":
            boundary = torch.zeros_like(ids, dtype=torch.bool)
            boundary[:, 0] = True
            boundary[:, 1:] = ids[:, :-1] == self.tokenizer.eos_id
            segments = boundary.long().cumsum(dim=-1) - 1
            indices = torch.arange(self.seq_len)[None, :].expand_as(ids)
            last_start = torch.where(boundary, indices, 0).cummax(dim=-1).values
            positions = indices - last_start
            targets[ids == self.tokenizer.eos_id] = -100
        result = {"ids": ids, "targets": targets, "segment_ids": segments, "position_ids": positions}
        return {name: None if value is None else value.to(device) for name, value in result.items()}
