"""Embedding distributions and compatibility adapter to the common benchmark."""
import sys

import torch

from ..operators import reference, student
from .common import measure_pair, prepare_calls as prepare_operator_calls


PATTERNS = ("random", "same", "unique", "hot")


def make_ids(pattern, batch, length, vocab_size, hot_tokens, device):
    if pattern == "random":
        return torch.randint(vocab_size, (batch, length), device=device)
    if pattern == "same":
        return torch.zeros(batch, length, device=device, dtype=torch.int64)
    if pattern == "unique":
        if batch * length > vocab_size:
            raise ValueError("unique IDs require batch_size * seq_length <= vocab_size")
        return torch.arange(batch * length, device=device).reshape(batch, length)
    if pattern == "hot":
        return torch.randint(min(hot_tokens, vocab_size), (batch, length), device=device)
    raise ValueError(f"unknown pattern: {pattern}")



def prepare_calls(ids, weight, upstream, candidate=student.embedding):
    calls, errors = prepare_operator_calls(
        reference.embedding, candidate, (ids, weight), (1,),
        upstream=upstream, exact_forward=True)
    # Preserve the historical embedding helper's single-gradient return value.
    left, right = calls["backward"]
    calls["backward"] = (lambda: left()[0], lambda: right()[0])
    return calls, errors


def main(argv=None):
    from .operators import main as operator_main
    operator_main(["--operator", "embedding", "--workloads", "prefill",
                   "--output", "runs/embedding-performance.json"] +
                  (sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
