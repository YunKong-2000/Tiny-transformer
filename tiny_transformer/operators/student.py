"""Your CUDA / CuTe / CUTLASS implementation entry points.

Embedding supports contiguous CUDA FP32 forward only; other entries are unfinished.
There is no silent reference fallback.
Match reference.py semantics, device, shape, dtype, strides and gradients.
Use --op NAME=student to enable only a completed operator.
See docs/development.md before registering a compiled/custom operator.
"""

import torch

from ._extension import load_embedding_extension


def _todo(name):
    raise NotImplementedError(
        f"Student operator '{name}' is not implemented. "
        f"Implement tiny_transformer/operators/student.py::{name} or select reference."
    )


def embedding(ids, weight):
    """Gather [B,T] int64 IDs from [V,H] FP32 CUDA weights; no backward yet."""
    if not ids.is_cuda or not weight.is_cuda:
        raise RuntimeError("student embedding requires ids and weight to be CUDA tensors")
    if torch.is_grad_enabled() and weight.requires_grad:
        raise RuntimeError(
            "student embedding is forward-only; use torch.no_grad() or "
            "torch.inference_mode(), or select the reference backend for training"
        )
    return load_embedding_extension().embedding_forward(ids, weight)


def linear(x, weight):
    return _todo("linear")


def rms_norm(x, weight, eps):
    return _todo("rms_norm")


def rope(x, cos, sin):
    return _todo("rope")


def attention(q, k, v, past_len=0, segment_ids=None):
    return _todo("attention")


def swiglu(gate, up):
    return _todo("swiglu")


def residual(x, update):
    return _todo("residual")


def cross_entropy(logits, targets):
    return _todo("cross_entropy")
