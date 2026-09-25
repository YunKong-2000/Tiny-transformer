"""Your CUDA / CuTe / CUTLASS implementation entry points.

Embedding supports contiguous CUDA FP32 forward/backward; other entries are unfinished.
There is no silent reference fallback.
Match reference.py semantics, device, shape, dtype, strides and gradients.
Use --op NAME=student to enable only a completed operator.
See docs/development.md before registering a compiled/custom operator.
"""

import torch
from torch.autograd.function import once_differentiable

from ._extension import load_embedding_extension


def _todo(name):
    raise NotImplementedError(
        f"Student operator '{name}' is not implemented. "
        f"Implement tiny_transformer/operators/student.py::{name} or select reference."
    )


class _Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, weight):
        # Function.forward runs with grad mode disabled; the raw pybind call
        # supplies values while Function.apply creates the autograd node.
        output = load_embedding_extension().embedding_forward(ids, weight)
        ctx.save_for_backward(ids)
        ctx.vocab_size = weight.shape[0]
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (ids,) = ctx.saved_tensors
        # E.g. output.sum().backward() supplies an expanded, zero-stride tensor.
        dweight = load_embedding_extension().embedding_backward(
            ids, grad_output.contiguous(), ctx.vocab_size
        )
        # One result per forward argument: IDs are discrete, weight is trainable.
        return None, dweight


def embedding(ids, weight):
    """Gather [B,T] int64 IDs from [V,H] FP32 CUDA weights; first-order autograd."""
    if not ids.is_cuda or not weight.is_cuda:
        raise RuntimeError("student embedding requires ids and weight to be CUDA tensors")
    if torch.is_grad_enabled() and weight.requires_grad:
        return _Embedding.apply(ids, weight)
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
