"""Your CUDA / CuTe / CUTLASS implementation entry points.

Embedding supports contiguous CUDA FP32 forward/backward; RMSNorm supports FP32 forward.
There is no silent reference fallback.
Match reference.py semantics, device, shape, dtype, strides and gradients.
Use --op NAME=student to enable only a completed operator.
See docs/development.md before registering a compiled/custom operator.
"""

import torch
from torch.autograd.function import once_differentiable

from ._extension import load_embedding_extension, load_rms_norm_extension


def _todo(name):
    raise NotImplementedError(
        f"Student operator '{name}' is not implemented. "
        f"Implement tiny_transformer/operators/student.py::{name} or select reference."
    )


class _Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, weight, backward_impl):
        # Function.forward runs with grad mode disabled; the raw pybind call
        # supplies values while Function.apply creates the autograd node.
        output = load_embedding_extension().embedding_forward(ids, weight)
        ctx.save_for_backward(ids)
        ctx.vocab_size = weight.shape[0]
        ctx.backward_impl = backward_impl
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (ids,) = ctx.saved_tensors
        # E.g. output.sum().backward() supplies an expanded, zero-stride tensor.
        dweight = load_embedding_extension().embedding_backward(
            ids, grad_output.contiguous(), ctx.vocab_size, ctx.backward_impl
        )
        # IDs and implementation selector are not differentiable.
        return None, dweight, None


def embedding(ids, weight, *, backward_impl="grouped"):
    """Gather [B,T] int64 IDs from [V,H] FP32 CUDA weights; first-order autograd."""
    if backward_impl not in ("grouped", "baseline"):
        raise ValueError("embedding backward_impl must be 'grouped' or 'baseline'")
    if not ids.is_cuda or not weight.is_cuda:
        raise RuntimeError("student embedding requires ids and weight to be CUDA tensors")
    if torch.is_grad_enabled() and weight.requires_grad:
        return _Embedding.apply(ids, weight, backward_impl)
    return load_embedding_extension().embedding_forward(ids, weight)


def linear(x, weight):
    return _todo("linear")


def rms_norm(x, weight, eps):
    """CUDA FP32 [..., H] RMSNorm forward; strided inputs are explicitly copied.

    No backward yet: use no_grad/inference_mode for parameters requiring grad.
    The native entry point repeats validation and accepts only contiguous inputs.
    """
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("student rms_norm requires x and weight to be CUDA tensors")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise RuntimeError(
            "student rms_norm backward is not implemented; "
            "use torch.no_grad()/torch.inference_mode() for forward-only inference"
        )
    # Covers last-only prefill (batch stride > H), transposes and x[..., ::2].
    # Copies are part of end-to-end operator timing; there is no reference fallback.
    return load_rms_norm_extension().rms_norm_forward(
        x.contiguous(), weight.contiguous(), eps
    )


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
