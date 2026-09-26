"""Your CUDA / CuTe / CUTLASS implementation entry points.

Embedding supports contiguous CUDA FP32 forward/backward; RMSNorm supports FP32 forward/backward (backward H <= 1024).
Residual supports same-shape FP32/FP16/BF16 inputs via FP32 kernels and explicit casts.
Cross entropy supports FP32/FP16/BF16 logits, ignored labels and first-order autograd.
There is no silent reference fallback.
Match reference.py semantics, device, shape, dtype, strides and gradients.
Use --op NAME=student to enable only a completed operator.
See docs/development.md before registering a compiled/custom operator.
"""

import torch
from torch.autograd.function import once_differentiable

from ._extension import (
    load_cross_entropy_extension, load_embedding_extension,
    load_residual_extension, load_rms_norm_extension,
)


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


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        # apply() disables grad mode here; pybind itself does not create a graph.
        output, inv_rms = load_rms_norm_extension().rms_norm_forward(x, weight, eps)
        ctx.save_for_backward(x, weight, inv_rms)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight, inv_rms = ctx.saved_tensors
        dx, dweight = load_rms_norm_extension().rms_norm_backward(
            x, grad_output.contiguous(), weight, inv_rms
        )
        return (dx if ctx.needs_input_grad[0] else None,
                dweight if ctx.needs_input_grad[1] else None, None)


def rms_norm(x, weight, eps):
    """CUDA FP32 [..., H] RMSNorm with first-order autograd for 0 < H <= 1024.

    Strided inputs/upstream gradients are explicitly copied. Raw forward returns
    (Y, inv_rms); the public operator returns only Y and autograd owns inv_rms.
    """
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("student rms_norm requires x and weight to be CUDA tensors")
    needs_grad = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    if needs_grad and x.ndim >= 1 and x.shape[-1] > 1024:
        raise RuntimeError("rms_norm backward requires 0 < H <= 1024")
    # Copies stay in the graph so gradients propagate back through input views.
    xc, wc = x.contiguous(), weight.contiguous()
    if needs_grad:
        return _RMSNorm.apply(xc, wc, eps)
    output, _ = load_rms_norm_extension().rms_norm_forward(xc, wc, eps)
    return output


def rope(x, cos, sin):
    return _todo("rope")


def attention(q, k, v, past_len=0, segment_ids=None):
    return _todo("attention")


def swiglu(gate, up):
    return _todo("swiglu")


class _Residual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, update):
        # The derivative is constant: no input tensors need to be saved.
        return load_residual_extension().residual_forward(x, update)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        dx, dupdate = load_residual_extension().residual_backward(grad_output.contiguous())
        return (dx if ctx.needs_input_grad[0] else None,
                dupdate if ctx.needs_input_grad[1] else None)


def residual(x, update):
    """Same-shape CUDA add with dtype promotion and first-order autograd.

    FP16/BF16 inputs are explicitly converted to FP32 for the native kernels;
    the result is cast to the promoted dtype. Copies/casts remain in the graph
    so gradients return to the original input layout and dtype. No broadcasting.
    """
    if not x.is_cuda or not update.is_cuda:
        raise RuntimeError("student residual requires x and update to be CUDA tensors")
    if x.device != update.device:
        raise RuntimeError("x and update must be on the same CUDA device")
    if x.layout != torch.strided or update.layout != torch.strided:
        raise RuntimeError("x and update must have strided layout")
    if x.shape != update.shape:
        raise RuntimeError("x and update must have the same shape; broadcasting is not supported")
    supported = (torch.float32, torch.float16, torch.bfloat16)
    if x.dtype not in supported or update.dtype not in supported:
        raise RuntimeError("student residual supports only float32, float16 and bfloat16")
    dtype = torch.promote_types(x.dtype, update.dtype)
    xc, uc = x.float().contiguous(), update.float().contiguous()
    if torch.is_grad_enabled() and (xc.requires_grad or uc.requires_grad):
        output = _Residual.apply(xc, uc)
    else:
        output = load_residual_extension().residual_forward(xc, uc)
    return output.to(dtype)


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        # apply() disables grad mode here; pybind itself does not create a graph.
        loss, lse = load_cross_entropy_extension().cross_entropy_forward(logits, targets)
        count = (targets != -100).sum()
        ctx.save_for_backward(logits, targets, lse, count)
        # Match PyTorch: empty/all-ignored mean is NaN, with zero input gradients.
        return loss.sum() / count

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        logits, targets, lse, count = ctx.saved_tensors
        # Keep the shared scale on the GPU; the kernel broadcasts it to all rows.
        grad_loss = grad_output / count
        dz = load_cross_entropy_extension().cross_entropy_backward(
            logits, targets, lse, grad_loss
        )
        return dz, None


def cross_entropy(logits, targets):
    """Mean CUDA cross entropy: [B,T,V] logits, [B,T] int64 targets, ignore=-100.

    FP16/BF16 logits are explicitly converted to FP32; the scalar loss is FP32.
    Copies/casts preserve gradients to input views and their original dtype.
    Supports first-order eager autograd; empty/all-ignored targets yield NaN loss
    and zero gradients, matching the reference. No class weights or smoothing.
    """
    if not logits.is_cuda or not targets.is_cuda:
        raise RuntimeError("student cross_entropy requires logits and targets to be CUDA tensors")
    if logits.device != targets.device:
        raise RuntimeError("logits and targets must be on the same CUDA device")
    if logits.layout != torch.strided or targets.layout != torch.strided:
        raise RuntimeError("logits and targets must have strided layout")
    supported = (torch.float32, torch.float16, torch.bfloat16)
    if logits.dtype not in supported:
        raise RuntimeError("student cross_entropy logits supports only float32, float16 and bfloat16")
    if targets.dtype != torch.long:
        raise RuntimeError("cross_entropy targets must be int64")
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise RuntimeError("cross_entropy requires logits [B,T,V] and matching targets [B,T] shape")
    if logits.shape[-1] == 0:
        raise RuntimeError("vocabulary size must be positive")
    logits, targets = logits.float().contiguous(), targets.contiguous()
    if torch.is_grad_enabled() and logits.requires_grad:
        return _CrossEntropy.apply(logits, targets)
    loss, _ = load_cross_entropy_extension().cross_entropy_forward(logits, targets)
    return loss.sum() / (targets != -100).sum()
