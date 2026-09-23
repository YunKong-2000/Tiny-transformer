"""Your CUDA / CuTe / CUTLASS implementation entry points.

Every function is intentionally unfinished. There is no silent reference fallback.
Match reference.py semantics, device, shape, dtype, strides and gradients.
Use --op NAME=student to enable only a completed operator.
See docs/development.md before registering a compiled/custom operator.
"""


def _todo(name):
    raise NotImplementedError(
        f"Student operator '{name}' is not implemented. "
        f"Implement tiny_transformer/operators/student.py::{name} or select reference."
    )


def embedding(ids, weight):
    return _todo("embedding")


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
