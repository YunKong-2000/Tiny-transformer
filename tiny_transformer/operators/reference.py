"""Executable specifications. Students implement optimized versions in student.py.

Weights of linear are [out_features, in_features]. Attention tensors are [B,H,T,D].
The RoPE convention pairs adjacent dimensions (0,1), (2,3), ... .
"""
import math

import torch
import torch.nn.functional as F


def accumulation(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def embedding(ids, weight):
    return F.embedding(ids, weight)


def linear(x, weight):
    return F.linear(x, weight)


def rms_norm(x, weight, eps):
    xf = accumulation(x)
    normalized = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * weight.to(xf.dtype)).to(x.dtype)


def rope(x, cos, sin):
    # x: [B,H,T,D]; cos/sin: [B or 1,1,T,D/2]. Preserve strided input support.
    even, odd = x[..., 0::2], x[..., 1::2]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def causal_mask(q, k, past_len=0, segment_ids=None):
    q_pos = torch.arange(q.shape[-2], device=q.device) + past_len
    k_pos = torch.arange(k.shape[-2], device=q.device)
    allowed = k_pos[None, :] <= q_pos[:, None]
    allowed = allowed[None, None, :, :]
    if segment_ids is not None:
        if past_len or q.shape[-2] != k.shape[-2]:
            raise ValueError("isolated documents are supported only without a KV cache")
        allowed = allowed & (segment_ids[:, None, :, None] == segment_ids[:, None, None, :])
    return allowed


def attention(q, k, v, past_len=0, segment_ids=None):
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores = accumulation(scores).masked_fill(~causal_mask(q, k, past_len, segment_ids), float("-inf"))
    probability = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.matmul(probability, v)


def sdpa_attention(q, k, v, past_len=0, segment_ids=None):
    # Non-square causal SDPA is upper-left aligned; cached multi-token queries
    # need an explicit lower-right-aligned mask. Single-token decode sees all keys.
    if segment_ids is None and past_len == 0 and q.shape[-2] == k.shape[-2]:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    if segment_ids is None and q.shape[-2] == 1:
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)
    mask = causal_mask(q, k, past_len, segment_ids)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)


def swiglu(gate, up):
    return F.silu(gate) * up


def residual(x, update):
    return x + update


def cross_entropy(logits, targets):
    # Explicit FP32 loss computation under BF16/FP16 autocast.
    return F.cross_entropy(accumulation(logits).reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
