"""Pre-norm decoder with tied embeddings and an explicit contiguous KV cache."""
from typing import Optional

import torch
from torch import nn

from .config import ModelConfig
from .operators import Operators


class KVCache:
    """Inference-only cache. Capacity is allocated once per layer, on first use.

    Storage is [batch, heads, capacity, head_dim]; the live prefix is a view.
    length advances once per model call, not once per layer.
    """
    def __init__(self, config, batch_size, capacity):
        if not 0 < capacity <= config.max_seq_len or batch_size <= 0:
            raise ValueError("invalid KV cache batch size or capacity")
        self.batch_size = batch_size
        self.capacity = capacity
        self.length = 0
        self.keys = [None] * config.n_layers
        self.values = [None] * config.n_layers

    def reset(self):
        self.length = 0

    def update(self, layer, k, v):
        end = self.length + k.shape[-2]
        if end > self.capacity or k.shape[0] != self.batch_size:
            raise ValueError("KV cache capacity exceeded or batch size changed")
        if self.keys[layer] is None:
            shape = (self.batch_size, k.shape[1], self.capacity, k.shape[-1])
            self.keys[layer] = k.new_empty(shape)
            self.values[layer] = v.new_empty(shape)
        elif self.keys[layer].dtype != k.dtype or self.keys[layer].device != k.device:
            raise ValueError("KV cache dtype/device changed; create a new cache")
        self.keys[layer][:, :, self.length:end, :].copy_(k)
        self.values[layer][:, :, self.length:end, :].copy_(v)
        return self.keys[layer][:, :, :end, :], self.values[layer][:, :, :end, :]

    def allocated_bytes(self):
        return sum(x.numel() * x.element_size() for x in self.keys + self.values if x is not None)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn_norm = nn.Parameter(torch.ones(config.dim))
        self.ffn_norm = nn.Parameter(torch.ones(config.dim))
        self.qkv = nn.Parameter(torch.empty(3 * config.dim, config.dim))
        self.proj = nn.Parameter(torch.empty(config.dim, config.dim))
        self.gate_up = nn.Parameter(torch.empty(2 * config.hidden_dim, config.dim))
        self.down = nn.Parameter(torch.empty(config.dim, config.hidden_dim))

    def forward(self, x, ops, cos, sin, layer, cache=None, segment_ids=None):
        cfg = self.config
        batch, time, _ = x.shape
        normalized = ops.rms_norm(x, self.attn_norm, cfg.norm_eps)
        qkv = ops.linear(normalized, self.qkv).reshape(batch, time, 3, cfg.n_heads, cfg.dim // cfg.n_heads)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]
        q, k = ops.rope(q, cos, sin), ops.rope(k, cos, sin)
        past_len = 0 if cache is None else cache.length
        if cache is not None:
            k, v = cache.update(layer, k, v)
        attended = ops.attention(q, k, v, past_len, segment_ids)
        attended = attended.transpose(1, 2).reshape(batch, time, cfg.dim)
        x = ops.residual(x, ops.linear(attended, self.proj))
        normalized = ops.rms_norm(x, self.ffn_norm, cfg.norm_eps)
        gate, up = ops.linear(normalized, self.gate_up).chunk(2, dim=-1)
        return ops.residual(x, ops.linear(ops.swiglu(gate, up), self.down))


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.ops = Operators(config.operators)
        self.embedding = nn.Parameter(torch.empty(config.vocab_size, config.dim))
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = nn.Parameter(torch.ones(config.dim))
        if config.tie_embeddings:
            self.output_weight = self.embedding
        else:
            self.output_weight = nn.Parameter(torch.empty(config.vocab_size, config.dim))
        for parameter in self.parameters():
            if parameter.ndim >= 2:
                nn.init.normal_(parameter, mean=0.0, std=0.02)
        # Residual projection initialization is scaled by depth.
        for block in self.blocks:
            for parameter in (block.proj, block.down):
                nn.init.normal_(parameter, std=0.02 / (2 * config.n_layers) ** 0.5)

    def new_cache(self, batch_size, capacity=None):
        return KVCache(self.config, batch_size, self.config.max_seq_len if capacity is None else capacity)

    def forward(self, ids, cache: Optional[KVCache] = None, segment_ids=None, position_ids=None, last_only=False):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("ids must have shape [batch, nonzero time]")
        past_len = 0 if cache is None else cache.length
        time = ids.shape[1]
        if time + past_len > self.config.max_seq_len:
            raise ValueError("sequence exceeds model max_seq_len")
        if cache is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError("KV cache requires eval() and torch.no_grad()/inference_mode()")
            if segment_ids is not None or position_ids is not None:
                raise ValueError("custom segments/positions cannot be combined with KV cache")
            if time + past_len > cache.capacity or ids.shape[0] != cache.batch_size:
                raise ValueError("KV cache capacity exceeded or batch size changed")
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + time, device=ids.device)[None, :]
        # Reconstruct frequencies in FP32 even if model weights were cast to BF16.
        head_dim = self.config.dim // self.config.n_heads
        frequency = 1.0 / self.config.rope_theta ** (torch.arange(0, head_dim, 2, device=ids.device).float() / head_dim)
        angles = position_ids.float()[..., None] * frequency
        cos, sin = angles.cos()[:, None, :, :], angles.sin()[:, None, :, :]
        x = self.ops.embedding(ids, self.embedding)
        for layer, block in enumerate(self.blocks):
            x = block(x, self.ops, cos, sin, layer, cache, segment_ids)
        if cache is not None:
            cache.length += time
        if last_only:
            x = x[:, -1:, :]
        x = self.ops.rms_norm(x, self.final_norm, self.config.norm_eps)
        return self.ops.linear(x, self.output_weight)

    def loss(self, ids, targets, segment_ids=None, position_ids=None):
        logits = self(ids, segment_ids=segment_ids, position_ids=position_ids)
        return self.ops.cross_entropy(logits, targets)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
