import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    dim: int = 768
    n_layers: int = 8
    n_heads: int = 12
    hidden_dim: int = 2048
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    operators: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("vocab_size", "dim", "n_layers", "n_heads", "hidden_dim", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dim % self.n_heads or (self.dim // self.n_heads) % 2:
            raise ValueError("dim must be divisible by n_heads, and head dimension must be even")
        if self.norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("norm_eps and rope_theta must be positive")

    def to_dict(self):
        return asdict(self)


def load_config(path):
    with Path(path).open() as handle:
        raw = json.load(handle)
    return ModelConfig(**raw["model"]), raw["training"]


def apply_operator_overrides(config, overrides):
    for item in overrides or []:
        name, separator, backend = item.partition("=")
        if not separator:
            raise ValueError("operator override must look like rms_norm=student")
        config.operators[name] = backend
