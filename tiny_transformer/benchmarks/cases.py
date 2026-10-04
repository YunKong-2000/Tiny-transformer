"""Representative per-operator inputs; timing policy lives in common.py."""
from dataclasses import dataclass

import torch

from .embedding import make_ids


@dataclass
class Case:
    inputs: tuple
    grad_indices: tuple
    metadata: dict
    upstream_scale: float


LINEAR_PROJECTIONS = ("qkv", "o", "gate_up", "down", "lm_head")


def linear_spec(args, workload, projection):
    """Describe model GEMMs without allocating tensors, including skipped cases."""
    if workload not in ("train", "prefill", "decode"):
        raise ValueError(f"unknown Linear workload: {workload}")
    dimensions = {"qkv": (3 * args.dim, args.dim), "o": (args.dim, args.dim),
                  "gate_up": (2 * args.hidden_dim, args.dim),
                  "down": (args.dim, args.hidden_dim),
                  "lm_head": (args.vocab_size, args.dim),
                  "custom": (args.out_features, args.dim)}
    n, k = dimensions[projection]
    batch = (args.inference_batch_size or args.batch_size) if workload != "train" else args.batch_size
    # generate.py requests last_only=True even during prefill.
    time = 1 if workload == "decode" or (workload == "prefill" and projection == "lm_head") else args.seq_length
    m = batch * time
    training = workload == "train"
    return {"projection": projection, "execution": "training" if training else "inference",
            "x_shape": [batch, time, k], "weight_shape": [n, k],
            "output_shape": [batch, time, n],
            "gemm_shapes": {"forward": [m, n, k], **({"dx": [m, k, n], "dweight": [n, k, m]}
                                                        if training else {})}}


def make_case(name, args, device, dtype, workload, layout="contiguous", pattern="random",
              projection="custom"):
    b, t, h = args.batch_size, 1 if workload == "decode" else args.seq_length, args.dim
    if layout == "last-only" and name != "rms_norm":
        raise ValueError("last-only layout applies only to rms_norm")

    def rand(*shape):
        if layout == "strided":
            return torch.randn(*shape[:-1], shape[-1] * 2, device=device, dtype=dtype)[..., ::2]
        return torch.randn(*shape, device=device, dtype=dtype)

    if name == "embedding":
        ids = make_ids(pattern, b, t, args.vocab_size, args.hot_tokens, device)
        inputs, gradients = (ids, rand(args.vocab_size, h)), (1,)
    elif name == "linear":
        spec = linear_spec(args, workload, projection)
        b, t, _ = spec["x_shape"]
        inputs, gradients = (rand(*spec["x_shape"]), rand(*spec["weight_shape"])), (0, 1)
    elif name == "rms_norm":
        x = rand(b, t, h)
        if layout == "last-only":
            x = x[:, -1:, :]
        inputs, gradients = (x, rand(h), args.eps), (0, 1)
    elif name == "rope":
        d = h // args.heads
        # Actual model input is a transpose of [B,T,heads,head_dim].
        x = rand(b, t, args.heads, d).transpose(1, 2)
        angles = torch.randn(1, 1, t, d // 2, device=device)
        inputs, gradients = (x, angles.cos(), angles.sin()), (0,)
    elif name == "attention":
        d, length = h // args.heads, args.seq_length
        inputs = (rand(b, args.heads, t, d), rand(b, args.heads, length, d),
                  rand(b, args.heads, length, d), length - t, None)
        gradients = (0, 1, 2)
    elif name == "swiglu":
        # Chunk views retain the real fused gate/up projection's row stride.
        gate, up = rand(b, t, 2 * args.hidden_dim).chunk(2, dim=-1)
        inputs, gradients = (gate, up), (0, 1)
    elif name == "residual":
        inputs, gradients = (rand(b, t, h), rand(b, t, h)), (0, 1)
    elif name == "cross_entropy":
        targets = torch.randint(args.vocab_size, (b, t), device=device)
        if targets.numel() > 1:
            targets[0, 0] = -100
        inputs, gradients = (rand(b, t, args.vocab_size), targets), (0,)
    else:
        raise ValueError(f"unknown operator: {name}")
    if layout == "contiguous":
        inputs = tuple(x.contiguous() if torch.is_tensor(x) else x for x in inputs)
    metadata = {"workload": workload, "layout": layout,
                "inputs": [{"shape": list(x.shape), "stride": list(x.stride()),
                            "dtype": str(x.dtype), "storage_offset": x.storage_offset()}
                           if torch.is_tensor(x) else x for x in inputs],
                "grad_indices": list(gradients)}
    if name == "embedding":
        metadata.update(pattern=pattern, distinct_ids=inputs[0].unique().numel())
    if name == "linear":
        metadata.update(spec)
    # Loss already averages tokens; use a unit-scale scalar upstream for CE.
    rows = b if layout == "last-only" else b * t
    scale = 1.0 if name == "cross_entropy" else 1.0 / rows
    metadata["upstream"] = "normal random values scaled by " + str(scale)
    return Case(inputs, gradients, metadata, scale)
