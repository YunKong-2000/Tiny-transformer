import contextlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from .config import ModelConfig, apply_operator_overrides, load_config
from .model import Transformer


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use the NGC container on the GPU host")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("supported devices: cpu, cuda")
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    return device


def validate_precision(device, precision):
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training/inference is supported only on CUDA in this lab")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support BF16")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_tf32(enabled):
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def autocast(device, precision):
    return contextlib.nullcontext() if precision == "fp32" else torch.autocast(device.type, dtype=DTYPES[precision])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory(device):
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def environment(device):
    result = {"python": platform.python_version(), "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "device": str(device), "platform": platform.platform(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "time_unix": time.time()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update({"gpu": props.name, "gpu_memory_bytes": props.total_memory,
                       "compute_capability": [props.major, props.minor]})
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_checkpoint(path, device="cpu"):
    # Checkpoints include optimizer/RNG objects. Load ONLY your own trusted files.
    return torch.load(path, map_location=device, weights_only=False)


def inference_model(args):
    device = device_for(args.device)
    validate_precision(device, args.precision)
    seed_all(args.seed)
    set_tf32(args.tf32)
    checkpoint = None
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        config = ModelConfig(**checkpoint["model_config"])
    else:
        config, _ = load_config(args.config)
    apply_operator_overrides(config, args.op)
    model = Transformer(config)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    weight_dtype = torch.float32 if getattr(args, "phase", None) == "train" else DTYPES[args.precision]
    model.to(device=device, dtype=weight_dtype).eval()
    executable = torch.compile(model) if args.compile else model
    return model, executable, device, checkpoint


def add_inference_arguments(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", default="configs/smoke.json")
    source.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=list(DTYPES), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--op", action="append", default=[])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--tf32", action="store_true")
