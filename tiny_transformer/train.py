"""Single-GPU FP32/BF16/FP16 training with packed windows and resumable state."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .config import apply_operator_overrides, load_config
from .data import PackedDataset
from .model import Transformer
from .runtime import autocast, device_for, environment, load_checkpoint, peak_memory, reset_peak, seed_all, set_tf32, synchronize, validate_precision, write_json


def lr_at(step, training):
    warmup = training["warmup_steps"]
    if step < warmup:
        return training["lr"] * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(1, training["steps"] - warmup - 1)
    progress = min(max(progress, 0.0), 1.0)
    ratio = training["min_lr_ratio"] + (1 - training["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * progress))
    return training["lr"] * ratio


@torch.no_grad()
def evaluate(model, executable, dataset, training, device):
    model.eval()
    # Replay the same validation windows at every evaluation.
    dataset.rng = np.random.default_rng(training["seed"] + 1)
    weighted_loss, tokens = 0.0, 0
    for _ in range(training["eval_batches"]):
        batch = dataset.batch(training["batch_size"], device)
        with autocast(device, training["precision"]):
            logits = executable(batch["ids"], segment_ids=batch["segment_ids"], position_ids=batch["position_ids"])
            loss = model.ops.cross_entropy(logits, batch["targets"])
        count = int((batch["targets"] != -100).sum())
        weighted_loss += float(loss) * count
        tokens += count
    model.train()
    return weighted_loss / tokens


def save_checkpoint(path, model, optimizer, scaler, training, step, dataset):
    state = {"format_version": 1, "model_config": model.config.to_dict(), "training": training,
             "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "step": step, "sampler_rng": dataset.rng.bit_generator.state,
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "tokenizer": dataset.tokenizer.spec,
             "data_metadata": dataset.metadata}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def run(args):
    config, training = load_config(args.config)
    for name in ("steps", "batch_size", "grad_accum", "seq_len", "precision", "packing"):
        value = getattr(args, name, None)
        if value is not None:
            training[name] = value
    if args.tf32:
        training["tf32"] = True
    apply_operator_overrides(config, args.op)
    for name in ("steps", "batch_size", "grad_accum", "seq_len", "eval_every", "eval_batches", "save_every"):
        if training[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if training["seq_len"] > config.max_seq_len:
        raise ValueError("training seq_len exceeds model max_seq_len")
    if args.stop_after is not None and args.stop_after <= 0:
        raise ValueError("stop_after must be positive")
    device = device_for(args.device)
    validate_precision(device, training["precision"])
    set_tf32(training["tf32"])
    seed_all(training["seed"])
    train_data = PackedDataset(args.data, "train", training["seq_len"], training["seed"], training["packing"])
    val_data = PackedDataset(args.data, "val", training["seq_len"], training["seed"] + 1, training["packing"])
    if train_data.tokenizer.vocab_size > config.vocab_size:
        raise ValueError("tokenizer vocabulary is larger than model vocabulary")
    model = Transformer(config).to(device)
    # FP32 master parameters; autocast controls GEMM/attention compute precision.
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": training["weight_decay"]},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=training["lr"], fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and training["precision"] == "fp16")
    start_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint["model_config"] != config.to_dict():
            raise ValueError("resume model configuration differs from checkpoint")
        if checkpoint["tokenizer"] != train_data.tokenizer.spec:
            raise ValueError("resume tokenizer differs from checkpoint")
        if checkpoint.get("data_metadata") != train_data.metadata:
            raise ValueError("resume data manifest differs from checkpoint")
        for key in training:
            if key == "steps":
                continue
            if checkpoint["training"][key] != training[key]:
                raise ValueError(f"resume changes {key}; start a new run instead")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"]
        train_data.rng.bit_generator.state = checkpoint["sampler_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
    if start_step >= training["steps"]:
        raise ValueError("checkpoint already reached requested total steps")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists() and not args.resume:
        raise FileExistsError("output already contains a run; choose a new directory or --resume")
    executable = torch.compile(model) if args.compile else model
    metadata = {"environment": environment(device), "model": config.to_dict(), "training": training,
                "parameter_count": model.parameter_count(), "compile": args.compile,
                "data": json.loads((Path(args.data) / "metadata.json").read_text()),
                "timing": "wall time with CUDA synchronization; includes batch sampling/copy, forward, backward, optimizer; excludes eval/save"}
    write_json(output / ("resume_metadata.json" if args.resume else "metadata.json"), metadata)
    print(json.dumps({"parameter_count": model.parameter_count(), "device": str(device), "precision": training["precision"]}))
    model.train()
    reset_peak(device)
    stop = min(training["steps"], start_step + args.stop_after) if args.stop_after else training["steps"]
    tokens_per_step = training["batch_size"] * training["seq_len"] * training["grad_accum"]
    with (output / "metrics.jsonl").open("a", buffering=1) as log:
        for step in range(start_step, stop):
            synchronize(device)
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            lr = lr_at(step, training)
            for group in optimizer.param_groups:
                group["lr"] = lr
            total_loss = torch.zeros((), device=device)
            valid_tokens = torch.zeros((), dtype=torch.long, device=device)
            # Weight gradients by valid targets, including unequal isolated-document boundaries.
            batches = [train_data.batch(training["batch_size"], device) for _ in range(training["grad_accum"])]
            counts = [(batch["targets"] != -100).sum() for batch in batches]
            valid_count = sum(counts)
            for batch, count in zip(batches, counts):
                with autocast(device, training["precision"]):
                    logits = executable(batch["ids"], segment_ids=batch["segment_ids"], position_ids=batch["position_ids"])
                    loss = model.ops.cross_entropy(logits, batch["targets"])
                weighted = loss * (count / valid_count)
                scaler.scale(weighted).backward()
                total_loss += weighted.detach()
                valid_tokens += count
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            finite_gradient = bool(torch.isfinite(grad_norm))
            if not torch.isfinite(total_loss):
                raise FloatingPointError("non-finite loss; inspect precision and operators")
            if not finite_gradient and not scaler.is_enabled():
                raise FloatingPointError("non-finite gradient; inspect precision and operators")
            # FP16 overflow is handled by GradScaler: skip the update and reduce
            # the scale. BF16/FP32 have no loss scaling and fail on invalid grads.
            step_skipped = not finite_gradient
            scaler.step(optimizer)
            scaler.update()
            synchronize(device)
            seconds = time.perf_counter() - begin
            row = {"step": step + 1, "loss": float(total_loss), "lr": lr, "grad_norm": float(grad_norm) if finite_gradient else None,
                   "optimizer_step_skipped": step_skipped, "loss_scale": scaler.get_scale(),
                   "step_seconds": seconds, "tokens_per_second": tokens_per_step / seconds,
                   "input_tokens": tokens_per_step, "valid_target_tokens": int(valid_tokens),
                   "measurement_warmup": step - start_step < 2, **peak_memory(device)}
            if (step + 1) % training["eval_every"] == 0 or step + 1 == stop:
                row["val_loss"] = evaluate(model, executable, val_data, training, device)
            log.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if (step + 1) % training["save_every"] == 0 or step + 1 == stop:
                save_checkpoint(output / "last.pt", model, optimizer, scaler, training, step + 1, train_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_60m.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    for name in ("steps", "batch-size", "grad-accum", "seq-len", "stop-after"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--packing", choices=["continuous", "isolated"])
    parser.add_argument("--op", action="append", default=[])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--resume")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
