"""Export PyTorch trace and top-three self-time operators for one execution phase."""
import argparse
from pathlib import Path

import torch

from .runtime import add_inference_arguments, environment, inference_model, synchronize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_arguments(parser)
    parser.add_argument("--phase", choices=["train", "prefill", "decode"], default="decode")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--active-steps", type=int, default=3)
    parser.add_argument("--output", default="runs/profile")
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.active_steps) <= 0:
        parser.error("shapes and active steps must be positive")
    model, executable, device, _ = inference_model(args)
    required_length = args.seq_len + (1 if args.phase == "decode" else 0)
    if required_length > model.config.max_seq_len:
        parser.error("profile context exceeds max_seq_len (decode requires one extra position)")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ids = torch.randint(3, model.config.vocab_size, (args.batch_size, args.seq_len), device=device)
    targets = torch.randint(3, model.config.vocab_size, ids.shape, device=device)
    cache = None
    if args.phase == "train":
        # Match train.py: FP32 master parameters, autocast compute, fused AdamW on CUDA.
        model.float().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    elif args.phase == "decode":
        with torch.inference_mode():
            cache = model.new_cache(args.batch_size, args.seq_len + 1)
            executable(ids, cache=cache, last_only=True)
        # Reuse the prefix: each profiled iteration decodes at exactly the same context length.
        ids = ids[:, -1:].contiguous()

    def iteration():
        if args.phase == "train":
            from .runtime import autocast
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, args.precision):
                logits = executable(ids)
                loss = model.ops.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.inference_mode():
                if cache is not None:
                    cache.length = args.seq_len
                executable(ids, cache=cache, last_only=True)

    for _ in range(2):
        iteration()
    synchronize(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profile:
        for _ in range(args.active_steps):
            with torch.profiler.record_function(f"lab/{args.phase}"):
                iteration()
        synchronize(device)
    profile.export_chrome_trace(str(output / "trace.json"))
    key = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    averages = profile.key_averages(group_by_input_shape=True)
    (output / "operators.txt").write_text(averages.table(sort_by=key, row_limit=30))
    ranked = sorted((event for event in averages if not event.key.startswith("lab/")), key=lambda event: getattr(event, key), reverse=True)
    top = [{"operator": event.key, "input_shapes": str(event.input_shapes), "self_time_us": getattr(event, key), "calls": event.count} for event in ranked[:3]]
    write_json(output / "summary.json", {"environment": environment(device), "arguments": vars(args),
                                        "ranking": key, "top_three": top,
                                        "note": "PyTorch operator self-time, not a DRAM/Roofline measurement; use Nsight for kernel counters"})
    print(averages.table(sort_by=key, row_limit=10))


if __name__ == "__main__":
    main()
