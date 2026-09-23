"""Model-only latency benchmark; no tokenizer, network, queue, or continuous batching."""
import argparse
import statistics
import time

import torch

from .runtime import add_inference_arguments, environment, inference_model, peak_memory, reset_peak, synchronize, write_json


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


@torch.inference_mode()
def request(model, executable, prompt, new_tokens, device):
    synchronize(device)
    start = time.perf_counter()
    cache = model.new_cache(prompt.shape[0], prompt.shape[1] + new_tokens)
    token = executable(prompt, cache=cache, last_only=True)[:, -1, :].argmax(dim=-1, keepdim=True)
    synchronize(device)
    first = time.perf_counter()
    for _ in range(new_tokens - 1):
        token = executable(token, cache=cache, last_only=True)[:, -1, :].argmax(dim=-1, keepdim=True)
    synchronize(device)
    end = time.perf_counter()
    return {"ttft_ms": (first - start) * 1000,
            "tpot_ms": (end - first) * 1000 / (new_tokens - 1),
            "request_ms": (end - start) * 1000,
            "output_tokens_per_second": prompt.shape[0] * new_tokens / (end - start),
            "decode_tokens_per_second": prompt.shape[0] * (new_tokens - 1) / (end - first),
            "kv_allocated_bytes": cache.allocated_bytes()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="runs/benchmark.json")
    args = parser.parse_args()
    if min(args.batch_size, args.prompt_length, args.repeats) <= 0 or args.new_tokens < 2 or args.warmup < 1:
        parser.error("positive batch/prompt/repeats, at least 2 generated tokens and 1 warmup required")
    model, executable, device, _ = inference_model(args)
    if args.prompt_length + args.new_tokens > model.config.max_seq_len:
        parser.error("prompt + generated tokens exceeds max_seq_len")
    prompt = torch.randint(3, model.config.vocab_size, (args.batch_size, args.prompt_length), device=device)
    reset_peak(device)
    cold = request(model, executable, prompt, args.new_tokens, device)
    cold_memory = peak_memory(device)
    for _ in range(args.warmup):
        request(model, executable, prompt, args.new_tokens, device)
    synchronize(device)
    reset_peak(device)
    trials = [request(model, executable, prompt, args.new_tokens, device) for _ in range(args.repeats)]
    summary = {}
    for field in ("ttft_ms", "tpot_ms", "request_ms", "output_tokens_per_second", "decode_tokens_per_second"):
        values = [row[field] for row in trials]
        summary[field] = {"median": statistics.median(values), "p95_across_trials": percentile(values, 0.95),
                          "min": min(values), "max": max(values)}
    result = {"environment": environment(device), "model": model.config.to_dict(), "parameter_count": model.parameter_count(),
              "arguments": vars(args), "weights": "checkpoint" if args.checkpoint else "random; performance only",
              "measurement": {"scope": "fixed-batch model-only greedy generation; includes cache allocation and argmax; prompt already on device",
                              "ttft": "prefill through first generated token, host wall time with synchronization",
                              "tpot": "remaining decode wall time / (new_tokens - 1); no per-token host synchronization",
                              "percentiles": "across request trials; TPOT percentiles are NOT per-token p95",
                              "throughput": "output tokens only; prompt tokens excluded",
                              "cold": "first complete request, includes lazy compilation and allocator/library warmup; not pure compile time"},
              "cold_request": cold, "cold_peak_memory": cold_memory,
              "steady": summary, "steady_peak_memory": peak_memory(device), "trials": trials}
    if args.compile:
        from torch._dynamo.utils import counters, compile_times
        result["dynamo_counters"] = {name: {str(k): v for k, v in counts.items()} for name, counts in counters.items()}
        result["compiler_reported_times"] = compile_times()
    write_json(args.output, result)
    print(f"saved {args.output}")
    print(summary)


if __name__ == "__main__":
    main()
