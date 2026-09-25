"""Compare FP32 embedding forward/backward latency across token distributions."""
import argparse
from functools import partial
import statistics

import torch

from .operators import reference, student
from .runtime import device_for, environment, seed_all, write_json


PATTERNS = ("random", "same", "unique", "hot")


def make_ids(pattern, batch, length, vocab_size, hot_tokens, device):
    if pattern == "random":
        return torch.randint(vocab_size, (batch, length), device=device)
    if pattern == "same":
        return torch.zeros(batch, length, device=device, dtype=torch.int64)
    if pattern == "unique":
        if batch * length > vocab_size:
            raise ValueError("unique IDs require batch_size * seq_length <= vocab_size")
        return torch.arange(batch * length, device=device).reshape(batch, length)
    if pattern == "hot":
        return torch.randint(min(hot_tokens, vocab_size), (batch, length), device=device)
    raise ValueError(f"unknown pattern: {pattern}")


def prepare_calls(ids, weight, upstream, candidate=student.embedding):
    """Build graphs and check results before timing; reuse only graphs, not gradients."""
    expected = reference.embedding(ids, weight)
    actual = candidate(ids, weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def reference_backward():
        return torch.autograd.grad(expected, weight, upstream, retain_graph=True)[0]

    def candidate_backward():
        return torch.autograd.grad(actual, weight, upstream, retain_graph=True)[0]

    expected_gradient = reference_backward()
    actual_gradient = candidate_backward()
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-5, rtol=3e-4)
    errors = {
        "forward_max_abs_error": float((actual.detach() - expected.detach()).abs().max()),
        "backward_max_abs_error": float((actual_gradient - expected_gradient).abs().max()),
    }
    inference_weight = weight.detach()
    return {
        "forward": (lambda: reference.embedding(ids, inference_weight),
                    lambda: candidate(ids, inference_weight)),
        "backward": (reference_backward, candidate_backward),
    }, errors


def measure_pair(functions, warmup, repeats, trials):
    """CUDA stream intervals in microseconds, with alternating backend order."""
    for function in functions:
        for _ in range(warmup):
            function()
    # Initialize both events outside the measured interval, including lazy creation.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()

    samples = [[], []]
    for trial in range(trials):
        for index in ((0, 1) if trial % 2 == 0 else (1, 0)):
            function = functions[index]
            start.record()
            for _ in range(repeats):
                function()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000 / repeats)
    medians = [statistics.median(values) for values in samples]
    return {
        "reference_us": medians[0],
        "student_us": medians[1],
        "speedup": medians[0] / medians[1] if medians[1] > 0 else None,
        "reference_trials_us": samples[0],
        "student_trials_us": samples[1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--patterns", choices=PATTERNS, nargs="+", default=list(PATTERNS))
    parser.add_argument("--hot-tokens", type=int, default=16)
    parser.add_argument("--backward-impl", choices=("grouped", "baseline", "all"), default="grouped",
                        help="student backward kernel; all compares both on identical inputs")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/embedding-performance.json")
    args = parser.parse_args()
    if min(args.batch_size, args.seq_length, args.vocab_size, args.dim,
           args.hot_tokens, args.warmup, args.repeats, args.trials) <= 0:
        parser.error("shapes, hot-tokens, warmup, repeats and trials must be positive")
    if torch.device(args.device).type != "cuda":
        parser.error("this benchmark requires CUDA; CPU timings are not supported")
    device = device_for(args.device)
    seed_all(args.seed)

    rows = args.batch_size * args.seq_length
    implementations = ("grouped", "baseline") if args.backward_impl == "all" else (args.backward_impl,)
    results = []
    with torch.cuda.device(device):
        weight = torch.randn(args.vocab_size, args.dim, device=device, requires_grad=True)
        # Model a mean-reduced loss; both backends use exactly the same upstream.
        upstream = torch.randn(args.batch_size, args.seq_length, args.dim, device=device) / rows
        for pattern in dict.fromkeys(args.patterns):
            if pattern == "unique" and rows > args.vocab_size:
                reason = "batch_size * seq_length exceeds vocab_size; unique IDs are impossible"
                results.append({"pattern": pattern, "status": "skipped", "reason": reason})
                print(f"{pattern}: skipped ({reason})")
                continue
            ids = make_ids(pattern, args.batch_size, args.seq_length, args.vocab_size,
                           args.hot_tokens, device)
            distinct_ids = ids.unique().numel()
            for implementation in implementations:
                candidate = partial(student.embedding, backward_impl=implementation)
                calls, errors = prepare_calls(ids, weight, upstream, candidate=candidate)
                result = {"pattern": pattern, "backward_impl": implementation, "status": "passed",
                          "distinct_ids": distinct_ids, "validation": errors}
                for phase, functions in calls.items():
                    timings = measure_pair(functions, args.warmup, args.repeats, args.trials)
                    result[phase] = timings
                    speedup = timings["speedup"]
                    ratio = f"{speedup:.2f}x" if speedup is not None else "n/a"
                    print(f"{pattern:6s} {implementation:8s} {phase:8s}: "
                          f"reference={timings['reference_us']:.2f} us, "
                          f"student={timings['student_us']:.2f} us, speedup={ratio}")
                results.append(result)
                # Release retained graphs before constructing the next variant.
                del calls, functions

        write_json(args.output, {
            "environment": environment(device), "arguments": vars(args), "dtype": "float32",
            "measurement": {
                "timer": "CUDA events on current stream; median of per-trial mean call times",
                "forward": "inference forward, including output allocation; no autograd graph",
                "backward": "autograd.grad on a retained graph; includes gradient allocation/zeroing "
                            "and dispatch; excludes forward and accumulation into weight.grad",
                "excluded": "input generation, correctness checks, JIT compilation and warmup",
                "scope": "repeated fixed inputs, warm allocator/cache; eager stream intervals may "
                         "include CPU submission gaps, not isolated kernel execution time",
                "upstream": "contiguous FP32 normal random values divided by B*T",
                "hot_token_count": min(args.hot_tokens, args.vocab_size),
                "speedup": "reference_us / student_us; greater than 1 means student is faster",
            },
            "cases": results,
        })
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
