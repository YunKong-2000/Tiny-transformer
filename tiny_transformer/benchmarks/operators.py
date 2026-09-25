"""Uniform CUDA forward/backward benchmarks for all eight Transformer operators."""
import argparse
from functools import partial
from itertools import product
import math

import torch

from ..operators import reference, student
from ..operators.dispatch import NAMES
from ..runtime import DTYPES, device_for, environment, seed_all, validate_precision, write_json
from .cases import make_case
from .common import MEASUREMENT, measure_pair, prepare_calls
from .embedding import PATTERNS


# Explicit implementation status, not a fallback. Update as student kernels land.
STUDENT_PHASES = {"embedding": ("forward", "backward"), "rms_norm": ("forward",)}


def unsupported_reason(operator, backend, precision, phase, layout):
    if layout == "last-only" and operator != "rms_norm":
        return "last-only layout applies only to rms_norm"
    if backend == "sdpa" and operator != "attention":
        return "sdpa backend applies only to attention"
    if backend == "student":
        if phase not in STUDENT_PHASES.get(operator, ()):
            return f"student {operator} {phase} is not implemented"
        if precision != "fp32":
            return f"student {operator} currently supports only fp32"
        if operator == "embedding" and layout != "contiguous":
            return "student embedding requires contiguous inputs"
    return None


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=(*NAMES, "all"), required=True)
    parser.add_argument("--backend", choices=("student", "reference", "sdpa"), default="student")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=DTYPES, default="fp32")
    parser.add_argument("--phases", nargs="+", choices=("forward", "backward"),
                        default=["forward", "backward"])
    parser.add_argument("--workloads", nargs="+", choices=("prefill", "decode"),
                        default=["prefill", "decode"])
    parser.add_argument("--layouts", nargs="+", choices=("contiguous", "strided", "last-only"),
                        default=["contiguous"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--out-features", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--patterns", choices=PATTERNS, nargs="+", default=list(PATTERNS))
    parser.add_argument("--hot-tokens", type=int, default=16)
    parser.add_argument("--backward-impl", choices=("grouped", "baseline", "all"), default="grouped")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/operators-performance.json")
    return parser


def validate_args(parser, args):
    if min(args.batch_size, args.seq_length, args.vocab_size, args.dim, args.heads,
           args.out_features, args.hidden_dim, args.hot_tokens, args.warmup,
           args.repeats, args.trials) <= 0:
        parser.error("shapes, hot-tokens, warmup, repeats and trials must be positive")
    if args.operator in ("all", "rope", "attention") and args.dim % args.heads:
        parser.error("dim must be divisible by heads for attention/RoPE")
    if args.operator in ("all", "rope") and (args.dim // args.heads) % 2:
        parser.error("RoPE head dimension must be even")
    if not math.isfinite(args.eps) or args.eps <= 0:
        parser.error("eps must be finite and positive for benchmark inputs")
    if torch.device(args.device).type != "cuda":
        parser.error("this benchmark requires CUDA; CPU timings are not supported")
    if args.backend == "sdpa" and args.operator not in ("attention", "all"):
        parser.error("sdpa is only an attention backend")


def run_case(operator, args, case, phases, implementation):
    """All variants share case inputs/upstream; validate before either phase is timed."""
    expected_function = getattr(reference, operator)
    if args.backend == "reference":
        candidate = expected_function
    elif args.backend == "sdpa":
        candidate = reference.sdpa_attention
    else:
        candidate = getattr(student, operator)
        if operator == "embedding":
            candidate = partial(candidate, backward_impl=implementation)
    # Re-seeding per variant gives grouped/baseline exactly the same upstream.
    # fork_rng restores state so validation does not change subsequent case inputs.
    devices = [case.inputs[0].device] if case.inputs[0].is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(args.seed)
        if devices:
            torch.cuda.manual_seed(args.seed)
        calls, errors = prepare_calls(
            expected_function, candidate, case.inputs, case.grad_indices, phases=phases,
            upstream_scale=case.upstream_scale, exact_forward=operator == "embedding")
    result = {"validation": errors}
    for phase, functions in calls.items():
        result[phase] = {"status": "passed", **measure_pair(
            functions, args.warmup, args.repeats, args.trials)}
    return result


def run(args, device):
    operators = NAMES if args.operator == "all" else (args.operator,)
    results = []
    for operator in operators:
        patterns = tuple(dict.fromkeys(args.patterns)) if operator == "embedding" else (None,)
        implementations = (("grouped", "baseline") if args.backward_impl == "all" else
                           (args.backward_impl,)) if operator == "embedding" and args.backend == "student" else (None,)
        for workload, layout, pattern in product(dict.fromkeys(args.workloads),
                                                dict.fromkeys(args.layouts), patterns):
            reasons = {phase: unsupported_reason(operator, args.backend, args.precision, phase, layout)
                       for phase in dict.fromkeys(args.phases)}
            rows = args.batch_size * (1 if workload == "decode" else args.seq_length)
            if pattern == "unique" and rows > args.vocab_size:
                reasons = {phase: "batch_size * workload_length exceeds vocab_size; unique IDs are impossible"
                           for phase in reasons}
            phases = tuple(phase for phase, reason in reasons.items() if reason is None)
            # Unsupported cases need no GPU allocation or compilation.
            case = make_case(operator, args, device, DTYPES[args.precision], workload, layout, pattern) if phases else None
            for implementation in implementations:
                result = {"operator": operator, "backend": args.backend, "workload": workload,
                          "layout": layout, "status": "passed" if phases else "skipped"}
                if pattern is not None:
                    result["pattern"] = pattern
                if implementation is not None:
                    result["backward_impl"] = implementation
                for phase, reason in reasons.items():
                    if reason is not None:
                        result[phase] = {"status": "skipped", "reason": reason}
                if case is not None:
                    result["inputs"] = case.metadata
                    result.update(run_case(operator, args, case, phases, implementation))
                results.append(result)
                for phase in reasons:
                    value = result[phase]
                    label = f"{operator} {workload} {layout} {pattern or ''} {implementation or ''} {phase}"
                    if value["status"] == "skipped":
                        print(f"{label}: skipped ({value['reason']})")
                    else:
                        ratio = f"{value['speedup']:.2f}x" if value["speedup"] is not None else "n/a"
                        print(f"{label}: reference={value['reference_us']:.2f} us, "
                              f"{args.backend}={value['candidate_us']:.2f} us, speedup={ratio}")
            del case
    return results


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    device = device_for(args.device)
    previous = torch.get_float32_matmul_precision()
    try:
        # All FP32 operator comparisons use full precision, with the policy recorded.
        torch.set_float32_matmul_precision("highest")
        with torch.cuda.device(device):
            validate_precision(device, args.precision)
            seed_all(args.seed)
            results = run(args, device)
            write_json(args.output, {
                "schema_version": 1, "environment": environment(device), "arguments": vars(args),
                "measurement": MEASUREMENT,
                "notice": "reference backend is a harness baseline; skipped phases have no timings; "
                          "representative cases do not certify the full operator contract",
                "cases": results,
            })
    finally:
        torch.set_float32_matmul_precision(previous)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
