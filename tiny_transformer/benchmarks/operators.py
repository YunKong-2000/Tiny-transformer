"""Uniform CUDA forward/backward benchmarks for all eight Transformer operators."""
import argparse
from functools import partial
from itertools import product
import math

import torch

from ..operators import reference, student
from ..operators.dispatch import NAMES
from ..runtime import DTYPES, device_for, environment, seed_all, validate_precision, write_json
from .cases import LINEAR_PROJECTIONS, linear_spec, make_case
from .common import MEASUREMENT, measure_pair, prepare_calls
from .linear import LinearFp32Validator
from .linear_kernels import KERNEL_MEASUREMENT, measure_graph_calls, run_kernel_case
from .embedding import PATTERNS


# Explicit implementation status, not a fallback. Update as student kernels land.
STUDENT_PHASES = {name: ("forward", "backward")
                  for name in ("embedding", "linear", "rms_norm", "residual", "cross_entropy", "rope", "swiglu")}
STUDENT_PHASES["attention"] = ("forward",)

ATTENTION_GRAPH_MEASUREMENT = {
    "timer": "CUDA events around graph replay; median of per-trial mean call times",
    "order": "alternate baseline/candidate order across trials after warming both graphs",
    "forward": "capture complete no_grad operator calls; replay includes device kernels and required copies",
    "excluded": "Python/host dispatch, host allocation, input generation, validation, JIT, capture and warmup",
    "scope": "fixed inputs and graph-pool buffers, warm caches; includes device scheduling gaps; not profiler-exclusive kernel time",
    "speedup": "reference_us / candidate_us",
}


def unsupported_reason(operator, backend, precision, phase, layout, dim=None, heads=None):
    if layout == "last-only" and operator != "rms_norm":
        return "last-only layout applies only to rms_norm"
    if backend == "sdpa" and operator != "attention":
        return "sdpa backend applies only to attention"
    if backend == "student":
        if phase not in STUDENT_PHASES.get(operator, ()):
            return f"student {operator} {phase} is not implemented"
        if operator == "attention" and precision not in ("fp32", "bf16"):
            return "student attention currently supports only fp32 and bf16"
        if precision != "fp32" and operator not in ("residual", "cross_entropy", "attention"):
            return f"student {operator} currently supports only fp32"
        if operator == "rms_norm" and phase == "backward" and dim is not None and dim > 1024:
            return "student rms_norm backward requires H <= 1024"
        if operator == "embedding" and layout != "contiguous":
            return "student embedding requires contiguous inputs"
        if operator == "attention" and dim is not None and heads is not None and dim != 64 * heads:
            return "student attention currently requires head_dim=64"
    return None


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=(*NAMES, "all"), required=True)
    parser.add_argument("--backend", choices=("student", "reference", "sdpa"), default="student")
    parser.add_argument("--baseline", choices=("reference", "sdpa"), default="reference",
                        help="comparison baseline; sdpa is available for --operator attention")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=DTYPES, default="fp32")
    parser.add_argument("--phases", nargs="+", choices=("forward", "backward"),
                        default=["forward", "backward"])
    parser.add_argument("--workloads", nargs="+", choices=("train", "prefill", "decode"),
                        default=None, help="Linear defaults to train/prefill/decode; other operators to prefill/decode")
    parser.add_argument("--linear-projections", nargs="+", choices=(*LINEAR_PROJECTIONS, "custom"),
                        default=list(LINEAR_PROJECTIONS))
    parser.add_argument("--linear-timing", choices=("kernel", "operator"), default="kernel",
                        help="Linear: prepared CUDA Graph kernel timing (default), or full operator calls")
    parser.add_argument("--attention-timing", choices=("operator", "graph"), default="operator",
                        help="Attention forward: eager operator calls (default), or CUDA Graph replay without host submission gaps")
    parser.add_argument("--inference-batch-size", type=int,
                        help="Linear inference batch size; defaults to --batch-size")
    parser.add_argument("--layouts", nargs="+", choices=("contiguous", "strided", "last-only"),
                        default=["contiguous"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--out-features", type=int, default=768,
                        help="output width for --linear-projections custom")
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
    if args.attention_timing == "graph" and (args.operator != "attention" or set(args.phases) != {"forward"}):
        parser.error("--attention-timing graph requires --operator attention --phases forward")
    if args.inference_batch_size is not None and args.inference_batch_size <= 0:
        parser.error("inference-batch-size must be positive")
    if args.workloads and "train" in args.workloads and args.operator not in ("linear", "all"):
        parser.error("train workload is currently supported only for linear")
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
    if args.baseline == "sdpa" and args.operator != "attention":
        parser.error("--baseline sdpa requires --operator attention")


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
    # Select the baseline independently: --backend reference still means the
    # unfused operator, even when it is measured against SDPA.
    if args.baseline == "sdpa":
        expected_function = reference.sdpa_attention
    # Re-seeding per variant gives grouped/baseline exactly the same upstream.
    # fork_rng restores state so validation does not change subsequent case inputs.
    devices = [case.inputs[0].device] if case.inputs[0].is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(args.seed)
        if devices:
            torch.cuda.manual_seed(args.seed)
        calls, errors = prepare_calls(
            expected_function, candidate, case.inputs, case.grad_indices, phases=phases,
            upstream_scale=case.upstream_scale, exact_forward=operator == "embedding",
            validator=LinearFp32Validator(*case.inputs)
            if operator == "linear" and case.inputs[0].dtype == torch.float32 else None)
    result = {"validation": errors}
    graph_timing = operator == "attention" and args.attention_timing == "graph"
    if operator == "attention":
        result["timing_mode"] = args.attention_timing
        result["measurement"] = ATTENTION_GRAPH_MEASUREMENT if graph_timing else MEASUREMENT
    for phase, functions in calls.items():
        if graph_timing:
            baseline, candidate = measure_graph_calls(
                functions, case.inputs[0].device, args.warmup, args.repeats, args.trials)
            timing = {"reference_us": baseline["us"], "candidate_us": candidate["us"],
                      "reference_trials_us": baseline["trials_us"],
                      "candidate_trials_us": candidate["trials_us"],
                      "speedup": baseline["us"] / candidate["us"] if candidate["us"] > 0 else None}
        else:
            timing = measure_pair(functions, args.warmup, args.repeats, args.trials)
        result[phase] = {"status": "passed", **timing}
    return result


def run(args, device):
    operators = NAMES if args.operator == "all" else (args.operator,)
    results = []
    for operator in operators:
        if operator == "linear":
            results.extend(run_linear(args, device))
            continue
        workloads = [w for w in (args.workloads or ("prefill", "decode")) if w != "train"]
        patterns = tuple(dict.fromkeys(args.patterns)) if operator == "embedding" else (None,)
        implementations = (("grouped", "baseline") if args.backward_impl == "all" else
                           (args.backward_impl,)) if operator == "embedding" and args.backend == "student" else (None,)
        for workload, layout, pattern in product(dict.fromkeys(workloads),
                                                dict.fromkeys(args.layouts), patterns):
            reasons = {phase: unsupported_reason(operator, args.backend, args.precision, phase, layout, args.dim, args.heads)
                       for phase in dict.fromkeys(args.phases)}
            rows = args.batch_size * (1 if workload == "decode" else args.seq_length)
            if pattern == "unique" and rows > args.vocab_size:
                reasons = {phase: "batch_size * workload_length exceeds vocab_size; unique IDs are impossible"
                           for phase in reasons}
            phases = tuple(phase for phase, reason in reasons.items() if reason is None)
            # Unsupported cases need no GPU allocation or compilation.
            case = make_case(operator, args, device, DTYPES[args.precision], workload, layout, pattern) if phases else None
            for implementation in implementations:
                result = {"operator": operator, "backend": args.backend, "baseline": args.baseline,
                          "workload": workload,
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
                        print(f"{label}: {args.baseline}={value['reference_us']:.2f} us, "
                              f"{args.backend}={value['candidate_us']:.2f} us, speedup={ratio}")
            del case
    return results


def run_linear(args, device):
    """Measure each projection; kernel mode separates dX, dW and split-K stages."""
    results = []
    for workload, projection, layout in product(
            dict.fromkeys(args.workloads or ("train", "prefill", "decode")),
            dict.fromkeys(args.linear_projections), dict.fromkeys(args.layouts)):
        spec = linear_spec(args, workload, projection)
        # Inference has no backward. An explicit backward-only request produces
        # a visible skip instead of silently running a forward timing.
        requested = tuple(p for p in dict.fromkeys(args.phases)
                          if workload == "train" or p == "forward")
        if not requested:
            requested = ("backward",)
        reasons = {phase: ("inference has no backward" if workload != "train" and phase == "backward"
                           else unsupported_reason("linear", args.backend, args.precision, phase, layout, args.dim))
                   for phase in requested}
        phases = tuple(phase for phase, reason in reasons.items() if reason is None)
        result = {"operator": "linear", "backend": args.backend, "baseline": "reference", "workload": workload,
                  "layout": layout, "status": "passed" if phases else "skipped",
                  "timing_mode": args.linear_timing, **spec}
        if workload == "train":
            result["backward_scope"] = ("separate prepared dX and dWeight GEMMs; no autograd in timing"
                                        if args.linear_timing == "kernel" else
                                        "combined autograd dX and dWeight; not separate kernel timings")
        for phase, reason in reasons.items():
            if reason is not None:
                result[phase] = {"status": "skipped", "reason": reason}
        if phases:
            case = make_case("linear", args, device, DTYPES[args.precision], workload, layout,
                             projection=projection)
            result["inputs"] = case.metadata
            if args.linear_timing == "kernel":
                result.update(run_kernel_case(args, case, phases))
            else:
                result.update(run_case("linear", args, case, phases, None))
            del case
        results.append(result)
        reported = [phase for phase in ("forward", "backward", "dx", "dweight", "split_k_partials", "split_k_reduce")
                    if phase in result]
        for phase in reported:
            value = result[phase]
            shapes = (f"M,N,K={spec['gemm_shapes']['forward']}" if phase in ("forward", "split_k_partials", "split_k_reduce") else
                      f"M,N,K={spec['gemm_shapes'][phase]}" if phase in ("dx", "dweight") else
                      f"dX={spec['gemm_shapes'].get('dx')}, dW={spec['gemm_shapes'].get('dweight')}")
            label = f"linear {workload} {projection} {layout} {phase} {shapes}"
            if value["status"] == "skipped":
                print(f"{label}: skipped ({value['reason']})")
            elif phase in ("split_k_partials", "split_k_reduce"):
                print(f"{label}: student={value['candidate_us']:.2f} us, "
                      f"performance={value['tflops_per_second']:.4f} TFLOP/s")
            else:
                ratio = f"{value['speedup']:.2f}x" if value["speedup"] is not None else "n/a"
                performance = (f", reference={value['reference_tflops_per_second']:.4f} TFLOP/s, "
                               f"{args.backend}={value['candidate_tflops_per_second']:.4f} TFLOP/s"
                               if 'candidate_tflops_per_second' in value else "")
                print(f"{label}: reference={value['reference_us']:.2f} us, "
                      f"{args.backend}={value['candidate_us']:.2f} us, speedup={ratio}{performance}")
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
                "schema_version": 2, "environment": environment(device), "arguments": vars(args),
                "measurement": (ATTENTION_GRAPH_MEASUREMENT if args.attention_timing == "graph" else
                                KERNEL_MEASUREMENT if args.operator == "linear" and args.linear_timing == "kernel" else MEASUREMENT),
                "linear_kernel_measurement": KERNEL_MEASUREMENT if args.linear_timing == "kernel" else None,
                "notice": "reference backend is a harness baseline; skipped phases have no timings; "
                          "representative cases do not certify the full operator contract",
                "cases": results,
            })
    finally:
        torch.set_float32_matmul_precision(previous)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
