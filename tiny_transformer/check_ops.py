"""Compare a selected student operator with its executable reference contract."""
import argparse
import json
import time

import torch

from .operators import reference, student
from .runtime import DTYPES, device_for, environment, seed_all, synchronize, validate_precision, write_json


def cases(name, device, dtype, decode=False):
    batch, time, dim, heads = 2, 1 if decode else 17, 64, 4
    rand = lambda *shape: torch.randn(*shape, device=device, dtype=dtype)
    if name == "linear":
        return (rand(batch, time, dim), rand(96, dim))
    if name == "rms_norm":
        # Odd row width exercises masked tails; non-contiguous input exercises stride contracts.
        return (rand(batch, time, 130)[..., ::2], rand(65), 1e-6)
    if name == "rope":
        x = rand(batch, time, heads, 16).transpose(1, 2)
        angles = torch.randn(1, 1, time, 8, device=device)
        return x, angles.cos(), angles.sin()
    if name == "attention":
        length = 23 if decode else time
        return (rand(batch, heads, time, 16), rand(batch, heads, length, 16), rand(batch, heads, length, 16), length - time, None)
    if name in ("swiglu", "residual"):
        return rand(batch, time, 65), rand(batch, time, 65)
    if name == "embedding":
        return torch.randint(0, 259, (batch, time), device=device), rand(259, dim)
    if name == "cross_entropy":
        targets = torch.randint(0, 259, (batch, time), device=device)
        targets[0, 0] = -100
        return rand(batch, time, 259), targets
    raise ValueError(name)


def differentiable_args(args):
    # cos/sin are constants in the model, but this helper intentionally checks all
    # floating inputs except RoPE constants (handled below).
    return tuple(x.detach().clone().requires_grad_(True) if torch.is_tensor(x) and x.is_floating_point() else x for x in args)


def timed(function, args, device, repeats):
    with torch.no_grad():
        for _ in range(5):
            function(*args)
        synchronize(device)
        begin = time.perf_counter()
        for _ in range(repeats):
            function(*args)
        synchronize(device)
    return (time.perf_counter() - begin) * 1e6 / repeats


def main():
    from .operators.dispatch import NAMES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=NAMES, required=True)
    parser.add_argument("--backend", choices=["student", "reference", "sdpa"], default="student")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=DTYPES, default="fp32")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", default="runs/operator_check.json")
    args = parser.parse_args()
    if args.repeats <= 0 or (args.backend == "sdpa" and args.operator != "attention"):
        parser.error("positive repeats required; sdpa is only an attention backend")
    device = device_for(args.device)
    validate_precision(device, args.precision)
    seed_all(42)
    function = reference.sdpa_attention if args.backend == "sdpa" else getattr(student if args.backend == "student" else reference, args.operator)
    expected_function = getattr(reference, args.operator)
    results = []
    atol, rtol = (1e-5, 1e-4) if args.precision == "fp32" else ((1e-2, 3e-2) if args.precision == "bf16" else (2e-3, 5e-3))
    for decode in (False, True):
        inputs = cases(args.operator, device, DTYPES[args.precision], decode)
        with torch.no_grad():
            expected, actual = expected_function(*inputs), function(*inputs)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        row = {"case": "decode" if decode else "prefill", "max_abs_error": float((actual - expected).abs().max()),
               "reference_us": timed(expected_function, inputs, device, args.repeats), "candidate_us": timed(function, inputs, device, args.repeats)}
        if args.backward:
            left, right = differentiable_args(inputs), differentiable_args(inputs)
            if args.operator == "rope":
                for index in (1, 2):
                    left[index].requires_grad_(False)
                    right[index].requires_grad_(False)
            target = expected_function(*left)
            upstream = torch.randn_like(target)
            expected_inputs = [x for x in left if torch.is_tensor(x) and x.requires_grad]
            actual_inputs = [x for x in right if torch.is_tensor(x) and x.requires_grad]
            gradients = torch.autograd.grad(target, expected_inputs, upstream)
            candidate_gradients = torch.autograd.grad(function(*right), actual_inputs, upstream)
            for expected_gradient, actual_gradient in zip(gradients, candidate_gradients):
                torch.testing.assert_close(actual_gradient, expected_gradient, atol=atol * 3, rtol=rtol * 3)
            row["backward"] = "passed"
        results.append(row)
    write_json(args.output, {"environment": environment(device), "arguments": vars(args), "atol": atol, "rtol": rtol,
                             "notice": "development smoke cases; extend to real model shapes before performance claims", "cases": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
