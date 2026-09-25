"""Shared validation, retained-graph backward and paired latency measurement."""
import statistics
import time

import torch


MEASUREMENT = {
    "timer": "CUDA events on current stream; median of per-trial mean call times",
    "order": "alternate reference/candidate order across trials after warming both",
    "forward": "no_grad forward; includes dispatch, output allocation and wrapper copies",
    "backward": "autograd.grad on retained graphs; includes gradient allocation/zeroing and "
                "dispatch; excludes forward and accumulation into leaf .grad",
    "excluded": "input generation, correctness checks, JIT compilation and warmup",
    "scope": "fixed inputs, warm allocator/cache; eager stream intervals may include CPU "
             "submission gaps; not isolated kernel execution time",
    "speedup": "reference_us / candidate_us",
}


def tolerances(dtype, exact_forward=False):
    atol, rtol = {torch.float32: (1e-5, 1e-4), torch.float64: (1e-8, 1e-6),
                  torch.bfloat16: (1e-2, 3e-2), torch.float16: (2e-3, 5e-3)}[dtype]
    return ((0, 0) if exact_forward else (atol, rtol)), (atol * 3, rtol * 3)


def max_error(actual, expected):
    return float((actual.detach() - expected.detach()).abs().max()) if actual.numel() else 0.0


def prepare_calls(reference, candidate, inputs, grad_indices, *, phases=("forward", "backward"),
                  upstream=None, upstream_scale=1.0, exact_forward=False):
    """Validate requested phases before timing; preserve input strides and constant args.

    Each backend gets separate detached leaves with the original storage layout.
    Operators must not mutate inputs. Only explicitly listed arguments get gradients
    (e.g. RoPE cos/sin stay constant). No .backward() or accumulation into .grad.
    """
    forward_inputs = tuple(x.detach() if torch.is_tensor(x) else x for x in inputs)

    @torch.no_grad()
    def reference_forward():
        return reference(*forward_inputs)

    @torch.no_grad()
    def candidate_forward():
        return candidate(*forward_inputs)

    expected, actual = reference_forward(), candidate_forward()
    # Output dtype can differ from input dtype (cross entropy accumulates FP32).
    dtype = next(x.dtype for x in inputs if torch.is_tensor(x) and x.is_floating_point())
    forward_tol, backward_tol = tolerances(dtype, exact_forward)
    torch.testing.assert_close(actual, expected, atol=forward_tol[0], rtol=forward_tol[1])
    errors = {"forward_max_abs_error": max_error(actual, expected),
              "forward_atol": forward_tol[0], "forward_rtol": forward_tol[1]}
    calls = {}
    if "forward" in phases:
        calls["forward"] = (reference_forward, candidate_forward)
    if "backward" in phases:
        if not grad_indices:
            raise ValueError("backward needs differentiable input indices")

        def graph(function):
            args = tuple(x.detach().requires_grad_(i in grad_indices) if torch.is_tensor(x)
                         and x.is_floating_point() else x for i, x in enumerate(inputs))
            leaves = tuple(args[i] for i in grad_indices)
            with torch.enable_grad():
                output = function(*args)
            return output, leaves

        expected_graph, expected_leaves = graph(reference)
        actual_graph, actual_leaves = graph(candidate)
        torch.testing.assert_close(actual_graph, expected_graph,
                                   atol=forward_tol[0], rtol=forward_tol[1])
        if upstream is None:
            upstream = torch.randn_like(expected_graph) * upstream_scale

        def reference_backward():
            return torch.autograd.grad(expected_graph, expected_leaves, upstream, retain_graph=True)

        def candidate_backward():
            return torch.autograd.grad(actual_graph, actual_leaves, upstream, retain_graph=True)

        expected_grads, actual_grads = reference_backward(), candidate_backward()
        gradient_errors = []
        for expected_grad, actual_grad in zip(expected_grads, actual_grads):
            torch.testing.assert_close(actual_grad, expected_grad,
                                       atol=backward_tol[0], rtol=backward_tol[1])
            gradient_errors.append(max_error(actual_grad, expected_grad))
        errors.update(backward_max_abs_error=max(gradient_errors),
                      backward_errors_by_input=dict(zip(grad_indices, gradient_errors)),
                      backward_atol=backward_tol[0], backward_rtol=backward_tol[1])
        calls["backward"] = (reference_backward, candidate_backward)
    return calls, errors


def _summary(samples):
    medians = [statistics.median(values) for values in samples]
    return {"reference_us": medians[0], "candidate_us": medians[1],
            "speedup": medians[0] / medians[1] if medians[1] > 0 else None,
            "reference_trials_us": samples[0], "candidate_trials_us": samples[1]}


def measure_pair(functions, warmup, repeats, trials):
    """CUDA event timing; caller owns the device/current stream context."""
    if len(functions) != 2 or min(warmup, repeats, trials) <= 0:
        raise ValueError("two functions and positive warmup/repeats/trials required")
    for function in functions:
        for _ in range(warmup):
            function()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    samples = [[], []]
    for trial in range(trials):
        for index in ((0, 1) if trial % 2 == 0 else (1, 0)):
            start.record()
            for _ in range(repeats):
                functions[index]()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000 / repeats)
    return _summary(samples)


def measure_cpu_smoke_pair(functions, warmup, repeats, trials):
    """CPU-only check_ops diagnostics; never used to claim CUDA performance."""
    for function in functions:
        for _ in range(warmup):
            function()
    samples = [[], []]
    for trial in range(trials):
        for index in ((0, 1) if trial % 2 == 0 else (1, 0)):
            start = time.perf_counter()
            for _ in range(repeats):
                functions[index]()
            samples[index].append((time.perf_counter() - start) * 1e6 / repeats)
    return _summary(samples)
