"""Prepared Linear GEMMs timed on the GPU with repeated CUDA Graph nodes."""
import math
import statistics
from functools import partial

import torch

from ..operators._extension import load_linear_extension
from .common import max_error, tolerances
from .linear import LinearFp32Validator


KERNEL_MEASUREMENT = {
    "timer": "CUDA events around one CUDA Graph replay containing repeats operations; median across trials",
    "scope": "GPU graph execution interval per operation, including device scheduling gaps; not profiler-exclusive kernel duration",
    "excluded": "Python/host dispatch, allocation, copies, argument initialization, JIT, validation, graph capture and warmup",
    "reference": "preallocated torch.mm for each mathematical GEMM; backend may launch multiple kernels",
    "backward": "dX and dWeight measured separately without autograd or forward in the timed region",
    "split_k": "forward is measured as the complete two-kernel pipeline; partials and reduction are also measured in isolation",
    "flops": "useful GEMM work=2*M*N*K (FMA=2); split reduction=partitions*M*N additions from zero; excludes epilogue/address/padding work",
    "cache": "fixed inputs, preallocated buffers, warmed caches; no cache flush",
}


def throughput(flops, microseconds):
    if not math.isfinite(microseconds) or microseconds <= 0:
        raise ValueError("kernel latency must be finite and positive")
    return {"flops": int(flops), "flops_per_second": flops * 1e6 / microseconds,
            "tflops_per_second": flops / (microseconds * 1e6)}


def capture_repeated(function, device, warmup, repeats):
    """Capture fixed-pointer calls, eliminating host gaps between repetitions."""
    current = torch.cuda.current_stream(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            function()
    stream.synchronize()  # initialization/warmup is explicitly outside capture/timing
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            function()
    with torch.cuda.stream(stream):
        graph.replay()  # upload/warm the executable graph outside timing
    current.wait_stream(stream)
    return graph


def measure_graph_calls(functions, device, warmup, repeats, trials):
    if not functions or min(warmup, repeats, trials) <= 0:
        raise ValueError("nonempty calls and positive warmup/repeats/trials required")
    graphs = [capture_repeated(fn, device, warmup, repeats) for fn in functions]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    samples = [[] for _ in graphs]
    for trial in range(trials):
        order = range(len(graphs)) if trial % 2 == 0 else reversed(range(len(graphs)))
        for index in order:
            start.record()
            graphs[index].replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000 / repeats)
    return [{"us": statistics.median(values), "trials_us": values} for values in samples]


def paired_report(measurements, flops):
    ref, candidate = measurements
    return {"status": "passed", "flops": flops,
            "reference_us": ref['us'], "candidate_us": candidate['us'],
            "reference_trials_us": ref['trials_us'], "candidate_trials_us": candidate['trials_us'],
            "speedup": ref['us'] / candidate['us'],
            "reference_flops_per_second": throughput(flops, ref['us'])['flops_per_second'],
            "candidate_flops_per_second": throughput(flops, candidate['us'])['flops_per_second'],
            "reference_tflops_per_second": throughput(flops, ref['us'])['tflops_per_second'],
            "candidate_tflops_per_second": throughput(flops, candidate['us'])['tflops_per_second']}


@torch.no_grad()
def run_kernel_case(args, case, phases):
    """Validate production and prepared paths before timing any graph."""
    device = case.inputs[0].device
    with torch.cuda.device(device), torch.random.fork_rng(devices=[device]):
        torch.random.default_generator.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        x, weight = (value.detach().contiguous() for value in case.inputs)
        m, n, k = math.prod(x.shape[:-1]), weight.shape[0], weight.shape[1]
        x = x.reshape(1, m, k)
        x2 = x.reshape(m, k)
        has_backward = 'backward' in phases
        dy = torch.randn(1, m, n, device=device, dtype=x.dtype) * case.upstream_scale if has_backward else x.new_empty(0)
        expected = x.new_empty(m, n)
        ref_calls = {'forward': lambda: torch.mm(x2, weight.T, out=expected)}
        if has_backward:
            dy2 = dy.reshape(m, n)
            ref_dx, ref_dw = torch.empty_like(x2), torch.empty_like(weight)
            ref_calls.update(dx=lambda: torch.mm(dy2, weight, out=ref_dx),
                             dweight=lambda: torch.mm(dy2.T, x2, out=ref_dw))
        for fn in ref_calls.values():
            fn()  # establish reference outputs before validation
        prepared = None
        if args.backend == 'student':
            extension = load_linear_extension()
            prepared = extension.prepare_linear_benchmark(x, weight, dy, has_backward)
            calls = {name: partial(prepared.run, name) for name in ref_calls}
            for fn in calls.values():
                fn()
            actual = prepared.output
            # Prepared paths must match the real operator, including split-K.
            torch.testing.assert_close(actual, extension.linear_forward(x, weight), atol=0, rtol=0)
            if has_backward:
                actual_grads = (prepared.dx, prepared.dweight)
                production_grads = extension.linear_backward(dy, x, weight)
                for value, production in zip(actual_grads, production_grads):
                    torch.testing.assert_close(value, production, atol=0, rtol=0)
            config = {"forward_kind": prepared.forward_kind, "split_k_slices": prepared.split_k_slices,
                      "tiles": prepared.tiles,
                      "workspace_bytes": prepared.workspace.numel() if prepared.forward_kind == 'split_k' else 0}
        else:
            # Reference-only mode remains a baseline self-comparison, using its
            # own output buffers so there is no alias between paired calls.
            actual = x.new_empty(1, m, n)
            calls = {'forward': lambda: torch.mm(x2, weight.T, out=actual.view(m, n))}
            if has_backward:
                actual_grads = (torch.empty_like(x), torch.empty_like(weight))
                calls.update(dx=lambda: torch.mm(dy2, weight, out=actual_grads[0].view(m, k)),
                             dweight=lambda: torch.mm(dy2.T, x2, out=actual_grads[1]))
            for fn in calls.values():
                fn()
            config = {"forward_kind": "reference", "split_k_slices": 1, "workspace_bytes": 0}
        expected3 = expected.view(1, m, n)
        validation = {"forward_max_abs_error": max_error(actual, expected3)}
        if x.dtype == torch.float32:
            validator = LinearFp32Validator(x, weight)
            validation.update(validation_policy=validator.policy,
                              forward_fp64=validator.forward(actual, expected3))
            if has_backward:
                validation['backward_fp64'] = validator.backward(
                    actual_grads, (ref_dx.view_as(x), ref_dw), dy, (0, 1))
        else:
            forward_tol, backward_tol = tolerances(x.dtype)
            torch.testing.assert_close(actual, expected3, atol=forward_tol[0], rtol=forward_tol[1])
            if has_backward:
                for value, target in zip(actual_grads, (ref_dx.view_as(x), ref_dw)):
                    torch.testing.assert_close(value, target, atol=backward_tol[0], rtol=backward_tol[1])
        if prepared is not None and prepared.forward_kind == 'split_k' and 'forward' in phases:
            # Isolated reduction must consume valid partials. Check exactly that
            # these public Params produce the same result as the device operator.
            production = actual.clone()
            prepared.run('split_k_partials')
            prepared.output.fill_(float('nan'))
            prepared.run('split_k_reduce')
            torch.testing.assert_close(prepared.output, production, atol=0, rtol=0)
            validation['split_k_stages_match_production'] = True

        result = {"validation": validation, "kernel_config": config, "kernel_measurement": KERNEL_MEASUREMENT}
        result['prepared_inputs'] = {"x_shape": list(x.shape), "weight_shape": list(weight.shape),
                                     "contiguous": True, "copies_in_timing": False}
        requested = (['forward'] if 'forward' in phases else []) + (['dx', 'dweight'] if has_backward else [])
        flops = 2 * m * n * k
        for name in requested:
            measurements = measure_graph_calls((ref_calls[name], calls[name]), device,
                                                args.warmup, args.repeats, args.trials)
            result[name] = paired_report(measurements, flops)
            result[name]['component'] = ('split_k_pipeline' if name == 'forward' and
                                         config['forward_kind'] == 'split_k' else 'gemm')
            result[name]['candidate_kernel_count'] = (None if prepared is None else
                                                      2 if result[name]['component'] == 'split_k_pipeline' else 1)
        if prepared is not None and prepared.forward_kind == 'split_k' and 'forward' in phases:
            for name, operations in (('split_k_partials', flops),
                                     ('split_k_reduce', config['split_k_slices'] * m * n)):
                measured, = measure_graph_calls((partial(prepared.run, name),), device,
                                                 args.warmup, args.repeats, args.trials)
                result[name] = {"status": "passed", "candidate_us": measured['us'],
                                "candidate_trials_us": measured['trials_us'],
                                **throughput(operations, measured['us'])}
                result[name]['flop_convention'] = ('2*M*N*K useful GEMM work' if name == 'split_k_partials'
                                                  else 'S*M*N additions into zero accumulator; epilogue excluded')
        return result
