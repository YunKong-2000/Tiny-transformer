"""Independent FP64 validation for FP32 Linear, outside all timing regions.

Two FP32 GEMMs need not agree at cancellation-dominated outputs. Validate both
against FP64 instead, using the FP32 dot-product roundoff bound AND an RMS check.
"""
import math

import torch


class LinearFp32Validator:
    policy = "fp64_dot_product_bound_and_rms"
    rms_atol = 1e-5
    rms_rtol = 1e-4

    def __init__(self, x, weight):
        self.x = x.detach()
        self.weight = weight.detach()
        if self.x.dtype != torch.float32 or self.weight.dtype != torch.float32:
            raise ValueError("FP64 Linear validation requires FP32 inputs")
        self.rows = math.prod(self.x.shape[:-1])
        self.k = self.x.shape[-1]
        self.n = self.weight.shape[0]

    @torch.no_grad()
    def product(self, actual, expected, left, right, shape):
        """Check all entries, chunking output rows to bound FP64 workspace.

        For round-to-nearest FP32 FMA, |error| <= gamma_L * (|A| @ |B|),
        gamma_L = L*u/(1-L*u), u=2^-24, L=the reduction length. The additional
        RMS gate retains a strict overall accuracy requirement when that worst-
        case bound is conservative. This is not proof of the kernel's math mode.
        """
        for name, value in (("candidate", actual), ("reference", expected)):
            if tuple(value.shape) != tuple(shape) or value.dtype != torch.float32:
                raise AssertionError(f"Linear {name} shape/dtype mismatch: {value.shape}, {value.dtype}")
            if value.device != left.device:
                raise AssertionError(f"Linear {name} device mismatch")
        reduction = left.shape[1]
        unit_roundoff = torch.finfo(torch.float32).eps / 2
        if reduction * unit_roundoff >= 1:
            raise ValueError("reduction too long for the FP32 gamma_L validation bound")
        gamma = reduction * unit_roundoff / (1 - reduction * unit_roundoff)
        rows, columns = left.shape[0], right.shape[1]
        values = {"candidate": actual.detach().reshape(rows, columns),
                  "reference": expected.detach().reshape(rows, columns)}
        stats = {name: {"max_abs_error": 0.0, "max_roundoff_ratio": 0.0, "squared_error": 0.0}
                 for name in values}
        squared_oracle = 0.0
        if rows and columns:
            rhs = right.detach().double()
            abs_rhs = rhs.abs()
            if not bool(torch.isfinite(rhs).all()):
                raise AssertionError("Linear validation requires finite operands")
            for start in range(0, rows, 128):
                lhs = left[start:start + 128].detach().double()
                if not bool(torch.isfinite(lhs).all()):
                    raise AssertionError("Linear validation requires finite operands")
                oracle = lhs @ rhs
                bound = gamma * (lhs.abs() @ abs_rhs)
                # Conservative allowance for subnormal/flush-to-zero effects.
                bound += reduction * torch.finfo(torch.float32).tiny
                squared_oracle += float(oracle.square().sum())
                for name, value in values.items():
                    delta = (value[start:start + 128].double() - oracle).abs()
                    if not bool(torch.isfinite(delta).all()):
                        raise AssertionError(f"Linear {name} contains non-finite results")
                    ratio = delta / bound.clamp_min(torch.finfo(torch.float64).tiny)
                    max_ratio = float(ratio.max())
                    stats[name]["max_abs_error"] = max(stats[name]["max_abs_error"], float(delta.max()))
                    stats[name]["max_roundoff_ratio"] = max(stats[name]["max_roundoff_ratio"], max_ratio)
                    stats[name]["squared_error"] += float(delta.square().sum())
                    if bool((delta > bound).any()):
                        raise AssertionError(
                            f"Linear {name} exceeds FP64 dot-product roundoff bound "
                            f"(reduction={reduction}, max error/bound={max_ratio:.6g}, "
                            f"max abs error={float(delta.max()):.6g})")
        count = max(rows * columns, 1)
        oracle_rms = math.sqrt(squared_oracle / count)
        rms_limit = self.rms_atol + self.rms_rtol * oracle_rms
        for name, report in stats.items():
            report["rms_error"] = math.sqrt(report.pop("squared_error") / count)
            if report["rms_error"] > rms_limit:
                raise AssertionError(
                    f"Linear {name} exceeds FP64 RMS limit: {report['rms_error']:.6g} > {rms_limit:.6g}")
        return {"reduction_length": reduction, "gamma": gamma, "oracle_rms": oracle_rms,
                "rms_limit": rms_limit, **stats}

    def forward(self, actual, expected):
        return self.product(actual, expected, self.x.reshape(self.rows, self.k), self.weight.T,
                            (*self.x.shape[:-1], self.n))

    def backward(self, actual_grads, expected_grads, upstream, grad_indices):
        dy = upstream.detach().reshape(self.rows, self.n)
        x = self.x.reshape(self.rows, self.k)
        products = {0: (dy, self.weight, self.x.shape), 1: (dy.T, x, self.weight.shape)}
        return {str(index): self.product(actual, expected, *products[index])
                for index, actual, expected in zip(grad_indices, actual_grads, expected_grads)}
