from . import reference, student


NAMES = ("embedding", "linear", "rms_norm", "rope", "attention", "swiglu", "residual", "cross_entropy")


class Operators:
    def __init__(self, overrides=None):
        overrides = overrides or {}
        unknown = set(overrides) - set(NAMES)
        if unknown:
            raise ValueError(f"Unknown operators: {sorted(unknown)}")
        for name in NAMES:
            backend = overrides.get(name, "reference")
            if backend in ("reference", "student"):
                implementation = getattr(reference if backend == "reference" else student, name)
            elif backend == "sdpa" and name == "attention":
                implementation = reference.sdpa_attention
            else:
                raise ValueError(f"Unsupported backend {backend!r} for {name}")
            setattr(self, name, implementation)
