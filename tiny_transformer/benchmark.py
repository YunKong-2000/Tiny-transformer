"""Compatibility entry point; use python -m tiny_transformer.benchmarks.model."""
from .benchmarks.model import main, percentile, request

if __name__ == "__main__":
    main()
