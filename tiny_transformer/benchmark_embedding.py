"""Compatibility entry point; use python -m tiny_transformer.benchmarks.embedding."""
from .benchmarks.embedding import PATTERNS, main, make_ids, measure_pair, prepare_calls

if __name__ == "__main__":
    main()
