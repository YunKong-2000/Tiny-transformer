"""Check CUTLASS/CuTe headers, then compile and run a small CUDA kernel."""
import argparse
from pathlib import Path

from ._cutlass import cutlass_include_paths


def run_smoke_test():
    includes = cutlass_include_paths()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUTLASS smoke test requires CUDA-enabled PyTorch and a CUDA device")
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUTLASS smoke test requires nvcc; set CUDA_HOME")
    source = Path(__file__).resolve().parents[1] / "csrc" / "cutlass_smoke" / "smoke.cu"
    extension = load(
        name="tiny_transformer_cutlass_smoke",
        sources=[str(source)],
        extra_include_paths=includes,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo"],
        with_cuda=True,
    )
    # Check a partial block and the PyTorch current-stream integration as well.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.no_grad():
        x = torch.arange(257, device="cuda", dtype=torch.float32)
        actual = extension.add_one(x)
        torch.testing.assert_close(actual, x + 1, rtol=0, atol=0)
    stream.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers-only", action="store_true",
                        help="check header paths without importing PyTorch or compiling")
    args = parser.parse_args()
    for path in cutlass_include_paths():
        print(f"Include: {path}")
    if args.headers_only:
        print("CUTLASS/CuTe headers OK (CUDA compilation not checked).")
    else:
        run_smoke_test()
        print("CUTLASS/CuTe CUDA compilation and kernel output OK.")


if __name__ == "__main__":
    main()
