"""Resolve the header-only CUTLASS/CuTe dependency without importing CUDA tools."""
import os
from pathlib import Path


def cutlass_include_paths():
    """Use CUTLASS_PATH (repository root), or this checkout's pinned submodule."""
    configured = os.environ.get("CUTLASS_PATH")
    root = (Path(configured).expanduser() if configured else
            Path(__file__).resolve().parents[1] / "third_party" / "cutlass").resolve()
    required = ("include/cutlass/cutlass.h", "include/cute/tensor.hpp",
                "tools/util/include/cutlass/util/host_tensor.h")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"CUTLASS/CuTe headers are missing at {root}: {', '.join(missing)}. "
            "Run `git submodule update --init --recursive third_party/cutlass` "
            "from the repository root, or set CUTLASS_PATH to a complete CUTLASS checkout "
            "(not its include directory)."
        )
    return [str(root / "include"), str(root / "tools" / "util" / "include")]
