"""Lazy JIT build of the student CUDA extension from this source checkout."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_embedding_extension():
    # Importing the model/reference operators must not require a CUDA toolkit,
    # Ninja, or a C++ compiler. Only the first student CUDA call builds anything.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("student embedding requires CUDA-enabled PyTorch and a CUDA device")

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("student embedding requires a CUDA toolkit with nvcc; set CUDA_HOME")

    source_root = Path(__file__).resolve().parents[2] / "csrc"
    sources = [
        source_root / "bindings.cpp",
        source_root / "embedding" / "embedding.cu",
        source_root / "embedding" / "embedding_backward.cu",
    ]
    required = sources + [source_root / "embedding" / "embedding.h"]
    if not all(path.is_file() for path in required):
        raise RuntimeError(
            "student CUDA sources are missing; run from the repository or an editable install"
        )

    # PyTorch owns the on-disk build cache and honors TORCH_EXTENSIONS_DIR,
    # MAX_JOBS and TORCH_CUDA_ARCH_LIST. No compilation at package import time.
    return load(
        name="tiny_transformer_embedding_cuda",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
    )
