#include <torch/extension.h>

#include "embedding/embedding.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "embedding_forward",
        &embedding_forward_cuda,
        "Embedding CUDA forward (contiguous int64 IDs, FP32 weight; no backward)",
        pybind11::arg("ids"),
        pybind11::arg("weight")
    );
}
