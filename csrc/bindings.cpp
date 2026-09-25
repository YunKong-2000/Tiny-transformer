#include <torch/extension.h>

#include "embedding/embedding.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "embedding_forward",
        &embedding_forward_cuda,
        "Embedding CUDA forward (contiguous int64 IDs, FP32 weight)",
        pybind11::arg("ids"),
        pybind11::arg("weight")
    );
    m.def(
        "embedding_backward",
        &embedding_backward_cuda,
        "Embedding CUDA backward (contiguous int64 IDs, FP32 gradient, vocabulary size)",
        pybind11::arg("ids"),
        pybind11::arg("gradient"),
        pybind11::arg("vocab_size"),
        pybind11::arg("implementation") = "grouped"
    );
}
