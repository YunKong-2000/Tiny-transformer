#include "rope.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rope_forward",
        &rope_forward_cuda,
        "Rope CUDA forward (strided FP32 X, cos, sin)",
        pybind11::arg("X"),
        pybind11::arg("cos"),
        pybind11::arg("sin")
    );
    m.def(
        "rope_backward",
        &embedding_backward_cuda,
        "Rope CUDA backward (strided FP32 gradient, cos, sin)",
        pybind11::arg("gradient"),
        pybind11::arg("cos"),
        pybind11::arg("sin")
    );
}