#include "residual.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "residual_forward",
        &residual_forward_cuda,
        "Residual CUDA forward (same shape, contiguous FP32; no autograd)",
        pybind11::arg("X"),
        pybind11::arg("update")
    );
    m.def(
        "residual_backward",
        &residual_backward_cuda,
        "Residual CUDA backward (returns dX and dUpdate; contiguous FP32)",
        pybind11::arg("dY")
    );
}
