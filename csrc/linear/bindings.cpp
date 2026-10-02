#include "linear.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "linear_forward",
        &linear_forward,
        "Linear CUDA forward (contiguous FP32 x, weight; no autograd)",
        pybind11::arg("x"),
        pybind11::arg("weight")
    );
    m.def(
        "linear_backward",
        &linear_backward,
        "Linear CUDA backward (returns dX and dWeight; contiguous FP32)",
        pybind11::arg("gradient"),
        pybind11::arg("x"),
        pybind11::arg("weight")
    );
}
