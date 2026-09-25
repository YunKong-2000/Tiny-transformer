#include "rms_norm.h"

// Separate module: building RMSNorm does not link embedding (or vice versa).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rms_norm_forward",
        &rms_norm_forward_cuda,
        "RMSNorm CUDA forward (returns Y and inv_rms; contiguous FP32; no autograd)",
        pybind11::arg("X"),
        pybind11::arg("weight"),
        pybind11::arg("epsilon")
    );
    m.def(
        "rms_norm_backward",
        &rms_norm_backward_cuda,
        "RMSNorm CUDA backward (returns dX and dGamma; FP32, H <= 1024)",
        pybind11::arg("X"),
        pybind11::arg("gradient"),
        pybind11::arg("weight"),
        pybind11::arg("R")
    );
}
