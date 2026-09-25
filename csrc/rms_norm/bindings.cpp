#include "rms_norm.h"

// Separate module: building RMSNorm does not link embedding (or vice versa).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rms_norm_forward",
        &rms_norm_forward_cuda,
        "RMSNorm CUDA forward (contiguous FP32 input and weight; no autograd)",
        pybind11::arg("X"),
        pybind11::arg("weight"),
        pybind11::arg("epsilon")
    );
}
