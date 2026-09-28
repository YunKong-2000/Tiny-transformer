#include "rope.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_forward", &rope_forward,
        "Adjacent-pair RoPE CUDA forward (strided FP32 inputs)",
        pybind11::arg("x"), pybind11::arg("cos"), pybind11::arg("sin"));
  m.def("rope_backward", &rope_backward,
        "RoPE CUDA input gradient (constant cos/sin, first-order only)",
        pybind11::arg("gradient"), pybind11::arg("cos"), pybind11::arg("sin"));
}
