#include "swiglu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swiglu_forward", &swiglu_forward,
        "Swiglu CUDA forward (strided FP32 inputs)",
        pybind11::arg("gate"), pybind11::arg("up"));
  m.def("swiglu_backward", &swiglu_backward,
        "SwiGLU CUDA dgate and dup (strided FP32 inputs and dz)",
        pybind11::arg("gradient"), pybind11::arg("gate"), pybind11::arg("up"));
}
