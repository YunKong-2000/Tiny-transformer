#include "attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_forward", &attention_forward,
        "Attention CUDA forward (FP32/BF16, returns O and LSE)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("past_len") = 0,
        pybind11::arg("segment_ids") = pybind11::none());
  m.def("attention_backward", &attention_backward,
        "Attention CUDA BF16 backward (preprocess, dQ, dK/dV)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("dO"), pybind11::arg("O"), pybind11::arg("lse"),
        pybind11::arg("past_len") = 0,
        pybind11::arg("segment_ids") = pybind11::none());
}
