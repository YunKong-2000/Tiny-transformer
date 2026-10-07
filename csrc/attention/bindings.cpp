#include "attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_forward", &attention_forward,
        "Attention CUDA forward (strided FP32 inputs)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("past_len") = 0,
        pybind11::arg("segment_ids") = pybind11::none());
//   m.def("attention_backward_preprocess", &attention_backward_preprocess,
//         "Attention CUDA backward preprocess (strided FP32 inputs)",
//         pybind11::arg("dO"), pybind11::arg("O"));
//   m.def("attention_backward_dQ", &attention_backward_dQ,
//         "Attention CUDA backward dQ (strided FP32 inputs)",
//         pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
//         pybind11::arg("dO"), pybind11::arg("LSE"), pybind11::arg("D"),
//         pybind11::arg("scale"));
//   m.def("attention_backward_dKV", &attention_backward_dKV,
//         "Attention CUDA backward dK/dV (strided FP32 inputs)",
//         pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
//         pybind11::arg("dO"), pybind11::arg("LSE"), pybind11::arg("D"),
//         pybind11::arg("scale"));
}
