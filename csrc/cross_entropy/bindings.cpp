#include "cross_entropy.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cross_entropy_forward", &cross_entropy_forward_cuda,
        "Cross entropy row losses and split LSE cache (CUDA FP32)",
        pybind11::arg("logits"), pybind11::arg("targets"));
  m.def("cross_entropy_backward", &cross_entropy_backward_cuda,
        "Cross entropy first-order backward (CUDA FP32, scalar tensor grad_loss)",
        pybind11::arg("logits"), pybind11::arg("targets"),
        pybind11::arg("lse"), pybind11::arg("grad_loss"));
}
