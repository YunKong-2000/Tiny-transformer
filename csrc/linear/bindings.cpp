#include "linear.h"
#include "linear_benchmark.h"
#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<LinearBenchmark, std::shared_ptr<LinearBenchmark>>(m, "LinearBenchmark")
        .def("run", &LinearBenchmark::run)
        .def_readonly("output", &LinearBenchmark::output)
        .def_readonly("dx", &LinearBenchmark::dx)
        .def_readonly("dweight", &LinearBenchmark::dweight)
        .def_readonly("workspace", &LinearBenchmark::workspace)
        .def_readonly("forward_kind", &LinearBenchmark::forward_kind)
        .def_readonly("split_k_slices", &LinearBenchmark::split_k_slices)
        .def_readonly("tiles", &LinearBenchmark::tiles);
    m.def("prepare_linear_benchmark", &prepare_linear_benchmark,
          "Prepare preallocated FP32 GEMMs for CUDA Graph kernel timing",
          pybind11::arg("x"), pybind11::arg("weight"),
          pybind11::arg("gradient"), pybind11::arg("backward"));
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
