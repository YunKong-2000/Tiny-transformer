#pragma once

#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Benchmark-only prepared calls. Owners keep all captured pointers alive.
struct LinearBenchmark {
  torch::Tensor x, weight, gradient, output, dx, dweight, workspace;
  std::string forward_kind;
  int split_k_slices = 1;
  std::map<std::string, std::vector<int64_t>> tiles;
  std::map<std::string, std::function<void(cudaStream_t)>> calls;
  void run(const std::string& name);
};

std::shared_ptr<LinearBenchmark> prepare_linear_benchmark(
    torch::Tensor x, torch::Tensor weight, torch::Tensor gradient, bool backward);
