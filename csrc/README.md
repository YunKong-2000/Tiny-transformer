# CUDA / CuTe / CUTLASS 算子实现

每个算子目录包含 CUDA 源码、头文件和独立 `bindings.cpp`。PyTorch 入口位于 [student.py](../tiny_transformer/operators/student.py)，通过 [_extension.py](../tiny_transformer/operators/_extension.py) 延迟编译，在 `torch.autograd.Function` 中连接一阶反向。

## 模块

| 目录 | 实现 | 主要约束 |
| --- | --- | --- |
| `embedding/` | warp 查表；baseline / grouped 原子累加反向 | FP32 权重、连续 int64 IDs；不支持二阶梯度 |
| `linear/` | CUTLASS FP32 SIMT 前向、dX、dW；小 M / Split-K | 不支持 BF16/FP16/autocast；wrapper 处理非连续输入 |
| `rms_norm/` | warp 前向归约；缓存统计量的一阶反向 | FP32；反向 H≤1024 |
| `rope/` | CuTe pair 布局与前反向模板 | FP32；支持 stride；cos/sin 不求梯度 |
| `attention/` | FP32 SIMT / BF16 Tensor Core 前向；BF16 preprocess/dQ/dK/dV | D=64；BF16 SM80+；无 autocast、二阶梯度 |
| `swiglu/` | gate/up 融合与双输入梯度 | FP32；支持 chunk view 和非连续上游梯度 |
| `residual/` | FP32 向量/标量加法及双路梯度 | wrapper 提供低精度转换、dtype promotion |
| `cross_entropy/` | 逐行稳定归一化、LSE 缓存和反向 | FP32 kernel；wrapper 支持低精度 logits、ignore_index |
| `cutlass_smoke/` | CUTLASS/CuTe 编译与运行检查 | 依赖完整子模块及 CUDA 工具链 |

BF16 Attention 反向按当前 stream 执行 preprocess → dQ → dK/dV，保存前向 Q/K/V/O/LSE。已具备完整接口与测试用例，GPU 验收记录范围见 [验证说明](../docs/validation.md)。

## 构建与调用

```text
Python public function
  → torch.autograd.Function（需要梯度时）
  → 单算子 JIT loader
  → pybind / C++ 参数校验
  → 当前 CUDA stream 上的 kernel
```

首次加载需要 CUDA 版 PyTorch、nvcc、C++ 编译器、Ninja，以及初始化后的 `third_party/cutlass`。扩展统一使用 C++17。A100 对应 `TORCH_CUDA_ARCH_LIST=8.0`。目前要求源码 checkout 或 editable install，普通 wheel 不保证包含 `csrc`。

```bash
python -m tiny_transformer.check_cutlass
python -m unittest discover -s tests -p 'test_student_*.py' -v
python -m tiny_transformer.benchmarks --operator embedding \
  --backward-impl all --output runs/embedding-performance.json
```

Python wrapper 负责必要的布局转换与 autograd 状态；native kernel 的输入契约不能通过直接调用 pybind 绕过。原子累加存在浮点非确定性，相关路径在严格 deterministic 模式下显式拒绝。

## 性能测试与实现记录

Linear 提供 `prepare_linear_benchmark()`，预分配 tensor/workspace 并测量 forward、dX、dW 及 Split-K 两阶段；它不是模型 `torch.compile` 注册接口。其他测量边界见 [benchmarks](../tiny_transformer/benchmarks/README.md)。

- [接口与支持矩阵](../docs/operators/README.md)
- [Attention 实现记录](attention/development.md)
- [Linear 配置及结果](linear/development.md)
- [性能记录索引](../docs/performance.md)
- [扩展开发规范](../docs/development.md)
