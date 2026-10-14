# Tiny Transformer

面向 Transformer 训练与推理负载的 **CUDA 算子实现与性能优化项目**。以 decoder-only Transformer 为计算载体，在 PyTorch 参考实现之外提供 CUDA C++、CUTLASS 和 CuTe 后端，覆盖 GEMM、归约、逐元素融合及因果 Attention 的前反向。

主要开发与测量平台：**NVIDIA A100 80GB PCIe / SM80**。参考配置为 8 层、hidden size 768、12 个 attention heads、8K 词表，约 62.93M 参数。

## 主要实现

- **BF16 Attention 前反向**：CuTe Tensor Core 融合 QKᵀ、causal mask、online softmax 和 PV；反向分为 preprocess、dQ、dK/dV，利用前向 LSE 分块重算概率，并接入一阶 autograd。
- **Prefill / decode 分工**：prefill 使用异步搬运、共享内存 swizzle、寄存器预算与完整 tile 特化；单 token decode 使用 KV 分块和 FP32 softmax 合并。
- **模型形状驱动的 Linear**：CUTLASS FP32 SIMT 前向、dX、dW；针对常规矩阵、小 M 和 Split-K 分别配置执行路径。
- **完整算子调用边界**：处理 stride、alignment、尾块、当前 CUDA stream 和梯度接口，提供数值对照、模型接入测试及独立性能入口。

自定义后端的现有 API 名为 `student`，例如 `--op linear=student`。本文称其为“自定义 CUDA 后端”；命令与源码标识保持一致。`reference` 为 PyTorch 数学参考，Attention 另提供 `sdpa` 基线。

## 性能摘要

下表为已保存的 **BF16 Attention 前向**测量，B=8、H=12、D=64，平台为 A100 80GB PCIe。速度比定义为 `SDPA 耗时 / 自定义实现耗时`。

| 负载 | Tq / Tk | 自定义实现 | 同轮 SDPA | 速度比 |
| --- | --- | ---: | ---: | ---: |
| Prefill | 512 / 512 | 39.78 μs | 39.35 μs | 0.989× |
| 单 token decode | 1 / 512 | 12.73 μs | 15.44 μs | 1.213× |

Decode 使用 CUDA Graph 计时；prefill 最新记录沿用同一 Graph 测量上下文，但结果片段未附完整 CLI。以上是固定形状的历史记录，不代表所有形状或当前工作区复测结果。BF16 反向已有实现及接入，尚未附完整 GPU 验收和性能报告。

[性能口径与复现](docs/performance.md) · [Attention 优化报告](docs/reports/attention_bf16_optimization_report.md) · [原始测量汇总](docs/reports/attention_bf16_measurements.json)

## 算子支持范围

下表描述源码接口范围；实现、测试覆盖和已执行的验证记录分别见对应文档。

| 算子 | 自定义前向 / 一阶反向 | 关键边界 |
| --- | --- | --- |
| [Embedding](docs/operators/embedding.md) | FP32 / FP32 | 连续 int64 IDs 与 FP32 权重；baseline / grouped 反向 |
| [Linear](docs/operators/linear.md) | FP32 SIMT / FP32 SIMT | 常规、小 M、Split-K；不支持 autocast |
| [RMSNorm](docs/operators/rms_norm.md) | FP32 / FP32 | 反向 H≤1024；非连续输入显式复制 |
| [RoPE](docs/operators/rope.md) | FP32 / FP32 | CuTe；支持 stride；cos/sin 作为常量 |
| [Attention](docs/operators/attention.md) | FP32、BF16 / BF16 | D=64；BF16 要求 SM80+；无 autocast；segments 仅 FP32 前向 |
| [SwiGLU](docs/operators/swiglu.md) | FP32 / FP32 | 支持 chunk view 和非连续梯度 |
| [Residual](docs/operators/residual.md) | FP32 kernel / FP32 kernel | wrapper 支持 FP16/BF16 转换及 dtype promotion |
| [Cross Entropy](docs/operators/cross_entropy.md) | FP32 kernel / FP32 kernel | wrapper 接受低精度 logits；ignore_index=-100；输出 FP32 |

范围外输入会显式报错，不自动回退到参考实现。单个算子的 BF16 支持不等于全自定义后端已支持 BF16 模型训练。项目不包含分布式训练、量化、paged attention 或在线服务调度。

## 环境与快速运行

需要 Linux CUDA 开发环境、CUDA 版 PyTorch、nvcc、C++ 编译器及 Ninja。参考容器为 `nvcr.io/nvidia/pytorch:25.08-py3`，CUTLASS 子模块固定为 v3.9.2。

```bash
git clone --recurse-submodules https://github.com/YunKong-2000/Tiny-transformer.git
cd Tiny-transformer
# 已有 checkout 可执行：git submodule update --init --recursive third_party/cutlass

docker run --rm -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/tiny-transformer -w /workspace/tiny-transformer \
  nvcr.io/nvidia/pytorch:25.08-py3 bash
```

容器内：

```bash
python -m pip install -e '.[data]'
export TORCH_CUDA_ARCH_LIST=8.0
python -m tiny_transformer.check_cutlass

# 数值/梯度和模型接入回归；首次调用会 JIT 编译扩展。
python -m unittest discover -s tests -p 'test_student_*.py' -v

# BF16 Attention 前向与同 dtype SDPA 对照。
python -m tiny_transformer.benchmarks --operator attention \
  --backend student --baseline sdpa --precision bf16 --phases forward \
  --workloads prefill decode --layouts contiguous --attention-timing graph \
  --batch-size 8 --heads 12 --dim 768 --seq-length 512 \
  --warmup 20 --repeats 100 --trials 5 --output runs/attention-bf16-graph.json
```

GPU 用例在无 CUDA 环境会跳过。CPU smoke 验证框架流程，不作为 CUDA 正确性或性能证据：

```bash
bash scripts/smoke.sh
```

## 模型运行

参考框架支持 TinyStories / 本地 JSONL 数据准备、训练、断点续训、KV cache 生成、TTFT/TPOT 测量及 profiling。`data/`、`runs/` 和 checkpoint 不随仓库分发，需按流程生成。

```bash
python -m tiny_transformer.prepare --source tinystories --output data/tinystories-8k \
  --train-docs 100000 --val-docs 2000 --tokenizer-docs 20000 --vocab-size 8192
python -m tiny_transformer.train --config configs/model_60m.json \
  --data data/tinystories-8k --output runs/60m-sdpa --device cuda --op attention=sdpa
```

[训练与推理流程](docs/model_workflow.md) · [离线环境](docs/offline.md) · [依赖配置](third_party/README.md)

## 文档与代码导航

| 路径 | 内容 |
| --- | --- |
| [docs/](docs/README.md) | 文档入口、阅读路径与测量记录 |
| [csrc/](csrc/README.md) | CUDA / CuTe / CUTLASS kernel 和 C++ 绑定 |
| [operators/student.py](tiny_transformer/operators/student.py) | 自定义算子接口与一阶 autograd |
| [operators/reference.py](tiny_transformer/operators/reference.py) | 数学参考实现 |
| [benchmarks/](tiny_transformer/benchmarks/README.md) | 算子和模型性能测试 |
| [tests/](tests/README.md) | 数值、梯度、边界和模型集成测试 |
| [算子接口](docs/operators/README.md) | shape、dtype、stride 与反向返回项 |
| [开发规范](docs/development.md) | 扩展接入、stream、数值验证和测量约定 |

底层实现参考 [CUTLASS / CuTe](https://github.com/NVIDIA/cutlass) 和 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 的相关设计；依赖及上游许可保留在 `third_party/` 中。
