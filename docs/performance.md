# 性能记录与测量方法

## BF16 Attention 前向

平台：NVIDIA A100 80GB PCIe。B=8、H=12、D=64；warmup=20、repeats=100、trials=5。耗时单位为 μs。

| 场景 | Tq / Tk | 自定义实现 | SDPA | SDPA / 自定义 |
| --- | --- | ---: | ---: | ---: |
| Prefill | 512 / 512 | 39.78 | 39.35 | 0.989× |
| Decode | 1 / 512 | 12.73 | 15.44 | 1.213× |

数据来源：[测量汇总](reports/attention_bf16_measurements.json) 的 G11 / D03，以及 [优化报告](reports/attention_bf16_optimization_report.md)。报告记录日期为 2026-10-07 至 2026-10-08，数据来自保存的终端日志和 profiler 记录，未在文档整理环境重新测量。

- Decode 为明确的 CUDA Graph 记录；G11 沿用 Graph 上下文，但结果片段缺完整 CLI，不能视为完整独立复现记录。
- 历史 profiler 记录确认当轮 SDPA 使用 FlashAttention 后端；当前 `--baseline sdpa` 允许 PyTorch 自动选择后端，不强制绑定该实现。
- 数值仅覆盖上述形状，不表示整个模型的加速，也不覆盖反向性能。BF16 反向已实现 preprocess/dQ/dK/dV 并接入 autograd；目前未附完整 GPU 验收及耗时报告。
- 不混合 operator、Graph 和挂载 Nsight Compute 的时间计算优化收益。报告保留未获收益及回退实验。

## 实现与性能证据索引

| 算子 | 实现重点 | 现有记录及解释 |
| --- | --- | --- |
| Attention | Tensor Core、online softmax、swizzle、异步搬运、decode 分块 | [优化报告](reports/attention_bf16_optimization_report.md)；固定形状前向结果 |
| Linear | 常规/小 M/Split-K、前向/dX/dW 分项 | [开发记录](../csrc/linear/development.md)；早期 operator 与后期 Graph 需分别解读 |
| Embedding | 向量化查表、atomic baseline 与 warp 内分组 | [对照记录](../csrc/embedding/development&test.md)；grouped 在已记录输入分布中未取得预期收益 |
| RMSNorm | warp reduction、缓存统计量、参数梯度 | [实现记录](../csrc/rms_norm/rms_norm.md)；基线为 PyTorch eager 组合实现 |
| RoPE | CuTe 索引、stride、前反向模板 | [测量记录](../csrc/rope/development_test_report.md)；区分 prefill/decode |
| SwiGLU | chunk view、逐元素融合与反向 | [测量记录](../csrc/swiglu/develop_test.md)；保留慢于基线的形状 |

## 复现入口

```bash
# Attention：固定 BF16 形状的 Graph 前向对照。
python -m tiny_transformer.benchmarks --operator attention \
  --backend student --baseline sdpa --precision bf16 --phases forward \
  --workloads prefill decode --layouts contiguous --attention-timing graph \
  --batch-size 8 --heads 12 --dim 768 --seq-length 512 \
  --warmup 20 --repeats 100 --trials 5 --output runs/attention-bf16-graph.json

# 反向：完整算子调用，包括 preprocess、dQ、dK/dV。
python -m tiny_transformer.benchmarks --operator attention \
  --backend student --baseline sdpa --precision bf16 --phases forward backward \
  --workloads prefill decode --layouts contiguous strided \
  --output runs/attention-bf16-forward-backward.json

# Linear：前向、dX、dW 的预分配 GPU 路径。
python -m tiny_transformer.benchmarks --operator linear --precision fp32 \
  --linear-timing kernel --output runs/linear-kernel-performance.json
```

以上命令是当前入口，不声称能逐字复现缺失完整命令的历史记录。新结果需保存源码 commit、设备/软件版本、原始 trial、误差、输入布局和计时范围。

## 计时约定

- **Operator**：完整调用，计入输出分配、必要的连续化/类型转换及 dispatch；CUDA Event 区间可能包含主机提交间隙。
- **CUDA Graph**：预热后捕获固定输入调用，以图重放测量 GPU 执行区间；包含实际设备复制、清零和调度间隙，排除捕获阶段的主机工作。
- **模型指标**：TTFT、TPOT、token/s 和峰值显存独立记录，不能由单算子速度比推算。
- 正确性检查和 JIT 编译在计时外完成；比较同 dtype、同输入、同精度与等价后处理。speedup 为 baseline/candidate。

完整参数说明见 [benchmarks](../tiny_transformer/benchmarks/README.md)。
