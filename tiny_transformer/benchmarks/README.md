# 统一性能测试

所有性能测试实现集中在本目录。八个算子共用校验、计时和 JSON 格式；
模型端到端测试保留 TTFT/TPOT 等请求指标。

| 文件 | 职责 |
|---|---|
| `common.py` | 前向/逐输入梯度检查、保留图反向、交替计时、样本与中位数 |
| `cases.py` | 八个算子的形状、stride、常量参数、上游梯度缩放 |
| `operators.py` / `__main__.py` | 统一 CLI、已实现阶段判断、结果输出 |
| `embedding.py` | 四种 ID 分布和 embedding 专用 CLI；调用统一 runner |
| `model.py` | 完整模型请求的冷启动、稳态 TTFT/TPOT、吞吐与显存 |

原 `tiny_transformer.benchmark_embedding` 和 `tiny_transformer.benchmark` 仅保留转发，
没有另一份计时实现。新结果统一使用 `candidate_us/candidate_trials_us`，
替代旧 embedding JSON 的 `student_us/student_trials_us`；`backend` 标识候选实现。

## 使用方法

在 CUDA 开发环境从仓库根目录运行；A100 可设置 `TORCH_CUDA_ARCH_LIST=8.0`。
默认 FP32、`B=8,T=512,H=768,V=8192`、20 次预热、每轮 100 次调用、5 轮采样。
默认分别测 prefill 和 decode，并请求 forward/backward。只支持前向时，backward 明确跳过。

```bash
# RMSNorm：前向与反向，包含连续、末维 stride=2 和 last-only prefill。
python -m tiny_transformer.benchmarks --operator rms_norm \
  --layouts contiguous strided last-only --phases forward backward \
  --output runs/rmsnorm-performance.json

# Embedding：同一输入/上游梯度，对比四种分布与两个反向实现。
python -m tiny_transformer.benchmarks --operator embedding \
  --backward-impl all --patterns random same unique hot \
  --output runs/embedding-performance.json

# 所有 student 算子：未实现的算子/阶段/dtype 明确记录 skipped，无 fallback。
python -m tiny_transformer.benchmarks --operator all \
  --output runs/all-operators-performance.json

# 用 reference 验证八个算子的测量流程；这是基线自比较，不是优化收益。
python -m tiny_transformer.benchmarks --operator all --backend reference \
  --output runs/reference-operators-performance.json

# Attention 已有 SDPA 候选，可以独立比较前向、反向和缓存 decode。
python -m tiny_transformer.benchmarks --operator attention --backend sdpa \
  --precision bf16 --output runs/attention-sdpa-performance.json

# 整模型请求指标。
python -m tiny_transformer.benchmarks.model --config configs/smoke.json \
  --device cuda --precision fp32 --op rms_norm=student \
  --prompt-length 16 --new-tokens 8 --output runs/rmsnorm-inference.json
```

`python -m tiny_transformer.benchmarks.embedding` 是统一 CLI 的快捷入口，默认只测 prefill，
保留原 embedding 默认行为；可传 `--workloads prefill decode`。
所有入口都可用 `--help` 查看选项。

## 相同的方法，不同的输入契约

| 算子 | 输入/测试维度 | 反向需要的输入 | 当前 student |
|---|---|---|---|
| embedding | `[B,T]` IDs、`[V,H]` weight；random/same/unique/hot；grouped/baseline | weight | FP32 前向、反向，连续输入 |
| linear | `[B,T,H]`、`[out_features,H]` | x、weight | 未实现 |
| rms_norm | `[B,T,H]`、`[H]`、eps；可测 last-only `[B,1,H]` | x、weight | FP32 前向/反向（H <= 1024）；wrapper 复制跨步输入 |
| rope | `[B,heads,T,H/heads]` 与共享 cos/sin | x；cos/sin 是常量 | 未实现 |
| attention | prefill `Q=K=T`；decode `Q=1,K=seq_length`，携带 past_len | q、k、v | 未实现；可用 SDPA 比较 |
| swiglu | `[B,T,hidden_dim]` gate/up；跨步用例保留 chunk view | gate、up | 未实现 |
| residual | 两个 `[B,T,H]` | 两项输入 | FP32 kernel 前向/反向；wrapper 支持 FP16/BF16 与跨步输入，计入转换/复制成本 |
| cross_entropy | `[B,T,V]` logits、含 ignore_index 的 targets | logits | 未实现 |

`--dim/--heads/--out-features/--hidden-dim/--vocab-size` 调整对应形状；
RoPE 要求 head_dim 是偶数。`--seq-length` 在 decode 中仍决定 attention 的 KV 长度。
`--layouts contiguous` 使所有输入连续；`strided` 将浮点输入的末维 stride 设为 2，
并保留 RoPE transpose/SwiGLU chunk 布局；`last-only` 仅对 RMSNorm 有效。
JSON 逐项记录实际 shape、stride、storage offset 和 dtype，last-only 的 prefill 行数为 B。
这些是代表性性能场景，不替代各算子专属边界与模型正确性测试。

当前 student 阶段表在 `operators.py::STUDENT_PHASES`，dtype/layout 限制在
`unsupported_reason`。实现新 kernel 后更新这些声明即可沿用统一流程。
已声明支持的路径发生编译错误、运行错误或数值错误时直接失败，不把错误转换成 skipped。
`unique` 在行数超过 V 时明确跳过，不取模制造重复 ID。

## 测量边界

1. 在计时前完成 JIT 编译和与 reference 的前向比较。需要测 backward 时，还检查每个可微输入的梯度。
   Embedding 前向要求精确相等；其他算子按 dtype 使用公用阈值，JSON 保存实际阈值与误差。
   RMSNorm 前向验证/计时使用 `no_grad`；反向计时保存每图的 R 并直接复用。H > 1024 时跳过反向。
2. 两个后端都先预热；CUDA events 在计时区间外完成首次初始化。
   每轮用当前设备/stream 的 events 包围 repeats 次调用，得到每次调用的平均微秒数；
   交替 reference/candidate 顺序，最后报告 trials 轮的中位数、全部原始样本和加速比。
3. forward 包含输出分配、Python dispatch 和 wrapper 内的显式复制。
   backward 调用 `autograd.grad(..., retain_graph=True)`，复用计时前的图，
   包含梯度分配/清零、autograd 调度和梯度计算；不包含前向、不累积到叶子的 `.grad`。
4. 两边使用完全相同的输入与上游梯度。一般算子使用正态上游梯度除以本次输出的 token 行数；
   cross entropy 已自行取均值，因此使用单位尺度的随机标量上游梯度。
   Embedding grouped/baseline 使用同一份输入和相同随机种子生成的上游梯度。
5. 独立算子基准使用 FP32 `highest` matmul precision，退出后恢复；环境字段记录 TF32 状态。
   `--precision` 指实际输入 dtype，不是 AMP 模型训练模式。
6. 这是固定输入、热缓存/allocator 的 eager stream 区间，可能含 CPU 提交产生的 GPU 空隙，
   不能宣称纯 kernel 时间。模型请求基准使用同步后的主机时钟，因为 TTFT/TPOT 的测量目标不同。

输出的每个 forward/backward 字段均带 `status`。已测阶段包含
`reference_us`、`candidate_us`、`speedup`、`reference_trials_us`、`candidate_trials_us`；
跳过阶段只带原因，没有伪造的 0 延迟。`speedup=reference_us/candidate_us`。

`check_ops.py` 仍是小形状正确性检查；CUDA 前向时延也调用本目录的同一计时函数，
`--backward` 只增加梯度正确性检查。反向性能请用统一 CLI。
CPU 上的 check_ops 使用多轮主机时钟，并标注为 smoke diagnostics；独立算子性能 CLI 只接受 CUDA。

## 测试

```bash
python -m unittest discover -s tests -p 'test_benchmarks.py' -v
```

主机测试覆盖八个算子的输入/梯度契约、非连续布局、反向不重跑前向或积累 `.grad`、
错误结果拒绝、CUDA event 交替顺序/单位换算/中位数、未实现路径和变体间的输入公平性。
该文件只测试公共框架，不重复跑学生 kernel 的数值或性能。
学生 kernel 正确性运行 `python -m unittest discover -s tests -p 'test_student_*.py' -v`；
真实性能运行上方 benchmark CLI，其中已包含计时前的数值校验。完整测试分工见 [测试说明](../../tests/README.md)。
