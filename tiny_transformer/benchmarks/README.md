# 统一性能测试

所有性能测试实现集中在本目录。八个算子共用校验、计时和 JSON 格式；
模型端到端测试保留 TTFT/TPOT 等请求指标。

| 文件 | 职责 |
|---|---|
| `common.py` | 前向/逐输入梯度检查、保留图反向、交替计时、样本与中位数 |
| `cases.py` | 八个算子的形状、stride、常量参数、上游梯度缩放 |
| `operators.py` / `__main__.py` | 统一 CLI、已实现阶段判断、结果输出 |
| `embedding.py` | 四种 ID 分布和 embedding 专用 CLI；调用统一 runner |
| `linear.py` | FP32 Linear 的独立 FP64 对照、逐元素舍入误差界和 RMS 校验；不参与计时 |
| `linear_kernels.py` | Linear 预分配 GEMM、CUDA Graph/Event 分项计时及 FLOP/s |
| `model.py` | 完整模型请求的冷启动、稳态 TTFT/TPOT、吞吐与显存 |

原 `tiny_transformer.benchmark_embedding` 和 `tiny_transformer.benchmark` 仅保留转发，
没有另一份计时实现。新结果统一使用 `candidate_us/candidate_trials_us`，
替代旧 embedding JSON 的 `student_us/student_trials_us`；`backend` 标识候选实现。

## 使用方法

在 CUDA 开发环境从仓库根目录运行；A100 可设置 `TORCH_CUDA_ARCH_LIST=8.0`。
默认 FP32、`B=8,T=512,H=768,V=8192`、20 次预热、每轮 100 次调用、5 轮采样。
Linear 默认覆盖五种投影的训练前向/反向、prefill 前向、decode 前向。
默认 `--linear-timing kernel` 分别测 forward、dX、dWeight 的 GPU 执行区间；
`--linear-timing operator` 保留原有完整调用的计时方式。
其他算子保持 prefill/decode 和 forward/backward；只支持前向时，backward 明确跳过。

```bash
# RMSNorm：前向与反向，包含连续、末维 stride=2 和 last-only prefill。
python -m tiny_transformer.benchmarks --operator rms_norm \
  --layouts contiguous strided last-only --phases forward backward \
  --output runs/rmsnorm-performance.json

# Embedding：同一输入/上游梯度，对比四种分布与两个反向实现。
python -m tiny_transformer.benchmarks --operator embedding \
  --backward-impl all --patterns random same unique hot \
  --output runs/embedding-performance.json

# Linear：五种投影 × 训练/prefill/decode，包含连续和跨步输入。
python -m tiny_transformer.benchmarks --operator linear --precision fp32 \
  --layouts contiguous strided --output runs/linear-performance.json

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
| linear | QKV/O/Gate-Up/Down/LM head 的 `[B,T,K]`、`[N,K]`；训练与推理分别生成 | x、weight（仅训练反向） | FP32 SIMT 前向/反向；wrapper 复制跨步输入；尚不支持 AMP/BF16/FP16 |
| rms_norm | `[B,T,H]`、`[H]`、eps；可测 last-only `[B,1,H]` | x、weight | FP32 前向/反向（H <= 1024）；wrapper 复制跨步输入 |
| rope | `[B,heads,T,H/heads]` 与共享 cos/sin | x；cos/sin 是常量 | FP32 CuTe 前向/反向；直接消费输入及梯度的 stride |
| attention | prefill `Q=K=T`；decode `Q=1,K=seq_length`，携带 past_len | q、k、v | 未实现；可用 SDPA 比较 |
| swiglu | `[B,T,hidden_dim]` gate/up；跨步用例保留 chunk view | gate、up | FP32 CuTe 前向/反向；支持独立输入 stride 和非连续/零 stride 上游梯度 |
| residual | 两个 `[B,T,H]` | 两项输入 | FP32 kernel 前向/反向；wrapper 支持 FP16/BF16 与跨步输入，计入转换/复制成本 |
| cross_entropy | `[B,T,V]` logits、含 ignore_index 的 targets | logits | FP32 kernel 前向/反向；wrapper 支持 FP16/BF16 与跨步输入，返回 FP32 平均 loss，计入转换/复制成本 |

`--dim/--heads/--out-features/--hidden-dim/--vocab-size` 调整对应形状；
RoPE 要求 head_dim 是偶数。`--seq-length` 在 decode 中仍决定 attention 的 KV 长度。
`--layouts contiguous` 使所有输入连续；`strided` 将浮点输入的末维 stride 设为 2，
并保留 RoPE transpose/SwiGLU chunk 布局；`last-only` 仅对 RMSNorm 有效。
JSON 逐项记录实际 shape、stride、storage offset 和 dtype，last-only 的 prefill 行数为 B。
这些是代表性性能场景，不替代各算子专属边界与模型正确性测试。

### Linear 的完整形状矩阵

默认命令每种 layout 生成 15 组 case、25 项成对计时：五种投影各有训练 forward/dx/dweight，
以及推理 prefill/decode forward。推理没有 backward。split-K 场景还增加分区 GEMM 和归约两项
student 独立计时，默认形状只有 decode Down 使用 split-K，因此共输出 27 项。
`--phases backward` 在 kernel 模式展开为 dx 和 dweight；每组先校验，再计时。
选择 `--linear-timing operator` 时仍是 20 项，训练 backward 为 autograd 的完整 dX+dWeight 耗时。

默认维度对应 model_60m 的 `B=8,T=512,H=768,I=2048,V=8192`（不自动读取配置文件）：

| 投影 | N | K | 训练 M | Prefill M | Decode M |
|---|---:|---:|---:|---:|---:|
| qkv | 2304 | 768 | 4096 | 4096 | 8 |
| o | 768 | 768 | 4096 | 4096 | 8 |
| gate_up | 4096 | 768 | 4096 | 4096 | 8 |
| down | 768 | 2048 | 4096 | 4096 | 8 |
| lm_head | 8192 | 768 | 4096 | 8 | 8 |

推理 LM head 在 prefill 也只处理最后一个 token，与生成路径 `last_only=True` 一致。
形状由 `--dim/--hidden-dim/--vocab-size/--batch-size/--seq-length` 控制；
`--inference-batch-size` 可单独设置推理 batch，省略时使用 `--batch-size`。
每种相同投影形状只测一次，不按层数重复，也不将 grad_accum 乘入 M。
这里的训练场景表示形状和前反向工作量；`--precision` 仍是实际输入 dtype，不是 AMP。

```bash
# 全覆盖，训练 batch=8、推理 batch=1。
python -m tiny_transformer.benchmarks --operator linear \
  --inference-batch-size 1 --output runs/linear-performance.json

# 只调训练中的 Down 和 LM head 反向。
python -m tiny_transformer.benchmarks --operator linear \
  --workloads train --linear-projections down lm_head --phases backward

# 只测五种 decode 投影的前向。
python -m tiny_transformer.benchmarks --operator linear --workloads decode

# 保留原始完整调用口径，包含分配、wrapper 和 autograd 调度。
python -m tiny_transformer.benchmarks --operator linear --linear-timing operator

# 手动指定一个非模型形状；out-features 仅用于 custom 投影。
python -m tiny_transformer.benchmarks --operator linear --linear-projections custom \
  --dim 1024 --out-features 3072 --workloads train
```

控制台显示 workload、projection、phase 和 GEMM 尺寸；JSON 每个 case 保存 `execution`、
`projection`、`x_shape`、`weight_shape`、`output_shape`、`gemm_shapes`。
训练 `gemm_shapes` 同时包含 forward=`[M,N,K]`、dx=`[M,K,N]`、dweight=`[N,K,M]`。
显式请求推理 `--phases backward` 会记录 skip；默认混合阶段请求会自动只测推理 forward。
`--operator all` 同样展开 Linear；`--workloads train` 在 all 中只作用于 Linear。

### Linear kernel 计时口径

- 输入复制/展平、输出和 workspace 分配、CUTLASS 参数初始化全部在计时外完成。
  原始 stride 保存在 `inputs`，实际参与 kernel 的连续输入记录在 `prepared_inputs`。
  因此 kernel 模式的 `strided` 用例不衡量复制成本；要衡量复制请使用 operator 模式。
- CUDA Graph 中预先捕获 `repeats` 次固定指针调用，正式计时只重放图。
  使用当前设备/stream 的 CUDA Event，按 trial 交替 reference/student 顺序，
  取每轮图执行时间除以 repeats 的中位数，保留所有原始样本。
  此口径排除 host 提交间隙，但包含图在 GPU 上的调度间隙，不等同于 profiler 的 kernel 独占时间。
  不清空缓存，是固定输入、热缓存下的吞吐测试，不代表模型端到端延迟。
- reference 使用预分配输出的 `torch.mm(out=...)`：forward=`X@W.T`、dx=`dY@W`、
  dweight=`dY.T@X`。CUDA 库可能为一次 GEMM 发射多个 kernel，reference 数值代表完整设备操作，
  不宣称某个 cuBLAS kernel 的独占耗时。reference-only 模式是基线自比较。
- student 普通 GEMM 直接运行初始化后的生产 kernel；split-K 的 forward 运行生产 device operator，
  包含分区 GEMM+归约。为分别测量，两阶段使用相同公开 kernel 类型和与 device initialize 相同的参数。
  校验时先验证 prepared 输出与正式算子一致，再作 FP64 对照；独立 split-K 阶段必须精确匹配生产结果。
  归约计时前先生成有效 workspace；禁止把未初始化部分和当输入。
- `forward` 的 split-K pipeline 耗时单独实测，不是两项独立测量中位数的和。
  `split_k_partials`/`split_k_reduce` 是 student 诊断项，没有伪造 reference 时间或 speedup。
- FLOP/s=`FLOPs/(us*1e-6)`，TFLOP/s=`FLOP/s/1e12`。forward/dx/dweight 各自使用
  `2*M*N*K`（这里的 M/N/K 为对应 GEMM 的维度，FMA 计 2）；仅计有效数学工作，不计填充和地址指令。
  split-K partials 和完整 forward 同样使用 `2*M*N*K`。归约按当前实现从零累加 S 个分区，
  使用 `S*M*N` 次加法，不计 epilogue 缩放，也不将它误算为完整 GEMM FLOPs。
  归约主要受访存/启动影响，其 FLOP/s 不适合直接与 GEMM 或理论峰值比较。

JSON schema 为 2。每项 forward/dx/dweight 保存 `flops`、`reference_flops_per_second`、
`candidate_flops_per_second`、两方 TFLOP/s、微秒、原始 trials 和 speedup；
split-K 分项保存 `flops_per_second`、`tflops_per_second` 及 student 微秒。
`kernel_config` 记录实际分派、CTA/Warp 形状、split 路数和 workspace 字节数。
kernel 模式与历史 operator 数据口径不同，不能把二者时间直接比较为优化收益。

当前 student 阶段表在 `operators.py::STUDENT_PHASES`，dtype/layout 限制在
`unsupported_reason`。实现新 kernel 后更新这些声明即可沿用统一流程。
已声明支持的路径发生编译错误、运行错误或数值错误时直接失败，不把错误转换成 skipped。
`unique` 在行数超过 V 时明确跳过，不取模制造重复 ID。

## 测量边界

以下为完整算子模式（其他算子及 Linear 的 `--linear-timing operator`）。
Linear 默认 kernel 模式遵循上方单独说明，并复用 FP64 校验规则。

1. 在计时前完成 JIT 编译和与 reference 的前向比较。需要测 backward 时，还检查每个可微输入的梯度。
   Embedding 前向要求精确相等；其他算子按 dtype 使用公用阈值，JSON 保存实际阈值与误差。
   FP32 Linear 例外：cuBLAS 与 CUTLASS 的累加顺序不同，接近零的点积可能不满足统一的逐元素绝对容差。
   两者分别与完整的 FP64 GEMM 对照，所有元素都须满足
   `abs(result - fp64) <= gamma_L * (abs(A) @ abs(B)) + L * float32.tiny`，
   其中 `u=2^-24`、`gamma_L=L*u/(1-L*u)`，L 是本次 GEMM 的归约长度。
   同时要求 `RMS(result-fp64) <= 1e-5 + 1e-4 * RMS(fp64)`，防止长归约的最坏误差界过于宽松。
   FP64 按输出行分块计算；前向、dX、dW 的 L 分别是 K、N、M。错误布局、非有限结果和超界误差直接失败。
   JSON 的 `validation_policy=fp64_dot_product_bound_and_rms`、`forward_fp64`、`backward_fp64`
   记录 candidate/reference 各自的最大绝对误差、最大误差/界限比和 RMS；这些检查不计入性能数据。
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
