# linear：五类投影共用的 CUTLASS 入口

[返回算子总契约](README.md) · [参考实现](../../tiny_transformer/operators/reference.py) · [调用位置](../../tiny_transformer/model.py)

当前 student 实现为 **CUDA FP32 SIMT 前向和一阶反向**，权重仍为 `[N,K]`。
Python 入口支持 `[...,K]`，显式复制非连续输入并恢复输出形状，复制保留梯度链；
原生入口接收连续三维 FP32 输入。支持零 M/N/K，使用当前 CUDA stream，检查 int32 GEMM 维度范围。
BF16/FP16 和 CUDA autocast 目前明确拒绝，尚未实现 `torch.compile` 自定义算子注册。
下面的 AMP/BF16 要求描述完整目标契约，不代表当前已支持。

正确性测试（有 CUDA 时会真实编译扩展；无 CUDA 时只执行 CPU 接线测试）：

```bash
python -m unittest discover -s tests -p 'test_student_linear.py' -v
python -m unittest discover -s tests -p 'test_student_integration.py' -v
```

Linear 测试覆盖五种默认训练投影、batch=8/1 的 decode、非方形和尾块、
输入/上游梯度的 stride、空维度、非法输入和非默认 stream。
GPU 编译、数值和性能仍需在目标 CUDA 主机验收；CPU 测试替身不验证 CUTLASS。

FP32 性能基准和 Linear CUDA 测试使用独立 FP64 结果校验 reference 和 student。
由于累加顺序不同，不能仅凭接近零的元素相对误差很大就认定布局错误。
每个结果须同时满足逐元素点积舍入界和 RMS 限制；性能 JSON 保留两种实现各自的误差。
完整公式见 [性能测量说明](../../tiny_transformer/benchmarks/README.md#测量边界)。
五种投影的测试输入使用未缩小的标准正态分布，与性能基准一致，避免小幅值输入掩盖舍入问题。

小 M 的 K_tile=8 基线使用 `CTA<8,32,8>`、`Warp<8,32,8>`（每 CTA 一个 warp）。
当前 K_tile=16 调优配置使用 `CTA<8,32,16>`、`Warp<8,16,16>`（每 CTA 两个 warp），
尚需 GPU 编译、正确性及性能验证。K_tile=16 时共享内存 padding 为 2，必须同时能被 LaneM/LaneN 整除；
`Warp<8,32,16>` 的 LaneN=4 不满足约束，不能只增大原配置的 K。
当前两级、标量访问的 SIMT 路径需要操作数 tile 的元素数能被 CTA 线程数整除，头文件已增加检查。
例如 `Warp<4,16,8>` 会产生 128 个线程，但 A tile 只有 64 个元素；这版共享内存写入
不屏蔽多余线程，会覆盖有效 A 数据。`can_implement` 不会检查这种内部线程映射问题。
新增小整数精确结果测试覆盖共享内存 stage 复用、尾块和 `[8,8192,768]` 的 LM head 场景。

前向在 `M<128 && K>=2048` 时使用两路 `GemmSplitKParallel`，其余小 M 使用普通 GEMM，
大 M 配置和反向不变。部分和与归约均为 FP32；workspace 通过 PyTorch allocator 在当前流分配，
不额外清零或同步，分区 GEMM 和最终归约均在当前流执行。两路所需 workspace 为 `2*M*N*sizeof(float)`。
测试覆盖 `K=2047/2048/2049`、`M=1/8/127/128`、奇数 N、两分区与最后一个 K 元素的精确贡献，
以及非默认流和重复调用下的 workspace 生命周期。split-K 修改仍需在 CUDA 主机编译与验证。

## 1. 接口与职责

```python
linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor
```

此入口会被 QKV、Attention O、合并 gate/up、FFN down 和 LM head 共用。
选择 `linear=student` 后，所有这些投影都会调用你的实现，没有“只替换 QKV”的隐含限制。
每个 block 4 次，最后 LM head 1 次；默认模型每次 forward 共 33 次。

## 2. 输入与输出

| 项目 | 含义 | shape | 梯度 |
|---|---|---|---|
| x | 本次投影的输入激活 | 模型中为 $[B,T,K]$；一般可记为 $[\ldots,K]$ | 训练需要 dx |
| weight | 按输出行存储的参数 | $[N,K]$，不是 $[K,N]$ | 训练需要 dweight |
| 返回值 y | 未经过非线性的投影结果 | $[B,T,N]$ 或 $[\ldots,N]$ | 接入下游计算图 |

张量位于同一设备。无 autocast 时，本项目输入与 weight 使用同一种浮点 dtype，输出保持该 dtype。
输出前导维度必须保留，不能把内部展平后的二维结果直接返回。

## 3. 数学语义和边界

将前导维度展平为 $M$ 行：

$$
Y_{M\times N}=X_{M\times K}W_{N\times K}^{\top},\qquad M=B T.
$$

等价于 GEMM 的 $\alpha=1,\ \beta=0$：没有额外 C 输入、bias、残差、激活或量化 scale。
也不执行 QKV 拆分、head transpose、RoPE、SwiGLU 或 softmax。

CUTLASS 中描述逻辑 B 操作数时，要同时考虑 weight 的原始 $[N,K]$ 存储与转置访问。
不能为了得到逻辑 $[K,N]$ 就假设传进来的 tensor 已经物理转置。

## 4. 默认模型里的五种调用

| 角色 | x shape | weight shape | y shape | 下游用途 |
|---|---|---|---|---|
| QKV | $[8,512,768]$ | $[2304,768]$ | $[8,512,2304]$ | reshape、拆出 Q/K/V |
| O projection | $[8,512,768]$ | $[768,768]$ | $[8,512,768]$ | 与 residual 相加 |
| Gate/Up | $[8,512,768]$ | $[4096,768]$ | $[8,512,4096]$ | 沿末维均分 gate 与 up |
| Down | $[8,512,2048]$ | $[768,2048]$ | $[8,512,768]$ | 与 residual 相加 |
| LM head | $[8,512,768]$ | $[8192,768]$ | $[8,512,8192]$ | 训练交叉熵 |

prefill 中 block 投影的 $M=B T_{\mathrm{prompt}}$；当前推理使用 `last_only=True`，
LM head 只处理最后一个位置，所以 LM head 的 $M=B$。
逐 token decode 中，所有投影的 $M=B$。训练 GEMM 的最优配置不能直接用于小 batch decode。
测试还使用 64/96 等小维度，第一版只支持默认模型时要写清楚为何测试用例可能被拒绝。

## 5. AMP 是这个接口的重要边界

BF16 AMP 训练的实际入口：

| 调用 | x dtype | weight dtype | 参考输出 dtype |
|---|---|---|---|
| QKV / Gate-Up / LM head | FP32 | FP32 | BF16 |
| O / Down | BF16 | FP32 | BF16 |

这是 `F.linear` 在 autocast 区间内的行为，而不是“参数已经全部变成 BF16”。
扩展 wrapper 需要明确遵循 autocast 目标精度：内部进行必要转换，选用对应 GEMM，并保留正确的反向转换链。
不能按 BF16 解释 FP32 指针，也不能永久把 FP32 master 参数改成 BF16。
无 autocast 的 BF16 inference 则是 x、weight、y 都为 BF16。

FP32 路径要遵守实验的 TF32 设置。初始 GEMM 建议 FP32 accumulation、输出按当前计算 dtype 转换。
FP64 不属于首阶段训练要求；若未支持，应明确拒绝而非默默转 FP32。

## 6. Stride 和参数共享

当前 reference 路径传入此算子的主要激活通常是连续的，参数是连续二维矩阵。
但不能从 shape 推断连续性：其他 student 算子可能返回不同布局，attention 的 transpose/reshape 也可能产生复制。
可先声明支持连续 x/weight，在 wrapper 中检查；额外 `.contiguous()` 的时间与分配需要纳入模型测量。

LM head 的 weight 与 embedding 表是同一个参数，不能在 kernel 中转置覆盖、重新初始化或更新它。
内部打包缓存若依赖参数值，必须处理训练更新后的失效，不能复用过期权重。

## 7. 训练反向

设上游梯度为 $dY\in\mathbb{R}^{M\times N}$：

$$
dX=dY W,\qquad dW=dY^{\top}X.
$$

`dx` 恢复 x 的完整 shape；`dweight` 为 $[N,K]$，对本次输入的所有 $M$ 行求和。
不要再除 batch size 或 sequence length；loss 的平均与梯度累积权重已由上游处理。
AMP 下对原 FP32 x/weight 返回的梯度需与其 dtype 对应；中间 BF16 激活的梯度对应 BF16。
计算时要使用与前向一致的有效精度/转换语义，并对参考梯度做数值检查。

forward、dx、dweight 是三类不同 shape 的 GEMM，通常需要分别选 kernel。
仅接入 forward 的 CUTLASS GEMM，可以先用于 inference，不能依赖 autograd 自动推导外部 CUDA 调用。

## 8. 开发与验收建议

先做 BF16 inference 的五类投影，再完成 FP32 对照、AMP wrapper、dx/dweight 和 compile 注册。
同一个入口内部可根据 $M,N,K$ 做 dispatch，但不要把训练大 $M$ 的速度代表所有调用。

```bash
python -m tiny_transformer.check_ops --operator linear --backend student \
  --device cuda --precision fp32 --backward --output runs/linear-fp32.json
```

该命令只检查小尺寸、同 dtype 输入。BF16 支持实现后，还必须检查 FP32/BF16 混合输入的 autocast、
较小 $M$、不整除 tile 的尾部、LM head 的共享权重梯度，以及训练与缓存生成。
接口目前不包含 fused epilogue 的额外输出；跨 Linear/SwiGLU/Residual 融合应另行定义明确的新契约。

## 统一性能测试入口

本算子与其余七个算子共用 [benchmarks 测量框架](../../tiny_transformer/benchmarks/README.md)：
先校验数值与可用梯度，再用 CUDA events、交替后端顺序、多轮中位数分别测前向/反向。

```bash
python -m tiny_transformer.benchmarks --operator linear \
  --output runs/linear-performance.json
```

默认按 QKV、O、Gate/Up、Down、LM head 五种投影展开，共 15 组 case。
默认 `--linear-timing kernel` 在预分配后用 CUDA Graph/Event 分别测 forward、dx、dweight，
推理 prefill/decode 只测 forward，共 25 项成对计时。split-K 前向还报告分区 GEMM 和归约，
默认 decode Down 因此增加两项 student 计时。所有 GEMM 报告微秒、FLOP/s、TFLOP/s；
FMA 计 2 FLOPs，完整 GEMM 按 `2MNK` 计算，归约按从零累加 S 个部分和的 `SMN` 次加法计算。
`--linear-timing operator` 保留原始完整调用模式，包含分配和 autograd 调度，仍为 20 项计时。
两种模式的时间不能直接作为同口径加速比；CUDA Graph 时间也包含设备调度间隙。
prefill 的 LM head 按 `last_only=True` 使用 M=B，其余 block 投影使用 M=B*T。
终端与 JSON 均标明投影和问题规模；训练同时记录前向、dX、dWeight 三组 GEMM 尺寸。
默认推理 batch 与训练相同，可用 `--inference-batch-size 1` 单独测试小 batch 推理。
可通过 `--workloads train`、`--linear-projections qkv down`、`--phases backward` 筛选。
完整参数和形状表见 [Linear 性能用例](../../tiny_transformer/benchmarks/README.md#linear-的完整形状矩阵)。

未实现的 student 算子/阶段会记录为 `skipped`，没有隐式 reference fallback；
可用 `--backend reference` 验证完整测量流程。`check_ops --backward` 的结果不能替代反向性能数据。
