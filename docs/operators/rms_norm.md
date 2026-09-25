# rms_norm：逐 token 归一化与可训练缩放

[返回算子总契约](README.md) · [参考实现](../../tiny_transformer/operators/reference.py) · [调用位置](../../tiny_transformer/model.py)

## 1. 接口与职责

```python
rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor
```

每个 block 在 attention 和 FFN 之前各调用一次；全部 blocks 之后还调用一次 final norm。
默认模型每次 forward 共 17 次。输出交给 QKV、Gate/Up 或 LM head 的 Linear。
本接口只做 RMSNorm，不包含 residual add、Linear 或 LayerNorm 的减均值步骤。

**当前 student 实现**：CUDA FP32 前向与一阶反向，支持 `[..., H]` 与 `[H]`；
前向要求 `H > 0`，共享内存反向要求 `0 < H <= 1024`。需要梯度且超出该范围时，在前向入口报错；
Python 入口显式复制非连续输入，原生入口只接受连续张量。前导维度可以为空，
eps 必须是有限非负 FP32 标量。尚未支持 BF16/FP16 输入、二阶梯度或 torch.compile 注册。
FP32 模型参数在 AMP 训练中保持 FP32，RMSNorm 也保持 FP32；这不同于把模型转成 BF16 推理。
当前接入步骤和验证命令见第 7、8 节。

## 2. 输入与输出

| 项目 | 含义 | shape | dtype / 梯度 |
|---|---|---|---|
| x | residual stream 上的激活 | $[B,T,H]$，一般为 $[\ldots,H]$ | 浮点 tensor；训练需要 dx |
| weight | 公式中的 $\gamma$，不是矩阵投影 | $[H]$ | 可训练参数；需要 dweight |
| eps | 防止分母为零的常数 | Python float，默认 $10^{-6}$ | 不是 tensor，无梯度 |
| 返回值 y | 归一化后按通道缩放的激活 | 与 x 相同 | **与 x 同 dtype/device** |

weight 不在 batch/sequence 维度变化，所有行共享同一份 $\gamma$。
项目的 x/weight 组合：FP32 训练为 FP32/FP32；BF16 AMP 训练此处也为 FP32/FP32；
BF16 inference 为 BF16/BF16。参考会将 weight 转为计算累加 dtype，因此输出 dtype 由 x 决定。

## 3. 精确的前向语义

将 x 的前导维度视为多行，对每行分别计算：

$$
r=\left(\frac{1}{H}\sum_{j=0}^{H-1}x_j^2+\epsilon\right)^{-1/2},
\qquad y_i=x_i r\gamma_i.
$$

参考的顺序是：将低精度 x 转 FP32 → 平方均值 → 加 eps → rsqrt → 乘 x 和 weight → 转回 x dtype。
FP32 保持 FP32；FP64 reference 路径保留 FP64，方便 gradcheck。
你可以融合这些步骤，但应保持相同数学定义与精度意图。

边界：

- 只沿最后一维归约，不能混合不同 token 或 batch。
- 分母是平方和的**均值**，不能遗漏除以 $H$。
- eps 加在平方根内，不是对 rsqrt 结果加 eps。
- 不减均值，不添加 beta，不做 affine bias。
- 不修改 x；它仍是后续残差相加所需的原始 residual。
- 输出只有 y，不返回 r；训练要保存 r 时由 autograd context 管理。

## 4. 真实形状与 stride 陷阱

| 场景 | x | weight | 输出 |
|---|---|---|---|
| 默认训练的 block/final norm | $[8,512,768]$ | $[768]$ | $[8,512,768]$ |
| prefill 的 block norm | $[B,T_{\mathrm{prompt}},768]$ | $[768]$ | 同 x |
| prefill 的 final norm | $[B,1,768]$ | $[768]$ | 同 x |
| 单 token decode | $[B,1,768]$ | $[768]$ | 同 x |

prefill 的 final norm 只处理最后一个位置，因为模型在它之前执行 `x[:, -1:, :]`。
当 $B>1$ 且 prompt 长度大于 1 时，该 view 的 batch stride 通常仍为 $T_{\mathrm{prompt}}H$，
不能按连续的 $BH$ 个元素扁平读取。

本地小模型示例：x shape 为 $[2,1,64]$，stride 为 $(320,64,1)$，storage offset 为 256。
其两行之间相隔 320 个元素，而不是 64。kernel 应读取实际行 stride。

`check_ops.py` 还构造 `x[..., ::2]`，使归约轴本身的 stride 为 2，宽度为 65。
这超出“最后一维连续”的常见 fast path：可先提供明确限制，但必须知道该检查为何失败，
并在声称支持此类 view 前补齐读取或计入显式拷贝。

## 5. 训练反向

上游梯度 $g$ 与 y 同 shape。对单行定义 $u_i=g_i\gamma_i$，则：

$$
dx_i=r u_i-\frac{r^3x_i}{H}\sum_j u_jx_j.
$$

跨所有行的 weight 梯度为：

$$
d\gamma_i=\sum_{a\in\mathrm{rows}}g_{a,i}x_{a,i}r_a.
$$

backward 返回 `dx, dweight, None`。dx 是逐行计算，dweight 需要跨 batch/sequence 行归约。
dtype 分别对应原始 x、weight；不要把所有梯度都强制输出为 BF16。
还要保留上游梯度中的 loss scaling / microbatch 权重，不额外平均 dweight。

可保存 x、weight、FP32 r；也可以在 backward 重算 r。若重算，必须使用相同 eps 和归约定义。
归约次序不同引起的舍入允许在合理阈值内变化，但 forward 正确不能替代 dweight 的独立验证。

## 6. 功能范围和优化路线

第一版可先支持 CUDA、末维连续、FP32/BF16、特定 $H$ 的 forward，明确返回范围外错误。
模型训练默认 $H=768$，但 smoke 的 $H=64$，开发工具还用到 $H=65$；尾部处理是独立任务。
接训练时需要 FP32 输入路径，即使训练命令选择了 BF16。

先优化行内归约、向量化 load/store，再考虑 residual+norm 融合。
当前接口没有 residual update 参数，不能私自将相加塞入这个函数。
跨算子融合通常需要同时返回原 residual 更新结果与 normalized 结果，需另立契约。

```bash
python -m tiny_transformer.check_ops --operator rms_norm --backend student \
  --device cuda --precision fp32 --output runs/rmsnorm-fp32.json
```

补充用例：零输入、weight 非全 1、较大/较小幅值、不同 eps、奇数宽度、跨步输入、
batch 大于 1 的 last-only prefill，以及 dweight 跨多行累加。
仅完成 BF16 inference kernel，不代表默认 AMP 训练的 FP32 norm 路径已经支持。

## 7. 前向缓存与 PyTorch 调用链

前向保留一 warp 一行的 scalar/float4 两条路径。向量化要求 x、weight 地址 16 字节对齐且 H 能被 4 整除。
每行只有 lane 0 写入 `R[row]`；两条路径都必须保存这个值。
R 存的是 `rsqrt(mean(x²) + eps)`，形状 `[rows]`，dtype FP32。
原生前向现在返回 `(Y, R)`，包括空输入时的两个空张量；Python 公共接口仍只返回 Y。
当前推理也调用该原生入口，因此前向性能计时包含 R 的分配/写入。

```text
student.rms_norm(x, weight, eps)
  → x.contiguous(), weight.contiguous()        # 拷贝保留在 autograd 图中
  → _RMSNorm.apply(xc, wc, eps)                # 需要梯度时
      → extension.rms_norm_forward(xc, wc, eps)
      ← Y, R
      → ctx.save_for_backward(xc, wc, R)
      ← Y
loss.backward()
  → _RMSNorm.backward(grad_y)
      → ctx.saved_tensors                     # 取回本次前向的缓存
      → extension.rms_norm_backward(xc, grad_y.contiguous(), wc, R)
      ← dX, dGamma
      ← dX, dGamma, None                      # 对应 x、weight、eps
```

- [rms_norm.h](../../csrc/rms_norm/rms_norm.h) 声明两个返回 `std::tuple<Tensor, Tensor>` 的接口。
- [bindings.cpp](../../csrc/rms_norm/bindings.cpp) 导出 `rms_norm_forward(X, weight, epsilon)` 和
  `rms_norm_backward(X, gradient, weight, R)`。反向不再接收 epsilon，因为 R 已包含它。
- [_extension.py](../../tiny_transformer/operators/_extension.py) 延迟编译 bindings、前向 `.cu` 和反向 `.cu`；
  模块名仍为 `tiny_transformer_rms_norm_cuda`，导入模型不会编译，与 embedding 模块独立。
- [student.py](../../tiny_transformer/operators/student.py) 用 `_RMSNorm` 建立一阶 autograd 节点。
  `save_for_backward` 保存张量引用，由图管理生命周期；多个前向有各自的 R，无全局缓存。
  只有 x 或只有 weight 需要梯度时，也能正确回传；二阶梯度通过 `once_differentiable` 明确拒绝。

原生 pybind 本身不会建立 autograd 图。因此直接调用前向且输入需要梯度时，必须处于 `no_grad`，
或者改用 `student.rms_norm`。`Function.forward` 自动关闭 grad mode；`Function.apply` 负责建立节点。

最小训练调用：

```python
import torch
from tiny_transformer.operators import Operators

ops = Operators({"rms_norm": "student"})
x = torch.randn(2, 17, 65, device="cuda", requires_grad=True)
weight = torch.nn.Parameter(torch.ones(65, device="cuda"))
y = ops.rms_norm(x, weight, 1e-6)
y.square().mean().backward()
assert x.grad.shape == x.shape
assert weight.grad.shape == weight.shape
```

## 8. 共享内存反向与运行命令

反向每个 block 使用 256 线程处理一行。三个长度 1024 的共享数组保存 x、dy、u，
八个 warp 小计先写入共享内存，再由首个 warp 归约。每行结束后同步，才能安全复用下一行的共享内存。
固定共享内存为 `3 * 1024 * 4 + 8 * 4 = 12320` 字节，H=1024 合法。

wrapper 在主机端校验 H 上限、gradient 与 x 完整形状相同、R 是同设备连续 FP32 `[rows]`。
每次调用分配 dX 并将 dGamma 清零；每行使用 `atomicAdd(dGamma + col, ...)` 跨行累加。
两条 wrapper 都使用设备 guard、current stream 和 kernel launch 检查，空行输入不发射 kernel。
浮点原子加不保证确定性；严格 deterministic 模式下非空反向明确报错。

在 A100 上从仓库根目录执行（其他 GPU 按实际计算能力设置架构）：

```bash
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=2
python -m unittest discover -s tests -p 'test_student_rms_norm.py' -v
python -m tiny_transformer.check_ops --operator rms_norm --backend student \
  --device cuda --precision fp32 --backward --output runs/rmsnorm-fp32.json
python -m tiny_transformer.benchmarks --operator rms_norm \
  --layouts contiguous strided last-only --phases forward backward \
  --output runs/rmsnorm-performance.json
python -m tiny_transformer.benchmarks.model --config configs/smoke.json \
  --device cuda --precision fp32 --op rms_norm=student \
  --prompt-length 16 --new-tokens 8 --output runs/rmsnorm-inference.json
```

统一性能框架在 backward 阶段复用前向图和 R，包含梯度清零、分配和 autograd 调度，排除前向。
H 超过 1024 时仍可测前向，反向记录为 skipped；已声明支持的路径发生编译或数值错误时直接失败。
eps=0 可用于非零行；零行加零 eps 时与 reference 一样产生 NaN，不额外 clamp。

[test_student_rms_norm.py](../../tests/test_student_rms_norm.py) 用同一组输入检查 Y、缓存 R、dX 和 dgamma，
覆盖 scalar/vector 边界、地址偏移、H=1024/1025、空行、跨步输入、grid 循环、清零与 current stream。
主机只保留一个测试替身用例，检查每图的 R 和跨步梯度累积；不视为 GPU 数值验收。

模型训练及缓存推理统一放在 [test_student_integration.py](../../tests/test_student_integration.py)，
分别启用 embedding、RMSNorm 和两者，对照 FP32/AMP 参数梯度及 last-only prefill/decode。
运行两个算子和集成回归：

```bash
python -m unittest discover -s tests -p 'test_student_*.py' -v
```

前向使用 `atol=1e-5, rtol=1e-4`，单算子反向与 FP32 模型梯度使用 `atol=3e-5, rtol=3e-4`；
模型推理使用 `atol=2e-5, rtol=1e-4`，AMP 模型梯度使用 `atol=3e-3, rtol=3e-2`。
测试职责及删除的重复覆盖见 [测试说明](../../tests/README.md)。
本机无 CUDA/nvcc 时跳过 GPU 用例；GPU 编译、数值和性能仍须在目标环境验收。
