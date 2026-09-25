# rms_norm：逐 token 归一化与可训练缩放

[返回算子总契约](README.md) · [参考实现](../../tiny_transformer/operators/reference.py) · [调用位置](../../tiny_transformer/model.py)

## 1. 接口与职责

```python
rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor
```

每个 block 在 attention 和 FFN 之前各调用一次；全部 blocks 之后还调用一次 final norm。
默认模型每次 forward 共 17 次。输出交给 QKV、Gate/Up 或 LM head 的 Linear。
本接口只做 RMSNorm，不包含 residual add、Linear 或 LayerNorm 的减均值步骤。

**当前 student 实现**：CUDA FP32 前向，支持 `[..., H]` 与 `[H]`，`H > 0`；
Python 入口显式复制非连续输入，原生入口只接受连续张量。前导维度可以为空，
eps 必须是有限非负 FP32 标量。尚未实现 backward、BF16/FP16 或 torch.compile 注册。
下文的低精度与反向说明是完整目标契约；当前接入步骤和验证命令见第 7、8 节。

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

## 7. 本次代码检查与 PyTorch 接入步骤

原来的 warp 分工与平方和归约符合 RMSNorm 公式。`H < 32` 或不是 32 的倍数时，
没有列可读的 lane 保留零值，仍参与 shuffle；同一 warp 的 row 条件一致，因此 full mask 合法，
不需要 block 级共享内存或 `__syncthreads()`。

修复的主要问题：

| 原问题 | 修复与原因 |
|---|---|
| `torch/extention.h`、`rms_nrom` 拼写 | 改为实际头文件与源码路径，否则不能编译/加载 |
| `std::min{...}` 写法和 kernel 调用缺 epsilon | 使用 `std::min<int64_t>(..., 65535)`，按签名传入 epsilon |
| 缺少 CUDA stream、launch check 等头文件 | 显式包含对应 c10 头文件，避免依赖间接 include |
| 共享绑定引用另一算子和未实现的 backward | RMSNorm 使用独立 bindings，只导出已有前向，避免未定义符号 |
| 使用当前设备直接 launch | 加入 `CUDAGuard(X.device())`，在该设备的 PyTorch current stream 上执行 |
| `int` 保存形状和 `row * H` | 改为 64 位索引，防止大张量地址计算溢出 |
| 先算 `x * weight` 再除以 RMS | 改为 `(x * rsqrtf(mean_square + eps)) * weight`；先归一化可避免缩放乘积提前溢出 |
| 没有 autograd 却允许训练调用 | Python 和原生入口均检查 grad mode 与两项输入的 `requires_grad`，明确报错 |

接入链路：

```text
Transformer / Operators({"rms_norm": "student"})
  → student.rms_norm(x, weight, eps)
  → load_rms_norm_extension()                   # 首次调用才编译
  → module.rms_norm_forward(x, weight, eps)    # pybind11
  → rms_norm_forward_cuda(...)                # 校验、分配、device/stream
  → rms_norm_kernel<<<blocks, 256, 0, stream>>>
```

1. [rms_norm.h](../../csrc/rms_norm/rms_norm.h) 声明 C++ 前向签名，
   [rms_norm.cu](../../csrc/rms_norm/rms_norm.cu) 定义 wrapper 和 kernel。
   wrapper 分配与 x 同 shape/dtype/device 的新输出，空行输入直接返回，不 launch 空 grid。
2. [rms_norm/bindings.cpp](../../csrc/rms_norm/bindings.cpp) 使用
   `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` 将 C++ 函数导出为 `rms_norm_forward`。
   一个扩展只编译一个 module 定义；这里不再使用 embedding 的 `csrc/bindings.cpp`。
3. [_extension.py](../../tiny_transformer/operators/_extension.py) 的
   `load_rms_norm_extension()` 将上述 `.cpp` 和 `.cu` 传给 `torch.utils.cpp_extension.load`，
   模块名为 `tiny_transformer_rms_norm_cuda`。`lru_cache` 缓存进程内模块，
   PyTorch 管理磁盘编译缓存；导入模型不会触发构建。embedding 的加载保持独立。
4. [student.py](../../tiny_transformer/operators/student.py) 检查 CUDA 与前向使用条件，
   用 `x.contiguous()` 和 `weight.contiguous()` 显式处理 view，然后调用扩展。
   连续输入不发生额外复制；非连续输入的复制成本包含在 `check_ops` 和模型基准中。
   原生入口仍会独立检查 device、dtype、shape、连续性、eps 和 autograd 条件。
5. `Operators` 已按名字选择 student/reference，无需修改模型；只设置
   `Operators({"rms_norm": "student"})` 或命令行 `--op rms_norm=student`。

最小 PyTorch 调用（首次执行需要 CUDA 版 PyTorch、nvcc、C++ 编译器和 Ninja）：

```python
import torch
from tiny_transformer.operators import Operators

ops = Operators({"rms_norm": "student"})
x = torch.randn(2, 17, 65, device="cuda", dtype=torch.float32)
weight = torch.nn.Parameter(torch.ones(65, device="cuda"))
with torch.inference_mode():
    y = ops.rms_norm(x, weight, 1e-6)
assert y.shape == x.shape
```

`model.eval()` 不关闭 autograd，含可训练参数的推理仍须使用 `no_grad()` 或 `inference_mode()`。
pybind 不会为手写 CUDA 自动生成反向；当前不创建一个返回假梯度的 autograd 节点。
后续实现第 5 节的 `dx/dweight` 后，再导出 backward，用 `torch.autograd.Function` 保存输入/eps
并返回 `(dx, dweight, None)`，最后增加独立梯度对照与模型训练测试。

## 8. 测试与运行命令

在 A100 上从仓库根目录执行（其他 GPU 按实际计算能力设置架构）：

```bash
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=2
python -m unittest discover -s tests -p 'test_student_rms_norm.py' -v
python -m tiny_transformer.check_ops --operator rms_norm --backend student \
  --device cuda --precision fp32 --output runs/rmsnorm-fp32.json
python -m tiny_transformer.benchmark --config configs/smoke.json \
  --device cuda --precision fp32 --op rms_norm=student \
  --prompt-length 16 --new-tokens 8 --output runs/rmsnorm-inference.json
```

当前不要加 `--backward` 或选择 BF16/FP16；这些路径会明确报错。
eps=0 可用于非零行；零行加零 eps 时与 reference 一样产生 NaN，不额外 clamp。

[test_student_rms_norm.py](../../tests/test_student_rms_norm.py) 包含：

- 主机测试：import/CPU 拒绝不加载编译器、两个扩展的构建隔离与缓存、CUDA/toolkit 缺失报错。
  编译器 mock 仅验证加载配置，不作为 GPU 正确性证据。
- CUDA 数值：warp 边界、奇数 H、默认训练形状与 decode 形状、任意前导维度、超过 grid 上限的行数、
  零输入、不同幅值/eps、非全一权重、先缩放会溢出的输入、空行、输入不变与输出不别名。
- 布局：末维 stride=2、转置、广播 view、last-only prefill、非零 storage offset；
  原生入口拒绝非连续输入，Python 入口复制后正确计算。
- 工程：非法参数、两项输入各自需要梯度时的拒绝、no_grad/inference_mode、autocast 保持 FP32、
  非默认 stream、双 GPU 的 device guard 和设备不匹配。
- 模型：FP32 完整前向、batch=2 的 last-only prefill、KV cache decode，与 reference 对照。

单算子使用 `atol=1e-5, rtol=1e-4`；模型使用 `atol=2e-5, rtol=1e-4`，测试期间关闭 TF32 后恢复。
有 CUDA 时测试会真实编译扩展，缺少 nvcc 会构建失败；无 CUDA 时跳过 GPU 用例。
本次本地环境为 macOS / PyTorch 2.8.0，无 CUDA 和 nvcc，GPU 编译、数值及性能仍须按上述命令验收。
