# rms_norm：逐 token 归一化与可训练缩放

[返回算子总契约](README.md) · [参考实现](../../tiny_transformer/operators/reference.py) · [调用位置](../../tiny_transformer/model.py)

## 1. 接口与职责

```python
rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor
```

每个 block 在 attention 和 FFN 之前各调用一次；全部 blocks 之后还调用一次 final norm。
默认模型每次 forward 共 17 次。输出交给 QKV、Gate/Up 或 LM head 的 Linear。
本接口只做 RMSNorm，不包含 residual add、Linear 或 LayerNorm 的减均值步骤。

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
  --device cuda --precision fp32 --backward --output runs/rmsnorm-fp32.json
```

补充用例：零输入、weight 非全 1、较大/较小幅值、不同 eps、奇数宽度、跨步输入、
batch 大于 1 的 last-only prefill，以及 dweight 跨多行累加。
仅完成 BF16 inference kernel，不代表默认 AMP 训练的 FP32 norm 路径已经支持。
