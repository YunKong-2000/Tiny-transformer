# swiglu：SiLU gate 与 up 分支逐元素相乘

[返回算子总契约](README.md) · [参考实现](../../tiny_transformer/operators/reference.py) · [调用位置](../../tiny_transformer/model.py)

## 1. 接口与职责

```python
swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor
```

每个 block 的 FFN 中调用一次。上游先计算合并的 Gate/Up Linear，
再沿最后一维 `.chunk(2, dim=-1)`：前半段是 gate，后半段是 up。
此函数只执行 SiLU 与乘法；两次输入投影已经由 Linear 完成，输出还要经过 Down Linear。

## 2. 输入与输出

| 项目 | 含义 | shape | dtype / 梯度 |
|---|---|---|---|
| gate | 输入投影的第一半，不是激活后的结果 | $[B,T,I]$ | 浮点；训练需要 dgate |
| up | 输入投影的第二半 | $[B,T,I]$ | 与 gate 同 dtype/device；训练需要 dup |
| 返回值 z | 非线性门控后的中间激活 | $[B,T,I]$ | 与输入同 dtype/device |

默认训练为 $[8,512,2048]$；单 token decode 为 $[B,1,2048]$。
BF16 AMP 训练中，gate/up 都来自 Linear 的 BF16 输出；FP16 AMP 时是 FP16。
FP32 训练或同 dtype 的 inference 则按相应 dtype 执行。
这里的宽度是 FFN 中间维 $I$，不是 hidden dimension $H$。

## 3. 数学语义和边界

$$
z_i=\operatorname{SiLU}(g_i)u_i
=g_i\sigma(g_i)u_i,
\qquad \sigma(g_i)=\frac{1}{1+e^{-g_i}}.
$$

- 激活的是 gate，不是 up；不能交换两者。
- 不使用 ReLU/GELU，不做额外 sigmoid 门控或 bias。
- 不做 reduction、归一化、residual、Down Linear。
- 不修改 gate/up；它们在训练反向时仍有用途。
- 当前模型两输入 shape 相同。PyTorch 的逐元素表达式可能支持广播等更广范围，首版不必实现，但应明确限制。

reference 先计算 `F.silu(gate)`，再与 up 相乘。低精度 SiLU 的中间输出有一次舍入。
融合 kernel 可以采用 FP32 中间值再写回，产生的小误差需验证；不能只因为数学表达等价就忽略数值差异。
应采用数值稳定的 sigmoid 计算，尤其检查较大负数，避免出现不必要的 NaN。

## 4. 两个输入通常是非连续视图

合并输出 shape 为 $[B,T,2I]$，chunk 后 gate/up 都是 $[B,T,I]$，但行 stride 仍然按 $2I$ 计算：

$$
\operatorname{stride}(\mathrm{gate})=
\operatorname{stride}(\mathrm{up})=(2TI,\ 2I,\ 1).
$$

相对于合并 tensor 的 storage，gate 的 offset 为 0，up 的 offset 为 $I$。
例如 $B=2,T=5,I=128$ 时：shape 为 $[2,5,128]$，stride 为 $(1280,256,1)$，up offset 为 128。
默认训练的行宽 2048 并不意味着下一行从当前地址加 2048 开始；实际要加 4096。

因此，把 `gate.data_ptr()` 后的全部 $BTI$ 个元素看成连续数组是错误的。
应按行 stride 与列偏移读取；使用 tensor 的 `data_ptr()` 时不重复加 up 的 storage offset。
输出可以是新的连续张量，下游 Down Linear 处理该输出。
`check_ops.py` 当前直接生成两个独立连续输入，单独通过它不能证明真实 chunk view 正确。

## 5. 训练反向

令上游梯度为 $a_i=\partial\mathcal{L}/\partial z_i$，则：

$$
\begin{aligned}
\operatorname{SiLU}'(g_i)&=\sigma(g_i)+g_i\sigma(g_i)(1-\sigma(g_i)),\\
dg_i&=a_i u_i\operatorname{SiLU}'(g_i),\\
du_i&=a_i\operatorname{SiLU}(g_i).
\end{aligned}
$$

返回 `dgate, dup`，shape 分别与输入相同，dtype 对应各自输入。
不需要在此处计算 Gate/Up Linear 的权重梯度；autograd 会把两个切片的梯度拼回合并输出，
由上游 Linear backward 继续传播。
可保存原 gate/up，反向重算 sigmoid；也可保存中间量，需权衡显存和计算。

一个有用边界是 $g_i=0$：此时 $z_i=0$，但 $dg_i=a_i u_i/2$ 通常并不为零。
不要用“输出为零”推断整个梯度为零。

## 6. 开发与验收建议

这是学习 CuTe view 和 pointwise 融合的合适入口。先做真实 chunk stride 的 forward，再加入 backward。
沿 $I$ 向量化读取，处理不整除向量宽度的尾部；减少中间张量，但保留清晰的输入输出契约。

```bash
python -m tiny_transformer.check_ops --operator swiglu --backend student \
  --device cuda --precision bf16 --backward --output runs/swiglu-bf16.json
```

补充真实 `.chunk(2, -1)` 输入、正负大幅值、零 gate/零 up、$I=65$ 尾部和 $I=2048$ 主配置。
分别检查 dgate/dup，并通过完整 FFN 梯度验证 chunk 的反向链路。
与 Gate/Up GEMM epilogue 融合时，上游输出的组织方式会变化，应明确增加融合入口，而不是改变此函数的参数含义。
