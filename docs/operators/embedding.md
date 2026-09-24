# embedding：token ID 查表

[返回算子总契约](README.md) · [参考实现](../../tiny_transformer/operators/reference.py) · [调用位置](../../tiny_transformer/model.py)

## 1. 接口与职责

```python
embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor
```

每次模型前向开始时调用一次。`ids` 来自 packed batch、prompt 编码或上一步生成的 token；
`weight` 是 `model.embedding` 参数。输出送入第一个 block 的 RMSNorm。
本接口只负责查表；tokenizer、位置编码、mask、Linear 和 loss 均由其他模块处理。

## 2. 输入与输出

| 项目 | 实际数据 | shape | dtype / 梯度 |
|---|---|---|---|
| ids | 离散 token 编号，不是 one-hot 或浮点向量 | $[B,T]$ | 项目使用 `torch.int64`；无梯度 |
| weight | 可训练 embedding 表，一行对应一个 token | $[V,H]$ | 训练 FP32；推理为所选浮点 dtype；训练需要梯度 |
| 返回值 x | 每个 token 的向量表示 | $[B,T,H]$ | 与 weight 相同的 dtype/device |

合法 ID 满足 $0\le\mathrm{ids}_{b,t}<V$。负数、越界 ID 不是正常输入，wrapper 应检查支持约束并明确失败。
不要把 targets 中的 `-100` 当作 embedding 输入；该值只属于 loss mask。

BF16/FP16 AMP 训练不会自动把此处 FP32 embedding 参数和输出转为低精度。
如果 kernel 强制输出 BF16，就改变了后续 norm/residual 的计算路径。
PyTorch 参考也可接受部分其他整数 dtype/输入维度，但本项目的首要契约是二维 int64 IDs。

## 3. 数学语义和功能边界

$$
X_{b,t,h}=W_{\mathrm{ids}_{b,t},h}.
$$

- 不加 bias，不缩放输出，不加 position embedding。
- 没有 `padding_idx`：ID 0 如果出现，就正常查表并参与梯度。
- 没有按 token 出现频率缩放梯度，也没有 max-norm clipping。
- 不进行去重后改变输出顺序；同一个 ID 出现在多个位置时，都需要对应输出。
- 输入 weight 是只读参数，不得在 forward 内更新。

## 4. 真实形状和布局

| 场景 | ids | weight | 输出 |
|---|---|---|---|
| 默认训练 | $[8,512]$ | $[8192,768]$ | $[8,512,768]$ |
| batch 1、512-token prefill | $[1,512]$ | $[8192,768]$ | $[1,512,768]$ |
| batch 4、单 token decode | $[4,1]$ | $[8192,768]$ | $[4,1,768]$ |

当前 dataset 构造连续 IDs，weight 也是连续参数；这些可以作为第一版 fast path。
参考的逻辑查表不要求 ids 连续，缓存一致性测试或手动切片可能传入非连续 IDs；
只支持连续输入时应显式报错或在 wrapper 中显式复制并计入成本。
返回新的逻辑输出张量，不可把输出伪装成对 weight 的可写 view。

## 5. 训练反向

上游梯度 $G$ 的形状为 $[B,T,H]$：

$$
\frac{\partial\mathcal{L}}{\partial W_{v,h}}
=\sum_{b,t:\,\mathrm{ids}_{b,t}=v}G_{b,t,h}.
$$

反向返回 `None, dweight`，`dweight` 的形状为 $[V,H]$，dtype 与 weight 对应。
只考虑 embedding 这一条分支时，未出现的 token 行梯度为零；重复 ID 的梯度是**相加**，不是覆盖或求平均。
用 atomics 时要考虑竞争、累加精度和非确定性误差。

默认模型的 embedding 与 LM head 共享同一个 Parameter。整模型训练时，该权重还会收到
输出 Linear 的梯度，因此不能用“未作为输入出现的 token 行必须为零”检查共享权重的最终 `.grad`。
交由 autograd 累加两条分支，不要在 embedding backward 中清空或覆盖共享参数梯度。
本项目参考使用 dense gradient；第一版无需实现 sparse embedding optimizer 路径。

## 6. 建议的实现阶段与验收

1. FP32/BF16 inference：按 ID gather 一整行，保证向量 load/store 对齐与行尾处理。
2. 加入重复 ID 的 backward 累加，验证不同 batch/sequence 的重复情况。
3. 验证 FP32 master 参数的 AMP 训练，再考虑查表缓存与带宽优化。

```bash
python -m tiny_transformer.check_ops --operator embedding --backend student \
  --device cuda --precision fp32 --backward --output runs/embedding-fp32.json
```

补充用例：所有 ID 相同、ID 0 与最大合法 ID、重复 BOS/EOS、非连续 IDs、真实词表，
以及共享 embedding/LM head 权重的整模型梯度。BF16 推理和 BF16 AMP 训练要分别检查。
不需要为了这一个算子实现 tokenizer 或 vocab projection。
