# cross_entropy：带 ignore_index 的平均 next-token 损失

[返回算子总契约](README.md) · [参考实现](../../tiny_transformer/operators/reference.py) · [训练调用](../../tiny_transformer/train.py) · [标签构造](../../tiny_transformer/data.py)

## 1. 接口与职责

```python
cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor
```

输入是 LM head 输出的**未归一化分数**与真实下一个 token ID。
每个训练/验证 microbatch 调用一次，不在 `generate` 或推理 benchmark 中调用。
该接口包括稳定 log-sum-exp、目标类别分数提取、忽略标签和平均归约。
不包括 LM head GEMM、采样、tokenizer 或 optimizer。

## 2. 输入与输出

| 项目 | shape | dtype | 语义 |
|---|---|---|---|
| logits | $[B,T,V]$ | FP32、BF16、FP16；reference 也接受 FP64 | 不是概率，不应预先 softmax |
| targets | $[B,T]$ | 项目使用 `torch.int64` | 合法 token ID 或 `-100` |
| 返回值 loss | 标量 tensor，`shape == ()` | FP16/BF16/FP32 logits 返回 FP32；FP64 reference 返回 FP64 | 对有效 target 的平均损失 |

两输入位于同一设备，loss 也必须留在该设备。
不能返回 Python float、逐 token loss 数组或长度为 1 的向量替代标量 tensor。
训练需要 loss 保持可微，`.item()` 只适合外部日志，不应出现在算子返回路径中。

## 3. 标签已经由 dataset 偏移过一次

dataset 从 token 流取长度 $T+1$ 的窗口，构造：

```text
完整窗口： BOS  A  B  C  EOS
ids：      BOS  A  B  C
targets：   A  B  C  EOS
```

因此此算子直接比较同一位置的 logits 与 targets，不能再次 shift、删除首尾位置或改写标签。
连续模式允许跨文档预测；isolated 模式由 dataset 将 EOS 后跨到下一文档的目标置为 `-100`。
CE 只看最终 targets，不接收 segment_ids，也不自行重新构建 causal mask。

## 4. 数学语义与数值精度

将前导维度展平为 $R=BT$。记有效行集合为
$\mathcal{A}=\{r:y_r\ne-100\}$，$N=\lvert\mathcal{A}\rvert$：

$$
\ell_r=-z_{r,y_r}+\log\sum_{j=0}^{V-1}e^{z_{r,j}},
\qquad
\mathcal{L}=\frac{1}{N}\sum_{r\in\mathcal{A}}\ell_r.
$$

稳定计算应使用每行最大值：

$$
m_r=\max_j z_{r,j},\qquad
\ell_r=-z_{r,y_r}+m_r+\log\sum_j e^{z_{r,j}-m_r}.
$$

reference 对低精度 logits 先转为 FP32，再计算交叉熵。
分母是有效 target 数，不是始终等于 $BT$；不能把 ignored 行计入平均。
例如只有一行有效，增加十行 ignored targets 不应改变有效行的平均 loss。

边界：

- 只有 `-100` 是 ignore index；ID 0、BOS、EOS 不自动忽略。
- 其他 target 必须满足 $0\le y_r<V$，不允许把其他负数当作 padding。
- 没有 class weights、label smoothing、soft labels 或 sum/none reduction 选项。
- 不需要返回 softmax 概率，也不能把概率再次当作 logits 传入。
- 全部 targets 为 `-100` 时，有效数为 0；当前 mean reference 返回 NaN，正常训练路径应避免此情况。
  首版可明确拒绝该输入，但不能默默返回 0 并声称与完整 reference 边界一致。

## 5. 默认形状、布局与显存

默认 logits 为 $[8,512,8192]$，targets 为 $[8,512]$，有效数最多为 4096。
BF16 AMP 训练中，logits 是 BF16，loss 是 FP32。
logits 共 $8\times512\times8192=33{,}554{,}432$ 个元素：BF16 本体约 64 MiB，
完整转成 FP32 还需约 128 MiB，因此避免大中间张量具有实际价值。

当前 LM head 和 dataset 通常提供连续张量；reference 使用 `.reshape`，必要时会处理非连续布局。
第一版可明确要求词表维连续，其他 stride 通过 wrapper 检查；不能因逻辑 shape 相同而直接假设所有行紧邻。
返回 loss 不需要连续性约定之外的复杂布局，但不能改写输入 logits 作为临时 workspace。

## 6. 训练反向与外部 loss 权重

设有效行上的概率为 $p_{r,j}$，算子收到的标量上游梯度为 $a$：

$$
\frac{\partial(a\mathcal{L})}{\partial z_{r,j}}
=\begin{cases}
\displaystyle\frac{a}{N}\left(p_{r,j}-\mathbf{1}_{j=y_r}\right),&r\in\mathcal{A},\\
0,&r\notin\mathcal{A}.
\end{cases}
$$

backward 返回 `dlogits, None`。dlogits 的 shape 与 logits 相同，返回 dtype 对应原 logits。
内部概率、归约可用 FP32，再按所需输入梯度 dtype 写回。

不要假定 $a=1$：训练脚本按每个 microbatch 的有效 target 数为 loss 加权，FP16 还会使用 GradScaler。
kernel 必须接收并应用实际上游梯度。算子内部只对**本次调用**的有效 target 数平均，
不再除 grad_accum 或全局 batch size。
优化时可保存每行 log-sum-exp，在 backward 重算概率，避免存储完整 FP32 softmax。

## 7. 开发与验收建议

先实现 FP32、连续 logits、有限有效 targets 的 mean CE；随后覆盖 ignore_index、低精度输入与 backward。
推理的 `--op cross_entropy=student` 不会触发它，必须运行训练、验证或直接算子检查。

```bash
python -m tiny_transformer.check_ops --operator cross_entropy --backend student \
  --device cuda --precision bf16 --backward --output runs/ce-bf16.json
```

补充用例：所有 logits 为零时 loss 为 $\log V$；极大正负 logits；一个有效 target 与多个 ignored；
忽略位置梯度为零；不同有效数的 microbatches；非单位上游梯度；FP16 GradScaler；真实 $V=8192$。
对全部 ignored 的边界单独检查并记录策略。
融合 LM head 与 CE 需要隐藏状态及输出权重，当前接口只接收 logits，无法无接口变化地完成该融合。

## 统一性能测试入口

本算子与其余七个算子共用 [benchmarks 测量框架](../../tiny_transformer/benchmarks/README.md)：
先校验数值与可用梯度，再用 CUDA events、交替后端顺序、多轮中位数分别测前向/反向。

```bash
python -m tiny_transformer.benchmarks --operator cross_entropy \
  --output runs/cross_entropy-performance.json
```

未实现的 student 算子/阶段会记录为 `skipped`，没有隐式 reference fallback；
可用 `--backend reference` 验证完整测量流程。`check_ops --backward` 的结果不能替代反向性能数据。
