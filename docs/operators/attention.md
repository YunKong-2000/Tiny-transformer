# attention：包含 scale、causal mask、softmax 和 V 汇总

[返回算子总契约](README.md) · [参考实现](../../tiny_transformer/operators/reference.py) · [调用位置](../../tiny_transformer/model.py)

## 1. 接口与职责

```python
from typing import Optional

attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
          past_len: int = 0, segment_ids: Optional[torch.Tensor] = None) -> torch.Tensor
```

每层调用一次。输入 Q/K 已应用 RoPE，V 未旋转；有 cache 时，K/V 已由框架写入并取出有效前缀。
输出随后由模型合并 heads，再调用 O Linear 和 residual。

**此入口不是单独的 QK GEMM，也不是单独的 softmax，而是整个 attention 核心。**
它负责 scale、mask、softmax 和 PV；不负责 QKV/O 投影、RoPE、KV 写入、残差或输出 token 选择。

## 2. 输入与输出

| 参数 | shape / 类型 | 语义与约束 |
|---|---|---|
| q | $[B,N_h,T_q,D_h]$ 浮点 tensor | 当前待计算位置的 Q |
| k | $[B,N_h,T_k,D_h]$ 浮点 tensor | 当前可访问的全部 K，已做 RoPE |
| v | $[B,N_h,T_k,D_h]$ 浮点 tensor | 与 K 一一对应的 V |
| past_len | Python int，记为 $p\ge0$ | **本次输入之前**已有的缓存长度，不是分配容量 |
| segment_ids | None 或 $[B,T_q]$ 的 int64 tensor | 隔离文档的 segment 编号；不是 bool attention mask |
| 返回值 O | $[B,N_h,T_q,D_h]$ | 与 v 相同的计算 dtype/device |

项目 q/k/v 使用相同的 FP32、BF16 或 FP16 dtype。batch/head/head_dim 一致；当前是普通 MHA，
不包含 GQA/MQA、dropout、额外 attention bias、ALiBi 或可选 scale 参数。
返回一个 Tensor，不返回概率矩阵、log-sum-exp 或 cache tuple；反向所需状态可在内部保存。

## 3. 三种真实调用

| 场景 | $T_q$ | $T_k$ | $p$ | segment_ids |
|---|---:|---:|---:|---|
| 训练 / 普通 prefill | 当前序列长度 | 与 $T_q$ 相同 | 0 | 连续模式为 None；隔离训练为 $[B,T_q]$ |
| 单 token decode | 1 | 历史长度加 1 | 历史长度 | None |
| 带 cache 的 chunk 输入 | 当前 chunk 长度 | 历史长度加 chunk 长度 | 历史长度 | None |

当前模型的有效缓存调用满足 $T_k=p+T_q$。
默认训练 q/k/v 的 shape 都是 $[8,12,512,64]$。
例如已有 512 个 token、batch 为 4 的 decode：q 为 $[4,12,1,64]$，
k/v 为 $[4,12,513,64]$，`past_len` 为 512。
框架的 cache length 要在所有层完成后才增加，所以不能在此函数内再次增量更新它。

## 4. 计算、mask 与功能边界

$$
S_{b,h,i,j}=\frac{Q_{b,h,i,:}K_{b,h,j,:}^{\top}}{\sqrt{D_h}}.
$$

普通 causal 条件为 $j\le p+i$。
训练的文档隔离模式额外要求 $\mathrm{segment}_{b,i}=\mathrm{segment}_{b,j}$：

$$
\mathrm{allowed}_{b,i,j}=
\begin{cases}
j\le p+i,&\text{无 segment IDs},\\
(j\le i)\land(\mathrm{segment}_{b,i}=\mathrm{segment}_{b,j}),&\text{隔离文档}.
\end{cases}
$$

非法位置的 score 设为 $-\infty$，随后：

$$
P=\operatorname{softmax}(S+M),\qquad O=PV,
$$

其中 $M$ 在允许处为 0、禁止处为 $-\infty$，softmax 沿最后一维 $T_k$。
不能先 softmax 再简单清零非法权重，也不能在 decode 上直接使用从左上角开始的非方形三角 mask。
例如 $p=4,T_q=2,T_k=6$，两个 query 分别允许访问前 5 和前 6 个 key。

隔离模式只支持 $p=0$ 且 $T_q=T_k$。reference 遇到其他组合会抛出 `ValueError`；
模型更进一步，直接拒绝同时传入 cache 与 segments/custom positions。
第一版可以明确不支持隔离模式，但此时必须限制为 continuous，不能忽略 segment_ids 继续计算。

## 5. 数值与 dtype

reference 的顺序是 QK matmul → 除以 $\sqrt{D_h}$ → 低精度 score 转 FP32 → mask/softmax →
probability 转回 v dtype → PV matmul。FP64 reference 保留双精度。
BF16 AMP 训练中，q/k/v 通常已是 BF16；这与 norm/residual 的 FP32 路径不同。

融合 attention 可以用 FP32 score/softmax 累加，并避免完整中间矩阵落地，
但要与 reference/SDPA 做误差对照。更高精度的中间计算也可能产生不同的舍入，不要求逐 bit 相同。
softmax 应使用减最大值或等价的稳定 online softmax；合法模型输入每行至少能访问自己。
任意外部“整行全部屏蔽”的输入不在本项目正常契约内。

## 6. Stride：训练与缓存路径不同

训练时，reference RoPE 返回的 Q/K 通常连续，**V 仍是 QKV 拆分后的非连续 view**。
当合并 QKV 连续时，V 的 stride 为 $(3TH,D_h,3H,1)$，storage offset 为 $2H$。

缓存存储为 $[B,N_h,C,D_h]$，传入的是长度 $T_k$ 的前缀，stride 仍为：

$$
(N_h C D_h,\ C D_h,\ D_h,\ 1).
$$

不能将 head stride 写死成 $T_kD_h$。例如小模型 $B=2,N_h=4,C=16,D_h=16$：
cache 前缀 shape 为 $[2,4,6,16]$，stride 为 $(1024,256,16,1)$；
如果错误使用逻辑长度 6 计算 head stride，就会跨到错误位置甚至读取未初始化容量。
capacity 不作为独立参数传入，你通过 tensor stride 读取真实地址，只遍历有效 $T_k$。

输入 q/k/v 只读；禁止读有效前缀之外的数据。输出可为连续张量，后续 transpose/reshape 由框架处理。

## 7. 训练反向

上游梯度 $dO$ 与 O 同 shape。核心结构为：

$$
\begin{aligned}
dV&=P^{\top}dO, & dP&=dO\,V^{\top},\\
dS_{i,j}&=P_{i,j}\left(dP_{i,j}-\sum_k dP_{i,k}P_{i,k}\right),\\
dQ&=\frac{dS\,K}{\sqrt{D_h}}, & dK&=\frac{dS^{\top}Q}{\sqrt{D_h}}.
\end{aligned}
$$

这些矩阵运算分别在每个 batch/head 内进行；非法位置的 dS 为零。
backward 返回 `dq, dk, dv, None, None`，各梯度的 shape/dtype 对应原输入。
本项目 cache 只用于 eval + no_grad/inference_mode，因此训练反向主要对应方形、无 cache 的 attention；
直接算子测试仍可单独验证非方形张量的梯度。
高效反向可以保存归一化统计量并重算概率，不需要强制保存完整 $T^2$ 矩阵。

## 8. 分阶段验收

第一阶段可只做 continuous、固定 $D_h=64$ 的 forward；随后补 decode、chunk mask、cache stride、
隔离文档和训练 backward。是否支持每条路径要在启用 student 前明确检查。

```bash
python -m tiny_transformer.check_ops --operator attention --backend student \
  --device cuda --precision bf16 --backward --output runs/attention-bf16.json
```

开发工具没有覆盖真实缓存 stride、isolated segments 和全部 chunk 情况，需要补充：
改变未来 token 不影响过去输出；改变前一文档不影响后一文档；prefill+decode 与完整前向一致；
cache capacity 大于有效长度；batch 大于 1；前后向各自的误差与性能。
SDPA 是独立的可用后端。`attention=student` 中的优化不能仅凭调用名宣称已使用 FlashAttention。

## 统一性能测试入口

本算子与其余七个算子共用 [benchmarks 测量框架](../../tiny_transformer/benchmarks/README.md)：
先校验数值与可用梯度，再用 CUDA events、交替后端顺序、多轮中位数分别测前向/反向。

```bash
python -m tiny_transformer.benchmarks --operator attention \
  --output runs/attention-performance.json
```

未实现的 student 算子/阶段会记录为 `skipped`，没有隐式 reference fallback；
可用 `--backend reference` 验证完整测量流程。`check_ops --backward` 的结果不能替代反向性能数据。
