# rope：对单个 Q 或 K 张量应用旋转位置编码

[返回算子总契约](README.md) · [参考实现](../../tiny_transformer/operators/reference.py) · [调用位置](../../tiny_transformer/model.py)

## 1. 接口与职责

```python
rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor
```

模型从合并 QKV 输出拆出 Q、K、V 后，分别调用一次 `rope(q, cos, sin)` 和 `rope(k, cos, sin)`。
一次调用只处理一个张量，默认每层 2 次。V 不经过 RoPE。

位置、频率、cos/sin 已由 `Transformer.forward` 算好；此函数不用接收 position、theta 或 cache。
旋转后的 K 随后由框架写入 cache，你的算子不负责 cache 分配、写入或更新长度。

## 2. 输入与输出

| 项目 | shape | dtype | 说明 |
|---|---|---|---|
| x | $[B,N_h,T,D_h]$ | FP32/BF16/FP16；reference 还支持 FP64 | Q 或 K 激活；$D_h$ 必须为偶数 |
| cos | $[1,1,T,D_h/2]$ 或 $[B,1,T,D_h/2]$ | 模型中为 FP32 | 每个相邻通道对使用一个系数 |
| sin | 与 cos 相同 | 模型中为 FP32 | 与 cos 对应的正弦系数 |
| 返回值 | 与 x 相同 | 与 x 同 dtype/device | 新的旋转结果，不能覆盖输入 |

普通连续语料和缓存推理中，position 在 batch 间共享，因此 cos/sin 的 batch 维是 1。
isolated packing 为每个样本分别重置 position，cos/sin 的 batch 维为 $B$。
head 维始终广播，不能把一个 head 的结果广播代替每个 head 自己的输入计算。

## 3. 必须匹配的旋转定义

本项目采用 adjacent-pair，而不是 split-half：

$$
\begin{aligned}
y_{2j}&=x_{2j}c_j-x_{2j+1}s_j,\\
y_{2j+1}&=x_{2j}s_j+x_{2j+1}c_j.
\end{aligned}
$$

其中 $j\in\{0,\ldots,D_h/2-1\}$。每个 batch/head/position 独立执行上述操作。
reference 先将 cos/sin 转为 x dtype，再计算旋转。融合 kernel 若使用 FP32 中间计算，
需对低精度系数舍入与最终误差作对照；不能无说明地改变系数精度或配对方式。

功能边界：不计算 sin/cos，不加 bias，不做 attention scaling，不旋转 V，不更改 position。
decode 的 cos/sin 已对应正确的绝对位置，不能根据输入长度为 1 而擅自使用位置 0。

## 4. 模型实际传入的布局

默认训练 x 为 $[8,12,512,64]$，cos/sin 为 $[1,1,512,32]$；isolated 时后者为 $[8,1,512,32]$。
单 token decode 的 x 为 $[B,12,1,64]$，cos/sin 为 $[1,1,1,32]$。

Q/K 源自合并 QKV 的 reshape、unbind 和 transpose，所以**输入通常不连续**。
当合并 QKV 是连续输出时，传入 x 的 stride 为：

$$
\operatorname{stride}(x)=(3TH,\ D_h,\ 3H,\ 1).
$$

Q 和 K 的 shape/stride 相同，但相对于合并 QKV storage，K 的 offset 比 Q 多 $H$。
例如小模型 $B=2,T=5,H=64,N_h=4$：

| 张量 | shape | stride | storage offset |
|---|---|---|---:|
| Q | $[2,4,5,16]$ | $(960,16,192,1)$ | 0 |
| K | $[2,4,5,16]$ | $(960,16,192,1)$ | 64 |

不要按标准连续 $[B,N_h,T,D_h]$ 地址公式读取输入，也不要在 `data_ptr()` 上重复加 offset。
第一版可以输出新的连续张量；reference 当前输出也是连续的，但下游契约看逻辑结果而非特定分配器地址。

## 5. 训练反向

本项目 cos/sin 不需要梯度，x 需要梯度。设上游梯度为 $g$：

$$
\begin{aligned}
dx_{2j}&=g_{2j}c_j+g_{2j+1}s_j,\\
dx_{2j+1}&=-g_{2j}s_j+g_{2j+1}c_j.
\end{aligned}
$$

即对上游梯度施加旋转矩阵的转置，返回 `dx, None, None`。
dx 的 shape/dtype 对应 x；不需要复制 x 的原始 storage offset，但必须按逻辑坐标返回正确梯度。

若单独将 cos/sin 设置为 `requires_grad=True`，PyTorch reference 本身可求其梯度；
这属于比本项目更广的功能。若 student 只实现 x 的梯度，应声明该范围并拒绝可训练 cos/sin，
不能在声称完整支持时静默丢弃它们的梯度。现有 `check_ops --backward` 明确把这两个参数设为常量。

## 6. 开发与验收建议

优先学会 CuTe 的 Q/K strided view 和相邻通道配对，再做向量化 load/store。
可以从固定 $D_h=64$ 的推理开始，但 smoke 使用 $D_h=16$；同时覆盖 batch 广播与逐样本 cos/sin。
RoPE 与 K-cache write 融合属于新接口设计，不能在当前纯函数中偷偷改 cache。

```bash
python -m tiny_transformer.check_ops --operator rope --backend student \
  --device cuda --precision bf16 --backward --output runs/rope-bf16.json
```

该工具的 transpose 输入与真实 QKV view 的 stride 并不完全相同，必须增加上表布局的用例。
检查位置 0、较大位置、isolated position reset、Q/K 不同 offset、batch 大于 1，以及反向。
额外检查每对分量的平方和近似保持不变，并对比整段前向与缓存解码的 logits。

## 统一性能测试入口

本算子与其余七个算子共用 [benchmarks 测量框架](../../tiny_transformer/benchmarks/README.md)：
先校验数值与可用梯度，再用 CUDA events、交替后端顺序、多轮中位数分别测前向/反向。

```bash
python -m tiny_transformer.benchmarks --operator rope \
  --output runs/rope-performance.json
```

未实现的 student 算子/阶段会记录为 `skipped`，没有隐式 reference fallback；
可用 `--backend reference` 验证完整测量流程。`check_ops --backward` 的结果不能替代反向性能数据。
