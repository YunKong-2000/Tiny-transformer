# Student 算子开发契约与导航

这组文档以当前仓库的实际调用为准，回答“模型会传入什么、算子必须返回什么”。
算法推导见 [Transformer 手册](../transformer.md)，CUDA/CuTe 工程规则见 [开发手册](../development.md)。
核对日期：2026-09-24。

## 1. 是否必须完全匹配 reference？

**对于声明支持且实际启用的路径，必须与 reference 的数学语义、逻辑形状、dtype、设备和梯度语义一致。**
不要求照抄实现、使用相同 kernel 数，也不要求浮点结果逐 bit 相同。
融合或不同归约顺序引起的误差需要数值对照和训练回归验证，不能靠任意放宽 tolerance 掩盖问题。

范围分三层：

| 层级 | 必须完成什么 | 可以明确暂不支持什么 |
|---|---|---|
| 单算子实验版 | 指定 shape/dtype/stride 的正确前向 | backward、其他 shape、其他设备；遇到范围外输入须明确报错 |
| 项目推理版 | 所选配置下 prefill、decode 和全部相关调用点 | 训练；例如只做 BF16 CUDA inference 时要写明限制 |
| 项目训练版 | 所选训练精度下的 AMP、输入梯度、参数梯度、验证前向 | PyTorch 通用 API 中本项目没有使用的其他选项 |

这不是要求第一天复刻完整的 `torch.nn.functional`。例如 embedding 没有 `padding_idx` 参数，
Linear 没有 bias，Attention 没有 dropout 或 GQA，CE 没有 label smoothing。
但只支持某个固定维度的 kernel，不应被标记为已经覆盖整个模型。

`--op 名称=student` 会替换该名称的**所有调用**，并不会按层自动选择或对不支持的 shape 自动 fallback。
例如只完成 QKV 的 GEMM 时，不能直接声称 `linear=student` 已能运行完整模型。
你可以在 student wrapper 内对已声明支持的不同 shape 做 dispatch；未覆盖输入应先检查再报错。

“支持 stride”指正确读取实际传入的视图。输出通常可以选择新的连续张量，
不要求复刻 reference 每个输出的物理 stride；必须满足后续消费者的逻辑接口并统计必要复制开销。
这 8 个接口都是函数式接口：不要原位改写输入/权重，不要将输出偷偷 alias 到需要保留的输入。

## 2. 八个算子的文档

| 算子 | 负责的工作 | 训练时的梯度 | 每次模型前向的调用次数 |
|---|---|---|---|
| [embedding](embedding.md) | token ID 查表 | weight；ids 无梯度 | 1 |
| [linear](linear.md) | QKV/O/gate-up/down/LM head 投影 | x、weight | $4L+1$ |
| [rms_norm](rms_norm.md) | 最后一维归一化与缩放 | x、weight | $2L+1$ |
| [rope](rope.md) | 旋转 Q 或 K | x；本项目 cos/sin 为常量 | $2L$ |
| [attention](attention.md) | 因果注意力、加权汇总 V | q、k、v | $L$ |
| [swiglu](swiglu.md) | gate 激活与 up 相乘 | gate、up | $L$ |
| [residual](residual.md) | 残差相加 | x、update | $2L$ |
| [cross_entropy](cross_entropy.md) | logits 与目标 ID 的平均损失 | logits；targets 无梯度 | 训练/验证每个 microbatch 1 次，生成时 0 次 |

这里 $L=8$。梯度累积不改变算子单次输入的 batch size：默认一个 microbatch 是 $B=8$，
一个 optimizer step 处理 4 个 microbatches，不是把每个算子的 batch size 变成 32。

## 3. 共用符号和真实尺寸

| 符号 | 含义 | 默认 60M 配置 | smoke 配置 |
|---|---|---:|---:|
| $B$ | 当前 microbatch/request 的 batch size | 训练 8；推理可变 | 训练 2 |
| $T$ | 当前输入长度 | 训练 512；prefill 为 prompt 长度；decode 通常为 1 | 训练 32 |
| $H$ | hidden dimension | 768 | 64 |
| $N_h$ | head 数 | 12 | 4 |
| $D_h=H/N_h$ | head dimension | 64 | 16 |
| $I$ | SwiGLU 中间宽度 | 2048 | 128 |
| $V$ | 词表大小 | 8192 | 259 |
| $C$ | cache 分配容量 | 请求指定，不超过 4096 | 不超过 256 |
| $p$ | 此次调用之前已经缓存的 token 数 | prefill 为 0，decode 递增 | 同左 |

除 Python 标量外，所有输入张量必须位于同一个设备；输出也在该设备。
本文 stride 均以**元素**为单位，地址偏移还需乘元素字节数。
Tensor 的 `data_ptr()` 已指向该视图的起始元素；若使用它，不要再重复加 `storage_offset()`。

## 4. BF16 训练与 BF16 推理的输入不同

`train.py` 保留 FP32 参数，在 `autocast` 区间内运行前向。
`generate.py` / `benchmark.py` 则直接将模型权重转换为所选推理精度。

以下是其他算子仍用 reference 时的预期数据流：

| 算子入口 | BF16 AMP 训练：真实输入 dtype | 输出 dtype | BF16 推理 |
|---|---|---|---|
| embedding | ids int64，weight FP32 | FP32 | weight/output BF16 |
| rms_norm | x FP32，weight FP32 | FP32 | x/weight/output BF16，内部统计 FP32 |
| linear：QKV、gate-up、LM head | x FP32，weight FP32；处于 autocast 内 | BF16 | x/weight/output BF16 |
| linear：O、down | x BF16，weight FP32；处于 autocast 内 | BF16 | x/weight/output BF16 |
| rope | x BF16，cos/sin FP32 | BF16 | 同左 |
| attention | q/k/v BF16，segment_ids 为 int64 或 None | BF16 | 同左，生成时无 segments |
| swiglu | gate/up BF16 | BF16 | 同左 |
| residual | x FP32，update BF16 | FP32 | x/update/output BF16 |
| cross_entropy | logits BF16，targets int64 | FP32 标量 | 文本生成不调用 |

FP32 训练中，浮点数据主路径为 FP32；FP16 AMP 的结构与上表相同，将低精度分支换成 FP16，
并由框架使用 GradScaler。BF16/FP16 推理的 sin/cos 仍由模型用 FP32 构造。

这些边界已用本地 PyTorch 2.8.0 的 CPU FP32、BF16 autocast、BF16 cached inference 实际调用核对，
并结合源码分析。CUDA AMP 行为还应在目标 NGC 镜像上验证，不能将 CPU 检查当作 A100 验收。
自定义 pybind kernel 不会自动获得 `F.linear` 的 autocast 规则，需在 wrapper/算子注册中处理。

特别注意：训练模式、梯度模式、autocast 是不同开关。验证是 eval + no_grad，但仍有 autocast 和 FP32 参数；
参数的 `requires_grad=True` 也不能单独证明当前需要执行 backward。

## 5. 前向接口之外，训练还需要什么

`student.py` 只列出 8 个前向入口，不代表训练框架能自动理解你写的 CUDA kernel。
若前向仅分配输出并调用扩展，需通过 `torch.autograd.Function` 或自定义算子注册提供 backward。

| forward 参数顺序 | backward 对应返回槽位 |
|---|---|
| `embedding(ids, weight)` | `None, dweight` |
| `linear(x, weight)` | `dx, dweight` |
| `rms_norm(x, weight, eps)` | `dx, dweight, None` |
| `rope(x, cos, sin)` | 本项目常量 cos/sin 路径为 `dx, None, None` |
| `attention(q, k, v, past_len, segment_ids)` | `dq, dk, dv, None, None` |
| `swiglu(gate, up)` | `dgate, dup` |
| `residual(x, update)` | `dx, dupdate` |
| `cross_entropy(logits, targets)` | `dlogits, None` |

梯度的 shape/dtype 应与对应浮点输入兼容，FP32 master 参数最终获得 FP32 梯度。
返回梯度给 autograd，由它累加到参数；kernel 不负责 optimizer step，也不要绕过 autograd 直接覆盖 `.grad`。
上游梯度可能含 microbatch 权重或 loss scaling，不能擅自假设其为 1。
激活梯度、参数梯度以及跨 microbatch 的累积是不同层次，不要重复归一化。

暂时只实现 forward 时，将该版本标为 inference-only，并先用于 `generate` / `benchmark`。
若需要训练，可先使用自定义 backward 中的 PyTorch 参考运算验证链路，再独立优化 backward；
文档与性能报告要明确这时仍使用参考反向。

## 6. 如何查看自己收到的数据

下面的诊断代码只包装 reference，打印一个小模型的第一个调用，不实现学生 kernel：

```python
import torch
from tiny_transformer.config import ModelConfig
from tiny_transformer.model import Transformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(ModelConfig(vocab_size=259, dim=64, n_layers=1,
                               n_heads=4, hidden_dim=128, max_seq_len=32)).to(device)
operator = "linear"  # 改为 embedding / rms_norm / rope / attention / swiglu / residual / cross_entropy
original = getattr(model.ops, operator)
seen = False

def describe(value):
    if not torch.is_tensor(value):
        return value
    return dict(shape=tuple(value.shape), dtype=str(value.dtype),
                stride=value.stride(), offset=value.storage_offset(),
                requires_grad=value.requires_grad)

def traced(*args, **kwargs):
    global seen
    result = original(*args, **kwargs)
    if not seen:
        print(operator, [describe(arg) for arg in args], "->", describe(result))
        seen = True
    return result

setattr(model.ops, operator, traced)
ids = torch.randint(0, 259, (2, 5), device=device)
targets = torch.randint(0, 259, (2, 5), device=device)
with torch.autocast(device, dtype=torch.bfloat16):
    loss = model.loss(ids, targets)
loss.backward()
```

诊断打印只用于理解，不能留在正式 benchmark 或 compile 路径中。
要检查 Linear 的所有角色，移除 `seen` 限制；要检查缓存，改用 eval + inference_mode，
先传入 prompt，再传入单 token，并观察 attention 的 stride 和 `past_len`。

## 7. 验收工具能证明什么

```bash
python -m tiny_transformer.check_ops --operator rms_norm --backend student \
  --device cuda --precision fp32 --backward --output runs/rmsnorm-contract.json
```

各算子文档列出专属补充用例。`check_ops.py` 默认生成小尺寸、**同 dtype** 的浮点参数，
因此它没有覆盖训练里的 FP32/BF16 混合输入，也未覆盖所有模型 shape、isolated mask 和 cache stride。
其中所谓 cross_entropy 的 decode 用例仅表示短序列算子测试；正常 decode 不调用 loss。
反向 helper 的 clone 也可能使原本跨步的输入变为连续，不能据此宣布非连续反向已经全面验收。

通过该工具后，再检查实际模型的 forward/backward、短程训练、验证 loss 和缓存 logits 对齐。
只测 reference 的单元测试通过，不能证明 student 已经正确；应显式选用 `student`。
FP64 gradcheck 是参考推导的工具；第一版 CUDA kernel 可以明确不支持 FP64，并用参考解析梯度作对照。

实现要求来源：[reference.py](../../tiny_transformer/operators/reference.py)、
[model.py](../../tiny_transformer/model.py)、[train.py](../../tiny_transformer/train.py)、
[data.py](../../tiny_transformer/data.py)、[check_ops.py](../../tiny_transformer/check_ops.py)。
