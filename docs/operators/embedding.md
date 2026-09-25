# embedding：token ID 查表

[返回算子总契约](README.md) · [参考实现](../../tiny_transformer/operators/reference.py) · [调用位置](../../tiny_transformer/model.py)

## 1. 接口与职责

```python
embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor
```

student 另提供实验用关键字参数 `backward_impl="grouped"`，可设为 `"baseline"` 选择
一 warp 一 token 的反向实现；默认模型调用仍使用 grouped，前向计算相同。

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

student 已接入 CUDA FP32 的一阶反向与 eager autograd；下文给出输入输出契约和调用链。
CUDA 编译、数值和性能验收须在目标 GPU 上执行，不能以 CPU 接线测试代替。

### 5.1 反向算子的输入与输出

前向是从 weight 查出各个 token 的向量；反向则把各位置的上游梯度累加回对应的 weight 行。
记 `grad_output` 为 $G=\partial\mathcal{L}/\partial X$，它由下游计算图传入，
不是前向的 weight，也不是前向输出 X 本身。

底层 CUDA 接口显式接收词表大小（C++ 参数名为 `gradient`，对应下文的 grad_output）：

```cpp
torch::Tensor embedding_backward_cuda(
    torch::Tensor ids, torch::Tensor gradient, int64_t vocab_size,
    const std::string& implementation = "grouped");
```

| 方向 | 项目 | 含义 / 来源 | shape | dtype / device |
|---|---|---|---|---|
| 输入 | ids | 前向保存的 token ID，决定每个梯度累加到哪一行 | $[B,T]$ | `torch.int64`；与 grad_output 同设备 |
| 输入 | grad_output | 下游传入的 $\partial\mathcal{L}/\partial X$ | $[B,T,H]$ | 与前向输出 X 对应；本项目 FP32/AMP 训练均为 FP32 |
| 输入 | vocab_size | 前向保存的 `weight.shape[0]`，即 V | 整数标量 | C++ `int64_t`，不是 tensor |
| 输入 | implementation | 选择 `grouped` 或 `baseline`，默认 grouped | 字符串 | 不是 tensor，无梯度 |
| 输出 | dweight | 本次 embedding 调用贡献的 $\partial\mathcal{L}/\partial W$ | $[V,H]$ | dense tensor；与前向 weight 同 dtype/device |

反向不需要读取 weight 或 X 的数值。当前 autograd wrapper 使用 `ctx.save_for_backward(ids)`
保存索引，使用 `ctx.vocab_size = weight.shape[0]` 保存 V。autograd 校验上游梯度与前向输出的
shape/dtype/device；H 从 grad_output 的末维取得，FP32 输出梯度表在相同设备上分配。
**不能用 `ids.max() + 1` 推断 V**：一个 batch 通常只包含词表中的一部分 token，空 IDs 也没有最大值。
若底层只接收 `ids, grad_output`，还必须通过其他明确约定提供 V，例如传入预分配的 `[V,H]` 输出；
这两个输入本身不足以确定完整梯度表的行数。

底层 CUDA 算子只返回 `dweight`。当前 Python wrapper 内部调用
`_Embedding.apply(ids, weight, backward_impl)`，因此 backward 按这三个参数的顺序返回
**`(None, dweight, None)`**。ids 是离散整数索引，backward_impl 是实现选择字符串，
两者的梯度均为 `None`，不是全零 tensor。V 保存于 ctx，不是 apply 的参数，不增加返回项。

默认训练时，反向输入为 ids `[8,512]`、grad_output `[8,512,768]` 和 V=8192，
输出 dweight `[8192,768]`。kernel 内可将 ids 展平为 `[B*T]`、grad_output 展平为 `[B*T,H]`，
但 `[B*T,H]` 表示每个位置的上游梯度，不是 weight 的形状；输出始终保留完整 `[V,H]`。

### 5.2 梯度计算过程

对每个词表行 v 和通道 h：

$$
\frac{\partial\mathcal{L}}{\partial W_{v,h}}
=\sum_{b,t:\,\mathrm{ids}_{b,t}=v}G_{b,t,h}.
$$

1. 为本次调用分配独立的 `[V,H]` 梯度表，并初始化为零。
2. 遍历每个位置 `(b,t)`，读取 `v = ids[b,t]`，将 `grad_output[b,t,:]` 累加到 `dweight[v,:]`。
3. 返回该梯度表，由 autograd 将它传递并累加到 weight 对应的梯度。

只考虑 embedding 这一条分支时，未出现的 token 行梯度为零；重复 ID 的梯度是**相加**，不是覆盖或求平均。
例如 `ids = [[2,1,2]]`，三个位置的梯度为 `g0, g1, g2`，则
`dweight[2] = g0 + g2`、`dweight[1] = g1`，其余行全零。ID 0 也按相同规则参与累加。
空 IDs 对应空的 grad_output，仍返回完整 `[V,H]` 全零梯度表。

并行实现可使用 scatter-add / atomicAdd，或先按 ID 分组再归约；重复 ID 会产生写竞争，
不能直接赋值或使用没有同步保护的读改写。用 atomics 时要考虑累加精度和非确定性误差。
上游已包含 loss 的平均、microbatch 权重及可能的 loss scaling，反向不再除以 B、T 或 token 出现次数，
也不自行解除 loss scaling。

底层反向 wrapper 检查 grad_output 的前两维与 ids 一致、H/V 为正、dtype/device 符合契约，
并在 kernel 内检查 ID 在 `[0,V)` 内。底层只接受连续输入；Python backward 显式执行
`grad_output.contiguous()` 并将复制计入反向成本。特别是 `x.sum().backward()` 可能传入
零 stride 的展开梯度，不能仅凭 shape 按连续内存读取。

### 5.3 与共享参数和 autograd 的衔接

默认模型的 embedding 与 LM head 共享同一个 Parameter。整模型训练时，该权重还会收到
输出 Linear 的梯度，因此不能用“未作为输入出现的 token 行必须为零”检查共享权重的最终 `.grad`。
交由 autograd 累加两条分支，不要在 embedding backward 中清空或覆盖共享参数梯度。
本项目参考使用 dense gradient；第一版无需实现 sparse embedding optimizer 路径。

### 5.4 从绑定到 loss.backward() 的调用链

1. [扩展加载器](../../tiny_transformer/operators/_extension.py) 将 `bindings.cpp`、
   `embedding.cu` 和 `embedding_backward.cu` 一起传给 `torch.utils.cpp_extension.load`，
   延迟编译并加载同一个扩展模块。只在头文件声明函数不会编译其实现。
2. [bindings.cpp](../../csrc/bindings.cpp) 中的
   `m.def("embedding_backward", &embedding_backward_cuda, ...)` 将 C++ host wrapper
   暴露为 Python 的 `extension.embedding_backward(ids, gradient, vocab_size, implementation="grouped")`。
   **pybind 只提供可调用函数，不会自动把 forward 与 backward 关联起来。**
3. [student.py](../../tiny_transformer/operators/student.py) 在梯度开启且 weight 需要梯度时，
   调用 `_Embedding.apply(ids, weight, backward_impl)`。`apply` 创建 autograd 节点；其 `forward`
   在关闭梯度记录的上下文中调用扩展前向，并保存 ids、V 和 backward_impl。
   实现选择保存在每个节点的 ctx 中，同时存在 grouped/baseline 两张图也不会串用。
4. 下游执行 `loss.backward()` 或 `torch.autograd.grad(...)` 时，autograd 沿计算图把
   `[B,T,H]` 上游梯度传给 `_Embedding.backward(ctx, grad_output)`；用户不需要手动调用 kernel。
5. Python backward 将上游梯度转为连续张量，再调用扩展的 `embedding_backward`。
   C++ wrapper 校验输入，设置 device guard，分配 `[V,H]` 全零输出，并在当前 CUDA stream
   上启动选择的 kernel。grouped 在 warp 内合并相同 ID 后原子累加，baseline 由每个 warp
   直接原子累加一个 token 的梯度。两者共用输入检查、输出分配和清零代码。
6. Python backward 返回 `(None, dweight, None)`；autograd 将 dweight 与 LM head 分支、已有
   microbatch 梯度一起累加到同一个参数。算子自身不直接修改 `weight.grad`，也不更新 weight。

```python
import torch
from tiny_transformer.operators import student

ids = torch.tensor([[2, 1, 2]], device="cuda", dtype=torch.int64)
weight = torch.randn(7, 65, device="cuda", requires_grad=True)
x = student.embedding(ids, weight)  # x.grad_fn 对应 _EmbeddingBackward
x.sum().backward()                 # 自动调用绑定的 CUDA backward
# weight.grad.shape == (7, 65)
# 第 2 行全为 2，第 1 行全为 1，其余行全为 0。
```

`no_grad` / `inference_mode` 或 weight 不需要梯度时，直接调用扩展前向，不创建此节点。
直接调用 pybind 前向不会建立梯度关系，因此原生前向仍在需要 autograd 时拒绝调用，并提示使用
`student.embedding`。当前 backward 用 `once_differentiable` 明确限定一阶梯度；
尚未提供二阶反向或 `torch.compile` 的自定义算子注册。

单独使用 baseline 时调用 `student.embedding(ids, weight, backward_impl="baseline")`，
然后正常执行 `loss.backward()`。直接测试底层入口可调用
`extension.embedding_backward(ids, grad_output, V, "baseline")`；此时上游梯度必须连续。

反向使用浮点原子加，不保证逐 bit 确定性。开启 `torch.use_deterministic_algorithms(True)`
时，非空反向明确报错；`warn_only=True` 时警告后执行。空输入只返回全零表。

## 6. 建议的实现阶段与验收

当前 student 已接入 **CUDA FP32 连续输入的前向与一阶反向**，前向保留一 warp 一 ID 的标量和 float4 两条路径。
weight/output 指针均为 16 字节对齐且 H 能被 4 整除时使用 float4，否则使用标量 kernel。
支持空 IDs，拒绝 V/H 为零、非连续输入和低精度 weight。
扩展在首次调用时延迟编译，构建依赖和完整命令见 [csrc 说明](../../csrc/README.md)。
反向每个 warp 对最多 32 个输入位置按 ID 分组，所有 lane 协作处理通道；索引与地址计算使用 int64。
SM70 及以上使用 `__match_any_sync`，更早架构使用 shuffle/ballot 分组。
采用限制 grid 大小的 warp-stride 循环；尾部 token 和 H 不整除 32 均有对应处理。
baseline 每个 warp 只处理一个 token，每个 256 线程 block 处理 8 个 token；
因此默认 4096 个位置启动 512 个 block，而 grouped 启动 16 个 block。
baseline 同样使用 int64 索引和 warp-stride 循环，以覆盖 grid 截断后的剩余位置。
非法 ID 通过异步设备断言报错；不会静默跳过或通过 CPU 读回检查。

1. 在目标 GPU 上验收 FP32 前向/反向、尾部形状、重复 ID 和共享权重梯度。
2. 验证 FP32 master 参数的 AMP 训练；BF16/FP16 weight 仍不支持。
3. 比较高/低重复率下分组反向与逐 token 原子累加的耗时，再考虑通道切分等并行度优化。

```bash
python -m tiny_transformer.check_ops --operator embedding --backend student \
  --device cuda --precision fp32 --backward --output runs/embedding-fp32.json
python -m unittest discover -s tests -p 'test_student_embedding.py' -v
```

`check_ops --backward` 对照随机上游梯度；其中的计时字段仍只测前向，不能当作 backward 性能数据。
本文件对应的算子测试覆盖反向分组、空输入、非连续上游梯度、对齐、非法 ID 和 current stream。
AMP、共享参数训练与缓存推理统一放在 [test_student_integration.py](../../tests/test_student_integration.py)；
完整算子回归使用 `python -m unittest discover -s tests -p 'test_student_*.py' -v`。
测试职责见 [tests/README.md](../../tests/README.md)。
CPU 环境只验证 autograd wrapper 的接线（使用测试替身），CUDA 用例会明确跳过。

补充用例：所有 ID 相同、ID 0 与最大合法 ID、重复 BOS/EOS、非连续 IDs、真实词表，
以及共享 embedding/LM head 权重的整模型梯度。BF16 推理和 BF16 AMP 训练要分别检查。
不需要为了这一个算子实现 tokenizer 或 vocab projection。

## 7. 前向与反向性能：四种 token 分布

使用 [benchmarks/embedding.py](../../tiny_transformer/benchmarks/embedding.py) 在 CUDA 机器上比较
student 和 PyTorch reference。默认测量 FP32、`B=8,T=512,V=8192,H=768`，每组先验证
前向/反向数值，再分别测量两个阶段。首次扩展编译和预热不计入结果。

```bash
# A100；其他 GPU 请按实际架构设置。
export TORCH_CUDA_ARCH_LIST=8.0
python -m tiny_transformer.benchmarks.embedding \
  --device cuda --backward-impl all --patterns random same unique hot \
  --warmup 20 --repeats 100 --trials 5 \
  --output runs/embedding-performance.json
```

| pattern | 输入分布 | 观察重点 |
|---|---|---|
| `random` | 在整个词表中均匀随机采样 | 一般分布下的前向与反向耗时 |
| `same` | 所有位置的 ID 都为 0 | warp 内合并收益和跨 warp 原子竞争 |
| `unique` | ID 为 `0..B*T-1`，每个位置各不相同 | 没有重复 ID 时的分组开销 |
| `hot` | 在前 `min(hot_tokens,V)` 个 ID 中随机采样 | 热门 token 的重复与竞争；默认 16 个 |

当 `B*T > V` 时，`unique` 明确记录为 `skipped`，不会通过取模制造重复 ID。
控制台和 JSON 分别报告各分布的 `forward` / `backward`：`reference_us`、`candidate_us`、
`speedup`，JSON 另存每轮原始计时、`inputs.distinct_ids`、误差、参数和硬件环境。
`speedup = reference_us / candidate_us`，大于 1 表示 student 更快。

`--backward-impl grouped|baseline|all` 选择反向实现，默认 grouped。
`all` 在每种分布下使用相同的 ids、weight 和上游梯度，分别测量两个实现；控制台增加实现名，
JSON 的每条测量记录增加 `backward_impl`。两种实现都先与 PyTorch 检查数值再计时。
它们共用同一个前向，因此两行前向数据是重复测量，差异不代表前向算法发生了变化。
只测试 baseline 可运行：

```bash
python -m tiny_transformer.benchmarks.embedding \
  --backward-impl baseline --patterns random same unique hot \
  --output runs/embedding-baseline.json
```

计时使用当前设备/stream 上的 CUDA Event，每轮执行 repeats 次并计算平均耗时，最后取
trials 轮的中位数；轮次交替 reference/student 的测量顺序。前向使用不需要梯度的 weight。
反向复用计时前构建的计算图，调用 `autograd.grad(..., retain_graph=True)`：
**包含梯度表分配、清零、autograd 调度及梯度计算，不包含前向，也不累积到 `weight.grad`。**
两种实现使用相同的连续 FP32 上游梯度，随机正态值除以 `B*T`，模拟平均 loss 的缩放。

这是固定输入、缓存和 allocator 预热后的 eager 调用区间，可能包含 CPU 提交造成的 GPU 空隙，
不等于单个 kernel 的纯执行时间，也不代表完整训练 step。需要分离清零和累加 kernel 时使用
Nsight Systems/Compute；不要把清零移出算子反向计时来计算加速比。

可改变形状或单独比较热门 token 数量：

```bash
python -m tiny_transformer.benchmarks.embedding \
  --batch-size 1 --seq-length 512 --vocab-size 8192 --dim 768 \
  --output runs/embedding-b1-t512.json
python -m tiny_transformer.benchmarks.embedding \
  --patterns hot --hot-tokens 4 --output runs/embedding-hot4.json
```

该脚本不提供 CPU 性能替代结果；当前 student 只支持 FP32 weight，不能据此推断低精度性能。

## 统一性能测试入口

本算子与其余七个算子共用 [benchmarks 测量框架](../../tiny_transformer/benchmarks/README.md)：
先校验数值与可用梯度，再用 CUDA events、交替后端顺序、多轮中位数分别测前向/反向。

```bash
python -m tiny_transformer.benchmarks --operator embedding --backward-impl all \
  --output runs/embedding-performance.json
```

未实现的 student 算子/阶段会记录为 `skipped`，没有隐式 reference fallback；
可用 `--backend reference` 验证完整测量流程。`check_ops --backward` 的结果不能替代反向性能数据。
