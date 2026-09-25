# residual：保留 dtype promotion 的残差相加

[返回算子总契约](README.md) · [参考实现](../../tiny_transformer/operators/reference.py) · [调用位置](../../tiny_transformer/model.py)

## 1. 接口与职责

```python
residual(x: torch.Tensor, update: torch.Tensor) -> torch.Tensor
```

每个 block 调用两次：attention 的 O Linear 之后一次，FFN 的 Down Linear 之后一次。
`x` 是进入对应分支之前保留下来的 residual stream；`update` 是分支新计算的更新量。
输出成为下一阶段的 residual stream，默认每次模型前向调用 16 次。

## 2. 输入与输出

| 项目 | shape | 含义 | 梯度 |
|---|---|---|---|
| x | $[B,T,H]$ | 原 residual | 训练需要 dx |
| update | 与 x 相同 | O/Down Linear 的输出 | 训练需要 dupdate |
| 返回值 y | 与 x 相同 | 两者相加后的新 residual | 继续向后传播 |

默认训练为 $[8,512,768]$，单 token decode 为 $[B,1,768]$。
两输入位于同一设备，但**训练中不一定同 dtype**。

| 模式 | x | update | 参考输出 |
|---|---|---|---|
| FP32 训练 | FP32 | FP32 | FP32 |
| BF16 AMP 训练 | FP32 | BF16 | FP32 |
| FP16 AMP 训练 | FP32 | FP16 | FP32 |
| BF16 inference | BF16 | BF16 | BF16 |
| FP16 inference | FP16 | FP16 | FP16 |

dtype 遵守本项目浮点输入下的 `torch.add` promotion 规则，不是强制跟随 update 或 autocast dtype。
验证阶段即使 `model.eval()`，也仍可能是 FP32 参数配合 AMP，所以不能仅由 eval/train 判断两输入类型。

## 3. 数学语义和边界

$$
Y_{b,t,h}=X_{b,t,h}+U_{b,t,h}.
$$

没有缩放系数、bias、dropout、RMSNorm 或其他激活；不能除以 2 或引入残差缩放。
BF16 AMP 训练应在 FP32 输出语义下相加：先将 update 的值转换到合适计算类型，再与 FP32 x 相加。
先把 x 降为 BF16 相加、最后转回 FP32，会永久丢失 x 的低位信息，不符合参考路径。

该操作返回新的结果，不能直接执行 `x.add_(update)`。
原 x 可能已被分支中的 norm/Linear 保存供 backward 使用，原位覆盖会破坏数值或触发 autograd version 错误。
即使 inference-only 版本希望复用 storage，也应通过新的明确契约处理，而不是静默改变当前接口。

## 4. 布局与广播范围

当前模型传入相同 shape 的浮点张量，通常连续；从其他 student 算子返回时可能有不同 stride。
可先为连续、相同 shape 提供 fast path，对范围外输入明确报错。
PyTorch 的 `x + update` 还支持广播和更广的 dtype，本项目首版不要求实现整数、复数或任意广播。

如果声明支持广播，backward 需要把广播扩展维度的梯度求和回原 shape，不能直接返回整张上游梯度。
如果只支持本项目相同 shape，应在 wrapper 中检查，不能将 shape 不同的张量按同元素数误读。

## 5. 训练反向

相同 shape 下，设上游梯度为 $G$：

$$
dX=G,\qquad dU=G.
$$

这是数学值的关系；实际返回 dtype 仍对应原输入。
例如 x 是 FP32、update 是 BF16 时，dx 为 FP32，dupdate 对应 BF16，
从输出梯度到 update 梯度的低精度转换不能遗漏。
backward 返回 `dx, dupdate`，不额外除 batch/sequence，也不修改参数梯度。

残差使上游激活有两条梯度路径：一条经过分支，一条经过此处直接相加。
应交由 autograd 汇总两条路径，而不是在 residual kernel 内手动寻找和更新其他 tensor 的 `.grad`。

## 6. 开发与验收建议

此算子的数学很简单，但可用于练习多 dtype dispatch、向量化访存和输出类型契约。
单独相加通常受带宽和 launch 限制；是否值得替换需要看整个模型中的时间占比。

```bash
python -m tiny_transformer.check_ops --operator residual --backend student \
  --device cuda --precision fp32 --backward --output runs/residual-fp32.json
```

该工具只生成同 dtype 输入，必须额外测试 FP32+BF16、FP32+FP16，以及两条输入的梯度 dtype。
加入 x 含低精度难以表示的小增量的用例，能发现“先降精度再相加”的错误。
另检查输出不修改输入、非连续输入、不同 CUDA stream、尾部元素和完整 block 的 backward。

residual+RMSNorm 的融合不在当前接口内：norm 需要 weight/eps，且后续仍需要未归一化的 residual。
到融合阶段应明确设计参数和两个结果，而不是把当前输出悄悄改成 normalized x。

## 7. 当前 student 实现与调用链

`student.residual(x, update)` → `load_residual_extension()` 延迟编译独立扩展 →
`residual/bindings.cpp` → `residual_forward_cuda`。首次调用需要 CUDA 版 PyTorch、
nvcc、C++ 编译器和 Ninja；import 模型或 reference 路径不会触发编译。

- 原生接口只接受同设备、同 shape、连续的 FP32 CUDA 张量；支持标量和任意维度，
  包括空张量。空输入直接返回，不发射 kernel。不支持广播。
- Python 入口接受 FP32、FP16、BF16 及三者的混合输入。先显式转换为连续 FP32，
  经自定义 kernel 相加，再转回 `torch.promote_types` 的结果类型。
  这是正确性基线，低精度路径尚未使用专用 kernel；转换和分配成本计入性能测试。
- 训练通过 `_Residual.apply` 建立 autograd 节点，反向调用 `residual_backward(dY)`，
  返回独立分配的 `dX` 和 `dUpdate`。导数为常数，不需要缓存 X、update 或 Y。
  wrapper 的转换/复制留在计算图中，将梯度映射回输入 dtype 和原始视图。
- 非连续上游梯度（例如 `sum()` 产生的展开视图）会先显式复制。前后向都检查 float4
  地址对齐和元素数整除条件，否则用标量路径；使用 64 位索引、device guard、
  current CUDA stream 和 launch error 检查，无设备同步或原子累加。
- 仅支持一阶梯度。直接调用 native API 不会建立 autograd 图，梯度模式下传入
  需要梯度的张量会明确报错；训练应使用 `student.residual`。尚未接入 `torch.compile`。

正确性回归：

```bash
python -m unittest discover -s tests -p 'test_student_residual.py' -v
python -m unittest discover -s tests -p 'test_student_*.py' -v
```

专属用例覆盖精确前向/反向、空输入/标量/尾部、各输入地址对齐、非连续视图、
相同输入的梯度累加、混合 dtype 和 AMP 精度、非法参数、非默认 stream、grid 上限循环。
最大 grid 用例需要约 1.4 GB 显存；共享 integration 覆盖单独启用 residual 和三个算子
共同启用时的 FP32/AMP 模型梯度、prefill/decode、native grad guard 和确定性模式。
无 CUDA 时只执行 CPU autograd 接线测试并跳过 GPU 项目，不能据此认定 CUDA 验收通过。

## 统一性能测试入口

本算子与其余七个算子共用 [benchmarks 测量框架](../../tiny_transformer/benchmarks/README.md)：
先校验数值与可用梯度，再用 CUDA events、交替后端顺序、多轮中位数分别测前向/反向。

```bash
python -m tiny_transformer.benchmarks --operator residual \
  --output runs/residual-performance.json
```

未实现的 student 算子/阶段会记录为 `skipped`，没有隐式 reference fallback；
可用 `--backend reference` 验证完整测量流程。`check_ops --backward` 的结果不能替代反向性能数据。
