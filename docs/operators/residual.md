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

## 统一性能测试入口

本算子与其余七个算子共用 [benchmarks 测量框架](../../tiny_transformer/benchmarks/README.md)：
先校验数值与可用梯度，再用 CUDA events、交替后端顺序、多轮中位数分别测前向/反向。

```bash
python -m tiny_transformer.benchmarks --operator residual \
  --output runs/residual-performance.json
```

未实现的 student 算子/阶段会记录为 `skipped`，没有隐式 reference fallback；
可用 `--backend reference` 验证完整测量流程。`check_ops --backward` 的结果不能替代反向性能数据。
