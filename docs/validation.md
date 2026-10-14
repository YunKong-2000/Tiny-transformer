# 验证记录与范围

实现范围、已编写测试和已经执行的验证是不同信息。本页索引现有记录，不将代码存在或测试用例存在描述为全量 GPU 验收通过。

| 范围 | 已有证据 | 边界 |
| --- | --- | --- |
| 参考框架与数据流程 | [2026-09-23 框架验收](history/framework_validation_20260923.md)：CPU 测试、短程训练、续训、KV cache、数据 hash | 不用于 GPU 性能结论 |
| BF16 Attention 前向 | [A100 报告](reports/attention_bf16_optimization_report.md)：前向延迟、NCU 和优化取舍 | 固定形状历史记录；部分后期记录缺完整 CLI |
| BF16 Attention 反向 | [2026-10-13 接入记录](../csrc/attention/development.md)：preprocess/dQ/dK/dV、一阶 autograd 和测试覆盖 | 尚未附完整 GPU 数值/竞态/性能验收记录 |
| 其他自定义算子 | [测试职责](../tests/README.md) 与 [模块记录索引](performance.md) | 逐项查看 dtype、shape、版本和基线；无全算子/全精度保证 |

## 运行回归

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p 'test_student_*.py' -v
python -m unittest discover -s tests -p 'test_attention_backward_*.py' -v
```

无 CUDA 时，GPU 用例会跳过；跳过不是通过。首次 CUDA 测试需要 nvcc、C++ 编译器、Ninja 和完整 CUTLASS 子模块。目标平台可使用 `bash scripts/a100_validate.sh` 检查参考模型训练、SDPA、compile 和 profiling 流程；该脚本不等于全部自定义 kernel 的验收。

新增测量应保留 commit、GPU/驱动/CUDA/PyTorch 版本、命令、数值误差、测试结果和原始 trial。历史 `data/`、`runs/` 不随 Git 分发，运行方法见 [模型流程](model_workflow.md) 与 [离线部署](offline.md)。
