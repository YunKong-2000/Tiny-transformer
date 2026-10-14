# 文档导航

## 项目与实现

- [项目概览](../README.md)：实现范围、性能摘要和快速运行。
- [算子接口](operators/README.md)：八类算子的输入、输出、精度与梯度契约。
- [CUDA 源码结构](../csrc/README.md)：JIT、绑定和 autograd 调用链。
- [Transformer 数学与数据流](transformer.md)：模型结构、张量布局、梯度和 KV cache。
- [开发规范](development.md)：新增或调整算子时的接口与验证要求。

## 性能与验证

- [性能结果与测量方法](performance.md)：Attention 数据及其他算子的记录索引。
- [BF16 Attention 优化报告](reports/attention_bf16_optimization_report.md)：tile、流水线、寄存器、mask 和归约的取舍，包含退化实验。
- [性能测试 CLI](../tiny_transformer/benchmarks/README.md)：operator / Graph / 模型计时边界。
- [测试说明](../tests/README.md)：数值、梯度、缓存、数据和续训回归。
- [验证记录](validation.md)：历史执行结果与当前验证边界。

## 运行与部署

- [训练、生成和模型测量](model_workflow.md)。
- [离线部署](offline.md)。
- [CUTLASS 依赖](../third_party/README.md)。

`csrc/*/development*.md` 等文件保存具体模块的工程记录。历史日志按当时的版本和计时方法解释；当前接口范围以算子文档和源码入口为准。
