# 学生 CUDA / CuTe / CUTLASS 源码目录

此目录刻意不提供 kernel。建议逐步加入 `rms_norm.cu`、`rope.cu`、
`linear_cutlass.cu` 和绑定代码，从 `tiny_transformer/operators/student.py` 调用。

先用 `torch.utils.cpp_extension.load` 做单算子迭代，再决定是否迁移为预编译扩展。
将 CUTLASS 固定在一个明确的 commit，记录在实验元数据中；不要使用浮动 main 做最终报告。
在 A100 上设置 `TORCH_CUDA_ARCH_LIST=8.0`。不要在 Python import 时自动下载依赖或编译全部算子。

工程契约、autograd、current stream、非连续张量和 CuTe 学习任务见
[开发指南](../docs/development.md)。
