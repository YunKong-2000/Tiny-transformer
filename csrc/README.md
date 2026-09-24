# 学生 CUDA / CuTe / CUTLASS 源码目录

当前包含 embedding 的 FP32 CUDA 前向：每个 warp 搬运一个 ID 对应的行，
256 线程的 block 同时处理 8 个 ID；按地址与行宽对齐选择 float4 或标量 kernel。
其他算子尚未实现。

调用链：`student.embedding` → `operators/_extension.py` 的延迟 JIT 加载 →
`bindings.cpp::embedding_forward` → `embedding_forward_cuda` → CUDA kernel。
首次调用需要 CUDA 版 PyTorch、CUDA toolkit/nvcc、C++ 编译器和 Ninja；
后续使用 PyTorch 的编译缓存。当前从源码 checkout 或 editable install 加载 `csrc`，
不支持不含这些源码的普通 wheel。此算子不依赖 CUTLASS。

在 A100 的 CUDA 开发环境中，从仓库根目录运行：

```bash
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=2
python -m tiny_transformer.check_ops --operator embedding --backend student \
  --device cuda --precision fp32 --output runs/embedding-fp32.json
python -m unittest discover -s tests -p 'test_student_embedding.py' -v
python -m tiny_transformer.benchmark --config configs/smoke.json \
  --device cuda --precision fp32 --op embedding=student \
  --prompt-length 16 --new-tokens 8 --output runs/embedding-inference.json
```

测试会在有 CUDA 时真实编译扩展；没有 CUDA 时跳过 GPU 用例。
支持连续二维 int64 IDs、连续二维 FP32 weight，以及非零 storage offset；
支持空 IDs，但 weight 的 V/H 必须为正。非连续输入显式报错。
weight 需要梯度且梯度模式开启时显式拒绝；验证/推理请用 `no_grad` 或 `inference_mode`。
尚不支持 backward、BF16/FP16 weight 或 torch.compile 集成。
AMP 验证中的 FP32 weight/output 保持 FP32。

非法 ID 在 kernel 内通过设备断言报告，不做主机读回或设备同步。
错误可能在后续同步时才被观察到，触发后须重启该 CUDA 进程；
测试用独立子进程分别验证向量与标量路径的非法 ID。

使用 `torch.utils.cpp_extension.load` 做单算子迭代，再决定是否迁移为预编译扩展。
将 CUTLASS 固定在一个明确的 commit，记录在实验元数据中；不要使用浮动 main 做最终报告。
在 A100 上设置 `TORCH_CUDA_ARCH_LIST=8.0`。不要在 Python import 时自动下载依赖或编译全部算子。

工程契约、autograd、current stream、非连续张量和 CuTe 学习任务见
[开发指南](../docs/development.md)。
