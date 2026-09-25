# 学生 CUDA / CuTe / CUTLASS 源码目录

当前包含 embedding 的 FP32 CUDA 前向与一阶反向。前向每个 warp 搬运一个 ID 对应的行，
256 线程的 block 同时处理 8 个 ID；按地址与行宽对齐选择 float4 或标量 kernel。
反向每个 warp 对最多 32 个位置按 ID 分组，合并组内梯度后原子累加到全零梯度表。
另提供一 warp 一 token 的 baseline 反向，共用输入检查、清零和 stream 管理。
另包含 RMSNorm 的 FP32 CUDA 前向，每个 warp 对一行归约；Python 入口显式复制非连续输入，
支持 `[..., H]` 与 `[H]`，前向返回 Y 和每行的 R。共享内存反向支持 H <= 1024，
通过 autograd 缓存 R；dGamma 使用原子加，尚不支持低精度输入或二阶梯度。
其他算子尚未实现；CUDA 编译、数值及性能需在目标 GPU 上验收。

RMSNorm 使用独立的 `rms_norm/bindings.cpp` 和 `load_rms_norm_extension()`，
只编译本算子的绑定与 CUDA 源码，避免与 embedding 互相产生未定义符号。
逐步接入说明、代码检查结果与测试命令见 [RMSNorm 文档](../docs/operators/rms_norm.md#7-前向缓存与-pytorch-调用链)。

调用链：`student.embedding` → `operators/_extension.py` 的延迟 JIT 加载 →
`bindings.cpp::embedding_forward` → `embedding_forward_cuda` → CUDA kernel。
训练时由 `student._Embedding.apply` 建立 autograd 节点，保存 ids、V 和 backward_impl；执行
`loss.backward()` 时进入 `_Embedding.backward`，再经 `bindings.cpp::embedding_backward`
调用 `embedding_backward_cuda` 和选定的反向 kernel，返回 `(None, dweight, None)` 给 autograd。
pybind 绑定只暴露函数，forward/backward 的关联由 Python `torch.autograd.Function` 建立。
逐步说明见 [embedding 开发文档](../docs/operators/embedding.md#54-从绑定到-lossbackward-的调用链)。
首次调用需要 CUDA 版 PyTorch、CUDA toolkit/nvcc、C++ 编译器和 Ninja；
后续使用 PyTorch 的编译缓存。当前从源码 checkout 或 editable install 加载 `csrc`，
不支持不含这些源码的普通 wheel。此算子不依赖 CUTLASS。

在 A100 的 CUDA 开发环境中，从仓库根目录运行：

```bash
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=2
python -m tiny_transformer.check_ops --operator embedding --backend student \
  --device cuda --precision fp32 --backward --output runs/embedding-fp32.json
python -m unittest discover -s tests -p 'test_student_*.py' -v
python -m tiny_transformer.benchmarks.embedding \
  --device cuda --backward-impl all --patterns random same unique hot \
  --output runs/embedding-performance.json
python -m tiny_transformer.benchmarks.model --config configs/smoke.json \
  --device cuda --precision fp32 --op embedding=student \
  --prompt-length 16 --new-tokens 8 --output runs/embedding-inference.json
```

测试会在有 CUDA 时真实编译扩展；没有 CUDA 时跳过 GPU 用例。
`benchmarks.embedding` 通过统一框架测量四种 token 分布的前向与反向，并与 PyTorch 比较。
全部八个算子共用 [benchmarks/](../tiny_transformer/benchmarks/README.md) 的测量方法；
RMSNorm 性能入口为 `python -m tiny_transformer.benchmarks --operator rms_norm --phases forward backward`。
`--backward-impl all` 同时测试 grouped/baseline；也可仅指定其中一个，默认 grouped。
反向计时包含梯度表清零，不含前向；参数和结果含义见
[embedding 性能测试说明](../docs/operators/embedding.md#7-前向与反向性能四种-token-分布)。
模型缓存对照测试临时将 FP32 matmul precision 设为 `highest`，结束后恢复原设置，
避免 TF32 下不同 GEMM 尺寸的数值差异干扰 FP32 验收。
纯 reference 缓存与完整前向的一致性由 `test_model.py` 负责；两个学生算子的共同训练/推理
放在 `test_student_integration.py`。各算子文件保留独有的数值与边界用例，职责见 [测试说明](../tests/README.md)。
支持连续二维 int64 IDs、连续二维 FP32 weight，以及非零 storage offset；
支持空 IDs，但 weight 的 V/H 必须为正。非连续输入显式报错。
训练通过 `student.embedding` 使用一阶反向；直接调用原生前向不会建立 autograd 图，
因此原生前向在梯度开启且 weight 需要梯度时拒绝直接调用。
反向原生入口要求连续 FP32 `[B,T,H]` 梯度；Python wrapper 显式复制非连续上游梯度。
尚不支持二阶梯度、BF16/FP16 weight 或 torch.compile 集成。
AMP 下 FP32 weight/output/gradient 保持 FP32。
浮点原子加存在非确定性，严格 deterministic 模式下非空 backward 明确报错。

非法 ID 在 kernel 内通过设备断言报告，不做主机读回或设备同步。
错误可能在后续同步时才被观察到，触发后须重启该 CUDA 进程；
测试用独立子进程分别验证向量与标量路径的非法 ID。

使用 `torch.utils.cpp_extension.load` 做单算子迭代，再决定是否迁移为预编译扩展。
将 CUTLASS 固定在一个明确的 commit，记录在实验元数据中；不要使用浮动 main 做最终报告。
在 A100 上设置 `TORCH_CUDA_ARCH_LIST=8.0`。不要在 Python import 时自动下载依赖或编译全部算子。

工程契约、autograd、current stream、非连续张量和 CuTe 学习任务见
[开发指南](../docs/development.md)。
