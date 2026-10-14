# 算子开发与集成规范

本页说明自定义 CUDA 后端的接口、布局、梯度和测量要求。数学定义见 [Transformer 数据流](transformer.md)，各算子契约见 [接口导航](operators/README.md)。

## 后端与职责

| 层 | 职责 |
| --- | --- |
| `operators/reference.py` | PyTorch 数学参考与解析梯度 |
| `operators/student.py` | 自定义后端公共接口、布局转换与一阶 autograd |
| `operators/_extension.py` | 独立 JIT 构建、CUTLASS 头文件配置 |
| `csrc/<operator>/` | 参数检查、kernel 分派和 CUDA 计算 |
| `model.py` | 参数共享、block 编排、位置及 KV cache 管理 |
| `benchmarks/` | 校验、预热、计时、样本和环境元数据 |

`student` 是现有 API 后端标识。`--op linear=student` 替换模型中所有 Linear 调用点，包括 QKV/O/gate-up/down/LM head；未覆盖的 shape 或精度显式报错，不进行隐式 fallback。当前支持范围见 [首页](../README.md#算子支持范围)。

## 数学与精度

- 保持输入输出的数学语义、逻辑 shape、dtype、device 与梯度语义；无需逐 bit 等价。
- 明确区分 FP32 SIMT、TF32 乘法与 BF16/FP16 输入的 FP32 累加。仅比较输入 dtype 不足以判断基线精度一致。
- 归约顺序、融合及低精度舍入应使用最大绝对误差、相对误差和必要的 RMS 判据验证。
- AMP、训练模式和梯度模式是不同开关。自定义 pybind 不自动继承 PyTorch 内置算子的 autocast 规则。
- Attention 当前支持 BF16 一阶反向，FP32 仅前向；不据此推断全模型自定义 BF16 训练已支持。

## 布局与 CUDA 执行

1. 输入检查在 kernel 启动前完成，覆盖 dtype、device、shape、stride、alignment 和索引范围。
2. 使用输入设备的 device guard 与 PyTorch 当前 stream；不在普通算子调用中做全设备同步。
3. `data_ptr()` 已包含 view 起点，避免重复累加 `storage_offset`。元素 stride 与字节偏移分别处理。
4. 必要的 contiguous/clone/cast 在 wrapper 中显式执行，完整算子计时包含这些开销。
5. 不原位修改输入，不把输出 alias 到仍需保留的激活。空张量、尾块、未对齐 view 必须有确定行为。
6. `cp.async` 的线程内等待与 CTA barrier 分别保证完成及跨线程可见性，不能互相替代。

CuTe 的布局检查应覆盖逻辑坐标→物理地址、tile→线程/value 映射、MMA fragment 与 copy fragment 的兼容性。地址唯一性、转置/swizzle 和尾块有效范围可独立检查，最终仍需 CUDA 数值与竞态验证。

## Autograd

`torch.autograd.Function` 保存反向所需状态；backward 按 forward 参数顺序返回梯度，离散参数返回 `None`。上游梯度可能非连续、零 stride 或包含外部缩放。kernel 不直接覆盖参数 `.grad`，不执行 optimizer step。

- Embedding 保存 IDs 与表尺寸；原子累加前清零梯度表。
- RMSNorm 缓存每行逆 RMS；dGamma 的跨行归约具有非确定性。
- Attention 保存 Q/K/V/O/LSE，反向 preprocess 计算行归约后生成 dQ/dK/dV。
- Cross Entropy 缓存 LSE 与有效样本数，处理 ignore_index 和外部 loss 梯度。

目前自定义路径以一阶梯度为范围；JIT/pybind 接入不等于 `torch.compile` / 自定义算子注册支持。

## 回归验证

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p 'test_student_*.py' -v
python -m tiny_transformer.check_ops --operator rms_norm --backend student \
  --device cuda --precision fp32 --backward --output runs/rmsnorm-check.json
```

覆盖单算子前向/梯度、非法输入、stride/offset、tail、stream、参数共享及模型调用。`check_ops` 是小形状入口检查，专属测试负责固定 D=64 Attention 等接口的完整边界。FP64 gradcheck 仅用于支持 FP64 的参考函数；低精度 kernel 与独立解析梯度对照。

GPU 用例在无 CUDA 时跳过，host 测试不能替代 GPU 验收。需要检查异步流水线时，可运行：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python -m unittest discover -s tests -p 'test_student_attention.py'
compute-sanitizer --tool racecheck --error-exitcode 1 \
  python -m unittest discover -s tests -p 'test_student_attention.py'
```

测试职责见 [tests](../tests/README.md)，已执行记录见 [validation](validation.md)。

## 性能分析

先确认计时边界，再评估改动。完整 operator、预分配 Graph 与模型请求指标不能互换。对照需匹配输入、精度、后处理、缓存状态及预热；新结果保存原始样本、环境与 commit。

Nsight Compute 用于定位而非代替无 profiler 的延迟：关注 registers/thread、shared memory/CTA、occupancy、bank conflict、memory throughput 和 warp stall。单一指标改善不保证总耗时降低，性能结论以同口径计时为准。

参考案例：[Attention 优化报告](reports/attention_bf16_optimization_report.md) 保留更紧 launch bounds、寄存器预取和输出重排的退化/持平结果。[性能指南](performance.md) 汇总已保存的数据与复现入口。

## 资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS / CuTe](https://github.com/NVIDIA/cutlass)
- [PyTorch C++/CUDA Extension](https://pytorch.org/docs/stable/cpp_extension.html)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
