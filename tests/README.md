# 测试入口与职责

日常开发只需要区分“正确性回归”和“性能测量”。正确性测试放在 `tests/`，
性能实现放在 `tiny_transformer/benchmarks/`。不把“耗时大于零”当作算子正确性测试。

## 正确性回归：统一用 unittest

在仓库根目录运行：

```bash
# 全部回归；没有 CUDA 时，GPU 用例自动跳过。
python -m unittest discover -s tests -v

# 只改了两个学生算子时：包含 kernel、autograd 和共用的模型集成测试。
python -m unittest discover -s tests -p 'test_student_*.py' -v
```

首次运行 GPU 测试会 JIT 编译真实扩展，需要 CUDA 版 PyTorch、nvcc、C++ 编译器和 Ninja。
有 GPU 但缺少构建工具时会报错，不当成测试通过。A100 可设置 `TORCH_CUDA_ARCH_LIST=8.0`。
单独迭代一个 kernel，也可将 pattern 改成 `test_student_embedding.py` 或 `test_student_rms_norm.py`；
提交前仍运行包含 integration 的命令。

| 文件 | 唯一职责 |
|---|---|
| `test_student_embedding.py` | embedding 的精确前向、grouped/baseline 梯度、行/通道边界、地址对齐、空输入、非法 ID、stream |
| `test_student_rms_norm.py` | 一组用例检查 Y/R/dX/dgamma；H=1024/1025、非连续输入、缓存 R、清零、行循环与 stream |
| `test_student_integration.py` | 两个算子的共用接入：CPU 拒绝/延迟加载、native grad guard、deterministic 模式、FP32/AMP 模型梯度、prefill/decode |
| `test_operators.py` | reference 数学定义、double gradcheck、SDPA、算子分发契约 |
| `test_model.py` | 模型结构、因果性、文档隔离、reference KV cache、训练和 compile 基础行为 |
| `test_benchmarks.py` | 公共测量框架自身：输入分布、校验后计时、反向不重跑前向、采样顺序、统计与跳过规则 |
| `test_data_training.py` | 数据准备、packing、tokenizer、续训一致性 |
| `test_offline_bundle.py` | 离线文件校验与路径合法性 |

每个学生算子只保留一个 CPU autograd 接线测试，使用明确的测试替身。
它用于没有 GPU 时检查 Python 接线，不证明 CUDA 数值正确。
GPU 用例不会使用 reference fallback；比较 reference 只是提供预期结果。

## 性能测量：统一用 benchmarks

```bash
python -m tiny_transformer.benchmarks --operator embedding \
  --backward-impl all --output runs/embedding-performance.json
python -m tiny_transformer.benchmarks --operator rms_norm \
  --layouts contiguous strided last-only --output runs/rmsnorm-performance.json
```

命令自身会先校验结果，再预热、用 CUDA events 测量，输出原始样本、中位数和加速比。
未实现阶段明确跳过，编译和数值错误直接失败。完整参数和计时边界见
[性能说明](../tiny_transformer/benchmarks/README.md)。模型 TTFT/TPOT 使用 `benchmarks.model`。

`python -m tiny_transformer.check_ops` 是单算子小形状开发检查，不是另一套完整验收入口；
其 `--backward` 只检查梯度，不测反向耗时。需要回归时运行 unittest，需要性能时运行 benchmarks。

## 本次精简

- 删除 `test_benchmark_embedding.py`。ID 分布的独有检查并入 `test_benchmarks.py`；
  删除与公共框架重复的梯度、错误候选和 CUDA 计时检查。
- 删除两个算子各自的模型训练/缓存测试，合并为公共集成测试；保留单独启用及组合启用的对照。
  reference 缓存与完整前向的一致性继续由 `test_model.py` 负责。
- 删除加载器文件名/缓存调用次数等实现细节 mock，以及主要重复验证 PyTorch 自身版本计数、
  二阶梯度装饰器和 SGD 更新行为的测试。
- 合并 RMSNorm 前向/反向的 shape、stride 扫描，移除同一输入的重复 native/public 数值对照。
  embedding 按行边界和通道边界分别选代表值，减少全组合；非法 ID 保留每条实现的负数/越上界检查。
- 删除多 GPU 专项，聚焦当前单 GPU 项目；保留非默认 stream、边界检查和原子累加的确定性契约。

保留 grid 上限和非法 ID 用例，是因为它们分别覆盖容易遗漏的循环复用及设备越界。
精简不改变算子实现，也不提高原有数值误差阈值。
