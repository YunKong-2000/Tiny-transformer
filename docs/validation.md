# 当前验证记录

日期：2026-09-23。此记录说明实际运行过哪些路径，不能视为 A100 性能报告。

## 离线数据准备更新

后续使用本机已配置的系统代理成功访问官方 Hugging Face 数据源，完成真实数据准备。

- 固定 revision：`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`。
- 前 100,000 条 train 记录去掉 17 条空文本，得到 99,983 篇、21,729,332 tokens。
- validation：2,000 篇、385,234 tokens。
- BPE：8,192 vocab；只使用前 20,000 篇非空 train 文本训练；原文、数据卡片和许可均保留。
- 所有文本/token SHA256、文档/token 数、token ID 范围已核对。
- 在独立环境仅安装 tokenizers wheel，未安装 datasets/huggingface_hub，7 段文本的离线 encode/decode 往返通过。
- 使用真实 token 数据和 8K 词表的小模型完成 3 步 CPU 训练与验证；用于数据接入验收，不是质量训练。
- 新增离线包完整性与损坏检测测试；最新测试共 26 项，25 通过、1 项 CUDA 测试跳过。
- Linux x86_64 tokenizer wheel 已下载；Linux/NGC GPU 执行仍待目标服务器验证。

使用方法见 [离线部署指南](offline.md)。以下保留初次框架验收记录。

## 环境与限制

- 本地 macOS，Python 3.9.6，PyTorch 2.8.0，CPU。
- 当前没有 CUDA GPU，也没有 Docker CLI，因此未拉取/启动 `nvcr.io/nvidia/pytorch:25.08-py3`。
- 初次验收时未安装数据依赖；后续已在项目专用 `.venv-data` 内安装并完成上方真实数据验收。
- 初次直连 TinyStories 页面超时；后续通过本机系统代理解决。synthetic smoke 与真实 TinyStories 分别保存在不同目录。
- 学生 kernel 全部留空，未声明任何 GPU 优化收益。

## 已通过

| 项目 | 结果 |
|---|---|
| 单元/集成测试 | 24 项：23 通过，1 项 CUDA BF16 测试跳过 |
| 60M 配置参数量 | meta device 核对为 62,927,616 |
| 数据准备 | 500 篇训练、50 篇验证；59,385 / 5,871 byte tokens；manifest 含 hash |
| 完整 CPU smoke 脚本 | 数据、测试、10 步训练、checkpoint、生成、benchmark、profile 全部执行成功 |
| 短程训练 | 98,816 参数 smoke 模型，训练 loss $5.5493\to4.8858$；最后验证 loss $4.9083$ |
| 精确续训 | 固定总 schedule，4 步连续训练与 $2+2$ 步恢复后所有权重逐元素相同 |
| 文档隔离 + SDPA | 4 步 CPU 训练完成，跨 EOS 的 target 被正确排除 |
| SDPA 前向与反向 | prefill/decode 对照 reference 通过；FP32 前向最大绝对误差分别约为 $2.98\times10^{-7}$、$1.79\times10^{-7}$ |
| CPU Inductor benchmark | 完成冷请求、预热和稳态采样；记录编译分项与 Dynamo counters |
| CPU Inductor 正确性 | compiled prefill+逐 token decode 对照完整 eager 前向，最大绝对误差约 $2.38\times10^{-7}$ |
| Profiler | train/decode 的 trace、算子表、前三热点 JSON 均生成 |
| 配图 | 4 张 PNG 与对应 SVG 已生成并检查可读性 |

生成样例仍会重复字符，这是仅 10 步 smoke 训练的预期局限，不代表完成语言模型质量训练。
CPU benchmark 数字仅验证测量流程，不能用于 A100 或 60M 项目的简历指标。

本地原始产物位于：

```text
data/smoke/
runs/verified-smoke/train/metrics.jsonl
runs/verified-smoke/train/last.pt
runs/verified-smoke/benchmark.json
runs/verified-smoke/profile/
runs/local-smoke/sdpa-check.json
runs/local-smoke/compile-cpu.json
runs/local-smoke/profile-train/
runs/isolated-sdpa-smoke/
```

`data/`、`runs/` 被 gitignore 排除；需要保存结果时主动归档，并使用脚本复现。

## A100 上的下一步验收

按 [README](../README.md) 启动官方容器并安装可选数据依赖，然后：

```bash
bash scripts/a100_validate.sh

python -m tiny_transformer.prepare --source tinystories \
  --output data/tinystories-8k --train-docs 100000 --val-docs 2000 --vocab-size 8192

python -m tiny_transformer.train --config configs/model_60m.json \
  --data data/tinystories-8k --output runs/60m-initial \
  --device cuda --op attention=sdpa --stop-after 100
```

GPU 验收脚本包括三种训练精度、BF16 SDPA 对照、compile benchmark 和 decode profile。
随后再测真实模型的训练/prefill/decode，检查 Flash 后端选择、FP16 overflow、图中断/重编译，
并保存容器 digest、驱动和编译器版本。

Nsight DRAM/Roofline、学生 CuTe/CUTLASS kernel、真实语料质量训练均属于尚未执行的实验。
