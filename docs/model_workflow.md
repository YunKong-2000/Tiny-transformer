# 训练、推理与模型性能测试

以下命令在仓库根目录的 CUDA/PyTorch 环境中执行。安装流程见 [项目首页](../README.md)。这些命令默认运行参考算子和 SDPA；启用自定义算子前检查其 dtype、shape 和 autocast 范围。

## 真实数据：TinyStories + 8K BPE

```bash
python -m tiny_transformer.prepare \
  --source tinystories --output data/tinystories-8k \
  --train-docs 100000 --val-docs 2000 \
  --tokenizer-docs 20000 --vocab-size 8192
```

脚本解析并记录数据集实际 revision，只在 train split 上训练 tokenizer，分开生成 train/val
token 文件，记录文本、token 文件和 tokenizer 的 SHA256。目标目录已存在时拒绝覆盖。
首次下载需要能访问 Hugging Face。历史数据准备记录为 99,983 篇训练故事（21,729,332 tokens）和 2,000 篇验证故事（385,234 tokens）；本次生成的数据以 manifest 为准。
已存在的数据目录无需再次运行 prepare；输出目录存在时脚本会拒绝覆盖。
使用前阅读数据集卡片和使用条款。

没有外网的 GPU 主机可以在联网机器执行上述步骤，再复制整个 `data/tinystories-8k` 目录。
也支持自备 JSONL（每行 `{"text":"..."}`）：

```bash
python -m tiny_transformer.prepare --source local \
  --train-jsonl /path/to/train.jsonl --val-jsonl /path/to/val.jsonl \
  --output data/local-8k --vocab-size 8192
```

## 60M 训练

```bash
# 算法参考基线，默认 eager attention。约 62.93M 参数。
python -m tiny_transformer.train --config configs/model_60m.json \
  --data data/tinystories-8k --output runs/60m-eager --device cuda

# 成熟库性能基线：相同配置，单独启用 SDPA。
python -m tiny_transformer.train --config configs/model_60m.json \
  --data data/tinystories-8k --output runs/60m-sdpa --device cuda \
  --op attention=sdpa

# FP32 对照：保持同一数据、seed、batch、seq_len；默认关闭 TF32。
python -m tiny_transformer.train --config configs/model_60m.json \
  --data data/tinystories-8k --output runs/60m-fp32 --device cuda --precision fp32

# 保持训练总 schedule 为 1000 steps，只执行前 20 steps。
python -m tiny_transformer.train --config configs/model_60m.json \
  --data data/tinystories-8k --output runs/resume-demo --device cuda --stop-after 20
python -m tiny_transformer.train --config configs/model_60m.json \
  --data data/tinystories-8k --output runs/resume-demo --device cuda \
  --resume runs/resume-demo/last.pt
```

默认每步处理 `8 × 512 × 4 = 16384` 个 input tokens；1000 steps 约 16.38M tokens。
采样是可复现的随机 token 窗口、有放回抽样；step 不等于 epoch。
100M tokens 可设置 `--steps 6104`。正式长训前先跑 100–200 步实测速度和显存。

日志：`metadata.json`、逐步 `metrics.jsonl`、原子写入的 `last.pt`。
训练含验证、梯度裁剪、AdamW、warmup/cosine、FP16 GradScaler 与 RNG/sampler 恢复。
只加载自己信任的 checkpoint；当前 checkpoint 含 optimizer 和 RNG 的 Python 对象。

## 推理、benchmark 与热点

```bash
python -m tiny_transformer.generate --checkpoint runs/60m-sdpa/last.pt \
  --device cuda --precision bf16 --prompt "Once upon a time" --max-new-tokens 128

python -m tiny_transformer.benchmarks.model --checkpoint runs/60m-sdpa/last.pt \
  --device cuda --precision bf16 --batch-size 1 --prompt-length 512 \
  --new-tokens 128 --repeats 10 --output runs/bench-b1.json

TORCH_LOGS="graph_breaks,recompiles" python -m tiny_transformer.benchmarks.model \
  --checkpoint runs/60m-sdpa/last.pt --device cuda --precision bf16 --compile \
  --prompt-length 512 --new-tokens 128 --output runs/bench-compile.json \
  2> runs/compile.log

python -m tiny_transformer.profile --checkpoint runs/60m-sdpa/last.pt \
  --device cuda --precision bf16 --phase decode --seq-len 512 --output runs/profile-decode
```

`--phase train/prefill/decode` 分别分析三类负载。生成 Chrome/Perfetto trace、算子时间表和前三个热点。
benchmark 报告冷请求、稳态 TTFT/TPOT、输出 token 吞吐、allocated/reserved 峰值显存、环境和原始 trial。
它是固定 batch 的模型基准，不是 HTTP 服务或动态调度器。
