# 离线服务器使用指南

本机负责下载原文、训练 8K BPE、编码 train/val token 文件；服务器直接使用离线包中的数据。
数据源为 `roneneldan/TinyStories`，固定 revision：
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`。
选取前 100,000 条 train 与前 2,000 条 validation，空文本过滤后数量以 `metadata.json` 为准。

## 1. 传输哪些文件

```text
dist/tiny-transformer-offline.tar.gz
dist/tiny-transformer-offline.tar.gz.sha256
```

压缩包包含项目代码、配置、文档、测试，以及：

```text
data/tinystories-8k/
  train.jsonl / val.jsonl       选定的原始故事，便于检查和重新编码
  train.bin / val.bin           编码后的 token，训练直接读取
  tokenizer.json               已训练好的 8K BPE
  metadata.json                数据版本、文档/token 数、hash
  DATASET_CARD.md               对应数据版本的官方说明
  LICENSE-CDLA-Sharing-1.0.txt   数据许可文本
  PREPARATION.json              数据准备环境与选择规则
wheelhouse/
  tokenizers-0.21.4-...manylinux...x86_64.whl
BUNDLE.json
CHECKSUMS.sha256
```

目标环境为 Linux x86_64 的 `nvcr.io/nvidia/pytorch:25.08-py3`，其 Python 版本为 3.12。
wheel 使用 CPython 3.9+ 的稳定 ABI，适用于该容器。这里没有把 macOS 虚拟环境拷给 Linux。

包内不含 NGC 镜像、NVIDIA 驱动、CUTLASS 源码或已训练模型权重。服务器需已具备目标容器与 GPU 运行环境。
后续开发学生 CUTLASS/CuTe kernel 时，需要另外准备并固定 CUTLASS 源码版本；参考框架不依赖它。

## 2. 在服务器校验并解压

将两个文件放在同一目录，执行：

```bash
sha256sum -c tiny-transformer-offline.tar.gz.sha256
tar -xzf tiny-transformer-offline.tar.gz
cd tiny-transformer-offline
```

如果容器尚未启动，在这个目录执行（镜像已经导入本机 Docker）：

```bash
docker run --rm -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/tiny-transformer \
  -w /workspace/tiny-transformer \
  nvcr.io/nvidia/pytorch:25.08-py3 bash
```

## 3. 容器内安装与验证：不访问网络

```bash
bash scripts/offline_setup.sh
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python3 -m unittest discover -s tests -v
```

setup 先校验每个打包文件，再从 `wheelhouse` 安装 tokenizer。
它使用 `--no-index --no-deps`，不会连接 PyPI，也不会替换镜像中的 torch、numpy 或其他依赖。
本项目仅调用 tokenizer 的本地 JSON 加载与 encode/decode；其联网下载辅助 API 未使用。
服务器不需要安装 `datasets`，也不必执行 `pip install -e '.[data]'`。
从项目根目录执行模块即可运行，不需要为了 editable install 再下载构建依赖。

## 4. 使用准备好的真实数据训练

```bash
# 保留完整 1000 steps 的 schedule，先运行 100 steps 验收。
python3 -m tiny_transformer.train \
  --config configs/model_60m.json \
  --data data/tinystories-8k \
  --output runs/60m-offline \
  --device cuda --op attention=sdpa --stop-after 100

# 接着训练，数据和 tokenizer 都从本地读取。
python3 -m tiny_transformer.train \
  --config configs/model_60m.json \
  --data data/tinystories-8k \
  --output runs/60m-offline \
  --device cuda --op attention=sdpa \
  --resume runs/60m-offline/last.pt
```

推理同样不联网，checkpoint 内保存 tokenizer 定义：

```bash
python3 -m tiny_transformer.generate \
  --checkpoint runs/60m-offline/last.pt \
  --device cuda --precision bf16 \
  --prompt "Once upon a time" --max-new-tokens 128
```

## 5. 本机如何重建数据和离线包

本机准备环境为 `.venv-data`，不会修改系统 PyTorch。
如网络需要代理，给命令设置本机实际的 `HTTPS_PROXY` / `HTTP_PROXY`，不要把代理配置带到离线服务器。

```bash
.venv-data/bin/python -m tiny_transformer.prepare \
  --source tinystories \
  --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --output data/tinystories-8k-new \
  --train-docs 100000 --val-docs 2000 \
  --tokenizer-docs 20000 --vocab-size 8192 --keep-text

.venv-data/bin/python -m pip download \
  --dest data/offline-wheelhouse --only-binary=:all: --no-deps \
  --platform manylinux2014_x86_64 --python-version 312 \
  --implementation cp --abi abi3 'tokenizers==0.21.4'

# 补入该版本的数据卡片和许可文件后，在目标输出不存在时打包。
python3 scripts/build_offline_bundle.py \
  --data data/tinystories-8k \
  --wheelhouse data/offline-wheelhouse \
  --output dist/tiny-transformer-offline.tar.gz
```

保留生成的 tokenizer 文件和 hash，才能重用完全相同的 token IDs。
重新训练 BPE 在词频并列时可能产生不同合并顺序，因此不要仅凭相同 vocab size 判断 tokenizer 相同。
`data/` 和 `dist/` 被版本控制忽略，迁移时要主动传输上述归档文件。

NGC Python 版本依据：[NVIDIA 25.08 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-08.html)。
