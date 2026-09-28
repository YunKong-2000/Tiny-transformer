# CUTLASS / CuTe

`cutlass/` 是 NVIDIA 官方仓库的 Git 子模块，固定在 **v3.9.2**：
`ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`。
版本也记录在 `cutlass.json` 中，更新依赖时需与子模块一起更新。
CuTe C++ 随 CUTLASS 提供，无需另装 Python 包或编译 CUTLASS 静态库。

## 初始化

在项目根目录运行（新 clone 可直接使用 `git clone --recurse-submodules`）：

```bash
git submodule update --init --recursive third_party/cutlass
python -m tiny_transformer.check_cutlass --headers-only
```

在有 CUDA 版 PyTorch、CUDA toolkit/nvcc、C++17 编译器及 Ninja 的机器上验证真实编译与执行：

```bash
export TORCH_CUDA_ARCH_LIST=8.0  # A100；其他 GPU 按实际架构调整
export MAX_JOBS=2
python -m tiny_transformer.check_cutlass
```

该命令 JIT 编译 `csrc/cutlass_smoke/smoke.cu`，用 CuTe 构造 tensor/layout，
用 CUTLASS 执行加法，并校验 GPU 输出。`--headers-only` 只检查路径，不代表 CUDA 验收通过。
项目目标环境仍为 README 中的 NGC 25.08；CUTLASS 上游要求 C++17，支持范围以该固定版本文档为准。

## 在算子中使用

现有四个 JIT loader 已添加 `cutlass/include` 和 `cutlass/tools/util/include`，
并启用 C++17。可以直接在 `.cu` 中写：

```cpp
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cute/tensor.hpp>
```

新增算子的 loader 应使用相同配置：

```python
from tiny_transformer._cutlass import cutlass_include_paths

# torch.utils.cpp_extension.load(...,
#     extra_include_paths=cutlass_include_paths(),
#     extra_cflags=["-O3", "-std=c++17"],
#     extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo"])
```

如需使用已准备的其他源码目录，可在第一次编译前设置：

```bash
export CUTLASS_PATH=/absolute/path/to/cutlass
```

变量指向 CUTLASS 仓库根目录，不能指向 `include`；显式指定的路径无效时会报错。
替换版本后重启 Python 进程，必要时使用新的 `TORCH_EXTENSIONS_DIR` 避免复用旧编译缓存；
实验需记录实际使用的提交。Python import、CPU reference 路径不会检查依赖或触发编译/下载。

## 离线使用与许可

联网机器先初始化子模块，再运行 `scripts/build_offline_bundle.py`。
新离线包会包含 `include/`、`tools/util/include/`、`LICENSE.txt` 和版本记录，
目标服务器不需要访问 GitHub。离线包不含 CUTLASS 的示例、测试及 Git 历史；
旧离线包需要重新生成。构建 Docker 镜像前同样需要初始化子模块。

CUTLASS 使用 BSD-3-Clause 许可，原文保留在 `cutlass/LICENSE.txt`。
