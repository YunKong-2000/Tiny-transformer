# flashAttention

当前接入 FP32 SIMT 和 BF16 前向，固定 `head_dim=64`。
FP32 和 BF16 通用 Tensor Core 路径每 CTA 128 线程；BF16 单 query 专用路径见下文。
两条路径均支持 causal prefill、decode/chunk 和序列尾块，Q/K/V 必须具有相同 dtype。
FP32 支持文档 segment IDs；host 将 Q/K 和 segment IDs 连续化，V 直接使用实际 stride。
BF16 要求 SM80+，不支持 segment IDs；host 将 Q/K/V 连续化，必要时 clone 保证起点
16-byte 对齐，kernel 本身不处理任意 stride。BF16 通用路径当前实验配置为 BQ=BK=BH=64，
Q 整块常驻 shared memory，另两个完整 K/V slot 共用 union 类型，共 24 KiB/CTA。
当前 slot 0 固定保存 K、slot 1 固定保存 V，使两者的预取和消费可以重叠。
score、softmax 统计量和输出累加器为 FP32，P 直接转换为 BF16 A fragment 留在寄存器。
原生接口返回 `(O, LSE)`：O 与输入同 dtype，LSE 始终为 FP32；Python 只返回 O。
反向、AMP/autocast 和 FP16 尚未实现；需要梯度时明确报错。
输入 batch、heads、序列长度必须为正，且 `Tk = past_len + Tq`。
FP32 二维 grid 要求 `B*H <= 65535`、`ceil(Tq/32) <= INT_MAX`；
BF16 通用路径一维 grid 要求 `B*H*ceil(Tq/64) <= INT_MAX`，乘法前检查溢出。
BF16 另要求 `Tk <= INT32_MAX`，host 在连续化/输出分配前检查，再将 Tq/Tk/past_len
传为 int32；完整元素地址偏移仍为 int64。
单 query 专用路径 grid 为 `(B*H, ceil(Tk/128))`，第二维最多 32，沿用上述边界检查。

2026-10-08 完整序列 tile 编译期特化（待 GPU 验收）：

- Host 在 `Tq % BQ == 0 && Tk % BK == 0` 时启动 `forward<true>`，其余通用
  BF16 输入启动 `forward<false>`；单 query decode 分派仍在此前处理。
- 完整 tile 的 Q/K/V 使用无 zero-fill 操作数的 `SM80_CP_ASYNC_CACHEGLOBAL`，
  去掉逐向量行边界判断和无效源地址选择。Q tile 终点直接为 `q0+BQ`。
- 完整 tile 的 mask 只检查 causal 条件，fully_valid 只检查最早 query 能否看到
  最后 key；输出无需 `qi<tq` 判断。尾块版本仍保留行边界检查和 zero-fill。
- 保持 BQ/BK/BH=64、24 KiB smem、launch_bounds(128,4)、PV 寄存器预取、
  base-2 softmax、分母延迟归约及直接 O pair stores。没有同时引入首轮 softmax 特化。

GPU 测试新增：同一组 query 的完整 tile 路径与追加一个 query/key 后的尾块路径
对照 O/LSE，并分别对照 FP64 和 SDPA；覆盖 cached causal 对角线。
NaN cache-capacity 用例补充 full/full、tail/full、full/tail，验证不能只检查一个长度。
本地 attention 测试 25 项：1 项 host 通过、24 项 CUDA 跳过。独立 CPU 索引检查覆盖
116 组长度、982 对抽样 tile（含 int32 上限），完整 tile 访存边界与 causal mask
等价性通过；不代表 CUDA 编译或运行验证。性能待与用户实测 42.38 us 基线比较。
编译日志/NCU 中现在应关注 `attention_bf16::forward<true>`；原有
`regex:attention_bf16::forward` 过滤器仍可匹配两个特化版本。

CUDA 验收（包含编译、FP64 oracle、tail/cache/segment/stream 检查）：

```bash
TORCH_CUDA_ARCH_LIST=8.0 python -m unittest discover -s tests -p 'test_student_attention.py' -v
compute-sanitizer --tool memcheck --error-exitcode 1 python -m unittest discover -s tests -p 'test_student_attention.py'
compute-sanitizer --tool racecheck --error-exitcode 1 python -m unittest discover -s tests -p 'test_student_attention.py'
compute-sanitizer --tool synccheck --error-exitcode 1 python -m unittest discover -s tests -p 'test_student_attention.py'
```

本地无 CUDA 时只能运行 Python 接线检查，GPU 用例会跳过；不能据此宣称 CUDA 已验证。

新增 BF16 测试直接调用生产 attention 扩展，比较 FP64 oracle 与 BF16 SDPA，
覆盖多轮 stage 复用、尾块、cache 前缀、非连续/未对齐 view、精确常数输出、
大 logits、因果隔离、非默认 stream、混合 dtype、segment 拒绝和 grid 上限。
2026-10-07 BF16 修复后的 student 回归：87 项（11 项通过、76 项 CUDA 跳过）。
Clang 对输入检查头文件和绑定的 C++17 语法检查通过；此检查不包含 CUDA kernel。
尚未运行 nvcc、GPU 数值测试或 compute-sanitizer。

BF16 性能排查：host 使用 `at::cuda::getDeviceProperties` 的设备缓存，避免每次调用
查询 CUDA runtime。每次复用某个 slot 前，必须完成该 slot 的旧数据消费和
CTA barrier；另一个 slot 可以仍在计算。当前完整 QK 计算时预取 V，完整 PV 计算时
预取下一 KV tile 的 K。同步改动仍需重跑数值测试和
compute-sanitizer racecheck/synccheck 验证。
性能入口新增 `--attention-timing graph --phases forward`，完整算子图重放与原有
operator 耗时分开报告，具体口径见 benchmarks README。新增 GPU 测试检查未对齐
prefill/decode 输入在 CUDA Graph 中捕获和重复重放的 O/LSE。

后续 GPU 反馈：Graph 下原实现 prefill 100.42 us、decode 34.06 us，对应 SDPA
39.37 us、15.42 us，说明 host 提交不是 student 的主要开销。当前改动针对 GPU：

- Q_i 的完整 `(64,64)` tile 保存在 shared memory，共 8 KiB。每次 QK 只加载
  当前 MMA 的 K=16 Q fragment 到寄存器，避免 Q 的完整寄存器 fragment 跨循环存活。
- K/V 当前均为 `(BK,BH)=(64,64)`，每个 8 KiB，两个 slot 共 16 KiB；加上 Q
  总计 24 KiB，没有 shared P。Q/K/V 的 64-BF16 行都使用 `Swizzle<3,3,3>`，
  将元素地址第 6/7/8 位异或到第 3/4/5 位，保留 16-byte 向量对齐。
- 每轮 KV tile 直接消费完整 Q/K，MMA 内部仍按 K=16 遍历 DH；
  完成全部 DH 后才做 mask/softmax。每线程 S 为 `(4,1,8)`，32 个 FP32。
- softmax 直接将 `rS(vi,0,n)` 对应的概率写到
  `rP(vi+4*(n%2),0,n/2)`，编译期断言校验 atom 的 A/C 布局关系；转换不跨线程。
  每线程 P 为 `(8,1,4)`，32 个 BF16。FP32 分母在转换前计算；进入 PV 前 S 结束使用。
- 先对完整 O 的 64 列缩放一次 alpha，再直接计算 `O += P @ V`；
  PV 内部按 K=16 归约 BK，不再切分输出特征。O 仍为 `(4,1,8)`，32 个 FP32。
- 完整 V 的加载与 QK/softmax 重叠，下一 tile 的完整 K 与当前 PV 重叠。
  每个 CTA 消费 N 个 KV tile 时共 2N 次 barrier（含 prologue），输出直接写回。
- 通用 softmax 与已保存的 int32/base-2 基线一致，使用 `exp2f` 和 FMA，LSE
  写回时转换为自然对数 FP32。该轮实验只改变 K/V 宽度、相应 swizzle 和流水线。
- `Tq=1 && Tk<=4096` 使用专用 SIMT split-KV 路径，每 CTA 256 线程处理 128 个 key。
  QK、P、PV 使用 FP32；单块直接输出，多块写 FP32 `(numerator, max, sum)` 并由第二个
  kernel 按全局 max 重缩放后合并。无需补齐 64 行 query，Tk>4096 仍走通用路径。
  例如 B=8,H=12,Tk=512 时，partial 阶段 384 个 CTA，另有 96 个 merge CTA；
  FP32 中间缓冲为 101376 字节，分配/第二次启动的成本仍包含在 operator benchmark 中。

本地 CPU 检查已覆盖 2048 个 P 元素的 C→A 映射、K/V swizzle 与转置的地址一致性，
以及特征分块的 online softmax 数学模型（随机 tail/chunk 和 8 个独立特征探针），均通过。
benchmark 主机测试 29 项通过；attention 测试 1 项通过、15 项因无 CUDA 跳过。
这些检查不替代 CUDA 编译、数值或竞态验证。
新增 GPU 测试覆盖 127/128/129、255/256/257、4095/4096/4097 的 decode 分派边界，
不同分块最大值、精确常数输出及非默认 stream。通用路径增加 31/32/33 的 BK 尾块，
以及 QK 特征 15/16/31/32/47/48/63 的独立探针；随 key 增大的 score 检查 alpha
是否覆盖 O 的全部列。仍需检查 ptxas register/spill 报告和 Nsight Compute；
当前没有此版本 GPU 耗时结论。

NCU 后续反馈包含 61440 次 local-memory spilling requests、global-store 每 sector
约 8.19/32 bytes 有效数据，以及 shared-load bank-conflict wavefront 占比 43.60%。
已将 K/V swizzle 从 `<2,3,2>` 修为 `<2,3,3>`：8x8 物理子矩阵的 8 个行起始 bank
由 `[0,20,8,28,0,20,8,28]` 变为 `[0,16,4,20,8,24,12,28]`，CPU 地址检查通过。
输出将同线程相邻两个 BF16 的原始位打包为一个对齐 uint32 store，减少零散的 16-bit 写入；
这仍不是完整的 epilogue 数据重排。以上改动需 GPU 数值与 NCU 复测，不能视作 spill 已修复。
spill 的来源需要结合 Source Counters/SASS 和 ptxas 输出定位；新增可选编译诊断：

```bash
TORCH_CUDA_ARCH_LIST=8.0 TINY_TRANSFORMER_CUDA_VERBOSE=1 \
python -u -m unittest discover -s tests -p 'test_student_attention.py' -k bf16 -v
```

该环境变量仅对 attention 扩展启用 verbose 和 `--ptxas-options=-v,--warn-on-spills`，
输出每个 kernel 的 registers、stack frame、spill loads/stores。开关会改变构建参数，
切换时可能重新编译。应保留 forward kernel 的报告与 NCU Source Counters 对照，
不要仅凭 local-memory 流量断言某个 C++ fragment 一定发生了寄存器容量溢出。

2026-10-08 根据新版 NCU（BF16 约 120 registers/thread、零 spill、理论 occupancy 25%）
实现以下待 GPU 验证的优化；日志里的 96 registers 属于 FP32 SIMT kernel：

- QK/PV 的 B 加载粒度缩到一个 `N=8,K=16` atom，每线程 rB 从 16 个 BF16 减为
  4 个。QK 的 8 个 BF16 A 元素跨 4 个 N atom 复用，P 继续保存在寄存器。
  用编译期 ki/ni 索引切片 C；不强制 maxrregcount/最小驻留块数，避免人为引入 spill。
  C++ 作用域缩小不保证编译器最终减少寄存器，必须复查 ptxas 和独立性能数据。
- Q 和首个 K[0] 一起 commit/wait。流水线为 K0(0)→K1(1)→V0(0)→V1(1)→
  下一 K0(0)，括号为 stage。每个阶段在另一 stage 的旧读者已完成 barrier 后发起预取。
  K[1] 结束后的 barrier 合并到 V[0] ready barrier；下一 tile 的 K[0] ready barrier
  合并到前一 tile 的 PV 末尾。每轮通常 4 次 CTA barrier，最后一轮 3 次；
  含 Q prologue 总计 `4*kv_tiles` 次，之前为 `1+6*kv_tiles` 次，共享内存仍为 12 KiB。
- 一维 grid 的 block 编号按 query tile 降序、同一 tile 内遍历 batch/head：
  `bh = blockIdx.x % batch_heads`，`q_tile = q_tiles-1-blockIdx.x/batch_heads`。
  每对 `(bh,q_tile)` 仍恰好一个 CTA，只改变枚举顺序；CUDA 不保证实际调度顺序。
  长 CTA 靠前可能改善 causal 尾效应，也可能影响 cache locality，收益须实测。
- 新增不同 head 常数偏移及随 token 变化的 V 测试，覆盖 query 尾块、chunk 和
  各 pipeline 阶段切换，检查 CTA 重排后没有漏写或写错 head。

本地验证：CTA 映射的 28 组 head/长度组合覆盖检查、N=8 拆分 GEMM 数学检查、
1/2/3/16 个 KV tile 的 stage 发布/释放状态模型均通过。attention 测试 1 项通过、
16 项因缺少 CUDA 跳过；没有执行 nvcc、GPU 数值/竞态检查或性能测试。

2026-10-08 编译反馈：上述三个改动合并后的 BF16 forward 使用 128 registers，
16-byte stack frame，20-byte spill stores、28-byte spill loads；原版本约 120 registers
且零 spill。显式缩小 rB 并没有证明能缩小最终分配，三个改动的独立影响尚未测量。
当前先撤回 QK/PV 的手写 N=8 内层展开，恢复完整 N fragment 的 CuTe gemm；
保留跨阶段预取和 CTA 重排，以单独观察这次撤回的编译结果。
不添加 maxrregcount 或更强的 launch_bounds，当前代码不能宣称 spill 已消失。
如果仍有 spill，应进一步单独比较预取与 CTA 重排，或恢复原先的零-spill版本作为基线。

2026-10-08 完整合法 KV tile 的 mask fast path：只将 scale/mask 循环抽成
`scale_mask_scores<FULL_TILE>`，online softmax、PV、tile 大小和流水线共用。
CTA 内统一分支的条件是 query tile 完整、key tile 完整，且第一行 query 已可见
该 key tile 的最后一个 key：`q_end-q0==BQ && tk-key0>=BK &&
past_len+q0-key0>=BK-1`。完整块仅乘 SCALE，编译期移除每元素的坐标与 mask 判断；
其余块使用原先的尾部/causal 检查。采用减法避免构造可能越界的 padded key 端点。
测试新增 Tq=63/64/65/128、past=0/30/31/32/33，在 key 31/63 放入脉冲 V，
对照 FP64/SDPA 并检查改变未来 V 不影响首行，以覆盖 inclusive causal 边界和路径切换。
该改动基于当前工作区（已撤回 N=8 手写展开），没有恢复该展开或改变 BK/BH。
未执行 CUDA 编译或计时，寄存器和性能影响仍需单独比较。

2026-10-08 下一步单变量实验：在 mask fast path 的基础上只将 BK 从 32 改成 64，
BH=32、BQ=64、128 线程、swizzle、union 预取、CTA 排列和 decode 分派保持原配置。
基线用户报告为 operator prefill：SDPA 41.11 us、student 62.86 us。

| 项目 | BK=32 基线 | BK=64 候选 |
|---|---:|---:|
| shared memory / CTA | 12 KiB | 16 KiB |
| 每线程 FP32 S 元素 | 16 | 32 |
| 每线程 BF16 P 元素 | 16 | 32 |
| PV 的 K=16 归约迭代数 / 输出 subtile | 2 | 4 |
| T=512 每 head 的 KV tile 总数 | 72 | 36 |
| 其中完整合法 / 边界 tile | 56 / 16 | 28 / 8 |
| T=512 每 head 的 CTA barrier 总次数 | 288 | 144 |

每个输出元素的 softmax 合并次数减少，QK/PV 总 MMA 数在这个整齐形状下保持相同。
BF16 P 的分块舍入可能变化，需重新通过精度检查；寄存器可能增加甚至引入更多 spill，
不预设新配置更快。新增 BK 边界、past=62/63/64/65、key 32–63 的 PV 归约探针，
并覆盖完整 512 prefill。有效性以 GPU 正确性、ptxas 和无 NCU 的同条件 benchmark 为准。

此轮本地检查：4096 个 P 坐标、2048 个 K/V 地址、16-byte 对齐、8x8 bank 分组、
753 个 mask tile 分类通过；BK=64 的分块数学模型在 10 个随机形状和 13 个 key 探针下
通过 FP64 对照。benchmark 主机测试 29 项通过；attention 测试 1 项通过、18 项
CUDA 跳过。未运行 nvcc、GPU 数值/竞态或性能测试。

2026-10-08 实现 base-2 online softmax，并恢复 `__launch_bounds__(THREADS, 4)`。
比较基线应使用用户测得的 minBlocks=4 版本（operator 52.03 us、graph 50.61 us），
而不是 minBlocks=5 的 graph 53.16 us。仅恢复驻留目标并修改通用 softmax，
没有修改 tile、预取、CTA 顺序、索引宽度或 decode 的数值域。

当前 `rS` 始终保存未缩放的 QK 分数；完整合法块不再执行逐元素 scale/mask 循环，
边界块只将非法位置写为 -inf。正比例因子允许先对 raw score 求 max，随后每行计算：

```text
scale2 = (1/sqrt(DH)) * log2(e)
m2_new = max(m2_old, rowmax(raw_score) * scale2)
alpha = m2_old == -inf ? 0 : exp2(m2_old - m2_new)
p = raw_score == -inf ? 0 : exp2(fma(raw_score, scale2, -m2_new))
l = alpha*l + rowsum(p)
O_acc = alpha*O_acc + BF16(p) @ V
LSE = fma(m2, ln(2), ln(l))  // 自然对数接口
```

全 mask 的 padded query 行在指数运算前屏蔽 -inf，避免 -inf-(-inf) 导致 NaN。
FMA 和 base-2 最大值的舍入与旧实现不同，最大概率可能因舍入略大于 1，不能假设
指数参数在浮点计算中严格非正；沿用原 O/LSE 容差做验证，不放宽阈值。
实现参考 Triton 官方 v3.4.0 `python/tutorials/06-fused-attention.py` 的
`_attn_fwd_inner`（raw max / scale / exp2）以及 CUTLASS example 41 的
`iterative_softmax`（base-2 最大值）；教程的内部 log2 LSE 不能直接作为本接口输出。

新增 GPU 用例用正负非零常数 logits 验证自然对数 LSE、精确常数 V 输出，包含
cache chunk、full/masked tile 切换及 query padding。CPU 分块模型使用 FP64
乘加后舍入 FP32 模拟 FMA，49 个完整 tile 和 60 个 mask tile 通过 FP64 对照，
最大 O 绝对误差 0.006991，最大 LSE 绝对误差 0.00006104（均在原阈值内）。
attention 主机测试 1 项通过、19 项 CUDA 跳过。未运行 nvcc、CUDA exp2f 精度检查、
GPU racecheck 或性能测试，寄存器分配与加速效果需由目标 A100 验证。

CUDA 编译修复：两处 base-2 指数改用 CUDA device math 的 `exp2f`，与 CUTLASS
example 41 的调用一致。此前误用了 `__exp2f`，目标工具链将其识别为 host 函数，
不能在 device code 调用。保留 base-2 最大值、FMA、自然对数 LSE 和 launch bounds；
不添加全局 fast-math 开关。具体指令和性能仍以 nvcc/SASS 和 A100 测量为准。

2026-10-08 索引宽度实验：通用 BF16 forward 的 Tq/Tk/past_len 参数、CTA 映射、
query/key/tile 循环变量及 mask 比较改为 int32。保持当前 base-2 softmax、BK=64、
BH=32、launch_bounds(128,4) 和流水线，以用户 base-2 Graph 51.11 us 为直接对照。
未同时退回自然指数，避免将两项改动的收益混在一起。FP32 和专用 decode kernel 未改。

安全边界：由 host 的 `Tk=past_len+Tq`、正尺寸及 `Tk<=INT32_MAX` 推导 Tq/past_len
也可安全缩窄；grid.x 的原有 INT_MAX 检查保证 block 编号、batch_heads 可转为 int32。
固定 BQ=BK=64 下，末块的 padded row/key 最大为 INT32_MAX；query 结束位置使用
`q0 + min(tq-q0,BQ)`，避免 `q0+BQ` 超界。causal 条件使用 `kj-past_len<=qi`，
避免在 padded query 行求 `past_len+qi` 时溢出。下一 key tile 起点只在确实有下一块时求值。
global load/store 的 `row*DH`、batch/head base 偏移在乘法前显式提升为 int64，
所以单 head 或整个 tensor 的元素/字节偏移不受 int32 限制。

新增 CUDA 测试用零 stride view 检查超长序列在复制前被拒绝，无需分配巨大输入。
本地另用 Clang UBSan 编译 kernel 中提取的真实索引表达式，对 INT32_MAX 附近的
尾块、非零 past_len、CTA 映射与原 int64 mask 语义进行对照。该检查不替代 nvcc、
CUDA 数值/竞态或性能测试；attention 主机测试 1 项通过、20 项 CUDA 跳过。

2026-10-08 保存的基线与当前 full-width KV 实验：

- 48.71 us 基线保存在 Git 提交 `b1482b45222875dcff2b893086a52dcf38418a2c`。
  本地快照目录 `runs/attention-tuning/baseline-int32-base2-b1482b4/` 包含源码和
  `manifest.json`（SHA256、commit、用户报告的性能及 ptxas 数据）。runs 被 gitignore
  排除，跨机器恢复以 Git 提交为准。基线为 A100 80GB PCIe、BF16、B=8,H=12,
  T=512,D=64、Graph、warmup=20,repeats=100,trials=5；student 48.71 us、SDPA
  39.23 us，128 registers、8-byte stack、4-byte spill stores/loads。
- 自然指数候选未经 GPU 测量已撤回。当前候选从上述快照恢复相同 base-2 数学后，
  仅将 BH=32 增大为 BH=64，BK/BQ=64、int32 索引及 launch_bounds(128,4) 保持不变。
- 共享内存 16→24 KiB，S/P/O 的逻辑寄存器元素数不变；QK 仍逐 K=16 加载 Q 片段，
  未将整个 Q 常驻寄存器。PV 当前 B fragment 由每线程 16 个 BF16 增大到 32 个，
  所以寄存器分配/spill 必须重测，不能仅凭 barrier 减少预设加速。
- 流水线：prologue 加载 Q 与 K(slot 0)；每轮预取 V(slot 1)，执行 QK/softmax，
  wait+CTA barrier 发布 V 并释放 K；若有下一轮，预取下一 K(slot 0)，执行当前 PV，
  wait+CTA barrier 发布 K 并释放 V。最后一轮 PV 后无需同步，无未完成 cp.async。
  原来的两个特征循环被移除。每 CTA 的 barrier 为 `1+N+(N-1)=2N`，基线为 4N。
  在 T=512 下，每 head 的 KV tile 仍为 36，barrier 总次数由 144 降至 72。
- GPU 测试增加单 head 的连续 cache-prefix view，在有效 Tk 后放入 NaN，覆盖
  Tk=63/64/65、query tail、多轮缓冲复用、单轮收尾。该 view 不会被 host 连续化
  复制，能检查原 allocation 内的越界尾部读取。沿用已有 FP64/SDPA、特征探针、
  非默认 stream 和 CUDA Graph 测试。

候选需先通过正确性和 racecheck/synccheck，再用相同 Graph 参数对照 48.71 us。
建议输出到 `runs/attention-bh64-graph.json`，避免覆盖基线报告。若需恢复基线 kernel：

```bash
git restore --source=b1482b45222875dcff2b893086a52dcf38418a2c -- csrc/attention/attention_forward_BF16_kernel.cuh
```

该命令会覆盖 kernel 的工作区修改；已有基线快照和 Git 提交均保留。

本地检查：保存文件的 SHA256 和 softmax/指数常量与基线一致；full-width swizzle
的 4096 个地址、16-byte 向量、8x8 物理子矩阵 bank 分组通过。1/2/3/8/16 个 KV
tile 的 slot 状态模型确认每次覆盖前释放读者、收尾无 pending copy、barrier 为 2N。
12 组随机 BF16 输入的 QK/PV 分组数学模型与原 BH=32 版本一致。attention 测试
1 项通过、21 项 CUDA 跳过；未完成 nvcc、GPU 数值/竞态检查或性能测试。

2026-10-07 本地验证：student 回归 82 项（11 项通过、71 项 CUDA 跳过），
benchmark 回归 25 项通过，operator 回归 10 项通过；Clang 对绑定和输入检查头文件的
C++17 语法检查通过。未运行 nvcc、GPU 数值测试或 compute-sanitizer。

## 整体算法 overall algorithm

2026-10-08 PV shared→register 预取实验：基线提交
`8732172b52eaf63f9ca7a0541d86cb727bec908e`，用户报告 student 42.34 us、SDPA 39.27 us。
本轮只改 `gemm_pv_subtile`：保留两个独立的 V register fragment rB0/rB1，
ki 和 slot 选择通过 `make_seq`/`if constexpr` 编译期展开，不使用动态索引数组。

```text
load V[0] → rB0
load V[1] → rB1; MMA(P[0], rB0)
load V[2] → rB0; MMA(P[1], rB1)
load V[3] → rB1; MMA(P[2], rB0)
                 MMA(P[3], rB1)
```

每个片段覆盖 K=16，每线程 32 个 BF16；只有前一内容已被 MMA 消费的 slot 才可复用。
最后一轮不预取，加载与 MMA 各执行 4 次，累加顺序与基线相同。共享内存仍为 24 KiB，
未增添 CTA barrier，也未改 QK、softmax、global→shared 流水线或 epilogue。
两个片段增加显式存活的 V 值；编译器可能进一步调整加载次序与寄存器分配，源码预取
不保证硬件上已经发生有效重叠，必须对照 ptxas、SASS 和同条件 Graph benchmark。

本地检查确认 kernel 除 PV helper 外与基线逐字相同；寄存器 slot 状态模型无提前覆盖、
无越界预取，20 组随机 BF16 操作数的 PV 数学模型与串行 K=16 累加逐位一致。
GPU key 探针扩展到两个完整 KV tile 中每个 K=16 fragment 的首尾，以及最后一个 key
的 tail。attention 测试 1 项通过、23 项 CUDA 跳过；未执行 nvcc、GPU 数值/竞态
或性能验收。候选只有在正确性通过并稳定优于 42.34 us 基线时才值得保留。

2026-10-08 输出重排实验（已撤回）：复用 `storage.q` 的 8 KiB，先由 MMA 原线程将归一化的
BF16 pair 写到 QLayout 对应位置，再执行一次全 CTA barrier。随后按每行 8 个线程、
每线程 8 个 BF16 重新读取，使用对齐的 uint4 向 global memory 写回。共享内存仍为
24 KiB，没有额外分配；原来的逐元素除法和 LSE 计算保留，只测试输出访问布局变化。
最后一次 QK→PV barrier 保证所有 Q 读取已经结束，因此写 Q 缓冲之前无需新增 barrier；
输出写入与 vector 读取的线程分工不同，中间 barrier 必须由包括 padded query 线程在内
的所有线程执行。global vector store 只对有效 query 行执行。

本地覆盖 1/2/7/16/31/32/33/63/64 个有效行的逐位重排、唯一覆盖、4/16-byte 对齐和
完整 32-byte sector 检查，均通过；新增 GPU 精确输出用例使用各 head/feature 不同的
常量 V，覆盖 2/63/64/65/129 行。attention 测试 1 项通过、22 项 CUDA 跳过。
尚未运行 nvcc、racecheck/synccheck 或性能测试，新增 shared 往返及 barrier 的代价
必须与更好的 global store 合并效果一起测量。分母延迟归约和每行倒数是另两项独立建议，
本轮没有同时实现。

用户实测该输出重排版本 student 48.17 us、SDPA 39.27 us；直接写回版本 student
46.73 us、SDPA 39.35 us，本次对照约慢 3.1%。撤回该 epilogue 时，kernel
逐字节恢复为 `e13905e` 的直接 uint32 pair 写回版本。失败实验完整保存在提交
`f14345a`，不同 head/feature 和 query tail 的精确输出测试继续保留。
这次数据表明更好的 global-store sector 利用率没有带来净收益；新增 shared 往返、
barrier 及编译调度变化的具体占比尚未单独量化。后续实验以 46.73 us 版本为基线。

2026-10-08 在直接写回基线上实现分母延迟归约与倒数复用：

- 每线程仍保存两个 FP32 分母状态，但语义改为本 lane 的累计部分和：
  `l_partial[a] = alpha*l_partial[a] + local_sum`。行最大值仍每轮跨四个 lane 归约，
  同一行的 alpha 因而一致。local_sum 是 P 转 BF16 前的 FP32 指数和。
- Tk 循环后才调用 `row_sum(l_partial[a])`，取得完整 denominator；此 shuffle 在
  `qi<tq` 判断之前执行，包括 padded 行的 lane，满足 full-mask warp 参与要求。
  遍历 N 个 KV tile 的 CTA，分母相关 shuffle 由每线程 4N 次降为 4 次。
- 最终对每线程负责的每行计算 `inv_l=1/denominator`，该行的所有输出元素都乘该
  倒数；源码中的除法从每线程最多 32 次降为 2 次。LSE 使用完整 denominator，
  仍输出 `fma(m2,ln(2),log(denominator))`。保留 denominator<=0 的零输出/-inf LSE。
- 没有引入 shared O 或额外 CTA barrier，维持 24 KiB shared memory、原来的 Q/K/V
  预取、tile、int32 索引、base-2 指数和 launch_bounds(128,4)。

新增 GPU 用例使同一行四个 lane 的权重明显不同，KV 各块的最大值交替上升/下降，
并覆盖 2/63/65/129 个 query 的尾块，防止丢失历史部分和或漏乘 alpha。
本地 CPU 分块模型对 40 个 query tile 通过 FP64 对照，最大 O 误差 0.006991、
LSE 误差 0.000001908；新旧分母因求和顺序变化的最大差值 0.000007629。
attention 测试 1 项通过、23 项因无 CUDA 跳过；没有执行 nvcc、GPU 精度/竞态或
性能测试。收益以与 46.73 us 基线的同条件 Graph benchmark 对照为准。

$$
\mathbf{Q}_i \in \mathbb{R}^{B_q \times D_n} \\ 
\mathbf{K}_{jd} \in \mathbb{R}^{B_k \times B_d} \\
\mathbf{m}_{new}, \mathbf{m}_{old}, \mathbf{\alpha} \in \mathbb{R}^{B_q} \\
\mathbf{l} \in \mathbb{R}^{B_q} \\
\mathbf{S}_{ij} \in \mathbb{R}^{B_q \times B_k} \\
\mathbf{P}_{ij} \in \mathbb{R}^{B_q \times B_k} \\
\mathbf{V}_{jn} \in \mathbb{R}^{B_k \times B_n} \\
\mathbf{N}_i \in \mathbb{R}^{B_q \times D_n} \\ 
\mathbf{O}_i \in \mathbb{R}^{B_q \times D_n} \\ 
$$


````markdown
load Q_i
for j in (1, T_k / B_k):
    for d in (1, D_n / B_d):
        load K_jd
        S_ij += Q_id * K_jd^T
    m_new = max(rowmax(S_ij), m_old)
    P_ij = exp(S_ij - m_new)
    alpha = exp(m_old - m_new)
    l = sum(P_ij) + alpha * l
    for k in (1, D_n / B_n):
        load V_jk
        N_ik = alpha * N_ik + P_ij @ V_jk
    m_old = m_new
O_i = N_i / l
store O_i, LSE_i=log(l)+m_old
````

## 核函数实现 implementation of kernel
### 并行策略
使用一个CTA对应一个O矩阵的$B_q$行的最终结果,也就是$\mathbf{O}_i \in \mathbb{R}^{B_q \times D_n}$，CTA中的不同warp负责不同行的计算，通常是一个warp对应多行，这样可以实现m和l的行内归约。
所以最终的CTA总数是$B \times N_h \times (T_q / B_q)$。实际启动时，可令grid shape为$(B*N_h, T_q / B_q)$。
### 共享内存分配
每个CTA在kernel开始时分配$Q_i$对应的shared memory，大小为$B_q \times D_h \times bytes$。BF16 路径每个 stage 的 K_tile/V_tile 通过 union 共享 $B_k \times B_h \times bytes$ 空间，两个 stage 交替预取；其余中间量 S、P 和输出累加器存储在寄存器中。
kernel开始时直接从global memory读取$Q_i$至shared memory。  
### 循环设计
然后进入streaming loop,这个循环的循环变量j在KV的序列方向$T_k$,循环次数是$T_k/B_k$。  
然后进入第一个归约循环，这个循环的循环变量d在K矩阵的特征方向进行内积累加，循环次数是$D_n/B_d$。每次循环中都会从GMEM中读取一个K_tile，也即是$K_{jd}:(B_k, B_d)$，大小为$B_k \times B_d$。然后和$Q_i$中对应的一段$Q_{id}:(B_q, B_d)$进行矩阵乘法。并将其结果累加至$S_{ij}:(B_q, B_k)$。
第一个归约循环结束后就得到了完整的$S_{ij}:(B_q, B_k)$，他的值将会暂时存放在寄存器中。然后可以直接进行scale和mask。  
然后开始更新行最大值和指数和，并且计算$P_{ij}:(B_q, B_k)$,$S_{ij}$和$P_{ij}$复用了寄存器中的空间。
接下来遍历输出特征子块，每次从 GMEM 读取 $V_{jn}:(B_k,B_n)$。
PV 真正的归约维是 key token：BF16 MMA 按 K=16 准备 P/V 寄存器片段，遍历 BK，累加到同一个输出子块。
代码在 softmax 阶段先对完整的输出累加器乘一次 alpha，再计算各子块的 PV，不能重复缩放。
完整 streaming 循环后，将输出累加器逐行除以 l 得到 O，并写回 GMEM；LSE=m+log(l) 单独保存。
### 流水线
每个 KV tile 先进行 K 特征循环，再进行 V 输出特征循环。每个循环都在计算当前 stage 时预取下一 subtile；完成异步等待和 CTA barrier 后交换 stage。K/V 阶段切换前完成所有旧 stage 的读取，再复用 union。Q 始终保留在独立的 shared memory 中。

### swizzle技术
本质上Swizzle操作是对元素偏移offset的二进制数的分析和操作。数据类型决定了offset对应的bank的起始位置，例如fp32的起始位就是0，因为一个bank恰好对应一个元素，而对于fp16或者bf16起始位就是1，因为一个bank对应两个数，如果有int8,fp8这样的类型，起始位就是2，因为一个bank对应8个元素;而读取数据的段长决定了最终的保留位，段长以元素个数位单位，之前的例子是一个线程读连续4个fp32,那就保留最低两位，线的例子事以8个bf16位一段，那就保留最低8位；其次就是同一次访问指令中同一bank的不同地址的stride，在这个例子中每隔64个元素就会落入到同一个bank,所以就是发生bank conflict的两个地址第6位以上才不同，这个数字恰好也是bank的起始位加上5，也就是bank的终止位的下一位；最后需要决定低位中哪些位需要被修改，很明显，最低的保留位是不能动的，因此需要从保留位下一位开始，根据bank conflict的way的数量决定，比如是8way，就需要log(8)=3位做修改，这样才能得到8个不同的新目标地址。
