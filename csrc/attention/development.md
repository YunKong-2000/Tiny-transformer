# flashAttention

当前接入 FP32 SIMT 和 BF16 前向，固定 `head_dim=64`。
FP32 和 BF16 通用 Tensor Core 路径每 CTA 128 线程；BF16 单 query 专用路径见下文。
两条路径均支持 causal prefill、decode/chunk 和序列尾块，Q/K/V 必须具有相同 dtype。
FP32 支持文档 segment IDs；host 将 Q/K 和 segment IDs 连续化，V 直接使用实际 stride。
BF16 要求 SM80+，不支持 segment IDs；host 将 Q/K/V 连续化，必要时 clone 保证起点
16-byte 对齐，kernel 本身不处理任意 stride。BF16 通用路径当前实验配置为 BQ=BK=64、BH=32，
Q 整块常驻 shared memory，两个 K/V union stage 保存特征 subtile，共 16 KiB/CTA。
score、softmax 统计量和输出累加器为 FP32，P 直接转换为 BF16 A fragment 留在寄存器。
原生接口返回 `(O, LSE)`：O 与输入同 dtype，LSE 始终为 FP32；Python 只返回 O。
反向、AMP/autocast 和 FP16 尚未实现；需要梯度时明确报错。
输入 batch、heads、序列长度必须为正，且 `Tk = past_len + Tq`。
FP32 二维 grid 要求 `B*H <= 65535`、`ceil(Tq/32) <= INT_MAX`；
BF16 通用路径一维 grid 要求 `B*H*ceil(Tq/64) <= INT_MAX`，乘法前检查溢出。
单 query 专用路径 grid 为 `(B*H, ceil(Tk/128))`，第二维最多 32，沿用上述边界检查。

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
查询 CUDA runtime。每次复用某个 union stage 前，必须完成该 stage 的旧数据消费和
CTA barrier；另一个 stage 可以仍在计算。K[1] 计算时预取 V[0]，V[1] 计算时预取
下一 KV tile 的 K[0]。同步改动仍需重跑数值测试和
compute-sanitizer racecheck/synccheck 验证。
性能入口新增 `--attention-timing graph --phases forward`，完整算子图重放与原有
operator 耗时分开报告，具体口径见 benchmarks README。新增 GPU 测试检查未对齐
prefill/decode 输入在 CUDA Graph 中捕获和重复重放的 O/LSE。

后续 GPU 反馈：Graph 下原实现 prefill 100.42 us、decode 34.06 us，对应 SDPA
39.37 us、15.42 us，说明 host 提交不是 student 的主要开销。当前改动针对 GPU：

- Q_i 的完整 `(64,64)` tile 保存在 shared memory，共 8 KiB。每次 QK 只加载
  当前 MMA 的 K=16 Q fragment 到寄存器，避免 Q 的完整寄存器 fragment 跨循环存活。
- K/V 的 subtile 当前均为 `(BK,BH)=(64,32)`，每个 4 KiB；每个 stage 使用 union 复用
  K/V 存储，两个 stage 共 8 KiB。共享内存总量为 `8+2*4=16 KiB`，没有 shared P。
  Q 使用 `Swizzle<3,3,3>`；K/V 的行宽为 32，使用 `Swizzle<2,3,3>`。
  32 个 BF16 的行跨越 16 个 bank，行低位已经选择 bank 的前/后半；swizzle 应使用
  行的第 1/2 位，即元素地址第 6/7 位。旧的 shift=2 会使相隔 4 行的片段重复占用 banks。
- 每轮 KV tile 中，先遍历 Q/K 特征 `d=0,32`，将两个部分点积累加到同一个 S；
  完成全部 DH 后才做 scale/mask/softmax。每线程 S 为 `(4,1,8)`，32 个 FP32。
- softmax 直接将 `rS(vi,0,n)` 对应的概率写到
  `rP(vi+4*(n%2),0,n/2)`，编译期断言校验 atom 的 A/C 布局关系；转换不跨线程。
  每线程 P 为 `(8,1,4)`，32 个 BF16。FP32 分母在转换前计算；进入 PV 前 S 结束使用。
- 先对完整 O 的 64 列缩放一次 alpha，再遍历输出特征 `h=0,32`，
  计算 `O[:,h:h+BH] += P @ V[:,h:h+BH]`；PV 的归约维为 BK，不能误用 BH/DH。
  O 仍为 `(4,1,8)`，32 个 FP32。输出切片索引在编译期确定，避免动态索引寄存器数组。
- 两个 union stage 交替预取，跨越 QK/PV 阶段与相邻 KV tile；V[0] 的加载与 K[1]
  的计算及 softmax 重叠，下一 tile 的 K[0] 与当前 V[1] 的计算重叠。
  该方案减少 shared memory 和 fragment 大小，但增加 subtile 切换与同步，收益需实测。
- softmax 的非正指数参数使用 `__expf`，LSE 仍为自然对数和 FP32；需按原阈值重验精度。
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

2026-10-07 本地验证：student 回归 82 项（11 项通过、71 项 CUDA 跳过），
benchmark 回归 25 项通过，operator 回归 10 项通过；Clang 对绑定和输入检查头文件的
C++17 语法检查通过。未运行 nvcc、GPU 数值测试或 compute-sanitizer。

## 整体算法 overall algorithm
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
