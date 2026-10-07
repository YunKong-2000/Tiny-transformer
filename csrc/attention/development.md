# flashAttention

当前接入 FP32 SIMT 和 BF16 Tensor Core 前向，固定 `head_dim=64`，每 CTA 128 线程。
两条路径均支持 causal prefill、decode/chunk 和序列尾块，Q/K/V 必须具有相同 dtype。
FP32 支持文档 segment IDs；host 将 Q/K 和 segment IDs 连续化，V 直接使用实际 stride。
BF16 要求 SM80+，不支持 segment IDs；host 将 Q/K/V 连续化，必要时 clone 保证起点
16-byte 对齐，kernel 本身不处理任意 stride。BF16 使用 BQ=BK=64、双缓冲和 48 KiB
动态共享内存；score、softmax 统计量和输出累加器为 FP32，P 在 PV 前转成 BF16。
原生接口返回 `(O, LSE)`：O 与输入同 dtype，LSE 始终为 FP32；Python 只返回 O。
反向、AMP/autocast 和 FP16 尚未实现；需要梯度时明确报错。
输入 batch、heads、序列长度必须为正，且 `Tk = past_len + Tq`。
FP32 二维 grid 要求 `B*H <= 65535`、`ceil(Tq/32) <= INT_MAX`；
BF16 一维 grid 要求 `B*H*ceil(Tq/64) <= INT_MAX`，乘法前检查溢出。

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
查询 CUDA runtime。每轮 PV 后的缓冲区释放同步和下一 stage 的发布同步合并：
先等待预取完成，再做一次 CTA barrier；到达 barrier 时所有线程都已完成当前 PV。
最后一轮不再复用 shared memory，无需额外 barrier。P 写完后的消费前同步保留。
同步改动仍需重跑数值测试和 compute-sanitizer racecheck/synccheck 验证。
性能入口新增 `--attention-timing graph --phases forward`，完整算子图重放与原有
operator 耗时分开报告，具体口径见 benchmarks README。新增 GPU 测试检查未对齐
prefill/decode 输入在 CUDA Graph 中捕获和重复重放的 O/LSE。

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
每个CTA在kernel开始时分配$Q_i$对应的shared memory。大小为$B_q \times D_n \times bytes$。另外还需为K_tile和V_tile分配对应的shared memory，大小分别为$B_k \times B_d \times bytes \times stages$和$B_k \times B_n \times bytes \times stages$。其余中间量均存储在寄存器中，特别是S,P和N。
kernel开始时直接从global memory读取$Q_i$至shared memory。  
### 循环设计
然后进入streaming loop,这个循环的循环变量j在KV的序列方向$T_k$,循环次数是$T_k/B_k$。  
然后进入第一个归约循环，这个循环的循环变量d在K矩阵的特征方向进行内积累加，循环次数是$D_n/B_d$。每次循环中都会从GMEM中读取一个K_tile，也即是$K_{jd}:(B_k, B_d)$，大小为$B_k \times B_d$。然后和$Q_i$中对应的一段$Q_{id}:(B_q, B_d)$进行矩阵乘法。并将其结果累加至$S_{ij}:(B_q, B_k)$。
第一个归约循环结束后就得到了完整的$S_{ij}:(B_q, B_k)$，他的值将会暂时存放在寄存器中。然后可以直接进行scale和mask。  
然后开始更新行最大值和指数和，并且计算$P_{ij}:(B_q, B_k)$,$S_{ij}$和$P_{ij}$复用了寄存器中的空间。
接下来遍历输出特征子块，每次从 GMEM 读取 $V_{jn}:(B_k,B_n)$。
PV 真正的归约维是 key token：按 RK=8 准备 P/V 寄存器片段，累加到同一个输出子块。
代码在 softmax 阶段先对完整的输出累加器乘一次 alpha，再计算各子块的 PV，不能重复缩放。
完整 streaming 循环后，将输出累加器逐行除以 l 得到 O，并写回 GMEM；LSE=m+log(l) 单独保存。
### 流水线
streaming循环可以使用pipeline进行延迟隐藏，可以将整个streaming循环分为第一个

### swizzle技术
本质上Swizzle操作是对元素偏移offset的二进制数的分析和操作。数据类型决定了offset对应的bank的起始位置，例如fp32的起始位就是0，因为一个bank恰好对应一个元素，而对于fp16或者bf16起始位就是1，因为一个bank对应两个数，如果有int8,fp8这样的类型，起始位就是2，因为一个bank对应8个元素;而读取数据的段长决定了最终的保留位，段长以元素个数位单位，之前的例子是一个线程读连续4个fp32,那就保留最低两位，线的例子事以8个bf16位一段，那就保留最低8位；其次就是同一次访问指令中同一bank的不同地址的stride，在这个例子中每隔64个元素就会落入到同一个bank,所以就是发生bank conflict的两个地址第6位以上才不同，这个数字恰好也是bank的起始位加上5，也就是bank的终止位的下一位；最后需要决定低位中哪些位需要被修改，很明显，最低的保留位是不能动的，因此需要从保留位下一位开始，根据bank conflict的way的数量决定，比如是8way，就需要log(8)=3位做修改，这样才能得到8个不同的新目标地址。
