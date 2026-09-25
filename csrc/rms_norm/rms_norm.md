# 前向计算
输入：$\mathbf{X}:(B, L, d)$, $\mathbf{gamma}:(d)$, $\epsilon$

前向的并行划分策略比较直接，使用一个warp对应X中完整一行，这样可以使得reduction完全在warp中进行，避免了跨warp的使用共享内存的归约操作。同时一个warp内的线程合并访问全局内存中的数据使得内存访问较为高效。仍然根据输入数组的地址对齐实现了向量化的访存和标量访存。

# 反向计算
输入：$\mathbf{dY}:(B, L, d)$, $\mathbf{X}:(B, L, d)$, $\mathbf{gamma}:(d)$, $\mathbf{a}:(B \times L)$
 
先计算$u_i = dy_i * \gamma_i$  
然后计算$S=\sum_{i} u_i * x_i * a$  
然后得到$dx_i = a[u_i - x_i*a\frac{S}{d}]$  
最终计算$d\gamma_i = \sum_n dy_{n,i}x_{n,i}a_n$

$dy_i$,$u_i$,$x_i$被反复使用，可以考虑使用shared memory缓存。  
一个block处理一行，总共需要使用的shared memory大小为$3d*sizeof(float)$，使用静态共享内存，按照$d=1024$预分配共享内存。最终使用的共享内存大小是12KiB.

当前实现使用 3 个长度 1024 的 FP32 缓存，加 8 个 warp 小计，共 12320 字节共享内存。
主机 wrapper 限制 `0 < H <= 1024`，避免设备断言破坏 CUDA 上下文。
前向输出 `(Y, R)`，其中 R 就是这里的 a；反向输入 `(X, dY, gamma, R)`，输出 `(dX, dGamma)`。
前向与反向的 PyTorch 缓存、绑定、测试及性能命令见 [算子文档](../../docs/operators/rms_norm.md#7-前向缓存与-pytorch-调用链)。

# 性能测试
对不同的两种形状进行了测试，对前后向都进行了测试。  
prefill：X=[8,512,768]，也是默认训练时的激活形状。  
decode：X=[8,1,768]，用于测试少量行的情况。  
推理阶段的测试结果如下，  
rms_norm prefill contiguous   forward: reference=60.41 us, student=13.38 us, speedup=4.51x  
rms_norm prefill contiguous   backward: reference=348.98 us, student=124.38 us, speedup=2.81x  
rms_norm decode contiguous   forward: reference=56.65 us, student=11.81 us, speedup=4.80x  
rms_norm decode contiguous   backward: reference=337.75 us, student=114.10 us, speedup=2.96x  
测试结果显示，对于$d=768$这样规模的特征，使用warp进行归约的策略是较为高效的，加速比均在4。5以上。对于后向计算来说，由于算法本身存在大量的访存读取，而且目前的缓存策略使用了较多的共享内存，因此可能一定程度上限制了kernel的并发度，但是从结果来看，加速比在2.5以上，kernel的优化效果相当好。  
目前算子的反向计算受限，由于使用预分配的共享内存，目前每个block使用了12Kib左右的共享内存，限制$d<1024$，对于极小$d$的算例，浪费了较多的共享内存，kernel的并发度将比较低；对于较大$d$的算例，共享内存不足。因此后续需要针对不同$d$的规模需要实现不同的kernel。目前的实现仅适用于当前tiny transformer的训练推理。