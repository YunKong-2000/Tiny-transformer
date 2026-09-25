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
