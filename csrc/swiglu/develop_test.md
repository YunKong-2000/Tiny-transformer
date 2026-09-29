# 前向
swiglu算子接收的输入张量可能是chunk布局，导致其第二维开始出现stride。因此在前向中使用了cute。并行策略上使用一个warp对应一个token，一个block包含4个warp。

# 反向
反向计算的实现和前向相同，对比前向多读取了gradient张量，另外反向计算中的浮点数运算大幅增加。

# 测试
swiglu prefill contiguous   forward: reference=109.66 us, student=114.23 us, speedup=0.96x
swiglu prefill contiguous   backward: reference=193.68 us, student=142.04 us, speedup=1.36x
swiglu decode contiguous   forward: reference=23.56 us, student=31.15 us, speedup=0.76x
swiglu decode contiguous   backward: reference=105.51 us, student=100.02 us, speedup=1.05x
saved runs/swiglu-performance.json
测试结果显示prefill形状下的算子在前向计算时的性能较差，原因可能是由于并行策略较差或者尚未使用向量化读取。decode形状下的测试结果性能明显下降，原因确定为并行策略较差导致有效warp和cta数太少，限制了并发度。
下一步先进行并行策略上的优化，使用一整个或多个CTA处理一个token，增加有效warp数，先提升并发度，后续视情况增加向量化读取kernel.