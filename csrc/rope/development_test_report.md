# 前向
使用cute实现tile的划分和thread->value的映射。尚未使用向量化访存

# 反向
使用模版函数实现前反向计算，当模版参数为false时实现反向，仅需调整一个输入参数和计算系数。

# 测试结果
root@1e55f708e731:/workspace/tiny-transformer# python -m tiny_transformer.benchmarks --operator rope --output 
runs/rope-performance.json
rope prefill contiguous   forward: reference=343.40 us, student=164.80 us, speedup=2.08x
rope prefill contiguous   backward: reference=545.14 us, student=163.88 us, speedup=3.33x
rope decode contiguous   forward: reference=83.37 us, student=4.39 us, speedup=18.98x
rope decode contiguous   backward: reference=241.37 us, student=53.69 us, speedup=4.50x
saved runs/rope-performance.json
对小规模的算例加速比较明显。
向量化访存待实现。