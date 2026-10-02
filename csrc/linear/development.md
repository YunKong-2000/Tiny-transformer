# 前向
前向算子使用FP32乘法路径，使用SIMT指令进行计算。

# 反向
反向类似，对应dX和dW有不同的输入layout。其余设定相同。

# 测试
初版测试使用prefill场景形状（8，512， 768）*（768， 768）和，decode场景（1，512， 768）*（768， 768）进行前向反向计算性能测试，结果如下，  
linear prefill contiguous   forward: reference=2441.49 us, student=2303.54 us, speedup=1.06x
linear prefill contiguous   backward: reference=5353.64 us, student=6153.02 us, speedup=0.87x
linear decode contiguous   forward: reference=14.37 us, student=380.26 us, speedup=0.04x
linear decode contiguous   backward: reference=147.46 us, student=237.52 us, speedup=0.62x
saved runs/linear-performance.json
可见对于prefill场景性能出现下降，对于decode场景，性能大幅下降，需要进一步优化device gemm的模版参数设置。后续还需开发混合精度训练支持BF16乘法。