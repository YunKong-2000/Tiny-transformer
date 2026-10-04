# 前向
前向算子使用FP32乘法路径，使用SIMT指令进行计算。

# 反向
反向类似，对应dX和dW有不同的输入layout。其余设定相同。

# 测试
## 初版
初版测试使用prefill场景形状（8，512， 768）*（768， 768）和，decode场景（1，512， 768）*（768， 768）进行前向反向计算性能测试，结果如下，  
linear prefill contiguous   forward: reference=2441.49 us, student=2303.54 us, speedup=1.06x
linear prefill contiguous   backward: reference=5353.64 us, student=6153.02 us, speedup=0.87x
linear decode contiguous   forward: reference=14.37 us, student=380.26 us, speedup=0.04x
linear decode contiguous   backward: reference=147.46 us, student=237.52 us, speedup=0.62x
saved runs/linear-performance.json
可见对于prefill场景性能出现下降，对于decode场景，性能大幅下降，需要进一步优化device gemm的模版参数设置。后续还需开发混合精度训练支持BF16乘法。  
10.4全形状测试:  
root@8cfa9e1a2f81:/workspace/tiny-transformer# python -m tiny_transformer.benchmarks --operator linear --output runs/linear-performance.json --device cuda:1
linear train qkv contiguous forward M,N,K=[4096, 2304, 768]: reference=821.21 us, student=894.86 us, speedup=0.92x
linear train qkv contiguous backward dX=[4096, 768, 2304], dW=[2304, 768, 4096]: reference=1716.45 us, student=2226.95 us, speedup=0.77x
linear train o contiguous forward M,N,K=[4096, 768, 768]: reference=293.11 us, student=301.79 us, speedup=0.97x
linear train o contiguous backward dX=[4096, 768, 768], dW=[768, 768, 4096]: reference=620.41 us, student=836.50 us, speedup=0.74x
linear train gate_up contiguous forward M,N,K=[4096, 4096, 768]: reference=1433.25 us, student=1498.92 us, speedup=0.96x
linear train gate_up contiguous backward dX=[4096, 768, 4096], dW=[4096, 768, 4096]: reference=3219.77 us, student=3717.41 us, speedup=0.87x
linear train down contiguous forward M,N,K=[4096, 768, 2048]: reference=736.88 us, student=792.46 us, speedup=0.93x
linear train down contiguous backward dX=[4096, 2048, 768], dW=[768, 2048, 4096]: reference=1540.87 us, student=1964.81 us, speedup=0.78x
linear train lm_head contiguous forward M,N,K=[4096, 8192, 768]: reference=2943.30 us, student=2886.96 us, speedup=1.02x
linear train lm_head contiguous backward dX=[4096, 768, 8192], dW=[8192, 768, 4096]: reference=6005.36 us, student=7059.92 us, speedup=0.85x
linear prefill qkv contiguous forward M,N,K=[4096, 2304, 768]: reference=833.81 us, student=911.06 us, speedup=0.92x
linear prefill o contiguous forward M,N,K=[4096, 768, 768]: reference=297.13 us, student=306.78 us, speedup=0.97x
linear prefill gate_up contiguous forward M,N,K=[4096, 4096, 768]: reference=1452.09 us, student=1520.79 us, speedup=0.95x
linear prefill down contiguous forward M,N,K=[4096, 768, 2048]: reference=745.24 us, student=801.58 us, speedup=0.93x
linear prefill lm_head contiguous forward M,N,K=[8, 8192, 768]: reference=30.63 us, student=150.45 us, speedup=0.20x
linear decode qkv contiguous forward M,N,K=[8, 2304, 768]: reference=23.54 us, student=148.35 us, speedup=0.16x
linear decode o contiguous forward M,N,K=[8, 768, 768]: reference=22.99 us, student=148.15 us, speedup=0.16x
linear decode gate_up contiguous forward M,N,K=[8, 4096, 768]: reference=23.72 us, student=148.40 us, speedup=0.16x
linear decode down contiguous forward M,N,K=[8, 768, 2048]: reference=23.63 us, student=385.98 us, speedup=0.06x
linear decode lm_head contiguous forward M,N,K=[8, 8192, 768]: reference=30.25 us, student=148.79 us, speedup=0.20x
saved runs/linear-performance.json

## 增加适用于推理场景小M问题规模kernel
增加了适用于小M场景下的kernel配置，设定CTAShape为<8, 64, 8>
root@8cfa9e1a2f81:/workspace/tiny-transformer# python -m tiny_transformer.benchmarks --operator linear --output runs/linear-performance.json --device cuda:1
linear train qkv contiguous forward M,N,K=[4096, 2304, 768]: reference=810.53 us, student=885.53 us, speedup=0.92x
linear train qkv contiguous backward dX=[4096, 768, 2304], dW=[2304, 768, 4096]: reference=1698.26 us, student=2222.49 us, speedup=0.76x
linear train o contiguous forward M,N,K=[4096, 768, 768]: reference=292.84 us, student=299.93 us, speedup=0.98x
linear train o contiguous backward dX=[4096, 768, 768], dW=[768, 768, 4096]: reference=616.80 us, student=832.73 us, speedup=0.74x
linear train gate_up contiguous forward M,N,K=[4096, 4096, 768]: reference=1421.67 us, student=1491.37 us, speedup=0.95x
linear train gate_up contiguous backward dX=[4096, 768, 4096], dW=[4096, 768, 4096]: reference=3190.00 us, student=3715.79 us, speedup=0.86x
linear train down contiguous forward M,N,K=[4096, 768, 2048]: reference=730.00 us, student=785.50 us, speedup=0.93x
linear train down contiguous backward dX=[4096, 2048, 768], dW=[768, 2048, 4096]: reference=1528.91 us, student=1594.59 us, speedup=0.96x
linear train lm_head contiguous forward M,N,K=[4096, 8192, 768]: reference=2907.54 us, student=2864.20 us, speedup=1.02x
linear train lm_head contiguous backward dX=[4096, 768, 8192], dW=[8192, 768, 4096]: reference=5953.67 us, student=6997.74 us, speedup=0.85x
linear prefill qkv contiguous forward M,N,K=[4096, 2304, 768]: reference=823.89 us, student=902.41 us, speedup=0.91x
linear prefill o contiguous forward M,N,K=[4096, 768, 768]: reference=295.80 us, student=303.83 us, speedup=0.97x
linear prefill gate_up contiguous forward M,N,K=[4096, 4096, 768]: reference=1442.88 us, student=1507.55 us, speedup=0.96x
linear prefill down contiguous forward M,N,K=[4096, 768, 2048]: reference=738.08 us, student=793.91 us, speedup=0.93x
linear prefill lm_head contiguous forward M,N,K=[8, 8192, 768]: reference=30.78 us, student=56.44 us, speedup=0.55x
linear decode qkv contiguous forward M,N,K=[8, 2304, 768]: reference=32.66 us, student=45.60 us, speedup=0.72x
linear decode o contiguous forward M,N,K=[8, 768, 768]: reference=32.16 us, student=44.24 us, speedup=0.73x
linear decode gate_up contiguous forward M,N,K=[8, 4096, 768]: reference=32.96 us, student=45.43 us, speedup=0.73x
linear decode down contiguous forward M,N,K=[8, 768, 2048]: reference=32.50 us, student=112.80 us, speedup=0.29x
linear decode lm_head contiguous forward M,N,K=[8, 8192, 768]: reference=30.46 us, student=55.51 us, speedup=0.55x
saved runs/linear-performance.json
测试结果显示，对于小M场景的矩阵乘法的性能下降程度明显缓解，但是还是较差。