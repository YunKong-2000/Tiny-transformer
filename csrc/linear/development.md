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
## 修改了small-M kernle的CTAShape和WarpShape，减小tile_N来增加CTA数量，增加tile_K减小循环次数
设定CTAShape为<8, 32, 16>，WarpShape为<8, 16, 16>,测试结果如下  
root@8cfa9e1a2f81:/workspace/tiny-transformer# python -m tiny_transformer.benchmarks --operator linear --output runs/linear-performance-CTA8-32-16.json --device cuda:1
linear train qkv contiguous forward M,N,K=[4096, 2304, 768]: reference=809.52 us, student=887.48 us, speedup=0.91x
linear train qkv contiguous backward dX=[4096, 768, 2304], dW=[2304, 768, 4096]: reference=1688.90 us, student=2221.29 us, speedup=0.76x
linear train o contiguous forward M,N,K=[4096, 768, 768]: reference=292.58 us, student=299.95 us, speedup=0.98x
linear train o contiguous backward dX=[4096, 768, 768], dW=[768, 768, 4096]: reference=616.86 us, student=832.62 us, speedup=0.74x
linear train gate_up contiguous forward M,N,K=[4096, 4096, 768]: reference=1409.16 us, student=1476.45 us, speedup=0.95x
linear train gate_up contiguous backward dX=[4096, 768, 4096], dW=[4096, 768, 4096]: reference=3162.50 us, student=3712.97 us, speedup=0.85x
linear train down contiguous forward M,N,K=[4096, 768, 2048]: reference=724.64 us, student=782.18 us, speedup=0.93x
linear train down contiguous backward dX=[4096, 2048, 768], dW=[768, 2048, 4096]: reference=1505.30 us, student=1966.99 us, speedup=0.77x
linear train lm_head contiguous forward M,N,K=[4096, 8192, 768]: reference=2879.33 us, student=2850.28 us, speedup=1.01x
linear train lm_head contiguous backward dX=[4096, 768, 8192], dW=[8192, 768, 4096]: reference=5920.91 us, student=6946.37 us, speedup=0.85x
linear prefill qkv contiguous forward M,N,K=[4096, 2304, 768]: reference=822.84 us, student=896.27 us, speedup=0.92x
linear prefill o contiguous forward M,N,K=[4096, 768, 768]: reference=292.73 us, student=300.40 us, speedup=0.97x
linear prefill gate_up contiguous forward M,N,K=[4096, 4096, 768]: reference=1429.17 us, student=1496.55 us, speedup=0.95x
linear prefill down contiguous forward M,N,K=[4096, 768, 2048]: reference=731.84 us, student=787.18 us, speedup=0.93x
linear prefill lm_head contiguous forward M,N,K=[8, 8192, 768]: reference=30.60 us, student=33.30 us, speedup=0.92x
linear decode qkv contiguous forward M,N,K=[8, 2304, 768]: reference=30.74 us, student=31.38 us, speedup=0.98x
linear decode o contiguous forward M,N,K=[8, 768, 768]: reference=30.39 us, student=31.60 us, speedup=0.96x
linear decode gate_up contiguous forward M,N,K=[8, 4096, 768]: reference=30.96 us, student=31.84 us, speedup=0.97x
linear decode down contiguous forward M,N,K=[8, 768, 2048]: reference=31.61 us, student=60.09 us, speedup=0.53x
linear decode lm_head contiguous forward M,N,K=[8, 8192, 768]: reference=30.31 us, student=33.09 us, speedup=0.92x
saved runs/linear-performance-CTA8-32-16.json
推理阶段已经十分接近参考实现的性能，但是Down阶段的性能还是较差

## 实现split-K
增加一个split-k kernel,使得推理阶段的down走split-k路径，每两个cta处理一整行，测试结果  
root@8cfa9e1a2f81:/workspace/tiny-transformer# python -m tiny_transformer.benchmarks --operator linear --device cuda:1 --output runs/linear-kernel-performance.json

linear train qkv contiguous forward M,N,K=[4096, 2304, 768]: reference=811.44 us, student=892.00 us, speedup=0.91x, reference=17.8638 TFLOP/s, student=16.2506 TFLOP/s
linear train qkv contiguous dx M,N,K=[4096, 768, 2304]: reference=925.90 us, student=1068.20 us, speedup=0.87x, reference=15.6556 TFLOP/s, student=13.5701 TFLOP/s
linear train qkv contiguous dweight M,N,K=[2304, 768, 4096]: reference=766.20 us, student=1162.01 us, speedup=0.66x, reference=18.9188 TFLOP/s, student=12.4745 TFLOP/s

linear train o contiguous forward M,N,K=[4096, 768, 768]: reference=292.62 us, student=297.88 us, speedup=0.98x, reference=16.5124 TFLOP/s, student=16.2207 TFLOP/s
linear train o contiguous dx M,N,K=[4096, 768, 768]: reference=295.95 us, student=371.94 us, speedup=0.80x, reference=16.3267 TFLOP/s, student=12.9910 TFLOP/s
linear train o contiguous dweight M,N,K=[768, 768, 4096]: reference=313.58 us, student=447.38 us, speedup=0.70x, reference=15.4087 TFLOP/s, student=10.8004 TFLOP/s

linear train gate_up contiguous forward M,N,K=[4096, 4096, 768]: reference=1419.66 us, student=1489.10 us, speedup=0.95x, reference=18.1521 TFLOP/s, student=17.3056 TFLOP/s
linear train gate_up contiguous dx M,N,K=[4096, 768, 4096]: reference=1646.04 us, student=1877.03 us, speedup=0.88x, reference=15.6556 TFLOP/s, student=13.7290 TFLOP/s
linear train gate_up contiguous dweight M,N,K=[4096, 768, 4096]: reference=1519.38 us, student=1855.70 us, speedup=0.82x, reference=16.9607 TFLOP/s, student=13.8868 TFLOP/s

linear train down contiguous forward M,N,K=[4096, 768, 2048]: reference=730.53 us, student=785.83 us, speedup=0.93x, reference=17.6377 TFLOP/s, student=16.3966 TFLOP/s
linear train down contiguous dx M,N,K=[4096, 2048, 768]: reference=734.91 us, student=819.42 us, speedup=0.90x, reference=17.5325 TFLOP/s, student=15.7245 TFLOP/s
linear train down contiguous dweight M,N,K=[768, 2048, 4096]: reference=780.85 us, student=787.44 us, speedup=0.99x, reference=16.5011 TFLOP/s, student=16.3631 TFLOP/s

linear train lm_head contiguous forward M,N,K=[4096, 8192, 768]: reference=2912.39 us, student=2865.90 us, speedup=1.02x, reference=17.6967 TFLOP/s, student=17.9837 TFLOP/s
linear train lm_head contiguous dx M,N,K=[4096, 768, 8192]: reference=2817.00 us, student=3727.05 us, speedup=0.76x, reference=18.2959 TFLOP/s, student=13.8285 TFLOP/s
linear train lm_head contiguous dweight M,N,K=[8192, 768, 4096]: reference=3165.01 us, student=3274.61 us, speedup=0.97x, reference=16.2842 TFLOP/s, student=15.7392 TFLOP/s

linear prefill qkv contiguous forward M,N,K=[4096, 2304, 768]: reference=825.79 us, student=902.37 us, speedup=0.92x, reference=17.5534 TFLOP/s, student=16.0638 TFLOP/s

linear prefill o contiguous forward M,N,K=[4096, 768, 768]: reference=295.80 us, student=302.28 us, speedup=0.98x, reference=16.3347 TFLOP/s, student=15.9844 TFLOP/s

linear prefill gate_up contiguous forward M,N,K=[4096, 4096, 768]: reference=1443.63 us, student=1508.49 us, speedup=0.96x, reference=17.8508 TFLOP/s, student=17.0832 TFLOP/s

linear prefill down contiguous forward M,N,K=[4096, 768, 2048]: reference=742.68 us, student=794.09 us, speedup=0.94x, reference=17.3493 TFLOP/s, student=16.2260 TFLOP/s

linear prefill lm_head contiguous forward M,N,K=[8, 8192, 768]: reference=28.83 us, student=30.94 us, speedup=0.93x, reference=3.4921 TFLOP/s, student=3.2540 TFLOP/s

linear decode qkv contiguous forward M,N,K=[8, 2304, 768]: reference=13.21 us, student=23.35 us, speedup=0.57x, reference=2.1433 TFLOP/s, student=1.2126 TFLOP/s

linear decode o contiguous forward M,N,K=[8, 768, 768]: reference=9.41 us, student=23.26 us, speedup=0.40x, reference=1.0028 TFLOP/s, student=0.4058 TFLOP/s

linear decode gate_up contiguous forward M,N,K=[8, 4096, 768]: reference=20.59 us, student=28.26 us, speedup=0.73x, reference=2.4442 TFLOP/s, student=1.7809 TFLOP/s

linear decode down contiguous forward M,N,K=[8, 768, 2048]: reference=16.11 us, student=32.01 us, speedup=0.50x, reference=1.5624 TFLOP/s, student=0.7862 TFLOP/s
linear decode down contiguous split_k_partials M,N,K=[8, 768, 2048]: student=30.26 us, performance=0.8317 TFLOP/s
linear decode down contiguous split_k_reduce M,N,K=[8, 768, 2048]: student=1.69 us, performance=0.0073 TFLOP/s

linear decode lm_head contiguous forward M,N,K=[8, 8192, 768]: reference=28.82 us, student=31.41 us, speedup=0.92x, reference=3.4934 TFLOP/s, student=3.2052 TFLOP/s
saved runs/linear-kernel-performance.json