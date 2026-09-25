# 前向实现
为了保证全局内存的合并访问以及避免warp内的跨stride或者随机访问，直接使用一个warp对应一个token的并行策略，使用一个warp搬运一个token对应的完整词表参数。这样最大程度上保持了合并内存访问，同时warp数和block数足够多，保证了并发度。对应输入输出数组的地址是否满足地址对齐设计了向量化访存和标量访存，由wrapper调度。

# 反向实现
反向算子由两种实现思路，第一是类似于前向的一个warp对应一个token并行策略，优点在于搬运梯度时保证合并访问，缺点在于存在token被重复使用时，需要atomicAdd来解决write race，这样带来了较大的开销；第二种实现是每个线程负责一个token，在warp内将重复token分组，再本地对重复token完成reduction后再使用atomicAdd完成write。这样的好处是可以降低atomicAdd指令数，理论上最高可以降低31倍的atomicAdd指令数，但是每个线程负责一个token的模式导致warp数和block数太少，并发度很低，实际测试下来比较reference实现性能甚至倒退。

# 测试及结果
测试使用了不同模式的ids对前向和反向进行测试，ids中token的分布模式主要影响反向的性能。这四种模式分别是random,same,unique,hot。random模式接近真实token分布，token随机分布；same模式为极端测试，所有梯度只对应同一个token；unique模式下每一个梯度对应的token都不同；hot模式下部分token比较热门会被多次重复；
测试ids(8, 512), gradient(8, 512, 768), output(8192, 768).结果如下，2026.9.25
random grouped  forward : reference=15.46 us, student=10.02 us, speedup=1.54x  
random grouped  backward: reference=243.54 us, student=505.52 us, speedup=0.48x  
random baseline forward : reference=16.04 us, student=9.97 us, speedup=1.61x  
random baseline backward: reference=220.41 us, student=87.21 us, speedup=2.53x  
same   grouped  forward : reference=22.30 us, student=10.41 us, speedup=2.14x  
same   grouped  backward: reference=230.42 us, student=427.01 us, speedup=0.54x  
same   baseline forward : reference=20.51 us, student=10.17 us, speedup=2.02x  
same   baseline backward: reference=231.18 us, student=153.65 us, speedup=1.50x  
unique grouped  forward : reference=20.74 us, student=10.34 us, speedup=2.00x  
unique grouped  backward: reference=249.26 us, student=509.19 us, speedup=0.49x  
unique baseline forward : reference=20.35 us, student=10.19 us, speedup=2.00x  
unique baseline backward: reference=235.34 us, student=94.99 us, speedup=2.48x  
hot    grouped  forward : reference=20.83 us, student=10.28 us, speedup=2.03x  
hot    grouped  backward: reference=259.92 us, student=454.62 us, speedup=0.57x  
hot    baseline forward : reference=20.62 us, student=10.21 us, speedup=2.02x  
hot    baseline backward: reference=230.52 us, student=96.99 us, speedup=2.38x  
可以观察出，对于前向实现，四种模式下加速效果显著，最差的random模式也有1.5倍以上加速比。但是对于反向来说，基础实现加速效果也比较可观，再最差的same模式下也具有1.5加速比，在random和unique模式下甚至达到了2.5倍加速比；与之相对的是，grouped kennel 并没有获得预期收益，反而大幅降低了性能，相对来讲对same模式确实有所改善，但是kernel整体性能衰减太大，原因应该就是前文分析的并发度不够。
反向baseline 实现向量化访存后测试结果，
random grouped  forward : reference=18.95 us, student=10.02 us, speedup=1.89x
random grouped  backward: reference=281.83 us, student=506.41 us, speedup=0.56x
random baseline forward : reference=15.48 us, student=10.01 us, speedup=1.55x
random baseline backward: reference=259.23 us, student=106.25 us, speedup=2.44x
same   grouped  forward : reference=19.22 us, student=9.65 us, speedup=1.99x
same   grouped  backward: reference=264.23 us, student=426.77 us, speedup=0.62x
same   baseline forward : reference=16.05 us, student=8.66 us, speedup=1.85x
same   baseline backward: reference=278.98 us, student=389.43 us, speedup=0.72x
unique grouped  forward : reference=15.90 us, student=9.92 us, speedup=1.60x
unique grouped  backward: reference=270.22 us, student=510.10 us, speedup=0.53x
unique baseline forward : reference=15.95 us, student=9.94 us, speedup=1.60x
unique baseline backward: reference=235.23 us, student=85.51 us, speedup=2.75x
hot    grouped  forward : reference=20.42 us, student=9.36 us, speedup=2.18x
hot    grouped  backward: reference=226.36 us, student=455.33 us, speedup=0.50x
hot    baseline forward : reference=19.99 us, student=10.28 us, speedup=1.94x
hot    baseline backward: reference=228.46 us, student=97.01 us, speedup=2.35x
向量化访存对与原子和操作不友好，原因需要排查