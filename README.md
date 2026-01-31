# LISFLOOD-GPU-2026

GPU加速的LISFLOOD点源扩散计算程序

## 项目目标

实现点源（QVAR类型）+ 时间流量过程曲线的GPU并行扩散计算
- 输入：DEM + 多个点源位置(bci) + 各点源的时间流量曲线(bdy)
- 输出：水深分布随时间的变化
- 场景：城市内涝模拟，管网溢流点作为点源输入

## 文件说明

### 核心程序
| 文件 | 说明 |
|------|------|
| `lisflood_gpu.cpp` | **主程序**，完整的GPU点源扩散，读取dem/bci/bdy文件 |

### 参考文件
| 文件 | 说明 |
|------|------|
| `lisflood_gpu_core.cpp` | GPU核心函数库（有Qy索引bug，仅供参考） |
| `test_point_source.cpp` | 50x50小网格测试程序 |
| `test_kokkos_cuda.cpp` | Kokkos+CUDA环境测试 |

### 数据文件
| 文件 | 说明 |
|------|------|
| `dem.ascii` | 1500x1500 DEM数据，cellsize=10m |
| `bci.bci` | 10个QVAR点源定义 |
| `bdy.bdy` | 时变流量数据（5 m²/s） |
| `gpu_test.par` | 参数文件 |

## 编译方法

```bash
cd build
cmake .. -DKokkos_DIR=/usr/local/lib/cmake/Kokkos
make -j4
```

## 运行方法

```bash
./lisflood_gpu ../gpu_test.par
```

## 当前状态（2026-01-28）

- [x] 程序能编译运行
- [x] 能读取dem/bci/bdy文件
- [x] 能输出水深栅格
- [ ] **待修复：质量守恒问题**（输出体积只有理论值的约10%）

## 算法参考

- 原始LISFLOOD代码：`/mnt/d/2024/2025-1/17/lisflood-yuanshi/`
- Kokkos架构参考：`/mnt/d/2024/2025-1/17/serghei-master/`

## 开发环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8GB) |
| CUDA | 12.6 |
| Kokkos | 4.4.99 (CUDA+OpenMP后端) |
| 操作系统 | WSL2 Linux |
