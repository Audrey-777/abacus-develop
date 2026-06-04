# DPMD 与 MD 温控扩展 OpenMP 优化修改说明

日期：2026-06-03

本次根据 collaborator 在 `refactor/parallel-optimize` 分支中的优化记录，整理 DPMD 接口层和 MD 温控相关的 OpenMP 并行尝试。该分支主要处理此前 0531 基础优化中暂未覆盖的 DPMD、Verlet、MSST 和 NHC 局部逐原子循环；未改动随机数相关循环，也未进入 DPMD 外部库内部。

原始分项记录保留在：

- `opt_logs/dpmd_interface_20260603.md`
- `opt_logs/verlet_thermalize_20260603.md`
- `opt_logs/msst_rescale_20260603.md`
- `opt_logs/nhc_particle_thermo_20260603.md`

## 修改范围

### DPMD 接口层

- `source/source_esolver/esolver_dp.h`
  - 新增 `atom_type_index` 和 `atom_local_index` 缓存，用于把全局原子序号 `iat` 映射到 `UnitCell` 的元素类型 `it` 和类型内局部序号 `ia`。

- `source/source_esolver/esolver_dp.cpp`
  - `before_all_runners()`：初始化全局原子到 `UnitCell` 存储位置的索引缓存。
  - `runner()`：
    - 并行 DPMD 坐标缓冲区 `coord` 填充；
    - 保持 cell 填充和 `dp.compute()` 外部库调用串行；
    - 并行 DPMD force 从外部库返回数组到 `dp_force` 的逐原子回填与单位转换；
    - virial 仅为 3x3 小矩阵，保持串行回填。

### MD 温控相关循环

- `source/source_md/verlet.cpp`
  - `Verlet::thermalize()`：对 rescaling / rescale_v / berendsen 分支中每个原子的速度缩放加入 OpenMP 并行。
  - Anderson thermostat 涉及 `std::rand()` 和 `gaussrand()`，保持串行。

- `source/source_md/msst.cpp`
  - `MSST::rescale()`：在晶胞 dilation 更新和 `unitcell::setup_cell_after_vc()` 完成后，并行缩放每个原子在冲击方向 `sd` 上的速度分量。

- `source/source_md/nhchain.cpp`
  - `Nose_Hoover::particle_thermo()`：保持 thermostat chain 变量递推串行，只并行最终 `scale` 应用到每个原子速度的循环。

## 并行策略

- 所有新增循环使用 `#pragma omp parallel for schedule(static)`。
- 对短循环使用 `if (nat >= 256)` 或同类阈值，降低小体系线程启动开销。
- 只并行逐原子独立写入循环，不改变 MD 时间步顺序、thermostat chain 递推顺序、晶胞更新顺序和 DPMD 外部库调用边界。
- 本次优化不涉及 reduction，因此理论上串行和并行结果应保持 bitwise 一致。

## 审查记录

本次对照四份优化记录与 `source/` 下实际代码进行了审查。结论是：各优化点的数据依赖判断基本合理，新增 OpenMP 循环均为每原子独立读写，没有发现需要立即停止测试的关键性错误。

需要注意的健壮性建议：

- `ESolver_DP::before_all_runners()` 构建 `atom_type_index` / `atom_local_index` 后，建议补充 `assert(iat == ucell.nat)`。原记录中说明“可通过索引构建时的 assert 覆盖”坐标填充后的 `assert`，但当前源码实际没有该断言。该问题不影响常规输入下的并行正确性，但补上断言有助于尽早发现 `UnitCell` 原子计数与类型遍历不一致的问题。

## 未执行项

- 未并行随机数相关循环，例如 Verlet Anderson thermostat 中的随机速度重采样。
- 未并行 NHC thermostat chain 递推、MSST 晶胞更新和 DPMD `dp.compute()` 外部库调用。
- 未做完整 ABACUS 端到端 DPMD / MD case。本次测试使用独立 microbenchmark 隔离评估改动循环本身的 OpenMP 收益。
- 未在 16 或 32 个物理核机器上验证更大并行扩展区间；当前 16 线程结果来自 8 物理核 / 16 逻辑线程服务器。

## 性能测试记录

### 测试文件与运行方式

测试文件放在独立 review worktree 的测试目录中：

- `Review/parallel_optimize_tests/tests/parallel_optimize_benchmark.cpp`
- `Review/parallel_optimize_tests/tests/run_parallel_optimize_benchmark.sh`
- `Review/parallel_optimize_tests/reports/parallel_optimize_benchmark.csv`
- `Review/parallel_optimize_tests/reports/parallel_optimize_benchmark.log`
- `Review/parallel_optimize_tests/reports/parallel_optimize_benchmark_report.md`

本次测试使用独立 C++ OpenMP microbenchmark，而不是完整 ABACUS 端到端输入。原因是 DPMD 端到端测试依赖外部 DeePMD 模型文件、编译宏和运行环境；本 benchmark 直接复现本次改动中真正发生变化的逐原子循环，可以隔离测量 DPMD 接口层数据准备/回填与 MD 温控速度缩放的 OpenMP 收益。

编译与运行命令：

```bash
cd /root/abacus-parallel-optimize-review
THREADS=1,2,4,8,16 ./Review/parallel_optimize_tests/tests/run_parallel_optimize_benchmark.sh
```

脚本内部编译命令：

```bash
g++ -O3 -std=c++17 -fopenmp \
    Review/parallel_optimize_tests/tests/parallel_optimize_benchmark.cpp \
    -o Review/parallel_optimize_tests/build/parallel_optimize_benchmark
```

测试参数：

| 参数 | 值 |
| --- | --- |
| CPU | Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz |
| 物理核 / 逻辑 CPU | 8 / 16 |
| NUMA 节点 | 1 |
| 编译器 | g++ |
| 原子数 | 2,000,000 |
| 每项重复次数 | 5 |
| 线程数 | 1 / 2 / 4 / 8 / 16 |
| OpenMP 绑定 | `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |

每个 kernel 同时运行串行版本和 OpenMP 版本，表中 `serial_ms` 与 `omp_ms` 均为单次调用平均耗时。`speedup = serial_ms / omp_ms`，`efficiency = speedup / threads`。

### 测试结果

| kernel | threads | serial_ms | omp_ms | speedup | efficiency | max_abs_diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dpmd_coord_fill | 1 | 9.821983 | 11.213064 | 0.875941 | 0.875941 | 0 |
| dpmd_force_copy_back | 1 | 10.948092 | 11.029436 | 0.992625 | 0.992625 | 0 |
| verlet_thermalize_velocity | 1 | 5.784265 | 5.888199 | 0.982349 | 0.982349 | 0 |
| msst_rescale_velocity_component | 1 | 5.083415 | 5.303124 | 0.958570 | 0.958570 | 0 |
| nhc_particle_thermo_velocity | 1 | 5.578181 | 5.733537 | 0.972904 | 0.972904 | 0 |
| dpmd_coord_fill | 2 | 9.864032 | 5.818020 | 1.695428 | 0.847714 | 0 |
| dpmd_force_copy_back | 2 | 10.754669 | 5.345393 | 2.011951 | 1.005976 | 0 |
| verlet_thermalize_velocity | 2 | 5.509689 | 2.923828 | 1.884410 | 0.942205 | 0 |
| msst_rescale_velocity_component | 2 | 4.801548 | 2.717274 | 1.767046 | 0.883523 | 0 |
| nhc_particle_thermo_velocity | 2 | 5.497487 | 2.985008 | 1.841699 | 0.920850 | 0 |
| dpmd_coord_fill | 4 | 9.963194 | 2.984800 | 3.337977 | 0.834494 | 0 |
| dpmd_force_copy_back | 4 | 10.834987 | 2.853441 | 3.797165 | 0.949291 | 0 |
| verlet_thermalize_velocity | 4 | 5.770737 | 1.520700 | 3.794790 | 0.948698 | 0 |
| msst_rescale_velocity_component | 4 | 5.183090 | 1.563954 | 3.314093 | 0.828523 | 0 |
| nhc_particle_thermo_velocity | 4 | 5.637649 | 1.894768 | 2.975377 | 0.743844 | 0 |
| dpmd_coord_fill | 8 | 9.865107 | 1.792455 | 5.503683 | 0.687960 | 0 |
| dpmd_force_copy_back | 8 | 10.797789 | 1.501847 | 7.189673 | 0.898709 | 0 |
| verlet_thermalize_velocity | 8 | 5.689709 | 0.729527 | 7.799179 | 0.974897 | 0 |
| msst_rescale_velocity_component | 8 | 5.084523 | 0.697988 | 7.284536 | 0.910567 | 0 |
| nhc_particle_thermo_velocity | 8 | 5.714033 | 0.792147 | 7.213346 | 0.901668 | 0 |
| dpmd_coord_fill | 16 | 9.921995 | 1.419156 | 6.991476 | 0.436967 | 0 |
| dpmd_force_copy_back | 16 | 10.716615 | 1.220650 | 8.779432 | 0.548714 | 0 |
| verlet_thermalize_velocity | 16 | 5.733320 | 0.620529 | 9.239400 | 0.577463 | 0 |
| msst_rescale_velocity_component | 16 | 4.985979 | 0.590147 | 8.448709 | 0.528044 | 0 |
| nhc_particle_thermo_velocity | 16 | 5.743777 | 0.587939 | 9.769343 | 0.610584 | 0 |

### 8 线程优化程度汇总

| 优化点 | 对应源码位置 | 8 线程加速比 | 优化程度判断 |
| --- | --- | ---: | --- |
| DPMD 坐标填充 | `source/source_esolver/esolver_dp.cpp` | 5.50x | 有明显收益，但低于纯连续写入循环 |
| DPMD force 回填 | `source/source_esolver/esolver_dp.cpp` | 7.19x | 接近线性扩展 |
| Verlet 速度缩放 | `source/source_md/verlet.cpp` | 7.80x | 接近线性扩展 |
| MSST 单方向速度缩放 | `source/source_md/msst.cpp` | 7.28x | 接近线性扩展 |
| NHC 速度缩放 | `source/source_md/nhchain.cpp` | 7.21x | 接近线性扩展 |

### 16 线程扩展情况

| 优化点 | 对应源码位置 | 16 线程加速比 | 扩展情况 |
| --- | --- | ---: | --- |
| DPMD 坐标填充 | `source/source_esolver/esolver_dp.cpp` | 6.99x | 继续提升，但效率下降明显 |
| DPMD force 回填 | `source/source_esolver/esolver_dp.cpp` | 8.78x | 继续提升，受超线程和内存带宽限制 |
| Verlet 速度缩放 | `source/source_md/verlet.cpp` | 9.24x | 继续提升，16 线程效率低于 8 线程 |
| MSST 单方向速度缩放 | `source/source_md/msst.cpp` | 8.45x | 继续提升，扩展效率下降 |
| NHC 速度缩放 | `source/source_md/nhchain.cpp` | 9.77x | 继续提升，扩展效率下降 |

### 分析

本次 OpenMP 优化收益最明确的是逐原子速度缩放和 DPMD force 回填。这些循环的迭代之间没有写冲突，每个线程处理独立原子编号，读入的缩放因子或单位换算系数均为只读标量。因此在 8 物理核机器上普遍达到约 7x 到 8x 的加速，属于低风险、高收益优化点。

Verlet、MSST 和 NHC 三个温控相关循环都只并行最终速度缩放步骤，不改变 thermostat 参数计算、晶胞更新或 chain 递推顺序。测试中三者 `max_abs_diff` 均为 0，说明并行结果与串行参考完全一致。MSST 只缩放单个速度分量，单次迭代计算量比完整 `Vec3` 缩放更少，因此更容易受线程调度开销和内存写入开销影响，但 8 线程仍达到约 7.28x。

DPMD force 回填是典型连续数组读取和逐原子写入循环，8 线程加速约 7.19x，扩展性较好。该优化不进入 `dp.compute()` 外部库内部，只减少 ABACUS 侧后处理开销，风险较低。

DPMD 坐标填充的 8 线程加速约 5.50x，低于其它优化点。主要原因是并行版本为了消除嵌套遍历中的串行 `iat++`，改为按全局原子编号访问 `atom_type_index[iat]` 和 `atom_local_index[iat]`，每个原子额外引入两次数组索引读取和一次间接访问 `ucell.atoms[it].tau[ia]`。该写法仍有明显收益，但更容易受内存访问模式和缓存局部性限制。

16 线程结果来自 8 物理核 / 16 逻辑线程机器，因此不能直接代表 16 物理核扩展效果。所有 kernel 在 16 线程下仍继续提升，但效率明显低于 8 线程，符合超线程共享执行单元、缓存和内存带宽后的预期。更大物理核数上的线性扩展区间建议放到最终优化和总结报告阶段，在 16 或 32 物理核机器上统一测试。

正确性方面，本次 benchmark 中所有 kernel 的 `max_abs_diff` 均为 0。由于本次优化不涉及 reduction，仅涉及每原子独立写入，因此理论上不应产生浮点求和顺序变化；测试结果符合预期。

## 正式构建与单测验证

本次已完成独立 OpenMP microbenchmark 的编译和运行，未执行完整 ABACUS 构建、MD 单测或 DPMD 端到端 case。

microbenchmark 编译结果：

```text
g++ -O3 -std=c++17 -fopenmp Review/parallel_optimize_tests/tests/parallel_optimize_benchmark.cpp -o Review/parallel_optimize_tests/build/parallel_optimize_benchmark
```

结果：编译通过，随后完成 1 / 2 / 4 / 8 / 16 线程测试。

后续正式验证建议：

- 构建相关 MD 单测目标，至少覆盖 `MODULE_MD_verlet`、`MODULE_MD_msst` 和 `MODULE_MD_nhc`。
- 在具备 DeePMD 依赖和模型文件的环境中补充 DPMD 端到端 MD case。
- 在最终集成分支上重复 microbenchmark，确认与其它 OpenMP 优化合并后没有性能回退。
- 下周在更高物理核数机器上补充 1 / 2 / 4 / 8 / 16 / 32 线程 scaling 测试，用于判断线性区间和饱和点。

## 回溯方式

本次 review/test 相关修改已记录在分支：

- `review/parallel-optimize-tests`

测试与报告提交：

- `test: add parallel optimize microbenchmarks`

如需回滚测试与报告，可对该提交执行 `git revert <commit>`；collaborator 的优化源码提交仍保留在 `refactor/parallel-optimize` 分支中。
