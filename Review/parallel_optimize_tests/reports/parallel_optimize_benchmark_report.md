# Parallel Optimize Microbenchmark Report

Date: 2026-06-04

## Scope

This report records microbenchmark results for the OpenMP changes in
`origin/refactor/parallel-optimize`:

- DPMD coordinate buffer fill in `ESolver_DP::runner()`
- DPMD force copy back in `ESolver_DP::runner()`
- Verlet velocity scaling in `Verlet::thermalize()`
- MSST velocity component scaling in `MSST::rescale()`
- Nose-Hoover velocity scaling in `Nose_Hoover::particle_thermo()`

The benchmark reproduces the changed loops outside the full ABACUS runtime.
It checks serial-vs-OpenMP numerical consistency and measures 1/2/4/8/16
thread performance.

## Environment

- CPU: Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz
- Logical CPUs: 16
- Physical cores: 8
- Threads per core: 2
- NUMA nodes: 1
- Compiler: g++
- OpenMP binding: `OMP_PROC_BIND=close`, `OMP_PLACES=cores`
- Atoms: 2,000,000
- Repeats: 5

## Correctness

All kernels produced `max_abs_diff = 0` against their serial reference
implementations for all tested thread counts.

## Performance Summary

| Kernel | 1 thread speedup | 2 threads speedup | 4 threads speedup | 8 threads speedup | 16 threads speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPMD coordinate fill | 0.88x | 1.70x | 3.34x | 5.50x | 6.99x |
| DPMD force copy back | 0.99x | 2.01x | 3.80x | 7.19x | 8.78x |
| Verlet velocity scaling | 0.98x | 1.88x | 3.79x | 7.80x | 9.24x |
| MSST velocity component scaling | 0.96x | 1.77x | 3.31x | 7.28x | 8.45x |
| NHC velocity scaling | 0.97x | 1.84x | 2.98x | 7.21x | 9.77x |

## Notes

- 1-thread OpenMP results are slightly slower than serial for most kernels,
  which is expected from OpenMP dispatch overhead.
- 8-thread results scale well for all kernels, especially force copy back and
  velocity scaling loops.
- 16-thread results improve further but with lower efficiency because the
  machine has 8 physical cores and 16 logical CPUs.
- DPMD coordinate fill scales less strongly than the other kernels because the
  optimized version replaces nested sequential traversal with flat index lookup,
  adding two indexed reads per atom.

Full CSV:

- `parallel_optimize_benchmark.csv`

Run log:

- `parallel_optimize_benchmark.log`
