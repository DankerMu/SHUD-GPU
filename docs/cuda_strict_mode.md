# CUDA strict mode (deterministic reduce / strict FP)

This repo supports a “strict(er) reproducibility” configuration for the CUDA backend, aimed at improving CPU↔CUDA consistency when you need tighter tolerances (e.g. `rel_max <= 1e-6`).

## Runtime switches

- `SHUD_DETERMINISTIC_REDUCE=1` (default: `1`)
  - Enables deterministic reductions for key CUDA hotspots (notably river/segment aggregation).
  - When set to `0`, the CUDA backend falls back to the legacy `atomicAdd` accumulation path (faster on some GPUs/cases, but non-deterministic and often slower under contention).

- `SHUD_STRICT_FP=1` (default: `0`)
  - Requests strict floating-point behavior for CUDA runs.
  - At runtime, this also forces `SHUD_DETERMINISTIC_REDUCE=1`.
  - **Note:** disabling FMA reliably requires a strict build (see below). If `SHUD_STRICT_FP=1` is set on a non-strict build, SHUD prints a warning and continues.

## Build switch (recommended for real strict-FP)

Rebuild the CUDA binary with strict FP flags:

```bash
make shud_cuda STRICT_FP=1
```

This adds NVCC flags:
- `--fmad=false` (disable fused multiply-add contraction)
- `--prec-div=true --prec-sqrt=true` (force precise div/sqrt)
- `-DSHUD_STRICT_FP_BUILD` (lets the binary report whether it was built in strict mode)

## Performance notes

- Deterministic reductions may add extra kernel(s) compared to the all-atomic path, but often reduce overall time by avoiding heavy `atomicAdd` contention.
- Disabling FMA can noticeably slow down double-precision code. If you enable strict FP, consider also enabling CUDA Graph capture for small problems to reduce launch overhead:
  - `SHUD_CUDA_GRAPH=auto|1` (see `README.md` / `BENCH_STATS` output).

## How to evaluate accuracy

On a CUDA machine:

```bash
make clean && make shud && make shud_omp && make shud_cuda

# Baseline CUDA
./shud_cuda ccw

# Strict configuration
SHUD_STRICT_FP=1 SHUD_DETERMINISTIC_REDUCE=1 ./shud_cuda ccw

python3 post_analysis/accuracy_comparison.py \
  output/ccw_cpu output/ccw_cuda \
  --tol_cuda 1e-6
```

If `rel_max<=1e-6` is still not achievable on `ccw`, prefer documenting a per-file tolerance (or focusing on key state variables) and include CVODE stats (`CVODE_STATS` line) so “accuracy improvements” are not just a solver-path change.

