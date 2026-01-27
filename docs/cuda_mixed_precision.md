# CUDA mixed-precision preconditioner (experimental)

This repo supports an **experimental** mixed-precision mode for the CUDA CVODE preconditioner (Block-Jacobi).

Goal: reduce preconditioner memory traffic and use faster fp32 math inside `PSolve`, while keeping the state vector and RHS in `realtype=double`.

## Switch

- `SHUD_CUDA_PRECOND_FP=fp64|fp32` (default: `fp64`)
  - `fp64`: legacy behavior (store/apply preconditioner in double).
  - `fp32`: store/apply preconditioner in float (mixed precision).

Accepted aliases: `double`/`float`, `0`/`1`, `on`/`off`.

## Notes / expected impact

- Mixed precision can change GMRES/Newton convergence behavior (different solver path), even if RHS math is unchanged.
- Always compare both **accuracy** and **CVODE stats** (`CVODE_STATS` line) when evaluating.

## How to evaluate

On a CUDA machine (with `shud_cuda` built):

```bash
# fp64 precond (baseline)
SHUD_CUDA_PRECOND=1 SHUD_CUDA_PRECOND_FP=fp64 ./shud_cuda ccw

# fp32 precond (mixed precision)
SHUD_CUDA_PRECOND=1 SHUD_CUDA_PRECOND_FP=fp32 ./shud_cuda ccw

python3 post_analysis/accuracy_comparison.py \
  output/ccw_cpu output/ccw_cuda \
  --tol_cuda 1e-3
```

If you need tighter reproducibility, consider combining with deterministic reductions / strict-fp build flags:
- `docs/cuda_strict_mode.md`

