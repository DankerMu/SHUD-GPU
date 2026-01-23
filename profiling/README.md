# Profiling (Nsight Systems / Nsight Compute)

This folder contains a lightweight workflow to find GPU performance bottlenecks in SHUD’s CUDA RHS.

## Prerequisites

- CUDA build enabled (`make shud_cuda`)
- NVIDIA Nsight Systems CLI (`nsys`) and/or Nsight Compute CLI (`ncu`) available in `PATH`
- NVTX ranges are enabled by default in the CUDA build and are linked via `-lnvToolsExt`

## What is instrumented (NVTX)

NVTX ranges are added in:

- `src/GPU/f_cuda.cpp`: top-level `f_cuda` call + stream sync/wait + RHS launch.
- `src/GPU/rhs_kernels.cu`: `launch_rhs_kernels()` stages:
  - `RHS/0_init`
  - `RHS/1_apply_bc`
  - `RHS/2_ele_local`
  - `RHS/3_ele_edge_surface`
  - `RHS/4_ele_edge_subsurface`
  - `RHS/5_segment_exchange`
  - `RHS/6_river_down_up`
  - `RHS/7_lake_toparea`
  - `RHS/8_apply_dy_element`
  - `RHS/9_apply_dy_river`
  - `RHS/10_apply_dy_lake`

These show up as nested ranges in Nsight Systems and help attribute GPU work to RHS phases.

## CVODE statistics (nfe, nli, nni, netf)

At the end of a run, SHUD prints a single parseable line:

```
CVODE_STATS nfe=... nli=... nni=... netf=...
```

This is emitted from `src/Model/shud.cpp`.

## Quick start

Build:

```bash
make shud_cuda CUDA_HOME=/usr/local/cuda
```

Run on the example project (`ccw`) with the provided script:

```bash
# Baseline wall time + CVODE stats
bash profiling/run_profiling.sh --mode baseline --project ccw

# Nsight Systems timeline (kernel time distribution + NVTX)
bash profiling/run_profiling.sh --mode nsys --project ccw

# Nsight Compute kernel metrics (bandwidth + atomics)
bash profiling/run_profiling.sh --mode ncu --project ccw
```

Outputs are written under `profiling/results/<timestamp>/`.

## Suggested analysis checklist

- Wall time: use the model’s existing timing output (see “Time used by model”), and/or `profiling/run_profiling.sh --mode baseline` (uses `/usr/bin/time -p` if available).
- Kernel time distribution: open the `.nsys-rep` in Nsight Systems UI and inspect the CUDA kernel summary / timeline grouped by NVTX ranges.
- Memory bandwidth utilization: use Nsight Compute sections like `SpeedOfLight` and `MemoryWorkloadAnalysis`.
- Atomic operation overhead: in Nsight Compute, inspect the atomics-related subsections inside `MemoryWorkloadAnalysis` (or run a broader capture with `NCU_ARGS='--set full'` if needed).

## Overrides

The wrapper script accepts optional env vars to adjust profiler flags:

- `NSYS_ARGS="..."` overrides default `nsys profile` arguments.
- `NCU_ARGS="..."` overrides default `ncu` arguments.

Example:

```bash
NSYS_ARGS="--trace=cuda,nvtx --sample=none --stats=true" \
  bash profiling/run_profiling.sh --mode nsys --project ccw
```

