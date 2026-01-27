# GPU ET unit test (k_ele_local)

This is a minimal CPU-vs-GPU reproducible check for the ET portion of `k_ele_local`
(`qEs/qEu/qEg/qTu/qTg`) on a single synthetic element.

It compiles a small standalone binary that links `src/GPU/rhs_kernels.cu` and runs
`k_ele_local` for 1 element, then compares the output against a host reference.

## Run

```bash
./validation/gpu_et_unit_test/run_et_unit_test.sh
```

### Environment variables

- `SUNDIALS_DIR` (default: `$HOME/sundials`) — used for headers only.
- `CUDA_HOME` (default: `/usr/local/cuda`)
- `NVCC` (optional) — overrides `nvcc` path.

