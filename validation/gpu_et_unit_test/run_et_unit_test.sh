#!/usr/bin/env bash
set -euo pipefail

sd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${sd}/../.." && pwd)"

SUNDIALS_DIR="${SUNDIALS_DIR:-$HOME/sundials}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

NVCC_BIN="${NVCC:-}"
if [[ -z "${NVCC_BIN}" ]]; then
  if [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
    NVCC_BIN="${CUDA_HOME}/bin/nvcc"
  else
    NVCC_BIN="nvcc"
  fi
fi

out="${sd}/et_unit_test"

echo "==> build ${out}"
"${NVCC_BIN}" -O2 -std=c++14 -D_CUDA_ON \
  -I "${root}/src/GPU" -I "${root}/src/Model" -I "${root}/src/Equations" -I "${root}/src" \
  -I "${SUNDIALS_DIR}/include" -I "${CUDA_HOME}/include" \
  -L "${CUDA_HOME}/lib64" -lnvToolsExt \
  -o "${out}" \
  "${sd}/et_unit_test.cu" \
  "${root}/src/GPU/rhs_kernels.cu"

echo "==> run"
"${out}"

