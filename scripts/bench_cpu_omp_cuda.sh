#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bench_cpu_omp_cuda.sh [project]
  bash scripts/bench_cpu_omp_cuda.sh --project ccw [--log-dir output/benchmark_logs] [--report output/benchmark_logs/report_ccw.md]

Runs SHUD with three backends (CPU-serial / OpenMP / CUDA), captures stdout/stderr logs,
records wall time via /usr/bin/time, then generates a Markdown report.

Notes:
  - Each binary uses its natural default backend (no --backend flag needed):
      ./shud     -> cpu (serial)
      ./shud_omp -> omp
      ./shud_cuda-> cuda
  - Missing backend executables are skipped with a warning.
  - Requires conda env "DA-RL" (this script will try to activate it).
EOF
}

PROJECT="ccw"
LOG_DIR="output/benchmark_logs"
REPORT=""
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project)
      PROJECT="${2:-}"; shift 2 ;;
    --log-dir|--out)
      LOG_DIR="${2:-}"; shift 2 ;;
    --report)
      REPORT="${2:-}"; shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage
      exit 2 ;;
    *)
      POSITIONAL+=("$1")
      shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -gt 1 ]]; then
  echo "ERROR: too many positional args: ${POSITIONAL[*]}" >&2
  usage
  exit 2
fi
if [[ ${#POSITIONAL[@]} -eq 1 ]]; then
  PROJECT="${POSITIONAL[0]}"
fi
if [[ -z "$REPORT" ]]; then
  REPORT="${LOG_DIR}/report_${PROJECT}.md"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

# Ensure expected directories exist.
mkdir -p scripts
mkdir -p "$LOG_DIR"

activate_conda_env() {
  local env_name="DA-RL"
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then
    return 0
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found; please run: conda activate ${env_name}" >&2
    exit 1
  fi

  # Enable `conda activate` for non-interactive shells.
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$env_name"
}

activate_conda_env

echo "[config] project=${PROJECT}"
echo "[config] log_dir=${LOG_DIR}"
echo "[config] report=${REPORT}"
echo "[config] conda_env=${CONDA_DEFAULT_ENV:-<none>}"

run_backend() {
  local key="$1" label="$2" bin="$3"

  if [[ ! -x "$bin" ]]; then
    echo "WARN: skip ${label} (missing executable: ${bin})" >&2
    return 0
  fi

  local log_file="${LOG_DIR}/${PROJECT}_${key}.log"
  local time_file="${LOG_DIR}/${PROJECT}_${key}.time"

  echo "[run] ${label}: ${bin} ${PROJECT}"
  if /usr/bin/time -f "%e" -o "$time_file" "$bin" "$PROJECT" >"$log_file" 2>&1; then
    echo "[done] ${label}: log=${log_file} time=${time_file}"
  else
    local rc=$?
    echo "WARN: ${label} exited with code ${rc}; see log=${log_file}" >&2
  fi
}

run_backend "cpu"  "CPU serial" "./shud"
run_backend "omp"  "OpenMP"     "./shud_omp"
run_backend "cuda" "CUDA"       "./shud_cuda"

python "${SCRIPT_DIR}/compare_outputs.py" \
  --project "$PROJECT" \
  --log-dir "$LOG_DIR" \
  --report "$REPORT"

echo "[done] Report written to: ${REPORT}"
