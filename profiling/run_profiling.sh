#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash profiling/run_profiling.sh [--mode baseline|nsys|ncu|all] [--bin ./shud_cuda] [--project ccw] [--out profiling/results/...]

Examples:
  # Baseline wall time + CVODE stats (no profiler)
  bash profiling/run_profiling.sh --mode baseline --project ccw

  # Nsight Systems: kernel time distribution + NVTX ranges
  bash profiling/run_profiling.sh --mode nsys --project ccw

  # Nsight Compute: memory bandwidth + atomic overhead (per-kernel)
  bash profiling/run_profiling.sh --mode ncu --project ccw

  # Everything (runs the model multiple times)
  bash profiling/run_profiling.sh --mode all --project ccw

Notes:
  - Build the CUDA binary first: `make shud_cuda`
  - Override profiler flags via env vars:
      NSYS_ARGS="..." NCU_ARGS="..." bash profiling/run_profiling.sh --mode nsys
  - Extra args after `--` are passed to the SHUD binary.
EOF
}

MODE="all"
BIN="./shud_cuda"
PROJECT="ccw"
OUT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"; shift 2 ;;
    --bin)
      BIN="${2:-}"; shift 2 ;;
    --project)
      PROJECT="${2:-}"; shift 2 ;;
    --out)
      OUT_DIR="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="profiling/results/$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found or not executable: $BIN" >&2
  echo "Build with: make shud_cuda" >&2
  exit 1
fi

run_baseline() {
  echo "[baseline] ${BIN} ${PROJECT} ${EXTRA_ARGS[*]-}" | tee "$OUT_DIR/baseline_cmd.txt"
  if command -v /usr/bin/time >/dev/null 2>&1; then
    (/usr/bin/time -p "$BIN" "$PROJECT" "${EXTRA_ARGS[@]}" 2>&1) | tee "$OUT_DIR/baseline.log"
  else
    ("$BIN" "$PROJECT" "${EXTRA_ARGS[@]}" 2>&1) | tee "$OUT_DIR/baseline.log"
  fi
  grep -a "CVODE_STATS" "$OUT_DIR/baseline.log" || true
}

run_nsys() {
  if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: nsys not found in PATH. Install Nsight Systems (CLI)." >&2
    exit 1
  fi

  local rep_base="$OUT_DIR/nsys_${PROJECT}"
  local args_default=(--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=true --cuda-memory-usage=true --stats=true --force-overwrite=true)
  local args=()
  if [[ -n "${NSYS_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    args=(${NSYS_ARGS})
  else
    args=("${args_default[@]}")
  fi

  echo "[nsys] output base: ${rep_base}" | tee "$OUT_DIR/nsys_cmd.txt"
  echo "[nsys] nsys profile ${args[*]} -o ${rep_base} -- ${BIN} ${PROJECT} ${EXTRA_ARGS[*]-}" >>"$OUT_DIR/nsys_cmd.txt"
  (nsys profile "${args[@]}" -o "$rep_base" -- "$BIN" "$PROJECT" "${EXTRA_ARGS[@]}" 2>&1) | tee "$OUT_DIR/nsys_stdout.log"
  grep -a "CVODE_STATS" "$OUT_DIR/nsys_stdout.log" || true
}

run_ncu() {
  if ! command -v ncu >/dev/null 2>&1; then
    echo "ERROR: ncu not found in PATH. Install Nsight Compute (CLI)." >&2
    exit 1
  fi

  local rep_base="$OUT_DIR/ncu_${PROJECT}"
  local args_default=(--target-processes all --section SpeedOfLight --section MemoryWorkloadAnalysis --section LaunchStats)
  local args=()
  if [[ -n "${NCU_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    args=(${NCU_ARGS})
  else
    args=("${args_default[@]}")
  fi

  echo "[ncu] output base: ${rep_base}" | tee "$OUT_DIR/ncu_cmd.txt"
  echo "[ncu] ncu ${args[*]} -o ${rep_base} ${BIN} ${PROJECT} ${EXTRA_ARGS[*]-}" >>"$OUT_DIR/ncu_cmd.txt"
  (ncu "${args[@]}" -o "$rep_base" "$BIN" "$PROJECT" "${EXTRA_ARGS[@]}" 2>&1) | tee "$OUT_DIR/ncu_stdout.log"
  grep -a "CVODE_STATS" "$OUT_DIR/ncu_stdout.log" || true
}

case "$MODE" in
  baseline) run_baseline ;;
  nsys) run_nsys ;;
  ncu) run_ncu ;;
  all)
    run_baseline
    run_nsys
    run_ncu
    ;;
  *)
    echo "ERROR: unknown --mode=$MODE (expected baseline|nsys|ncu|all)" >&2
    usage
    exit 2
    ;;
esac

echo "[done] Outputs in: $OUT_DIR"

