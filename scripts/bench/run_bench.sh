#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bench/run_bench.sh [options]

Options:
  --project <name>         Project name (default: ccw)
  --backend <cpu|omp|cuda|all>
                           Backend to run (default: all)
  --repeat <N>             Repeat count per backend (default: 3)
  --out-dir <dir>          Output directory (default: output/bench)
  --io <groups>            Output groups passed to SHUD (--io)
                           (default: all)
  --cuda-precond <default|on|off|auto>
                           CUDA CVODE preconditioner mode (default: default)
  --profile <none|nsys|nvprof>
                           Profiling mode (CUDA only; default: none)
  -h, --help               Show this help

Outputs (under --out-dir):
  - Per-run logs: <out-dir>/<project>/<backend>/run_XXX.log
  - Per-run time: <out-dir>/<project>/<backend>/run_XXX.time
  - Machine log:  <out-dir>/<project>/bench.log (TSV)
  - Summary:      <out-dir>/<project>/bench_summary.md

Notes:
  - Uses existing executables in repo root:
      cpu  -> ./shud
      omp  -> ./shud_omp
      cuda -> ./shud_cuda
  - Parses stdout for:
      CVODE_STATS  nfe=... nli=... nni=... netf=... npe=... nps=...
      BENCH_STATS  wall_s=... cvode_s=... io_s=... forcing_s=...
                  rhs_calls=... rhs_kernels=... rhs_launch_us=... rhs_graph=...
EOF
}

PROJECT="ccw"
BACKEND="all"
REPEAT=3
OUT_DIR="output/bench"
PROFILE="none"
CUDA_PRECOND="default"
IO_GROUPS="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="${2:-}"; shift 2 ;;
    --backend) BACKEND="${2:-}"; shift 2 ;;
    --repeat) REPEAT="${2:-}"; shift 2 ;;
    --out-dir|--out) OUT_DIR="${2:-}"; shift 2 ;;
    --io) IO_GROUPS="${2:-}"; shift 2 ;;
    --cuda-precond) CUDA_PRECOND="${2:-}"; shift 2 ;;
    --profile) PROFILE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${PROJECT}" ]]; then
  echo "ERROR: --project is required" >&2
  exit 2
fi
if ! [[ "${REPEAT}" =~ ^[0-9]+$ ]] || [[ "${REPEAT}" -lt 1 ]]; then
  echo "ERROR: --repeat must be a positive integer" >&2
  exit 2
fi
case "${BACKEND}" in
  cpu|omp|cuda|all) ;;
  *) echo "ERROR: invalid --backend '${BACKEND}' (expect cpu|omp|cuda|all)" >&2; exit 2 ;;
esac
case "${PROFILE}" in
  none|nsys|nvprof) ;;
  *) echo "ERROR: invalid --profile '${PROFILE}' (expect none|nsys|nvprof)" >&2; exit 2 ;;
esac
case "${CUDA_PRECOND}" in
  default|on|off|auto) ;;
  *) echo "ERROR: invalid --cuda-precond '${CUDA_PRECOND}' (expect default|on|off|auto)" >&2; exit 2 ;;
esac
if [[ -z "${IO_GROUPS}" ]]; then
  echo "ERROR: --io requires a non-empty value" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

require_bin() {
  local path="$1"
  if [[ ! -x "${path}" ]]; then
    echo "ERROR: missing executable: ${path}" >&2
    return 1
  fi
}

extract_kv() {
  local key="$1"
  local line="$2"
  awk -v key="${key}" '{
    for (i = 1; i <= NF; i++) {
      split($i, kv, "=")
      if (kv[1] == key) { print kv[2]; exit }
    }
  }' <<<"${line}"
}

run_one() {
  local backend="$1"
  local run_idx="$2"
  local bin="$3"

  local run_dir="${OUT_DIR}/${PROJECT}/${backend}"
  mkdir -p "${run_dir}"

  local run_tag
  run_tag="$(printf "run_%03d" "${run_idx}")"
  local log_file="${run_dir}/${run_tag}.log"
  local time_file="${run_dir}/${run_tag}.time"

  local cmd=( "${bin}" "${PROJECT}" )
  if [[ "${IO_GROUPS}" != "all" ]]; then
    cmd=( "${bin}" "--io" "${IO_GROUPS}" "${PROJECT}" )
  fi
  local cmd_prefix=()
  local env_prefix=()
  if [[ "${backend}" == "cuda" && "${CUDA_PRECOND}" != "default" ]]; then
    case "${CUDA_PRECOND}" in
      on) env_prefix=( SHUD_CUDA_PRECOND=1 ) ;;
      off) env_prefix=( SHUD_CUDA_PRECOND=0 ) ;;
      auto) env_prefix=( SHUD_CUDA_PRECOND=auto ) ;;
    esac
  fi
  if [[ "${backend}" == "cuda" && "${PROFILE}" != "none" ]]; then
    case "${PROFILE}" in
      nsys)
        if ! command -v nsys >/dev/null 2>&1; then
          echo "ERROR: --profile nsys requested but 'nsys' not found" >&2
          return 1
        fi
        cmd_prefix=( nsys profile --force-overwrite=true -o "${run_dir}/${run_tag}_nsys" )
        ;;
      nvprof)
        if ! command -v nvprof >/dev/null 2>&1; then
          echo "ERROR: --profile nvprof requested but 'nvprof' not found" >&2
          return 1
        fi
        cmd_prefix=( nvprof -o "${run_dir}/${run_tag}.nvprof" )
        ;;
    esac
  fi

  echo "[run] backend=${backend} ${run_tag}: ${cmd_prefix[*]:-} ${cmd[*]}"
  : >"${log_file}"
  if /usr/bin/time -f "%e" -o "${time_file}" env "${env_prefix[@]}" "${cmd_prefix[@]}" "${cmd[@]}" >>"${log_file}" 2>&1; then
    :
  else
    local rc=$?
    echo "WARN: backend=${backend} ${run_tag} exited with code ${rc} (see ${log_file})" >&2
  fi

  local wall_s=""
  if [[ -f "${time_file}" ]]; then
    wall_s="$(tr -d '[:space:]' <"${time_file}" || true)"
  fi

  local cvode_stats_line bench_stats_line
  cvode_stats_line="$(rg -n "CVODE_STATS" "${log_file}" | tail -n 1 | sed 's/^.*CVODE_STATS/CVODE_STATS/' || true)"
  bench_stats_line="$(rg -n "BENCH_STATS" "${log_file}" | tail -n 1 | sed 's/^.*BENCH_STATS/BENCH_STATS/' || true)"

  local nfe nli nni netf npe nps
  nfe="$(extract_kv nfe "${cvode_stats_line}")"
  nli="$(extract_kv nli "${cvode_stats_line}")"
  nni="$(extract_kv nni "${cvode_stats_line}")"
  netf="$(extract_kv netf "${cvode_stats_line}")"
  npe="$(extract_kv npe "${cvode_stats_line}")"
  nps="$(extract_kv nps "${cvode_stats_line}")"

  local bench_wall bench_cvode bench_io bench_forcing
  bench_wall="$(extract_kv wall_s "${bench_stats_line}")"
  bench_cvode="$(extract_kv cvode_s "${bench_stats_line}")"
  bench_io="$(extract_kv io_s "${bench_stats_line}")"
  bench_forcing="$(extract_kv forcing_s "${bench_stats_line}")"

  local rhs_calls rhs_kernels rhs_launch_us rhs_graph cuda_graph_mode
  rhs_calls="$(extract_kv rhs_calls "${bench_stats_line}")"
  rhs_kernels="$(extract_kv rhs_kernels "${bench_stats_line}")"
  rhs_launch_us="$(extract_kv rhs_launch_us "${bench_stats_line}")"
  rhs_graph="$(extract_kv rhs_graph "${bench_stats_line}")"
  cuda_graph_mode="$(extract_kv cuda_graph_mode "${bench_stats_line}")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${backend}" \
    "${run_idx}" \
    "${wall_s}" \
    "${bench_wall}" \
    "${bench_cvode}" \
    "${bench_io}" \
    "${bench_forcing}" \
    "${rhs_calls}" \
    "${rhs_kernels}" \
    "${rhs_launch_us}" \
    "${rhs_graph}" \
    "${cuda_graph_mode}" \
    "${nfe}" \
    "${nli}" \
    "${nni}" \
    "${netf}" \
    "${npe}" \
    "${nps}" \
    "${IO_GROUPS}" \
    "${CUDA_PRECOND}" \
    "${log_file}"
}

BACKENDS=()
if [[ "${BACKEND}" == "all" ]]; then
  BACKENDS=( cpu omp cuda )
else
  BACKENDS=( "${BACKEND}" )
fi

mkdir -p "${OUT_DIR}/${PROJECT}"
BENCH_LOG="${OUT_DIR}/${PROJECT}/bench.log"
SUMMARY_MD="${OUT_DIR}/${PROJECT}/bench_summary.md"

printf "backend\trun\twall_s\trun_wall_s\tcvode_s\tio_s\tforcing_s\trhs_calls\trhs_kernels\trhs_launch_us\trhs_graph\tcuda_graph_mode\tnfe\tnli\tnni\tnetf\tnpe\tnps\tio_groups\tcuda_precond\tlog\n" >"${BENCH_LOG}"

for backend in "${BACKENDS[@]}"; do
  bin=""
  case "${backend}" in
    cpu) bin="./shud" ;;
    omp) bin="./shud_omp" ;;
    cuda) bin="./shud_cuda" ;;
  esac

  if ! require_bin "${bin}"; then
    echo "WARN: skip backend=${backend} (missing ${bin})" >&2
    continue
  fi

  for ((i = 1; i <= REPEAT; i++)); do
    run_one "${backend}" "${i}" "${bin}" >>"${BENCH_LOG}"
  done
done

python3 - <<'PY' "${BENCH_LOG}" "${SUMMARY_MD}" "${PROJECT}"
import math
import sys
from collections import defaultdict

bench_log, out_md, project = sys.argv[1:4]

rows = []
with open(bench_log, "r", encoding="utf-8") as f:
  header = f.readline().rstrip("\n").split("\t")
  for line in f:
    line = line.rstrip("\n")
    if not line:
      continue
    parts = line.split("\t")
    rows.append(dict(zip(header, parts)))

def to_float(x):
  try:
    return float(x)
  except Exception:
    return None

by_backend = defaultdict(list)
for r in rows:
  by_backend[r["backend"]].append(r)

def mean_std(vals):
  vals = [v for v in vals if v is not None]
  if not vals:
    return None, None
  m = sum(vals) / len(vals)
  if len(vals) < 2:
    return m, 0.0
  var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
  return m, math.sqrt(var)

with open(out_md, "w", encoding="utf-8") as f:
  f.write(f"# SHUD benchmark summary ({project})\\n\\n")
  f.write(f"Source: `{bench_log}`\\n\\n")
  for backend in sorted(by_backend.keys()):
    rs = by_backend[backend]
    wall = [to_float(r["wall_s"]) for r in rs]
    cvode = [to_float(r["cvode_s"]) for r in rs]
    io = [to_float(r["io_s"]) for r in rs]
    forcing = [to_float(r["forcing_s"]) for r in rs]
    rhs_kernels = [to_float(r.get("rhs_kernels")) for r in rs]
    rhs_launch_us = [to_float(r.get("rhs_launch_us")) for r in rs]

    wall_m, wall_s = mean_std(wall)
    cvode_m, cvode_s = mean_std(cvode)
    io_m, io_s = mean_std(io)
    forcing_m, forcing_s = mean_std(forcing)
    rhs_kernels_m, rhs_kernels_s = mean_std(rhs_kernels)
    rhs_launch_us_m, rhs_launch_us_s = mean_std(rhs_launch_us)

    f.write(f"## {backend}\\n\\n")
    f.write("| metric | mean (s) | stdev (s) |\\n")
    f.write("|---|---:|---:|\\n")
    if wall_m is not None:
      f.write(f"| wall | {wall_m:.3f} | {wall_s:.3f} |\\n")
    if cvode_m is not None:
      f.write(f"| cvode | {cvode_m:.3f} | {cvode_s:.3f} |\\n")
    if io_m is not None:
      f.write(f"| io | {io_m:.3f} | {io_s:.3f} |\\n")
    if forcing_m is not None:
      f.write(f"| forcing | {forcing_m:.3f} | {forcing_s:.3f} |\\n")
    if rhs_kernels_m is not None:
      f.write(f"| rhs kernels/call | {rhs_kernels_m:.3f} | {rhs_kernels_s:.3f} |\\n")
    if rhs_launch_us_m is not None:
      f.write(f"| rhs launch (us/call) | {rhs_launch_us_m:.3f} | {rhs_launch_us_s:.3f} |\\n")

    # CVODE stats: show last run's values for quick reference
    last = rs[-1]
    stats = {k: last.get(k) for k in ("nfe", "nli", "nni", "netf", "npe", "nps")}
    if any(v for v in stats.values()):
      f.write("\\nCVODE stats (last run):\\n\\n")
      f.write("```\\n")
      f.write(" ".join([f\"{k}={v}\" for k, v in stats.items() if v]) + "\\n")
      f.write("```\\n\\n")
PY

echo "[done] bench_log=${BENCH_LOG}"
echo "[done] summary=${SUMMARY_MD}"
