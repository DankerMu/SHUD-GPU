#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BACKENDS: list[tuple[str, str]] = [
    ("cpu", "CPU serial"),
    ("omp", "OpenMP"),
    ("cuda", "CUDA"),
]

_CVODE_KV_RE = re.compile(r"\b(nfe|nli|nni|netf|npe|nps)=([0-9]+)\b")


@dataclass(frozen=True)
class BackendResult:
    key: str
    name: str
    time_s: Optional[float]
    nfe: Optional[int]
    nli: Optional[int]
    nni: Optional[int]
    netf: Optional[int]
    npe: Optional[int]
    nps: Optional[int]


def _read_time_seconds(path: Path) -> Optional[float]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    try:
        return float(text.split()[0])
    except ValueError:
        print(f"WARN: failed to parse time from {path}: {text!r}", file=sys.stderr)
        return None


def _parse_cvode_stats(path: Path) -> dict[str, int]:
    last_stats: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "CVODE_STATS" not in line:
                    continue
                pairs = dict(_CVODE_KV_RE.findall(line))
                if not pairs:
                    continue
                last_stats = {k: int(v) for k, v in pairs.items()}
    except FileNotFoundError:
        return {}
    return last_stats


def _fmt_float(value: Optional[float], *, digits: int) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _fmt_int(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    return str(value)


def _compute_speedup(cpu_time: Optional[float], time_s: Optional[float], *, is_cpu: bool) -> Optional[float]:
    if cpu_time is None or time_s is None or time_s <= 0:
        return None
    if is_cpu:
        return 1.0
    return cpu_time / time_s


def _collect_results(project: str, log_dir: Path) -> list[BackendResult]:
    results: list[BackendResult] = []
    for key, name in BACKENDS:
        time_s = _read_time_seconds(log_dir / f"{project}_{key}.time")
        stats = _parse_cvode_stats(log_dir / f"{project}_{key}.log")

        results.append(
            BackendResult(
                key=key,
                name=name,
                time_s=time_s,
                nfe=stats.get("nfe"),
                nli=stats.get("nli"),
                nni=stats.get("nni"),
                netf=stats.get("netf"),
                npe=stats.get("npe"),
                nps=stats.get("nps"),
            )
        )
    return results


def _render_markdown(project: str, log_dir: Path, results: list[BackendResult]) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cpu_time = next((r.time_s for r in results if r.key == "cpu"), None)

    lines: list[str] = []
    lines.append("# SHUD Benchmark Report")
    lines.append("")
    lines.append(f"- Project: `{project}`")
    lines.append(f"- Log dir: `{log_dir.as_posix()}`")
    lines.append(f"- Generated: `{now}`")
    lines.append("")
    lines.append("## Wall time")
    lines.append("")
    lines.append("| Backend | Time(s) | Speedup |")
    lines.append("|---|---:|---:|")
    for r in results:
        speedup = _compute_speedup(cpu_time, r.time_s, is_cpu=(r.key == "cpu"))
        lines.append(
            f"| {r.name} | {_fmt_float(r.time_s, digits=3)} | {_fmt_float(speedup, digits=2)} |"
        )
    lines.append("")
    lines.append("## CVODE stats")
    lines.append("")
    lines.append("| Backend | nfe | nli | nni | netf | npe | nps |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            "| {name} | {nfe} | {nli} | {nni} | {netf} | {npe} | {nps} |".format(
                name=r.name,
                nfe=_fmt_int(r.nfe),
                nli=_fmt_int(r.nli),
                nni=_fmt_int(r.nni),
                netf=_fmt_int(r.netf),
                npe=_fmt_int(r.npe),
                nps=_fmt_int(r.nps),
            )
        )
    lines.append("")
    lines.append(
        "_Note: CUDA results may differ slightly from CPU due to floating-point precision._"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare SHUD backend outputs (time + CVODE_STATS) and emit a Markdown report."
    )
    parser.add_argument("--project", default="ccw", help="Project name (default: ccw)")
    parser.add_argument(
        "--log-dir",
        default="output/benchmark_logs",
        help="Directory containing <project>_<backend>.{log,time} files",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Output Markdown report file path (default: <log-dir>/report_<project>.md)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    report_path = Path(args.report) if args.report else log_dir / f"report_{args.project}.md"

    results = _collect_results(args.project, log_dir)
    md = _render_markdown(args.project, log_dir, results)

    report_dir = report_path.parent
    if report_dir and not report_dir.exists():
        report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
    print(f"[ok] wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
