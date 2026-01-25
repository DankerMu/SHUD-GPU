#!/usr/bin/env python3
from __future__ import annotations

"""
Compare SHUD outputs across backends (CPU / OMP / CUDA) and generate a Markdown report.

Expected workflow:
  1) Run each backend and save outputs to separate directories (recommended):
       bash post_analysis/run_accuracy_test.sh --project ccw
  2) Compare accuracy:
       python3 post_analysis/accuracy_comparison.py --project ccw

.dat format (see src/classes/Model_Control.cpp):
  - 1024 bytes header text
  - StartTime (double)
  - NumVar (double, int-like)
  - icol[NumVar] (double array; typically element/river IDs)
  - For each record:
      t (double) + values[NumVar] (double array)

This script treats CPU as the reference and reports CPU-vs-OMP and CPU-vs-CUDA errors.

Error metrics:
  - abs_* : absolute error statistics on all aligned samples
  - rel_* : absolute error normalized by max(|CPU|) per file (global scale)
"""

import argparse
import datetime as _dt
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


def _import_shud_reader() -> tuple[object, object]:
    root = Path(__file__).resolve().parent.parent
    reader_dir = root / "validation" / "tsr" / "py"
    sys.path.insert(0, str(reader_dir))
    try:
        from shud_reader import DatFormatError, DatReader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "failed to import SHUD .dat reader from "
            f"{(reader_dir / 'shud_reader.py').as_posix()}; "
            "ensure the repo is intact."
        ) from e
    return DatReader, DatFormatError


DatReader, DatFormatError = _import_shud_reader()


try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("numpy is required; please install it (e.g., `conda install numpy`).") from e


KEY_VARS: dict[str, str] = {
    "eleysurf": "ySf (Surface water depth)",
    "eleyunsat": "yUs (Unsaturated zone)",
    "eleygw": "yGw (Groundwater)",
    "rivystage": "yRiv (River stage)",
}


def _var_code_from_filename(name: str) -> str:
    base = name[:-4] if name.lower().endswith(".dat") else name
    if "." in base:
        return base.split(".", 1)[1]
    return base


def _classify_var(code: str) -> str:
    if code in KEY_VARS:
        return "State(y*)"
    if code.startswith(("eleq", "rivq", "lakq")) or "q" in code:
        return "Flux(Q*)"
    if code.startswith(("elev", "rn_")):
        return "Flux/Diag"
    return "Other"


def _float_key(a: np.ndarray, *, scale: float = 1e6) -> np.ndarray:
    # Robust key for align-by-time/ID without depending on exact FP equality.
    return np.rint(a * scale).astype(np.int64, copy=False)


@dataclass(frozen=True)
class DatArrays:
    path: Path
    var_code: str
    times: np.ndarray  # float64, (T,)
    time_key: np.ndarray  # int64, (T,)
    col_ids: np.ndarray  # float64, (N,)
    col_key: np.ndarray  # int64, (N,)
    values: np.ndarray  # float64, (T, N)


@dataclass(frozen=True)
class CompareStats:
    # Absolute error statistics.
    abs_max: float
    abs_mean: float
    abs_std: float
    # Relative error statistics (normalized by global max(|ref|)).
    rel_max: float
    rel_mean: float
    rel_std: float
    # Reference scale used for relative errors.
    ref_scale: float
    # Alignment info.
    n_time_ref: int
    n_time_test: int
    n_time_common: int
    n_col_ref: int
    n_col_test: int
    n_col_common: int
    n_total: int
    n_finite: int
    n_nonfinite_ref: int
    n_nonfinite_test: int
    n_nonfinite_mismatch: int
    # Worst-case location (on aligned subset).
    worst_time: Optional[float]
    worst_col_id: Optional[float]
    worst_ref: Optional[float]
    worst_test: Optional[float]


@dataclass(frozen=True)
class FileResult:
    filename: str
    var_code: str
    var_type: str
    cpu_path: Path
    omp_path: Optional[Path]
    cuda_path: Optional[Path]
    omp: Optional[CompareStats]
    cuda: Optional[CompareStats]
    omp_error: Optional[str] = None
    cuda_error: Optional[str] = None


@dataclass(frozen=True)
class PlotItem:
    label: str
    path: Path


def _read_dat_arrays(path: Path) -> DatArrays:
    reader = DatReader(path)
    meta = reader.meta

    # Fast payload read (no pandas required).
    total_doubles = meta.num_records * (1 + meta.num_var)
    if total_doubles <= 0:
        times = np.zeros((0,), dtype=np.float64)
        values = np.zeros((0, meta.num_var), dtype=np.float64)
    else:
        from array import array

        a = array("d")
        with meta.path.open("rb") as f:
            f.seek(meta.data_offset)
            try:
                a.fromfile(f, total_doubles)
            except EOFError as e:
                raise DatFormatError("truncated data section while reading payload", path=meta.path) from e

        if len(a) != total_doubles:
            raise DatFormatError(
                f"truncated data section: expected {total_doubles} doubles, got {len(a)}",
                path=meta.path,
            )

        file_little = meta.endianness == "<"
        host_little = sys.byteorder == "little"
        if file_little != host_little:
            a.byteswap()

        payload = np.frombuffer(a, dtype=np.float64).reshape(meta.num_records, 1 + meta.num_var)
        times = payload[:, 0]
        values = payload[:, 1:]

    col_ids = np.asarray(meta.col_ids, dtype=np.float64)

    return DatArrays(
        path=path,
        var_code=_var_code_from_filename(path.name),
        times=times.astype(np.float64, copy=False),
        time_key=_float_key(times),
        col_ids=col_ids,
        col_key=_float_key(col_ids),
        values=values.astype(np.float64, copy=False),
    )


def _align_indices(ref_key: np.ndarray, test_key: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common, ref_idx, test_idx = np.intersect1d(ref_key, test_key, return_indices=True)
    return common, ref_idx, test_idx


def _compute_stats(ref: DatArrays, test: DatArrays) -> CompareStats:
    common_t, t_ref_idx, t_test_idx = _align_indices(ref.time_key, test.time_key)
    common_c, c_ref_idx, c_test_idx = _align_indices(ref.col_key, test.col_key)

    # If nothing overlaps, return "empty" stats (caller should treat as error).
    if common_t.size == 0 or common_c.size == 0:
        return CompareStats(
            abs_max=float("nan"),
            abs_mean=float("nan"),
            abs_std=float("nan"),
            rel_max=float("nan"),
            rel_mean=float("nan"),
            rel_std=float("nan"),
            ref_scale=float("nan"),
            n_time_ref=int(ref.times.size),
            n_time_test=int(test.times.size),
            n_time_common=int(common_t.size),
            n_col_ref=int(ref.col_ids.size),
            n_col_test=int(test.col_ids.size),
            n_col_common=int(common_c.size),
            n_total=0,
            n_finite=0,
            n_nonfinite_ref=0,
            n_nonfinite_test=0,
            n_nonfinite_mismatch=0,
            worst_time=None,
            worst_col_id=None,
            worst_ref=None,
            worst_test=None,
        )

    ref_sel = ref.values[np.ix_(t_ref_idx, c_ref_idx)]
    test_sel = test.values[np.ix_(t_test_idx, c_test_idx)]

    # Compute reference scale (global max abs); ignore non-finite values.
    ref_scale = float(np.nanmax(np.abs(ref_sel))) if ref_sel.size else 0.0
    if not math.isfinite(ref_scale) or ref_scale == 0.0:
        ref_scale = 1.0

    # Non-finite tracking (NaN/Inf in either backend).
    ref_finite = np.isfinite(ref_sel)
    test_finite = np.isfinite(test_sel)
    common_finite = ref_finite & test_finite
    n_total = int(ref_sel.size)
    n_finite = int(np.count_nonzero(common_finite))
    n_nonfinite_ref = int(n_total - np.count_nonzero(ref_finite))
    n_nonfinite_test = int(n_total - np.count_nonzero(test_finite))
    n_nonfinite_mismatch = int(np.count_nonzero(ref_finite ^ test_finite))

    # Use one work array to reduce peak memory: abs_err := abs(test - ref)
    work = np.empty_like(ref_sel)
    np.subtract(test_sel, ref_sel, out=work)
    np.abs(work, out=work)

    if n_finite < n_total:
        # Ignore non-finite locations for numeric statistics.
        work[~common_finite] = np.nan

    abs_max = float(np.nanmax(work)) if n_finite else float("nan")
    abs_mean = float(np.nanmean(work)) if n_finite else float("nan")
    abs_std = float(np.nanstd(work)) if n_finite else float("nan")

    rel_max = abs_max / ref_scale
    rel_mean = abs_mean / ref_scale
    rel_std = abs_std / ref_scale

    worst_time: Optional[float] = None
    worst_col_id: Optional[float] = None
    worst_ref: Optional[float] = None
    worst_test: Optional[float] = None
    if n_finite:
        try:
            flat = int(np.nanargmax(work))
        except ValueError:
            flat = -1
        if flat >= 0:
            i, j = divmod(flat, work.shape[1])
            worst_time = float(ref.times[t_ref_idx[i]])
            worst_col_id = float(ref.col_ids[c_ref_idx[j]]) if ref.col_ids.size else None
            worst_ref = float(ref_sel[i, j])
            worst_test = float(test_sel[i, j])

    return CompareStats(
        abs_max=abs_max,
        abs_mean=abs_mean,
        abs_std=abs_std,
        rel_max=rel_max,
        rel_mean=rel_mean,
        rel_std=rel_std,
        ref_scale=ref_scale,
        n_time_ref=int(ref.times.size),
        n_time_test=int(test.times.size),
        n_time_common=int(common_t.size),
        n_col_ref=int(ref.col_ids.size),
        n_col_test=int(test.col_ids.size),
        n_col_common=int(common_c.size),
        n_total=n_total,
        n_finite=n_finite,
        n_nonfinite_ref=n_nonfinite_ref,
        n_nonfinite_test=n_nonfinite_test,
        n_nonfinite_mismatch=n_nonfinite_mismatch,
        worst_time=worst_time,
        worst_col_id=worst_col_id,
        worst_ref=worst_ref,
        worst_test=worst_test,
    )


def _fmt_num(x: float, *, digits: int = 3) -> str:
    if x is None or not math.isfinite(x):
        return "N/A"
    return f"{x:.{digits}e}"


def _fmt_time(x: Optional[float]) -> str:
    if x is None or not math.isfinite(x):
        return "N/A"
    # Output is minutes; also provide days for easier interpretation.
    return f"{x:.0f} min ({x/1440.0:.2f} d)"


def _fmt_loc(stats: CompareStats) -> str:
    if stats.worst_time is None or stats.worst_col_id is None:
        loc = "N/A"
    else:
        loc = f"t={_fmt_time(stats.worst_time)}, id={stats.worst_col_id:g}"

    notes: list[str] = []
    if stats.n_nonfinite_ref or stats.n_nonfinite_test:
        notes.append(f"nonfinite(ref/test)={stats.n_nonfinite_ref}/{stats.n_nonfinite_test}")
    if stats.n_nonfinite_mismatch:
        notes.append(f"nonfinite_mismatch={stats.n_nonfinite_mismatch}")
    if notes:
        return f"{loc}; " + ", ".join(notes)
    return loc


def _discover_dat_files(dir_path: Path) -> dict[str, Path]:
    if not dir_path.exists():
        return {}
    return {p.name: p for p in sorted(dir_path.glob("*.dat")) if p.is_file()}


def _default_dir(project: str, *, kind: str) -> Path:
    if kind == "cpu":
        preferred = Path("output") / f"{project}_cpu"
        fallback = Path("output") / f"{project}.out"
        return preferred if preferred.exists() else fallback
    return Path("output") / f"{project}_{kind}"


def _render_report(
    *,
    project: str,
    cpu_dir: Path,
    omp_dir: Optional[Path],
    cuda_dir: Optional[Path],
    results: list[FileResult],
    total_cpu_files: int,
    skipped_cpu_files: Sequence[tuple[str, str]] = (),
    tol_omp: float,
    tol_cuda: float,
    plots: Sequence[PlotItem] = (),
) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def status(stats: Optional[CompareStats], tol: float, err: Optional[str]) -> str:
        if err:
            return "ERROR"
        if stats is None:
            return "MISSING"
        if stats.n_nonfinite_mismatch:
            return "ERROR"
        if not math.isfinite(stats.rel_max):
            return "ERROR"
        return "PASS" if stats.rel_max <= tol else "FAIL"

    # Summary (count across all scanned CPU files).
    omp_pass = sum(1 for r in results if status(r.omp, tol_omp, r.omp_error) == "PASS")
    omp_fail = sum(1 for r in results if status(r.omp, tol_omp, r.omp_error) == "FAIL")
    omp_missing = sum(1 for r in results if status(r.omp, tol_omp, r.omp_error) == "MISSING")
    omp_error = sum(1 for r in results if status(r.omp, tol_omp, r.omp_error) == "ERROR")

    cuda_pass = sum(1 for r in results if status(r.cuda, tol_cuda, r.cuda_error) == "PASS")
    cuda_fail = sum(1 for r in results if status(r.cuda, tol_cuda, r.cuda_error) == "FAIL")
    cuda_missing = sum(1 for r in results if status(r.cuda, tol_cuda, r.cuda_error) == "MISSING")
    cuda_error = sum(1 for r in results if status(r.cuda, tol_cuda, r.cuda_error) == "ERROR")

    def worst(
        items: Iterable[FileResult], *, which: str, tol: float
    ) -> Optional[tuple[str, float]]:
        best: Optional[tuple[str, float]] = None
        for r in items:
            st = r.omp if which == "omp" else r.cuda
            err = r.omp_error if which == "omp" else r.cuda_error
            if err or st is None or not math.isfinite(st.rel_max):
                continue
            if best is None or st.rel_max > best[1]:
                best = (r.filename, st.rel_max)
        return best

    worst_omp = worst(results, which="omp", tol=tol_omp)
    worst_cuda = worst(results, which="cuda", tol=tol_cuda)

    lines: list[str] = []
    lines.append("# SHUD Backend Accuracy Comparison Report")
    lines.append("")
    lines.append(f"- Project: `{project}`")
    lines.append(f"- CPU dir: `{cpu_dir.as_posix()}`")
    lines.append(f"- OMP dir: `{omp_dir.as_posix() if omp_dir else '<none>'}`")
    lines.append(f"- CUDA dir: `{cuda_dir.as_posix() if cuda_dir else '<none>'}`")
    lines.append(f"- Generated: `{now}`")
    lines.append("")
    lines.append("## Tolerances (relative to max(|CPU|) per file)")
    lines.append("")
    lines.append(f"- CPU vs OMP: `rel_max <= {tol_omp:g}` (expect ~1e-10 or smaller)")
    lines.append(f"- CPU vs CUDA: `rel_max <= {tol_cuda:g}` (expect ~1e-6 or smaller)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Files scanned: `{total_cpu_files}` (`*.dat` under CPU dir), compared: `{len(results)}`, skipped: `{len(skipped_cpu_files)}`"
    )
    if omp_dir:
        lines.append(
            f"- CPU vs OMP: PASS `{omp_pass}`, FAIL `{omp_fail}`, MISSING `{omp_missing}`, ERROR `{omp_error}`"
        )
        if worst_omp:
            lines.append(f"- Worst CPU vs OMP: `{worst_omp[0]}` (`rel_max={worst_omp[1]:.3e}`)")
    else:
        lines.append("- CPU vs OMP: skipped (OMP dir not provided / not found)")
    if cuda_dir:
        lines.append(
            f"- CPU vs CUDA: PASS `{cuda_pass}`, FAIL `{cuda_fail}`, MISSING `{cuda_missing}`, ERROR `{cuda_error}`"
        )
        if worst_cuda:
            lines.append(f"- Worst CPU vs CUDA: `{worst_cuda[0]}` (`rel_max={worst_cuda[1]:.3e}`)")
    else:
        lines.append("- CPU vs CUDA: skipped (CUDA dir not provided / not found)")
    lines.append("")

    if skipped_cpu_files:
        lines.append("## Skipped files (CPU unreadable)")
        lines.append("")
        for fn, reason in skipped_cpu_files:
            lines.append(f"- `{fn}`: {reason}")
        lines.append("")

    # Key variables
    key_files = [r for r in results if r.var_code in KEY_VARS]
    if key_files:
        lines.append("## Key variables")
        lines.append("")
        lines.append("| Variable | File | CPU vs OMP rel_max | CPU vs CUDA rel_max | Notes |")
        lines.append("|---|---|---:|---:|---|")
        for r in sorted(key_files, key=lambda x: x.var_code):
            omp_rel = _fmt_num(r.omp.rel_max, digits=3) if r.omp else ("ERROR" if r.omp_error else "N/A")
            cuda_rel = _fmt_num(r.cuda.rel_max, digits=3) if r.cuda else ("ERROR" if r.cuda_error else "N/A")
            notes: list[str] = []
            if r.omp:
                notes.append(f"OMP worst@{_fmt_loc(r.omp)}")
            if r.cuda:
                notes.append(f"CUDA worst@{_fmt_loc(r.cuda)}")
            lines.append(
                f"| {KEY_VARS.get(r.var_code, r.var_code)} | `{r.filename}` | {omp_rel} | {cuda_rel} | {'; '.join(notes) if notes else ''} |"
            )
        lines.append("")

    # Full tables
    def render_table(which: str, tol: float) -> None:
        label = "CPU vs OMP" if which == "omp" else "CPU vs CUDA"
        lines.append(f"## {label} (all .dat)")
        lines.append("")
        lines.append("| Type | File | Shape(TxN) | Scale(max|CPU|) | rel_max | rel_mean | rel_std | abs_max | Status | Worst |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")

        def key(r: FileResult) -> tuple[int, float, str]:
            st = r.omp if which == "omp" else r.cuda
            err = r.omp_error if which == "omp" else r.cuda_error
            st_status = status(st, tol, err)
            # Sort: FAIL first, then PASS, then MISSING/ERROR. Within each, by rel_max desc.
            bucket = {"FAIL": 0, "PASS": 1, "MISSING": 2, "ERROR": 3}.get(st_status, 9)
            rel = st.rel_max if (st is not None and math.isfinite(st.rel_max)) else float("-inf")
            return bucket, -rel, r.filename

        for r in sorted(results, key=key):
            st = r.omp if which == "omp" else r.cuda
            err = r.omp_error if which == "omp" else r.cuda_error
            st_status = status(st, tol, err)
            if err:
                lines.append(
                    f"| {r.var_type} | `{r.filename}` | N/A | N/A | N/A | N/A | N/A | N/A | {st_status} | {err} |"
                )
                continue
            if st is None:
                lines.append(
                    f"| {r.var_type} | `{r.filename}` | N/A | N/A | N/A | N/A | N/A | N/A | {st_status} | N/A |"
                )
                continue

            shape = f"{st.n_time_common}x{st.n_col_common}"
            lines.append(
                "| {typ} | `{fn}` | {shape} | {scale} | {rmax} | {rmean} | {rstd} | {amax} | {status} | {worst} |".format(
                    typ=r.var_type,
                    fn=r.filename,
                    shape=shape,
                    scale=_fmt_num(st.ref_scale, digits=3),
                    rmax=_fmt_num(st.rel_max, digits=3),
                    rmean=_fmt_num(st.rel_mean, digits=3),
                    rstd=_fmt_num(st.rel_std, digits=3),
                    amax=_fmt_num(st.abs_max, digits=3),
                    status=st_status,
                    worst=_fmt_loc(st),
                )
            )
        lines.append("")

    if omp_dir:
        render_table("omp", tol_omp)
    if cuda_dir:
        render_table("cuda", tol_cuda)

    if plots:
        lines.append("## Plots")
        lines.append("")
        for p in plots:
            lines.append(f"- {p.label}: `{p.path.as_posix()}`")
        lines.append("")

    # Notes on alignment
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Metrics are computed on the intersection of time steps and column IDs (icol) between CPU and the target backend."
    )
    lines.append(
        "- Relative errors use `max(|CPU|)` per file as the normalization scale (avoids undefined relative errors near zero)."
    )
    lines.append("")
    return "\n".join(lines)


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        return None
    return plt


def _sanitize_filename(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in s)


def _plot_one(
    *,
    cpu_path: Path,
    test_path: Path,
    backend_label: str,
    which: str,
    out_dir: Path,
    stats: CompareStats,
) -> Optional[Path]:
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    cpu = _read_dat_arrays(cpu_path)
    test = _read_dat_arrays(test_path)

    common_t, t_ref_idx, t_test_idx = _align_indices(cpu.time_key, test.time_key)
    common_c, c_ref_idx, c_test_idx = _align_indices(cpu.col_key, test.col_key)
    if common_t.size == 0 or common_c.size == 0:
        return None

    # Choose column to plot: worst column if available, else the first common one.
    col_pos = 0
    if stats.worst_col_id is not None and math.isfinite(stats.worst_col_id):
        target_key = int(np.rint(stats.worst_col_id * 1e6))
        hits = np.nonzero(common_c == target_key)[0]
        if hits.size:
            col_pos = int(hits[0])

    col_ref = int(c_ref_idx[col_pos])
    col_test = int(c_test_idx[col_pos])

    t_min = cpu.times[t_ref_idx]
    t_days = t_min / 1440.0
    ref_series = cpu.values[t_ref_idx, col_ref]
    test_series = test.values[t_test_idx, col_test]
    diff_series = test_series - ref_series

    var_code = _var_code_from_filename(cpu_path.name)
    col_id = float(cpu.col_ids[col_ref]) if cpu.col_ids.size else float("nan")
    id_tag = f"id{int(round(col_id))}" if math.isfinite(col_id) else "idNA"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = _sanitize_filename(f"{which}__{var_code}__{id_tag}.png")
    out_path = out_dir / out_name

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    fig.suptitle(f"{cpu_path.name}: CPU vs {backend_label} ({id_tag})")

    ax1.plot(t_days, ref_series, label="CPU", linewidth=1.2)
    ax1.plot(t_days, test_series, label=backend_label, linewidth=1.2, alpha=0.9)
    ax1.set_ylabel(var_code)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.25)

    ax2.plot(t_days, diff_series, label=f"{backend_label}-CPU", linewidth=1.2)
    ax2.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Diff")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare SHUD outputs across backends (CPU/OMP/CUDA) and emit a Markdown report."
    )
    parser.add_argument("--project", default="ccw", help="Project name (default: ccw)")
    parser.add_argument("--cpu", default="", help="CPU output directory (default: output/<project>_cpu or output/<project>.out)")
    parser.add_argument("--omp", default="", help="OMP output directory (default: output/<project>_omp)")
    parser.add_argument("--cuda", default="", help="CUDA output directory (default: output/<project>_cuda)")
    parser.add_argument(
        "--out",
        default="",
        help="Output Markdown report path (default: post_analysis/accuracy_report_<project>.md)",
    )
    parser.add_argument("--tol-omp", type=float, default=1e-10, help="Tolerance for CPU vs OMP rel_max (default: 1e-10)")
    parser.add_argument("--tol-cuda", type=float, default=1e-6, help="Tolerance for CPU vs CUDA rel_max (default: 1e-6)")
    parser.add_argument("--plot", action="store_true", help="Generate time-series diff plots (requires matplotlib)")
    parser.add_argument(
        "--plot-dir",
        default="",
        help="Directory to write plots (default: post_analysis/figures/accuracy_<project>/)",
    )
    parser.add_argument(
        "--plot-top",
        type=int,
        default=3,
        help="Also plot top-N worst files for each backend (default: 3)",
    )
    args = parser.parse_args()

    project = args.project
    cpu_dir = Path(args.cpu) if args.cpu else _default_dir(project, kind="cpu")
    omp_dir = Path(args.omp) if args.omp else _default_dir(project, kind="omp")
    cuda_dir = Path(args.cuda) if args.cuda else _default_dir(project, kind="cuda")
    report_path = Path(args.out) if args.out else Path("post_analysis") / f"accuracy_report_{project}.md"

    if not cpu_dir.exists():
        print(f"ERROR: CPU dir not found: {cpu_dir}", file=sys.stderr)
        return 2

    cpu_files = _discover_dat_files(cpu_dir)
    if not cpu_files:
        print(f"ERROR: no .dat files found under CPU dir: {cpu_dir}", file=sys.stderr)
        return 2

    omp_files = _discover_dat_files(omp_dir) if omp_dir.exists() else {}
    cuda_files = _discover_dat_files(cuda_dir) if cuda_dir.exists() else {}

    results: list[FileResult] = []
    skipped_cpu_files: list[tuple[str, str]] = []
    for filename, cpu_path in cpu_files.items():
        var_code = _var_code_from_filename(filename)
        var_type = _classify_var(var_code)

        omp_path = omp_files.get(filename)
        cuda_path = cuda_files.get(filename)

        omp_stats: Optional[CompareStats] = None
        cuda_stats: Optional[CompareStats] = None
        omp_err: Optional[str] = None
        cuda_err: Optional[str] = None

        try:
            cpu_arr = _read_dat_arrays(cpu_path)
        except Exception as e:
            skipped_cpu_files.append((filename, str(e)))
            continue

        if omp_path is not None:
            try:
                omp_arr = _read_dat_arrays(omp_path)
                omp_stats = _compute_stats(cpu_arr, omp_arr)
            except Exception as e:
                omp_err = str(e)

        if cuda_path is not None:
            try:
                cuda_arr = _read_dat_arrays(cuda_path)
                cuda_stats = _compute_stats(cpu_arr, cuda_arr)
            except Exception as e:
                cuda_err = str(e)

        results.append(
            FileResult(
                filename=filename,
                var_code=var_code,
                var_type=var_type,
                cpu_path=cpu_path,
                omp_path=omp_path,
                cuda_path=cuda_path,
                omp=omp_stats,
                cuda=cuda_stats,
                omp_error=omp_err,
                cuda_error=cuda_err,
            )
        )

    plots: list[PlotItem] = []
    if args.plot:
        plt = _try_import_matplotlib()
        if plt is None:
            print("WARN: matplotlib not available; skip --plot.", file=sys.stderr)
        else:
            plot_root = (
                Path(args.plot_dir)
                if args.plot_dir
                else (Path("post_analysis") / "figures" / f"accuracy_{project}")
            )

            def top_files(which: str, n: int) -> list[FileResult]:
                items: list[FileResult] = []
                for r in results:
                    st = r.omp if which == "omp" else r.cuda
                    err = r.omp_error if which == "omp" else r.cuda_error
                    if err or st is None or not math.isfinite(st.rel_max):
                        continue
                    items.append(r)
                items.sort(key=lambda x: (x.omp.rel_max if which == "omp" and x.omp else x.cuda.rel_max), reverse=True)  # type: ignore[arg-type]
                return items[: max(0, n)]

            # Always include key variables (if available), plus top-N worst.
            plot_set: set[tuple[str, str]] = set()
            for r in results:
                if r.var_code not in KEY_VARS:
                    continue
                if r.omp is not None and r.omp_path is not None:
                    plot_set.add(("omp", r.filename))
                if r.cuda is not None and r.cuda_path is not None:
                    plot_set.add(("cuda", r.filename))

            top_n = int(args.plot_top)
            if top_n > 0:
                for r in top_files("omp", top_n):
                    if r.omp_path is not None:
                        plot_set.add(("omp", r.filename))
                for r in top_files("cuda", top_n):
                    if r.cuda_path is not None:
                        plot_set.add(("cuda", r.filename))

            for which, filename in sorted(plot_set):
                r = next((x for x in results if x.filename == filename), None)
                if r is None:
                    continue
                if which == "omp":
                    if r.omp_path is None or r.omp is None:
                        continue
                    out_path = _plot_one(
                        cpu_path=r.cpu_path,
                        test_path=r.omp_path,
                        backend_label="OMP",
                        which="cpu_vs_omp",
                        out_dir=plot_root / "cpu_vs_omp",
                        stats=r.omp,
                    )
                else:
                    if r.cuda_path is None or r.cuda is None:
                        continue
                    out_path = _plot_one(
                        cpu_path=r.cpu_path,
                        test_path=r.cuda_path,
                        backend_label="CUDA",
                        which="cpu_vs_cuda",
                        out_dir=plot_root / "cpu_vs_cuda",
                        stats=r.cuda,
                    )
                if out_path is not None:
                    plots.append(PlotItem(label=f"{which}: {filename}", path=out_path))

    md = _render_report(
        project=project,
        cpu_dir=cpu_dir,
        omp_dir=omp_dir if omp_dir.exists() else None,
        cuda_dir=cuda_dir if cuda_dir.exists() else None,
        results=results,
        total_cpu_files=len(cpu_files),
        skipped_cpu_files=skipped_cpu_files,
        tol_omp=float(args.tol_omp),
        tol_cuda=float(args.tol_cuda),
        plots=plots,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
    print(f"[ok] wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
