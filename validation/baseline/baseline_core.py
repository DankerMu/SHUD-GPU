from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

HEADER_BYTES = 1024


class BaselineError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatMeta:
    path: Path
    header_text: str
    start_time: float
    num_var: int
    num_records: int
    dt_min: Optional[float]
    col_ids: np.ndarray
    endianness: Literal["<", ">"]


@dataclass(frozen=True)
class DatMatrix:
    meta: DatMeta
    time_min: np.ndarray  # (T,)
    values: np.ndarray  # (T, N)


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return script_dir().parent.parent


def utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_executable(path: Path) -> None:
    if not path.exists():
        raise BaselineError(f"missing required file: {path}")
    if not os.access(path, os.X_OK):
        raise BaselineError(f"file is not executable: {path}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise BaselineError(f"missing required directory: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _cfg_key_re(key: str) -> re.Pattern[str]:
    # Match leading key token (case-insensitive), allowing arbitrary whitespace after it.
    return re.compile(rf"^(\s*)({re.escape(key)})\b", flags=re.IGNORECASE)


def upsert_cfg_kv(path: Path, key: str, value: str) -> None:
    if not path.exists():
        raise BaselineError(f"missing cfg file: {path}")

    key_pat = _cfg_key_re(key)
    replaced = False
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue

        m = key_pat.match(line)
        if m:
            indent = m.group(1)
            out.append(f"{indent}{key}\t{value}\n")
            replaced = True
        else:
            out.append(line)

    if not replaced:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        out.append(f"{key}\t{value}\n")

    path.write_text("".join(out), encoding="utf-8")


def set_forcing_csv_basepath(forc_list: Path, new_path: str) -> None:
    if not forc_list.exists():
        raise BaselineError(f"missing forcing list file: {forc_list}")

    lines = forc_list.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) < 2:
        raise BaselineError(f"forcing list too short (<2 lines): {forc_list}")
    nl = "\n" if lines[1].endswith("\n") else ""
    lines[1] = f"{new_path}{nl}"
    forc_list.write_text("".join(lines), encoding="utf-8")


def write_ccw_project_file(project_file: Path, in_rel: str, out_rel: str) -> None:
    project_file.write_text(
        "\n".join(
            [
                "PRJ\t ccw",
                f"INPATH\t {in_rel}",
                f"OUTPATH\t {out_rel}",
                f"MESH\t {in_rel}/ccw.sp.mesh",
                f"ATT\t {in_rel}/ccw.sp.att",
                f"RIV\t {in_rel}/ccw.sp.riv",
                f"RIVSEG\t {in_rel}/ccw.sp.rivseg",
                f"CALIB\t {in_rel}/ccw.cfg.calib",
                f"PARA\t {in_rel}/ccw.cfg.para",
                f"INIT\t {in_rel}/ccw.cfg.ic",
                f"LC\t {in_rel}/ccw.para.lc",
                f"SOIL\t {in_rel}/ccw.para.soil",
                f"GEOL\t {in_rel}/ccw.para.geol",
                f"FORC\t {in_rel}/ccw.tsd.forc",
                f"LAI\t {in_rel}/ccw.tsd.lai",
                f"MF\t {in_rel}/ccw.tsd.mf",
                f"RL\t {in_rel}/ccw.tsd.rl",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_shud(*, shud_bin: Path, project_file: Path, out_dir: Path) -> Path:
    ensure_executable(shud_bin)
    if not project_file.exists():
        raise BaselineError(f"missing project file: {project_file}")
    ensure_dir(out_dir)

    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.run(
            [str(shud_bin), "-p", str(project_file)],
            cwd=str(repo_root()),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if p.returncode != 0:
        raise BaselineError(f"shud failed (exit={p.returncode}); see {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "openMP: ON" in text:
        raise BaselineError("baseline must run with OpenMP OFF (serial build); got 'openMP: ON' in run.log")
    if "openMP: OFF" not in text:
        raise BaselineError("could not confirm OpenMP is OFF; expected 'openMP: OFF' in run.log")

    return log_path


def _validate_dat_shape(*, path: Path, size: int, num_var: int) -> tuple[int, int, int]:
    if num_var < 0:
        raise BaselineError(f"invalid NumVar={num_var} in {path}")
    data_offset = HEADER_BYTES + 16 + 8 * num_var
    if size < data_offset:
        raise BaselineError(f"file too small for header+metadata: {path} (size={size}, need>={data_offset})")
    record_size = 8 * (1 + num_var)
    rem = size - data_offset
    if rem % record_size != 0:
        raise BaselineError(
            f"data section not aligned in {path} (size={size}, data_offset={data_offset}, record_size={record_size})"
        )
    num_records = rem // record_size
    return data_offset, record_size, num_records


def _read_meta_for_endianness(path: Path, endianness: Literal["<", ">"]) -> DatMeta:
    if not path.exists():
        raise BaselineError(f"missing .dat file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise BaselineError(f"empty .dat file: {path}")
    if size < HEADER_BYTES:
        raise BaselineError(f"short header (<{HEADER_BYTES}B) in {path}")

    with path.open("rb") as f:
        header_raw = f.read(HEADER_BYTES)
        header_text = header_raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()
        start_time = struct.unpack(endianness + "d", f.read(8))[0]
        num_var_raw = struct.unpack(endianness + "d", f.read(8))[0]
        if not math.isfinite(num_var_raw):
            raise BaselineError(f"invalid NumVar={num_var_raw!r} in {path}")
        num_var = int(num_var_raw)
        if abs(num_var_raw - num_var) > 1e-9:
            raise BaselineError(f"NumVar is not an integer in {path}: {num_var_raw!r}")

        data_offset, record_size, num_records = _validate_dat_shape(path=path, size=size, num_var=num_var)

        col_ids_raw = f.read(8 * num_var)
        if len(col_ids_raw) != 8 * num_var:
            raise BaselineError(f"truncated icol array in {path}")
        col_ids = np.frombuffer(col_ids_raw, dtype=endianness + "f8").copy()

        dt_min: Optional[float] = None
        if num_records >= 2:
            f.seek(data_offset, os.SEEK_SET)
            rec1 = f.read(record_size)
            rec2 = f.read(record_size)
            if len(rec1) != record_size or len(rec2) != record_size:
                raise BaselineError(f"truncated records in {path}")
            t0 = struct.unpack_from(endianness + "d", rec1, 0)[0]
            t1 = struct.unpack_from(endianness + "d", rec2, 0)[0]
            dt_min = float(t1 - t0)
            if not math.isfinite(dt_min) or dt_min <= 0:
                raise BaselineError(f"non-positive dt derived from first two records in {path}: {dt_min!r}")

    return DatMeta(
        path=path,
        header_text=header_text,
        start_time=float(start_time),
        num_var=int(num_var),
        num_records=int(num_records),
        dt_min=dt_min,
        col_ids=col_ids,
        endianness=endianness,
    )


def read_shud_dat(path: Path, *, endianness: Literal["auto", "<", ">"] = "auto") -> DatMatrix:
    if endianness == "auto":
        errors: list[tuple[str, Exception]] = []
        for fmt in ("<", ">"):
            try:
                meta = _read_meta_for_endianness(path, fmt)  # type: ignore[arg-type]
                break
            except Exception as exc:
                errors.append((fmt, exc))
        else:
            msg = "; ".join(f"{fmt}: {exc}" for fmt, exc in errors)
            raise BaselineError(f"failed to parse .dat file in either endian for {path}: {msg}") from errors[0][1]
    else:
        meta = _read_meta_for_endianness(path, endianness)

    data_offset = HEADER_BYTES + 16 + 8 * meta.num_var
    total_doubles = meta.num_records * (1 + meta.num_var)
    with meta.path.open("rb") as f:
        f.seek(data_offset, os.SEEK_SET)
        payload = np.fromfile(f, dtype=meta.endianness + "f8", count=total_doubles)
    if payload.size != total_doubles:
        raise BaselineError(f"truncated payload in {meta.path} (need {total_doubles} doubles, got {payload.size})")

    mat = payload.reshape((meta.num_records, 1 + meta.num_var))
    time_min = mat[:, 0].astype(np.float64, copy=True)
    values = mat[:, 1:].astype(np.float64, copy=True)
    return DatMatrix(meta=meta, time_min=time_min, values=values)


def read_time_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise BaselineError(f"missing time csv: {path}")

    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.lower().startswith("time_minutes"):
            continue
        parts = re.split(r"\s+", s)
        if len(parts) < 6:
            raise BaselineError(f"unexpected time.csv row format in {path}: {line!r}")
        try:
            row = [float(x) for x in parts[:6]]
        except ValueError as exc:
            raise BaselineError(f"invalid number in time csv {path}: {line!r}") from exc
        if not all(math.isfinite(x) for x in row):
            raise BaselineError(f"non-finite number in time csv {path}: {line!r}")
        rows.append(row)

    if not rows:
        raise BaselineError(f"no rows parsed from time csv: {path}")

    a = np.array(rows, dtype=np.float64)
    return {
        "time_Minutes": a[:, 0].copy(),
        "Time_Days": a[:, 1].copy(),
        "Task_perc": a[:, 2].copy(),
        "Num_fcall": a[:, 5].copy(),
    }


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise BaselineError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    d = np.max(np.abs(a - b))
    if not np.isfinite(d):
        raise BaselineError("non-finite diff (NaN/Inf) detected")
    return float(d)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev(root: Path) -> Optional[str]:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip() or None


@dataclass(frozen=True)
class BaselineConfig:
    case: str = "ccw"
    end_days: float = 2.0
    dt_min: int = 60
    terrain_radiation: int = 0

    @property
    def end_minutes(self) -> float:
        return float(self.end_days) * 1440.0


def _apply_ccw_overrides(cfg_path: Path, cfg: BaselineConfig) -> None:
    upsert_cfg_kv(cfg_path, "END", f"{cfg.end_days:g}")
    upsert_cfg_kv(cfg_path, "SCR_INTV", f"{cfg.dt_min:d}")

    # Ensure deterministic binary output and a single output cadence across key channels.
    upsert_cfg_kv(cfg_path, "ASCII_OUTPUT", "0")
    upsert_cfg_kv(cfg_path, "BINARY_OUTPUT", "1")
    upsert_cfg_kv(cfg_path, "TERRAIN_RADIATION", str(int(cfg.terrain_radiation)))

    dt_keys = [
        "DT_YE_SNOW",
        "DT_YE_SURF",
        "DT_YE_UNSAT",
        "DT_YE_GW",
        "DT_QE_SURF",
        "DT_QE_SUB",
        "DT_QE_ET",
        "DT_QE_PRCP",
        "DT_QE_INFIL",
        "DT_QE_RECH",
        "DT_YR_STAGE",
        "DT_QR_DOWN",
        "DT_QR_SURF",
        "DT_QR_SUB",
        "DT_QR_UP",
    ]
    for k in dt_keys:
        upsert_cfg_kv(cfg_path, k, str(int(cfg.dt_min)))


def prepare_ccw_case(
    *,
    root: Path,
    cfg: BaselineConfig,
    tmp_dir: Path,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    input_src = root / "input" / cfg.case
    ensure_dir(tmp_dir.parent)
    ensure_dir(out_dir.parent)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    copy_tree(input_src, tmp_dir)

    cfg_path = tmp_dir / "ccw.cfg.para"
    forc_path = tmp_dir / "ccw.tsd.forc"
    _apply_ccw_overrides(cfg_path, cfg)

    tmp_rel = f"./{tmp_dir.relative_to(root).as_posix()}"
    set_forcing_csv_basepath(forc_path, tmp_rel + "/")

    out_rel = f"./{out_dir.relative_to(root).as_posix()}"
    project_file = tmp_dir / "ccw.SHUD"
    write_ccw_project_file(project_file, tmp_rel, out_rel)

    return tmp_dir, out_dir, project_file


def _required_outputs() -> dict[str, str]:
    return {
        # Global Y components (used to build y vector).
        "eleysurf": "ccw.eleysurf.dat",
        "eleyunsat": "ccw.eleyunsat.dat",
        "eleygw": "ccw.eleygw.dat",
        "rivystage": "ccw.rivystage.dat",
        # Key flux arrays (RHS-related).
        "elevinfil": "ccw.elevinfil.dat",
        "elevrech": "ccw.elevrech.dat",
        "eleveta": "ccw.eleveta.dat",
        "eleqsurf": "ccw.eleqsurf.dat",
        "eleqsub": "ccw.eleqsub.dat",
        "rivqdown": "ccw.rivqdown.dat",
        "rivqup": "ccw.rivqup.dat",
        "rivqsurf": "ccw.rivqsurf.dat",
        "rivqsub": "ccw.rivqsub.dat",
    }


@dataclass(frozen=True)
class BaselineRun:
    time_min: np.ndarray
    y: np.ndarray
    arrays: dict[str, np.ndarray]
    col_ids: dict[str, np.ndarray]
    cvode: dict[str, np.ndarray]


def extract_run(out_dir: Path) -> BaselineRun:
    required = _required_outputs()
    missing = [name for name in required.values() if not (out_dir / name).exists()]
    if missing:
        raise BaselineError(f"missing expected output files in {out_dir}: {missing}")

    mats: dict[str, DatMatrix] = {k: read_shud_dat(out_dir / v) for k, v in required.items()}

    ref_t = mats["eleysurf"].time_min
    for k, m in mats.items():
        if not np.array_equal(m.time_min, ref_t):
            raise BaselineError(f"time axis mismatch for {k}: {m.meta.path}")

    y_sf = mats["eleysurf"].values
    y_us = mats["eleyunsat"].values
    y_gw = mats["eleygw"].values
    y_riv = mats["rivystage"].values
    y = np.concatenate([y_sf, y_us, y_gw, y_riv], axis=1)

    arrays: dict[str, np.ndarray] = {"y": y}
    col_ids: dict[str, np.ndarray] = {"y_sf": mats["eleysurf"].meta.col_ids, "y_riv": mats["rivystage"].meta.col_ids}

    for k in sorted(required.keys()):
        if k in {"eleysurf", "eleyunsat", "eleygw", "rivystage"}:
            continue
        arrays[k] = mats[k].values
        col_ids[k] = mats[k].meta.col_ids

    cvode = read_time_csv(out_dir / "ccw.time.csv")

    return BaselineRun(
        time_min=ref_t.copy(),
        y=y.copy(),
        arrays=arrays,
        col_ids=col_ids,
        cvode=cvode,
    )


def write_baseline(
    *,
    output_dir: Path,
    run: BaselineRun,
    meta: dict[str, Any],
) -> tuple[Path, Path]:
    ensure_dir(output_dir)
    npz_path = output_dir / "golden.npz"
    meta_path = output_dir / "metadata.json"

    npz_payload: dict[str, np.ndarray] = {"time_min": run.time_min, "y": run.y}
    for k, v in run.arrays.items():
        if k in {"y"}:
            continue
        npz_payload[k] = v
    for k, v in run.cvode.items():
        npz_payload[f"cvode__{k}"] = v

    np.savez_compressed(npz_path, **npz_payload)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return npz_path, meta_path


def load_baseline(dir_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    meta_path = dir_path / "metadata.json"
    npz_path = dir_path / "golden.npz"
    if not meta_path.exists() or not npz_path.exists():
        raise BaselineError(f"baseline missing metadata.json or golden.npz in {dir_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as z:
        for k in z.files:
            arrays[k] = z[k]
    return meta, arrays


def compare_runs(
    *,
    baseline_arrays: dict[str, np.ndarray],
    run: BaselineRun,
    tol: float,
) -> dict[str, float]:
    diffs: dict[str, float] = {}
    if "time_min" not in baseline_arrays:
        raise BaselineError("baseline missing array: time_min")
    if "y" not in baseline_arrays:
        raise BaselineError("baseline missing array: y")
    diffs["time_min"] = max_abs_diff(baseline_arrays["time_min"], run.time_min)
    diffs["y"] = max_abs_diff(baseline_arrays["y"], run.y)

    for k in sorted(run.arrays.keys()):
        if k == "y":
            continue
        if k not in baseline_arrays:
            raise BaselineError(f"baseline missing array: {k}")
        diffs[k] = max_abs_diff(baseline_arrays[k], run.arrays[k])

    for k, v in run.cvode.items():
        bk = f"cvode__{k}"
        if bk not in baseline_arrays:
            raise BaselineError(f"baseline missing cvode array: {bk}")
        diffs[bk] = max_abs_diff(baseline_arrays[bk], v)

    bad = {k: d for k, d in diffs.items() if d > tol}
    if bad:
        worst = max(bad.items(), key=lambda kv: kv[1])
        raise BaselineError(f"baseline mismatch (tol={tol:g}). worst={worst[0]} diff={worst[1]:.3e}")
    return diffs


def build_metadata(*, root: Path, cfg: BaselineConfig, shud_bin: Path, npz_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_utc": utc_iso(),
        "case": cfg.case,
        "config": {
            "end_days": cfg.end_days,
            "dt_min": cfg.dt_min,
            "terrain_radiation": cfg.terrain_radiation,
        },
        "tolerance": 1e-12,
        "required_outputs": _required_outputs(),
        "repo": {
            "git_rev": git_rev(root),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "artifacts": {
            "golden_npz": str(npz_path.relative_to(root)),
            "golden_npz_sha256": sha256_file(npz_path),
            "shud_bin": str(shud_bin.relative_to(root)) if shud_bin.is_relative_to(root) else str(shud_bin),
        },
    }


def generate_golden(
    *,
    cfg: BaselineConfig,
    baseline_dir: Path,
    verify_repeat: int = 2,
) -> tuple[Path, Path]:
    root = repo_root()
    shud_bin = root / "shud"
    ensure_executable(shud_bin)

    tmp_dir = root / "validation" / "baseline" / "tmp" / f"{cfg.case}.input"
    out_dir = root / "validation" / "baseline" / "tmp" / f"{cfg.case}.output"
    prepare_ccw_case(root=root, cfg=cfg, tmp_dir=tmp_dir, out_dir=out_dir)

    project_file = tmp_dir / "ccw.SHUD"
    run_shud(shud_bin=shud_bin, project_file=project_file, out_dir=out_dir)
    run1 = extract_run(out_dir)

    if verify_repeat >= 2:
        # Run a second time into a fresh output directory to catch any nondeterminism.
        out_dir2 = root / "validation" / "baseline" / "tmp" / f"{cfg.case}.output.repeat"
        if out_dir2.exists():
            shutil.rmtree(out_dir2)
        prepare_ccw_case(root=root, cfg=cfg, tmp_dir=tmp_dir, out_dir=out_dir2)
        run_shud(shud_bin=shud_bin, project_file=project_file, out_dir=out_dir2)
        run2 = extract_run(out_dir2)

        tol = 1e-12
        baseline_arrays = {"time_min": run1.time_min, "y": run1.y}
        baseline_arrays.update({k: v for k, v in run1.arrays.items() if k != "y"})
        baseline_arrays.update({f"cvode__{k}": v for k, v in run1.cvode.items()})
        _ = compare_runs(baseline_arrays=baseline_arrays, run=run2, tol=tol)

    ensure_dir(baseline_dir)
    meta_stub: dict[str, Any] = {}
    npz_path, meta_path = write_baseline(output_dir=baseline_dir, run=run1, meta=meta_stub)

    meta = build_metadata(root=root, cfg=cfg, shud_bin=shud_bin, npz_path=npz_path)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return npz_path, meta_path


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--case", default="ccw", choices=["ccw"], help="fixed input project (default: ccw)")
    p.add_argument("--end-days", type=float, default=2.0, help="simulation END in days (default: 2)")
    p.add_argument("--dt-min", type=int, default=60, help="output interval in minutes (default: 60)")
    p.add_argument(
        "--terrain-radiation",
        type=int,
        default=0,
        choices=[0, 1],
        help="set TERRAIN_RADIATION in cfg.para (default: 0)",
    )


def parse_cfg(args: argparse.Namespace) -> BaselineConfig:
    return BaselineConfig(
        case=str(args.case),
        end_days=float(args.end_days),
        dt_min=int(args.dt_min),
        terrain_radiation=int(args.terrain_radiation),
    )
