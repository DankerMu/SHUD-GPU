#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


Table = Tuple[int, int, List[str], str, List[List[str]]]


def _read_table(fp) -> Table:
    dim_line = fp.readline()
    if not dim_line:
        raise ValueError("unexpected EOF while reading table dimensions")
    dim_tokens = dim_line.strip().split()
    if len(dim_tokens) < 2:
        raise ValueError(f"invalid table dimension line: {dim_line!r}")
    nrow = int(dim_tokens[0])
    ncol = int(dim_tokens[1])
    extras = dim_tokens[2:]
    header = fp.readline()
    if not header:
        raise ValueError("unexpected EOF while reading table header")
    rows: List[List[str]] = []
    for _ in range(nrow):
        line = fp.readline()
        if not line:
            raise ValueError("unexpected EOF while reading table rows")
        parts = line.strip().split()
        if len(parts) < ncol:
            raise ValueError(f"expected {ncol} columns, got {len(parts)}: {line!r}")
        rows.append(parts[:ncol])
    return nrow, ncol, extras, header.rstrip("\n"), rows


def _write_table(
    fp,
    nrow: int,
    ncol: int,
    extras: Sequence[str],
    header: str,
    rows: Iterable[Sequence[str]],
) -> None:
    if extras:
        fp.write(f"{nrow}\t{ncol}\t" + "\t".join(extras) + "\n")
    else:
        fp.write(f"{nrow}\t{ncol}\n")
    fp.write(header.rstrip("\n") + "\n")
    for row in rows:
        fp.write("\t".join(row) + "\n")


def _ensure_empty_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"ERROR: output dir already exists: {path} (use --force to overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _rewrite_cfg_para(text: str, start_day: Optional[float], end_day: Optional[float]) -> str:
    if start_day is None and end_day is None:
        return text

    out_lines: List[str] = []
    for line in text.splitlines(True):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            out_lines.append(line)
            continue

        parts = stripped.split()
        key = parts[0].upper()
        if key == "START" and start_day is not None:
            out_lines.append(f"START {start_day}\n")
            continue
        if key == "END" and end_day is not None:
            out_lines.append(f"END {end_day}\n")
            continue
        out_lines.append(line)
    return "".join(out_lines)


def generate_scaled_case(
    base_project: str,
    out_project: str,
    target_ny: Optional[int],
    factor: Optional[int],
    start_day: Optional[float],
    end_day: Optional[float],
    force: bool,
) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    base_dir = repo_root / "input" / base_project
    out_dir = repo_root / "input" / out_project

    if not base_dir.exists():
        raise SystemExit(f"ERROR: base project dir does not exist: {base_dir}")

    def p(name: str) -> Path:
        return base_dir / f"{base_project}.{name}"

    base_mesh = p("sp.mesh")
    base_att = p("sp.att")
    base_riv = p("sp.riv")
    base_rivseg = p("sp.rivseg")
    base_para = p("cfg.para")
    base_calib = p("cfg.calib")
    base_ic = p("cfg.ic")
    base_geol = p("para.geol")
    base_lc = p("para.lc")
    base_soil = p("para.soil")
    base_tsd_forc = p("tsd.forc")
    base_tsd_lai = p("tsd.lai")
    base_tsd_mf = p("tsd.mf")
    base_forcing_csv = base_dir / "forcing.csv"

    required = [
        base_mesh,
        base_att,
        base_riv,
        base_rivseg,
        base_para,
        base_calib,
        base_ic,
        base_geol,
        base_lc,
        base_soil,
        base_tsd_forc,
        base_tsd_lai,
        base_tsd_mf,
        base_forcing_csv,
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise SystemExit("ERROR: missing required base files:\n  - " + "\n  - ".join(missing))

    # Read base tables.
    with base_mesh.open("r", encoding="utf-8") as fp:
        ele_nrow, ele_ncol, ele_extras, ele_hdr, ele_rows = _read_table(fp)
        node_nrow, node_ncol, node_extras, node_hdr, node_rows = _read_table(fp)

    with base_att.open("r", encoding="utf-8") as fp:
        att_nrow, att_ncol, att_extras, att_hdr, att_rows = _read_table(fp)

    with base_rivseg.open("r", encoding="utf-8") as fp:
        seg_nrow, seg_ncol, seg_extras, seg_hdr, seg_rows = _read_table(fp)

    with base_riv.open("r", encoding="utf-8") as fp:
        riv_nrow, riv_ncol, riv_extras, riv_hdr, riv_rows = _read_table(fp)
        rtype_nrow, rtype_ncol, rtype_extras, rtype_hdr, rtype_rows = _read_table(fp)

    with base_ic.open("r", encoding="utf-8") as fp:
        ic_ele_nrow, ic_ele_ncol, ic_ele_extras, ic_ele_hdr, ic_ele_rows = _read_table(fp)
        ic_riv_nrow, ic_riv_ncol, ic_riv_extras, ic_riv_hdr, ic_riv_rows = _read_table(fp)
        # Lake table is optional; many cases (e.g., ccw) don't have it.
        # This generator currently does not support lake cases.
        rest = fp.read()
        if rest.strip():
            raise SystemExit("ERROR: lake cases are not supported by this generator yet (found extra cfg.ic tables)")

    if ele_nrow != att_nrow or ele_nrow != ic_ele_nrow:
        raise SystemExit(
            "ERROR: base case inconsistency: NumEle mismatch among sp.mesh/sp.att/cfg.ic "
            f"({ele_nrow}/{att_nrow}/{ic_ele_nrow})"
        )
    if riv_nrow != ic_riv_nrow:
        raise SystemExit(
            "ERROR: base case inconsistency: NumRiv mismatch among sp.riv/cfg.ic "
            f"({riv_nrow}/{ic_riv_nrow})"
        )

    base_num_ele = ele_nrow
    base_num_node = node_nrow
    base_num_riv = riv_nrow
    base_num_seg = seg_nrow
    base_num_lake = 0

    base_ny = 3 * base_num_ele + base_num_riv + base_num_lake
    if factor is None:
        if target_ny is None:
            raise SystemExit("ERROR: either --ny or --factor is required")
        if target_ny <= 0:
            raise SystemExit("ERROR: --ny must be > 0")
        factor = int(math.ceil(float(target_ny) / float(base_ny)))
    if factor <= 0:
        raise SystemExit("ERROR: --factor must be > 0")

    out_num_ele = base_num_ele * factor
    out_num_node = base_num_node * factor
    out_num_riv = base_num_riv * factor
    out_num_seg = base_num_seg * factor
    out_ny = 3 * out_num_ele + out_num_riv + base_num_lake

    _ensure_empty_dir(out_dir, force=force)

    # Copy small parameter/time-series files, rewrite cfg.para, and create a forcing list with correct path.
    _write_text(out_dir / f"{out_project}.cfg.para", _rewrite_cfg_para(_read_text(base_para), start_day, end_day))
    shutil.copy2(base_calib, out_dir / f"{out_project}.cfg.calib")
    shutil.copy2(base_geol, out_dir / f"{out_project}.para.geol")
    shutil.copy2(base_lc, out_dir / f"{out_project}.para.lc")
    shutil.copy2(base_soil, out_dir / f"{out_project}.para.soil")
    shutil.copy2(base_tsd_lai, out_dir / f"{out_project}.tsd.lai")
    shutil.copy2(base_tsd_mf, out_dir / f"{out_project}.tsd.mf")
    shutil.copy2(base_forcing_csv, out_dir / "forcing.csv")

    forc_lines = base_tsd_forc.read_text(encoding="utf-8").splitlines(True)
    if len(forc_lines) < 3:
        raise SystemExit(f"ERROR: unexpected tsd.forc format: {base_tsd_forc}")
    forc_lines[1] = f"./input/{out_project}/\n"
    _write_text(out_dir / f"{out_project}.tsd.forc", "".join(forc_lines))

    # Geometry tiling offsets (simple 1D tiling along X).
    xs = [float(r[1]) for r in node_rows]
    x_min = min(xs)
    x_max = max(xs)
    dx = (x_max - x_min) + 10000.0

    # sp.mesh (elements + nodes)
    out_mesh = out_dir / f"{out_project}.sp.mesh"
    with out_mesh.open("w", encoding="utf-8") as fp:
        def gen_ele_rows() -> Iterable[List[str]]:
            for tile in range(factor):
                ele_off = tile * base_num_ele
                node_off = tile * base_num_node
                for r in ele_rows:
                    idx = int(r[0]) + ele_off
                    n1 = int(r[1]) + node_off
                    n2 = int(r[2]) + node_off
                    n3 = int(r[3]) + node_off
                    nb1 = int(r[4])
                    nb2 = int(r[5])
                    nb3 = int(r[6])
                    nb1 = nb1 + ele_off if nb1 > 0 else 0
                    nb2 = nb2 + ele_off if nb2 > 0 else 0
                    nb3 = nb3 + ele_off if nb3 > 0 else 0
                    yield [str(idx), str(n1), str(n2), str(n3), str(nb1), str(nb2), str(nb3), r[7]]

        _write_table(fp, out_num_ele, ele_ncol, ele_extras, ele_hdr, gen_ele_rows())

        def gen_node_rows() -> Iterable[List[str]]:
            for tile in range(factor):
                node_off = tile * base_num_node
                shift_x = tile * dx
                for r in node_rows:
                    idx = int(r[0]) + node_off
                    x = float(r[1]) + shift_x
                    y = float(r[2])
                    yield [str(idx), f"{x:.6f}", f"{y:.6f}", r[3], r[4]]

        _write_table(fp, out_num_node, node_ncol, node_extras, node_hdr, gen_node_rows())

    # sp.att
    out_att = out_dir / f"{out_project}.sp.att"
    with out_att.open("w", encoding="utf-8") as fp:
        def gen_att_rows() -> Iterable[List[str]]:
            for tile in range(factor):
                ele_off = tile * base_num_ele
                for r in att_rows:
                    idx = int(r[0]) + ele_off
                    yield [str(idx)] + r[1:]

        _write_table(fp, out_num_ele, att_ncol, att_extras, att_hdr, gen_att_rows())

    # sp.rivseg
    out_rivseg = out_dir / f"{out_project}.sp.rivseg"
    with out_rivseg.open("w", encoding="utf-8") as fp:
        def gen_seg_rows() -> Iterable[List[str]]:
            for tile in range(factor):
                seg_off = tile * base_num_seg
                ele_off = tile * base_num_ele
                riv_off = tile * base_num_riv
                for r in seg_rows:
                    idx = int(r[0]) + seg_off
                    i_riv = int(r[1]) + riv_off
                    i_ele = int(r[2]) + ele_off
                    yield [str(idx), str(i_riv), str(i_ele), r[3]]

        _write_table(fp, out_num_seg, seg_ncol, seg_extras, seg_hdr, gen_seg_rows())

    # sp.riv (reaches + types)
    out_riv = out_dir / f"{out_project}.sp.riv"
    with out_riv.open("w", encoding="utf-8") as fp:
        def gen_riv_rows() -> Iterable[List[str]]:
            for tile in range(factor):
                riv_off = tile * base_num_riv
                for r in riv_rows:
                    idx = int(r[0]) + riv_off
                    down = int(float(r[1]))
                    down = down + riv_off if down > 0 else down
                    yield [str(idx), str(down), r[2], r[3], r[4], r[5]]

        _write_table(fp, out_num_riv, riv_ncol, riv_extras, riv_hdr, gen_riv_rows())
        _write_table(fp, rtype_nrow, rtype_ncol, rtype_extras, rtype_hdr, rtype_rows)

    # cfg.ic (elements + river stage)
    out_ic = out_dir / f"{out_project}.cfg.ic"
    with out_ic.open("w", encoding="utf-8") as fp:
        def gen_ic_ele_rows() -> Iterable[List[str]]:
            for tile in range(factor):
                ele_off = tile * base_num_ele
                for r in ic_ele_rows:
                    idx = int(r[0]) + ele_off
                    yield [str(idx)] + r[1:]

        _write_table(fp, out_num_ele, ic_ele_ncol, ic_ele_extras, ic_ele_hdr, gen_ic_ele_rows())

        def gen_ic_riv_rows() -> Iterable[List[str]]:
            for tile in range(factor):
                riv_off = tile * base_num_riv
                for r in ic_riv_rows:
                    idx = int(r[0]) + riv_off
                    yield [str(idx)] + r[1:]

        _write_table(fp, out_num_riv, ic_riv_ncol, ic_riv_extras, ic_riv_hdr, gen_ic_riv_rows())

    print(
        "\n".join(
            [
                "Generated scaled case:",
                f"  base: {base_project} (NY={base_ny}, NumEle={base_num_ele}, NumRiv={base_num_riv})",
                f"  out:  {out_project} (factor={factor}, NY={out_ny}, NumEle={out_num_ele}, NumRiv={out_num_riv})",
                f"  dir:  {out_dir}",
            ]
        )
    )
    return out_dir


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="Generate a larger synthetic SHUD case by tiling an existing case.")
    ap.add_argument("--base", default="ccw", help="Base project name under input/ (default: ccw)")
    ap.add_argument("--output", default=None, help="Output project name under input/ (default: derived from --ny/--factor)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--ny", type=int, help="Target NY (unknown count); chooses the smallest integer factor >= target")
    group.add_argument("--factor", type=int, help="Exact tiling factor (integer)")
    ap.add_argument("--start-day", type=float, default=None, help="Override START in cfg.para (days)")
    ap.add_argument("--end-day", type=float, default=None, help="Override END in cfg.para (days)")
    ap.add_argument("--force", action="store_true", help="Overwrite output dir if it exists")
    args = ap.parse_args(argv)

    out_project = args.output
    if out_project is None:
        if args.factor is not None:
            out_project = f"{args.base}_x{args.factor}"
        else:
            out_project = f"{args.base}_ny{args.ny}"
    generate_scaled_case(
        base_project=args.base,
        out_project=out_project,
        target_ny=args.ny,
        factor=args.factor,
        start_day=args.start_day,
        end_day=args.end_day,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
