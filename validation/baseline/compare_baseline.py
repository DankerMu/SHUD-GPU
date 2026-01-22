#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from baseline_core import (
    BaselineError,
    _add_common_args,
    compare_runs,
    extract_run,
    load_baseline,
    parse_cfg,
    prepare_ccw_case,
    repo_root,
    run_shud,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare current CPU-Serial run against golden baseline.")
    _add_common_args(p)
    p.add_argument(
        "--baseline-dir",
        default="validation/baseline/ccw",
        help="baseline directory containing metadata.json + golden.npz (default: validation/baseline/ccw)",
    )
    p.add_argument("--tol", type=float, default=1e-12, help="absolute tolerance (default: 1e-12)")
    p.add_argument(
        "--use-output-dir",
        default="",
        help="skip running shud; compare an existing output directory instead",
    )
    args = p.parse_args(argv)

    root = repo_root()
    cfg = parse_cfg(args)
    baseline_dir = (root / Path(args.baseline_dir)).resolve()

    try:
        _meta, baseline_arrays = load_baseline(baseline_dir)

        if args.use_output_dir:
            out_dir = (root / Path(args.use_output_dir)).resolve()
        else:
            tmp_dir = root / "validation" / "baseline" / "tmp" / f"{cfg.case}.input.compare"
            out_dir = root / "validation" / "baseline" / "tmp" / f"{cfg.case}.output.compare"
            prepare_ccw_case(root=root, cfg=cfg, tmp_dir=tmp_dir, out_dir=out_dir)
            shud_bin = root / "shud"
            run_shud(shud_bin=shud_bin, project_file=tmp_dir / "ccw.SHUD", out_dir=out_dir)

        run = extract_run(out_dir)
        diffs = compare_runs(baseline_arrays=baseline_arrays, run=run, tol=float(args.tol))
    except BaselineError as e:
        print(f"ERROR: {e}")
        return 1

    # Compact summary: only print worst few diffs.
    worst = sorted(diffs.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print("OK: baseline match")
    for k, d in worst:
        print(f"- {k}: {d:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

