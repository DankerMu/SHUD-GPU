#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from baseline_core import BaselineError, _add_common_args, generate_golden, parse_cfg, repo_root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate CPU-Serial RHS golden baseline (ccw).")
    _add_common_args(p)
    p.add_argument(
        "--baseline-dir",
        default="validation/baseline/ccw",
        help="output directory for golden data (default: validation/baseline/ccw)",
    )
    p.add_argument(
        "--verify-repeat",
        type=int,
        default=2,
        help="run the same case N times and verify determinism (default: 2)",
    )
    args = p.parse_args(argv)

    cfg = parse_cfg(args)
    baseline_dir = (repo_root() / Path(args.baseline_dir)).resolve()
    try:
        npz_path, meta_path = generate_golden(cfg=cfg, baseline_dir=baseline_dir, verify_repeat=int(args.verify_repeat))
    except BaselineError as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Wrote: {npz_path}")
    print(f"Wrote: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

