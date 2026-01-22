from __future__ import annotations

import struct
import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest import mock

import numpy as np

import baseline_core as bc


def _write_dat(
    path: Path,
    *,
    header_lines: list[str],
    start_time: float,
    col_ids: list[float],
    records: list[tuple[float, list[float]]],
    endianness: str = "<",
) -> None:
    header_text = "\n".join(header_lines).rstrip() + "\n"
    header_raw = header_text.encode("utf-8")
    if len(header_raw) > bc.HEADER_BYTES:
        raise ValueError("header too long for test helper")
    header_raw = header_raw.ljust(bc.HEADER_BYTES, b"\x00")

    num_var = len(col_ids)
    with path.open("wb") as f:
        f.write(header_raw)
        f.write(struct.pack(endianness + "d", float(start_time)))
        f.write(struct.pack(endianness + "d", float(num_var)))
        for x in col_ids:
            f.write(struct.pack(endianness + "d", float(x)))
        for t, vals in records:
            if len(vals) != num_var:
                raise ValueError("record width mismatch")
            f.write(struct.pack(endianness + "d", float(t)))
            for v in vals:
                f.write(struct.pack(endianness + "d", float(v)))


class TestCfgEdits(unittest.TestCase):
    def test_upsert_cfg_kv_replaces_existing_key_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.cfg.para"
            p.write_text("# comment\nEND 10\nDt_YE_SURF 1440\n", encoding="utf-8")
            bc.upsert_cfg_kv(p, "END", "2")
            bc.upsert_cfg_kv(p, "DT_YE_SURF", "60")
            text = p.read_text(encoding="utf-8")
            self.assertIn("END\t2\n", text)
            self.assertIn("DT_YE_SURF\t60\n", text)
            self.assertIn("# comment\n", text)

    def test_upsert_cfg_kv_appends_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.cfg.para"
            p.write_text("A 1\n", encoding="utf-8")
            bc.upsert_cfg_kv(p, "B", "2")
            text = p.read_text(encoding="utf-8")
            self.assertTrue(text.strip().endswith("B\t2"))

    def test_set_forcing_csv_basepath_updates_second_line(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.tsd.forc"
            p.write_text("1 20000101\n./input/x/\nID ...\n", encoding="utf-8")
            bc.set_forcing_csv_basepath(p, "./tmp/x/")
            lines = p.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[1], "./tmp/x/")

    def test_set_forcing_csv_basepath_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "missing.tsd.forc"
            with self.assertRaises(bc.BaselineError):
                bc.set_forcing_csv_basepath(p, "./x/")

    def test_set_forcing_csv_basepath_short_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "short.tsd.forc"
            p.write_text("1 20000101\n", encoding="utf-8")
            with self.assertRaises(bc.BaselineError):
                bc.set_forcing_csv_basepath(p, "./x/")

    def test_write_ccw_project_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ccw.SHUD"
            bc.write_ccw_project_file(p, "./in", "./out")
            text = p.read_text(encoding="utf-8")
            self.assertIn("PRJ\t ccw", text)
            self.assertIn("INPATH\t ./in", text)
            self.assertIn("OUTPATH\t ./out", text)
            self.assertIn("MESH\t ./in/ccw.sp.mesh", text)


class TestFiles(unittest.TestCase):
    def test_ensure_executable_errors_and_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing"
            with self.assertRaises(bc.BaselineError):
                bc.ensure_executable(missing)

            f = Path(td) / "file"
            f.write_text("x", encoding="utf-8")
            f.chmod(0o644)
            with self.assertRaises(bc.BaselineError):
                bc.ensure_executable(f)

            f.chmod(0o755)
            bc.ensure_executable(f)  # should not raise

    def test_copy_tree_errors_and_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            dst = Path(td) / "dst"
            with self.assertRaises(bc.BaselineError):
                bc.copy_tree(src, dst)

            src.mkdir()
            (src / "a.txt").write_text("hello", encoding="utf-8")
            bc.copy_tree(src, dst)
            self.assertTrue((dst / "a.txt").exists())


class TestDatReader(unittest.TestCase):
    def test_read_shud_dat_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ccw.eleysurf.dat"
            _write_dat(
                p,
                header_lines=["# SHUD output"],
                start_time=20000101.0,
                col_ids=[1.0, 2.0, 3.0],
                records=[(0.0, [1.0, 2.0, 3.0]), (60.0, [4.0, 5.0, 6.0])],
            )
            m = bc.read_shud_dat(p, endianness="auto")
            self.assertEqual(m.meta.num_var, 3)
            self.assertEqual(m.meta.num_records, 2)
            np.testing.assert_allclose(m.time_min, np.array([0.0, 60.0]))
            np.testing.assert_allclose(m.values, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_read_shud_dat_big_endian_auto(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ccw.eleysurf.dat"
            _write_dat(
                p,
                header_lines=["# SHUD output"],
                start_time=1.0,
                col_ids=[1.0],
                records=[(0.0, [1.0]), (10.0, [2.0])],
                endianness=">",
            )
            m = bc.read_shud_dat(p, endianness="auto")
            self.assertEqual(m.meta.endianness, ">")
            np.testing.assert_allclose(m.values[:, 0], np.array([1.0, 2.0]))

    def test_read_shud_dat_alignment_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ccw.eleysurf.dat"
            _write_dat(
                p,
                header_lines=["# SHUD output"],
                start_time=1.0,
                col_ids=[1.0],
                records=[(0.0, [1.0])],
            )
            with p.open("ab") as f:
                f.write(b"\x00")  # break alignment
            with self.assertRaises(bc.BaselineError) as ctx:
                _ = bc.read_shud_dat(p)
            self.assertIn("not aligned", str(ctx.exception))

    def test_read_shud_dat_truncated_payload_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ccw.eleysurf.dat"
            _write_dat(
                p,
                header_lines=["# SHUD output"],
                start_time=1.0,
                col_ids=[1.0],
                records=[(0.0, [1.0])],
            )
            real = np.fromfile
            try:
                np.fromfile = lambda *a, **k: real(*a, **k)[:0]  # type: ignore[assignment]
                with self.assertRaises(bc.BaselineError) as ctx:
                    _ = bc.read_shud_dat(p)
                self.assertIn("truncated payload", str(ctx.exception))
            finally:
                np.fromfile = real  # type: ignore[assignment]


class TestExtractRun(unittest.TestCase):
    def test_extract_run_builds_y_concat_and_flux_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            # 2 elements, 1 river, 2 time records
            t = [0.0, 60.0]
            cols_ele = [1.0, 2.0]
            cols_riv = [1.0]
            _write_dat(out / "ccw.eleysurf.dat", header_lines=["#"], start_time=0.0, col_ids=cols_ele, records=[(t[0], [1, 2]), (t[1], [3, 4])])
            _write_dat(out / "ccw.eleyunsat.dat", header_lines=["#"], start_time=0.0, col_ids=cols_ele, records=[(t[0], [5, 6]), (t[1], [7, 8])])
            _write_dat(out / "ccw.eleygw.dat", header_lines=["#"], start_time=0.0, col_ids=cols_ele, records=[(t[0], [9, 10]), (t[1], [11, 12])])
            _write_dat(out / "ccw.rivystage.dat", header_lines=["#"], start_time=0.0, col_ids=cols_riv, records=[(t[0], [13]), (t[1], [14])])

            for name in [
                "ccw.elevinfil.dat",
                "ccw.elevrech.dat",
                "ccw.eleveta.dat",
                "ccw.eleqsurf.dat",
                "ccw.eleqsub.dat",
                "ccw.rivqdown.dat",
                "ccw.rivqup.dat",
                "ccw.rivqsurf.dat",
                "ccw.rivqsub.dat",
            ]:
                if ".riv" in name:
                    _write_dat(out / name, header_lines=["#"], start_time=0.0, col_ids=cols_riv, records=[(t[0], [1.0]), (t[1], [2.0])])
                else:
                    _write_dat(out / name, header_lines=["#"], start_time=0.0, col_ids=cols_ele, records=[(t[0], [1.0, 2.0]), (t[1], [3.0, 4.0])])

            # time csv (tab/space separated)
            (out / "ccw.time.csv").write_text(
                "time_Minutes \t Time_Days \t Task_perc \t CPUTime_s \t WallTime_s \t Num_fcall\n"
                "0 0 0 0 0 10\n"
                "60 0.0416667 10 1 1 20\n",
                encoding="utf-8",
            )

            run = bc.extract_run(out)
            np.testing.assert_allclose(run.time_min, np.array(t))
            # y: [sf(2), us(2), gw(2), riv(1)] => 7 cols
            self.assertEqual(run.y.shape, (2, 7))
            np.testing.assert_allclose(run.y[0, :], np.array([1, 2, 5, 6, 9, 10, 13], dtype=np.float64))
            self.assertIn("elevinfil", run.arrays)
            self.assertIn("rivqdown", run.arrays)
            self.assertIn("Num_fcall", run.cvode)

    def test_extract_run_missing_outputs_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            with self.assertRaises(bc.BaselineError):
                _ = bc.extract_run(out)


class TestBaselineIO(unittest.TestCase):
    def test_write_and_load_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            run = bc.BaselineRun(
                time_min=np.array([0.0, 1.0]),
                y=np.zeros((2, 3)),
                arrays={"y": np.zeros((2, 3)), "elevinfil": np.ones((2, 1))},
                col_ids={},
                cvode={"time_Minutes": np.array([0.0, 1.0]), "Num_fcall": np.array([10.0, 20.0])},
            )
            npz_path, meta_path = bc.write_baseline(output_dir=d, run=run, meta={"schema_version": 1})
            self.assertTrue(npz_path.exists())
            self.assertTrue(meta_path.exists())
            meta, arrays = bc.load_baseline(d)
            self.assertEqual(meta["schema_version"], 1)
            np.testing.assert_allclose(arrays["time_min"], run.time_min)
            np.testing.assert_allclose(arrays["elevinfil"], run.arrays["elevinfil"])

    def test_load_baseline_missing_files_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            with self.assertRaises(bc.BaselineError):
                _ = bc.load_baseline(d)

    def test_compare_runs_success_and_missing_key(self) -> None:
        run = bc.BaselineRun(
            time_min=np.array([0.0]),
            y=np.array([[1.0, 2.0]]),
            arrays={"y": np.array([[1.0, 2.0]]), "elevinfil": np.array([[3.0]])},
            col_ids={},
            cvode={"time_Minutes": np.array([0.0]), "Num_fcall": np.array([10.0])},
        )
        baseline_arrays = {
            "time_min": np.array([0.0]),
            "y": np.array([[1.0, 2.0]]),
            "elevinfil": np.array([[3.0]]),
            "cvode__time_Minutes": np.array([0.0]),
            "cvode__Num_fcall": np.array([10.0]),
        }
        diffs = bc.compare_runs(baseline_arrays=baseline_arrays, run=run, tol=1e-12)
        self.assertTrue(all(d <= 1e-12 for d in diffs.values()))

        with self.assertRaises(bc.BaselineError):
            _ = bc.compare_runs(baseline_arrays={"time_min": run.time_min, "y": run.y}, run=run, tol=1e-12)


class TestRunShud(unittest.TestCase):
    def test_run_shud_invokes_subprocess_and_validates_log(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shud = root / "shud"
            shud.write_text("", encoding="utf-8")
            shud.chmod(0o755)

            proj = root / "ccw.SHUD"
            proj.write_text("PRJ ccw\n", encoding="utf-8")
            out = root / "out"
            out.mkdir()

            def fake_run(*args, **kwargs):
                stdout = kwargs.get("stdout")
                if stdout is not None:
                    stdout.write("openMP: OFF\n")
                    stdout.flush()
                return mock.Mock(returncode=0)

            with mock.patch.object(bc, "repo_root", return_value=root), mock.patch.object(
                bc.subprocess, "run", side_effect=fake_run
            ) as mrun:
                _ = bc.run_shud(shud_bin=shud, project_file=proj, out_dir=out)
                mrun.assert_called_once()

    def test_run_shud_error_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shud = root / "shud"
            shud.write_text("", encoding="utf-8")
            shud.chmod(0o755)
            out = root / "out"
            out.mkdir()

            with mock.patch.object(bc, "repo_root", return_value=root):
                with self.assertRaises(bc.BaselineError):
                    _ = bc.run_shud(shud_bin=shud, project_file=root / "missing.SHUD", out_dir=out)

            proj = root / "ccw.SHUD"
            proj.write_text("PRJ ccw\n", encoding="utf-8")

            with mock.patch.object(bc, "repo_root", return_value=root), mock.patch.object(
                bc.subprocess, "run", return_value=mock.Mock(returncode=2)
            ):
                with self.assertRaises(bc.BaselineError):
                    _ = bc.run_shud(shud_bin=shud, project_file=proj, out_dir=out)

            def fake_run_writes(log_line: str):
                def _f(*args, **kwargs):
                    stdout = kwargs.get("stdout")
                    if stdout is not None:
                        stdout.write(log_line)
                        stdout.flush()
                    return mock.Mock(returncode=0)

                return _f

            with mock.patch.object(bc, "repo_root", return_value=root), mock.patch.object(
                bc.subprocess, "run", side_effect=fake_run_writes("openMP: ON\n")
            ):
                with self.assertRaises(bc.BaselineError):
                    _ = bc.run_shud(shud_bin=shud, project_file=proj, out_dir=out)

            with mock.patch.object(bc, "repo_root", return_value=root), mock.patch.object(
                bc.subprocess, "run", side_effect=fake_run_writes("no openmp marker\n")
            ):
                with self.assertRaises(bc.BaselineError):
                    _ = bc.run_shud(shud_bin=shud, project_file=proj, out_dir=out)


class TestDiff(unittest.TestCase):
    def test_max_abs_diff(self) -> None:
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([1.0, 2.5], dtype=np.float64)
        self.assertAlmostEqual(bc.max_abs_diff(a, b), 0.5)

    def test_max_abs_diff_shape_mismatch_raises(self) -> None:
        with self.assertRaises(bc.BaselineError):
            _ = bc.max_abs_diff(np.zeros((2, 2)), np.zeros((2, 3)))

    def test_max_abs_diff_nan_raises(self) -> None:
        with self.assertRaises(bc.BaselineError):
            _ = bc.max_abs_diff(np.array([1.0, float("nan")]), np.array([1.0, 2.0]))


class TestUtilities(unittest.TestCase):
    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.bin"
            p.write_bytes(b"abc")
            self.assertEqual(
                bc.sha256_file(p),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_git_rev_variants(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(bc.subprocess, "run", side_effect=RuntimeError("boom")):
                self.assertIsNone(bc.git_rev(root))
            with mock.patch.object(bc.subprocess, "run", return_value=mock.Mock(returncode=1, stdout="")):
                self.assertIsNone(bc.git_rev(root))
            with mock.patch.object(bc.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="abc\n")):
                self.assertEqual(bc.git_rev(root), "abc")

    def test_baseline_config_end_minutes(self) -> None:
        cfg = bc.BaselineConfig(end_days=2.0)
        self.assertEqual(cfg.end_minutes, 2880.0)

    def test_build_metadata_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shud = root / "shud"
            shud.write_text("", encoding="utf-8")
            shud.chmod(0o755)
            npz = root / "golden.npz"
            np.savez_compressed(npz, time_min=np.array([0.0]), y=np.array([[0.0]]))
            with mock.patch.object(bc, "git_rev", return_value="deadbeef"), mock.patch.object(
                bc, "sha256_file", return_value="sha"
            ):
                meta = bc.build_metadata(root=root, cfg=bc.BaselineConfig(), shud_bin=shud, npz_path=npz)
            self.assertEqual(meta["repo"]["git_rev"], "deadbeef")
            self.assertEqual(meta["artifacts"]["golden_npz_sha256"], "sha")

    def test_generate_golden_mocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shud = root / "shud"
            shud.write_text("", encoding="utf-8")
            shud.chmod(0o755)
            baseline_dir = root / "baseline"

            run = bc.BaselineRun(
                time_min=np.array([0.0]),
                y=np.array([[1.0, 2.0]]),
                arrays={"y": np.array([[1.0, 2.0]]), "elevinfil": np.array([[3.0]])},
                col_ids={},
                cvode={"time_Minutes": np.array([0.0]), "Num_fcall": np.array([10.0])},
            )

            with mock.patch.object(bc, "repo_root", return_value=root), mock.patch.object(
                bc, "prepare_ccw_case", return_value=(Path("tmp"), Path("out"), Path("proj"))
            ), mock.patch.object(
                bc, "run_shud", return_value=Path("run.log")
            ), mock.patch.object(
                bc, "extract_run", side_effect=[run, run]
            ), mock.patch.object(
                bc, "compare_runs", return_value={}
            ):
                npz_path, meta_path = bc.generate_golden(cfg=bc.BaselineConfig(), baseline_dir=baseline_dir, verify_repeat=2)
            self.assertTrue(npz_path.exists())
            self.assertTrue(meta_path.exists())


class TestCLIs(unittest.TestCase):
    def test_generate_golden_cli_success_and_error(self) -> None:
        import generate_golden as gg_cli

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            npz = root / "golden.npz"
            meta = root / "metadata.json"
            with mock.patch.object(gg_cli, "repo_root", return_value=root), mock.patch.object(
                gg_cli, "generate_golden", return_value=(npz, meta)
            ):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = gg_cli.main(["--baseline-dir", "x"])
                self.assertEqual(rc, 0)
                self.assertIn("Wrote:", buf.getvalue())

            with mock.patch.object(gg_cli, "repo_root", return_value=root), mock.patch.object(
                gg_cli, "generate_golden", side_effect=bc.BaselineError("nope")
            ):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = gg_cli.main(["--baseline-dir", "x"])
                self.assertEqual(rc, 1)
                self.assertIn("ERROR:", buf.getvalue())

    def test_compare_baseline_cli_success_and_error(self) -> None:
        import compare_baseline as cb_cli

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = bc.BaselineRun(
                time_min=np.array([0.0]),
                y=np.array([[1.0]]),
                arrays={"y": np.array([[1.0]])},
                col_ids={},
                cvode={"time_Minutes": np.array([0.0]), "Num_fcall": np.array([10.0])},
            )
            baseline_arrays = {
                "time_min": np.array([0.0]),
                "y": np.array([[1.0]]),
                "cvode__time_Minutes": np.array([0.0]),
                "cvode__Num_fcall": np.array([10.0]),
            }
            diffs = {"time_min": 0.0, "y": 0.0, "cvode__Num_fcall": 0.0}
            with mock.patch.object(cb_cli, "repo_root", return_value=root), mock.patch.object(
                cb_cli, "load_baseline", return_value=({}, baseline_arrays)
            ), mock.patch.object(
                cb_cli, "extract_run", return_value=run
            ), mock.patch.object(
                cb_cli, "compare_runs", return_value=diffs
            ):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = cb_cli.main(["--use-output-dir", "out"])
                self.assertEqual(rc, 0)
                self.assertIn("OK: baseline match", buf.getvalue())

            with mock.patch.object(cb_cli, "repo_root", return_value=root), mock.patch.object(
                cb_cli, "load_baseline", side_effect=bc.BaselineError("no baseline")
            ):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = cb_cli.main(["--use-output-dir", "out"])
                self.assertEqual(rc, 1)
                self.assertIn("ERROR:", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
