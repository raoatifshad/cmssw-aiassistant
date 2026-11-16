import gzip
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent))
import ib_test_log_problems as problems
import ib_test_log_scanner as scanner
from test_ib_test_log_scanner import DICTIONARY_NOT_FOUND, unit_log

BUILD = "el8_amd64_gcc13/www/fri/16.1-fri-11/CMSSW_16_1_X_2026-11-13-1100"


def _sorted(result):
    """Zip order and folder-walk order differ; the content must not."""
    key = lambda d: json.dumps(d, sort_keys=True, default=str)
    return dict(result, tests=sorted(result["tests"], key=key), problems=sorted(result["problems"], key=key))


def _unpack(zip_path: Path, folder: Path) -> None:
    """What es_ibs_log.py does: cd <build>; mkdir UT; cd UT; unzip ../<zip>."""
    folder.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(folder)


class DescribeZipPathTests(unittest.TestCase):
    def test_reads_the_build_from_the_server_path(self):
        info = problems.describe_zip_path(f"/data/sdt/buildlogs/{BUILD}/unitTestLogs.zip")

        self.assertEqual(info, dict(kind="unit", arch="el8_amd64_gcc13", tag="CMSSW_16_1_X_2026-11-13-1100",
                                    variant="", queue="16.1-fri-11", path=f"{BUILD}/unitTestLogs.zip"))

    def test_gpu_and_other_variants(self):
        gpu = problems.describe_zip_path(f"/data/sdt/buildlogs/{BUILD}/gpu/nvidia_h100/unitTestLogs.zip")
        other = problems.describe_zip_path(f"/x/{BUILD}/other/rntuple/addOnTests.zip")

        self.assertEqual((gpu["kind"], gpu["variant"]), ("unit", "gpu/nvidia_h100"))
        self.assertEqual((other["kind"], other["variant"]), ("addon", "other/rntuple"))

    def test_refuses_paths_it_cannot_place(self):
        for bad in ("/tmp/unitTestLogs.zip",                                   # no build folder
                    f"/data/sdt/buildlogs/{BUILD}/runTheMatrix.zip",           # not a test-log zip
                    f"/data/sdt/buildlogs/{BUILD}/odd/unitTestLogs.zip",       # unknown level below the tag
                    f"/data/sdt/buildlogs/{BUILD}/gpu/a/b/unitTestLogs.zip",
                    "/data/sdt/buildlogs/arch/www/fri/q/NOT_A_TAG/unitTestLogs.zip"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                problems.describe_zip_path(bad)

    def test_relval_zip_path(self):
        info = problems.describe_zip_path(f"/data/sdt/buildlogs/{BUILD}/gpu/nvidia_h100/pyRelValMatrixLogs.zip")
        self.assertEqual((info["kind"], info["variant"]), ("relval", "gpu/nvidia_h100"))

    def test_scan_file_sits_next_to_the_zip(self):
        self.assertEqual(problems.scan_file_path("/b/x/unitTestLogs.zip"), "/b/x/unitTestLogs-scan.json.gz")
        self.assertEqual(problems.scan_file_path("/b/x/addOnTests.zip"), "/b/x/addOnTests-scan.json.gz")


class ScanDirTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.build = Path(self.tmp.name) / "buildlogs" / BUILD
        self.build.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _unit_zip(self):
        zip_path = self.build / "unitTestLogs.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("unitTestLogs/Alignment/OfflineValidation/unitTest.log",
                        "".join(unit_log(("SubmitPVrbr", DICTIONARY_NOT_FOUND, "succeeded"),
                                         ("PrimaryVertex", "x\n", "had ERRORS"))))
            zf.writestr("unitTestLogs/Alignment/OfflineValidation/unitTest.log-read_config", "{}")
            zf.writestr("unitTestLogs/FWCore/Utilities/unitTest.log",
                        "".join(unit_log(("TestA", "%MSG-e Cat: mod\nbad\n%MSG\n", "succeeded"))))
        return zip_path

    def test_unit_folder_scan_matches_the_zip_scan(self):
        zip_path = self._unit_zip()
        _unpack(zip_path, self.build / "UT")

        out = problems.scan_unit_test_dir(str(self.build / "UT"), str(zip_path))

        self.assertEqual(out, str(self.build / "unitTestLogs-scan.json.gz"))
        with gzip.open(out, "rt") as fh:
            written = json.load(fh)
        expected = scanner.scan_unit_test_zip(str(zip_path))
        expected["problems"] = [problems._trim(p) for p in expected["problems"]]
        self.assertEqual(_sorted(dict(written, source=None)), _sorted(dict(expected, source=None)))
        self.assertEqual(len(written["tests"]), 3)
        src = written["source"]
        self.assertEqual((src["kind"], src["arch"], src["tag"], src["variant"]),
                         ("unit", "el8_amd64_gcc13", "CMSSW_16_1_X_2026-11-13-1100", ""))
        self.assertEqual(src["size"], os.path.getsize(zip_path))
        self.assertEqual(src["scanner_version"], scanner.SCANNER_VERSION)

    def test_addon_folder_scan_matches_the_zip_scan(self):
        zip_path = self.build / "addOnTests.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("cmsDriver-hlt_mc_GRun_step2.log", DICTIONARY_NOT_FOUND)
            zf.writestr("cmsDriver-hlt_mc_GRun_step2.log-read_config", "{}")
            zf.writestr("cmsDriver-fastsim_step1.log", "fine\n")
        status_text = ("[hlt_mc_GRun:2] cmsRun x.py : PASSED - elapsed time 1s\n"
                       "[fastsim:1] cmsDriver.py y : FAILED - elapsed time 3s\n")
        (self.build / "addOnTests.log").write_text(status_text)
        _unpack(zip_path, self.build / "AO")

        out = problems.scan_addon_dir(str(self.build / "AO"), str(zip_path))

        with gzip.open(out, "rt") as fh:
            written = json.load(fh)
        expected = scanner.scan_addon_zip(str(zip_path), status_text)
        self.assertEqual(_sorted(dict(written, source=None)), _sorted(dict(expected, source=None)))
        self.assertEqual(sorted((t["test"], t["status"]) for t in written["tests"]),
                         [("fastsim", "failed"), ("hlt_mc_GRun", "passed")])
        self.assertEqual(written["source"]["kind"], "addon")

    def test_writes_nothing_inside_the_unpacked_folder_and_leaves_no_tmp(self):
        zip_path = self._unit_zip()
        _unpack(zip_path, self.build / "UT")
        before = sorted(p.name for p in (self.build / "UT").rglob("*"))

        problems.scan_unit_test_dir(str(self.build / "UT"), str(zip_path))

        self.assertEqual(sorted(p.name for p in (self.build / "UT").rglob("*")), before)
        self.assertEqual(sorted(p.name for p in self.build.iterdir()),
                         ["UT", "unitTestLogs-scan.json.gz", "unitTestLogs.zip"])

    def test_running_twice_rewrites_the_same_file(self):
        zip_path = self._unit_zip()
        _unpack(zip_path, self.build / "UT")

        first = problems.scan_unit_test_dir(str(self.build / "UT"), str(zip_path))
        second = problems.scan_unit_test_dir(str(self.build / "UT"), str(zip_path))

        self.assertEqual(first, second)
        self.assertEqual(len(list(self.build.glob("*-scan.json.gz*"))), 1)

    def test_empty_unpacked_zip_gives_an_empty_result(self):
        zip_path = self.build / "gpu" / "amd_mi300x" / "unitTestLogs.zip"
        zip_path.parent.mkdir(parents=True)
        zipfile.ZipFile(zip_path, "w").close()
        _unpack(zip_path, zip_path.parent / "UT")

        out = problems.scan_unit_test_dir(str(zip_path.parent / "UT"), str(zip_path))

        with gzip.open(out, "rt") as fh:
            written = json.load(fh)
        self.assertEqual((written["tests"], written["problems"]), ([], []))
        self.assertEqual(written["source"]["variant"], "gpu/amd_mi300x")

    def test_bad_path_or_missing_folder_writes_nothing(self):
        zip_path = self._unit_zip()
        with self.assertRaises(ValueError):
            problems.scan_unit_test_dir(str(self.build / "UT"), str(zip_path))   # not unpacked
        with self.assertRaises(ValueError):
            problems.scan_addon_dir(str(self.build), str(zip_path))              # wrong kind
        self.assertEqual([p.name for p in self.build.iterdir()], ["unitTestLogs.zip"])

    def test_missing_addon_status_log_raises(self):
        zip_path = self.build / "addOnTests.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("cmsDriver-x_step1.log", "fine\n")
        _unpack(zip_path, self.build / "AO")

        with self.assertRaises(FileNotFoundError):
            problems.scan_addon_dir(str(self.build / "AO"), str(zip_path))
        self.assertFalse(list(self.build.glob("*-scan.json.gz*")))


def _relval_files(add):
    """The same RelVal content, written by `add(name, text)` into a zip or a folder."""
    wf = "140.051_RunMuonEG2022C"
    add(f"{wf}/workflow.log", f"{wf} Step0-PASSED Step1-FAILED ; exit: 0 256\n")
    add(f"{wf}/step1_RunMuonEG2022C.log", DICTIONARY_NOT_FOUND)
    add(f"{wf}/step1.log", DICTIONARY_NOT_FOUND)                          # identical copy: read once
    add(f"{wf}/step2_RunMuonEG2022C.log", "%MSG-e Cat: mod\nbad thing\n%MSG\n")
    add(f"{wf}/step2.log", "%MSG-e Cat: mod\nbad thinG\n%MSG\n")         # same size, other content: read too
    add(f"{wf}/step1_RunMuonEG2022C.log-read_config", "{}")
    add("11634.0_TTbar/workflow.log", "11634.0_TTbar Step0-PASSED ; exit: 0\n")
    add("11634.0_TTbar/step1_TTbar.log", "----- Begin Fatal Exception 01-Jan-2026\nboom\n----- End Fatal Exception\n")


class RelvalDirTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.build = Path(self.tmp.name) / "buildlogs" / BUILD
        self.partial = self.build / "pyRelValPartialLogs"
        self.partial.mkdir(parents=True)

        def add(name, text):
            path = self.partial / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
        _relval_files(add)

    def tearDown(self):
        self.tmp.cleanup()

    def test_folder_scan_matches_the_zip_scan(self):
        zip_path = Path(self.tmp.name) / "same.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            _relval_files(zf.writestr)
        (self.partial / "done.1of2").touch()                   # top-level markers and deeper
        (self.partial / "140.051_RunMuonEG2022C" / "sub").mkdir()  # folders are not logs
        (self.partial / "140.051_RunMuonEG2022C" / "sub" / "x.log").write_text(DICTIONARY_NOT_FOUND)

        from_dir = scanner.scan_relval_dir(str(self.partial))
        from_zip = scanner.scan_relval_zip(str(zip_path))

        self.assertEqual(_sorted(dict(from_dir, tests=[])), _sorted(dict(from_zip, tests=[])))
        self.assertEqual(from_dir["workflows"], from_zip["workflows"])
        members = sorted(p["member"] for p in from_dir["problems"])
        self.assertNotIn("140.051_RunMuonEG2022C/step1.log", members)      # the copy was skipped
        self.assertIn("140.051_RunMuonEG2022C/step2.log", members)         # different content: kept

    def test_writes_the_scan_file_without_a_zip_and_counts_jobs(self):
        for name in ("done.1of2", "done.2of2"):
            (self.partial / name).touch()

        out = problems.scan_relval_dir(str(self.partial))

        self.assertEqual(out, str(self.build / "pyRelValMatrixLogs-scan.json.gz"))
        with gzip.open(out, "rt") as fh:
            written = json.load(fh)
        src = written["source"]
        self.assertEqual((src["kind"], src["tag"], src["variant"], src["scanned_from"]),
                         ("relval", "CMSSW_16_1_X_2026-11-13-1100", "", "pyRelValPartialLogs"))
        self.assertEqual((src["jobs_done"], src["jobs_total"], src["complete"]), (2, 2, True))
        self.assertTrue(src["path"].endswith("/pyRelValMatrixLogs.zip"))
        self.assertEqual(len(written["workflows"]), 2)
        self.assertEqual(sorted(p.name for p in self.build.iterdir()),
                         ["pyRelValMatrixLogs-scan.json.gz", "pyRelValPartialLogs"])

    def test_a_job_that_comes_back_after_the_zip_completes_the_run(self):
        # 2 of 3 jobs, zipped after the 10-hour wait; cms-bot renames the folder to .tmp
        for name in ("done.1of3", "done.2of3"):
            (self.partial / name).touch()
        out = problems.scan_relval_dir(str(self.partial))
        self.partial.rename(self.partial.with_name("pyRelValPartialLogs.tmp"))
        # job 3 comes back: a new folder with only its workflows (one of them also ran before)
        wf = self.partial / "11634.0_TTbar"
        wf.mkdir(parents=True)
        (wf / "workflow.log").write_text("11634.0_TTbar Step0-PASSED ; exit: 0\n")
        (wf / "step1_TTbar.log").write_text("all good\n")
        new = self.partial / "999.0_New"
        new.mkdir()
        (new / "workflow.log").write_text("999.0_New Step0-PASSED ; exit: 0\n")
        (self.partial / "done.3of3").touch()

        self.assertEqual(problems.scan_relval_dir(str(self.partial)), out)

        with gzip.open(out, "rt") as fh:
            data = json.load(fh)
        src = data["source"]
        self.assertEqual((src["jobs_done"], src["jobs_total"], src["complete"]), (3, 3, True))
        self.assertEqual(src["jobs"], ["done.1of3", "done.2of3", "done.3of3"])
        self.assertTrue(src["merged_with_scan_of"])
        self.assertEqual([w["workflow"] for w in data["workflows"]],
                         ["11634.0_TTbar", "140.051_RunMuonEG2022C", "999.0_New"])        # kept + rerun + new
        owners = {p["workflow"] for p in data["problems"]}
        self.assertIn("140.051_RunMuonEG2022C", owners)       # earlier problems kept
        self.assertNotIn("11634.0_TTbar", owners)             # rerun log is clean now: old problem replaced

    def test_timed_out_run_is_marked_incomplete(self):
        (self.partial / "done.1of3").touch()

        out = problems.scan_relval_dir(str(self.partial))

        with gzip.open(out, "rt") as fh:
            src = json.load(fh)["source"]
        self.assertEqual((src["jobs_done"], src["jobs_total"], src["complete"]), (1, 3, False))

    def test_gpu_variant_and_bad_folders(self):
        gpu = self.build / "gpu" / "nvidia_h100" / "pyRelValPartialLogs"
        gpu.mkdir(parents=True)
        out = problems.scan_relval_dir(str(gpu))
        with gzip.open(out, "rt") as fh:
            self.assertEqual(json.load(fh)["source"]["variant"], "gpu/nvidia_h100")
        with self.assertRaises(ValueError):
            problems.scan_relval_dir(str(self.build))                          # wrong folder name
        with self.assertRaises(ValueError):
            problems.scan_relval_dir(str(self.build / "nothing" / "pyRelValPartialLogs"))   # missing

    def test_command_line_used_by_process_relval_logs(self):
        self.assertEqual(problems.main(["x", "relval", str(self.partial)]), 0)
        self.assertTrue((self.build / "pyRelValMatrixLogs-scan.json.gz").exists())

    def test_dbg_relvals_are_skipped_unless_switched_on(self):
        dbg = Path(self.tmp.name) / "buildlogs" / "el9_amd64_gcc14/www/fri/20.1.DBG-fri-23/CMSSW_20_1_DBG_X_2026-11-13-2300"
        (dbg / "pyRelValPartialLogs").mkdir(parents=True)
        out = dbg / "pyRelValMatrixLogs-scan.json.gz"
        with patch.dict(os.environ, {problems.DBG_RELVAL_ENV: ""}):
            self.assertEqual(problems.main(["x", "relval", str(dbg / "pyRelValPartialLogs")]), 0)
        self.assertFalse(out.exists())
        with patch.dict(os.environ, {problems.DBG_RELVAL_ENV: "1"}):
            self.assertEqual(problems.main(["x", "relval", str(dbg / "pyRelValPartialLogs")]), 0)
        self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
