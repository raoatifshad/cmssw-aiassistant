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

    def test_file_names(self):
        where = dict(arch="el8_amd64_gcc13", tag="CMSSW_16_1_X_2026-11-13-1100", variant="")
        self.assertEqual(problems.scan_file_name(where, "unit"),
                         "unit__el8_amd64_gcc13__CMSSW_16_1_X_2026-11-13-1100__main.json.gz")
        step = dict(where, variant="gpu/nvidia_t4", workflow="1.0_A", step=3)
        self.assertEqual(problems.scan_file_name(step, "relval"),
                         "relval__el8_amd64_gcc13__CMSSW_16_1_X_2026-11-13-1100__gpu_nvidia_t4__1.0_A__step3.json.gz")


class ScanDirTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.build = Path(self.tmp.name) / "buildlogs" / BUILD
        self.build.mkdir(parents=True)
        self.outbox = Path(self.tmp.name) / "outbox"
        self.outbox.mkdir()
        switch = patch.object(problems, "OUTBOX", str(self.outbox))
        switch.start()
        self.addCleanup(switch.stop)

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

        self.assertEqual(out, str(self.outbox / "unit" / "unit__el8_amd64_gcc13__CMSSW_16_1_X_2026-11-13-1100__main.json.gz"))
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
                         ["UT", "unitTestLogs.zip"])
        self.assertEqual([p.suffix for p in (self.outbox / "unit").iterdir()], [".gz"])

    def test_running_twice_rewrites_the_same_file(self):
        zip_path = self._unit_zip()
        _unpack(zip_path, self.build / "UT")

        first = problems.scan_unit_test_dir(str(self.build / "UT"), str(zip_path))
        second = problems.scan_unit_test_dir(str(self.build / "UT"), str(zip_path))

        self.assertEqual(first, second)
        self.assertEqual(len(list(self.outbox.rglob("*.json.gz*"))), 1)

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
        self.assertEqual(list(self.outbox.iterdir()), [])

    def test_without_an_outbox_nothing_is_scanned(self):
        zip_path = self._unit_zip()
        _unpack(zip_path, self.build / "UT")
        self.outbox.rmdir()
        with patch.object(scanner, "scan_unit_test_dir", side_effect=AssertionError("scanned")):
            self.assertIsNone(problems.scan_unit_test_dir(str(self.build / "UT"), str(zip_path)))
        self.assertIsNone(problems.scan_addon_dir(str(self.build), str(self.build / "addOnTests.zip")))
        self.assertFalse(self.outbox.exists())

    def test_missing_addon_status_log_raises(self):
        zip_path = self.build / "addOnTests.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("cmsDriver-x_step1.log", "fine\n")
        _unpack(zip_path, self.build / "AO")

        with self.assertRaises(FileNotFoundError):
            problems.scan_addon_dir(str(self.build / "AO"), str(zip_path))
        self.assertEqual(list(self.outbox.rglob("*.json.gz*")), [])


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


class RelvalStepTests(unittest.TestCase):
    """write_relval_step_scan: one small file per step log, for parseLog."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.build = Path(self.tmp.name) / "buildlogs" / BUILD
        self.partial = self.build / "pyRelValPartialLogs"
        self.partial.mkdir(parents=True)
        self.outbox = Path(self.tmp.name) / "outbox"
        self.outbox.mkdir()
        _relval_files(self.add)

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, name, text):
        path = self.partial / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)

    def scan(self, name):
        out = problems.write_relval_step_scan(str(self.partial / name), outbox=str(self.outbox))
        if out is None:
            return None
        with gzip.open(out, "rt") as fh:
            return json.load(fh)

    def test_one_file_per_step_with_status_exit_code_and_problems(self):
        wf = "140.051_RunMuonEG2022C"
        before = sorted(str(p) for p in self.partial.rglob("*"))
        one, two = self.scan(f"{wf}/step1_RunMuonEG2022C.log"), self.scan(f"{wf}/step2_RunMuonEG2022C.log")

        self.assertEqual((one["kind"], one["arch"], one["tag"], one["variant"], one["queue"]),
                         ("relval_step", "el8_amd64_gcc13", "CMSSW_16_1_X_2026-11-13-1100", "", "16.1-fri-11"))
        self.assertEqual((one["workflow"], one["step"], one["status"], one["exit_code"]), (wf, 1, "PASSED", 0))
        self.assertEqual((two["step"], two["status"], two["exit_code"]), (2, "FAILED", 256))
        self.assertEqual(one["workflow_steps"], {"1": "PASSED", "2": "FAILED"})
        self.assertNotIn("scan_error", one)
        self.assertEqual({p["member"] for p in one["problems"]}, {f"{wf}/step1_RunMuonEG2022C.log"})   # copy skipped
        self.assertEqual({p["member"] for p in two["problems"]},
                         {f"{wf}/step2_RunMuonEG2022C.log", f"{wf}/step2.log"})      # other content: kept
        name = "relval__el8_amd64_gcc13__CMSSW_16_1_X_2026-11-13-1100__main__140.051_RunMuonEG2022C__step"
        self.assertEqual(sorted(p.name for p in (self.outbox / "relval").iterdir()), [name + "1.json.gz", name + "2.json.gz"])
        self.assertEqual(sorted(str(p) for p in self.partial.rglob("*")), before)     # nothing written there

    def test_all_steps_together_equal_the_whole_folder_scan(self):
        wf = "17034.96_PU"
        self.add(f"{wf}/workflow.log", f"{wf} Step0-PASSED Step1-PASSED ; exit: 0 0\n")
        self.add(f"{wf}/step0_PU.log", "%MSG-w A: m\nzero\n%MSG\n")
        self.add(f"{wf}/step1_PU.log", "%MSG-w B: m\none\n%MSG\n")
        self.add(f"{wf}/step1_dasquery.log", "%MSG-w C: m\ndas\n%MSG\n")
        self.add(f"{wf}/step2_PU.log", "fine\n")
        self.add(f"{wf}/step11_PU.log", "%MSG-e D: m\neleven\n%MSG\n")
        self.add(f"{wf}/das-up-to-nevents.log", "%MSG-e E: m\nno step\n%MSG\n")
        self.add(f"{wf}/step7.log", "%MSG-e F: m\nno named copy\n%MSG\n")

        got = []
        for log in sorted(self.partial.glob("*/step*_*.log")):
            got += self.scan(str(log.relative_to(self.partial)))["problems"] if "dasquery" not in log.name else []

        key = lambda p: (p["member"], p["step"] or 0, p["signature_id"], p["count"])
        full = scanner.scan_relval_dir(str(self.partial))["problems"]
        self.assertEqual(sorted(map(key, got)), sorted(map(key, full)))

    def test_step0_has_no_status_and_no_exit_code(self):
        self.add("5.0_X/workflow.log", "5.0_X Step0-PASSED Step1-FAILED ; exit: 0 256\n")
        self.add("5.0_X/step0_X.log", "fine\n")
        doc = self.scan("5.0_X/step0_X.log")
        self.assertEqual((doc["step"], doc["status"], doc["exit_code"]), (0, None, None))

    def test_das_or_lumi_log_gives_the_same_file_as_the_named_log(self):
        wf = "140.051_RunMuonEG2022C"
        self.add(f"{wf}/step1_lumiRanges.log", "{}")
        named = self.scan(f"{wf}/step1_RunMuonEG2022C.log")
        lumi = self.scan(f"{wf}/step1_lumiRanges.log")       # what parseLog's glob may pick instead
        for doc in (named, lumi):
            doc.pop("scanned_at"), doc.pop("log_mtime")
        self.assertEqual(named, lumi)
        self.assertEqual(len(list((self.outbox / "relval").iterdir())), 1)

    def test_big_copy_is_found_from_its_first_and_last_megabyte(self):
        big = "line\n" * 600000 + "%MSG-e Cat: mod\nbad\n%MSG\n"
        self.add("6.0_Y/workflow.log", "6.0_Y Step0-PASSED ; exit: 0\n")
        self.add("6.0_Y/step1_Y.log", big)
        self.add("6.0_Y/step1.log", big)
        self.assertEqual([p["member"] for p in self.scan("6.0_Y/step1_Y.log")["problems"]], ["6.0_Y/step1_Y.log"])

    def test_clean_step_still_gets_a_file(self):
        self.add("7.0_Z/workflow.log", "7.0_Z Step0-PASSED ; exit: 0\n")
        self.add("7.0_Z/step1_Z.log", "fine\n")
        self.assertEqual(self.scan("7.0_Z/step1_Z.log")["problems"], [])

    def test_failed_scan_is_written_down(self):
        with patch.object(scanner, "_scan_relval_members", side_effect=RuntimeError("boom")):
            doc = self.scan("11634.0_TTbar/step1_TTbar.log")
        self.assertEqual(doc["scan_error"], "RuntimeError: boom")
        self.assertEqual((doc["workflow"], doc["step"], doc["problems"]), ("11634.0_TTbar", 1, []))

    def test_dbg_and_missing_outbox_write_nothing(self):
        dbg = Path(self.tmp.name) / "buildlogs/el9_amd64_gcc14/www/fri/20.1.DBG-fri-23/CMSSW_20_1_DBG_X_2026-11-13-2300"
        log = dbg / "pyRelValPartialLogs" / "1.0_A" / "step1_A.log"
        log.parent.mkdir(parents=True)
        log.write_text(DICTIONARY_NOT_FOUND)
        self.assertIsNone(problems.write_relval_step_scan(str(log), outbox=str(self.outbox)))
        self.assertIsNone(problems.write_relval_step_scan(
            str(self.partial / "11634.0_TTbar/step1_TTbar.log"), outbox=str(self.outbox / "no")))
        self.assertEqual(list(self.outbox.iterdir()), [])
        self.assertFalse((self.outbox / "no").exists())

    def test_variants_are_in_the_name_and_odd_paths_are_refused(self):
        for variant, word in (("gpu/nvidia_h100", "gpu_nvidia_h100"), ("other/rntuple", "other_rntuple")):
            log = self.build / variant / "pyRelValPartialLogs" / "1.0_A" / "step1_A.log"
            log.parent.mkdir(parents=True)
            log.write_text("x\n")
            out = problems.write_relval_step_scan(str(log), outbox=str(self.outbox))
            self.assertIn(f"__{word}__1.0_A__step1", out)
            with gzip.open(out, "rt") as fh:
                self.assertEqual(json.load(fh)["variant"], variant)
        for bad in ("/tmp/1.0_A/step1_A.log", f"/x/{BUILD}/odd/pyRelValPartialLogs/1.0_A/step1_A.log",
                    f"/x/{BUILD}/pyRelValPartialLogs/1.0_A/workflow.log",
                    f"/x/{BUILD}/pyRelValPartialLogs/step1_A.log"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                problems.describe_step_log(bad)

    def test_no_scan_when_the_sender_has_not_run_for_a_day(self):
        old = problems.time.time() - 25 * 3600
        os.utime(self.outbox, (old, old))
        self.assertIsNone(self.scan("11634.0_TTbar/step1_TTbar.log"))
        self.assertEqual(list(self.outbox.iterdir()), [])
        os.utime(self.outbox)
        self.assertIsNotNone(self.scan("11634.0_TTbar/step1_TTbar.log"))
        os.utime(self.outbox, (old, old))                                  # writing into relval/ does not keep it alive
        self.assertIsNone(self.scan("11634.0_TTbar/step1_TTbar.log"))

    def test_a_failed_write_leaves_no_half_file(self):
        with patch.object(problems.json, "dump", side_effect=OSError("No space left on device")):
            with self.assertRaises(OSError):
                self.scan("11634.0_TTbar/step1_TTbar.log")
        self.assertEqual(list((self.outbox / "relval").iterdir()), [])

    def test_rewritten_log_replaces_its_file_and_leaves_no_tmp(self):
        self.scan("11634.0_TTbar/step1_TTbar.log")
        self.add("11634.0_TTbar/step1_TTbar.log", "all good now\n")
        self.assertEqual(self.scan("11634.0_TTbar/step1_TTbar.log")["problems"], [])
        self.assertEqual([p.suffix for p in (self.outbox / "relval").iterdir()], [".gz"])


if __name__ == "__main__":
    unittest.main()
