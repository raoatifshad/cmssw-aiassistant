import gzip
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

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
                    f"/data/sdt/buildlogs/{BUILD}/pyRelValMatrixLogs.zip",     # not phase 1
                    f"/data/sdt/buildlogs/{BUILD}/odd/unitTestLogs.zip",       # unknown level below the tag
                    f"/data/sdt/buildlogs/{BUILD}/gpu/a/b/unitTestLogs.zip",
                    "/data/sdt/buildlogs/arch/www/fri/q/NOT_A_TAG/unitTestLogs.zip"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                problems.describe_zip_path(bad)

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


if __name__ == "__main__":
    unittest.main()
