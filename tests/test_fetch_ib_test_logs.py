import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
import fetch_ib_test_logs as fetch

TAG_DIR = "el9_amd64_gcc14/www/mon/20.1-mon-11/CMSSW_20_1_X_2026-10-05-1100"


class ListingTests(unittest.TestCase):
    def test_parse_listing_finds_zips_markers_and_addon_logs(self):
        listing = (
            f"280599646 1791230000.5 {TAG_DIR}/pyRelValMatrixLogs.zip\n"
            f"7303022 1791240000.0 {TAG_DIR}/unitTestLogs.zip\n"
            f"0 1791240100.0 {TAG_DIR}/unitTestLogs.zip.checked\n"
            f"104000000 1791230000.0 {TAG_DIR}/gpu/nvidia_t4/pyRelValMatrixLogs.zip\n"
            f"295000000 1791230000.0 {TAG_DIR}//other/rntuple/pyRelValMatrixLogs.zip\n"
            f"6397907 1791240000.0 {TAG_DIR}/addOnTests.zip\n"
            f"12000 1791240000.0 {TAG_DIR}/addOnTests.log\n"
        )
        zips, checked, addon_logs = fetch.parse_listing(listing)

        self.assertEqual(sorted((z["kind"], z["variant"]) for z in zips),
                         [("addon", ""), ("relval", ""), ("relval", "gpu/nvidia_t4"), ("relval", "other/rntuple"),
                          ("unit", "")])
        self.assertEqual(checked, {f"{TAG_DIR}/unitTestLogs.zip"})
        self.assertIn(f"{TAG_DIR}/addOnTests.log", addon_logs)
        relval = next(z for z in zips if z["kind"] == "relval" and not z["variant"])
        self.assertEqual((relval["arch"], relval["tag"], relval["size"]),
                         ("el9_amd64_gcc14", "CMSSW_20_1_X_2026-10-05-1100", 280599646))

    def test_unit_zip_without_checked_marker_is_not_ready(self):
        item = dict(kind="unit", path=f"{TAG_DIR}/unitTestLogs.zip", mtime=0)
        ready, why = fetch.is_ready(item, set(), now=10_000, min_age_minutes=60)
        self.assertFalse(ready)
        self.assertIn(".checked", why)

    def test_unit_zip_never_marked_by_cms_bot_is_scanned_after_a_day(self):
        item = dict(kind="unit", path=f"{TAG_DIR}/gpu/amd_mi300x/unitTestLogs.zip", mtime=0)
        self.assertFalse(fetch.is_ready(item, set(), now=23 * 3600, min_age_minutes=60)[0])
        self.assertTrue(fetch.is_ready(item, set(), now=25 * 3600, min_age_minutes=60)[0])

    def test_recently_modified_relval_zip_is_not_ready(self):
        item = dict(kind="relval", path=f"{TAG_DIR}/pyRelValMatrixLogs.zip", mtime=9_000)
        self.assertFalse(fetch.is_ready(item, set(), now=10_000, min_age_minutes=60)[0])
        self.assertTrue(fetch.is_ready(item, set(), now=9_000 + 3601, min_age_minutes=60)[0])

    def test_huge_dbg_zips_go_last_and_the_rest_stays_oldest_first(self):
        items = [dict(path="dbg-old", size=33_000_000_000, mtime=1), dict(path="new", size=7_000_000, mtime=30),
                 dict(path="old", size=280_000_000, mtime=10), dict(path="dbg-new", size=33_500_000_000, mtime=20)]
        self.assertEqual([i["path"] for i in fetch.order_todo(items)], ["old", "new", "dbg-old", "dbg-new"])

    def test_local_copies_already_deleted_on_the_server_are_still_processed(self):
        with tempfile.TemporaryDirectory() as tmp:
            gone, still = f"{TAG_DIR}/unitTestLogs.zip", f"{TAG_DIR}/addOnTests.zip"
            for rel in (gone, gone + ".checked", still):
                os.makedirs(os.path.join(tmp, os.path.dirname(rel)), exist_ok=True)
                Path(tmp, rel).write_bytes(b"x")
            zips, checked = fetch.local_only_zips(tmp, on_server={still})
        self.assertEqual([(z["path"], z["kind"], z["local_only"]) for z in zips], [(gone, "unit", True)])
        self.assertEqual(checked, {gone})

    def test_finished_zips_are_listed_for_rsync_exclude(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "results"))
            ledger = {"a": dict(path=f"{TAG_DIR}/addOnTests.zip", kind="addon", status="done"),
                      "b": dict(path=f"{TAG_DIR}/unitTestLogs.zip", kind="unit", status="failed")}
            fetch.write_rsync_exclude(tmp, ledger)
            lines = Path(tmp, "results", "rsync-exclude.txt").read_text().splitlines()
        self.assertEqual(lines, [f"/{TAG_DIR}/addOnTests.log", f"/{TAG_DIR}/addOnTests.zip",
                                 f"/{TAG_DIR}/addOnTests.zip.checked"])

    def test_result_path_keeps_ib_arch_and_variant_apart(self):
        item = dict(kind="relval", arch="el9_amd64_gcc14", tag="CMSSW_20_1_X_2026-10-05-1100",
                    variant="gpu/nvidia_t4", path=f"{TAG_DIR}/gpu/nvidia_t4/pyRelValMatrixLogs.zip")
        self.assertEqual(fetch.result_path("/b", item),
                         "/b/results/relval/el9_amd64_gcc14/CMSSW_20_1_X_2026-10-05-1100/gpu_nvidia_t4/"
                         "pyRelValMatrixLogs.json.gz")


class LedgerTests(unittest.TestCase):
    def test_last_entry_per_key_wins_so_a_retried_zip_counts_as_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            fetch.append_ledger(path, dict(key="k", path="p", status="failed"))
            fetch.append_ledger(path, dict(key="k", path="p", status="done"))
            self.assertEqual(fetch.read_ledger(path)["k"]["status"], "done")

    def test_changed_size_or_mtime_is_a_new_key(self):
        a = dict(path="p", size=1, mtime=100.2)
        self.assertEqual(fetch.ledger_key(a), fetch.ledger_key(dict(a, mtime=100.9)))
        self.assertNotEqual(fetch.ledger_key(a), fetch.ledger_key(dict(a, size=2)))


class ExpectedZipsTests(unittest.TestCase):
    def test_dashboard_entries_map_to_zip_paths(self):
        root = "/data/sdt/buildlogs/" + TAG_DIR
        data = {"comparisons": [{
            "relvals": [{"arch": "el9_amd64_gcc14", "file": f"{root}/pyRelValMatrixLogs/run/runall-report-step123-.log"},
                        {"arch": "el9_amd64_gcc14",
                         "file": f"{root}/gpu/nvidia_t4/pyRelValMatrixLogs/run/runall-report-step123-.log"}],
            "utests": [{"arch": "el9_amd64_gcc14", "file": f"{root}/unitTests-summary.log"},
                       {"arch": "el9_amd64_gcc15", "file": "['/data/sdt/buildlogs/x/new/logAnalysis.pkl']"}],
            "addons": [{"arch": "el9_amd64_gcc14", "file": f"{root}/addOnTests.log"}],
            "gpu_data": {"qa": {"amd_mi300x": {"el9_amd64_gcc14": {
                "utests": [{"file": f"{root}/gpu/amd_mi300x/unitTests-summary.log"}]}}}},
        }]}
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "CMSSW_20_1_X.json"), "w") as fh:
                json.dump(data, fh)
            expected = fetch.expected_zips(tmp)

        self.assertEqual(expected, {f"{TAG_DIR}/pyRelValMatrixLogs.zip",
                                    f"{TAG_DIR}/gpu/nvidia_t4/pyRelValMatrixLogs.zip",
                                    f"{TAG_DIR}/unitTestLogs.zip", f"{TAG_DIR}/addOnTests.zip",
                                    f"{TAG_DIR}/gpu/amd_mi300x/unitTestLogs.zip"})


if __name__ == "__main__":
    unittest.main()
