"""ingestion/merge_test_log_batches.py: received batches -> one result file per build."""
import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent))
import ingest_test_logs  # noqa: E402
import merge_test_log_batches as merge  # noqa: E402
from server import test_log_ingest as tli  # noqa: E402
from test_test_log_ingest import gz, scan, step, tar_of  # noqa: E402

TAG = "CMSSW_20_1_X_2026-11-16-2300"
BUILD = f"relval/el9_amd64_gcc14/{TAG}/_/pyRelValMatrixLogs.json.gz"
KEY = f"el9_amd64_gcc14/{TAG}/_"


def problem(wf, number, text="boom", kind="fatal_exception"):
    return dict(kind=kind, category="Cat", severity=None, origin="cmssw", message=text, summary=text,
                signature=f"{kind}|Cat|{text}", signature_id=text, block=[text], count=1, line_start=1,
                workflow=wf, step=number, test_status="passed", raw_status="PASSED", member=f"{wf}/step{number}_x.log")


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.received, self.results = Path(self.tmp.name) / "received", Path(self.tmp.name) / "results"
        self.received.mkdir()
        self.lines = []

    def batch(self, *docs, progress=None, **files):
        members = {f"relval/r{i}.json.gz": gz(d) for i, d in enumerate(docs)}
        if progress:
            members["progress.json"] = json.dumps(progress).encode()
        members.update({f"{k}/{k}.json.gz": gz(v) for k, v in files.items()})
        answer = tli.receive_batch(tar_of(members), self.received)
        self.assertEqual(answer["rejected"], {})

    def merge(self):
        return merge.merge_pending(self.received, self.results, log=self.lines.append)

    def result(self, rel=BUILD):
        with gzip.open(self.results / rel, "rt") as fh:
            return json.load(fh)

    def waiting(self):
        return len(list(self.received.glob("*.tar")))

    def steps(self, data):
        return {w: sorted(s) for w, s in data["steps_scanned"].items()}

    def test_steps_of_several_batches_become_one_build(self):
        self.batch(step("1.0_A", 1, problems=[problem("1.0_A", 1)]), step("1.0_A", 2))
        self.batch(step("2.0_B", 1, workflow_steps={"1": "FAILED"}, workflow_exit_codes=[256]))
        self.assertEqual(self.merge(), 0)
        data = self.result()
        self.assertEqual(self.steps(data), {"1.0_A": ["1", "2"], "2.0_B": ["1"]})
        self.assertEqual([(w["workflow"], w["status"]) for w in data["workflows"]], [("1.0_A", "passed"), ("2.0_B", "failed")])
        self.assertEqual(len(data["problems"]), 1)
        self.assertEqual((data["source"]["tag"], data["source"]["not_scanned"]), (TAG, []))
        self.assertEqual((self.waiting(), len(list((self.received / "merged").glob("*.tar")))), (0, 2))
        self.assertEqual([p.name for p in self.results.rglob("*.part")], [])

        self.batch(step("3.0_C", 1, workflow_steps={"1": "PASSED"}))                 # a later job
        self.merge()
        self.assertEqual(sorted(self.steps(self.result())), ["1.0_A", "2.0_B", "3.0_C"])
        self.assertEqual(len(self.result()["problems"]), 1)

    def test_what_the_importer_makes_of_it(self):
        self.batch(step("1.0_A", 1, problems=[problem("1.0_A", 1)]),
                   progress={KEY: dict(jobs=["done.1of2"], total=2, closed=False)})
        self.merge()
        run = ingest_test_logs.run_record(self.result())
        self.assertEqual((run["suite"], run["ib_tag"], run["total"], run["passed"]), ("relval", TAG, 1, 1))
        self.assertEqual((run["jobs_done"], run["jobs_total"], run["complete"], run["closed"]), (1, 2, False, False))
        self.assertEqual(run["not_scanned"], ["1.0_A:step2"])                        # ran, its file is not here yet
        self.assertTrue(run["server_path"].startswith("/data/sdt/buildlogs/el9_amd64_gcc14/www/mon/"))
        recs = ingest_test_logs.problem_records(self.result(), run["key"])
        self.assertEqual([r["owners"] for r in recs], [["1.0_A:step1"]])

    def test_merging_again_changes_nothing(self):
        self.batch(step("1.0_A", 1, problems=[problem("1.0_A", 1)]), step("1.0_A", 2), step("2.0_B", 1, problems=[problem("2.0_B", 1)]))
        self.merge()
        before = (self.results / BUILD).read_bytes()
        os.utime(self.results / BUILD, (1, 1))
        self.batch(step("1.0_A", 1, problems=[problem("1.0_A", 1)]), step("1.0_A", 2))
        self.assertEqual(self.merge(), 0)
        self.assertEqual((self.results / BUILD).read_bytes(), before)
        self.assertEqual((self.results / BUILD).stat().st_mtime, 1)                  # not even rewritten

    def test_newer_wins_in_any_order(self):
        old = step("1.0_A", 1, problems=[problem("1.0_A", 1, "old")])
        new = step("1.0_A", 1, log_mtime=2000.0, problems=[problem("1.0_A", 1, "new")])
        for docs in ((old, new), (new, old)):
            for doc in docs:
                self.batch(doc)
            self.merge()
            self.assertEqual([p["message"] for p in self.result()["problems"]], ["new"])
            (self.results / BUILD).unlink()

    def test_a_failed_scan_is_not_clean_and_never_replaces_a_good_one(self):
        failed = step("1.0_A", 1, scan_error="RuntimeError: boom", workflow_steps={}, workflow_exit_codes=[],
                      scanned_at="2026-11-17T09:00:00")
        self.batch(failed, step("1.0_A", 2))
        self.merge()
        data = self.result()
        self.assertEqual(data["source"]["not_scanned"], ["1.0_A:step1"])
        self.assertEqual(data["workflows"][0]["status"], "passed")                   # status from the other step's file
        self.batch(step("1.0_A", 1, problems=[problem("1.0_A", 1)]))                 # the good scan, even if earlier
        self.merge()
        self.assertEqual((self.result()["source"]["not_scanned"], len(self.result()["problems"])), ([], 1))
        self.batch(failed)
        self.merge()
        self.assertEqual((self.result()["source"]["not_scanned"], len(self.result()["problems"])), ([], 1))

    def test_only_failed_scans_give_an_unknown_workflow(self):
        self.batch(step("1.0_A", 1, scan_error="x", workflow_steps={}, workflow_exit_codes=[]))
        self.merge()
        run = ingest_test_logs.run_record(self.result())
        self.assertEqual((run["passed"], run["failed"], run["other_tests"], run["not_scanned"]),
                         (0, 0, ["1.0_A"], ["1.0_A:step1"]))

    def test_a_workflow_that_ran_again_drops_steps_it_did_not_run_this_time(self):
        self.batch(step("1.0_A", 1), step("1.0_A", 2, problems=[problem("1.0_A", 2)]))
        self.merge()
        self.batch(step("1.0_A", 1, log_mtime=2000.0, workflow_steps={"1": "FAILED", "2": "NOTRUN"},
                        workflow_exit_codes=[256, 0], problems=[problem("1.0_A", 1, "again")]))
        self.merge()
        data = self.result()
        self.assertEqual(self.steps(data), {"1.0_A": ["1"]})
        self.assertEqual([p["message"] for p in data["problems"]], ["again"])
        self.assertEqual(data["workflows"][0]["status"], "failed")
        self.assertEqual(data["source"]["not_scanned"], [])

    def test_step0_and_steps_without_a_log(self):
        self.batch(step("1.0_A", 0, status=None, exit_code=None), step("1.0_A", 1), step("1.0_A", 2))
        self.merge()
        self.assertEqual(self.steps(self.result()), {"1.0_A": ["0", "1", "2"]})
        self.assertEqual(self.result()["source"]["not_scanned"], [])

    def test_variants_and_builds_are_kept_apart(self):
        self.batch(step("1.0_A", 1), step("1.0_A", 1, variant="gpu/nvidia_t4"), step("1.0_A", 1, variant="other/rntuple"),
                   step("1.0_A", 1, arch="el8_amd64_gcc13", path=f"el8_amd64_gcc13/www/mon/q/{TAG}/x"))
        self.merge()
        self.assertEqual(sorted(str(p.relative_to(self.results).parent) for p in self.results.rglob("*.json.gz")), [
            f"relval/el8_amd64_gcc13/{TAG}/_", f"relval/el9_amd64_gcc14/{TAG}/_",
            f"relval/el9_amd64_gcc14/{TAG}/gpu_nvidia_t4", f"relval/el9_amd64_gcc14/{TAG}/other_rntuple"])

    def test_progress_before_with_and_after_the_steps(self):
        self.batch(progress={KEY: dict(jobs=["done.1of3"], total=3, closed=False)})
        self.assertEqual(self.merge(), 0)                                            # no steps yet: remembered
        self.assertFalse((self.results / BUILD).exists())
        self.batch(step("1.0_A", 1))
        self.merge()
        self.assertEqual((self.result()["source"]["jobs_done"], self.result()["source"]["complete"]), (1, False))
        self.batch(progress={KEY: dict(jobs=["done.2of3", "done.3of3"], total=3, closed=True)})
        self.merge()
        src = self.result()["source"]
        self.assertEqual((src["jobs"], src["complete"], src["closed"]),
                         (["done.1of3", "done.2of3", "done.3of3"], True, True))

    def test_into_a_whole_folder_scan_of_the_same_build(self):
        old = scan("relval")
        old["workflows"] = [dict(workflow=w, steps={"1": "PASSED"}, exit_codes=[0], status="passed") for w in ("1.0_A", "9.0_Z")]
        old["problems"] = [problem("1.0_A", 1, "old"), problem("9.0_Z", 1, "kept")]
        tli.receive(gz(old), self.results)
        self.batch(step("1.0_A", 1, workflow_steps={"1": "PASSED"}, problems=[problem("1.0_A", 1, "new")]))
        self.merge()
        data = self.result()
        self.assertEqual(sorted(p["message"] for p in data["problems"]), ["kept", "new"])
        self.assertEqual([w["workflow"] for w in data["workflows"]], ["1.0_A", "9.0_Z"])
        self.assertEqual(data["source"]["not_scanned"], [])

    def test_unit_and_addon_results_go_straight_to_their_place(self):
        self.batch(unit=scan("unit"), addon=scan("addon"))
        self.assertEqual(self.merge(), 0)
        self.assertEqual(sorted(p.name for p in self.results.rglob("*.json.gz")), ["addOnTests.json.gz", "unitTestLogs.json.gz"])

    def test_a_broken_batch_is_set_aside_and_the_others_are_merged(self):
        (self.received / "20260101T000000.0-x.tar").write_bytes(b"junk" * 500)
        self.batch(step("1.0_A", 1))
        self.assertEqual(self.merge(), 1)
        self.assertEqual(len(list((self.received / "bad").iterdir())), 1)
        self.assertTrue((self.results / BUILD).exists())
        self.assertEqual(self.merge(), 0)                                            # reported once

    def test_a_build_that_cannot_be_merged_keeps_its_batch(self):
        path = self.results / BUILD
        path.parent.mkdir(parents=True)
        path.write_bytes(gz(scan("relval"))[:60])                                    # damaged result file
        self.batch(step("1.0_A", 1))
        self.batch(step("1.0_A", 1, variant="gpu/nvidia_t4"))
        self.assertEqual(self.merge(), 1)
        self.assertEqual(path.read_bytes(), gz(scan("relval"))[:60])                 # never overwritten as "nothing yet"
        self.assertEqual(self.waiting(), 1)
        self.assertEqual(len(list((self.received / "merged").iterdir())), 1)
        path.unlink()
        self.assertEqual(self.merge(), 0)
        self.assertEqual((self.waiting(), self.steps(self.result())), (0, {"1.0_A": ["1"]}))

    def test_a_write_that_lost_something_keeps_the_batch(self):
        self.batch(step("1.0_A", 1, problems=[problem("1.0_A", 1)]))
        real = merge.write_result
        with patch.object(merge, "write_result", lambda p, d: real(p, dict(d, problems=[]))):
            self.assertEqual(self.merge(), 1)
        self.assertEqual(self.waiting(), 1)
        self.assertTrue(any("not complete after writing" in l for l in self.lines))

    def test_nothing_waiting(self):
        self.assertEqual(self.merge(), 0)
        self.assertEqual(merge.main(["--received", str(self.received / "no"), "--results", str(self.results)]), 1)


if __name__ == "__main__":
    unittest.main()
