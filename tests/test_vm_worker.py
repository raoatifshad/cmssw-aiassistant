"""deploy/vm/worker.py: when each job is due, what it skips, and that a failing job
is recorded without stopping the worker."""
import datetime as dt
import gzip
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    "vm_worker", Path(__file__).resolve().parent.parent / "deploy" / "vm" / "worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)

AT = lambda s: dt.datetime.fromisoformat(s)
DEFAULT_CMSREPO_AT = list(worker.CMSREPO_AT)   # as loaded, before any test patches it


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        # cmsrepo on the same two slots here: these tests are about the slot logic itself
        patcher = patch.multiple(worker, REFRESH_AT=[(0, 30), (12, 30)], CMSREPO_AT=[(0, 30), (12, 30)],
                                 REFRESH_STEPS=["cmsrepo", "exceptions", "docs"], TESTLOGS_EVERY=dt.timedelta(minutes=30))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_last_slot(self):
        self.assertEqual(worker.last_refresh_slot(AT("2026-11-16T12:29:00")), AT("2026-11-16T00:30:00"))
        self.assertEqual(worker.last_refresh_slot(AT("2026-11-16T12:30:00")), AT("2026-11-16T12:30:00"))
        self.assertEqual(worker.last_refresh_slot(AT("2026-11-16T00:10:00")), AT("2026-11-15T12:30:00"))

    def test_a_fresh_worker_runs_everything_once(self):
        self.assertEqual(worker.due_jobs({}, AT("2026-11-16T15:00:00")), ["cmsrepo", "exceptions", "docs", "testlogs"])

    def test_refresh_runs_once_per_slot_even_after_a_failure(self):
        state = {n: {"tried_at": "2026-11-16T12:30:05", "ok": n != "exceptions"} for n in ("cmsrepo", "exceptions", "docs")}
        state["testlogs"] = {"tried_at": "2026-11-16T14:50:00"}
        self.assertEqual(worker.due_jobs(state, AT("2026-11-16T15:00:00")), [])           # a failure waits for 00:30
        self.assertEqual(worker.due_jobs(state, AT("2026-11-17T00:30:00")), ["cmsrepo", "exceptions", "docs", "testlogs"])

    def test_a_slot_missed_while_down_runs_at_start(self):
        state = {n: {"tried_at": "2026-11-15T12:30:01"} for n in ("cmsrepo", "exceptions", "docs")}
        state["testlogs"] = {"tried_at": "2026-11-16T09:00:00"}
        self.assertEqual(worker.due_jobs(state, AT("2026-11-16T09:10:00")), ["cmsrepo", "exceptions", "docs"])

    def test_testlogs_every_30_minutes(self):
        state = {n: {"tried_at": "2026-11-16T12:30:01"} for n in ("cmsrepo", "exceptions", "docs")}
        state["testlogs"] = {"tried_at": "2026-11-16T14:40:00"}
        self.assertEqual(worker.due_jobs(state, AT("2026-11-16T15:09:00")), [])
        self.assertEqual(worker.due_jobs(state, AT("2026-11-16T15:10:00")), ["testlogs"])


class HourlyCmsrepoTests(unittest.TestCase):
    """cmsrepo every hour at :30 (the IB page's data is pushed about hourly); exceptions and docs
    stay at 00:30 and 12:30."""

    def setUp(self):
        patcher = patch.multiple(worker, REFRESH_AT=[(0, 30), (12, 30)], CMSREPO_AT=[(h, 30) for h in range(24)],
                                 REFRESH_STEPS=["cmsrepo", "exceptions", "docs"], TESTLOGS_EVERY=dt.timedelta(minutes=30))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _state(self, cmsrepo, others="2026-11-16T12:30:05"):
        state = {"cmsrepo": {"tried_at": cmsrepo}, "exceptions": {"tried_at": others}, "docs": {"tried_at": others}}
        state["testlogs"] = {"tried_at": "2026-11-16T14:50:00"}
        return state

    def test_default_is_every_hour_at_half_past(self):
        self.assertEqual(DEFAULT_CMSREPO_AT, [(h, 30) for h in range(24)])

    def test_cmsrepo_is_due_every_hour_the_others_are_not(self):
        self.assertEqual(worker.due_jobs(self._state("2026-11-16T13:30:04"), AT("2026-11-16T14:31:00")), ["cmsrepo"])
        self.assertEqual(worker.due_jobs(self._state("2026-11-16T14:30:40"), AT("2026-11-16T15:00:00")), [])

    def test_at_half_past_twelve_cmsrepo_runs_first(self):
        # exceptions attach to RelVals already in the graph: the fresh cmsrepo import comes first
        due = worker.due_jobs(self._state("2026-11-17T11:30:03", others="2026-11-17T00:30:05"), AT("2026-11-17T12:31:00"))
        self.assertEqual(due, ["cmsrepo", "exceptions", "docs", "testlogs"])   # testlogs: last run 11-16

    def test_own_times_can_be_set(self):
        self.assertEqual(worker._times("06:15,18:15"), [(6, 15), (18, 15)])
        with patch.object(worker, "CMSREPO_AT", worker._times("06:15,18:15")):
            self.assertEqual(worker.due_jobs(self._state("2026-11-16T06:15:02"), AT("2026-11-16T15:00:00")), [])
            self.assertEqual(worker.due_jobs(self._state("2026-11-16T06:15:02"), AT("2026-11-16T18:15:00")), ["cmsrepo", "testlogs"])


class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        patcher = patch.multiple(worker, RESULTS=root / "results", RECEIVED=root / "received", STATE_DIR=root / "worker",
                                 STATE=root / "worker" / "state.json", SOURCES=root / "sources")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.root = root

    def _result(self, name, mtime):
        path = self.root / "results" / "relval" / "el9_amd64_gcc14" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as fh:
            json.dump({}, fh)
        os.utime(path, (mtime, mtime))

    def test_testlogs_imports_only_when_a_file_changed(self):
        state = {}
        with patch.object(worker, "run") as run:
            self.assertEqual(worker.job_testlogs(state), "no results files yet")
            self._result("a.json.gz", 1_000_000)
            self.assertIn("imported", worker.job_testlogs(state))
            self.assertEqual(worker.job_testlogs(state), "nothing new")
            self._result("b.json.gz", 2_000_000)                       # a new or rewritten (merged) file
            self.assertIn("imported", worker.job_testlogs(state))
        self.assertEqual(run.call_count, 2)
        self.assertIn("ingest_test_logs.py", run.call_args.args[0][1])

    def _batch(self, name, folder="received", age_days=0.0):
        path = self.root / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        t = worker.time.time() - age_days * 86400
        os.utime(path, (t, t))
        return path

    def test_waiting_batches_are_merged_before_the_import(self):
        self._result("a.json.gz", 1_000_000)
        self._batch("1.tar")
        with patch.object(worker, "run") as run:
            worker.job_testlogs({})
        self.assertEqual([c.args[0][1] for c in run.call_args_list],
                         ["ingestion/merge_test_log_batches.py", "ingestion/ingest_test_logs.py"])

    def test_imported_batches_are_deleted_after_the_kept_days(self):
        self._result("a.json.gz", 1_000_000)
        old, new = self._batch("old.tar", "received/merged", 8), self._batch("new.tar", "received/merged", 1)
        waiting = self._batch("wait.tar", age_days=30)
        with patch.object(worker, "run"):
            self.assertIn("1 imported batches deleted", worker.job_testlogs({}))
        self.assertEqual([p.exists() for p in (old, new, waiting)], [False, True, True])
        with patch.object(worker, "run"), patch.object(worker, "RECEIVED_KEEP_DAYS", 0):
            worker.job_testlogs({})
        self.assertEqual([p.exists() for p in (new, waiting)], [False, True])

    def test_refused_files_are_counted_and_deleted_after_30_days(self):
        self._result("a.json.gz", 1_000_000)
        old, new = self._batch("a__x", "received/rejected", 31), self._batch("b__y", "received/rejected", 2)
        self._batch("b__y.reason.txt", "received/rejected", 2)
        state = {}
        with patch.object(worker, "run"):
            self.assertIn("1 refused files", worker.job_testlogs(state))
        self.assertEqual((state["testlogs"]["refused_files"], old.exists(), new.exists()), (1, False, True))

    def test_nothing_is_deleted_when_the_merge_or_the_import_failed(self):
        self._result("a.json.gz", 1_000_000)
        old = self._batch("old.tar", "received/merged", 8)
        self._batch("1.tar")
        for script in ("merge_test_log_batches", "ingest_test_logs"):
            calls = []

            def run(cmd, *a, script=script, calls=calls):
                calls.append(cmd[1])
                if script in cmd[1]:
                    raise worker.StepFailed("exit 1")
            with patch.object(worker, "run", run), self.assertRaises(worker.StepFailed):
                worker.job_testlogs({})
            self.assertTrue(old.exists())
            self.assertIn("ingestion/ingest_test_logs.py", calls)          # what did merge is still imported

    def test_an_unreadable_results_folder_fails_instead_of_saying_no_files(self):
        self._result("a.json.gz", 1_000_000)
        with patch.object(worker.os, "access", return_value=False):
            with self.assertRaises(worker.StepFailed):
                worker.job_testlogs({})
        sub = self.root / "results" / "relval" / "el9_amd64_gcc14"
        sub.chmod(0)                                    # a folder below it: os.walk would skip it silently
        self.addCleanup(sub.chmod, 0o755)
        if os.access(sub, os.R_OK):                     # running as root: permissions not enforced
            self.skipTest("root ignores folder permissions")
        with self.assertRaises(worker.StepFailed):
            worker.job_testlogs({})

    def test_a_failed_import_is_tried_again_next_time(self):
        self._result("a.json.gz", 1_000_000)
        state = {}
        with patch.object(worker, "run", side_effect=worker.StepFailed("exit 1")):
            self.assertFalse(worker.run_job("testlogs", state))
        self.assertNotIn("imported_up_to", state["testlogs"])
        with patch.object(worker, "run"):
            self.assertIn("imported", worker.job_testlogs(state))

    def test_a_failing_job_is_recorded_and_does_not_stop_the_worker(self):
        state = {}
        with patch.dict(worker.REFRESH_JOBS, {"cmsrepo": lambda: 1 / 0}):
            self.assertFalse(worker.run_job("cmsrepo", state))
        saved = json.loads(worker.STATE.read_text())
        self.assertFalse(saved["cmsrepo"]["ok"])
        self.assertIn("ZeroDivisionError", saved["cmsrepo"]["result"])
        self.assertNotIn("ok_at", saved["cmsrepo"])

    def test_an_empty_exceptions_export_is_never_imported(self):
        good, empty, broken = (self.root / n for n in ("good.json", "empty.json", "broken.json"))
        good.write_text(json.dumps({"hits": {"hits": [{"_id": 1}]}}))
        empty.write_text(json.dumps({"hits": {"hits": []}}))
        broken.write_text('{"hits": ')
        self.assertEqual([worker.export_is_usable(p) for p in (good, empty, broken)], [True, False, False])

        def fake_run(cmd, timeout_s, cwd=None, stdout=None):
            if "es-cmssw-exceptions.py" in cmd:
                stdout.write(json.dumps({"hits": {"hits": []}}))
            elif "build_graph.py" in " ".join(map(str, cmd)):
                raise AssertionError("an empty export reached the graph")
        with patch.object(worker, "git_update"), patch.object(worker, "run", side_effect=fake_run):
            with self.assertRaises(worker.StepFailed):
                worker.job_exceptions()


class StaleGitLockTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        (self.repo / ".git").mkdir()
        self.lock = self.repo / ".git" / "index.lock"
        patcher = patch.object(worker, "log")
        self.log = patcher.start()
        self.addCleanup(patcher.stop)

    def test_an_old_lock_is_removed_and_logged(self):
        self.lock.touch()
        old = worker.time.time() - 2 * 3600
        os.utime(self.lock, (old, old))
        worker.remove_stale_git_lock(self.repo)
        self.assertFalse(self.lock.exists())
        self.assertIn("stale", self.log.call_args.args[0])

    def test_a_fresh_lock_is_kept(self):
        self.lock.touch()
        worker.remove_stale_git_lock(self.repo)
        self.assertTrue(self.lock.exists())
        self.log.assert_not_called()

    def test_no_lock_nothing_happens(self):
        worker.remove_stale_git_lock(self.repo)
        self.log.assert_not_called()

    def test_git_update_checks_the_lock_before_fetch(self):
        calls = []
        with patch.object(worker, "remove_stale_git_lock", side_effect=lambda p: calls.append("lock")), \
             patch.object(worker, "run", side_effect=lambda cmd, *a, **k: calls.append(cmd[3])), \
             patch.object(worker.subprocess, "run"):
            worker.git_update(self.repo, "url")
        self.assertEqual(calls[:2], ["lock", "fetch"])


if __name__ == "__main__":
    unittest.main()
