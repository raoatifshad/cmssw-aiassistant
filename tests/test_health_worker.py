"""/api/health reports VM worker jobs that keep failing, and a worker that stopped."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import server.api_server as api


def _state(**jobs):
    return {name: dict(job) for name, job in jobs.items()}


class WorkerHealthTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "state.json"
        patcher = patch.object(api, "WORKER_STATE_FILE", str(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, state, age_s=0):
        self.path.write_text(json.dumps(state))
        then = time.time() - age_s
        os.utime(self.path, (then, then))

    def test_all_ok(self):
        self.write(_state(cmsrepo={"ok": True, "ok_at": "2026-11-27T13:30:00", "tried_at": "2026-11-27T13:30:00"},
                          testlogs={"ok": True, "ok_at": "2026-11-27T13:50:00", "tried_at": "2026-11-27T13:50:00"}))
        worker = api.health()["worker"]
        self.assertTrue(worker["ok"])
        self.assertEqual(worker["failing"], {})

    def test_a_job_failing_for_hours_is_reported(self):
        self.write(_state(cmsrepo={"ok": False, "ok_at": "2026-11-26T04:30:10", "tried_at": "2026-11-27T13:30:20",
                                   "result": "FAILED (exit 128: git -C /data/sources/cmsrepo)"},
                          testlogs={"ok": True, "ok_at": "2026-11-27T13:52:04", "tried_at": "2026-11-27T13:52:04"}))
        worker = api.health()["worker"]
        self.assertFalse(worker["ok"])
        self.assertEqual(list(worker["failing"]), ["cmsrepo"])
        self.assertEqual(worker["failing"]["cmsrepo"]["last_ok"], "2026-11-26T04:30:10")

    def test_one_recent_failure_is_not_reported_yet(self):
        self.write(_state(cmsrepo={"ok": False, "ok_at": "2026-11-27T12:30:00", "tried_at": "2026-11-27T13:30:00"},
                          testlogs={"ok": True, "ok_at": "2026-11-27T13:50:00", "tried_at": "2026-11-27T13:50:00"}))
        self.assertTrue(api.health()["worker"]["ok"])

    def test_a_job_that_never_succeeded_is_reported(self):
        self.write(_state(docs={"ok": False, "tried_at": "2026-11-27T12:30:00", "result": "FAILED"}))
        self.assertIn("docs", api.health()["worker"]["failing"])

    def test_a_stopped_worker_is_reported(self):
        self.write(_state(testlogs={"ok": True, "ok_at": "2026-11-27T13:50:00", "tried_at": "2026-11-27T13:50:00"}),
                   age_s=2 * 3600)
        worker = api.health()["worker"]
        self.assertFalse(worker["ok"])
        self.assertTrue(worker["stopped"])

    def test_unreadable_state_is_reported_not_raised(self):
        worker = api.health()["worker"]   # no file written
        self.assertFalse(worker["ok"])
        self.assertIn("cannot read", worker["error"])

    def test_no_worker_configured_leaves_health_as_before(self):
        with patch.object(api, "WORKER_STATE_FILE", ""):
            self.assertNotIn("worker", api.health())


if __name__ == "__main__":
    unittest.main()
