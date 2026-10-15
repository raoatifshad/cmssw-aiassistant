"""GET /api/test-log-problems and get_test_log_changes: what changed inside tests the
dashboard reports as PASSED between the two builds the Shift Console compares."""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from server.api_server import app, require_shifter
from tools import get_test_log_changes

TO, FROM = "CMSSW_20_1_X_2026-10-12-1100", "CMSSW_20_1_X_2026-10-11-2300"

TO_RUNS = [
    dict(key="T1", tag=TO, arch="el9_amd64_gcc14", suite="relval", run_variant="",
         ran=["1.0", "2.0", "3.0", "4.0", "5.0"]),
    dict(key="T2", tag="CMSSW_20_1_PY314_X_2026-10-12-1100", arch="el9_amd64_gcc14", suite="unit",
         run_variant="", ran=["Pkg/testX"]),
    dict(key="T3", tag=TO, arch="el8_amd64_gcc14", suite="addon", run_variant="", ran=["hlt"]),
    # another series that only shares the prefix text must never be picked up
    dict(key="T4", tag="CMSSW_20_10_X_2026-10-12-1100", arch="el9_amd64_gcc14", suite="unit",
         run_variant="", ran=[]),
]
BASE_RUNS = [
    dict(track="T1", key="B0", tag="CMSSW_20_1_X_2026-10-11-1300", ran=[]),
    dict(track="T1", key="B1", tag=FROM, ran=["1.0", "3.0", "5.0", "6.0"]),
    dict(track="T2", key="B2a", tag="CMSSW_20_1_PY314_X_2026-10-08-1100", ran=[]),
    dict(track="T2", key="B2", tag="CMSSW_20_1_PY314_X_2026-10-09-1100", ran=["Pkg/testX"]),
]


def _problem(run, sid, owners, statuses, kind="python_traceback"):
    return dict(run=run, sid=sid, kind=kind, category=None, summary=f"problem {sid}", message="",
                origin="cmssw", intentional_hint=None, owners=owners, statuses=statuses, evidence="ev",
                server_path="/data/x.zip", member="m.log", line=1)


PROBLEMS = [
    _problem("T1", "A", ["1.0:step1"], ["passed"]),          # first time ever -> new
    _problem("T1", "B", ["2.0:step1"], ["passed"]),          # seen 10-04, not in baseline -> returned
    _problem("T1", "C", ["3.0:step1"], ["passed"]),          # also in baseline -> still present
    _problem("T1", "D", ["4.0:step1"], ["failed"]),          # failed test: the dashboard shows it already
    _problem("B1", "C", ["3.0:step1"], ["passed"]),
    _problem("B1", "E", ["5.0:step1"], ["passed"]),          # 5.0 ran again without it -> gone
    _problem("B1", "F", ["6.0:step1"], ["passed"]),          # 6.0 did not run in T1 -> nothing said
    _problem("T2", "G", ["Pkg/testX"], ["passed"], kind="ubsan"),
    _problem("T3", "H", ["hlt:step1"], ["passed"]),          # no baseline at all -> not compared
]


class _Session:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, **params):
        if "UNWIND $tracks" in cypher:
            ids = {t["id"] for t in params["tracks"]}
            return [r for r in BASE_RUNS if r["track"] in ids]
        if "HAS_PROBLEM" in cypher:
            return [p for p in PROBLEMS if p["run"] in params["keys"]]
        return TO_RUNS


def _lookups(cypher, params=None):
    if "ProblemFirstSeen" in cypher:
        firsts = {"relval|CMSSW_20_1|1.0:step1|A": TO, "relval|CMSSW_20_1|2.0:step1|B": "CMSSW_20_1_X_2026-10-04-2300"}
        return [dict(key=k, tag=firsts[k]) for k in params["keys"] if k in firsts]
    return [dict(sid=s, first="CMSSW_20_1_X_2026-10-01-2300", last=TO) for s in params["sids"]]


def _changes(**kwargs):
    with patch("tools.get_driver") as get_driver, patch("tools._run_query_all", side_effect=_lookups), \
            patch("tools.test_log_coverage", return_value={"last": "2026-10-12-1100"}):
        get_driver.return_value.session.return_value = _Session()
        return get_test_log_changes(TO, FROM, **kwargs)


class TestLogChangesTests(unittest.TestCase):
    def setUp(self):
        self.result = _changes()
        self.archs = {a["arch"]: a for a in self.result["architectures"]}

    def test_new_returned_gone_and_still_present_are_told_apart(self):
        el9 = self.archs["el9_amd64_gcc14"]
        self.assertEqual({g["signature_id"] for g in el9["new"]}, {"A", "G"})
        self.assertEqual([g["signature_id"] for g in el9["returned"]], ["B"])
        self.assertEqual([g["signature_id"] for g in el9["gone"]], ["E"])
        self.assertEqual(el9["still_present"], 1)
        self.assertEqual(self.result["totals"], dict(new_problems=2, new_tests=2, returned_problems=1,
                                                     returned_tests=1, gone_problems=1, gone_tests=1,
                                                     still_present=1))

    def test_a_returned_problem_says_when_it_was_first_seen_for_that_test(self):
        test = self.archs["el9_amd64_gcc14"]["returned"][0]["tests"][0]
        self.assertEqual((test["name"], test["first_seen_tag"], test["returned"]),
                         ("2.0:step1", "CMSSW_20_1_X_2026-10-04-2300", True))

    def test_problems_in_failed_tests_and_tests_that_did_not_run_are_not_reported(self):
        listed = {g["signature_id"] for a in self.result["architectures"]
                  for part in ("new", "returned", "gone") for g in a[part]}
        self.assertNotIn("D", listed)
        self.assertNotIn("F", listed)

    def test_sub_ib_without_a_build_at_from_uses_its_own_previous_build(self):
        track = next(t for t in self.archs["el9_amd64_gcc14"]["tracks"] if t["variant"] == "PY314")
        self.assertEqual(track["baseline_tag"], "CMSSW_20_1_PY314_X_2026-10-09-1100")
        self.assertIn("own previous build", track["baseline_note"])
        primary = next(t for t in self.archs["el9_amd64_gcc14"]["tracks"] if t["variant"] == "primary")
        self.assertEqual((primary["baseline_tag"], primary["baseline_note"]), (FROM, None))

    def test_a_track_without_any_earlier_build_is_not_compared(self):
        el8 = self.archs["el8_amd64_gcc14"]
        self.assertEqual((el8["new"], el8["returned"]), ([], []))
        self.assertEqual([p["suite"] for p in el8["no_baseline"]], ["addon"])

    def test_another_series_sharing_the_prefix_is_ignored(self):
        tags = {t["tag"] for a in self.result["architectures"] for t in a["tracks"]}
        self.assertNotIn("CMSSW_20_10_X_2026-10-12-1100", tags)

    def test_no_imported_logs_is_not_reported_as_zero_problems(self):
        with patch("tools.get_driver") as get_driver, \
                patch("tools.test_log_coverage", return_value={"last": "2026-10-12-1100"}):
            get_driver.return_value.session.return_value.__enter__.return_value.run.return_value = []
            result = get_test_log_changes("CMSSW_20_1_X_2026-10-12-2300", TO)
        self.assertFalse(result["coverage"]["to_imported"])
        self.assertIn("not the same as zero problems", result["coverage"]["message"])

    def test_bad_tags_are_rejected(self):
        with self.assertRaises(ValueError):
            get_test_log_changes("latest", FROM)


class TestLogProblemsEndpointTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[require_shifter] = lambda: "alice"
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(require_shifter, None)

    @patch("server.api_server.get_test_log_changes", return_value={"totals": {}})
    def test_passes_the_window_and_architecture(self, changes):
        response = self.client.get("/api/test-log-problems", params={"from": FROM, "to": TO, "arch": "el9_amd64_gcc14"})
        self.assertEqual(response.status_code, 200)
        changes.assert_called_once_with(TO, FROM, "el9_amd64_gcc14")

    def test_both_tags_are_required(self):
        self.assertEqual(self.client.get("/api/test-log-problems", params={"to": TO}).status_code, 422)

    @patch("server.api_server.get_test_log_changes", side_effect=ValueError("bad tag"))
    def test_a_bad_tag_is_a_422(self, _changes):
        response = self.client.get("/api/test-log-problems", params={"from": "x", "to": TO})
        self.assertEqual((response.status_code, response.json()["detail"]), (422, "bad tag"))


if __name__ == "__main__":
    unittest.main()
