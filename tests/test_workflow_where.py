"""DAS_ERROR RelVal status (importer + backfill) and the cross-build "where is workflow X
failing" answer. Found 2026-10-15: 38551.85 had DAS_ERROR at step1 in all 250 imported
runs but was stored as PASSED, and "where is 38551.85 failing?" got the latest IB's
failure overview while a bare "38551.85" got the "couldn't match that" menu."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import backfill_relval_status
from build_graph import relval_status
from server.api_server import _relval_scope_summary, _workflow_where_failing_answer


class RelvalStatusTests(unittest.TestCase):
    def test_status_rule(self):
        self.assertEqual(relval_status([{"status": "PASSED"}, {"status": "FAILED"}]), "FAILED")
        self.assertEqual(relval_status([{"status": "DAS_ERROR"}]), "DAS_ERROR")
        self.assertEqual(relval_status([{"status": "DAS_ERROR"}, {"status": "FAILED"}]), "FAILED")
        self.assertEqual(relval_status([{"status": "PASSED"}]), "PASSED")
        self.assertEqual(relval_status([]), "PASSED")

    def test_backfill_reads_the_same_keys_build_graph_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            day = Path(tmp) / "el9_amd64_gcc14" / "2026-10-12-1100"
            day.mkdir(parents=True)
            (day / "CMSSW_20_1_X.json").write_text(json.dumps([
                {"id": "38551.85", "steps": [{"status": "DAS_ERROR"}]},
                {"id": "1.0", "steps": [{"status": "PASSED"}]},
                {"id": "2.0", "steps": [{"status": "FAILED"}]},
            ]))
            rows = backfill_relval_status.wanted_statuses(Path(tmp))
        self.assertEqual(rows, [
            {"key": "CMSSW_20_1_X_2026-10-12-1100:el9_amd64_gcc14:38551.85", "status": "DAS_ERROR"},
            {"key": "CMSSW_20_1_X_2026-10-12-1100:el9_amd64_gcc14:2.0", "status": "FAILED"},
        ])

    @patch("server.api_server.get_driver")
    def test_summary_counts_das_errors_separately(self, get_driver):
        get_driver.return_value.session.return_value.__enter__.return_value.run.return_value = [
            {"workflow_id": "38551.85"}]
        record = {"tag": "CMSSW_20_1_X_2026-10-12-1100", "arch": "el9_amd64_gcc14", "relval_total": 951,
                  "failed_relvals": 0, "passed_relvals": 950, "known_error_relvals": 0, "das_error_relvals": 1}
        answer = _relval_scope_summary([record], "RelVals")
        self.assertTrue(answer.startswith("🟠"))  # not green: one workflow could not run
        self.assertIn("950 passed, 1 DAS error (input files not found, nothing ran)", answer)
        self.assertIn("1 DAS error(s): `38551.85`", answer)


def _rows(tag, archs, variant="primary", status="DAS_ERROR", exitcode=1, name="HydjetQMinBias_5519GeV+Run4D128_hi"):
    return [{"tag": tag, "arch": arch, "variant": variant, "name": name, "status": status, "exitcode": exitcode,
             "known_error": False, "known_error_message": None} for arch in archs]


class WorkflowWhereTests(unittest.TestCase):
    def _patch_graph(self, relvals, details=(), exceptions=()):
        session = MagicMock()

        def run(query, **params):
            if "RAN_RELVAL" in query:
                return relvals
            if "TestLogRun" in query:
                return [{"detail": d} for d in details]
            return list(exceptions)
        session.run.side_effect = run
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        return patch("server.api_server.get_driver", return_value=driver)

    def test_das_error_everywhere(self):
        relvals = (_rows("CMSSW_20_1_X_2026-10-14-2300", ["el9_amd64_gcc14", "el8_amd64_gcc14"])
                   + _rows("CMSSW_20_1_CLANG_X_2026-10-14-2300", ["el9_amd64_gcc14"], variant="CLANG")
                   + _rows("CMSSW_20_0_X_2026-10-07-2300", ["el8_aarch64_gcc13"]))
        detail = "38551.85_HydjetQMinBias_5519GeV+Run4D128_hi: step1=DAS_ERROR step2=NOTRUN; exit 1 0"
        with self._patch_graph(relvals, [detail, detail]), \
                patch("server.api_server.get_exit_code_names", return_value={}):
            for question in ("where is 38551.85 failing?", "38551.85", "wf 38551.85", "is workflow 38551.85 failing"):
                answer = _workflow_where_failing_answer(question, "s")
                self.assertIn("**4 of 4** imported run(s) did not pass", answer, question)
        self.assertIn("🟠 **DAS error** in 4 run(s)", answer)
        self.assertIn("could not get its input files from DAS", answer)
        self.assertIn("did not pass **anywhere**", answer)
        self.assertIn("`CMSSW_20_0`, `CMSSW_20_1`", answer)
        self.assertIn("`step1=DAS_ERROR step2=NOTRUN; exit 1 0` -- 2 run(s)", answer)
        self.assertIn("Most recent: `CMSSW_20_1_X_2026-10-14-2300`", answer)

    def test_mixed_results_list_where_it_failed(self):
        relvals = (_rows("CMSSW_20_1_X_2026-10-13-2300", ["el9_amd64_gcc14"], status="FAILED", exitcode=256, name="W")
                   + _rows("CMSSW_20_1_X_2026-10-12-2300", ["el9_amd64_gcc14"], status="PASSED", exitcode=0, name="W")
                   + _rows("CMSSW_17_0_X_2026-10-13-2300", ["el8_amd64_gcc12"], status="PASSED", exitcode=0, name="W"))
        with self._patch_graph(relvals), patch("server.api_server.get_exit_code_names", return_value={}):
            answer = _workflow_where_failing_answer("where is 512.0 failing", "s")
        self.assertIn("**1 of 3** imported run(s) did not pass", answer)
        self.assertIn("| `CMSSW_20_1` | primary | `el9_amd64_gcc14` | 1 / 2 | `CMSSW_20_1_X_2026-10-13-2300` |", answer)
        self.assertIn("Passed in every run of: `CMSSW_17_0` primary.", answer)

    def test_why_without_tag_falls_back_to_every_build_when_latest_ib_lacks_it(self):
        from server.api_server import _workflow_why_failing_answer
        relvals = _rows("CMSSW_20_1_X_2026-10-14-2300", ["el9_amd64_gcc14"])
        session = MagicMock()

        def run(query, **params):
            if "ib.parent_tag = $tag" in query:
                return []  # the latest IB has no record of the workflow yet
            if "RAN_RELVAL" in query:
                return relvals
            return []
        session.run.side_effect = run
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch("server.api_server.get_driver", return_value=driver), \
                patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-10-15-0900"}]), \
                patch("server.api_server.get_exit_code_names", return_value={}):
            answer = _workflow_why_failing_answer("why is workflow 38551.85 failing", "why-no-tag")
        self.assertIn("🟠 **DAS error** in 1 run(s)", answer)
        self.assertNotIn("has no imported RelVal record", answer)

    def test_other_handlers_keep_their_questions(self):
        with self._patch_graph([]) as get_driver:
            for question in ("why is 38551.85 failing", "38551.85 any PR?", "is there an issue for workflow 38551.85",
                             "has 38551.85 failed before", "where is 38551.85 failing in CMSSW_20_1_X_2026-10-14-2300",
                             "relval exception for 38551.85", "what is failing in the latest IB"):
                self.assertIsNone(_workflow_where_failing_answer(question, "s"), question)
            get_driver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
