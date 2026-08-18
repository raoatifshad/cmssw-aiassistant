import unittest
from unittest.mock import patch

from graph.tools import get_relval_recurrence
from server.api_server import _execute_planned_capability, _recurring_failure_answer


class RecurringFailureTests(unittest.TestCase):
    @patch("graph.tools.get_driver")
    def test_get_relval_recurrence_computes_failure_rate(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{
            "name": "RunMuonEG2022C",
            "total_appearances": 441,
            "failed_appearances": 258,
            "failed_tags": ["CMSSW_20_1_X_2026-08-17-2300", "CMSSW_20_1_X_2026-08-17-1100"],
        }]

        result = get_relval_recurrence("140.051")

        self.assertEqual(result["total_appearances"], 441)
        self.assertEqual(result["failed_appearances"], 258)

    @patch("graph.tools.get_driver")
    def test_get_relval_recurrence_reports_zero_for_an_unknown_workflow(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []

        result = get_relval_recurrence("999999.0")

        self.assertEqual(result["total_appearances"], 0)

    def test_recurring_failure_answer_requires_a_recurrence_keyword(self):
        self.assertIsNone(_recurring_failure_answer("what does workflow 140.051 do"))

    @patch("server.api_server.get_relval_recurrence")
    def test_recurring_failure_answer_reports_the_rate(self, recurrence):
        recurrence.return_value = {
            "workflow_id": "140.051", "name": "RunMuonEG2022C",
            "total_appearances": 441, "failed_appearances": 258,
            "failed_tags": ["CMSSW_20_1_X_2026-08-17-2300"],
        }

        answer = _recurring_failure_answer("has workflow 140.051 recurred before?")

        self.assertIn("258 of 441", answer)
        self.assertIn("59%", answer)

    @patch("server.api_server._recurring_failure_answer", return_value="recurrence evidence")
    @patch("server.api_server._pr_evidence_answer", return_value=None)
    @patch("server.api_server._pr_number_answer", return_value=None)
    def test_known_issue_phrasing_falls_through_to_recurrence_not_shadowed(
        self, pr_number, pr_evidence, recurring
    ):
        # Regression: "is this a known issue" contains the word "issue", which
        # routes it to the pr_evidence capability — it must not dead-end there
        # once PR/Issue evidence itself finds nothing.
        answer, _ = _execute_planned_capability(
            "pr_evidence", "is workflow 140.051 a known issue?", "known-issue-session",
        )
        self.assertEqual(answer, "recurrence evidence")


if __name__ == "__main__":
    unittest.main()
