import unittest
from unittest.mock import patch

from graph.tools import get_relval_recurrence, top_recurring_relval_failures
from server.api_server import (
    _execute_planned_capability, _frequent_failures_report, _parse_recency_window_days,
    _pending_frequent_failure_scopes, _recurring_failure_answer,
)


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
        self.assertIsNone(_recurring_failure_answer("what does workflow 140.051 do", "s1"))

    @patch("server.api_server.get_relval_recurrence")
    def test_recurring_failure_answer_reports_the_rate(self, recurrence):
        recurrence.return_value = {
            "workflow_id": "140.051", "name": "RunMuonEG2022C",
            "total_appearances": 441, "failed_appearances": 258,
            "failed_tags": ["CMSSW_20_1_X_2026-08-17-2300"],
        }

        answer = _recurring_failure_answer("has workflow 140.051 recurred before?", "s1")

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


class FrequentFailuresTests(unittest.TestCase):
    """Regression coverage: a real user asked "What are the frequent failures
    over the last week" and got the generic "could not create a valid local
    evidence plan" rejection, because _recurring_failure_answer only ever
    handled one already-named workflow ID and had no aggregate mode."""

    @patch("graph.tools.get_driver")
    def test_top_recurring_relval_failures_applies_a_time_cutoff(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []

        top_recurring_relval_failures(days=7, limit=10)

        cutoff = session.run.call_args.args[1]["cutoff"]
        self.assertIsNotNone(cutoff)
        self.assertRegex(cutoff, r"^\d{4}-\d{2}-\d{2}-\d{4}$")

    @patch("graph.tools.get_driver")
    def test_top_recurring_relval_failures_all_time_has_no_cutoff(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []

        top_recurring_relval_failures(days=None, limit=10)

        self.assertIsNone(session.run.call_args.args[1]["cutoff"])

    def test_parse_recency_window_recognizes_explicit_phrasing(self):
        self.assertEqual(_parse_recency_window_days("failures over the last week"), 7)
        self.assertEqual(_parse_recency_window_days("failures in the last 3 days"), 3)
        self.assertEqual(_parse_recency_window_days("failures over the past month"), 30)
        self.assertIsNone(_parse_recency_window_days("failures of all time"))
        self.assertIsNone(_parse_recency_window_days("has this ever failed"))

    def test_parse_recency_window_defaults_to_a_week_when_unspecified(self):
        self.assertEqual(_parse_recency_window_days("what are the frequent failures"), 7)

    def tearDown(self):
        _pending_frequent_failure_scopes.pop("freq-session", None)

    @patch("server.api_server.top_recurring_relval_failures")
    @patch("server.api_server.relval_failure_architectures", return_value=["el9_amd64_gcc14"])
    def test_single_architecture_answers_immediately_without_asking(self, architectures, top_failures):
        # Only one architecture had failures in this period -- there's no real
        # choice to ask about, so this must not create a pending clarification.
        top_failures.return_value = [
            {
                "workflow_id": "37634.0", "name": "TTbar_14TeV+Run4D127",
                "total_appearances": 3, "failed_appearances": 3,
                "failed_tags": ["CMSSW_20_1_TF_X_2026-08-18-1100"],
            },
        ]

        report = _frequent_failures_report("What are the frequent failures over the last week", "freq-session")

        top_failures.assert_called_once_with(days=7, limit=10, arch="el9_amd64_gcc14")
        self.assertIn("37634.0", report)
        self.assertIn("3/3", report)
        self.assertIn("the last 7 days", report)
        self.assertNotIn("freq-session", _pending_frequent_failure_scopes)

    @patch("server.api_server.top_recurring_relval_failures", return_value=[])
    @patch("server.api_server.relval_failure_architectures", return_value=[])
    def test_no_architectures_reports_no_failures(self, architectures, top_failures):
        report = _frequent_failures_report("what are the frequent failures over the last week", "freq-session")
        self.assertIn("No RelVal failures are recorded", report)

    @patch("server.api_server.relval_failure_architectures", return_value=["el9_amd64_gcc14", "el9_aarch64_gcc14"])
    def test_multiple_architectures_asks_before_answering(self, architectures):
        # Real regression: RelVal failures over a given period genuinely spanned
        # 25 distinct architectures in production data, all silently combined
        # into one misleading answer before this clarification flow existed.
        answer = _frequent_failures_report("What are the frequent failures over the last week", "freq-session")

        self.assertIn("el9_amd64_gcc14", answer)
        self.assertIn("el9_aarch64_gcc14", answer)
        self.assertIn("Choose one", answer)
        self.assertEqual(
            _pending_frequent_failure_scopes["freq-session"],
            {"days": 7, "architectures": ["el9_amd64_gcc14", "el9_aarch64_gcc14"]},
        )

    @patch("server.api_server.top_recurring_relval_failures", return_value=[])
    @patch("server.api_server.relval_failure_architectures", return_value=["el9_amd64_gcc14", "el9_aarch64_gcc14"])
    def test_choosing_an_architecture_resolves_the_pending_choice(self, architectures, top_failures):
        _frequent_failures_report("What are the frequent failures over the last week", "freq-session")

        _frequent_failures_report("el9_amd64_gcc14", "freq-session")

        top_failures.assert_called_once_with(days=7, limit=10, arch="el9_amd64_gcc14")
        self.assertNotIn("freq-session", _pending_frequent_failure_scopes)

    @patch("server.api_server.top_recurring_relval_failures", return_value=[])
    @patch("server.api_server.relval_failure_architectures", return_value=["el9_amd64_gcc14", "el9_aarch64_gcc14"])
    def test_replying_all_combines_every_architecture(self, architectures, top_failures):
        _frequent_failures_report("What are the frequent failures over the last week", "freq-session")

        _frequent_failures_report("all", "freq-session")

        top_failures.assert_called_once_with(days=7, limit=10, arch=None)
        self.assertNotIn("freq-session", _pending_frequent_failure_scopes)

    @patch("server.api_server.relval_failure_architectures", return_value=["el9_amd64_gcc14", "el9_aarch64_gcc14"])
    def test_choosing_an_unlisted_architecture_is_rejected_without_clearing_the_pending_choice(self, architectures):
        _frequent_failures_report("What are the frequent failures over the last week", "freq-session")

        answer = _frequent_failures_report("el8_amd64_gcc12", "freq-session")

        self.assertIn("no recorded failures in this period", answer)
        self.assertIn("freq-session", _pending_frequent_failure_scopes)

    @patch("server.api_server._frequent_failures_report", return_value="frequent failures report")
    def test_recurring_failure_answer_delegates_to_aggregate_report_without_a_workflow_id(self, report):
        answer = _recurring_failure_answer("What are the frequent failures over the last week", "freq-session")

        self.assertEqual(answer, "frequent failures report")
        report.assert_called_once_with("What are the frequent failures over the last week", "freq-session")

    @patch("server.api_server._frequent_failures_report", return_value="frequent failures report")
    def test_recurring_failure_answer_resumes_a_pending_choice_with_no_keyword_in_the_reply(self, report):
        # A bare architecture reply like "el9_amd64_gcc14" has none of the
        # recurrence trigger words -- it must still resume the pending choice
        # instead of being rejected by the keyword gate.
        _pending_frequent_failure_scopes["freq-session"] = {
            "days": 7, "architectures": ["el9_amd64_gcc14", "el9_aarch64_gcc14"],
        }

        answer = _recurring_failure_answer("el9_amd64_gcc14", "freq-session")

        self.assertEqual(answer, "frequent failures report")


if __name__ == "__main__":
    unittest.main()
