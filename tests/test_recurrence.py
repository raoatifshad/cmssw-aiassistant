import unittest
from unittest.mock import patch

from graph.tools import get_relval_recurrence, search_workflow_ids, top_recurring_relval_failures
from server.api_server import (
    _execute_planned_capability, _frequent_failures_report, _last_workflow_contexts, _parse_recency_window_days,
    _pending_frequent_failure_scopes, _pending_workflow_choices, _recurring_failure_answer,
)


def _relval_row(tag, name="RunMuonEG2022C", status="FAILED", arch="el9_amd64_gcc14", variant="primary"):
    return {"name": name, "status": status, "tag": tag, "arch": arch, "variant": variant}


class RecurringFailureTests(unittest.TestCase):
    @patch("graph.tools.get_driver")
    def test_get_relval_recurrence_computes_failure_rate(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = (
            [_relval_row("CMSSW_20_1_X_2026-08-17-2300"), _relval_row("CMSSW_20_1_X_2026-08-17-1100")]
            + [_relval_row(f"CMSSW_20_1_X_2026-08-{n:02d}-2300", status="PASSED") for n in range(1, 17)]
        )

        result = get_relval_recurrence("140.051")

        self.assertEqual(result["total_appearances"], 18)
        self.assertEqual(result["failed_appearances"], 2)
        self.assertEqual(
            [record["tag"] for record in result["failed_records"]],
            ["CMSSW_20_1_X_2026-08-17-2300", "CMSSW_20_1_X_2026-08-17-1100"],
        )

    @patch("graph.tools.get_driver")
    def test_get_relval_recurrence_does_not_crowd_out_sub_ib_failures(self, get_driver):
        # Regression: found live 2026-09-18, workflow 537.0. A sub-IB's tag (e.g.
        # CMSSW_20_1_UBSAN_X_...) always sorts alphabetically BEFORE its primary's
        # tag (CMSSW_20_1_X_...) purely because every variant name starts with a
        # letter earlier than "X" -- the old Cypher-side `ORDER BY ib.tag DESC`
        # therefore always ranked every primary-build failure ahead of every sub-IB
        # failure regardless of actual date, so a genuinely more recent sub-IB
        # failure (ROOT6, 09-16) got silently dropped from a count-capped list full
        # of older primary-build entries (several architectures, 09-11 through
        # 09-15). Re-sorting by _ib_sort_key in Python must recover it and rank it
        # ahead of the older primary entries.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        primary_tags = [
            f"CMSSW_20_1_X_2026-09-{day}-2300" for day in ("11", "12", "13", "14", "15")
        ]
        session.run.return_value = (
            [_relval_row(tag, name="SomeWorkflow") for tag in primary_tags]
            + [_relval_row(
                "CMSSW_20_1_ROOT6_X_2026-09-16-2300", name="SomeWorkflow",
                arch="el9_aarch64_gcc14", variant="ROOT6",
            )]
        )

        result = get_relval_recurrence("537.0")

        tags_in_order = [record["tag"] for record in result["failed_records"]]
        self.assertEqual(tags_in_order[0], "CMSSW_20_1_ROOT6_X_2026-09-16-2300")
        self.assertIn("CMSSW_20_1_ROOT6_X_2026-09-16-2300", tags_in_order)

    @patch("graph.tools.get_driver")
    def test_get_relval_recurrence_scopes_to_the_current_era_when_a_number_is_reused(self, get_driver):
        # Regression: found live 2026-09-18. workflow_id "537.0" has 237 occurrences
        # named "DYToLL012Jets_5FS_..." (the current, actual workflow) and a separate
        # 5 occurrences named "DYToLL012Jets_5FS_...+...+HARVESTGEN" from an unrelated,
        # much older era -- the same number reassigned to a physically different
        # workflow. The old implementation aggregated with no explicit GROUP BY,
        # which implicitly grouped by `name` and picked between the two arbitrarily
        # (observed flipping across a plain database restart, no code change at
        # all). Must always resolve to the name/count of the MOST RECENT occurrence.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = (
            [_relval_row(f"CMSSW_20_1_X_2026-09-{n:02d}-2300", name="CurrentWorkflow",
                         status="FAILED" if n in (11, 17) else "PASSED") for n in range(10, 18)]
            + [_relval_row(f"CMSSW_10_6_ASAN_X_2020-01-{n:02d}-2300", name="OldEraWorkflow+HARVESTGEN",
                           arch="slc7_amd64_gcc700", variant="ASAN") for n in range(1, 6)]
        )

        result = get_relval_recurrence("537.0")

        self.assertEqual(result["name"], "CurrentWorkflow")
        self.assertEqual(result["total_appearances"], 8)
        self.assertEqual(result["failed_appearances"], 2)
        self.assertTrue(all(
            record["tag"].startswith("CMSSW_20_1_X_") for record in result["failed_records"]
        ))

    @patch("graph.tools.get_driver")
    def test_get_relval_recurrence_reports_zero_for_an_unknown_workflow(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []

        result = get_relval_recurrence("999999.0")

        self.assertEqual(result["total_appearances"], 0)

    def test_recurring_failure_answer_requires_a_recurrence_keyword(self):
        self.assertIsNone(_recurring_failure_answer("what does workflow 140.051 do", "s1"))

    @patch("server.api_server.get_relval_recurrence")
    def test_recurring_failure_answer_recognizes_when_did_it_start_failing_phrasing(self, recurrence):
        # Regression: a real user asked "when the WF starts failing 140.0" --
        # no "recurring"/"how often"/etc. keyword -- and it fell through to
        # the fallback clarification instead of looking up the workflow.
        recurrence.return_value = {
            "workflow_id": "140.0", "name": "SomeWorkflow",
            "total_appearances": 10, "failed_appearances": 4,
            "failed_tags": ["CMSSW_20_1_X_2026-08-18-2300"],
        }

        answer = _recurring_failure_answer("when the WF starts failing 140.0", "s1")

        self.assertIn("4 of 10", answer)

    @patch("server.api_server.get_relval_recurrence")
    def test_recurring_failure_answer_when_phrasing_survives_a_decimal_workflow_id(self, recurrence):
        # Regression: the bounded gap between "when" and "fail" excluded "."
        # to avoid crossing sentence boundaries, but workflow IDs are decimal
        # numbers (e.g. "140.0") -- the exclusion broke on the ID's own
        # period, not a real sentence break, so this never matched at all.
        recurrence.return_value = {
            "workflow_id": "140.0", "name": "SomeWorkflow",
            "total_appearances": 10, "failed_appearances": 4,
            "failed_tags": ["CMSSW_20_1_X_2026-08-18-2300"],
        }

        answer = _recurring_failure_answer("when the workflow 140.0 start failing in the Ibs", "s1")

        self.assertIn("4 of 10", answer)

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

    @patch("server.api_server.get_relval_recurrence")
    def test_recurring_failure_answer_remembers_the_workflow_for_pr_evidence_followups(self, recurrence):
        # Regression: "is there any issue/pr open for it?" as a follow-up
        # needs to know what "it" refers to -- _pr_evidence_answer has no
        # session memory of its own, so this must be recorded here.
        recurrence.return_value = {
            "workflow_id": "140.0", "name": "SomeWorkflow",
            "total_appearances": 14, "failed_appearances": 0, "failed_tags": [],
        }

        _recurring_failure_answer("when does workflow 140.0 start failing", "s-remember")

        self.assertEqual(_last_workflow_contexts.get("s-remember"), "140.0")
        _last_workflow_contexts.pop("s-remember", None)

    @patch("server.api_server._recurring_failure_answer", return_value="recurrence evidence")
    @patch("server.api_server._artifact_search_answer", return_value=None)
    @patch("server.api_server._release_artifact_answer", return_value=None)
    @patch("server.api_server._workflow_artifact_answer", return_value=None)
    @patch("server.api_server._batch_workflow_pr_answer", return_value=None)
    @patch("server.api_server._pr_evidence_answer", return_value=None)
    @patch("server.api_server._pr_number_answer", return_value=None)
    def test_known_issue_phrasing_falls_through_to_recurrence_not_shadowed(
        self, pr_number, pr_evidence, batch_workflow, workflow_artifact,
        release_artifact, artifact_search, recurring,
    ):
        # Regression: "is this a known issue" contains the word "issue", which
        # routes it to the pr_evidence capability — it must not dead-end there
        # once every PR/Issue evidence lookup in that capability's chain (see
        # CLAUDE.md's "Capability-handler coverage" section for the full list)
        # finds nothing.
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


class WorkflowDisambiguationTests(unittest.TestCase):
    """A typed workflow ID that's wrong or written as a wildcard (e.g.
    "140.999", "140.xx") must trigger a prefix search and a "did you mean"
    choice instead of a dead end -- never silently guess which one."""

    def tearDown(self):
        _pending_workflow_choices.pop("s1", None)

    @patch("graph.tools.get_driver")
    def test_search_workflow_ids_matches_by_prefix(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"workflow_id": "140.001", "name": "RunMinimumBias2022A"},
            {"workflow_id": "140.051", "name": "RunMuonEG2022C"},
        ]

        results = search_workflow_ids("140.", limit=15)

        self.assertEqual(len(results), 2)
        self.assertEqual(session.run.call_args.args[1]["prefix"], "140.")

    @patch("server.api_server.search_workflow_ids")
    def test_wildcard_with_multiple_candidates_asks_which_one(self, search_ids):
        search_ids.return_value = [
            {"workflow_id": "140.001", "name": "RunMinimumBias2022A"},
            {"workflow_id": "140.051", "name": "RunMuonEG2022C"},
        ]

        answer = _recurring_failure_answer("when does workflow 140.xx start failing", "s1")

        self.assertIn("140.001", answer)
        self.assertIn("140.051", answer)
        self.assertIn("s1", _pending_workflow_choices)
        search_ids.assert_called_once_with("140.", limit=15)

    @patch("server.api_server.search_workflow_ids", return_value=[])
    def test_wildcard_with_no_candidates_says_so_plainly(self, search_ids):
        answer = _recurring_failure_answer("when does workflow 99999.xx start failing", "s1")

        self.assertIn("No imported workflow starts with `99999.`", answer)
        self.assertNotIn("s1", _pending_workflow_choices)

    @patch("server.api_server.search_workflow_ids")
    @patch("server.api_server.get_relval_recurrence")
    def test_not_found_exact_id_falls_back_to_prefix_search(self, recurrence, search_ids):
        recurrence.return_value = {"workflow_id": "140.999", "total_appearances": 0}
        search_ids.return_value = [{"workflow_id": "140.051", "name": "RunMuonEG2022C"}]

        answer = _recurring_failure_answer("has workflow 140.999 recurred before?", "s1")

        self.assertIn("Workflow `140.999` is not reported", answer)
        self.assertIn("140.051", answer)
        search_ids.assert_called_once_with("140.", limit=15)

    @patch("server.api_server.get_relval_recurrence")
    def test_found_exact_id_answers_directly_without_searching(self, recurrence):
        recurrence.return_value = {
            "workflow_id": "140.0", "name": "SomeWorkflow",
            "total_appearances": 10, "failed_appearances": 0, "failed_tags": [],
        }

        answer = _recurring_failure_answer("has workflow 140.0 recurred before?", "s1")

        self.assertIn("0 of 10", answer)
        self.assertNotIn("s1", _pending_workflow_choices)

    @patch("server.api_server.get_relval_recurrence")
    def test_choosing_a_listed_workflow_resolves_the_pending_choice(self, recurrence):
        _pending_workflow_choices["s1"] = {
            "candidates": [
                {"workflow_id": "140.001", "name": "RunMinimumBias2022A"},
                {"workflow_id": "140.051", "name": "RunMuonEG2022C"},
            ],
        }
        recurrence.return_value = {
            "workflow_id": "140.051", "name": "RunMuonEG2022C",
            "total_appearances": 221, "failed_appearances": 131, "failed_tags": [],
        }

        answer = _recurring_failure_answer("140.051", "s1")

        self.assertIn("131 of 221", answer)
        self.assertNotIn("s1", _pending_workflow_choices)
        recurrence.assert_called_once_with("140.051")

    @patch("server.api_server.search_workflow_ids")
    @patch("server.api_server.get_relval_recurrence")
    def test_naming_a_fresh_workflow_id_abandons_the_stale_pending_choice(self, recurrence, search_ids):
        _pending_workflow_choices["s1"] = {
            "candidates": [{"workflow_id": "140.001", "name": "RunMinimumBias2022A"}],
        }
        recurrence.return_value = {
            "workflow_id": "37634.0", "name": "TTbar_14TeV+Run4D127",
            "total_appearances": 150, "failed_appearances": 44, "failed_tags": [],
        }

        answer = _recurring_failure_answer("has workflow 37634.0 recurred before?", "s1")

        self.assertIn("44 of 150", answer)
        search_ids.assert_not_called()

    def test_unmatched_reply_re_shows_the_pending_choice(self):
        _pending_workflow_choices["s1"] = {
            "candidates": [{"workflow_id": "140.001", "name": "RunMinimumBias2022A"}],
        }

        answer = _recurring_failure_answer("I'm not sure", "s1")

        self.assertIn("140.001", answer)
        self.assertIn("s1", _pending_workflow_choices)


if __name__ == "__main__":
    unittest.main()
