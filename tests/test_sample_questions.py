"""Owner 2026-11-13: sample questions (help, /api/sample-questions, "ask next" suggestions) and fixes 1-4 of the 11-13 smoke test:
"which PR caused it?" after a list of failing workflows, the old-Clang-data lead, PRs opened OR merged in a window, "what can I ask you?"."""
import unittest
from unittest.mock import patch

from server import pr_cause, sample_questions
from server.api_server import ChatRequest, _HELP_RE, _clang_stale_lead, chat


class HelpAndSamplesTests(unittest.TestCase):
    def test_help_phrasings(self):
        for q in ("what can I ask you?", "help", "what questions can I ask?", "show me some example questions", "examples",
                  "how do I use this chat?", "what can you do?"):
            self.assertTrue(_HELP_RE.fullmatch(q), q)
        for q in ("help me with 36434.0", "what can I ask about RelVal 537.0?", "which examples fail?"):
            self.assertFalse(_HELP_RE.fullmatch(q), q)

    @patch("server.sample_questions._failing_workflow", return_value="36434.0")
    @patch("server.api_server._store_chat_turns")
    def test_help_lists_the_failure_chain_and_suggests_starting_points(self, _store, _wf):
        result = chat(ChatRequest(question="what can I ask you?", session_id="help-samples"))
        self.assertEqual(result.route, "conversational:help")
        for text in ("where is RelVal 36434.0 failing?", "is it reported on GitHub?", "which PR could have caused it?", "what code did that PR touch?"):
            self.assertIn(text, result.answer)
        self.assertNotIn("clang", result.answer.lower())  # owner 2026-11-13: no Clang questions
        self.assertEqual(result.actions, [])  # suggestions, not "Choose a scope" buttons
        self.assertTrue(result.suggestions)

    @patch("server.sample_questions._failing_workflow", return_value=None)
    def test_without_a_failing_workflow_the_chain_starts_from_the_latest_ib(self, _wf):
        chain = [g for g in sample_questions.sample_groups() if g.get("chain")][0]
        self.assertEqual(chain["items"][0]["question"], "what is failing in the latest IB?")
        for group in sample_questions.sample_groups():
            for item in group["items"]:
                self.assertNotIn("<", item["question"])
                self.assertNotIn("clang", item["question"].lower())

    def test_next_suggestions_follow_the_chain(self):
        ns = sample_questions.next_suggestions
        labels = lambda route, answer: [s["label"] for s in ns(route, answer)]  # noqa: E731
        self.assertEqual(labels("deterministic:workflow_where", "Workflow `36434.0` is **now failing in 18 places**"),
                         ["Is it reported?", "Which PR caused it?"])
        self.assertEqual(labels("deterministic:pr_cause", "**Most likely: PR #51861** - MTD"), ["What code did it touch?", "Is it reported?"])
        self.assertEqual(labels("deterministic:pr_cause", "**No clear candidate:** none"), ["Is it reported?"])
        self.assertEqual(labels("ai_known:deterministic:github_reported", "Possibly reported"), ["Which PR caused it?"])
        self.assertEqual(labels("capability:unittest", "No failed unit-test packages were found for `CMSSW_20_1_X_2026-11-12-1100`."), [])
        self.assertEqual(labels("clarification", "Which architecture?"), [])
        self.assertEqual(labels("ai:llm_validated", "anything"), [])


class ListedWorkflowsCauseTests(unittest.TestCase):
    SUMMARY = ("🔴 Imported RelVal results for `CMSSW_20_1_X_2026-11-12-1100`: **5 failing RelVal(s)**\n"
               "- 🔴 **el8_amd64_gcc14**: `TTbar_14TeV+Run4D110FS` (wf 36434.0, exit 16640 (OtherCMS)), `TTbar_14TeV+Run4D121FS` (wf 36834.0, exit 16640)\n"
               "- 🔴 **el9_amd64_gcc14**: `DY` (wf 537.0, exit 62720 (SIGSEGV))")

    def test_listed_workflows_from_an_ib_summary(self):
        last = {"question": "are the RelVals ok there?", "answer": self.SUMMARY, "route": "capability:relval"}
        self.assertEqual(pr_cause._listed_workflows(last), ["36434.0", "36834.0", "537.0"])

    def test_resolved_workflows_and_single_workflow_questions_are_not_listed(self):
        comparison = "**Newly failing:** 1\n- `537.0` (`DY`)\n\n**Resolved:** 1\n- `10805.31` (`SingleGamma`)"
        self.assertEqual(pr_cause._listed_workflows({"question": "compare", "answer": comparison, "route": "capability:comparison"}), [])
        one = {"question": "where is 36434.0 failing?", "answer": self.SUMMARY, "route": "deterministic:workflow_where"}
        self.assertEqual(pr_cause._listed_workflows(one), [])

    def _failure(self, wf, arch=None):
        start = {"536.0": "CMSSW_20_1_X_2026-11-12-1100"}.get(wf, "CMSSW_20_1_X_2026-11-06-1100")
        return {"first_bad": start, "last_good": "CMSSW_20_1_X_2026-11-05-2300", "arch": "el9_amd64_gcc14", "still_failing": True,
                "subject": f"workflow `{wf}` (TTbar_14TeV)", "workflow": wf, "names": ["BTLBaseRecHitSoAProducer"], "packages": [],
                "error": {"category": "PluginNotFound", "message": "Unable to find plugin 'btlrechit::BTLBaseRecHitSoAProducer@alpaka'"}}

    def test_workflows_with_the_same_start_and_pr_are_grouped(self):
        prs = [{"number": 51861, "title": "MTD SoA", "merged_at": "2026-11-06", "patch_status": "ok", "tags": [],
                "files": ["src/RecoLocalFastTime/FTLRecProducers/plugins/alpaka/BTLBaseRecHitSoAProducerAlgo.h"]},
               {"number": 52397, "title": "Other", "merged_at": "2026-11-06", "patch_status": "ok", "tags": [], "files": ["src/Foo/Bar/x.cc"]}]
        with patch.object(pr_cause, "_workflow_failure", side_effect=self._failure), \
                patch.object(pr_cause, "_window_prs", return_value=(prs, ["CMSSW_20_1_X_2026-11-06-1100"])):
            answer = pr_cause._multi_cause_answer(["36434.0", "36834.0"], None, "multi-cause")
        self.assertIn("`36434.0`, `36834.0`", answer)
        self.assertIn("**Most likely: PR #51861**", answer)
        self.assertNotIn("#52397", answer)
        self.assertIn("not proof", answer)
        self.assertEqual(pr_cause._state["multi-cause"]["top"], 51861)


class ClangLeadAndTouchedWindowTests(unittest.TestCase):
    def test_old_clang_data_is_said_first(self):
        class Session:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def run(self, *a, **k): return [{"tag": "CMSSW_20_1_X_2026-08-12-2300"}]

        class Driver:
            def session(self): return Session()
            def close(self): pass

        with patch("server.api_server.get_driver", return_value=Driver()), \
                patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-12-1100"}]):
            lead = _clang_stale_lead()
        self.assertIn("92 days", lead)
        self.assertIn("CMSSW_20_1_X_2026-08-12-2300", lead)

    def test_touched_path_window_counts_opened_or_merged(self):
        from server.api_server import _touched_path_answer
        found = {"path": "RecoLocalFastTime/FTLRecProducers/", "matched_as": "directory", "total_prs": 1, "prs_with_unknown_files_in_window": 0,
                 "unknown_files_but_title_names_it": [], "prs": [{"number": 51861, "title": "MTD SoA", "author": "x", "created_at": "2026-09-11",
                                                                   "merged_at": "2026-11-06", "state": "closed", "files_touched": 22, "example_files": []}]}
        with patch("graph.analytics.prs_touching_path", return_value=found) as touching:
            answer = _touched_path_answer("which PRs touched RecoLocalFastTime/FTLRecProducers last week?")
        self.assertEqual(touching.call_args.kwargs["date_field"], "any")
        self.assertIn("opened or merged", answer)
        with patch("graph.analytics.prs_touching_path", return_value=found) as touching:
            _touched_path_answer("which PRs opened last week touched RecoLocalFastTime/FTLRecProducers?")
        self.assertEqual(touching.call_args.kwargs["date_field"], "created")


if __name__ == "__main__":
    unittest.main()
