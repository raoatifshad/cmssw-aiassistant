import unittest
from unittest.mock import patch

from server.api_server import (
    ChatRequest,
    FeedbackRequest,
    SYSTEM_PROMPT,
    _pending_ib_comparisons,
    _is_latest_ib_question,
    _is_latest_comparison_question,
    _latest_ib_answer,
    _latest_clang_warning_answer,
    _clang_warning_answer,
    _issue_author_answer,
    _issue_detail_answer,
    _merged_prs_for_build_answer,
    _resolve_issue_number,
    _latest_issue_answer,
    _last_issue_contexts,
    _open_issue_answer,
    _artifact_search_answer,
    _build_warning_answer,
    _ib_action_menu,
    _ib_relationship_answer,
    _ib_results_summary_answer,
    _pasted_exception_signature,
    _pasted_exception_search_answer,
    _relval_exception_context,
    _relval_exception_list_answer,
    _relval_exception_answer,
    _normalize_deterministic_question,
    _pr_evidence_answer,
    _pr_search_terms,
    _pending_relval_scopes,
    _pending_unittest_scopes,
    _failed_relval_list_answer,
    _relval_data_availability_answer,
    _relval_error_count_answer,
    _relval_failure_count_answer,
    _relval_ib_summary_answer,
    _relval_scope_or_choice,
    _unittest_summary_answer,
    _failed_unittest_list_answer,
    _workflow_reported_answer,
    _workflow_artifact_answer,
    _requested_ib_arch,
    _execute_planned_capability,
    _failing_relval_context,
    _failed_relval_contexts,
    _fallback_clarification,
    _latest_ib_arch_contexts,
    _latest_ib_contexts,
    _request_planner,
    chat,
    record_feedback,
)
from server.request_planner import Plan, PlanKind, RequestContext
from server.agent import KGChatSession
from graph.tools import TOOLS, get_failing_relvals, search_issues_by_text
from ingestion.build_graph import ib_parent_tag


class ChatRoutingTests(unittest.TestCase):
    @patch("server.api_server.get_driver")
    def test_primary_ib_summary_includes_sub_ib_variants(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {
                "ib_tag": "CMSSW_20_1_X_2026-08-11-2300",
                "variant": None,
                "arch": "el9_amd64_gcc14",
                "is_default": True,
                "build_status": "passed",
                "build_warnings": 0,
                "ignored_warnings": 2,
                "relval_total": 950,
                "relval_failed": 3,
                "unit_test_total": 372,
                "unit_test_failed": 0,
                "unit_test_errors": 0,
            },
            {
                "ib_tag": "CMSSW_20_1_ROOT640_X_2026-08-11-2300",
                "variant": "ROOT640",
                "arch": "el9_amd64_gcc14",
                "is_default": True,
                "build_status": "passed",
                "build_warnings": 4,
                "ignored_warnings": 0,
                "relval_total": 900,
                "relval_failed": 5,
                "unit_test_total": 360,
                "unit_test_failed": 2,
                "unit_test_errors": 7,
            },
        ]

        answer = _ib_results_summary_answer(
            "summary for CMSSW_20_1_X_2026-08-11-2300", None, None
        )

        self.assertIn("primary IB `CMSSW_20_1_X_2026-08-11-2300`", answer)
        self.assertIn("Scope: all imported architectures.", answer)
        self.assertIn("Imported sub-IB variants", answer)
        self.assertIn("`CMSSW_20_1_ROOT640_X_2026-08-11-2300`", answer)
        self.assertIn("ib.parent_tag = $tag", session.run.call_args.args[0])
        get_driver.return_value.close.assert_called_once()

    @patch("server.api_server.get_driver")
    def test_ib_summary_never_hides_non_default_architectures(self, get_driver):
        # Regression: found live via "give the summary for latest IB" -- a
        # primary tag with 4 imported architectures, only 2 flagged
        # is_relval_default, showed only those 2 with no indication the
        # other 2 (el8_amd64_gcc14, el9_amd64_gcc14) existed at all. Unlike
        # the RelVal/unit-test versions of this bug, this one had no
        # ask-when-ambiguous guard whatsoever -- it silently narrowed
        # whenever any default existed, for any count.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        tag = "CMSSW_20_1_X_2026-08-25-2300"
        session.run.return_value = [
            {"ib_tag": tag, "variant": None, "arch": "el10_amd64_gcc15", "is_default": True,
             "build_status": "passed", "build_warnings": 0, "ignored_warnings": 964,
             "relval_total": 40, "relval_failed": 1, "unit_test_total": 368, "unit_test_failed": 4,
             "unit_test_errors": 16},
            {"ib_tag": tag, "variant": None, "arch": "el9_aarch64_gcc14", "is_default": True,
             "build_status": "passed", "build_warnings": 0, "ignored_warnings": 1528,
             "relval_total": 951, "relval_failed": 51, "unit_test_total": 366, "unit_test_failed": 3,
             "unit_test_errors": 15},
            {"ib_tag": tag, "variant": None, "arch": "el8_amd64_gcc14", "is_default": False,
             "build_status": "passed", "build_warnings": 0, "ignored_warnings": 900,
             "relval_total": 951, "relval_failed": 1, "unit_test_total": 373, "unit_test_failed": 2,
             "unit_test_errors": 14},
            {"ib_tag": tag, "variant": None, "arch": "el9_amd64_gcc14", "is_default": False,
             "build_status": "passed", "build_warnings": 0, "ignored_warnings": 2397,
             "relval_total": 951, "relval_failed": 1, "unit_test_total": 373, "unit_test_failed": 2,
             "unit_test_errors": 14},
        ]

        answer = _ib_results_summary_answer(f"give the summary for {tag}", None, None)

        self.assertIn("el10_amd64_gcc15", answer)
        self.assertIn("el9_aarch64_gcc14", answer)
        self.assertIn("el8_amd64_gcc14", answer)
        self.assertIn("el9_amd64_gcc14", answer)

    @patch("server.api_server.get_driver")
    def test_sub_ib_question_explains_parent_relationship(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "tag": "CMSSW_20_1_TF_X_2026-08-14-1100",
            "parent_tag": "CMSSW_20_1_X_2026-08-14-1100",
            "variant": "TF",
        }

        answer = _ib_relationship_answer("why is CMSSW_20_1_TF_X_2026-08-14-1100 a sub IB?")

        self.assertIn("`TF` sub-IB variant", answer)
        self.assertIn("`CMSSW_20_1_X_2026-08-14-1100`", answer)

    def test_llm_knowledge_contract_exposes_live_coverage_tool(self):
        self.assertIn("get_database_overview", TOOLS)
        self.assertIn("get_database_overview", SYSTEM_PROMPT)
        self.assertIn("records from different architectures", SYSTEM_PROMPT)

    @patch("graph.tools._run_query")
    def test_failing_relvals_for_primary_ib_include_sub_ibs(self, run_query):
        run_query.return_value = []

        get_failing_relvals("CMSSW_20_1_X_2026-08-12-1100", arch="el9_amd64_gcc13")

        cypher, params = run_query.call_args.args
        self.assertIn("ib.parent_tag = $t", cypher)
        self.assertIn("ib.tag AS ib_tag", cypher)
        self.assertEqual(params["t"], "CMSSW_20_1_X_2026-08-12-1100")

    @patch("graph.tools._run_query")
    def test_issue_text_search_returns_counts_and_matches(self, run_query):
        run_query.return_value = [{
            "total_matches": 18,
            "open_matches": 2,
            "issues": [{"number": 51449, "title": "(Tiny) Memory Leak", "state": "open"}],
        }]

        result = search_issues_by_text("memory leak")

        self.assertEqual(result["total_matches"], 18)
        self.assertEqual(result["open_matches"], 2)
        self.assertEqual(result["issues"][0]["number"], 51449)
        self.assertEqual(
            run_query.call_args.args[1],
            {"term": "memory leak", "include_comments": False, "limit": 20},
        )

    def test_session_restores_history_and_remembers_deterministic_answer(self):
        history = [
            {"role": "user", "content": "What is Issue #51449?"},
            {"role": "assistant", "content": "It concerns a memory leak."},
        ]
        session = KGChatSession(
            openai_client=None,
            model="gpt-5.6-luna",
            tools_registry={},
            system_prompt="System prompt",
            history=history,
        )

        session.remember("What about its status?", "It is open.")

        self.assertEqual(session.messages[1:3], history)
        self.assertEqual(session.messages[-2:], [
            {"role": "user", "content": "What about its status?"},
            {"role": "assistant", "content": "It is open."},
        ])

    def test_search_terms_ignore_generic_fragments(self):
        self.assertEqual(_pr_search_terms("are there any issues open for them to fix them?"), [])

    def test_search_terms_keep_cmssw_identifiers(self):
        question = "Are there open issues for CMSSW_20_1_X_2026-08-04-2300 workflow 36834.0?"

        self.assertEqual(
            _pr_search_terms(question),
            ["CMSSW_20_1_X_2026-08-04-2300", "36834.0"],
        )

    def test_search_terms_reject_stop_word_package_paths(self):
        # Regression: found live -- "...any PR/Issue created for this?" was
        # searched as the literal term `PR/Issue`, returning an unrelated PR
        # (#29067) instead of the actual workflow ID in the question. The
        # package-path regex (meant for "RecoTracker/TrackProducer") had no
        # stop-word filtering, so any "word/word"-shaped phrase matched.
        # "Workflow" alone must also not survive as a pseudo-identifier --
        # it's a generic domain word, not a specific one, the same class of
        # gap _RELVAL_KEYWORD_RE already treats as a synonym everywhere else.
        question = "why it didnt answer. is this reported Workflow 2025.0010001?? and s there any PR/Issue created for this ?"

        self.assertEqual(_pr_search_terms(question), ["2025.0010001"])

    def test_search_terms_keep_genuine_package_paths(self):
        # The stop-word filter must not reject a real CMSSW package path just
        # because it has the same "word/word" shape as "PR/Issue".
        self.assertEqual(
            _pr_search_terms("RecoTracker/TrackProducer warnings"),
            ["RecoTracker/TrackProducer"],
        )

    @patch("server.api_server.find_prs_for_error")
    def test_pr_evidence_falls_back_to_last_discussed_workflow(self, find_prs):
        # Regression: a real user asked about workflow 140.0's recurrence,
        # then "is there any issue /pr open for it recently?" -- every word in
        # that follow-up is a stopword or has no digit/uppercase, so
        # _pr_search_terms found nothing and the question was rejected
        # outright even though "it" clearly meant the workflow just discussed.
        find_prs.return_value = [{
            "repo": "cmssw", "pr_number": 51000, "title": "Fix workflow 140.0",
            "merged_at": None, "evidence": ["pr_title"], "matched_files": [],
            "matched_unit_tests": False, "matched_unit_test_packages": [],
            "matched_unit_test_ibs": [], "matched_warnings": None,
        }]

        result = _pr_evidence_answer("is there any issue /pr open for it recently?", fallback_term="140.0")

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("140.0", answer)
        find_prs.assert_called_once_with("140.0")

    @patch("server.api_server.find_prs_for_error", return_value=[])
    def test_pr_evidence_prefers_an_explicit_identifier_over_the_fallback(self, find_prs):
        # An identifier actually present in the question must never be
        # overridden by stale remembered context from an earlier turn.
        _pr_evidence_answer("any PR fixing CMSSW_20_1_X_2026-08-04-2300?", fallback_term="140.0")

        find_prs.assert_called_once_with("CMSSW_20_1_X_2026-08-04-2300")

    def test_pr_evidence_without_a_fallback_or_identifier_returns_none(self):
        self.assertIsNone(_pr_evidence_answer("is there any issue /pr open for it recently?"))

    @patch("server.api_server.get_driver")
    def test_ib_results_summary_uses_explicit_architecture(self, get_driver):
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"ib_tag": tag, "arch": "el9_amd64_gcc13", "is_default": False, "build_status": "passed", "build_warnings": 0, "ignored_warnings": 0, "relval_total": 950, "relval_failed": 3, "unit_test_total": 367, "unit_test_failed": 0, "unit_test_errors": 0},
            {"ib_tag": tag, "arch": "el9_amd64_gcc14", "is_default": True, "build_status": "passed", "build_warnings": 0, "ignored_warnings": 0, "relval_total": 950, "relval_failed": 3, "unit_test_total": 367, "unit_test_failed": 0, "unit_test_errors": 0},
        ]

        answer = _ib_results_summary_answer(f"{tag} with el9_amd64_gcc14", tag, None)

        self.assertIn("el9_amd64_gcc14", answer)
        self.assertNotIn("el9_amd64_gcc13", answer)
        self.assertIn("3 / 950", answer)

    @patch("server.api_server._ib_results_summary_answer", return_value="scoped summary")
    def test_planned_summary_offers_all_architectures(self, summary_answer):
        response = chat(ChatRequest(
            question="summary for CMSSW_20_1_X_2026-08-10-1100",
            session_id="summary-actions",
        ))

        self.assertEqual(response.answer, "scoped summary")
        self.assertEqual(len(response.actions), 1)
        self.assertEqual(response.actions[0].label, "All architectures")
        self.assertEqual(response.actions[0].question, "summary for CMSSW_20_1_X_2026-08-10-1100 all")
        summary_answer.assert_called_once()

    @patch("server.api_server.get_driver")
    def test_ib_results_summary_followup_uses_selected_architecture(self, get_driver):
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{
            "ib_tag": tag, "arch": "el9_amd64_gcc14", "is_default": True, "build_status": "passed", "build_warnings": 0,
            "ignored_warnings": 0, "relval_total": 950, "relval_failed": 3, "unit_test_total": 367,
            "unit_test_failed": 0, "unit_test_errors": 0,
        }]

        answer = _ib_results_summary_answer("provide the results", tag, "el9_amd64_gcc14")

        self.assertIn(f"`{tag}`", answer)
        self.assertIn("el9_amd64_gcc14", answer)

    @patch("server.api_server.get_driver")
    def test_bare_ib_request_offers_only_available_graph_actions(self, get_driver):
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "architecture_count": 3,
            "has_build_status": True,
            "has_relvals": True,
            "has_unittests": True,
        }

        answer, actions = _ib_action_menu(tag)

        self.assertIn(f"`{tag}`", answer)
        self.assertEqual(
            [action["label"] for action in actions],
            ["Combined summary", "Build warnings", "RelVal results", "Unit-test results", "All architectures"],
        )
        self.assertEqual(actions[0]["question"], f"summary for {tag}")

    @patch("server.api_server.get_driver")
    def test_build_warning_answer_uses_default_architectures(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"arch": "el8_amd64_gcc13", "status": "passed", "comp_warnings": 0, "ignored_warnings": 3045, "is_default": False},
            {"arch": "el9_amd64_gcc13", "status": "passed", "comp_warnings": 0, "ignored_warnings": 3049, "is_default": False},
            {"arch": "el9_amd64_gcc15", "status": "passed", "comp_warnings": 0, "ignored_warnings": 1038, "is_default": True},
            {"arch": "el9_amd64_gcc16", "status": "warning", "comp_warnings": 464, "ignored_warnings": 1209, "is_default": True},
        ]

        answer = _build_warning_answer("any build warnings for CMSSW_20_1_X_2026-08-09-0000")

        self.assertIn("el9_amd64_gcc15", answer)
        self.assertIn("el9_amd64_gcc16", answer)
        self.assertNotIn("el8_amd64_gcc13", answer)
        self.assertNotIn("el9_amd64_gcc13", answer)
        get_driver.return_value.close.assert_called_once()

    def test_search_terms_keep_quoted_workflow_names(self):
        self.assertEqual(
            _pr_search_terms('find PR evidence for "TTbar_14TeV+Run4D121FS"'),
            ["TTbar_14TeV+Run4D121FS"],
        )

    @patch("server.api_server.get_driver")
    @patch("server.api_server.get_failing_relvals")
    def test_open_issue_search_uses_failed_relvals_when_only_ib_is_named(self, get_failing_relvals_mock, get_driver):
        get_failing_relvals_mock.return_value = [{
            "name": "HydjetQMinBias_5519GeV+Run4D121_hi",
            "workflow_id": "34551.85",
        }]
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{
            "number": 51638,
            "title": "WF 34551.85 step2 failure with FatalRootError",
            "state": "open",
            "matched_terms": ["34551.85"],
        }]

        answer = _open_issue_answer(
            "Are there any open Issues for failing RelVals in CMSSW_20_1_NONLTO_X_2026-08-09-0000?",
            None,
        )

        self.assertIn("Issue #51638", answer)
        get_failing_relvals_mock.assert_called_once_with(
            "CMSSW_20_1_NONLTO_X_2026-08-09-0000", arch=None, limit=100
        )
        self.assertEqual(
            session.run.call_args.kwargs["terms"],
            ["HydjetQMinBias_5519GeV+Run4D121_hi", "34551.85"],
        )

    def test_normalizes_common_deterministic_paraphrases(self):
        self.assertEqual(
            _normalize_deterministic_question("How much Rel-Val failures are in this integration-build?"),
            "how many relval failures are in this ib?",
        )

    def test_requested_ib_architecture_is_extracted(self):
        self.assertEqual(
            _requested_ib_arch("What is the latest IB for el9_amd64_gcc13?"),
            "el9_amd64_gcc13",
        )

    @patch("server.api_server.get_driver")
    def test_named_ib_relval_exception_list_reports_missing_import(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []
        question = "what are the relval exceptions in CMSSW_20_1_X_2026-08-12-1100?"

        answer = _relval_exception_list_answer(question, _relval_exception_context(question))

        self.assertIn("No imported RelVal exception signatures", answer)
        self.assertIn("CMSSW_20_1_X_2026-08-12-1100", answer)
        get_driver.return_value.close.assert_called_once()

    def test_relval_exception_list_requires_exception_keyword(self):
        # A plain relval question (no "exception"/"match") must never reuse a
        # cached exception context — regression guard for the stale-context bug.
        context = {"tag": "CMSSW_20_1_DEVEL_X_2026-08-17-2300", "arch": "el9_amd64_gcc14"}
        self.assertIsNone(
            _relval_exception_list_answer("what relvals are failing in CMSSW_20_1_CLANG_X_2026-08-16-2300", context)
        )

    @patch("server.api_server.get_driver")
    def test_relval_exception_list_answer_recognizes_natural_why_failing_phrasing(self, get_driver):
        # Regression: a real user asked "What are the failed RelVals?" (setting
        # _failed_relval_contexts via _failing_relval_context), then followed up
        # with "Analyze the logs why they are failing" -- a natural phrasing with
        # neither "exception" nor "match" in it. That must still reuse the
        # session's failing-RelVal context instead of falling through to the LLM
        # path with no memory of what was just discussed.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []
        context = _failing_relval_context(
            "what are the failed relvals for CMSSW_20_1_TF_X_2026-08-18-1100 el9_amd64_gcc14"
        )

        answer = _relval_exception_list_answer("Analyze the logs why they are failing", context)

        self.assertIsNotNone(answer)
        get_driver.return_value.close.assert_called_once()

    def test_relval_exception_list_rejects_stale_tag_context(self):
        # A question naming a *different* explicit IB than the cached context
        # must not silently answer using the stale tag.
        context = {"tag": "CMSSW_20_1_DEVEL_X_2026-08-17-2300", "arch": "el9_amd64_gcc14"}
        self.assertIsNone(
            _relval_exception_list_answer(
                "what relval exceptions does CMSSW_20_1_CLANG_X_2026-08-16-2300 have", context
            )
        )

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_relvals")
    def test_relval_capability_lists_failing_workflows_without_repeating_tag(
        self, get_failing_relvals, get_exit_code_names
    ):
        # Regression: a real session asked "how many relvals are failing in
        # <tag>" (which sets _latest_ib_contexts[sid]), then followed up with
        # "can you give the workflow that are failing" -- no tag repeated.
        # This used to hard-fail with "No imported local evidence is
        # available for the selected `relval` request" because (a) none of
        # the relval-* helpers consulted the session's last-mentioned tag,
        # and (b) even a same-turn explicit-tag phrasing of the same request
        # never reached _failed_relval_list_answer, since that function
        # wasn't in the planner's "relval" capability chain at all.
        session_id = "relval-followup-no-tag"
        tag = "CMSSW_20_1_ASAN_X_2026-08-24-2300"
        get_failing_relvals.return_value = [
            {"ib_tag": tag, "name": "136.899", "workflow_id": "136.899", "errors": 3, "exitcode": 1},
        ]
        try:
            _latest_ib_contexts[session_id] = tag
            result = _execute_planned_capability(
                "relval", "can you give the workflow that are failing", session_id
            )
        finally:
            _latest_ib_contexts.pop(session_id, None)
            _failed_relval_contexts.pop(session_id, None)

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("136.899", answer)
        get_failing_relvals.assert_called_once_with(tag, arch=None, limit=100)

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_relvals")
    def test_relval_capability_lists_failing_workflows_for_what_are_failing_phrasing(
        self, get_failing_relvals, get_exit_code_names
    ):
        # Regression: found live -- "what workflows are failing in
        # <tag>" gave the generic pass/fail-count summary table instead of
        # the actual list of failing workflow names. _failed_relval_list_answer
        # only recognized an imperative "list/show/give", not "what X are
        # failing" -- the same request, phrased as a question instead of a
        # command.
        tag = "CMSSW_20_1_X_2026-08-24-2300"
        get_failing_relvals.return_value = [
            {"ib_tag": tag, "name": "140.0", "workflow_id": "140.0", "errors": 2, "exitcode": 1},
        ]

        result = _execute_planned_capability(
            "relval", f"what workflows are failing in the {tag}", "relval-what-are-failing"
        )

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("140.0", answer)
        self.assertIn("failing RelVals for", answer)  # the list heading, not the count-table heading

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_relvals")
    def test_relval_list_answer_recognizes_which_phrasing(self, get_failing_relvals, get_exit_code_names):
        # Same gap as the "what X are failing" fix above, for "which" --
        # found live for the unittest-side equivalent ("which packages are
        # failing"), fixed for both at once since they share the same
        # narrow what-only regex.
        context = {"tag": "CMSSW_20_1_X_2026-08-24-2300", "arch": None}
        get_failing_relvals.return_value = [
            {"ib_tag": context["tag"], "name": "140.0", "workflow_id": "140.0", "errors": 2, "exitcode": 1},
        ]

        answer = _failed_relval_list_answer("which workflows are failing?", context)

        self.assertIsNotNone(answer)
        self.assertIn("140.0", answer)

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_relvals")
    def test_relval_capability_lists_failing_workflows_with_explicit_tag(
        self, get_failing_relvals, get_exit_code_names
    ):
        # Same gap as above, but for a single-turn question that already
        # names the tag -- the generic _relval_ib_summary_answer catch-all
        # used to answer first with a bare pass/fail count table instead of
        # ever reaching the actual workflow list.
        session_id = "relval-explicit-tag-list"
        tag = "CMSSW_20_1_ASAN_X_2026-08-24-2300"
        get_failing_relvals.return_value = [
            {"ib_tag": tag, "name": "136.899", "workflow_id": "136.899", "errors": 3, "exitcode": 1},
        ]
        try:
            result = _execute_planned_capability(
                "relval", f"can you give the workflow that are failing in relvals {tag}", session_id
            )
        finally:
            _failed_relval_contexts.pop(session_id, None)

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("136.899", answer)
        self.assertNotIn("across", answer)  # not the generic scope-summary heading

    @patch("server.api_server._workflow_artifact_answer", return_value=None)
    @patch("server.api_server.get_driver")
    @patch("server.api_server.find_prs_for_error")
    def test_relval_capability_falls_back_to_pr_and_issue_evidence(
        self, find_prs_for_error, get_driver, workflow_artifact_answer
    ):
        # Regression: found live -- "any PR/Issue open for Workflow
        # 2025.0010001?" hard-failed with "No imported local evidence is
        # available for the selected `relval` request". RequestPlanner routes
        # this to "relval" (not "pr_evidence") because "Workflow" triggers
        # the relval category, which statically outranks pr_evidence -- but
        # there's no IB tag and no relval-failure-specific phrasing, so every
        # relval-specific helper returns None. The question was never really
        # about RelVal pass/fail counts; _pr_and_issue_evidence_answer is now
        # tried as a last resort before giving up under the "relval" label.
        # (_workflow_artifact_answer, the more specific exact-text search
        # tried just before this fallback, is stubbed out here so this test
        # isolates the generic-term-search fallback specifically; see
        # test_relval_capability_finds_exact_workflow_text_match below for
        # that more specific path.)
        find_prs_for_error.return_value = [{
            "repo": "cmssw", "pr_number": 51000, "title": "Fix workflow 2025.0010001 step2",
            "merged_at": "2026-08-01T00:00:00Z", "evidence": ["pr_title"],
            "matched_files": [], "matched_unit_tests": [], "matched_unit_test_packages": [],
            "matched_unit_test_ibs": [], "matched_warnings": None, "matched_occurrences": None,
        }]
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []

        result = _execute_planned_capability(
            "relval", "any PR/Issue open for Workflow 2025.0010001?", "relval-pr-issue-fallback"
        )

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("51000", answer)

    @patch("server.api_server.get_driver")
    def test_relval_capability_finds_exact_workflow_text_match(self, get_driver):
        # Sweep finding: _workflow_artifact_answer (exact workflow-ID text
        # match in imported PR/Issue title/body/comments, with a
        # commit-provenance fallback) existed only in the legacy waterfall,
        # unreachable once the planner routed to "relval" or "pr_evidence"
        # deterministically -- same coverage-gap class as everything above,
        # found by auditing every _*_answer() function against
        # _execute_planned_capability's branches instead of one query at a
        # time. It's more specific than the generic term-search fallback
        # (_pr_and_issue_evidence_answer), so it's tried first.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{
            "kind": "Issue", "number": 51640,
            "title": "Workflow 2025.0010001 step2 produces different sized files for different runs",
            "state": "open", "merged_at": None, "matched_field": "title",
        }]

        result = _execute_planned_capability(
            "relval", "any PR/Issue open for Workflow 2025.0010001?", "relval-workflow-artifact"
        )

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("51640", answer)

    @patch("server.api_server.get_driver")
    def test_unittest_capability_uses_session_ib_context_when_tag_omitted(self, get_driver):
        # Regression guard: unlike "relval" (fixed above), the "unittest"
        # capability already threads _latest_ib_contexts.get(session_id) as a
        # fallback tag (_unittest_summary_answer's `context_tag` parameter).
        # Pins that behavior so a future refactor can't silently drop it.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"arch": "el9_amd64_gcc14", "is_default": True, "total": 10, "failed": 1, "passed": 9, "errors": 2},
        ]
        session_id = "unittest-followup-no-tag"
        tag = "CMSSW_20_1_ASAN_X_2026-08-24-2300"
        try:
            _latest_ib_contexts[session_id] = tag
            result = _execute_planned_capability("unittest", "what about unit tests", session_id)
        finally:
            _latest_ib_contexts.pop(session_id, None)

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn(tag, answer)

    @patch("server.api_server.get_failing_unittests")
    def test_unittest_capability_answers_failed_package_list_questions(self, get_failing_unittests):
        # Confirms _failed_unittest_list_answer is actually reachable through
        # the planner's deterministic "unittest" capability route, not just
        # callable in isolation, and that it's tried before the generic
        # summary for a phrasing both could match ("what 3 failed unit
        # tests?").
        get_failing_unittests.return_value = [{"name": "DQM/Physics", "errors": 5}]
        session_id = "unittest-failed-list-capability"
        tag = "CMSSW_20_1_X_2026-08-25-2300"
        try:
            _latest_ib_contexts[session_id] = tag
            _latest_ib_arch_contexts[session_id] = "el9_aarch64_gcc14"
            result = _execute_planned_capability("unittest", "what 3 failed unit tests?", session_id)
        finally:
            _latest_ib_contexts.pop(session_id, None)
            _latest_ib_arch_contexts.pop(session_id, None)

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("DQM/Physics", answer)

    @patch("server.api_server.get_driver")
    def test_clang_capability_uses_session_ib_context_when_tag_omitted(self, get_driver):
        # Regression: fixed alongside the "relval" bug above. A tag-less
        # clang follow-up ("show me the clang warnings again") used to
        # dead-end even with a tag already established this session, because
        # none of the "clang" capability's handlers accepted a fallback tag.
        # _clang_warning_answer (added for the no-handler-at-all gap below)
        # now takes context_tag the same way _unittest_summary_answer does.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"arch": "el9_amd64_gcc14", "is_default": True, "locations": 3, "occurrences": 7},
        ]
        session_id = "clang-followup-no-tag"
        tag = "CMSSW_20_1_ASAN_X_2026-08-24-2300"
        try:
            _latest_ib_contexts[session_id] = tag
            result = _execute_planned_capability("clang", "show me the clang warnings again", session_id)
        finally:
            _latest_ib_contexts.pop(session_id, None)

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn(tag, answer)

    @patch("server.api_server.get_driver")
    def test_clang_warning_answer_handles_plain_tag_qualified_question(self, get_driver):
        # Regression: found live against the real server, not by any mock --
        # "clang warnings for <tag>" failed on its very first turn, tag and
        # all, because the "clang" capability only ever had handlers for
        # "latest" and two-tag comparisons, none for "warnings for this one
        # named IB" (the same phrasing shape that already worked for
        # build/relval/unit-test). _clang_warning_answer closes that gap.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"arch": "el9_amd64_gcc14", "is_default": True, "locations": 3, "occurrences": 7},
        ]
        tag = "CMSSW_20_1_X_2026-08-24-2300"

        result = _execute_planned_capability("clang", f"clang warnings for {tag}", "clang-plain-tag")

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn(tag, answer)
        self.assertIn("7", answer)

    @patch("server.api_server.get_driver")
    def test_build_capability_uses_session_ib_context_when_tag_omitted(self, get_driver):
        # Regression: fixed alongside the "relval" bug above -- a tag-less
        # follow-up that still names the "build" domain (_build_warning_answer
        # requires that word, separately from the tag, to disambiguate from
        # Clang warnings -- untouched by this fix) used to dead-end even with
        # a tag already established this session, because the function only
        # ever looked for a tag in the current question text.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"arch": "el9_amd64_gcc14", "status": "PASSED", "comp_warnings": 2, "ignored_warnings": 1, "is_default": True},
        ]
        session_id = "build-followup-no-tag"
        tag = "CMSSW_20_1_ASAN_X_2026-08-24-2300"
        try:
            _latest_ib_contexts[session_id] = tag
            result = _execute_planned_capability("build", "what about build warnings", session_id)
        finally:
            _latest_ib_contexts.pop(session_id, None)

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn(tag, answer)

    @patch("server.api_server.get_driver")
    @patch("server.api_server.find_prs_for_error")
    def test_pr_evidence_capability_combines_pr_and_open_issue_evidence(self, find_prs_for_error, get_driver):
        # Regression: found live -- "any information about ticl::AssociationMap
        # any PR/issue open?" returned only PR evidence. The "pr_evidence"
        # capability's chain was `_pr_number_answer or _pr_evidence_answer or
        # _recurring_failure_answer`, so once _pr_evidence_answer matched,
        # _open_issue_answer was never even called -- open Issues mentioning
        # the same term (confirmed in the live graph: #42737, #37821, #34000
        # all mention AssociationMap and are open) were silently dropped, not
        # absent. Same "capability matched, one handler swallowed the rest"
        # bug class as the relval/clang/build fixes.
        find_prs_for_error.return_value = [{
            "repo": "cmssw", "pr_number": 51649,
            "title": "Rename SimDataformats TICL associator as TICLAssociationMap",
            "merged_at": "2026-08-21T07:33:48+00:00", "evidence": ["pr_title"],
            "matched_files": [], "matched_unit_tests": [], "matched_unit_test_packages": [],
            "matched_unit_test_ibs": [], "matched_warnings": None, "matched_occurrences": None,
        }]
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"number": 42737, "title": "edm::RefToBase is deprecated", "state": "open",
             "matched_terms": ["AssociationMap"]},
        ]

        result = _execute_planned_capability(
            "pr_evidence", "any information about ticl::AssociationMap any PR/issue open?", "pr-issue-combo"
        )

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("51649", answer)
        self.assertIn("42737", answer)

    @patch("server.api_server.find_prs_before_ib")
    def test_merged_prs_for_build_answer_reports_recently_merged_prs(self, find_prs_before_ib):
        # New feature: "which PRs are merged in <tag>" -- the same "recently
        # merged" list the CMSSDT dashboard shows under a tag's tab. Backed
        # by (PR)-[:MERGED_BEFORE]->(IB), populated by ingest_merged_prs
        # (previously designed and registered as an LLM tool via
        # find_prs_before_ib, but never actually populated by any ingestion
        # function -- always returned empty).
        # find_prs_before_ib is responsible for newest-first ordering itself
        # (see test_orders_newest_merged_at_first_with_undated_prs_last);
        # this fixture is already in that order to isolate what
        # _merged_prs_for_build_answer's own formatting adds -- the merge
        # date, which the user explicitly asked to see displayed, not just
        # used for sorting.
        find_prs_before_ib.return_value = [
            {"repo": "cmssw", "pr_number": 51741, "title": "Update the requested memory size in reused blocks",
             "merged_at": "2026-08-24T17:22:05+00:00", "author": "fwyzard"},
            {"repo": "cmssw", "pr_number": 51750, "title": "Remove testSkipEvent_cfg.py as obsolete",
             "merged_at": None, "author": "makortel"},
        ]

        answer, prs = _merged_prs_for_build_answer(
            "which PR are merged in default CMSSW_20_1_X_2026-08-25-2300"
        )

        self.assertIn("51750", answer)
        self.assertIn("makortel", answer)
        self.assertIn("2026-08-24T17:22:05+00:00", answer)
        self.assertIn("merge date unknown", answer)
        self.assertEqual(len(prs), 2)
        find_prs_before_ib.assert_called_once_with("CMSSW_20_1_X_2026-08-25-2300", limit=100)

    def test_merged_prs_for_build_answer_requires_merged_keyword_and_tag(self):
        self.assertIsNone(_merged_prs_for_build_answer("what is CMSSW_20_1_X_2026-08-25-2300"))
        self.assertIsNone(_merged_prs_for_build_answer("which PRs are merged"))

    @patch("server.api_server.find_prs_before_ib", return_value=[])
    def test_pr_evidence_capability_answers_merged_prs_questions(self, find_prs_before_ib):
        result = _execute_planned_capability(
            "pr_evidence", "which PR are merged in default CMSSW_20_1_X_2026-08-25-2300", "merged-prs-pr-evidence"
        )

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("No imported record", answer)

    @patch("server.api_server.find_prs_before_ib")
    def test_clang_capability_answers_merged_prs_questions_without_pr_word(self, find_prs_before_ib):
        # Regression: found live -- "Clang CMSSW_..._CLANG_X_... what are
        # merged" has no "PR"/"issue" word (only "clang"), so it routes to
        # the "clang" capability, not "pr_evidence" -- same
        # capability-priority pattern as the relval/pr_evidence fix. Without
        # this wiring it would hard-fail under the "clang" label even though
        # the merged-PR data itself was found fine.
        find_prs_before_ib.return_value = [
            {"repo": "cmssw", "pr_number": 51750, "title": "Remove testSkipEvent_cfg.py as obsolete",
             "merged_at": None, "author": "makortel"},
        ]

        result = _execute_planned_capability(
            "clang", "Clang CMSSW_20_1_CLANG_X_2026-08-25-2300 what are merged", "merged-prs-clang"
        )

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("51750", answer)

    @patch("server.api_server._artifact_search_answer", return_value="WRONG: fuzzy text match answer")
    @patch("server.api_server.get_driver")
    @patch("server.api_server.find_prs_before_ib")
    def test_chat_prefers_merged_prs_answer_over_generic_artifact_search(
        self, find_prs_before_ib, get_driver, artifact_search_answer
    ):
        # Regression: found live -- "how many merged in <tag>" has no "PR"
        # word, so RequestPlanner can't classify it into any capability at
        # all, and it falls to the legacy waterfall. _artifact_search_scope
        # treats bare "merged" as its own trigger and a full timestamped IB
        # tag is a real _pr_search_terms identifier, so _artifact_search_answer
        # used to run first and "win" with a technically-true but useless
        # "No imported Issues or PRs match `<tag>`" (a fuzzy text search for
        # the literal tag string in PR/Issue bodies) instead of the
        # structured MERGED_BEFORE lookup this phrasing actually needs.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.__iter__.return_value = iter([])
        session.run.return_value.single.return_value = None
        find_prs_before_ib.return_value = [
            {"repo": "cmssw", "pr_number": 51750, "title": "Remove testSkipEvent_cfg.py as obsolete",
             "merged_at": None, "author": "makortel"},
        ]

        result = chat(ChatRequest(question="how many merged in CMSSW_20_1_MULTIARCHS_X_2026-08-25-2300"))

        self.assertIn("51750", result.answer)
        artifact_search_answer.assert_not_called()

    def test_workflow_exception_search_requires_ib_context(self):
        self.assertEqual(
            _relval_exception_answer("Search GitHub for the exception in workflow 37834.0", None),
            "Please provide the Integration Build tag before searching an exception by workflow ID.",
        )

    @patch("server.api_server.get_driver")
    def test_pasted_exception_search_preserves_category_name(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.side_effect = [
            unittest.mock.Mock(single=lambda: {"total": 21}),
            [{
                "kind": "Issue",
                "number": 52123,
                "title": "ProductNotFound in data workflow",
                "state": "open",
                "merged_at": None,
                "matched_field": "comment",
                "matched_evidence": "An exception of category 'ProductNotFound' occurred while processing an event",
            }],
        ]

        exception = _pasted_exception_signature(
            "An exception of category 'ProductNotFound' occurred while processing the event"
        )
        answer, github_matches = _pasted_exception_search_answer(exception)

        self.assertIn("ProductNotFound", answer)
        self.assertIn("Issue #52123", answer)
        self.assertIn("1-1 of 21", answer)
        self.assertIn("more results", answer)
        self.assertEqual(github_matches, [{
            "number": 52123,
            "title": "ProductNotFound in data workflow",
            "url": "https://github.com/cms-sw/cmssw/issues/52123",
            "type": "Issue",
            "match_evidence": "An exception of category 'ProductNotFound' occurred while processing an event",
            "match_source": "comment",
            "exact_match": True,
        }])
        self.assertEqual(
            session.run.call_args_list[1].kwargs["exception"],
            "An exception of category 'ProductNotFound' occurred while",
        )

    def test_variant_ib_has_its_primary_run_as_parent(self):
        self.assertEqual(
            ib_parent_tag("CMSSW_20_1_NONLTO_X_2026-08-10-2300"),
            ("CMSSW_20_1_X_2026-08-10-2300", "NONLTO"),
        )
        self.assertEqual(
            ib_parent_tag("CMSSW_20_1_X_2026-08-10-2300"),
            ("CMSSW_20_1_X_2026-08-10-2300", None),
        )

    @patch("server.api_server._relval_scope_records")
    def test_named_relval_error_count_is_deterministic(self, scope_records):
        scope_records.return_value = [{
            "tag": "CMSSW_20_1_EVTGEN3_X_2026-08-09-2300", "arch": "el9_amd64_gcc14",
            "is_relval_default": True,
            "relval_total": 950, "error_count": 100777,
            "failed_relvals": 13, "failed_errors": 126,
            "passed_relvals": 937, "passed_errors": 100651,
        }]

        answer = _relval_error_count_answer(
            "how many relvals errors in CMSSW_20_1_EVTGEN3_X_2026-08-09-2300", "s1"
        )

        self.assertIn("13 failed RelVal(s)", answer)
        self.assertIn("950 RelVal record(s)", answer)
        self.assertIn("1 IB record(s)", answer)
        self.assertIn("100,651", answer)
        self.assertIn("126", answer)
        scope_records.assert_called_once_with("CMSSW_20_1_EVTGEN3_X_2026-08-09-2300", include_sub_ibs=True)

    @patch("server.api_server._relval_scope_records")
    def test_primary_relval_failure_count_lists_every_architecture_up_front(self, scope_records):
        # A plain question with no architecture marked is_relval_default at
        # all is no longer ambiguous enough to justify a round-trip -- with
        # nothing to guess as "the" architecture, show every one directly
        # instead of making the user pick (see
        # test_relval_scope_or_choice_asks_only_when_multiple_defaults_tie
        # for the one case that still asks: >1 simultaneous defaults).
        session_id = "relval-primary-scope"
        scope_records.return_value = [
            {
                "tag": "CMSSW_20_1_X_2026-08-13-2300",
                "arch": "el9_amd64_gcc14",
                "relval_total": 3700,
                "failed_relvals": 4,
                "passed_relvals": 3696,
            },
            {
                "tag": "CMSSW_20_1_X_2026-08-13-2300",
                "arch": "el10_amd64_gcc15",
                "relval_total": 3700,
                "failed_relvals": 4,
                "passed_relvals": 3696,
            },
        ]

        answer = _relval_failure_count_answer(
            "how many relvals failing in CMSSW_20_1_X_2026-08-13-2300", session_id
        )

        self.assertIn("el9_amd64_gcc14", answer)
        self.assertIn("el10_amd64_gcc15", answer)
        self.assertIn("8 failing RelVal(s)", answer)
        self.assertNotIn(session_id, _pending_relval_scopes)

    @patch("server.api_server._relval_scope_records")
    def test_relval_failure_count_all_returns_aggregate(self, scope_records):
        session_id = "relval-all-scope"
        scope_records.return_value = [
            {
                "tag": "CMSSW_20_1_CLANG_X_2026-08-13-2300",
                "arch": "el9_amd64_gcc14",
                "relval_total": 3692,
                "failed_relvals": 36,
                "passed_relvals": 3656,
            },
            {
                "tag": "CMSSW_20_1_X_2026-08-13-2300",
                "arch": "el9_amd64_gcc14",
                "relval_total": 3700,
                "failed_relvals": 4,
                "passed_relvals": 3696,
            },
        ]
        answer = _relval_failure_count_answer(
            "how many relvals failing in all CMSSW_20_1_X_2026-08-13-2300", session_id
        )
        self.assertIn("40 failing RelVal(s)", answer)
        self.assertIn("All imported RelVal results", answer)

    def test_relval_scope_or_choice_asks_only_when_multiple_defaults_tie(self):
        # The one remaining case _relval_scope_or_choice still asks about:
        # a real production incident where an IB had multiple architectures
        # simultaneously marked is_relval_default, which is genuinely
        # ambiguous (there's no single "the" architecture to default to).
        # Zero or exactly one default architecture is never ambiguous this
        # way -- every other case returns every record directly.
        tag = "CMSSW_20_1_X_2026-08-24-2300"
        two_defaults = [
            {"tag": tag, "arch": "el9_amd64_gcc14", "is_relval_default": True},
            {"tag": tag, "arch": "el9_aarch64_gcc14", "is_relval_default": True},
        ]
        try:
            result = _relval_scope_or_choice("scope-choice-tie", tag, two_defaults, all_requested=False)
            self.assertIsInstance(result, str)
            self.assertIn("architecture-specific", result)

            one_default = [
                {"tag": tag, "arch": "el9_amd64_gcc14", "is_relval_default": True},
                {"tag": tag, "arch": "el8_amd64_gcc14", "is_relval_default": False},
            ]
            result = _relval_scope_or_choice("scope-choice-one", tag, one_default, all_requested=False)
            self.assertEqual(result, one_default)

            no_defaults = [
                {"tag": tag, "arch": "el9_amd64_gcc14", "is_relval_default": False},
                {"tag": tag, "arch": "el8_amd64_gcc14", "is_relval_default": False},
            ]
            result = _relval_scope_or_choice("scope-choice-none", tag, no_defaults, all_requested=False)
            self.assertEqual(result, no_defaults)
        finally:
            _pending_relval_scopes.pop("scope-choice-tie", None)

    @patch("server.api_server._relval_scope_records")
    @patch("server.api_server.get_driver")
    def test_relval_failure_count_mentions_sub_ib_variants(self, get_driver, scope_records):
        # Sub-IB variants (ASAN, UBSAN, ROOT6, ...) are never merged into a
        # primary tag's RelVal numbers (different (tag, architecture)
        # identities, per CLAUDE.md), but their existence should be
        # mentioned rather than silently omitted -- found live: a real
        # question about a primary tag gave no hint that 10 sub-IB variants
        # with very different failure profiles existed for the same build.
        tag = "CMSSW_20_1_X_2026-08-24-2300"
        scope_records.return_value = [
            {"tag": tag, "arch": "el9_amd64_gcc14", "relval_total": 951, "failed_relvals": 1, "passed_relvals": 950, "is_relval_default": False},
        ]
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"variant": "ASAN"}, {"variant": "UBSAN"}]

        answer = _relval_failure_count_answer(f"how many relvals failing in {tag}", "relval-sub-ib-mention")

        self.assertIn("2 sub-IB variant(s)", answer)
        self.assertIn("ASAN", answer)
        self.assertIn("UBSAN", answer)

    @patch("server.api_server._relval_scope_records")
    def test_sub_ib_relval_failure_count_lists_every_architecture_up_front(self, scope_records):
        session_id = "relval-sub-ib-scope"
        sub_ib = "CMSSW_20_1_CLANG_X_2026-08-13-2300"
        scope_records.return_value = [
            {
                "tag": sub_ib,
                "arch": "el9_aarch64_gcc14",
                "relval_total": 500,
                "failed_relvals": 7,
                "passed_relvals": 493,
            },
            {
                "tag": sub_ib,
                "arch": "el9_amd64_gcc14",
                "relval_total": 500,
                "failed_relvals": 2,
                "passed_relvals": 498,
            },
        ]

        answer = _relval_failure_count_answer(
            f"how many failed relvals in {sub_ib}", session_id
        )

        self.assertIn("el9_aarch64_gcc14", answer)
        self.assertIn("el9_amd64_gcc14", answer)
        self.assertIn("9 failing RelVal(s)", answer)
        self.assertNotIn(session_id, _pending_relval_scopes)

    @patch("server.api_server._relval_scope_records")
    def test_relval_failure_count_rejects_invalid_architecture_choice(self, scope_records):
        # The pending-choice flow now only triggers when >1 architecture is
        # simultaneously marked is_relval_default (genuinely ambiguous which
        # one is "the" build) -- both records need that flag set for this
        # fixture to still exercise the choice/rejection path being tested.
        session_id = "relval-invalid-scope"
        primary_ib = "CMSSW_20_1_X_2026-08-13-2300"
        scope_records.return_value = [
            {
                "tag": primary_ib,
                "arch": "el9_amd64_gcc14",
                "relval_total": 3700,
                "failed_relvals": 4,
                "passed_relvals": 3696,
                "is_relval_default": True,
            },
            {
                "tag": primary_ib,
                "arch": "el10_amd64_gcc15",
                "relval_total": 3700,
                "failed_relvals": 1,
                "passed_relvals": 3699,
                "is_relval_default": True,
            },
        ]
        try:
            _relval_failure_count_answer(f"how many failed relvals in {primary_ib}", session_id)
            answer = _relval_failure_count_answer("el11_amd64_gcc16", session_id)
            self.assertIn("not available", answer)
            self.assertIn(session_id, _pending_relval_scopes)
        finally:
            _pending_relval_scopes.pop(session_id, None)

    @patch("server.api_server._relval_scope_records")
    def test_any_failing_relvals_with_one_architecture_returns_exact_ib(self, scope_records):
        session_id = "relval-any-scope"
        scope_records.return_value = [{
            "tag": "CMSSW_20_1_X_2026-08-10-1100",
            "arch": "el9_amd64_gcc14",
            "relval_total": 950,
            "failed_relvals": 3,
            "passed_relvals": 947,
        }]
        try:
            answer = _relval_failure_count_answer(
                "any relvals failing in CMSSW_20_1_X_2026-08-10-1100", session_id
            )
            self.assertIn("3 failing RelVal(s)", answer)
            self.assertNotIn(session_id, _pending_relval_scopes)
            scope_records.assert_called_once_with(
                "CMSSW_20_1_X_2026-08-10-1100", include_sub_ibs=False
            )
        finally:
            _pending_relval_scopes.pop(session_id, None)

    @patch("server.api_server._relval_scope_records")
    def test_relval_failure_count_never_hides_non_default_architectures(self, scope_records):
        # Regression: found live -- a primary tag's only is_relval_default
        # architecture had 51/951 RelVal failures while two other, non-default
        # architectures had 1/951 each. The old logic narrowed to just the
        # single default record and silently dropped the other two, so a
        # plain "how many relvals failing" answer showed only the worst
        # architecture's number with no indication the cleaner ones existed.
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        scope_records.return_value = [
            {"tag": tag, "arch": "el8_amd64_gcc13", "relval_total": 950, "failed_relvals": 1, "passed_relvals": 949, "is_relval_default": False},
            {"tag": tag, "arch": "el9_amd64_gcc13", "relval_total": 950, "failed_relvals": 1, "passed_relvals": 949, "is_relval_default": False},
            {"tag": tag, "arch": "el9_amd64_gcc14", "relval_total": 950, "failed_relvals": 51, "passed_relvals": 899, "is_relval_default": True},
        ]

        answer = _relval_failure_count_answer(f"any relvals failing in {tag}", "relval-default-arch")

        self.assertIn("el9_amd64_gcc14", answer)
        self.assertIn("el8_amd64_gcc13", answer)
        self.assertIn("el9_amd64_gcc13", answer)
        self.assertIn("53 failing RelVal(s)", answer)
        self.assertNotIn("architecture-specific", answer)

    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_asks_when_multiple_default_architectures_exist(self, scope_records):
        # Regression: a real IB had 4 architectures marked default
        # simultaneously, and their unit-test counts were silently added
        # together into one misleading headline number ("6 failed") instead
        # of asking which architecture, same bug class fixed for RelVals.
        scope_records.return_value = [
            {"arch": "el9_amd64_gcc13", "is_default": False, "total": 950, "failed": 3, "passed": 947, "errors": 3},
            {"arch": "el9_amd64_gcc15", "is_default": True, "total": 950, "failed": 3, "passed": 947, "errors": 3},
            {"arch": "el9_amd64_gcc16", "is_default": True, "total": 950, "failed": 4, "passed": 946, "errors": 4},
        ]

        answer = _unittest_summary_answer("and unit tests", "CMSSW_20_1_X_2026-08-09-0000", "s-multi-unittest")

        self.assertIn("Choose one", answer)
        self.assertIn("el9_amd64_gcc15", answer)
        self.assertIn("el9_amd64_gcc16", answer)
        self.assertNotIn("el9_amd64_gcc13", answer)
        self.assertIn("s-multi-unittest", _pending_unittest_scopes)
        _pending_unittest_scopes.pop("s-multi-unittest", None)

    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_never_hides_non_default_architectures(self, scope_records):
        # Regression: a build with only ONE architecture flagged is_default
        # among several imported used to silently narrow to just that one,
        # dropping the rest with no indication they existed -- the same bug
        # already fixed for RelVals
        # (test_relval_failure_count_never_hides_non_default_architectures).
        # Distinct from the >1-simultaneous-default "ask" case, covered by
        # test_unittest_summary_asks_when_multiple_default_architectures_exist
        # and test_unittest_summary_all_recovers_full_architecture_set below.
        scope_records.return_value = [
            {"arch": "el8_amd64_gcc14", "is_default": False, "total": 373, "failed": 2, "passed": 371, "errors": 2},
            {"arch": "el9_amd64_gcc14", "is_default": False, "total": 373, "failed": 2, "passed": 371, "errors": 2},
            {"arch": "el9_aarch64_gcc14", "is_default": True, "total": 366, "failed": 3, "passed": 363, "errors": 15},
        ]

        answer = _unittest_summary_answer("and unit tests", "CMSSW_20_1_X_2026-08-25-2300", "s-4arch-unittest")

        self.assertIn("el8_amd64_gcc14", answer)
        self.assertIn("el9_amd64_gcc14", answer)
        self.assertIn("el9_aarch64_gcc14", answer)
        self.assertIn("7 failed", answer)  # 2+2+3

    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_all_recovers_full_architecture_set(self, scope_records):
        # Regression: found live -- replying "all" to the multi-default
        # choice prompt only returned the pre-filtered tied-default subset
        # (2 of 4 architectures), because the pending state itself was
        # already narrowed before "all" was ever offered as an option.
        session_id = "s-4arch-all-unittest"
        scope_records.return_value = [
            {"arch": "el8_amd64_gcc14", "is_default": False, "total": 373, "failed": 2, "passed": 371, "errors": 2},
            {"arch": "el9_amd64_gcc14", "is_default": False, "total": 373, "failed": 2, "passed": 371, "errors": 2},
            {"arch": "el9_aarch64_gcc14", "is_default": True, "total": 366, "failed": 3, "passed": 363, "errors": 15},
            {"arch": "el10_amd64_gcc15", "is_default": True, "total": 368, "failed": 4, "passed": 364, "errors": 16},
        ]
        try:
            first = _unittest_summary_answer("and unit tests", "CMSSW_20_1_X_2026-08-25-2300", session_id)
            self.assertIn("Choose one", first)

            answer = _unittest_summary_answer("all", None, session_id)
        finally:
            _pending_unittest_scopes.pop(session_id, None)

        self.assertIn("el8_amd64_gcc14", answer)
        self.assertIn("el9_amd64_gcc14", answer)
        self.assertIn("el9_aarch64_gcc14", answer)
        self.assertIn("el10_amd64_gcc15", answer)
        self.assertIn("11 failed", answer)

    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_remembers_session_architecture_for_tagless_followup(self, scope_records):
        # Regression: found live -- "how many unit tests failing in <tag>
        # <arch>" (explicit tag+arch) resolved fine, but a tag-less,
        # arch-less follow-up in the same session ("what 3 failed unit
        # tests?") re-asked "choose an architecture" from scratch instead of
        # reusing the architecture already named this session, the same
        # pronoun-resolution pattern _latest_ib_contexts provides for tags.
        scope_records.return_value = [
            {"arch": "el9_aarch64_gcc14", "is_default": True, "total": 366, "failed": 3, "passed": 363, "errors": 15},
            {"arch": "el10_amd64_gcc15", "is_default": True, "total": 368, "failed": 4, "passed": 364, "errors": 16},
        ]

        answer = _unittest_summary_answer(
            "what 3 failed unit tests", "CMSSW_20_1_X_2026-08-25-2300", "s-remembered-arch",
            context_arch="el9_aarch64_gcc14",
        )

        self.assertIn("3 failed", answer)
        self.assertNotIn("Choose one", answer)

    @patch("server.api_server.get_failing_unittests")
    def test_failed_unittest_list_answer_lists_package_names(self, get_failing_unittests):
        # Regression: found live -- get_failing_unittests was already
        # registered as an LLM tool, but nothing in the deterministic path
        # ever called it; only the aggregate count existed
        # (_unittest_summary_answer), never the actual failed package
        # names, the same "designed but only half-built" pattern as
        # MERGED_BEFORE.
        get_failing_unittests.return_value = [
            {"name": "DQM/Physics", "errors": 5},
            {"name": "RecoTracker/TrackProducer", "errors": 2},
        ]

        answer = _failed_unittest_list_answer(
            "what 3 failed packages?", "CMSSW_20_1_X_2026-08-25-2300", "el9_aarch64_gcc14",
        )

        self.assertIn("DQM/Physics", answer)
        self.assertIn("RecoTracker/TrackProducer", answer)
        get_failing_unittests.assert_called_once_with(
            "CMSSW_20_1_X_2026-08-25-2300", arch="el9_aarch64_gcc14", limit=100
        )

    @patch("server.api_server.get_failing_unittests")
    def test_failed_unittest_list_answer_recognizes_which_phrasing(self, get_failing_unittests):
        # Regression: found live -- "which packages are failing" has neither
        # "list/show/give" nor "what", only "which"; the trigger only
        # recognized "what X are failing", the same narrow-synonym gap
        # already fixed for _failed_relval_list_answer's "which workflows
        # are failing" equivalent.
        get_failing_unittests.return_value = [{"name": "DQM/Physics", "errors": 5}]

        answer = _failed_unittest_list_answer(
            "which packages are failing?", "CMSSW_20_1_X_2026-08-25-2300", "el9_aarch64_gcc14",
        )

        self.assertIn("DQM/Physics", answer)

    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_resolves_after_choosing_an_architecture(self, scope_records):
        scope_records.return_value = [
            {"arch": "el9_amd64_gcc15", "is_default": True, "total": 950, "failed": 3, "passed": 947, "errors": 3},
            {"arch": "el9_amd64_gcc16", "is_default": True, "total": 950, "failed": 4, "passed": 946, "errors": 4},
        ]
        _unittest_summary_answer("and unit tests", "CMSSW_20_1_X_2026-08-09-0000", "s-resume-unittest")

        answer = _unittest_summary_answer("el9_amd64_gcc16", None, "s-resume-unittest")

        self.assertIn("4 failed", answer)
        self.assertNotIn("s-resume-unittest", _pending_unittest_scopes)

    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_honors_an_explicit_architecture_up_front(self, scope_records):
        # An architecture named directly in the first question must resolve
        # immediately -- no reason to ask when the user already said which one.
        scope_records.return_value = [
            {"arch": "el9_amd64_gcc15", "is_default": True, "total": 950, "failed": 3, "passed": 947, "errors": 3},
            {"arch": "el9_amd64_gcc16", "is_default": True, "total": 950, "failed": 4, "passed": 946, "errors": 4},
        ]

        answer = _unittest_summary_answer(
            "unit tests for CMSSW_20_1_X_2026-08-09-0000 el9_amd64_gcc16", None, "s-explicit-unittest",
        )

        self.assertIn("4 failed", answer)
        self.assertNotIn("s-explicit-unittest", _pending_unittest_scopes)

    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_recognizes_unit_without_the_word_test(self, scope_records):
        # Regression: a real user asked "unit failures in <tag>" -- no "test"
        # word at all -- and the old \bunit\s*tests?\b regex required "test"
        # to be present, so this fell through to the ambiguous-category
        # clarification instead of answering directly.
        scope_records.return_value = [
            {"arch": "el9_amd64_gcc14", "is_default": True, "total": 372, "failed": 2, "passed": 370, "errors": 0},
        ]

        answer = _unittest_summary_answer("unit failures in CMSSW_20_1_PY314_X_2026-08-19-1100", None, "s1")

        self.assertIsNotNone(answer)
        self.assertIn("2 failed", answer)

    @patch("server.api_server._relval_scope_records")
    def test_repeated_relval_question_restarts_pending_scope(self, scope_records):
        # Both records need is_relval_default set so the pending-choice flow
        # this test exercises still triggers (see
        # test_relval_failure_count_rejects_invalid_architecture_choice).
        session_id = "relval-repeated-question"
        primary_ib = "CMSSW_20_1_X_2026-08-10-1100"
        scope_records.return_value = [
            {
                "tag": primary_ib,
                "arch": "el9_aarch64_gcc13",
                "relval_total": 950,
                "failed_relvals": 53,
                "passed_relvals": 897,
                "is_relval_default": True,
            },
            {
                "tag": primary_ib,
                "arch": "el9_amd64_gcc13",
                "relval_total": 950,
                "failed_relvals": 3,
                "passed_relvals": 947,
                "is_relval_default": True,
            },
        ]
        question = f"any relvals failing in {primary_ib}"
        try:
            _relval_failure_count_answer(question, session_id)
            answer = _relval_failure_count_answer(question, session_id)

            self.assertIn("architecture-specific", answer)
            self.assertIn("el9_aarch64_gcc13", answer)
            self.assertNotIn("CMSSW_20_1_TF_X_2026-08-10-1100", answer)
        finally:
            _pending_relval_scopes.pop(session_id, None)

    @patch("server.api_server._relval_scope_records")
    def test_relval_data_availability_is_graph_backed(self, scope_records):
        scope_records.return_value = [{
            "tag": "CMSSW_20_1_X_2026-08-10-1100",
            "arch": "el9_amd64_gcc14",
            "relval_total": 950,
            "failed_relvals": 3,
            "passed_relvals": 947,
        }]

        answer = _relval_data_availability_answer(
            "do you have data for CMSSW_20_1_X_2026-08-10-1100?", "s1"
        )

        self.assertIn("Imported RelVal data", answer)
        self.assertIn("3 failing RelVal(s)", answer)
        scope_records.assert_called_once_with(
            "CMSSW_20_1_X_2026-08-10-1100", include_sub_ibs=False
        )

    @patch("server.api_server._relval_scope_records")
    def test_relval_data_availability_asks_when_multiple_default_architectures_exist(self, scope_records):
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        scope_records.return_value = [
            {"tag": tag, "arch": "el9_amd64_gcc15", "relval_total": 950, "failed_relvals": 3, "passed_relvals": 947, "is_relval_default": True},
            {"tag": tag, "arch": "el9_amd64_gcc16", "relval_total": 950, "failed_relvals": 4, "passed_relvals": 946, "is_relval_default": True},
        ]

        answer = _relval_data_availability_answer(f"do you have data for {tag}?", "s-avail-multi")

        self.assertIn("Choose one", answer)
        _pending_relval_scopes.pop("s-avail-multi", None)

    @patch("server.api_server._relval_scope_records")
    def test_relval_error_count_asks_when_multiple_default_architectures_exist(self, scope_records):
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        scope_records.return_value = [
            {"tag": tag, "arch": "el9_amd64_gcc15", "relval_total": 950, "failed_relvals": 3, "passed_relvals": 947,
             "error_count": 100, "failed_errors": 10, "passed_errors": 90, "is_relval_default": True},
            {"tag": tag, "arch": "el9_amd64_gcc16", "relval_total": 950, "failed_relvals": 4, "passed_relvals": 946,
             "error_count": 200, "failed_errors": 20, "passed_errors": 180, "is_relval_default": True},
        ]

        answer = _relval_error_count_answer(f"how many relvals errors in {tag}", "s-errcount-multi")

        self.assertIn("Choose one", answer)
        scope_records.assert_called_once_with(tag, include_sub_ibs=True)
        _pending_relval_scopes.pop("s-errcount-multi", None)

    @patch("server.api_server._relval_scope_records")
    def test_ambiguous_named_relval_question_defaults_to_graph_summary(self, scope_records):
        scope_records.return_value = [{
            "tag": "CMSSW_20_1_X_2026-08-10-1100",
            "arch": "el9_amd64_gcc14",
            "relval_total": 950,
            "failed_relvals": 3,
            "passed_relvals": 947,
        }]

        answer = _relval_ib_summary_answer(
            "what RelVal results are in CMSSW_20_1_X_2026-08-10-1100?", "s1"
        )

        self.assertIn("Imported RelVal results", answer)
        self.assertIn("3 failing RelVal(s)", answer)

    @patch("server.api_server._relval_scope_records")
    def test_generic_relval_summary_asks_when_multiple_default_architectures_exist(self, scope_records):
        # Regression: silently summing every "default" architecture into one
        # combined headline number is misleading when more than one exists
        # (a real production IB had 4 simultaneously marked default) -- this
        # must ask which architecture instead of guessing, same as the
        # already-correct _relval_failure_count_answer.
        tag = "CMSSW_20_1_X_2026-08-09-0000"
        scope_records.return_value = [
            {"tag": tag, "arch": "el9_amd64_gcc13", "relval_total": 950, "failed_relvals": 5, "passed_relvals": 945, "is_relval_default": False},
            {"tag": tag, "arch": "el9_amd64_gcc15", "relval_total": 950, "failed_relvals": 5, "passed_relvals": 945, "is_relval_default": True},
            {"tag": tag, "arch": "el9_amd64_gcc16", "relval_total": 950, "failed_relvals": 5, "passed_relvals": 945, "is_relval_default": True},
        ]

        answer = _relval_ib_summary_answer(f"show RelVal results for {tag}", "s-multi-default")

        self.assertIn("Choose one", answer)
        self.assertIn("el9_amd64_gcc15", answer)
        self.assertIn("el9_amd64_gcc16", answer)
        self.assertNotIn("el9_amd64_gcc13", answer)
        self.assertEqual(_pending_relval_scopes["s-multi-default"]["stage"], "arch")
        _pending_relval_scopes.pop("s-multi-default", None)

    def test_relval_summary_preserves_explicit_issue_intent(self):
        self.assertIsNone(_relval_ib_summary_answer(
            "are there open Issues for failed RelVals in CMSSW_20_1_X_2026-08-10-1100?", "s1"
        ))

    @patch("server.api_server.get_exit_code_names", return_value={16640: "OtherCMS"})
    @patch("server.api_server.get_driver")
    def test_workflow_reported_without_an_ib_lists_matching_records(self, get_driver, exit_code_names):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "total": 2,
            "records": [{
                "ib_tag": "CMSSW_20_1_X_2026-08-11-2300",
                "arch": "el10_amd64_gcc15",
                "status": "PASSED",
                "errors": 0,
                "exitcode": 0,
            }, {
                "ib_tag": "CMSSW_20_1_X_2026-08-12-1100",
                "arch": "el10_amd64_gcc15",
                "status": "FAILED",
                "errors": 2,
                "exitcode": 16640,
            }],
        }

        answer = _workflow_reported_answer("does WF 140.051 reported?")

        self.assertIn("Workflow `140.051` is reported in **2**", answer)
        self.assertIn("The 2 most recent", answer)
        self.assertIn("CMSSW_20_1_X_2026-08-11-2300", answer)
        self.assertIn("CMSSW_20_1_X_2026-08-12-1100", answer)
        self.assertIn("exit code: 16640 (OtherCMS)", answer)
        self.assertEqual(session.run.call_args.kwargs["workflow_id"], "140.051")
        get_driver.return_value.close.assert_called_once()

    @patch("server.api_server.get_driver")
    def test_workflow_artifact_search_requires_an_exact_workflow_mention(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{
            "kind": "PR",
            "number": 51667,
            "title": "comment failing wf 37834.0 until there is a suitable PU library",
            "state": "closed",
            "merged_at": "2026-08-10T00:00:00+00:00",
            "matched_field": "title",
        }]

        answer = _workflow_artifact_answer("can you specify the PR/issues for 37834.0")

        self.assertIn("PR #51667", answer)
        self.assertIn("exact match in title", answer)
        self.assertEqual(session.run.call_args.kwargs["workflow_id"], "37834.0")
        get_driver.return_value.close.assert_called_once()

    @patch("server.api_server.get_driver")
    def test_relval_exception_search_returns_imported_artifact_matches(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        exception_text = "An exception of category 'NoSecondaryFiles' occurred while"
        session.run.side_effect = [
            [{"exception": exception_text, "category": None, "source_url": None, "context": None, "name": "TTbar"}],
            [{
                "kind": "PR",
                "number": 44029,
                "title": "Tau DQM miniAOD validation for DATA",
                "state": "closed",
                "merged_at": "2024-02-20T00:00:00+00:00",
                "matched_terms": [exception_text],
            }],
            [],
        ]

        answer = _relval_exception_answer(
            "Search GitHub for exception in workflow 37834.0",
            {"tag": "CMSSW_20_1_X_2026-08-12-1100", "arch": "el9_amd64_gcc13"},
        )

        self.assertIn("NoSecondaryFiles", answer)
        self.assertIn("PR #44029", answer)
        self.assertIn("https://github.com/search?q=repo%3Acms-sw%2Fcmssw+An+exception", answer)
        self.assertEqual(session.run.call_args_list[0].kwargs["workflow_id"], "37834.0")
        self.assertIn(exception_text, session.run.call_args_list[1].kwargs["terms"])
        get_driver.return_value.close.assert_called_once()

    @patch("server.api_server.get_driver")
    def test_relval_exception_answer_labels_merged_pr_and_open_issue(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        exception_text = "LogicError: A muon candidate track has more than 5 stations"
        session.run.side_effect = [
            [{
                "exception": exception_text, "category": "LogicError", "source_url": "https://example/log#/1-1",
                "context": "Calling method for module EvaluateMuonMVAID/'muonMVAID'", "name": "RunMuonEG2022C",
            }],
            [
                {
                    "kind": "PR", "number": 51375, "title": "Unrelated PR", "state": "closed",
                    "merged_at": "2026-08-01T00:00:00+00:00", "matched_terms": ["RunMuonEG2022C", "140.051"],
                },
                {
                    "kind": "PR", "number": 50360, "title": "Touches EvaluateMuonMVAID", "state": "closed",
                    "merged_at": "2026-08-02T00:00:00+00:00", "matched_terms": ["EvaluateMuonMVAID"],
                },
                {
                    "kind": "Issue", "number": 51354, "title": "LogicError in EvaluateMuonMVAID", "state": "open",
                    "merged_at": None, "matched_terms": ["EvaluateMuonMVAID", "RunMuonEG2022C"],
                },
            ],
            [],
        ]

        answer = _relval_exception_answer(
            "what relval exception does workflow 140.051 have?",
            {"tag": "CMSSW_20_1_DEVEL_X_2026-08-17-2300", "arch": "el9_amd64_gcc14"},
        )

        # The PR matched only on the weak workflow-name/ID terms (routine CI-comment
        # noise) must be dropped; the one matched on the specific module name, and
        # the open Issue, must both survive.
        self.assertNotIn("PR #51375", answer)
        self.assertIn("Merged PR (candidate fix or cause", answer)
        self.assertIn("PR #50360", answer)
        self.assertIn("Open Issue: Issue #51354", answer)

    @patch("server.api_server.get_driver")
    def test_named_issue_author_is_deterministic(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "title": "LogicError in EvaluateMuonMVAID",
            "author": "cms-bot",
        }

        answer = _issue_author_answer("Who opened this issue #51354?")

        self.assertEqual(answer, "**Issue #51354** was opened by **cms-bot**.")
        self.assertEqual(session.run.call_args.kwargs["number"], 51354)
        get_driver.return_value.close.assert_called_once()

    def test_issue_author_recognizes_auxiliary_verb_phrasing(self):
        # Regression: a real question, "who has created the issue #51756",
        # didn't match the old trigger regex `who\s+(?:opened|created|...)`
        # because of the auxiliary "has" between "who" and "created" -- not
        # a data gap (the issue in question genuinely isn't imported), but a
        # phrasing gap that made even an imported issue's author unreachable
        # this way. _issue_author_answer's own driver-backed "not present"
        # path is covered by test_named_issue_author_is_deterministic; this
        # only pins the trigger regex, so no Neo4j mock is needed here --
        # returning non-None at all proves the regex matched.
        self.assertIsNotNone(_issue_author_answer("who has created the issue #51756?"))

    @patch("server.api_server.get_driver")
    def test_pr_evidence_capability_answers_issue_author_questions(self, get_driver):
        # Regression: same coverage-gap class as
        # test_pr_evidence_capability_combines_pr_and_open_issue_evidence --
        # _issue_author_answer lived only in the legacy waterfall, so once
        # the planner routed "who has created the issue #51756" to the
        # "pr_evidence" capability (matches on the "issue" category), it
        # hard-failed with "No imported local evidence is available for the
        # selected `pr_evidence` request" -- the real bug this reproduces,
        # not the issue's genuine absence from the imported graph.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = None

        result = _execute_planned_capability(
            "pr_evidence", "who has created the issue #51756?", "issue-author-combo"
        )

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertEqual(answer, "Issue #51756 is not present in the imported CMSSW graph.")

    @patch("server.api_server.get_driver")
    def test_issue_detail_answer_reports_title_and_state(self, get_driver):
        # Regression: no function anywhere answered "what is the title of
        # issue #X" or "do you have info about issue #X" -- only who opened
        # it existed (_issue_author_answer). Same "no handler exists at all"
        # gap class as _latest_issue_answer.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "title": "Workflow 2025.0010001 step2 produces different sized files for different runs",
            "state": "open", "created_at": "2026-08-19T00:00:00Z",
        }

        answer = _issue_detail_answer("what is the title of issue #51640?")

        self.assertIn("#51640", answer)
        self.assertIn("Workflow 2025.0010001", answer)

    def test_issue_number_recognizes_github_issues_url(self):
        # A pasted GitHub Issues URL is unambiguous -- no need to fall back
        # to the PR/Issue-ambiguous bare-number heuristic for this shape.
        self.assertEqual(
            _resolve_issue_number("you have no info about https://github.com/cms-sw/cmssw/issues/51640", None),
            51640,
        )

    @patch("server.api_server.get_driver")
    def test_issue_detail_answer_resolves_bare_number_to_issue(self, get_driver):
        # Regression: found live -- "what is the title of 51640" (no "#", no
        # "issue" word) fell all the way through to the generic "I couldn't
        # match that" fallback even though Issue #51640 is imported. The
        # "#"-only extraction regex simply never fired.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "title": "Workflow 2025.0010001 step2 produces different sized files for different runs",
            "state": "open", "created_at": "2026-08-19T00:00:00Z",
        }

        answer = _issue_detail_answer("what is the title of 51640")

        self.assertIn("#51640", answer)
        self.assertIn("Workflow 2025.0010001", answer)

    @patch("server.api_server.get_driver")
    def test_issue_detail_answer_bare_number_falls_back_to_pr(self, get_driver):
        # A bare number is ambiguous between a PR and an Issue -- when no
        # Issue with that number exists, check the PR table before giving up.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.side_effect = [
            None,  # no matching Issue
            {"repo": "cmssw", "title": "Fix workflow 2025.0010001 step2", "merged_at": "2026-08-01T00:00:00Z"},
        ]

        answer = _issue_detail_answer("what is the title of 51397")

        self.assertIn("cmssw PR #51397", answer)
        self.assertIn("Fix workflow 2025.0010001", answer)

    def test_issue_detail_answer_ignores_workflow_id_segments(self):
        # "136.899" is a workflow ID, not an Issue/PR number -- neither "136"
        # nor "899" (its dot-separated segments) may be misread as a bare
        # artifact number, or a routine workflow question would misfire this
        # handler and query for a nonexistent Issue/PR #136 or #899.
        self.assertIsNone(_issue_detail_answer("tell me about workflow 136.899"))

    @patch("server.api_server.get_driver")
    def test_pr_evidence_capability_resolves_bare_followup_to_last_discussed_issue(self, get_driver):
        # Regression: found live. A real conversation named an explicit
        # issue ("who created the issue #51640" -> answered correctly), then
        # asked a bare follow-up with no number and no "issue"/"pr" word at
        # all ("what is the title") -- RequestPlanner can't even classify
        # that into a capability (empty categories), so it fell all the way
        # through to the generic "I couldn't match that" fallback instead of
        # resolving to Issue #51640 via session memory the way IB-tag
        # follow-ups already do via _latest_ib_contexts. _last_issue_contexts
        # now provides the same pronoun resolution for Issues.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {"title": "opened-by-name", "author": "Dr15Jones"}
        session_id = "issue-followup-no-number"
        try:
            first = _execute_planned_capability(
                "pr_evidence", "who created the issue #51640", session_id
            )
            self.assertIsNotNone(first)

            session.run.return_value.single.return_value = {
                "title": "Workflow 2025.0010001 step2 produces different sized files for different runs",
                "state": "open", "created_at": "2026-08-19T00:00:00Z",
            }
            second = _execute_planned_capability("pr_evidence", "what is the title", session_id)
        finally:
            _last_issue_contexts.pop(session_id, None)

        # Note: "what is the title" alone has no "issue"/"pr" word, so in the
        # real chat() flow RequestPlanner can't classify it into any
        # capability at all -- it's the legacy-waterfall wiring (not this
        # capability branch) that actually reaches _issue_detail_answer for
        # that exact phrasing. This test isolates the handler + session
        # memory logic itself, which both paths share.
        self.assertIsNotNone(second)
        answer, _ = second
        self.assertIn("#51640", answer)
        self.assertIn("Workflow 2025.0010001", answer)

    @patch("server.api_server.get_driver")
    def test_ib_relationship_answer_reachable_via_legacy_waterfall(self, get_driver):
        # Sweep finding: _ib_relationship_answer existed only inside
        # _execute_planned_capability, with no fallback in the legacy
        # waterfall unlike every other major answer function. In practice
        # its trigger is perfectly synced with "ib_relationship"'s own
        # planner scoring condition (both use the identical why/parent/
        # sub-ib/variant regex), so RequestPlanner realistically never fails
        # to classify a question this function would answer -- this is pure
        # defensive coverage against that sync drifting apart later (the
        # same kind of drift behind several bugs found earlier this
        # session), tested here by forcing PlanKind.LLM directly rather than
        # via a real under-classified phrasing, since none exists today.
        tag = "CMSSW_20_1_TF_X_2026-08-14-1100"
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "tag": tag, "parent_tag": "CMSSW_20_1_X_2026-08-14-1100", "variant": "TF",
        }
        question = f"why did {tag} show as a sub IB?"

        with patch.object(
            _request_planner, "plan",
            return_value=Plan(PlanKind.LLM, confidence=0.0, context=RequestContext.extract(question)),
        ):
            result = chat(ChatRequest(question=question))

        self.assertIn("sub-IB", result.answer)
        self.assertIn(tag, result.answer)

    @patch("server.api_server.get_driver")
    def test_ib_results_summary_answer_reachable_via_legacy_waterfall(self, get_driver):
        # Same gap as above for _ib_results_summary_answer -- placed last in
        # the legacy chain (see the comment at its call site) since its
        # trigger matches any bare tag mention alone, the same low-priority
        # catch-all role "ib_summary" has in the planner (priority 50).
        # Placing it earlier in an initial attempt at this fix let it shadow
        # _merged_prs_for_build_answer for "how many merged in <tag>" --
        # exactly the "capability matched, but the wrong handler answered"
        # bug class this whole sweep exists to catch, reintroduced by hand
        # this time instead of found live. Forces PlanKind.LLM directly,
        # same as the ib_relationship test above -- defensive coverage in
        # case a future capability-scoring change (or a competing
        # capability outscoring "ib_summary") ever leaves a real
        # summary-shaped question unclassified.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        tag = "CMSSW_20_1_X_2026-08-25-2300"
        session.run.return_value = [
            {"ib_tag": tag, "variant": None, "arch": "el9_amd64_gcc14", "is_default": True,
             "build_status": "passed", "build_warnings": 0, "ignored_warnings": 2,
             "relval_total": 950, "relval_failed": 3, "unit_test_total": 372, "unit_test_failed": 0,
             "unit_test_errors": 0},
        ]
        # "all" (not "summary") avoids the earlier _build_summary_answer's
        # own "summary/analysis" trigger, which would otherwise intercept
        # first with an unrelated "no local log-analysis artifact" answer.
        question = f"{tag} all"

        with patch.object(
            _request_planner, "plan",
            return_value=Plan(PlanKind.LLM, confidence=0.0, context=RequestContext.extract(question)),
        ):
            result = chat(ChatRequest(question=question))

        self.assertIn("primary IB", result.answer)
        self.assertIn(tag, result.answer)

    @patch("server.api_server.get_driver")
    def test_bare_title_followup_resolves_via_chat_legacy_waterfall(self, get_driver):
        # End-to-end reproduction of the exact reported conversation: "who
        # created the issue #51640" (answers correctly) followed by a bare
        # "what is the title" with no number and no "issue"/"pr" word at
        # all. RequestPlanner can't classify that second question into any
        # capability (empty categories -> PlanKind.LLM), so it's the
        # unconditional legacy-waterfall wiring in chat() -- not the
        # "pr_evidence" capability branch -- that must reach
        # _issue_detail_answer via _last_issue_contexts. Before this fix it
        # fell through everything to "I couldn't match that to a specific
        # local answer."
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session_id = "bare-title-followup-e2e"
        try:
            session.run.return_value.single.return_value = {"title": "n/a", "author": "Dr15Jones"}
            first = chat(ChatRequest(question="who created the issue #51640", session_id=session_id))
            self.assertIn("Dr15Jones", first.answer)

            session.run.return_value.single.return_value = {
                "title": "Workflow 2025.0010001 step2 produces different sized files for different runs",
                "state": "open", "created_at": "2026-08-19T00:00:00Z",
            }
            second = chat(ChatRequest(question="what is the title", session_id=session_id))
        finally:
            _last_issue_contexts.pop(session_id, None)

        self.assertIn("#51640", second.answer)
        self.assertIn("Workflow 2025.0010001", second.answer)

    @patch("server.api_server.get_driver")
    def test_chat_handles_github_issues_url_without_crashing(self, get_driver):
        # Regression: found live -- a question containing a GitHub Issues
        # URL ("...cmssw/issues/51640") crashed with a 500 (TypeError:
        # int() argument ... not 'NoneType'). _ISSUE_NUMBER_RE gained a
        # second alternative/capture group for GitHub URLs, but two call
        # sites in _execute_planned_capability and the legacy waterfall
        # still read match.group(1) directly -- None whenever the URL
        # alternative matched instead of the "#N" one. The URL's own
        # "issues" segment satisfies RequestPlanner's "pr_evidence" category
        # on its own, so this question routes deterministically straight
        # into the code path that crashed.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "title": "Workflow 2025.0010001 step2 produces different sized files for different runs",
            "state": "open", "created_at": "2026-08-19T00:00:00Z",
        }

        result = chat(ChatRequest(
            question="you have no info about https://github.com/cms-sw/cmssw/issues/51640"
        ))

        self.assertIn("#51640", result.answer)

    @patch("server.api_server.get_driver")
    def test_latest_issue_answer_reports_newest_imported_issue(self, get_driver):
        # Regression: "what is the latest issue available in the database?"
        # hard-failed -- no function anywhere answered "latest Issue" at all
        # (only a *named* Issue's author, or open-Issue text search existed).
        # Same "no handler exists for this phrasing" gap as the clang fix.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "number": 51644, "title": "HGCAL digis are not reproducible across multithreaded runs",
            "state": "open", "created_at": "2026-08-07T03:23:46Z",
        }

        answer = _latest_issue_answer("what is the latest issue available in the database?")

        self.assertIn("#51644", answer)
        self.assertIn("HGCAL digis", answer)

    @patch("server.api_server.get_driver")
    def test_pr_evidence_capability_answers_latest_issue_questions(self, get_driver):
        # Same coverage-gap class as the other "pr_evidence" capability
        # fixes: confirms the planner's deterministic route actually reaches
        # _latest_issue_answer instead of hard-failing before ever calling it.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "number": 51644, "title": "HGCAL digis are not reproducible across multithreaded runs",
            "state": "open", "created_at": "2026-08-07T03:23:46Z",
        }

        result = _execute_planned_capability(
            "pr_evidence", "what is the latest issue available in the database?", "latest-issue"
        )

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("#51644", answer)

    @patch("server.api_server.find_prs_for_error")
    def test_pr_evidence_capability_answers_batch_workflow_questions(self, find_prs_for_error):
        # Sweep finding: _batch_workflow_pr_answer (exact PR evidence for a
        # pasted list of >=2 backtick-quoted failed workflow names) existed
        # only in the legacy waterfall, unreachable once the planner routed
        # to "pr_evidence" deterministically (it requires the word "pr",
        # which is also exactly what triggers that category).
        find_prs_for_error.side_effect = lambda workflow: (
            [{"repo": "cmssw", "pr_number": 51000, "title": f"Fix {workflow}",
              "evidence": ["pr_title"]}] if workflow == "140.051" else []
        )

        result = _execute_planned_capability(
            "pr_evidence", "any PR fixing `140.051` or `150.2`?", "batch-workflow-pr"
        )

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("140.051", answer)
        self.assertIn("51000", answer)

    @patch("server.api_server.get_driver")
    def test_artifact_search_returns_fuzzy_merged_pr_match(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{
            "kind": "PR",
            "number": 51378,
            "title": "[WarningFix]: Remove mutable global state from makeArrowPlots",
            "state": "closed",
            "merged_at": "2026-08-12T03:54:39+00:00",
            "score": 2.5,
        }]

        answer = _artifact_search_answer("What happened with makeArroPlots?")

        self.assertIn("Work items matching `makeArroPlots`", answer)
        self.assertIn("PR #51378", answer)
        self.assertIn("merged 2026-08-12T03:54:39+00:00", answer)
        session.run.assert_any_call(unittest.mock.ANY, search_query="makeArroPlots~1")
        get_driver.return_value.close.assert_called_once()

    @patch("server.api_server.get_driver")
    def test_artifact_search_keeps_comment_only_matches(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.side_effect = [
            [],
            [{
                "kind": "Issue",
                "number": 51354,
                "title": "LogicError in EvaluateMuonMVAID",
                "state": "open",
                "merged_at": None,
                "score": 0.0,
            }],
        ]

        answer = _artifact_search_answer("Is there an unresolved issue about makeArrowPlots?")

        self.assertIn("Open work items matching `makeArrowPlots`", answer)
        self.assertIn("Issue #51354", answer)
        self.assertEqual(session.run.call_count, 2)
        get_driver.return_value.close.assert_called_once()

    @patch("server.api_server.get_driver")
    def test_named_issue_reporter_is_deterministic(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "title": "LogicError in EvaluateMuonMVAID",
            "author": "cms-bot",
        }

        answer = _issue_author_answer("Who reported Issue #51354?")

        self.assertEqual(answer, "**Issue #51354** was opened by **cms-bot**.")

    @patch("server.api_server.get_driver")
    def test_artifact_search_filters_open_items(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{
            "kind": "Issue",
            "number": 51354,
            "title": "LogicError in EvaluateMuonMVAID",
            "state": "open",
            "merged_at": None,
            "score": 1.2,
        }]

        answer = _artifact_search_answer("Any open issue related to EvaluateMuonMVAID?")

        self.assertIn("Open work items matching `EvaluateMuonMVAID`", answer)
        self.assertIn("Issue #51354", answer)
        get_driver.return_value.close.assert_called_once()

    @patch("server.api_server.get_driver")
    def test_open_artifact_search_shows_related_merged_records(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{
            "kind": "PR",
            "number": 51378,
            "title": "[WarningFix]: Remove mutable global state from makeArrowPlots",
            "state": "closed",
            "merged_at": "2026-08-12T03:54:39+00:00",
            "score": 2.5,
        }]

        answer = _artifact_search_answer("Is anything open for makeArroPlots?")

        self.assertIn("No open Issues or PRs match `makeArroPlots`.", answer)
        self.assertIn("Related non-open records:", answer)
        self.assertIn("PR #51378", answer)
        get_driver.return_value.close.assert_called_once()

    @patch("server.api_server.get_driver")
    @patch("server.api_server._llm_feedback_verification", return_value=("accepted", "Graph facts support this."))
    def test_general_feedback_uses_llm_graph_verifier(self, llm_verification, get_driver):
        result = record_feedback(FeedbackRequest(
            question="Is Issue #51449 open?",
            answer="It is open.",
            verdict="correct",
        ))

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["verification_method"], "llm_graph_tools")
        llm_verification.assert_called_once()
        get_driver.return_value.close.assert_called_once()

    @patch("server.api_server._latest_ib_records")
    def test_latest_ib_answer_is_scoped_to_requested_architecture(self, latest_ib_record):
        latest_ib_record.return_value = [{
            "tag": "CMSSW_20_1_X_2026-08-10-1100",
            "release": "CMSSW_20_1_X",
            "architectures": ["el9_amd64_gcc13"],
            "sub_ib_tags": ["CMSSW_20_1_NONLTO_X_2026-08-10-1100"],
            "build_timestamp": "2026-08-10-1100",
        }]

        answer = _latest_ib_answer("What is the latest IB for el9_amd64_gcc13?")

        latest_ib_record.assert_called_once_with("el9_amd64_gcc13")
        self.assertIn("for `el9_amd64_gcc13`", answer)
        self.assertIn("Sub-IB variants", answer)

    @patch("server.api_server._latest_ib_records")
    def test_latest_ib_answer_lists_timestamp_ties(self, latest_ib_records):
        latest_ib_records.return_value = [
            {
                "tag": "CMSSW_20_1_X_2026-08-10-1100",
                "release": "CMSSW_20_1_X",
                "architectures": ["el8_amd64_gcc13", "el9_amd64_gcc13"],
                "build_timestamp": "2026-08-10-1100",
            },
            {
                "tag": "CMSSW_20_0_X_2026-08-10-1100",
                "release": "CMSSW_20_0_X",
                "architectures": ["el8_amd64_gcc13"],
                "build_timestamp": "2026-08-10-1100",
            },
        ]

        answer = _latest_ib_answer("What is the latest IB?")

        self.assertIn("CMSSW_20_1_X_2026-08-10-1100", answer)
        self.assertIn("CMSSW_20_0_X_2026-08-10-1100", answer)
        self.assertIn("no single tag is selected arbitrarily", answer)

    @patch("server.api_server.get_driver")
    @patch("server.api_server._latest_ib_records")
    def test_feedback_accepts_any_tag_at_newest_timestamp(self, latest_ib_records, get_driver):
        latest_ib_records.return_value = [
            {
                "tag": "CMSSW_20_1_X_2026-08-10-1100",
                "release": "CMSSW_20_1_X",
                "architectures": ["el8_amd64_gcc13", "el9_amd64_gcc13"],
                "build_timestamp": "2026-08-10-1100",
            },
            {
                "tag": "CMSSW_20_0_X_2026-08-10-1100",
                "release": "CMSSW_20_0_X",
                "architectures": ["el8_amd64_gcc13"],
                "build_timestamp": "2026-08-10-1100",
            },
        ]

        result = record_feedback(FeedbackRequest(
            question="What is the latest IB?",
            answer="CMSSW_20_1_X_2026-08-10-1100 is the latest.",
            verdict="correct",
        ))

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(
            result["verified_ib_tags"],
            ["CMSSW_20_1_X_2026-08-10-1100", "CMSSW_20_0_X_2026-08-10-1100"],
        )
        get_driver.return_value.close.assert_called_once()

    @patch("server.api_server._chat_response", side_effect=lambda *args, **kwargs: args[2])
    @patch("server.api_server._all_arch_ib_comparison_answer", return_value="comparison result")
    def test_everything_followup_uses_pending_two_ib_context(self, all_arch_comparison_answer, chat_response):
        session_id = "comparison-test"
        tags = [
            "CMSSW_20_1_X_2026-08-10-1100",
            "CMSSW_20_1_X_2026-08-10-2300",
        ]
        _pending_ib_comparisons[session_id] = {"tags": tags, "clang_warnings": False}
        try:
            result = chat(ChatRequest(question="everything", session_id=session_id))
        finally:
            _pending_ib_comparisons.pop(session_id, None)

        self.assertEqual(result, "comparison result")
        all_arch_comparison_answer.assert_called_once_with(
            f"compare everything for {tags[0]} and {tags[1]} across all architectures"
        )

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._chat_response", side_effect=lambda *args, **kwargs: args[2])
    @patch("server.api_server._execute_planned_capability")
    def test_ambiguous_failure_request_returns_choices_without_llm(
        self, execute, chat_response, restore, store_decision, store_pending, clear_pending
    ):
        session_id = "planner-clarification"
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        try:
            result = chat(ChatRequest(question=f"failures for {tag}", session_id=session_id))
        finally:
            from server.api_server import _request_planner
            _request_planner.pending.pop(session_id, None)

        self.assertIn("Do you want build warnings", result)
        execute.assert_not_called()
        store_pending.assert_called_once()

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._chat_response", side_effect=lambda *args, **kwargs: args[2])
    @patch("server.api_server._execute_planned_capability", return_value=("unit-test evidence", []))
    def test_category_reply_reuses_pending_ib_scope(
        self, execute, chat_response, restore, store_decision, store_pending, clear_pending
    ):
        session_id = "planner-category-reply"
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        try:
            chat(ChatRequest(question=f"failures for {tag}", session_id=session_id))
            result = chat(ChatRequest(question="unit tests", session_id=session_id))
        finally:
            from server.api_server import _request_planner
            _request_planner.pending.pop(session_id, None)

        self.assertEqual(result, "unit-test evidence")
        self.assertEqual(execute.call_args.args[:2], ("unittest", f"unit tests {tag}"))

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._chat_response", side_effect=lambda *args, **kwargs: args[2])
    @patch("server.api_server.propose_investigation_plan", return_value={"capability": "unknown"})
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.search_concepts", return_value=[])
    def test_invalid_llm_plan_is_rejected_before_tool_execution(
        self, concepts, propose_plan, chat_response, restore, store_decision, clear_pending
    ):
        # Regression: an invalid/rejected LLM plan must not be a dead end --
        # it offers concrete next-step actions instead of a flat rejection.
        result = chat(ChatRequest(question="Explain the graph topology", session_id="invalid-plan"))

        self.assertIn("couldn't match that to a specific local answer", result)
        propose_plan.assert_called_once()
        actions = chat_response.call_args.kwargs["actions"]
        self.assertIn({"label": "Latest IB", "question": "what is the latest IB"}, actions)

    # -- Regression tests for bugs found and fixed this session (Phase 0 golden set) --

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._chat_response", side_effect=lambda *args, **kwargs: args[2])
    @patch("server.api_server._execute_planned_capability", return_value=("relval evidence", []))
    @patch("server.api_server._latest_ib_records")
    def test_latest_ib_phrase_resolves_to_a_tag_before_routing(
        self, latest_ib_records, execute, chat_response, restore, store_decision, clear_pending
    ):
        # "the latest IB" with no explicit tag must resolve to a concrete tag
        # before capability dispatch, or relval/build/unittest questions phrased
        # this way silently fall through to "no imported local evidence."
        latest_ib_records.return_value = [{"tag": "CMSSW_20_1_X_2026-08-17-2300"}]

        result = chat(ChatRequest(
            question="how many relvals failing in the latest IB", session_id="latest-ib-resolve",
        ))

        self.assertEqual(result, "relval evidence")
        self.assertEqual(execute.call_args.args[0], "relval")
        self.assertIn("CMSSW_20_1_X_2026-08-17-2300", execute.call_args.args[1])

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._chat_response", side_effect=lambda *args, **kwargs: args[2])
    @patch("server.api_server._execute_planned_capability", return_value=("comparison evidence", []))
    @patch("server.api_server._latest_and_previous_ib_tags")
    def test_compare_latest_branches_resolves_to_a_tag_pair(
        self, latest_and_previous, execute, chat_response, restore, store_decision, clear_pending
    ):
        # "compare latest branches" with zero explicit tags must resolve to the
        # two most recent builds of the newest release series before routing.
        latest_and_previous.return_value = (
            "CMSSW_20_1_X_2026-08-17-2300", "CMSSW_20_1_X_2026-08-17-1100",
        )

        result = chat(ChatRequest(question="compare latest branches", session_id="compare-latest-resolve"))

        self.assertEqual(result, "comparison evidence")
        self.assertEqual(execute.call_args.args[0], "comparison")
        question_used = execute.call_args.args[1]
        self.assertIn("CMSSW_20_1_X_2026-08-17-2300", question_used)
        self.assertIn("CMSSW_20_1_X_2026-08-17-1100", question_used)

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._chat_response", side_effect=lambda *args, **kwargs: args[2])
    @patch("server.api_server._execute_planned_capability", return_value=("comparison evidence", []))
    def test_architecture_ack_with_leading_yes_completes_pending_comparison(
        self, execute, chat_response, restore, store_decision, clear_pending
    ):
        # A natural "yes el9_amd64_gcc14" reply must still be recognized as an
        # architecture-only answer — a bare fullmatch regex previously required
        # the reply to contain nothing but the architecture string.
        session_id = "ack-architecture-reply"
        tags = ["CMSSW_20_1_X_2026-08-17-1100", "CMSSW_20_1_X_2026-08-17-2300"]
        _pending_ib_comparisons[session_id] = {"tags": tags, "clang_warnings": False}
        try:
            result = chat(ChatRequest(question="yes el9_amd64_gcc14", session_id=session_id))
        finally:
            _pending_ib_comparisons.pop(session_id, None)

        self.assertEqual(result, "comparison evidence")
        self.assertEqual(
            execute.call_args.args[1],
            f"compare IB results {tags[0]} and {tags[1]} with el9_amd64_gcc14",
        )

    @patch("server.api_server._relval_ib_summary_answer", return_value="generic summary")
    @patch("server.api_server._relval_exception_answer", return_value="exception evidence")
    @patch("server.api_server._relval_exception_context")
    def test_relval_exception_answer_is_not_shadowed_by_generic_summary(
        self, exception_context, exception_answer, summary_answer
    ):
        # Regression for the capability-dispatch-order bug: a question containing
        # both "relval" and "exception" must reach _relval_exception_answer, not
        # the generic _relval_ib_summary_answer catch-all tried later in the chain.
        exception_context.return_value = {
            "tag": "CMSSW_20_1_X_2026-08-17-2300", "arch": "el9_amd64_gcc14",
        }

        answer, _ = _execute_planned_capability(
            "relval",
            "what relval exception does workflow 140.051 have in CMSSW_20_1_X_2026-08-17-2300?",
            "dispatch-order-session",
        )

        self.assertEqual(answer, "exception evidence")
        summary_answer.assert_not_called()

    def test_relval_exception_context_recognizes_aarch64_architecture(self):
        # Regression for a Phase 1 finding: this function used to build its own
        # amd64-only arch regex, so an aarch64 architecture in the question was
        # silently dropped (context["arch"] came back None, meaning "all
        # architectures") instead of scoping to the one actually named.
        context = _relval_exception_context(
            "relval exception for CMSSW_20_1_X_2026-08-17-2300 el9_aarch64_gcc14"
        )
        self.assertEqual(context["arch"], "el9_aarch64_gcc14")

    def test_failing_relval_context_recognizes_aarch64_architecture(self):
        context = _failing_relval_context(
            "failed relvals for CMSSW_20_1_X_2026-08-17-2300 el9_aarch64_gcc14"
        )
        self.assertEqual(context["arch"], "el9_aarch64_gcc14")

    def test_fallback_clarification_offers_tag_scoped_and_universal_actions(self):
        context = RequestContext.extract("explain the odd behavior for CMSSW_20_1_X_2026-08-17-2300")

        message, actions = _fallback_clarification(context)

        self.assertIn("CMSSW_20_1_X_2026-08-17-2300", message)
        labels = [action["label"] for action in actions]
        self.assertEqual(
            labels,
            ["Build warnings", "RelVal failures", "Unit-test results", "Combined summary",
             "Latest IB", "Frequent RelVal failures (last week)"],
        )
        self.assertIn(
            {"label": "RelVal failures", "question": "failed RelVals for CMSSW_20_1_X_2026-08-17-2300"}, actions,
        )

    def test_fallback_clarification_offers_only_universal_actions_without_a_tag(self):
        context = RequestContext.extract("explain the graph topology")

        message, actions = _fallback_clarification(context)

        self.assertNotIn("`None`", message)
        self.assertEqual(
            actions,
            [
                {"label": "Latest IB", "question": "what is the latest IB"},
                {"label": "Frequent RelVal failures (last week)",
                 "question": "what are the frequent failures over the last week"},
            ],
        )


class LatestTypoToleranceTests(unittest.TestCase):
    """Regression: found live -- "what workflows are failing in the lastest
    IB" (typo: "lastest", not "latest") hard-failed, because
    _is_latest_ib_question's trigger regex didn't recognize it, so the
    "latest IB" tag-resolution special-case never fired and the question
    reached the "relval" capability with no tag at all. The same word list
    had drifted into 5 separate copies (one per function below) before only
    the first one got fixed -- now a single shared _LATEST_WORD_RE."""

    def test_is_latest_ib_question_tolerates_the_typo(self):
        self.assertTrue(_is_latest_ib_question("what workflows are failing in the lastest IB"))

    def test_is_latest_comparison_question_tolerates_the_typo(self):
        self.assertTrue(_is_latest_comparison_question("compare the lastest IB with the previous one"))

    @patch("server.api_server.get_driver")
    def test_latest_clang_warning_answer_tolerates_the_typo(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"tag": "CMSSW_20_1_X_2026-08-25-2300", "release": "CMSSW_20_1_X",
             "arch": "el9_amd64_gcc14", "locations": 3, "occurrences": 7},
        ]

        answer = _latest_clang_warning_answer("what are the lastest clang warnings")

        self.assertIsNotNone(answer)

    @patch("server.api_server.get_driver")
    def test_latest_issue_answer_tolerates_the_typo(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "number": 51644, "title": "x", "state": "open", "created_at": None,
        }

        answer = _latest_issue_answer("what is the lastest issue")

        self.assertIsNotNone(answer)

    def test_clang_warning_answer_still_excludes_the_typo(self):
        # _clang_warning_answer must NOT answer a "lastest clang warnings"
        # question itself -- that's _latest_clang_warning_answer's job; this
        # only pins that the exclusion check also recognizes the typo (an
        # un-excluded "lastest" would let this generic handler answer with a
        # count-table instead of a "latest" one, an inconsistency, not a
        # dead end).
        self.assertIsNone(_clang_warning_answer("what are the lastest clang warnings"))


class ShiftSummaryCapabilityTests(unittest.TestCase):
    def test_trigger_regex_recognizes_shift_summary_phrasing(self):
        for question in [
            "give me a shift summary",
            "shift report please",
            "what changed since the last build",
            "what changed between builds",
            "any news from the previous IB",
            # Regression: "new" only means something relative to a previous
            # build -- "what are the new failures/errors" is the same
            # request as "what changed since the last IB", just phrased
            # without any of the words that regex looked for.
            "what are the new failures",
            "any new errors",
            "what are the new failures/errors",
        ]:
            with self.subTest(question=question):
                self.assertTrue(RequestContext.extract(question).has_shift_summary_request)

    def test_trigger_regex_does_not_fire_on_plain_comparison(self):
        context = RequestContext.extract(
            "compare CMSSW_20_1_X_2026-08-17-1100 and CMSSW_20_1_X_2026-08-17-2300"
        )
        self.assertFalse(context.has_shift_summary_request)

    def test_shift_summary_capability_outranks_relval(self):
        context = RequestContext.extract("what new relval failures showed up in a shift summary")
        plan = _request_planner.plan(context)
        self.assertEqual(plan.capability, "shift_summary")

    def test_shift_summary_question_with_explicit_tag_does_not_trigger_scope_clarification(self):
        # Regression: the generic "choose a scope" clarification fires when an explicit tag
        # + a failure word appear with no recognized category -- "shift summary" is neither
        # "relval"/"clang"/"unittest"/"build", so without an explicit guard this would wrongly
        # ask the user to pick a scope instead of routing to shift_summary.
        context = RequestContext.extract("shift summary for CMSSW_20_1_X_2026-08-17-2300, what failed")
        plan = _request_planner.plan(context)
        self.assertEqual(plan.kind.value, "deterministic")
        self.assertEqual(plan.capability, "shift_summary")

    @patch("server.api_server.generate_shift_summary")
    @patch("server.api_server.get_shift_summary")
    @patch("server.api_server._latest_and_previous_ib_tags")
    def test_uses_cached_summary_without_regenerating(self, latest_and_previous, get_cached, generate):
        latest_and_previous.return_value = ("CMSSW_20_1_X_2026-08-17-2300", "CMSSW_20_1_X_2026-08-17-1100")
        get_cached.return_value = {"markdown": "cached digest text"}

        answer, actions = _execute_planned_capability("shift_summary", "give me a shift summary", "s1")

        self.assertEqual(answer, "cached digest text")
        self.assertEqual(actions, [])
        generate.assert_not_called()

    @patch("server.api_server.generate_shift_summary", return_value="freshly generated digest")
    @patch("server.api_server.get_shift_summary", return_value=None)
    @patch("server.api_server._latest_and_previous_ib_tags")
    def test_falls_back_to_live_generation_when_uncached(self, latest_and_previous, get_cached, generate):
        latest_and_previous.return_value = ("CMSSW_20_1_X_2026-08-17-2300", "CMSSW_20_1_X_2026-08-17-1100")

        answer, _ = _execute_planned_capability("shift_summary", "give me a shift summary", "s1")

        self.assertEqual(answer, "freshly generated digest")
        generate.assert_called_once_with(
            "CMSSW_20_1_X_2026-08-17-2300", "CMSSW_20_1_X_2026-08-17-1100", None,
        )

    @patch("server.api_server.generate_shift_summary", return_value="live arch digest")
    @patch("server.api_server.get_shift_summary")
    @patch("server.api_server._latest_and_previous_ib_tags")
    def test_architecture_scoped_question_bypasses_the_combined_cache(
        self, latest_and_previous, get_cached, generate
    ):
        latest_and_previous.return_value = ("CMSSW_20_1_X_2026-08-17-2300", "CMSSW_20_1_X_2026-08-17-1100")

        answer, _ = _execute_planned_capability(
            "shift_summary", "shift summary for el9_amd64_gcc14", "s1",
        )

        self.assertEqual(answer, "live arch digest")
        get_cached.assert_not_called()
        generate.assert_called_once_with(
            "CMSSW_20_1_X_2026-08-17-2300", "CMSSW_20_1_X_2026-08-17-1100", "el9_amd64_gcc14",
        )

    @patch("server.api_server._latest_and_previous_ib_tags", return_value=None)
    def test_not_enough_history_returns_graceful_message(self, latest_and_previous):
        answer, actions = _execute_planned_capability("shift_summary", "give me a shift summary", "s1")

        self.assertIn("Not enough Integration Build history", answer)
        self.assertEqual(actions, [])


if __name__ == "__main__":
    unittest.main()