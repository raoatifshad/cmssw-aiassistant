import unittest
from unittest.mock import patch

from server.api_server import (
    ChatRequest,
    FeedbackRequest,
    SYSTEM_PROMPT,
    _pending_ib_comparisons,
    _latest_ib_answer,
    _issue_author_answer,
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
    _pr_search_terms,
    _pending_relval_scopes,
    _pending_unittest_scopes,
    _relval_data_availability_answer,
    _relval_error_count_answer,
    _relval_failure_count_answer,
    _relval_ib_summary_answer,
    _unittest_summary_answer,
    _workflow_reported_answer,
    _workflow_artifact_answer,
    _requested_ib_arch,
    _execute_planned_capability,
    _failing_relval_context,
    _fallback_clarification,
    chat,
    record_feedback,
)
from server.request_planner import RequestContext
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
        self.assertIn("Scope: imported DEFAULT architecture markers.", answer)
        self.assertIn("Imported sub-IB variants", answer)
        self.assertIn("`CMSSW_20_1_ROOT640_X_2026-08-11-2300`", answer)
        self.assertIn("ib.parent_tag = $tag", session.run.call_args.args[0])
        get_driver.return_value.close.assert_called_once()

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
    def test_primary_relval_failure_count_requires_architecture(self, scope_records):
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
        try:
            answer = _relval_failure_count_answer(
                "how many relvals failing in CMSSW_20_1_X_2026-08-13-2300", session_id
            )
            self.assertIn("architecture-specific", answer)

            answer = _relval_failure_count_answer("el9_amd64_gcc14", session_id)
            self.assertIn("4 failing RelVal(s)", answer)
            self.assertNotIn(session_id, _pending_relval_scopes)
        finally:
            _pending_relval_scopes.pop(session_id, None)

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

    @patch("server.api_server._relval_scope_records")
    def test_sub_ib_relval_failure_count_requires_architecture(self, scope_records):
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
        try:
            answer = _relval_failure_count_answer(
                f"how many failed relvals in {sub_ib}", session_id
            )
            self.assertIn("architecture-specific", answer)

            answer = _relval_failure_count_answer("el9-aarch64-gcc14", session_id)
            self.assertIn("7 failing RelVal(s)", answer)
            self.assertNotIn(session_id, _pending_relval_scopes)
        finally:
            _pending_relval_scopes.pop(session_id, None)

    @patch("server.api_server._relval_scope_records")
    def test_relval_failure_count_rejects_invalid_architecture_choice(self, scope_records):
        session_id = "relval-invalid-scope"
        primary_ib = "CMSSW_20_1_X_2026-08-13-2300"
        scope_records.return_value = [
            {
                "tag": primary_ib,
                "arch": "el9_amd64_gcc14",
                "relval_total": 3700,
                "failed_relvals": 4,
                "passed_relvals": 3696,
            },
            {
                "tag": primary_ib,
                "arch": "el10_amd64_gcc15",
                "relval_total": 3700,
                "failed_relvals": 1,
                "passed_relvals": 3699,
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
    def test_unqualified_relval_question_uses_imported_default_architecture(self, scope_records):
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        scope_records.return_value = [
            {"tag": tag, "arch": "el8_amd64_gcc13", "relval_total": 950, "failed_relvals": 3, "passed_relvals": 947, "is_relval_default": False},
            {"tag": tag, "arch": "el9_amd64_gcc13", "relval_total": 950, "failed_relvals": 3, "passed_relvals": 947, "is_relval_default": False},
            {"tag": tag, "arch": "el9_amd64_gcc14", "relval_total": 950, "failed_relvals": 3, "passed_relvals": 947, "is_relval_default": True},
        ]

        answer = _relval_failure_count_answer(f"any relvals failing in {tag}", "relval-default-arch")

        self.assertIn("el9_amd64_gcc14", answer)
        self.assertNotIn("el8_amd64_gcc13", answer)
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
        session_id = "relval-repeated-question"
        primary_ib = "CMSSW_20_1_X_2026-08-10-1100"
        scope_records.return_value = [
            {
                "tag": primary_ib,
                "arch": "el9_aarch64_gcc13",
                "relval_total": 950,
                "failed_relvals": 53,
                "passed_relvals": 897,
            },
            {
                "tag": primary_ib,
                "arch": "el9_amd64_gcc13",
                "relval_total": 950,
                "failed_relvals": 3,
                "passed_relvals": 947,
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


if __name__ == "__main__":
    unittest.main()