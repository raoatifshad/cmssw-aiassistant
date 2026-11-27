import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from server.api_server import (
    ChatRequest,
    FeedbackRequest,
    SYSTEM_PROMPT,
    _log_router_shadow_decision,
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
    _category_failure_snapshot,
    _ib_relationship_answer,
    _ib_results_summary_answer,
    _pasted_exception_signature,
    _pasted_exception_search_answer,
    _pasted_exception_workflow_matches,
    _exception_category_from_signature,
    _relval_exception_context,
    _relval_exception_list_answer,
    _relval_exception_answer,
    _normalize_deterministic_question,
    _pr_evidence_answer,
    _pr_search_terms,
    _looks_like_noise_artifact,
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
    _verification_session_context,
    _llm_answer_is_absence_claim,
    _question_has_scope_qualifier,
    _conversational_response,
    _relval_known_error_lines,
    _workflow_why_failing_answer,
    _failure_overview_answer,
    _parse_recency_window_days,
    _latest_and_days_ago_ib_tags,
    _shift_summary_answer,
    _ib_comparison_answer,
    _all_arch_ib_comparison_answer,
    _render_metric_deltas,
    _request_planner,
    _evidence_grounded,
    _shifter_doc_claims_grounded,
    _format_raw_evidence,
    chat,
    record_feedback,
)
from server.request_planner import Plan, PlanKind, RequestContext
from server.agent import KGChatSession, propose_pr_search_signature, verify_answer_matches_question
from graph.tools import TOOLS, get_failing_relvals, get_failing_unittests, search_issues_by_text
from ingestion.build_graph import ib_parent_tag


def _mock_kgchat_session(final_answer, tool_results=()):
    """Build a replacement constructor for server.api_server.KGChatSession
    (patched via @patch(..., side_effect=...)) that fires the real on_tool_call
    hook _run_llm_investigation passes in for each (name, args, result) in
    tool_results -- exactly how the real agentic loop reports evidence as it
    calls tools -- then returns final_answer from the mock session's .ask().
    factory.sessions holds every constructed mock session, in order, since
    .side_effect (not .return_value) is what actually runs per call."""
    sessions: list[MagicMock] = []

    def factory(*, openai_client, model, tools_registry, system_prompt, on_tool_call=None):
        if on_tool_call:
            for name, args, result in tool_results:
                on_tool_call(name, args, json.dumps(result, default=str))
        session = MagicMock()
        session.ask.return_value = final_answer
        sessions.append(session)
        return session

    factory.sessions = sessions
    return factory


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

    @patch("graph.tools.get_driver")
    def test_failing_relvals_for_primary_ib_include_sub_ibs(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []

        get_failing_relvals("CMSSW_20_1_X_2026-08-12-1100", arch="el9_amd64_gcc13")

        cypher, params = session.run.call_args.args
        self.assertIn("ib.parent_tag = $t", cypher)
        self.assertIn("ib.tag AS ib_tag", cypher)
        self.assertEqual(params["t"], "CMSSW_20_1_X_2026-08-12-1100")

    @patch("graph.tools.get_driver")
    def test_failing_relvals_limit_applies_per_ib_node_not_globally(self, get_driver):
        # Regression: a single heavily-failing sub-IB (found live, 2026-09-09: UBSAN
        # with 948/955 RelVals failing) used to consume the entire flat LIMIT before
        # the query ever reached OTHER variants -- including the primary itself --
        # silently returning zero rows for them even though they had real failures.
        # The fix caps rows per matched IB node via a correlated subquery, and reads
        # via get_driver directly (bypassing _run_query's own MAX_ROWS post-slice,
        # which would otherwise re-introduce the same crowd-out one layer down).
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []

        get_failing_relvals("CMSSW_20_1_X_2026-09-07-2300", arch="el9_amd64_gcc14", limit=100)

        cypher, params = session.run.call_args.args
        self.assertIn("CALL (ib) {", cypher)
        self.assertIn("LIMIT $lim", cypher)
        self.assertEqual(params["lim"], 100)

    @patch("graph.tools.get_driver")
    def test_failing_unittests_default_stays_exact_tag_only(self, get_driver):
        # include_sub_ibs defaults to False: _failed_unittest_package_listing already
        # loops over each exact sub-IB tag itself and would double-count sub-IB
        # failures under both their own section and the primary's if this pulled in
        # sub-IBs by default.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []

        get_failing_unittests("CMSSW_20_1_X_2026-09-07-2300", arch="el9_amd64_gcc14")

        cypher, params = session.run.call_args.args
        self.assertIn("ib:IB {tag:$t}", cypher)
        self.assertNotIn("parent_tag", cypher)

    @patch("graph.tools.get_driver")
    def test_failing_unittests_include_sub_ibs_caps_limit_per_ib_node(self, get_driver):
        # Same crowd-out fix as get_failing_relvals: sub-IB unit-test failures were
        # entirely invisible to generate_shift_summary before include_sub_ibs
        # existed at all (get_failing_unittests matched ib.tag = $t only, never
        # parent_tag) -- found live, 2026-09-09: UBSAN alone had 35 failing unit
        # tests never surfaced in any shift digest. The per-node LIMIT here exists
        # for the same reason it does on get_failing_relvals.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []

        get_failing_unittests("CMSSW_20_1_X_2026-09-07-2300", arch="el9_amd64_gcc14", limit=100, include_sub_ibs=True)

        cypher, params = session.run.call_args.args
        self.assertIn("ib.parent_tag = $t", cypher)
        self.assertIn("CALL (ib) {", cypher)
        self.assertIn("LIMIT $lim", cypher)
        self.assertEqual(params["lim"], 100)

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

    def test_search_terms_reject_ordinary_capitalized_words(self):
        # Regression: found live -- pasting a raw error message ("MIOpen Error:
        # filesystem error: cannot create directories: Permission denied
        # [/tmp/miopen-cache/3.5.2.]") extracted "Permission" as a search term,
        # since it has an uppercase letter and isn't a stop word. Against
        # find_prs_for_error's PR-comment search (~690k comments), a single
        # ordinary English word that merely starts a clause returned 20
        # completely unrelated PRs. "Permission" has its only uppercase letter
        # at position 0 (not a real CamelCase/acronym identifier); "MIOpen"
        # (internal capital) and the package-path-shaped "tmp/miopen" must
        # still survive.
        question = (
            "MIOpen Error: filesystem error: cannot create directories: "
            "Permission denied [/tmp/miopen-cache/3.5.2.]"
        )

        self.assertEqual(_pr_search_terms(question), ["tmp/miopen", "MIOpen"])

    def test_search_terms_keep_all_caps_acronyms_and_internal_caps(self):
        # Same fix as above, checked from the other direction: an all-caps
        # acronym/exit-code name (SIGSEGV) and a genuine CamelCase compound
        # identifier must still pass, since both carry uppercase beyond
        # position 0.
        self.assertEqual(_pr_search_terms("any PR for SIGSEGV crash"), ["SIGSEGV"])
        self.assertEqual(
            _pr_search_terms("is there any PR/issue open for testAlpakaModelCudaAsync"),
            ["testAlpakaModelCudaAsync"],
        )

    def test_noise_artifact_rejects_hex_hash_and_punctuation_run(self):
        # Regression: found live 2026-09-21 -- a pasted CUDA/Alpaka exception block
        # extracted a package-version hex hash and a decorative divider-line
        # fragment as if they were real search identifiers, and the hash (longer)
        # won the search over the workflow's actual error code, since
        # _pr_evidence_answer tries the longest term first.
        self.assertTrue(_looks_like_noise_artifact("caaac8d71f39d400ab2511b2403675a"))
        self.assertTrue(_looks_like_noise_artifact("CEST-----------------------"))
        # Real identifiers of similar shape/length must not be caught by the
        # same rules.
        self.assertFalse(_looks_like_noise_artifact("cudaErrorLaunchOutOfResources"))
        self.assertFalse(_looks_like_noise_artifact("CMSSW_20_1_X_2026-08-04-2300"))

    def test_search_terms_drop_hex_hash_and_punctuation_run_from_pasted_exception(self):
        # End-to-end version of the above: the noise tokens must never reach the
        # candidate set _pr_evidence_answer searches with, while the genuine error
        # code from the same pasted block survives.
        question = (
            "any PR or issue for this: ----- Begin Fatal Exception 19-Sep-2026 "
            "15:11:10 CEST-----------------------\n"
            "/data/cmsbld/.../external/alpaka/2.1.1-3caaac8d71f39d400ab2511b2403675a/"
            "include/alpaka/kernel/TaskKernelGpuUniformCudaHipRt.hpp(275) "
            "'cudaErrorLaunchOutOfResources': 'too many resources requested for launch'!"
        )

        terms = _pr_search_terms(question)

        self.assertIn("cudaErrorLaunchOutOfResources", terms)
        self.assertFalse(any("caaac8d71f39d400ab2511b2403675a" in term for term in terms))
        self.assertFalse(any(term.count("-") >= 3 for term in terms))

    def test_propose_pr_search_signature_strips_surrounding_quotes(self):
        # Regression: found live 2026-09-21 -- the same underlying phrase matched
        # find_prs_for_error with or without its decorative quotes, but a hosted
        # LLM's exact formatting isn't reliable enough call-to-call to depend on
        # the prompt instruction alone; defensive Python-side stripping is required.
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"error_signature": "\'cudaErrorLaunchOutOfResources\'"}'
            ))]
        )

        signature = propose_pr_search_signature(client, "some-model", "any PR for this error?")

        self.assertEqual(signature, "cudaErrorLaunchOutOfResources")

    def test_propose_pr_search_signature_returns_none_for_null_signature(self):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"error_signature": null}'))]
        )

        self.assertIsNone(propose_pr_search_signature(client, "some-model", "how many relvals failing?"))

    def test_verify_answer_matches_question_returns_true_when_llm_confirms(self):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"matches": true}'))]
        )

        result = verify_answer_matches_question(
            client, "some-model",
            "how many relvals are failing in CMSSW_20_1_X_2026-09-21-1100",
            "0 failing RelVal(s) across 592 RelVal record(s).",
        )

        self.assertTrue(result)

    def test_verify_answer_matches_question_returns_false_on_negation_mismatch(self):
        # Real bug found live 2026-09-22: "workflows that are NOT failing" and
        # "workflows that are failing" both trigger the same "failing" keyword
        # match in the deterministic relval handler and get the identical
        # answer -- this is the verifier catching exactly that class of miss.
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"matches": false, "reason": "question asks for workflows that are NOT '
                        'failing (i.e. passing), answer only addresses failing RelVals"}'
            ))]
        )

        result = verify_answer_matches_question(
            client, "some-model",
            "give me the workflows that are NOT failing in relvals CMSSW_20_1_X_2026-09-21-1100",
            "No failing RelVals were found for CMSSW_20_1_X_2026-09-21-1100.",
        )

        self.assertFalse(result)

    def test_verify_answer_matches_question_fails_open_on_malformed_json(self):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not json at all"))]
        )

        result = verify_answer_matches_question(client, "some-model", "some question", "some answer")

        self.assertTrue(result)

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
        find_prs.assert_called_once_with("140.0", limit=20)

    @patch("server.api_server.find_prs_for_error", return_value=[])
    def test_pr_evidence_prefers_an_explicit_identifier_over_the_fallback(self, find_prs):
        # An identifier actually present in the question must never be
        # overridden by stale remembered context from an earlier turn.
        _pr_evidence_answer("any PR fixing CMSSW_20_1_X_2026-08-04-2300?", fallback_term="140.0")

        find_prs.assert_called_once_with("CMSSW_20_1_X_2026-08-04-2300", limit=20)

    def test_pr_evidence_without_a_fallback_or_identifier_returns_none(self):
        self.assertIsNone(_pr_evidence_answer("is there any issue /pr open for it recently?"))

    @patch("server.api_server.find_prs_for_error")
    @patch("server.api_server.propose_pr_search_signature")
    @patch("server.api_server._llm_client", new="not-none-sentinel")
    def test_pr_evidence_falls_back_to_llm_signature_when_regex_terms_miss(
        self, propose_signature, find_prs
    ):
        # Pilot for the class of bug in project_pr_search_precision_fix: found live
        # 2026-09-21, a pasted MIOpen error ("...Permission denied...") got reduced to
        # searching the isolated word "Permission" -- the regex approach can only ever
        # score words in isolation, never read the message as a whole. When every
        # regex-extracted term fails to find a match, _pr_evidence_answer now falls
        # back once to propose_pr_search_signature (one constrained LLM call reading
        # the whole question) and searches with whatever it proposes.
        question = (
            "MIOpen Error: filesystem error: cannot create directories: "
            "Permission denied [/tmp/miopen-cache/3.5.2.]"
        )
        propose_signature.return_value = (
            "MIOpen Error: filesystem error: cannot create directories: "
            "Permission denied [/tmp/miopen-cache/3.5.2.]"
        )
        # First two calls are the regex-extracted terms ("tmp/miopen", "MIOpen"),
        # both miss; the third call is the LLM-derived full-message signature, hit.
        find_prs.side_effect = [
            [],
            [],
            [{
                "repo": "cmssw", "pr_number": 51900, "title": "Fix MIOpen cache permissions",
                "merged_at": None, "evidence": ["pr_comment"], "matched_files": [],
                "matched_unit_tests": False, "matched_unit_test_packages": [],
                "matched_unit_test_ibs": [], "matched_warnings": None,
            }],
        ]

        result = _pr_evidence_answer(f"any pr for this: {question}")

        self.assertIsNotNone(result)
        answer, rows = result
        self.assertIn("51900", answer)
        self.assertIn("LLM-assisted search", answer)
        self.assertEqual(len(rows), 1)
        propose_signature.assert_called_once()

    @patch("server.api_server.find_prs_for_error", return_value=[])
    @patch("server.api_server.propose_pr_search_signature", return_value=None)
    @patch("server.api_server._llm_client", new="not-none-sentinel")
    def test_pr_evidence_reports_no_evidence_when_llm_fallback_also_misses(
        self, propose_signature, find_prs
    ):
        # The LLM fallback is a best-effort second attempt, not a guarantee -- when
        # neither the regex terms nor the LLM-proposed signature find anything, this
        # must still report a plain no-evidence miss, never fabricate a match.
        result = _pr_evidence_answer("any pr for CMSSW_20_1_X_2026-08-04-2300?")

        self.assertIsNotNone(result)
        answer, rows = result
        self.assertIn("No PR evidence matching", answer)
        self.assertEqual(rows, [])

    @patch("server.api_server.find_prs_for_error", return_value=[])
    @patch("server.api_server.propose_pr_search_signature", side_effect=TimeoutError("gateway too slow"))
    @patch("server.api_server._llm_client", new="not-none-sentinel")
    def test_pr_evidence_degrades_gracefully_when_llm_fallback_times_out(
        self, propose_signature, find_prs
    ):
        # Same fail-safe contract as a null/unparseable LLM reply -- a
        # slow/unreachable gateway here must fall back to the plain
        # no-evidence miss, never propagate as an unhandled error.
        result = _pr_evidence_answer("any pr for CMSSW_20_1_X_2026-08-04-2300?")

        self.assertIsNotNone(result)
        answer, rows = result
        self.assertIn("No PR evidence matching", answer)
        self.assertEqual(rows, [])

    @patch("server.api_server.find_prs_for_error", return_value=[])
    @patch("server.api_server.propose_pr_search_signature")
    def test_pr_evidence_skips_llm_fallback_when_client_unconfigured(self, propose_signature, find_prs):
        # _llm_client defaults to None until the app's startup event configures it
        # (never fires in this test process) -- the fallback must no-op cleanly
        # rather than erroring, and never call the LLM proposal function at all.
        result = _pr_evidence_answer("any PR fixing CMSSW_20_1_X_2026-08-04-2300?")

        self.assertIsNotNone(result)
        propose_signature.assert_not_called()

    @patch("server.api_server.find_prs_for_error")
    @patch("server.api_server.propose_pr_search_signature")
    @patch("server.api_server._llm_client", new="not-none-sentinel")
    def test_pr_evidence_prefers_narrower_llm_match_for_qualified_code_reference(
        self, propose_signature, find_prs
    ):
        # Found live 2026-09-21: "any PR / issue having edm::FileInPath::initialize_()"
        # extracted only the bare class name "FileInPath" (word-splitting can't
        # preserve "::"/"()" , and the real specific part -- "initialize_", lowercase
        # per C++ private-method convention -- fails the capitalization check on its
        # own). The regex search still found 20 (the row limit) loosely-related PRs,
        # so the OLD zero-results-only trigger never even tried the LLM. Presence of
        # "::"/"()" must trigger the LLM check even though the regex path already
        # "succeeded", and a narrower LLM-derived match must be preferred over the
        # broad one.
        question = "any PR / issue having edm::FileInPath::initialize_()"
        propose_signature.return_value = "edm::FileInPath::initialize_()"
        broad_rows = [
            {"repo": "cmssw", "pr_number": 40000 + n, "title": f"Unrelated FileInPath change {n}",
             "merged_at": None, "evidence": ["pr_title"], "matched_files": [],
             "matched_unit_tests": False, "matched_unit_test_packages": [],
             "matched_unit_test_ibs": [], "matched_warnings": None}
            for n in range(20)
        ]
        narrow_rows = [{
            "repo": "cmssw", "pr_number": 46200, "title": "edm::FileInPath: return reference to paths; cleanup code",
            "merged_at": None, "evidence": ["pr_title"], "matched_files": [],
            "matched_unit_tests": False, "matched_unit_test_packages": [],
            "matched_unit_test_ibs": [], "matched_warnings": None,
        }]
        find_prs.side_effect = [broad_rows, narrow_rows]

        result = _pr_evidence_answer(question)

        self.assertIsNotNone(result)
        answer, rows = result
        self.assertIn("46200", answer)
        self.assertIn("LLM-assisted search", answer)
        self.assertEqual(len(rows), 1)
        propose_signature.assert_called_once()

    @patch("server.api_server.find_prs_for_error")
    @patch("server.api_server.propose_pr_search_signature")
    @patch("server.api_server._llm_client", new="not-none-sentinel")
    def test_pr_evidence_caps_broad_fallback_when_llm_finds_nothing_narrower(
        self, propose_signature, find_prs
    ):
        # Precise-first, capped fallback (per user-approved design 2026-09-21): the
        # LLM double-check must never make a broad-but-real regex match disappear
        # entirely -- but dumping the full _PR_SEARCH_ROW_LIMIT rows verbatim, even
        # with a trailing caveat, was itself the noise complaint that prompted this.
        # When a more specific search was warranted and comes up empty, show only a
        # small, clearly-labeled "closest related, not an exact match" preview.
        question = "any PR / issue having edm::FileInPath::initialize_()"
        propose_signature.return_value = "edm::FileInPath::initialize_()"
        broad_rows = [
            {"repo": "cmssw", "pr_number": 40000 + n, "title": f"Unrelated FileInPath change {n}",
             "merged_at": None, "evidence": ["pr_title"], "matched_files": [],
             "matched_unit_tests": False, "matched_unit_test_packages": [],
             "matched_unit_test_ibs": [], "matched_warnings": None}
            for n in range(20)
        ]
        find_prs.side_effect = [broad_rows, []]

        result = _pr_evidence_answer(question)

        self.assertIsNotNone(result)
        answer, rows = result
        self.assertNotIn("LLM-assisted search", answer)
        self.assertEqual(len(rows), 3)
        self.assertIn("closest related", answer.lower())
        self.assertIn("not an exact match", answer)
        self.assertIn("edm::FileInPath::initialize_()", answer)

    @patch("server.api_server.find_prs_for_error")
    @patch("server.api_server.propose_pr_search_signature")
    @patch("server.api_server._llm_client", new="not-none-sentinel")
    def test_pr_evidence_does_not_relabel_a_small_match_set(self, propose_signature, find_prs):
        # Regression: found live 2026-09-21 -- "any information about
        # ticl::AssociationMap any PR/issue open?" has qualified-code syntax
        # (triggering the LLM check) but its bare-term match was already a small,
        # reasonably-focused set (well under the row limit) of a genuinely relevant
        # real PR (see docs/DEVELOPER.md's "Capability-handler coverage" section -- this exact
        # question is the documented regression case for combining PR+Issue evidence
        # without short-circuiting). Relabeling a small set as "not an exact match"
        # just because the LLM's stricter qualified-string check didn't also confirm
        # it is wrong -- it implies hidden noise that doesn't exist. A small match set
        # must be shown normally, not capped/relabeled.
        question = "any information about ticl::AssociationMap any PR/issue open?"
        propose_signature.return_value = "ticl::AssociationMap"
        small_rows = [{
            "repo": "cmssw", "pr_number": 51649, "title": "Rename SimDataformats TICL associator as TICLAssociationMap",
            "merged_at": "2026-08-21T09:33:48Z", "evidence": ["pr_title"], "matched_files": [],
            "matched_unit_tests": False, "matched_unit_test_packages": [],
            "matched_unit_test_ibs": [], "matched_warnings": None,
        }]
        find_prs.side_effect = [small_rows, []]

        result = _pr_evidence_answer(question)

        self.assertIsNotNone(result)
        answer, rows = result
        self.assertIn("PR evidence matching `AssociationMap`:", answer)
        self.assertIn("51649", answer)
        self.assertNotIn("closest related", answer.lower())
        self.assertEqual(len(rows), 1)

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
            [t["term"] for t in session.run.call_args.kwargs["terms"]],
            ["HydjetQMinBias_5519GeV+Run4D121_hi", "34551.85"],
        )

    @patch("server.api_server.get_driver")
    def test_open_issue_search_does_not_require_the_word_open(self, get_driver):
        # Regression: found live 2026-09-21 -- "...where this comment lies in the
        # issue/pr" mentions "issue" but never "open"/"unresolved"/"active", so
        # _open_issue_answer returned None immediately and Issues were never even
        # searched, even though the sibling PR search ran fine for the same
        # question. The Cypher query itself already scopes to state='open' Issues
        # unconditionally, so requiring that word in the phrasing was a redundant,
        # over-narrow entry gate, not real filtering.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{
            "number": 51900, "title": "MIOpen cache permission failure",
            "state": "open", "matched_terms": ["MIOpen"],
        }]

        answer = _open_issue_answer(
            "MIOpen Error: filesystem error: cannot create directories: "
            "Permission denied [/tmp/miopen-cache/3.5.2.] where this comment "
            "lies in the issue/pr",
            None,
        )

        self.assertIsNotNone(answer)
        self.assertIn("Issue #51900", answer)

    @patch("server.api_server.get_driver")
    @patch("server.api_server.propose_pr_search_signature")
    @patch("server.api_server._llm_client", new="not-none-sentinel")
    def test_open_issue_search_adds_llm_signature_for_qualified_code_reference(
        self, propose_signature, get_driver
    ):
        # Found live 2026-09-21: "any PR / issue having edm::FileInPath::initialize_()"
        # matched Issue #51757 only via the bare term "FileInPath" -- but that Issue's
        # body literally contains the exact qualified string
        # "edm::FileInPath::initialize_()" verbatim, a precise hit the bare-word
        # search couldn't distinguish from four other issues that only coincidentally
        # mention "FileInPath". When the question has qualified-code syntax, the
        # LLM-derived signature is now added as an extra search term so an exact
        # match is visible in the row's own matched_terms, not indistinguishable from
        # a generic one.
        propose_signature.return_value = "edm::FileInPath::initialize_()"
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{
            "number": 51757, "title": "Infinite loop in edm::FileInPath", "state": "open",
            "matched_terms": ["FileInPath", "edm::FileInPath::initialize_()"],
        }]

        answer = _open_issue_answer("any PR / issue having edm::FileInPath::initialize_()", None)

        self.assertIsNotNone(answer)
        self.assertIn("Issue #51757", answer)
        self.assertIn("edm::FileInPath::initialize_()", answer)
        self.assertEqual([t["term"] for t in session.run.call_args.kwargs["terms"]],
                         ["FileInPath", "edm::FileInPath::initialize_()"])

    @patch("server.api_server.get_driver")
    @patch("server.api_server.propose_pr_search_signature")
    @patch("server.api_server._llm_client", new="not-none-sentinel")
    def test_open_issue_search_caps_broad_fallback_when_no_row_matches_precisely(
        self, propose_signature, get_driver
    ):
        # Precise-first, capped fallback (per user-approved design 2026-09-21), Issue
        # side: when NO row's matched_terms includes the LLM-derived signature (only
        # the bare, generic term matched anything) AND the match set is itself broad
        # (hit the row limit -- a small match set is shown as-is, see
        # test_open_issue_search_does_not_relabel_a_small_match_set below), show a
        # small, clearly-labeled "closest related, not an exact match" preview instead
        # of the full set.
        propose_signature.return_value = "edm::FileInPath::initialize_()"
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"number": 40000 + n, "title": f"Unrelated FileInPath issue {n}", "state": "open",
             "matched_terms": ["FileInPath"]}
            for n in range(20)
        ]

        answer = _open_issue_answer("any PR / issue having edm::FileInPath::initialize_()", None)

        self.assertIsNotNone(answer)
        self.assertIn("closest related", answer.lower())
        self.assertIn("not an exact match", answer)
        self.assertIn("edm::FileInPath::initialize_()", answer)
        self.assertEqual(answer.count("**Issue #"), 3)

    @patch("server.api_server.get_driver")
    @patch("server.api_server.propose_pr_search_signature")
    @patch("server.api_server._llm_client", new="not-none-sentinel")
    def test_open_issue_search_does_not_relabel_a_small_match_set(self, propose_signature, get_driver):
        # Regression: found live 2026-09-21 -- "any information about
        # ticl::AssociationMap any PR/issue open?" has qualified-code syntax
        # (triggering the LLM check) but its bare-term match was already a small,
        # reasonably-focused set (well under the row limit) that genuinely relevant
        # real Issues were part of (#42737 et al., confirmed live in
        # project_cross_variant... -- see docs/DEVELOPER.md's capability-handler-coverage
        # history). Relabeling a small set as "not an exact match" just because the
        # LLM's stricter qualified-string check didn't also confirm it (people often
        # drop a C++ namespace prefix in casual discussion) is wrong -- it implies
        # hidden noise that doesn't exist. A small match set must be shown as-is.
        propose_signature.return_value = "ticl::AssociationMap"
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"number": 42737, "title": "edm::RefToBase is deprecated", "state": "open",
             "matched_terms": ["AssociationMap"]},
        ]

        answer = _open_issue_answer("any information about ticl::AssociationMap any PR/issue open?", None)

        self.assertIsNotNone(answer)
        self.assertIn("Open CMSSW Issues matching the supplied identifiers:", answer)
        self.assertIn("Issue #42737", answer)
        self.assertNotIn("closest related", answer.lower())

    def test_normalizes_common_deterministic_paraphrases(self):
        self.assertEqual(
            _normalize_deterministic_question("How much Rel-Val failures are in this integration-build?"),
            "how many relval failures are in this ib?",
        )

    def test_does_not_mangle_relvals_inside_an_unrelated_hyphenated_identifier(self):
        # Regression: \b alone treats a hyphen as a word boundary, so the
        # "relvals" -> "relval" normalization used to also fire inside the
        # real Jenkins job name "ib-run-gpu-relvals", corrupting it to
        # "ib-run-gpu-relval" before it ever reached routing or the LLM --
        # traced back from a live shifter-doc answer that misattributed one
        # job's details to another.
        self.assertEqual(
            _normalize_deterministic_question("what does ib-run-gpu-relvals depend on"),
            "what does ib-run-gpu-relvals depend on",
        )

    def test_requested_ib_architecture_is_extracted(self):
        self.assertEqual(
            _requested_ib_arch("What is the latest IB for el9_amd64_gcc13?"),
            "el9_amd64_gcc13",
        )

    @patch("server.api_server.get_driver")
    def test_relval_exception_list_answer_never_traverses_has_exception(self, get_driver):
        # Real bug found live 2026-10-02: GRAPH_SCHEMA documented
        # (RelVal)-[:HAS_EXCEPTION]->(RelValException), and this function's
        # query traversed exactly that -- but ingestion never actually
        # creates that relationship (0 HAS_EXCEPTION edges exist even with
        # 9,612+ real RelValException nodes imported, confirmed directly
        # against the graph), so this silently returned "no exception
        # signatures were found" for every real exception ever imported.
        # Mocked session.run(return_value=[]) in the older tests below never
        # caught this, since it doesn't care what Cypher text was sent --
        # this test pins the query shape itself, not just its mocked result.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{
            "arch": "el9_amd64_gcc14", "workflow_id": "136.899",
            "exception": "LogicError: boom", "category": "LogicError", "source_url": None,
        }]
        question = "what are the relval exceptions in CMSSW_20_1_X_2026-08-12-1100?"

        answer = _relval_exception_list_answer(question, _relval_exception_context(question))

        query = session.run.call_args.args[0]
        self.assertNotIn("HAS_EXCEPTION", query)
        self.assertIn("exception.ib_tag", query)
        self.assertIn("136.899", answer)
        self.assertIn("LogicError", answer)

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
        get_failing_relvals.assert_called_once_with(tag, arch=None, limit=None)

    @patch("server.api_server.get_relval_recurrence")
    def test_relval_capability_answers_where_is_workflow_failing_with_no_tag(self, get_relval_recurrence):
        # Regression: found live 2026-09-18 -- "537.0 where this workflow failing" (no
        # IB tag at all, first turn of a fresh session) fell all the way through the
        # "relval" capability's chain to the flat "No imported local evidence" message.
        # _recurring_failure_answer is exactly the right handler for a bare workflow ID
        # with no tag -- it was already in the chain -- but its trigger regex,
        # _RECURRING_FAILURE_INTENT_RE, recognized "when ... fail" and never "where ...
        # fail", so it never matched this phrasing and the question never reached it.
        get_relval_recurrence.return_value = {
            "workflow_id": "537.0", "name": "DYToLL012Jets_5FS_TuneCH3_13TeV_amcatnloFxFx_herwig7",
            "total_appearances": 237, "failed_appearances": 36,
            "failed_records": [
                {"tag": "CMSSW_20_1_X_2026-09-17-2300", "arch": "el9_amd64_gcc16", "variant": "primary"},
                {"tag": "CMSSW_20_1_UBSAN_X_2026-09-14-2300", "arch": "el9_amd64_gcc14", "variant": "UBSAN"},
            ],
            "failed_tags": ["CMSSW_20_1_X_2026-09-17-2300", "CMSSW_20_1_UBSAN_X_2026-09-14-2300"],
        }
        session_id = "relval-where-is-it-failing"
        result = _execute_planned_capability("relval", "537.0 where this workflow failing", session_id)

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("failed in **36 of 237**", answer)
        # Each entry names its own variant and architecture -- not just a bare,
        # ambiguous tag string (see get_relval_recurrence's crowd-out fix).
        self.assertIn("`CMSSW_20_1_X_2026-09-17-2300` (primary, `el9_amd64_gcc16`)", answer)
        self.assertIn("`CMSSW_20_1_UBSAN_X_2026-09-14-2300` (UBSAN, `el9_amd64_gcc14`)", answer)
        get_relval_recurrence.assert_called_once_with("537.0")

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._latest_and_previous_ib_tags")
    def test_relval_capability_resolves_latest_release_phrasing_with_no_tag(
        self, latest_tags, get_failing_relvals, get_exit_code_names
    ):
        # Regression: found live 2026-09-18 -- "what relvals are failing in latest
        # release" (fresh session, no tag established yet) fell through to the flat
        # "No imported local evidence" message. _is_latest_ib_question didn't
        # recognize "release" as a synonym for "IB", so the "relval" capability's
        # latest-resolution fallback never resolved "latest" into an actual tag.
        tag = "CMSSW_20_1_ASAN_X_2026-08-24-2300"
        latest_tags.return_value = (tag, "CMSSW_20_1_ASAN_X_2026-08-24-1100")
        get_failing_relvals.return_value = [
            {"ib_tag": tag, "name": "136.899", "workflow_id": "136.899", "errors": 3, "exitcode": 1},
        ]
        session_id = "relval-latest-release-no-tag"
        result = _execute_planned_capability(
            "relval", "what relvals are failing in latest release", session_id
        )

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertIn("136.899", answer)
        get_failing_relvals.assert_called_once_with(tag, arch=None, limit=None)

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
        self.assertIn("failing RelVal(s)** for", answer)  # the list heading, not the count-table heading

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
    def test_relval_list_answer_omits_arch_clause_when_arch_is_none(self, get_failing_relvals, get_exit_code_names):
        # Regression: found live 2026-09-18 -- "what relvals are failing in
        # CMSSW_20_1_X_2026-09-16-1100" (no architecture named) rendered the literal
        # word "None" ("...and its sub-IBs on `None` are:") because context["arch"]
        # was f-string-interpolated directly instead of being guarded the way every
        # other architecture-scoped message in this codebase already is (see
        # _relval_error_count_answer's `f" on \`{arch}\`" if arch else " across all
        # imported architectures"` pattern).
        context = {"tag": "CMSSW_20_1_X_2026-09-16-1100", "arch": None}
        get_failing_relvals.return_value = [
            {"ib_tag": "CMSSW_20_1_PY314_X_2026-09-16-1100", "name": "RelValQCD", "workflow_id": "511.0",
             "errors": 0, "exitcode": 62720},
        ]

        answer = _failed_relval_list_answer("what relvals are failing in CMSSW_20_1_X_2026-09-16-1100", context)

        self.assertNotIn("None", answer)
        self.assertIn("and its sub-IBs:", answer)

    @patch("server.api_server.find_prs_for_error", return_value=[])
    @patch("server.api_server.get_failing_relvals")
    def test_relval_pr_answer_omits_arch_clause_when_arch_is_none(self, get_failing_relvals, find_prs_for_error):
        # Same bug class as test_relval_list_answer_omits_arch_clause_when_arch_is_none,
        # found in the sibling PR-evidence helper -- it had the exact same unguarded
        # f-string interpolation of context["arch"].
        from server.api_server import _failed_relval_pr_answer

        context = {"tag": "CMSSW_20_1_X_2026-09-16-1100", "arch": None}
        get_failing_relvals.return_value = [
            {"ib_tag": "CMSSW_20_1_PY314_X_2026-09-16-1100", "name": "RelValQCD", "workflow_id": "511.0",
             "errors": 0, "exitcode": 62720},
        ]

        result = _failed_relval_pr_answer("any pr related to these failures", context)

        self.assertIsNotNone(result)
        answer, _ = result
        self.assertNotIn("None", answer)

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

    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server.get_driver")
    def test_unittest_capability_uses_session_ib_context_when_tag_omitted(self, get_driver, _get_failing_unittests):
        # Regression guard: unlike "relval" (fixed above), the "unittest"
        # capability already threads _latest_ib_contexts.get(session_id) as a
        # fallback tag (_unittest_summary_answer's `context_tag` parameter).
        # Pins that behavior so a future refactor can't silently drop it.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        # Two distinct queries now run against this mocked driver: the
        # unit-test scope records, then _sub_ib_variant_names' separate
        # "does this tag have sub-IB siblings" lookup -- give each its own
        # shaped result instead of reusing one list for both.
        session.run.side_effect = [
            [{"arch": "el9_amd64_gcc14", "is_default": True, "total": 10, "failed": 1, "passed": 9, "errors": 2}],
            [],
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
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    def test_unittest_capability_answers_failed_package_list_questions(self, _sub_ib_variant_tags, get_failing_unittests):
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

    def test_exception_category_from_signature_recovers_bare_category(self):
        self.assertEqual(
            _exception_category_from_signature("An exception of category 'LogicError' occurred while"),
            "LogicError",
        )
        self.assertIsNone(_exception_category_from_signature("not a recognized signature"))

    @patch("server.api_server.get_driver")
    def test_pasted_exception_workflow_matches_lists_workflow_and_sub_ib_variant(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {
                "arch": "el10_amd64_gcc15", "variant": "primary", "workflow_id": "140.051",
                "name": "RunMuonEG2022C", "exception": "LogicError: A muon candidate track has too many stations.",
            },
            {
                "arch": "el9_amd64_gcc14", "variant": "CLANG", "workflow_id": "140.051",
                "name": "RunMuonEG2022C", "exception": "LogicError: A muon candidate track has too many stations.",
            },
        ]

        result = _pasted_exception_workflow_matches("LogicError", "CMSSW_20_1_X_2026-09-08-2300", None)

        self.assertIn("`CMSSW_20_1_X_2026-09-08-2300`", result)
        self.assertIn("`LogicError`", result)
        self.assertIn("- `el10_amd64_gcc15` workflow `140.051` - RunMuonEG2022C:", result)
        self.assertIn("- `el9_amd64_gcc14` (`CLANG`) workflow `140.051` - RunMuonEG2022C:", result)
        self.assertEqual(
            session.run.call_args.kwargs,
            {"tag": "CMSSW_20_1_X_2026-09-08-2300", "arch": None, "category": "LogicError", "limit": 20},
        )

    @patch("server.api_server.get_driver")
    def test_pasted_exception_workflow_matches_returns_none_without_rows(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []

        self.assertIsNone(
            _pasted_exception_workflow_matches("TotallyUnmatchedCategory", "CMSSW_20_1_X_2026-09-08-2300", None)
        )

    @patch("server.api_server._pasted_exception_workflow_matches")
    @patch("server.api_server._pasted_exception_search_answer")
    @patch("server.api_server.get_driver")
    def test_pasted_exception_search_prepends_session_workflow_matches_when_tag_known(
        self, get_driver, pasted_exception_search_answer, pasted_exception_workflow_matches
    ):
        # A pasted exception alone only names a *category* -- which workflow/IB is
        # actually hitting it today is a separate lookup that only makes sense once
        # a tag is already known for the session (an earlier turn named one), the
        # same _latest_ib_contexts session-fallback pattern used elsewhere in this
        # file. Driver calls unrelated to this path (planning-decision storage,
        # pending-clarification bookkeeping) are wrapped in try/except in the real
        # code and just log a warning on a mismatched mock, so a blanket empty
        # default is enough for them here.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.__iter__.return_value = iter([])
        session.run.return_value.single.return_value = None

        pasted_exception_search_answer.return_value = (
            "Imported GitHub matches 1-1 of 1 for exception `An exception of category 'LogicError' occurred while`:\n"
            "- **Issue #99999** - Some LogicError issue (state `open`; match: body)",
            [],
        )
        pasted_exception_workflow_matches.return_value = (
            "Workflows in `CMSSW_20_1_X_2026-09-08-2300` currently showing a `LogicError` exception:\n"
            "- `el10_amd64_gcc15` workflow `140.051` - RunMuonEG2022C: `LogicError: ...`"
        )

        session_id = "session-with-established-ib-tag"
        _latest_ib_contexts[session_id] = "CMSSW_20_1_X_2026-09-08-2300"
        try:
            result = chat(ChatRequest(
                session_id=session_id,
                question="An exception of category 'LogicError' occurred while .",
            ))
        finally:
            _latest_ib_contexts.pop(session_id, None)
            _latest_ib_arch_contexts.pop(session_id, None)

        self.assertIn("workflow `140.051`", result.answer)
        self.assertIn("Imported GitHub matches", result.answer)
        self.assertLess(
            result.answer.index("workflow `140.051`"), result.answer.index("Imported GitHub matches"),
            "the workflow/IB evidence should be prepended ahead of the GitHub evidence",
        )
        pasted_exception_workflow_matches.assert_called_once_with("LogicError", "CMSSW_20_1_X_2026-09-08-2300", None)

    @patch("server.api_server._pasted_exception_workflow_matches")
    @patch("server.api_server._pasted_exception_search_answer")
    @patch("server.api_server.get_driver")
    def test_pasted_exception_search_skips_workflow_lookup_without_session_tag(
        self, get_driver, pasted_exception_search_answer, pasted_exception_workflow_matches
    ):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.__iter__.return_value = iter([])
        session.run.return_value.single.return_value = None

        pasted_exception_search_answer.return_value = (
            "Imported GitHub matches 1-1 of 1 for exception `An exception of category 'LogicError' occurred while`:\n"
            "- **Issue #99999** - Some LogicError issue (state `open`; match: body)",
            [],
        )

        result = chat(ChatRequest(
            session_id="session-with-no-prior-ib-tag",
            question="An exception of category 'LogicError' occurred while .",
        ))

        self.assertIn("Imported GitHub matches", result.answer)
        self.assertNotIn("Workflows in", result.answer)
        pasted_exception_workflow_matches.assert_not_called()

    @patch("server.api_server._pasted_exception_search_answer")
    @patch("server.api_server.get_driver")
    def test_pasted_exception_wins_over_an_incidental_capability_keyword(
        self, get_driver, pasted_exception_search_answer
    ):
        # Real bug found live, 2026-10-02: a pasted CMSSW crash log's own
        # Jenkins build-log file path (".../build-any-ib/w/el9_amd64_gcc14/
        # ...") contains the literal substring "build", which matches the
        # "build" capability's keyword trigger. RequestPlanner used to score
        # this DETERMINISTIC for "build" -- a capability with no handler for
        # this shape -- so the pasted-exception search (only reachable from
        # the post-RequestPlanner legacy waterfall) was never even tried,
        # landing on the generic "I couldn't match that" fallback instead.
        # Pasted-exception detection must win regardless of what capability
        # keywords happen to appear incidentally in the pasted text.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.__iter__.return_value = iter([])
        session.run.return_value.single.return_value = None
        pasted_exception_search_answer.return_value = (
            "Imported GitHub matches 1-1 of 1 for exception "
            "`An exception of category 'StdException' occurred while`:\n"
            "- **Issue #51512** - Some StdException issue (state `open`; match: body)",
            [],
        )

        question = (
            "An exception of category 'StdException' occurred while\n"
            "Exception Message:\n"
            "A std::exception was thrown.\n"
            "/data/cmsbld/jenkins/workspace/build-any-ib/w/el9_amd64_gcc14/external/alpaka/"
            "include/alpaka/kernel/TaskKernelGpuUniformCudaHipRt.hpp(275) "
            "'cudaErrorLaunchOutOfResources': 'too many resources requested for launch'!"
        )
        result = chat(ChatRequest(session_id="pasted-exception-build-keyword", question=question))

        self.assertIn("Imported GitHub matches", result.answer)
        self.assertEqual(result.route, "deterministic")
        # the page size depends on whether the (mocked) name search found anything -- 5 when
        # precise evidence exists, else 20 -- which is not what this test is about
        pasted_exception_search_answer.assert_called_once()
        self.assertEqual(pasted_exception_search_answer.call_args.args[0],
                         "An exception of category 'StdException' occurred while")

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

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_primary_relval_failure_count_lists_every_architecture_up_front(self, scope_records, _sub_ib_variant_tags, _get_failing_relvals):
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

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_relval_failure_count_all_returns_aggregate(self, scope_records, _get_failing_relvals):
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

    def test_relval_scope_or_choice_returns_every_architecture(self):
        # Several flagged default used to ask, offering only those (2026-11-04: el10 and el8 of 4
        # imported); every answer has a per-architecture table, so all are shown instead.
        tag = "CMSSW_20_1_X_2026-08-24-2300"
        two_defaults = [
            {"tag": tag, "arch": "el9_amd64_gcc14", "is_relval_default": True},
            {"tag": tag, "arch": "el9_aarch64_gcc14", "is_relval_default": True},
        ]
        try:
            result = _relval_scope_or_choice("scope-choice-tie", tag, two_defaults, all_requested=False)
            self.assertEqual(result, two_defaults)
            self.assertNotIn("scope-choice-tie", _pending_relval_scopes)

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

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._relval_scope_records")
    @patch("server.api_server.get_driver")
    def test_relval_failure_count_mentions_sub_ib_variants(self, get_driver, scope_records, _get_failing_relvals):
        # Sub-IB variants (ASAN, UBSAN, ROOT6, ...) are never merged into a
        # primary tag's RelVal numbers (different (tag, architecture)
        # identities, per docs/DEVELOPER.md), but their existence should be
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

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_sub_ib_relval_failure_count_lists_every_architecture_up_front(self, scope_records, _sub_ib_variant_tags, _get_failing_relvals):
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
        # A choice still pending (the IB -> architecture flow) rejects an architecture not listed.
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
            _pending_relval_scopes[session_id] = {
                "root_tag": primary_ib, "stage": "arch", "records": scope_records.return_value,
            }
            answer = _relval_failure_count_answer("el11_amd64_gcc16", session_id)
            self.assertIn("not available", answer)
            self.assertIn(session_id, _pending_relval_scopes)
        finally:
            _pending_relval_scopes.pop(session_id, None)

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_any_failing_relvals_with_one_architecture_returns_exact_ib(self, scope_records, _sub_ib_variant_tags, _get_failing_relvals):
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

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_relval_failure_count_never_hides_non_default_architectures(self, scope_records, _sub_ib_variant_tags, _get_failing_relvals):
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

    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_shows_each_architecture_when_several_are_default(self, scope_records, *_):
        # Several flagged default: no question, and each architecture keeps its own row.
        scope_records.return_value = [
            {"arch": "el9_amd64_gcc13", "is_default": False, "total": 950, "failed": 3, "passed": 947, "errors": 3},
            {"arch": "el9_amd64_gcc15", "is_default": True, "total": 950, "failed": 3, "passed": 947, "errors": 3},
            {"arch": "el9_amd64_gcc16", "is_default": True, "total": 950, "failed": 4, "passed": 946, "errors": 4},
        ]

        answer = _unittest_summary_answer("and unit tests", "CMSSW_20_1_X_2026-08-09-0000", "s-multi-unittest")

        self.assertNotIn("Choose one", answer)
        self.assertIn("| `el9_amd64_gcc13` | 3 |", answer)
        self.assertIn("| `el9_amd64_gcc15` | 3 |", answer)
        self.assertIn("| `el9_amd64_gcc16` | 4 |", answer)
        self.assertNotIn("s-multi-unittest", _pending_unittest_scopes)

    @patch("server.api_server.get_failing_unittests")
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_names_the_failing_package_inline(self, scope_records, _sub_ib_variant_tags, get_failing_unittests):
        # Regression: found live 2026-09-21 -- a real user asked "unit tests
        # for <tag>", got "1 failed across 375 package result(s)" with no
        # indication of WHICH package, and had to ask a completely separate
        # follow-up ("which unit test package failed") to learn it was
        # HLTrigger/Tools. The summary itself must name the failing
        # package(s), not just the count -- a separate detailed-list handler
        # already existed (_failed_unittest_list_answer) but requiring a
        # second question for a single-failure build is a worse default.
        scope_records.return_value = [
            {"arch": "el9_amd64_gcc14", "is_default": True, "total": 375, "failed": 1, "passed": 374, "errors": 1},
        ]
        get_failing_unittests.return_value = [{"name": "HLTrigger/Tools", "errors": 1}]

        answer = _unittest_summary_answer(
            "unit tests for CMSSW_20_1_MULTIARCHSV4_X_2026-09-14-2300", None, "s-name-inline-unittest",
        )

        self.assertIn("HLTrigger/Tools", answer)
        get_failing_unittests.assert_called_once_with(
            "CMSSW_20_1_MULTIARCHSV4_X_2026-09-14-2300", arch="el9_amd64_gcc14", limit=10
        )

    @patch("server.api_server.get_failing_unittests")
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_caps_and_labels_more_when_many_packages_fail(self, scope_records, _sub_ib_variant_tags, get_failing_unittests):
        # A genuinely broad failure set must stay a short pointer, not turn
        # the summary into the full detailed list -- that's what
        # _failed_unittest_list_answer is for.
        scope_records.return_value = [
            {"arch": "el9_amd64_gcc14", "is_default": True, "total": 375, "failed": 15, "passed": 360, "errors": 15},
        ]
        get_failing_unittests.return_value = [
            {"name": f"Package/Number{i}", "errors": 1} for i in range(10)
        ]

        answer = _unittest_summary_answer(
            "unit tests for CMSSW_20_1_MULTIARCHSV4_X_2026-09-14-2300", None, "s-capped-unittest",
        )

        self.assertIn("Package/Number0", answer)
        self.assertIn("(+5 more;", answer)
        get_failing_unittests.assert_called_once_with(
            "CMSSW_20_1_MULTIARCHSV4_X_2026-09-14-2300", arch="el9_amd64_gcc14", limit=10
        )

    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_never_hides_non_default_architectures(self, scope_records, _sub_ib_variant_tags, _get_failing_unittests):
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

    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_all_recovers_full_architecture_set(self, scope_records, _sub_ib_variant_tags, _get_failing_unittests):
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
            _pending_unittest_scopes[session_id] = {"tag": "CMSSW_20_1_X_2026-08-25-2300",
                                                    "records": scope_records.return_value}
            answer = _unittest_summary_answer("all", None, session_id)
        finally:
            _pending_unittest_scopes.pop(session_id, None)

        self.assertIn("el8_amd64_gcc14", answer)
        self.assertIn("el9_amd64_gcc14", answer)
        self.assertIn("el9_aarch64_gcc14", answer)
        self.assertIn("el10_amd64_gcc15", answer)
        self.assertIn("11 failed", answer)

    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_remembers_session_architecture_for_tagless_followup(self, scope_records, _sub_ib_variant_tags, _get_failing_unittests):
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

    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server.get_driver")
    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_mentions_sub_ib_variants(self, scope_records, get_driver, _get_failing_unittests):
        # Regression: found live -- a primary tag's unit-test summary gave no
        # hint that CLANG/DEVEL/MULTIARCHS/NONLTO/ROOT6 sub-IB variants exist
        # for the same build, each running its own separate unit-test suite
        # with its own failure count (confirmed against the CMSSDT
        # dashboard) -- the same "mention, never merge" treatment RelVal
        # already gets (test_relval_failure_count_mentions_sub_ib_variants).
        tag = "CMSSW_20_1_X_2026-08-25-2300"
        # Only the primary tag has records in this fixture -- each sub-IB
        # variant gets an empty list, so its inlined section reports "no
        # imported records" instead of (wrongly) reusing the primary's data.
        primary_records = [
            {"arch": "el9_amd64_gcc14", "is_default": True, "total": 950, "failed": 3, "passed": 947, "errors": 3},
        ]
        scope_records.side_effect = lambda queried_tag: primary_records if queried_tag == tag else []
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"variant": "CLANG", "tag": "CMSSW_20_1_CLANG_X_2026-08-25-2300"},
            {"variant": "DEVEL", "tag": "CMSSW_20_1_DEVEL_X_2026-08-25-2300"},
            {"variant": "MULTIARCHS", "tag": "CMSSW_20_1_MULTIARCHS_X_2026-08-25-2300"},
        ]

        answer = _unittest_summary_answer("unit tests", tag, "unittest-sub-ib-mention")

        # Never-imported variants are still named (never silently merged or
        # omitted), just collapsed to one compact line each instead of a
        # full "no records" sentence per variant (refined 2026-09-30).
        self.assertIn("No imported unit-test records: `CLANG`, `DEVEL`, `MULTIARCHS`", answer)

    @patch("server.api_server.get_driver")
    @patch("server.api_server.get_failing_unittests")
    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_puts_failing_variants_first_and_collapses_clean_ones(
        self, scope_records, get_failing_unittests, get_driver
    ):
        # Real bug found live 2026-09-30 (comparing against the CMSSDT
        # dashboard): an 11-variant tag with only 2 real failures buried
        # them under 7 identical "0 failed" sub-IB sections. Failing
        # variants must appear in full, immediately after the primary
        # summary; clean ones collapse to one line; never-imported ones to
        # a separate line -- nothing omitted, just not spelled out at full
        # length when there's nothing to report.
        tag = "CMSSW_20_1_X_2026-09-20-2300"
        primary_records = [
            {"arch": "el9_amd64_gcc14", "is_default": True, "total": 376, "failed": 0, "passed": 376, "errors": 0},
        ]
        clang_records = [
            {"arch": "el9_amd64_gcc14", "is_default": True, "total": 376, "failed": 2, "passed": 374, "errors": 2},
        ]
        devel_records = [
            {"arch": "el9_amd64_gcc14", "is_default": True, "total": 376, "failed": 0, "passed": 376, "errors": 0},
        ]

        def scope_side_effect(queried_tag):
            if queried_tag == tag:
                return primary_records
            if "CLANG" in queried_tag:
                return clang_records
            if "DEVEL" in queried_tag:
                return devel_records
            return []

        scope_records.side_effect = scope_side_effect
        get_failing_unittests.return_value = [{"name": "SimG4Core/Geometry", "errors": 2}]
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"variant": "CLANG", "tag": "CMSSW_20_1_CLANG_X_2026-09-20-2300"},
            {"variant": "DEVEL", "tag": "CMSSW_20_1_DEVEL_X_2026-09-20-2300"},
            {"variant": "MULTIARCHS", "tag": "CMSSW_20_1_MULTIARCHS_X_2026-09-20-2300"},
        ]

        answer = _unittest_summary_answer("unit tests", tag, "unittest-mixed-variants")

        self.assertIn("Sub-IB variant `CLANG`", answer)
        self.assertIn("SimG4Core/Geometry", answer)
        self.assertIn("✅ Clean, no failures: `DEVEL`", answer)
        self.assertIn("No imported unit-test records: `MULTIARCHS`", answer)
        # Ordering: the failing variant's block must come before the clean
        # summary line, which must come before the missing-records line.
        self.assertLess(answer.index("Sub-IB variant `CLANG`"), answer.index("✅ Clean"))
        self.assertLess(answer.index("✅ Clean"), answer.index("No imported unit-test records"))

    @patch("server.api_server.get_driver")
    @patch("server.api_server.get_failing_unittests")
    def test_failed_unittest_list_answer_mentions_sub_ib_variants(self, get_failing_unittests, get_driver):
        tag = "CMSSW_20_1_X_2026-08-25-2300"
        get_failing_unittests.return_value = [{"name": "DQM/Physics", "errors": 5}]
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"variant": "ROOT6", "tag": "CMSSW_20_1_ROOT6_X_2026-08-25-2300"}]

        answer = _failed_unittest_list_answer(
            "what 3 failed packages?", tag, "el9_aarch64_gcc14",
        )

        self.assertIn("Sub-IB variant `ROOT6` (`CMSSW_20_1_ROOT6_X_2026-08-25-2300`)", answer)
        # DQM/Physics shows up under both the primary tag and the inlined
        # ROOT6 section since get_failing_unittests is mocked identically
        # for every tag in this fixture.
        self.assertEqual(answer.count("DQM/Physics"), 2)

    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server.get_failing_unittests")
    def test_failed_unittest_list_answer_lists_package_names(self, get_failing_unittests, _sub_ib_variant_tags):
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

    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server.get_failing_unittests")
    def test_failed_unittest_list_answer_recognizes_which_phrasing(self, get_failing_unittests, _sub_ib_variant_tags):
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

    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_resolves_after_choosing_an_architecture(self, scope_records, _sub_ib_variant_tags, _get_failing_unittests):
        scope_records.return_value = [
            {"arch": "el9_amd64_gcc15", "is_default": True, "total": 950, "failed": 3, "passed": 947, "errors": 3},
            {"arch": "el9_amd64_gcc16", "is_default": True, "total": 950, "failed": 4, "passed": 946, "errors": 4},
        ]
        _pending_unittest_scopes["s-resume-unittest"] = {"tag": "CMSSW_20_1_X_2026-08-09-0000",
                                                         "records": scope_records.return_value}

        answer = _unittest_summary_answer("el9_amd64_gcc16", None, "s-resume-unittest")

        self.assertIn("4 failed", answer)
        self.assertNotIn("s-resume-unittest", _pending_unittest_scopes)

    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_honors_an_explicit_architecture_up_front(self, scope_records, _sub_ib_variant_tags, _get_failing_unittests):
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

    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._unittest_scope_records")
    def test_unittest_summary_recognizes_unit_without_the_word_test(self, scope_records, _sub_ib_variant_tags, _get_failing_unittests):
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

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_repeated_relval_question_shows_every_architecture(self, scope_records, *_):
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

            self.assertNotIn("architecture-specific", answer)
            self.assertIn("el9_aarch64_gcc13", answer)
            self.assertIn("el9_amd64_gcc13", answer)
            self.assertNotIn(session_id, _pending_relval_scopes)
            self.assertNotIn("CMSSW_20_1_TF_X_2026-08-10-1100", answer)
        finally:
            _pending_relval_scopes.pop(session_id, None)

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_relval_data_availability_is_graph_backed(self, scope_records, _sub_ib_variant_tags, _get_failing_relvals):
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

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_relval_data_availability_shows_each_architecture_when_several_are_default(self, scope_records, *_):
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        scope_records.return_value = [
            {"tag": tag, "arch": "el9_amd64_gcc15", "relval_total": 950, "failed_relvals": 3, "passed_relvals": 947, "is_relval_default": True},
            {"tag": tag, "arch": "el9_amd64_gcc16", "relval_total": 950, "failed_relvals": 4, "passed_relvals": 946, "is_relval_default": True},
        ]

        answer = _relval_data_availability_answer(f"do you have data for {tag}?", "s-avail-multi")

        self.assertNotIn("Choose one", answer)
        self.assertIn("el9_amd64_gcc15", answer)
        self.assertIn("el9_amd64_gcc16", answer)

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_relval_error_count_shows_each_architecture_when_several_are_default(self, scope_records, *_):
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        scope_records.return_value = [
            {"tag": tag, "arch": "el9_amd64_gcc15", "relval_total": 950, "failed_relvals": 3, "passed_relvals": 947,
             "error_count": 100, "failed_errors": 10, "passed_errors": 90, "is_relval_default": True},
            {"tag": tag, "arch": "el9_amd64_gcc16", "relval_total": 950, "failed_relvals": 4, "passed_relvals": 946,
             "error_count": 200, "failed_errors": 20, "passed_errors": 180, "is_relval_default": True},
        ]

        answer = _relval_error_count_answer(f"how many relvals errors in {tag}", "s-errcount-multi")

        self.assertNotIn("Choose one", answer)
        self.assertIn("el9_amd64_gcc15", answer)
        self.assertIn("el9_amd64_gcc16", answer)
        scope_records.assert_called_once_with(tag, include_sub_ibs=True)

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_ambiguous_named_relval_question_defaults_to_graph_summary(self, scope_records, _sub_ib_variant_tags, _get_failing_relvals):
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

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_relval_ib_summary_answer_honors_an_explicit_architecture(
        self, scope_records, _sub_ib_variant_tags, _get_failing_relvals
    ):
        # Regression: found live 2026-09-21 -- "failed RelVals for <tag>
        # el9_amd64_gcc14" (the exact canned reply from this project's own
        # scope clarification) still answered with every imported
        # architecture, because this handler (unlike its sibling
        # _relval_failure_count_answer) never checked for an architecture
        # named explicitly in the question at all.
        tag = "CMSSW_20_1_X_2026-09-13-2300"
        scope_records.return_value = [
            {"tag": tag, "arch": "el8_amd64_gcc14", "relval_total": 954, "failed_relvals": 1, "passed_relvals": 953, "is_relval_default": False},
            {"tag": tag, "arch": "el9_aarch64_gcc14", "relval_total": 954, "failed_relvals": 0, "passed_relvals": 904, "known_error_relvals": 50, "is_relval_default": False},
            {"tag": tag, "arch": "el9_amd64_gcc14", "relval_total": 954, "failed_relvals": 0, "passed_relvals": 954, "is_relval_default": False},
        ]

        answer = _relval_ib_summary_answer(f"failed RelVals for {tag} el9_amd64_gcc14", "s-relval-arch")

        self.assertIn("el9_amd64_gcc14", answer)
        self.assertNotIn("el8_amd64_gcc14", answer)
        self.assertNotIn("el9_aarch64_gcc14", answer)

    @patch("server.api_server._relval_known_error_lines", new=lambda rows: [])
    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags")
    @patch("server.api_server._relval_scope_records")
    def test_relval_ib_summary_answer_breaks_down_every_sub_ib_for_one_architecture(
        self, scope_records, sub_ib_variant_tags, _get_failing_relvals
    ):
        # Regression: found live 2026-09-21 -- "how many ibs are failing for
        # el9_amd64_gcc14 for release <tag>" genuinely means "across every
        # IB in this release" (the DEFAULT/primary tag is only 1 of 11), but
        # the answer only ever showed the primary tag's own number plus a
        # flat "10 sub-IB variant(s) not included above -- ask individually"
        # footnote, requiring 10 separate follow-up questions to see them.
        tag = "CMSSW_20_1_X_2026-09-13-2300"
        devel_tag = "CMSSW_20_1_DEVEL_X_2026-09-13-2300"
        evtgen3_tag = "CMSSW_20_1_EVTGEN3_X_2026-09-13-2300"
        sub_ib_variant_tags.return_value = [
            {"variant": "DEVEL", "tag": devel_tag},
            {"variant": "EVTGEN3", "tag": evtgen3_tag},
        ]

        def scope_records_for(queried_tag, include_sub_ibs=False):
            data = {
                tag: [{"tag": tag, "arch": "el9_amd64_gcc14", "relval_total": 954, "failed_relvals": 0, "passed_relvals": 954}],
                devel_tag: [{"tag": devel_tag, "arch": "el9_amd64_gcc14", "relval_total": 954, "failed_relvals": 2, "passed_relvals": 952}],
                evtgen3_tag: [{"tag": evtgen3_tag, "arch": "el9_amd64_gcc14", "relval_total": 954, "failed_relvals": 11, "passed_relvals": 941, "known_error_relvals": 2}],
            }
            return data[queried_tag]
        scope_records.side_effect = scope_records_for

        answer = _relval_ib_summary_answer(
            f"how many relvals are failing for el9_amd64_gcc14 for release {tag}", "s-relval-breakdown"
        )

        self.assertIn(tag, answer)
        self.assertIn(devel_tag, answer)
        self.assertIn(evtgen3_tag, answer)
        self.assertIn("DEVEL", answer)
        self.assertIn("EVTGEN3", answer)
        self.assertIn("2 of 3 IB(s)", answer)

    @patch("server.api_server.get_exit_code_names", return_value={8021: "MethodFailure"})
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_relval_scope_summary_names_the_failing_workflow_inline(
        self, scope_records, _sub_ib_variant_tags, get_failing_relvals, _get_exit_code_names
    ):
        # Same "the summary itself should carry enough information" fix as
        # test_unittest_summary_names_the_failing_package_inline -- a
        # shifter asking for RelVal results on a build with only 1 failure
        # should see the actual failing workflow and its exit code, not
        # just "1 failing RelVal(s)" with no indication of which one.
        tag = "CMSSW_20_1_X_2026-09-13-2300"
        scope_records.return_value = [
            {"tag": tag, "arch": "el9_amd64_gcc14", "relval_total": 954, "failed_relvals": 1, "passed_relvals": 953, "is_relval_default": False},
        ]
        get_failing_relvals.return_value = [
            {"ib_tag": tag, "name": "TTbar_14TeV", "workflow_id": "140.0", "errors": 1, "exitcode": 8021},
        ]

        answer = _relval_ib_summary_answer(f"failed RelVals for {tag} el9_amd64_gcc14", "s-relval-name")

        self.assertIn("TTbar_14TeV", answer)
        self.assertIn("140.0", answer)
        self.assertIn("8021", answer)
        get_failing_relvals.assert_called_once_with(tag, arch="el9_amd64_gcc14", limit=10)

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_relval_scope_summary_excludes_sub_ib_workflows_from_inline_list(
        self, scope_records, _sub_ib_variant_tags, get_failing_relvals, _get_exit_code_names
    ):
        # get_failing_relvals also matches any sub-IB whose parent_tag is
        # this tag -- the inline "Failing workflows:" list must filter those
        # out client-side, the same way the table above it never includes
        # sub-IB rows either (see the "not included above" note).
        tag = "CMSSW_20_1_X_2026-09-13-2300"
        scope_records.return_value = [
            {"tag": tag, "arch": "el9_amd64_gcc14", "relval_total": 954, "failed_relvals": 1, "passed_relvals": 953, "is_relval_default": False},
        ]
        get_failing_relvals.return_value = [
            {"ib_tag": tag, "name": "TTbar_14TeV", "workflow_id": "140.0", "errors": 1, "exitcode": 8021},
            {"ib_tag": "CMSSW_20_1_DEVEL_X_2026-09-13-2300", "name": "OtherWF", "workflow_id": "141.0", "errors": 1, "exitcode": 1},
        ]

        answer = _relval_ib_summary_answer(f"failed RelVals for {tag} el9_amd64_gcc14", "s-relval-filter")

        self.assertIn("TTbar_14TeV", answer)
        self.assertNotIn("OtherWF", answer)

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_relval_scope_summary_uses_status_emoji(self, scope_records, _sub_ib_variant_tags, _get_failing_relvals):
        tag = "CMSSW_20_1_X_2026-09-13-2300"
        scope_records.return_value = [
            {"tag": tag, "arch": "el8_amd64_gcc14", "relval_total": 954, "failed_relvals": 1, "passed_relvals": 953, "is_relval_default": False},
        ]

        answer = _relval_ib_summary_answer(f"failed RelVals for {tag}", "s-relval-emoji")

        self.assertIn("🔴", answer)

    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[])
    @patch("server.api_server._relval_scope_records")
    def test_generic_relval_summary_shows_each_architecture_when_several_are_default(self, scope_records, *_):
        tag = "CMSSW_20_1_X_2026-08-09-0000"
        scope_records.return_value = [
            {"tag": tag, "arch": "el9_amd64_gcc13", "relval_total": 950, "failed_relvals": 5, "passed_relvals": 945, "is_relval_default": False},
            {"tag": tag, "arch": "el9_amd64_gcc15", "relval_total": 950, "failed_relvals": 5, "passed_relvals": 945, "is_relval_default": True},
            {"tag": tag, "arch": "el9_amd64_gcc16", "relval_total": 950, "failed_relvals": 5, "passed_relvals": 945, "is_relval_default": True},
        ]

        answer = _relval_ib_summary_answer(f"show RelVal results for {tag}", "s-multi-default")

        self.assertNotIn("Choose one", answer)
        for arch in ("el9_amd64_gcc13", "el9_amd64_gcc15", "el9_amd64_gcc16"):
            self.assertIn(arch, answer)
        self.assertNotIn("s-multi-default", _pending_relval_scopes)

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
        self.assertIn("in its title", answer)
        self.assertEqual(session.run.call_args.kwargs["lid"], "37834.0")
        get_driver.return_value.close.assert_called()

    @patch("server.api_server.get_driver")
    def test_relval_exception_answer_never_traverses_has_exception(self, get_driver):
        # Same bug, same function family as
        # test_relval_exception_list_answer_never_traverses_has_exception --
        # found live 2026-10-02 via a real "what is the latest exception"
        # question that confidently (and wrongly) reported zero exceptions
        # exist, when 9,612 real RelValException nodes were imported.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.side_effect = [
            [{
                "exception": "LogicError: boom", "category": "LogicError", "source_url": None,
                "context": None, "name": "TTbar",
            }],
            [],
            [],
        ]

        answer = _relval_exception_answer(
            "what exception does workflow 37834.0 have?",
            {"tag": "CMSSW_20_1_X_2026-08-12-1100", "arch": "el9_amd64_gcc13"},
        )

        exception_query = session.run.call_args_list[0].args[0]
        related_query = session.run.call_args_list[2].args[0]
        self.assertNotIn("HAS_EXCEPTION", exception_query)
        self.assertIn("exception.ib_tag", exception_query)
        self.assertNotIn("HAS_EXCEPTION", related_query)
        self.assertIn("LogicError: boom", answer)

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
    def test_issue_author_recognizes_auxiliary_verb_phrasing(self, get_driver):
        # Regression: a real question, "who has created the issue #51756",
        # didn't match the old trigger regex `who\s+(?:opened|created|...)`
        # because of the auxiliary "has" between "who" and "created" -- not
        # a data gap (the issue in question genuinely isn't imported), but a
        # phrasing gap that made even an imported issue's author unreachable
        # this way. This only pins the trigger regex, so the driver is
        # mocked to the real "not imported" shape (a driver call still
        # happens once the regex matches -- a prior version of this test
        # skipped the mock on the assumption it wouldn't, and only passed
        # because Neo4j happened to be reachable locally).
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = None

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
        self.assertTrue(answer.startswith("Issue #51756 is not present in the imported CMSSW graph."))
        self.assertIn("export", answer)  # says how far the imported GitHub data reaches (2026-11-04)

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

    @patch("server.api_server.get_driver")
    def test_issue_detail_answer_recognizes_bare_existence_check(self, get_driver):
        # Real bug found live, 2026-10-02: "do you have issue 52041?" names a
        # concrete issue number with no other qualifier word ("title"/
        # "about"/"who opened"/...) and fell all the way through to the
        # generic "I couldn't match that" fallback, even though the number
        # alone is already enough to answer -- same sensible default this
        # function already applies to an unqualified bare number.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = None  # genuinely not imported

        answer = _issue_detail_answer("do you have issue 52041?")

        self.assertTrue(answer.startswith("Neither an Issue nor a PR numbered #52041 is present in the imported graph."))
        self.assertIn("export", answer)  # says how far the imported GitHub data reaches (2026-11-04)

    @patch("server.api_server.get_driver")
    def test_issue_detail_answer_existence_check_reports_real_issue(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "title": "edm::RefToBase is deprecated", "state": "open", "created_at": "2023-09-07T20:16:18Z",
        }

        answer = _issue_detail_answer("do you have issue #42737?")

        self.assertIn("#42737", answer)
        self.assertIn("edm::RefToBase is deprecated", answer)

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

        latest_ib_record.assert_called_once_with("el9_amd64_gcc13", None)
        self.assertIn("for `el9_amd64_gcc13`", answer)
        self.assertIn("Sub-IB variants", answer)

    @patch("server.api_server._latest_ib_records")
    def test_latest_ib_answer_honors_an_explicit_release_series(self, latest_ib_record):
        # Found live 2026-10-06: "latest 20_0_X IB" returned the 20_1_X IB.
        latest_ib_record.return_value = [{
            "tag": "CMSSW_20_0_X_2026-10-06-1100", "release": "CMSSW_20_0_X",
            "architectures": ["el8_amd64_gcc13"], "sub_ib_tags": [], "build_timestamp": "2026-10-06-1100",
        }]

        answer = _latest_ib_answer("latest 20_0_X IB")

        latest_ib_record.assert_called_once_with(None, (20, 0))
        self.assertIn("CMSSW_20_0_X_2026-10-06-1100", answer)
        self.assertIn("release series `CMSSW_20_0_X`", answer)

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
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.KGChatSession")
    @patch("server.api_server._failure_overview_answer", return_value=("everything failing", [{"label": "a", "question": "b"}]))
    def test_ambiguous_failure_request_answers_with_failure_overview_without_llm(
        self, overview, session_cls, store_chat_turns, restore, store_decision, store_pending, clear_pending
    ):
        # "failures for <tag>" names no category. It used to get "Do you want
        # build warnings, failed RelVals, unit-test results, or a combined IB
        # summary?" -- 126 of 128 clarifications in the 2026-10-06 replay of
        # real sessions. Now answered directly with every failing item.
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        result = chat(ChatRequest(question=f"failures for {tag}", session_id="planner-failure-overview"))

        self.assertEqual(result.route, "capability:failure_overview")
        self.assertEqual(result.answer, "everything failing")
        overview.assert_called_once_with(tag, None)
        store_pending.assert_not_called()
        session_cls.assert_not_called()

    @patch("server.api_server.get_driver")
    def test_category_failure_snapshot_reports_real_counts_with_markers(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "record_count": 2, "build_failing": 0, "build_warnings": 0, "relval_failed": 3, "unit_test_failed": 0,
        }

        snapshot = _category_failure_snapshot("CMSSW_20_1_X_2026-08-10-1100")

        self.assertIn("2 imported architecture/variant record(s)", snapshot)
        self.assertIn("🔴 3 RelVal(s) failing", snapshot)
        self.assertIn("🟢 0 build(s) not passing", snapshot)
        self.assertIn("🟢 0 unit test(s) failing", snapshot)

    @patch("server.api_server.get_driver")
    def test_category_failure_snapshot_includes_sub_ib_variants(self, get_driver):
        # Regression: found live 2026-09-30 -- an earlier version matched
        # only ib.tag = $tag, silently excluding every sub-IB variant
        # (CLANG/CUDART/DEVEL/EVTGEN3/...), undercounting real failures.
        # Also confirms build_status ('error', a real build failure) is
        # reported even when comp_warnings (a different signal) is 0.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "record_count": 14, "build_failing": 1, "build_warnings": 0, "relval_failed": 15, "unit_test_failed": 4,
        }

        snapshot = _category_failure_snapshot("CMSSW_20_1_X_2026-09-20-2300")

        session.run.assert_called_once()
        query = session.run.call_args.args[0]
        self.assertIn("ib.tag = $tag OR ib.parent_tag = $tag", query)
        self.assertIn("14 imported architecture/variant record(s)", snapshot)
        self.assertIn("🔴 15 RelVal(s) failing", snapshot)
        self.assertIn("🔴 1 build(s) not passing", snapshot)
        self.assertIn("🔴 4 unit test(s) failing", snapshot)

    @patch("server.api_server.get_driver")
    def test_category_failure_snapshot_returns_none_for_unknown_tag(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "record_count": 0, "build_failing": None, "build_warnings": None,
            "relval_failed": None, "unit_test_failed": None,
        }

        self.assertIsNone(_category_failure_snapshot("CMSSW_UNKNOWN_TAG"))

    @patch("server.api_server.get_driver", side_effect=RuntimeError("boom"))
    def test_category_failure_snapshot_fails_open_on_backend_error(self, get_driver):
        # An enrichment failure must never take down the base clarification
        # -- the caller falls back to the plain question when this is None.
        self.assertIsNone(_category_failure_snapshot("CMSSW_20_1_X_2026-08-10-1100"))

    def test_planner_without_failure_overview_still_asks_for_category(self):
        # The category clarification remains the fallback for a planner built
        # without the failure_overview capability registered.
        from server.request_planner import Capability, PlanKind, RequestContext, RequestPlanner
        planner = RequestPlanner(capabilities=(
            Capability(name="relval", required_categories=frozenset({"relval"}), priority=65),
        ))
        plan = planner.plan(RequestContext.extract("failures for CMSSW_20_1_X_2026-08-10-1100"))
        self.assertEqual(plan.kind, PlanKind.CLARIFICATION)
        self.assertEqual(plan.clarification.missing_field, "category")

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._chat_response", side_effect=lambda *args, **kwargs: args[2])
    @patch("server.api_server._execute_planned_capability", return_value=("evidence", []))
    def test_category_follow_up_after_failure_overview_keeps_the_ib(
        self, execute, chat_response, restore, store_decision, store_pending, clear_pending
    ):
        session_id = "planner-category-reply"
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        try:
            chat(ChatRequest(question=f"failures for {tag}", session_id=session_id))
            self.assertEqual(execute.call_args.args[0], "failure_overview")
            self.assertEqual(_latest_ib_contexts[session_id], tag)
            chat(ChatRequest(question="unit tests", session_id=session_id))
        finally:
            _latest_ib_contexts.pop(session_id, None)
            _latest_ib_arch_contexts.pop(session_id, None)

        self.assertEqual(execute.call_args.args[0], "unittest")
        self.assertEqual(execute.call_args.args[2], session_id)  # handler resolves the session tag

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._chat_response", side_effect=lambda *args, **kwargs: args[2])
    @patch("server.api_server.KGChatSession")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.search_concepts", return_value=[])
    def test_invalid_llm_plan_is_rejected_before_tool_execution(
        self, concepts, session_cls, chat_response, restore, store_decision, clear_pending
    ):
        # Regression: a question the agentic loop can't ground in any tool
        # evidence (it never calls a tool at all) must not be a dead end --
        # it offers concrete next-step actions instead of a flat rejection.
        kgchat_factory = _mock_kgchat_session("I don't have a tool for that.", tool_results=())
        session_cls.side_effect = kgchat_factory

        result = chat(ChatRequest(question="Explain the graph topology", session_id="invalid-plan"))

        self.assertIn("couldn't match that to a specific local answer", result)
        kgchat_factory.sessions[0].ask.assert_called_once()
        actions = chat_response.call_args.kwargs["actions"]
        self.assertIn({"label": "Latest IB", "question": "what is the latest IB"}, actions)

    @patch("server.api_server._retriever", None)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.KGChatSession")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.search_concepts", return_value=[])
    def test_llm_fallback_answer_rejected_when_not_grounded_in_evidence(
        self, concepts, session_cls, store_chat_turns, restore, store_decision, clear_pending
    ):
        # Regression guard for the one path with no other verification: the
        # agentic loop's own final free-text answer is only trusted if every
        # number in it traces back to real tool evidence it actually called --
        # a fabricated number must not reach the user as if it were a graph fact.
        session_cls.side_effect = _mock_kgchat_session(
            "There are 42 failing workflows on CMSSW_20_1_X_2026-08-17-2300.",
            tool_results=[(
                "get_ib_summary", {},
                {"tag": "CMSSW_20_1_X_2026-08-17-2300", "failing_count": 5},
            )],
        )

        result = chat(ChatRequest(question="Explain the graph topology", session_id="llm-fallback-ungrounded"))

        self.assertEqual(result.route, "llm_ungrounded_fallback")
        self.assertNotIn("42", result.answer)
        self.assertIn("couldn't turn them into a verified answer", result.answer)
        self.assertIn("failing_count", result.answer)

    @patch("server.api_server._retriever", None)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.KGChatSession")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.search_concepts", return_value=[])
    def test_llm_fallback_answer_accepted_when_grounded_in_evidence(
        self, concepts, session_cls, store_chat_turns, restore, store_decision, clear_pending
    ):
        final_answer = "There are 5 failing workflows on CMSSW_20_1_X_2026-08-17-2300."
        session_cls.side_effect = _mock_kgchat_session(
            final_answer,
            tool_results=[(
                "get_ib_summary", {},
                {"tag": "CMSSW_20_1_X_2026-08-17-2300", "failing_count": 5},
            )],
        )

        result = chat(ChatRequest(question="Explain the graph topology", session_id="llm-fallback-grounded"))

        self.assertEqual(result.route, "llm_validated")
        self.assertEqual(result.answer, final_answer)

    @patch("server.api_server._retriever", None)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.KGChatSession")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.search_concepts", return_value=[])
    # The question routes to the `relval` capability (the job name contains
    # "relvals") and dead-ends there, so it only reaches the LLM loop via the
    # capability-fallback rescue -- patch it explicitly rather than relying on
    # whatever the local .env happens to set (CI has no .env).
    @patch("server.api_server.CAPABILITY_FALLBACK_TO_LLM_ENABLED", True)
    def test_llm_fallback_has_shifter_docs_tool_available(
        self, concepts, session_cls, store_chat_turns, restore, store_decision, clear_pending
    ):
        # The agentic LLM-fallback loop is the only place a question with no
        # recognizable graph entity (no IB tag, no package -- a CI-topology
        # question like this one) can still reach an answer. Confirm
        # search_shifter_docs (static CMSSW/CMSSDT documentation, not a live
        # IB fact) is actually wired into its tool list, not just registered
        # in graph.tools.TOOLS and never reaching the live call site.
        captured = {}

        def factory(*, openai_client, model, tools_registry, system_prompt, on_tool_call=None):
            captured["tools_registry"] = tools_registry
            session = MagicMock()
            session.ask.return_value = "I don't have evidence for that."
            return session

        session_cls.side_effect = factory

        chat(ChatRequest(question="what does ib-run-gpu-relvals depend on", session_id="shifter-docs-tool-wired"))

        self.assertIn("search_shifter_docs", captured["tools_registry"])

    @patch("server.api_server._retriever", None)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.KGChatSession")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.search_concepts", return_value=[])
    # The question routes to the `relval` capability (the job name contains
    # "relvals") and dead-ends there, so it only reaches the LLM loop via the
    # capability-fallback rescue -- patch it explicitly rather than relying on
    # whatever the local .env happens to set (CI has no .env).
    @patch("server.api_server.CAPABILITY_FALLBACK_TO_LLM_ENABLED", True)
    def test_llm_fallback_rejects_cross_entity_shifter_doc_attribution(
        self, concepts, session_cls, store_chat_turns, restore, store_decision, clear_pending
    ):
        # End-to-end regression for the live-found misattribution bug: the
        # chat()-level route must actually reach _shifter_doc_claims_grounded
        # and fall back to raw evidence, not just the unit-level function.
        session_cls.side_effect = _mock_kgchat_session(
            "ib-run-gpu-relvals' upstream project is ib-validation, and its "
            "downstream project is update-das-queries.",
            tool_results=[(
                "search_shifter_docs", {"query": "ib-run-gpu-relvals"},
                [self._gpu_relvals_chunk(), self._unrelated_job_chunk()],
            )],
        )

        result = chat(ChatRequest(
            question="what does ib-run-gpu-relvals depend on", session_id="shifter-doc-misattribution",
        ))

        self.assertTrue(result.route.endswith("llm_ungrounded_fallback"))
        self.assertIn("more than one retrieved document", result.answer)
        self.assertIn("ib-validation", result.answer)  # raw evidence still shown, just not as asserted prose

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.KGChatSession", side_effect=TimeoutError("gateway too slow"))
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.search_concepts", return_value=[])
    def test_llm_investigation_degrades_to_friendly_message_on_timeout(
        self, concepts, session_cls, store_chat_turns, restore, store_decision, clear_pending
    ):
        # Found live via scripts/stress_test_bot.py, 2026-09-22: concurrent
        # LLM-heavy traffic pushed the gateway past 30-60s. A hung/failed
        # call here must never surface as an unhandled 500 -- it degrades to
        # a friendly, retryable message instead.
        result = chat(ChatRequest(question="Explain the graph topology", session_id="llm-investigation-timeout"))

        self.assertEqual(result.route, "llm_timeout")
        self.assertIn("try again", result.answer.lower())

    @patch("server.api_server._newest_complete_primary_tag", new=lambda *a, **k: (None, ""))  # no live DB
    @patch("server.api_server._retriever", None)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.KGChatSession")
    @patch("server.api_server._execute_planned_capability", return_value=None)
    @patch("server.api_server._latest_ib_records")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.CAPABILITY_FALLBACK_TO_LLM_ENABLED", True)
    @patch("server.api_server.search_concepts", return_value=[])
    def test_capability_dead_end_falls_through_to_llm_investigation_when_flag_enabled(
        self, concepts, latest_ib_records, execute_capability, session_cls,
        store_chat_turns, restore, store_decision, clear_pending
    ):
        # RequestPlanner correctly identifies "relval" here (mirrors
        # test_latest_ib_phrase_resolves_to_a_tag_before_routing), but this time
        # none of the "relval" capability's own helpers recognize the phrasing
        # (_execute_planned_capability -> None), simulating a real handler-chain
        # miss. With the fallback flag on, this must not dead-end -- it should
        # reach the same evidence-gated LLM investigation path used elsewhere,
        # never fabricate the answer itself.
        latest_ib_records.return_value = [{"tag": "CMSSW_20_1_X_2026-08-17-2300"}]
        final_answer = "There are 7 failing workflows on CMSSW_20_1_X_2026-08-17-2300."
        kgchat_factory = _mock_kgchat_session(
            final_answer,
            tool_results=[(
                "get_failing_relvals", {},
                {"tag": "CMSSW_20_1_X_2026-08-17-2300", "failing_count": 7},
            )],
        )
        session_cls.side_effect = kgchat_factory

        result = chat(ChatRequest(
            question="how many relvals failing in the latest IB", session_id="capability-dead-end-rescue",
        ))

        self.assertEqual(result.answer, final_answer)
        self.assertEqual(result.route, "capability_dead_end_rescue:relval:llm_validated")
        execute_capability.assert_called_once()
        kgchat_factory.sessions[0].ask.assert_called_once()

    @patch("server.api_server.CAPABILITY_FALLBACK_TO_LLM_ENABLED", False)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server._execute_planned_capability", return_value=None)
    @patch("server.api_server._latest_ib_records")
    @patch("server.api_server.KGChatSession")
    def test_capability_dead_end_keeps_flat_message_when_flag_disabled(
        self, session_cls, latest_ib_records, execute_capability,
        store_chat_turns, restore, store_decision, clear_pending
    ):
        # Rollback-path guard: with the flag at its default (off), the same
        # handler-chain miss as above must still return today's flat message
        # and must never invoke the LLM investigation path at all.
        latest_ib_records.return_value = [{"tag": "CMSSW_20_1_X_2026-08-17-2300"}]

        result = chat(ChatRequest(
            question="how many relvals failing in the latest IB", session_id="capability-dead-end-no-flag",
        ))

        self.assertIn(
            "No imported local evidence is available for the selected `relval` request", result.answer
        )
        self.assertEqual(result.route, "capability:relval:no_evidence")
        session_cls.assert_not_called()

    @patch("server.api_server._newest_complete_primary_tag", new=lambda *a, **k: (None, ""))  # no live DB
    @patch("server.api_server._retriever", None)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.KGChatSession")
    @patch("server.api_server.verify_answer_matches_question", return_value=False)
    @patch("server.api_server._execute_planned_capability", return_value=("No failing RelVals were found.", []))
    @patch("server.api_server._latest_ib_records")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.DETERMINISTIC_ANSWER_VERIFICATION_ENABLED", True)
    @patch("server.api_server.search_concepts", return_value=[])
    def test_deterministic_answer_semantic_mismatch_falls_through_to_llm_investigation_when_flag_enabled(
        self, concepts, latest_ib_records, execute_capability, verify_answer, session_cls,
        store_chat_turns, restore, store_decision, clear_pending
    ):
        # Real bug found live 2026-09-22: "workflows that are NOT failing" and
        # "workflows that are failing" both trigger the same "failing" keyword
        # in the deterministic relval handler and get the identical answer,
        # which is wrong for the negated question. _execute_planned_capability
        # here simulates that exact mismatch (a truthy, but wrong, answer) --
        # with the verification flag on, this must not be trusted as-is; it
        # must reroute through the same evidence-gated LLM investigation path
        # used for genuine dead ends, never fabricate a "corrected" answer itself.
        latest_ib_records.return_value = [{"tag": "CMSSW_20_1_X_2026-08-17-2300"}]
        final_answer = "592 workflows are not failing on CMSSW_20_1_X_2026-08-17-2300."
        kgchat_factory = _mock_kgchat_session(
            final_answer,
            tool_results=[(
                "get_relval_results", {},
                {"tag": "CMSSW_20_1_X_2026-08-17-2300", "passing_count": 592},
            )],
        )
        session_cls.side_effect = kgchat_factory

        result = chat(ChatRequest(
            question="how many relvals are NOT failing in the latest IB", session_id="semantic-mismatch-rescue",
        ))

        self.assertEqual(result.answer, final_answer)
        self.assertEqual(result.route, "capability_semantic_mismatch_rescue:relval:llm_validated")
        execute_capability.assert_called_once()
        verify_answer.assert_called_once()
        kgchat_factory.sessions[0].ask.assert_called_once()

    @patch("server.api_server._retriever", None)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.KGChatSession")
    @patch("server.api_server.verify_answer_matches_question", return_value=False)
    @patch(
        "server.api_server._execute_planned_capability",
        return_value=("`CMSSW_20_1_EVTGEN3_X_2026-08-09-2300` has **13 failed RelVal(s)**.", []),
    )
    @patch("server.api_server._latest_ib_records")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.DETERMINISTIC_ANSWER_VERIFICATION_ENABLED", True)
    @patch("server.api_server.search_concepts", return_value=[])
    def test_semantic_mismatch_rescue_never_replaces_answer_with_llm_absence_claim(
        self, concepts, latest_ib_records, execute_capability, verify_answer, session_cls,
        store_chat_turns, restore, store_decision, clear_pending
    ):
        # Found live, 2026-10-06 in a replay of real questions: the verifier wrongly flagged a
        # correct "13 failed RelVals" answer, and the rescue's LLM -- having
        # called a tool with the wrong arguments -- replaced it with "No failing
        # RelVal workflows were recorded". An unverified absence claim must
        # never override a real deterministic answer.
        latest_ib_records.return_value = [{"tag": "CMSSW_20_1_X_2026-08-17-2300"}]
        session_cls.side_effect = _mock_kgchat_session(
            "No failing RelVal workflows were recorded for the build. The query returned an empty list.",
            tool_results=[("get_failing_relvals", {"ib_tag": "CMSSW_20_1_EVTGEN3_X_2026-08-09-2300"}, [])],
        )

        result = chat(ChatRequest(
            question="how many relvals errors in CMSSW_20_1_EVTGEN3_X_2026-08-09-2300, not counting warnings",
            session_id="rescue-absence-claim",
        ))

        self.assertIn("13 failed RelVal", result.answer)
        self.assertEqual(result.route, "capability:relval:verifier_rescue_unusable")

    @patch("server.api_server._retriever", None)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.KGChatSession")
    @patch("server.api_server.verify_answer_matches_question", return_value=False)
    @patch(
        "server.api_server._execute_planned_capability",
        return_value=("IB comparison: 2 newly failing workflows.", [{"label": "x", "question": "y"}]),
    )
    @patch("server.api_server._latest_ib_records")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.DETERMINISTIC_ANSWER_VERIFICATION_ENABLED", True)
    @patch("server.api_server.search_concepts", return_value=[])
    def test_semantic_mismatch_rescue_keeps_deterministic_answer_over_raw_evidence_dump(
        self, concepts, latest_ib_records, execute_capability, verify_answer, session_cls,
        store_chat_turns, restore, store_decision, clear_pending
    ):
        latest_ib_records.return_value = [{"tag": "CMSSW_20_1_X_2026-08-17-2300"}]
        session_cls.side_effect = _mock_kgchat_session(
            "There are 42 newly failing workflows.",  # 42 not in evidence -> ungrounded
            tool_results=[("get_failing_relvals", {}, [{"workflow_id": "1.0"}])],
        )

        result = chat(ChatRequest(
            question="compare CMSSW_20_1_X_2026-10-05-1100 and CMSSW_20_1_X_2026-10-05-2300 but only el9_amd64_gcc14",
            session_id="rescue-raw-dump",
        ))

        self.assertEqual(result.answer, "IB comparison: 2 newly failing workflows.")
        self.assertEqual(result.route, "capability:comparison:verifier_rescue_unusable")
        self.assertEqual([(a.label, a.question) for a in result.actions], [("x", "y")])

    @patch("server.api_server._retriever", None)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.KGChatSession")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.search_concepts", return_value=[])
    def test_llm_fallback_absence_claim_is_not_presented_as_fact(
        self, concepts, session_cls, store_chat_turns, restore, store_decision, clear_pending
    ):
        # Found live: "and on aarch64 only" -> the LLM called list_ib_tags with
        # arch="aarch64" (not a real arch name), got [], and answered "no IBs
        # with that architecture have been imported" -- false, there are
        # hundreds. An LLM's "no such data" is shown as an empty lookup, not a fact.
        session_cls.side_effect = _mock_kgchat_session(
            "I couldn't find any Integration Build records for the aarch64 architecture in the current graph.",
            tool_results=[("list_ib_tags", {"arch": "aarch64"}, [])],
        )

        result = chat(ChatRequest(question="Explain the graph topology", session_id="llm-absence-claim"))

        self.assertEqual(result.route, "llm_unverified_absence")
        self.assertNotIn("couldn't find any Integration Build", result.answer)
        self.assertIn("doesn't prove the data isn't there", result.answer)

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.verify_answer_matches_question", return_value=False)
    @patch("server.api_server._execute_planned_capability", return_value=("13 failed RelVal(s).", []))
    @patch("server.api_server._latest_ib_records")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.DETERMINISTIC_ANSWER_VERIFICATION_ENABLED", True)
    @patch("server.api_server.search_concepts", return_value=[])
    def test_semantic_verifier_skipped_without_scope_qualifier(
        self, concepts, latest_ib_records, execute_capability, verify_answer,
        store_chat_turns, restore, store_decision, clear_pending
    ):
        # 60 of 63 live verifier flags were false alarms on plain questions with
        # no negation/exclusion/scope word -- the verifier must not even run there.
        latest_ib_records.return_value = [{"tag": "CMSSW_20_1_X_2026-08-17-2300"}]

        result = chat(ChatRequest(
            question="how many relvals errors in CMSSW_20_1_EVTGEN3_X_2026-08-09-2300", session_id="no-qualifier",
        ))

        verify_answer.assert_not_called()
        self.assertEqual(result.route, "capability:relval")
        self.assertEqual(result.answer, "13 failed RelVal(s).")

    def test_scope_qualifier_detector(self):
        for q in ("workflows NOT failing", "which archs have no build warnings", "everything except ASAN",
                  "only aarch64", "relvals that didn't fail"):
            self.assertTrue(_question_has_scope_qualifier(q), q)
        for q in ("how many relvals errors in CMSSW_20_1_X_2026-08-09-2300", "compare the last two IBs",
                  "what is the latest IB", "is workflow 140.051 a known issue"):
            self.assertFalse(_question_has_scope_qualifier(q), q)

    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.KGChatSession")
    def test_conversational_replies_never_reach_routing_or_llm(self, session_cls, store_chat_turns):
        # Found live 2026-10-06: "hi", "thanks!", and "please rerun workflow
        # 512.0" all got the generic "I couldn't match that" menu.
        for question, route, phrase in (
            ("hi", "conversational:greeting", "What would you like to check"),
            ("Heelo!", "conversational:greeting", "What would you like to check"),
            ("thanks!", "conversational:thanks", "welcome"),
            ("please rerun workflow 512.0 for me", "conversational:read_only", "read-only"),
            ("delete all relval data for the latest IB", "conversational:read_only", "read-only"),
        ):
            result = chat(ChatRequest(question=question, session_id=f"conv-{route}"))
            self.assertEqual(result.route, route, question)
            self.assertIn(phrase, result.answer)
        session_cls.assert_not_called()

    def test_conversational_patterns_leave_real_questions_alone(self):
        from server.api_server import _GREETING_RE, _THANKS_RE
        self.assertIsNone(_GREETING_RE.fullmatch("hi, what is the latest IB"))
        self.assertIsNone(_THANKS_RE.fullmatch("ok"))  # may be a reply to a clarification
        self.assertIsNone(_conversational_response("s", "which relvals were rerun in the latest IB"))
        self.assertIsNone(_conversational_response("s", "merged PRs in the latest IB"))

    @patch("server.api_server.get_driver")
    def test_relval_known_error_lines_names_workflows_grouped_by_reason(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"workflow_id": "512.0", "message": "Architecture incompetibility. GridPacks were built for x86_64"},
            {"workflow_id": "513.0", "message": "Architecture incompetibility. GridPacks were built for x86_64"},
            {"workflow_id": "10805.31", "message": "Triton Server Instance for non-x86_64"},
        ]
        record = {"tag": "CMSSW_20_1_X_2026-10-05-2300", "arch": "el9_aarch64_gcc14", "known_error_relvals": 3}

        lines = _relval_known_error_lines([("el9_aarch64_gcc14", record), ("x", {"known_error_relvals": 0})])

        text = "\n".join(lines)
        self.assertIn("excluded from the failing count", text)
        self.assertIn("2 known error(s) -- \"Architecture incompetibility. GridPacks were built for x86_64\": `512.0`, `513.0`", text)
        self.assertIn("`10805.31`", text)
        session.run.assert_called_once()  # records without known errors are not queried

    def test_relval_known_error_lines_skips_query_when_none(self):
        self.assertEqual(_relval_known_error_lines([("a", {"known_error_relvals": 0})]), [])

    def test_workflow_follow_up_resolves_to_last_mentioned_workflow(self):
        # Found live 2026-10-06: after "tell me about workflow 512.0", "since
        # when is it failing" and "on which architectures" lost the workflow.
        from server.api_server import _last_workflow_contexts, _with_workflow_follow_up_context
        sid = "wf-follow-up"
        try:
            self.assertEqual(_with_workflow_follow_up_context("tell me about workflow 512.0", sid),
                             "tell me about workflow 512.0")
            self.assertEqual(_with_workflow_follow_up_context("since when is it failing", sid),
                             "since when is it failing (workflow 512.0)")
            self.assertEqual(_with_workflow_follow_up_context("on which architectures", sid),
                             "on which architectures (workflow 512.0)")
            # Never hijack a question that names its own entity or another category.
            for q in ("since when is CMSSW_20_1_X_2026-10-05-2300 failing", "is it in PR #52028",
                      "which unit tests are failing", "what is the latest IB"):
                self.assertEqual(_with_workflow_follow_up_context(q, sid), q)
        finally:
            _last_workflow_contexts.pop(sid, None)

    @patch("server.api_server.get_relval_recurrence",
           return_value={"total_appearances": 313, "failed_appearances": 41, "known_error_appearances": 33})
    @patch("server.api_server.get_exit_code_names", return_value={16640: "OtherCMS"})
    @patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-10-05-2300"}])
    @patch("server.api_server.get_driver")
    def test_workflow_why_answer_defaults_to_latest_ib_and_explains_known_error(
        self, get_driver, latest, names, recurrence
    ):
        # Found live: "why is workflow 512.0 failing on aarch64" answered
        # "Please provide the Integration Build tag".
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.side_effect = [
            [{"tag": "CMSSW_20_1_X_2026-10-05-2300", "arch": "el9_aarch64_gcc14", "variant": "primary",
              "name": "DYTollJets_LO_Mad_13TeV_py8", "status": "FAILED", "exitcode": 16640, "errors": 0,
              "known_error": True, "known_error_message": "GridPacks were built for x86_64"}],
            [{"tag": "CMSSW_20_1_X_2026-10-05-2300", "arch": "el9_aarch64_gcc14", "step": "step1",
              "category": "ExternalLHEProducer", "text": "Child failed with exit code 2.", "source_url": "https://x"}],
        ]

        answer = _workflow_why_failing_answer("why is workflow 512.0 failing on aarch64", "why-latest")

        self.assertIn("the latest IB", answer)
        self.assertIn("CMSSW_20_1_X_2026-10-05-2300", answer)
        self.assertIn("16640 (OtherCMS)", answer)
        self.assertIn("known error", answer)
        self.assertIn("GridPacks were built for x86_64", answer)
        self.assertIn("ExternalLHEProducer", answer)
        self.assertIn("41 of 313", answer)
        self.assertIn("CONTAINS $family", session.run.call_args_list[0].args[0])
        self.assertEqual(session.run.call_args_list[0].kwargs["family"], "aarch64")

    def test_workflow_why_answer_leaves_pr_and_unrelated_questions_alone(self):
        self.assertIsNone(_workflow_why_failing_answer("why is workflow 512.0 failing, any open issue?", "s"))
        self.assertIsNone(_workflow_why_failing_answer("why is the build red", "s"))

    @patch("server.api_server.get_exit_code_names", return_value={62720: "SIGSEGV"})
    @patch("server.api_server.get_driver")
    def test_failure_overview_lists_every_category_and_separates_known_errors(self, get_driver, names):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"tag": "T", "arch": "el9_aarch64_gcc14", "variant": "primary", "build_status": "passed",
             "failed_relvals": [], "known_errors": 50, "failed_unittests": ["SimG4Core/Geometry"]},
            {"tag": "T_ASAN", "arch": "el9_amd64_gcc14", "variant": "ASAN", "build_status": "warning",
             "failed_relvals": [{"wf": "136.817", "name": "RunDoubleEG2017E", "exitcode": 62720}],
             "known_errors": 0, "failed_unittests": []},
        ]

        answer, actions = _failure_overview_answer("T", None)

        self.assertIn("no build errors", answer)
        self.assertIn("Built with compilation warnings: `ASAN` `el9_amd64_gcc14`", answer)
        self.assertIn("`136.817` RunDoubleEG2017E (exit 62720 (SIGSEGV))", answer)
        self.assertIn("50 known-error RelVal(s), excluded", answer)
        self.assertIn("`SimG4Core/Geometry`", answer)
        self.assertNotIn("None", answer)
        self.assertEqual({a["label"] for a in actions} >= {"RelVal details", "Unit-test details"}, True)

    def test_recurrence_intent_covers_failed_before(self):
        from server.api_server import _RECURRING_FAILURE_INTENT_RE
        for q in ("has workflow 140.051 failed before", "did it fail previously", "was it failing in the past"):
            self.assertTrue(_RECURRING_FAILURE_INTENT_RE.search(q), q)

    @patch("server.api_server.get_driver")
    def test_arch_family_follow_up_resolves_only_when_unambiguous(self, get_driver):
        from server.api_server import _with_arch_family_context
        session = get_driver.return_value.session.return_value.__enter__.return_value
        sid = "arch-family"
        _latest_ib_contexts[sid] = "CMSSW_20_1_X_2026-10-05-2300"
        try:
            session.run.return_value = [{"arch": "el9_aarch64_gcc14"}]
            self.assertEqual(_with_arch_family_context("and on aarch64 only", sid),
                             "and on aarch64 only el9_aarch64_gcc14")
            session.run.return_value = [{"arch": "el8_aarch64_gcc13"}, {"arch": "el9_aarch64_gcc14"}]
            self.assertEqual(_with_arch_family_context("and on arm", sid), "and on arm")  # never guess
            self.assertEqual(_with_arch_family_context("on el9_amd64_gcc14", sid), "on el9_amd64_gcc14")
        finally:
            _latest_ib_contexts.pop(sid, None)

    def test_scope_only_follow_up_repeats_previous_capability(self):
        from server.api_server import _last_capability_contexts, _with_previous_capability_context
        sid = "prev-capability"
        _last_capability_contexts[sid] = "unittest"
        try:
            self.assertEqual(_with_previous_capability_context("and on aarch64 only el9_aarch64_gcc14", sid),
                             "unit tests and on aarch64 only el9_aarch64_gcc14")
            self.assertEqual(_with_previous_capability_context("all architectures", sid), "unit tests all architectures")
            for q in ("relvals on el9_amd64_gcc14", "what is the latest IB", "hello el9_amd64_gcc14 is great and I want to know more about it"):
                self.assertEqual(_with_previous_capability_context(q, sid), q)
        finally:
            _last_capability_contexts.pop(sid, None)

    @patch("server.api_server._latest_ib_records", return_value=[{"release": "CMSSW_20_1_X", "tag": "CMSSW_20_1_X_2026-10-05-2300"}])
    @patch("server.api_server.get_driver")
    def test_ib_slot_availability_resolves_yesterday_in_cern_time(self, get_driver, latest):
        # Found live: the LLM decided "yesterday" was two days ago.
        from server.api_server import _ib_slot_availability_answer
        from zoneinfo import ZoneInfo
        session = get_driver.return_value.session.return_value.__enter__.return_value
        yesterday = (datetime.now(ZoneInfo("Europe/Zurich")) - timedelta(days=1)).strftime("%Y-%m-%d")
        session.run.side_effect = [
            [{"tag": f"CMSSW_20_1_X_{yesterday}-2300", "arch": "el9_amd64_gcc14", "parent_tag": f"CMSSW_20_1_X_{yesterday}-2300", "variant": None}],
            [{"tag": f"CMSSW_20_1_X_{yesterday}-2300"}],
        ]

        answer = _ib_slot_availability_answer("is the 23:00 IB of yesterday available?")

        self.assertIn(f"{yesterday} 23:00", answer)
        self.assertIn(f"CMSSW_20_1_X_{yesterday}-2300", answer)
        self.assertEqual(session.run.call_args_list[0].kwargs["stamps"], [f"_{yesterday}-2300"])
        self.assertIsNone(_ib_slot_availability_answer("what is the latest IB"))
        self.assertIsNone(_ib_slot_availability_answer("did the latest IB build successfully?"))

    def test_planner_ignores_category_words_inside_jenkins_job_names(self):
        from server.request_planner import RequestContext
        self.assertEqual(RequestContext.extract("what does ib-run-gpu-relvals depend on").categories, frozenset())
        self.assertEqual(RequestContext.extract("what does the build-any-ib jenkins job do").categories, frozenset())
        self.assertNotIn("relval", RequestContext.extract("how does the PR approval workflow work").categories)
        self.assertIn("relval", RequestContext.extract("failing relvals in the latest IB").categories)
        self.assertIn("relval", RequestContext.extract("which workflows failed").categories)
        self.assertIn("pr_evidence", RequestContext.extract("any PRs for this").categories)

    def test_explicit_pr_question_outranks_the_category_it_mentions(self):
        from server.api_server import _request_planner
        from server.request_planner import RequestContext
        plan = _request_planner.plan(RequestContext.extract("any PR fixing the MPICore unit test failure?"))
        self.assertEqual(plan.capability, "pr_evidence")

    def test_latest_typos_are_normalized(self):
        self.assertIn("latest", _normalize_deterministic_question("lates ib relval failurs"))
        self.assertIn("failures", _normalize_deterministic_question("lates ib relval failurs"))
        self.assertIn("unit tests", _normalize_deterministic_question("UT failures"))
        self.assertEqual(_normalize_deterministic_question("but ut is fine"), "but ut is fine")

    def test_step_numbering_does_not_count_as_an_ungrounded_number(self):
        # Found live: every multi-step shifter-docs answer was rejected and
        # dumped as raw JSON solely because of its own "1.", "2." numbering.
        from server.api_server import _evidence_grounded
        evidence = [{"tool": "search_shifter_docs", "result": [{"text": "Open a PR, then wait for 5 approvals."}]}]
        self.assertTrue(_evidence_grounded("1. Open a PR\n2. Wait for 5 approvals\n| **3. Merge** |\nStep 4: done", evidence))
        self.assertFalse(_evidence_grounded("1. Open a PR\n2. Wait for 7 approvals", evidence))

    @patch("server.api_server.search_issues_by_text",
           return_value={"issues": [{"number": 49574, "title": "Multiple RelVals failed with cudaErrorLaunchOutOfResources", "state": "closed"}]})
    @patch("server.api_server.find_prs_for_error", return_value=[])
    def test_pasted_error_in_failure_overview_searches_evidence_instead(self, prs, issues):
        from server.api_server import _looks_like_pasted_error_text
        question = ("A previous API call (not this one) set the error : 'cudaErrorLaunchOutOfResources': "
                    "'too many resources requested for launch'! CMSSW_20_1_X_2026-10-05-2300")
        self.assertTrue(_looks_like_pasted_error_text(question))
        self.assertFalse(_looks_like_pasted_error_text("what is failing in the latest IB CMSSW_20_1_X_2026-10-05-2300"))

        answer, _ = _execute_planned_capability("failure_overview", question, "pasted-error")

        prs.assert_called_once_with("cudaErrorLaunchOutOfResources", limit=8)
        self.assertIn("Issue #49574", answer)
        self.assertIn("do not prove a fix or causal relationship", answer)

    @patch("server.api_server._store_chat_turns")
    def test_help_request_lists_what_can_be_asked(self, store_chat_turns):
        result = chat(ChatRequest(question="what can you help me with?", session_id="help"))
        self.assertEqual(result.route, "conversational:help")
        self.assertIn("which PR could have caused it?", result.answer)
        self.assertIn("what code did that PR touch?", result.answer)

    def test_recurrence_intent_covers_recurrence_pattern(self):
        from server.api_server import _RECURRING_FAILURE_INTENT_RE
        self.assertTrue(_RECURRING_FAILURE_INTENT_RE.search("tell me the recurrence pattern for workflow 136.899"))

    def test_ib_variant_glossary_answers_definitions_but_not_results(self):
        from server.api_server import _ib_variant_glossary_answer
        answer = _ib_variant_glossary_answer("what is a sub-IB and what does ASAN mean")
        self.assertIn("sub-IB", answer)
        self.assertIn("AddressSanitizer", answer)
        self.assertIn("cms-bot `config.map`", answer)
        self.assertNotIn("UBSAN", answer)  # only the named variant
        self.assertIsNone(_ib_variant_glossary_answer("ASAN failures in the latest IB"))
        self.assertIsNone(_ib_variant_glossary_answer("what is the latest IB"))

    @patch("server.api_server.variant_previous_tag", return_value="CMSSW_20_1_X_2026-10-05-1100")
    @patch("server.api_server.get_driver")
    def test_persistent_failures_splits_still_new_and_resolved(self, get_driver, previous):
        from server.api_server import _persistent_failures_answer
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.side_effect = [
            [{"a": "el9_amd64_gcc14"}],
            [{"n": "HeterogeneousCore/MPICore"}, {"n": "HLTrigger/Tools"}],   # now
            [{"n": "HeterogeneousCore/MPICore"}, {"n": "IOPool/Tests"}],      # before
        ]
        sid = "persistent"
        _latest_ib_contexts[sid] = "CMSSW_20_1_X_2026-10-05-2300"
        try:
            answer = _persistent_failures_answer("which unit tests also failed in the previous IB", sid)
        finally:
            _latest_ib_contexts.pop(sid, None)

        self.assertIn("**still failing** (also failed before): `HeterogeneousCore/MPICore`", answer)
        self.assertIn("**newly failing**: `HLTrigger/Tools`", answer)
        self.assertIn("**resolved** since then: `IOPool/Tests`", answer)
        self.assertIsNone(_persistent_failures_answer("which unit tests are failing", sid))

    def test_held_out_phrasings_reach_the_right_place(self):
        # Found live 2026-10-06 on a held-out question set (never tuned on).
        from server.api_server import _request_planner, _with_workflow_follow_up_context, _last_workflow_contexts
        from server.request_planner import RequestContext
        tag = " CMSSW_20_1_X_2026-10-05-2300"
        for q in ("anything red on aarch64 in the last build?", "how bad is the latest IB",
                  "give me the status of today's IBs", "is the latest build broken"):
            plan = _request_planner.plan(RequestContext.extract(_normalize_deterministic_question(q) + tag))
            self.assertEqual(plan.capability, "failure_overview", q)
        self.assertEqual(_normalize_deterministic_question("show the latest build warnings"), "show the latest build warnings")
        self.assertTrue(_is_latest_comparison_question("compare today and yesterday IBs"))
        sid = "wf-issue-follow-up"
        _last_workflow_contexts[sid] = "136.817"
        try:
            self.assertEqual(_with_workflow_follow_up_context("is there an issue for it?", sid),
                             "is there an issue for it? (workflow 136.817)")
        finally:
            _last_workflow_contexts.pop(sid, None)

    @patch("server.api_server.get_driver")
    def test_issue_author_accepts_issue_number_without_hash(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {"author": "iarspider", "title": "t", "state": "closed"}
        answer = _issue_author_answer("who opened issue 49574")
        self.assertIn("49574", answer)
        self.assertEqual(_resolve_issue_number("do you have issue 52041?", None), None)  # still PR-or-Issue ambiguous

    def test_llm_prompt_is_told_the_current_cern_date(self):
        from server.api_server import _current_date_note
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y-%m-%d")
        self.assertIn(today, _current_date_note())
        self.assertIn("yesterday", _current_date_note())

    @patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-10-05-2300"}])
    @patch("server.api_server.get_driver")
    def test_known_error_concept_is_explained_from_the_data(self, get_driver, latest):
        from server.api_server import _known_error_concept_answer
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"message": "GridPacks were built for x86_64", "n": 49, "archs": ["el9_aarch64_gcc14"]}]
        answer = _known_error_concept_answer("what is a known error")
        self.assertIn("excluded from failing counts", answer)
        self.assertIn("49 workflow(s): \"GridPacks were built for x86_64\"", answer)
        self.assertIsNone(_known_error_concept_answer("which known errors are in the latest IB"))

    def test_bare_why_is_not_an_ib_relationship_question(self):
        # Found live (held-out set 2): "why is HeterogeneousCore/MPICore failing"
        # was answered "<tag> is a primary IB, not a sub-IB variant".
        from server.request_planner import RequestContext
        self.assertFalse(RequestContext.extract("why is HeterogeneousCore/MPICore failing").has_relationship_request)
        self.assertTrue(RequestContext.extract("why did you show CMSSW_20_1_TF_X_2026-08-14-1100 as a sub IB?").has_relationship_request)
        self.assertTrue(RequestContext.extract("is everything green on el8_amd64_gcc14?").has_failure_request)

    @patch("server.api_server.search_issues_by_text",
           return_value={"issues": [{"number": 50547, "title": "testMPIAutosplitter appears to fail occasionally", "state": "open"}]})
    @patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-10-05-2300"}])
    @patch("server.api_server.get_driver")
    def test_package_status_answer_shows_every_record_history_and_open_issues(self, get_driver, latest, issues):
        from server.api_server import _package_status_answer
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.side_effect = [
            [{"variant": "primary", "arch": "el9_amd64_gcc14", "status": "FAILED", "errors": 1},
             {"variant": "ASAN", "arch": "el9_amd64_gcc14", "status": "PASSED", "errors": 0}],
            MagicMock(single=MagicMock(return_value={"total": 323, "failed": 62, "first_failed": "2026-09-29-2300"})),
        ]
        answer = _package_status_answer("why is HeterogeneousCore/MPICore failing in CMSSW_20_1_X_2026-10-05-2300", "pkg-status")
        self.assertIn("failing in 1 of 2", answer)
        self.assertIn("62 of 323", answer)
        self.assertIn("Issue #50547", answer)
        self.assertIn("relevance only", answer)
        self.assertIsNone(_package_status_answer("any PR for HeterogeneousCore/MPICore", "pkg-status"))

    @patch("server.api_server.get_driver")
    def test_package_status_without_tag_uses_each_tracks_newest_build(self, get_driver):
        # found 2026-10-16: the newest IB had unit tests for one architecture only, so
        # "is MPICore still failing?" said "failing in 0 of 2"
        from server.api_server import _package_status_answer
        session = get_driver.return_value.session.return_value.__enter__.return_value
        row = lambda tag, arch, status: {"tag": tag, "variant": "primary", "arch": arch, "status": status, "errors": 1}
        session.run.return_value = [
            row("CMSSW_20_1_X_2026-10-15-0900", "el9_amd64_gcc14", "PASSED"),
            row("CMSSW_20_1_X_2026-10-14-2300", "el9_aarch64_gcc14", "FAILED"),
            row("CMSSW_20_1_X_2026-10-14-1100", "el9_aarch64_gcc14", "FAILED"),
            row("CMSSW_20_1_X_2026-10-13-2300", "el9_aarch64_gcc14", "PASSED"),
            row("CMSSW_20_1_X_2026-10-12-2300", "el9_amd64_gcc14", "FAILED"),
            row("CMSSW_20_0_X_2026-10-14-2300", "el8_amd64_gcc13", "FAILED"),
        ]
        answer = _package_status_answer("is SimG4Core/Geometry still failing?", "pkg-newest")
        self.assertIn("Failing now in 1 of 2", answer)
        self.assertIn("since `CMSSW_20_1_X_2026-10-14-1100`", answer)
        self.assertIn("last failed in `CMSSW_20_1_X_2026-10-12-2300`", answer)
        self.assertNotIn("CMSSW_20_0", answer)  # newest release series only
        only = _package_status_answer("SimG4Core/Geometry failures on aarch64", "pkg-newest")
        self.assertIn("Failing now in 1 of 1", only)

    @patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-10-05-2300"}])
    @patch("server.api_server.get_driver")
    def test_named_sub_ib_flavour_resolves_to_its_newest_tag(self, get_driver, latest):
        from server.api_server import _with_variant_ib_context
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {"tag": "CMSSW_20_1_UBSAN_X_2026-10-05-2300"}
        self.assertEqual(_with_variant_ib_context("which unit tests broke in the UBSAN IB"),
                         "which unit tests broke in the UBSAN IB CMSSW_20_1_UBSAN_X_2026-10-05-2300")
        self.assertEqual(session.run.call_args.kwargs["prefix"], "CMSSW_20_1_UBSAN_X_")
        self.assertEqual(_with_variant_ib_context("what does the ASAN IB mean"), "what does the ASAN IB mean")
        self.assertEqual(_with_variant_ib_context("ASAN"), "ASAN")

    def test_docs_answer_must_mostly_use_the_retrieved_wording(self):
        # Found live: an invented six-step PR approval process ("GitHub Actions",
        # "squash-merge") passed every check once step numbers stopped tripping
        # the number guard -- nothing checked the content itself.
        from server.api_server import _doc_answer_terms_grounded
        doc = ("This page is a short summary of the process we use for accepting code changes from CMS "
               "developers into CMSSW. Go to the pull request page to have an overview of the status of all "
               "the pull requests. Click on a given pull request to have the whole discussion about the pull "
               "request approval process. Signatures from the L2 conveners of every touched category are required.")
        evidence = [{"tool": "search_shifter_docs", "result": [{"text": doc}]}]
        genuine = ("CMSSW accepts code changes from CMS developers through pull requests. The pull request page gives "
                   "an overview of the status of all pull requests, and each pull request shows the whole discussion "
                   "about the approval process, including signatures from the L2 conveners of every touched category.")
        invented = ("GitHub Actions run automated continuous integration pipelines, maintainers perform squash merges, "
                    "release managers publish nightly tarballs, reviewers attach labels like needs-review, and "
                    "quality gates enforce mandatory coverage thresholds before deployment to production servers.")
        self.assertTrue(_doc_answer_terms_grounded(genuine, evidence))
        self.assertFalse(_doc_answer_terms_grounded(invented, evidence))
        self.assertTrue(_doc_answer_terms_grounded(invented, [{"tool": "get_failing_relvals", "result": []}]))

    def test_llm_absence_claim_detector(self):
        for text in (
            "No failing RelVal workflows were recorded for the build.",
            "The graph does not contain `CMSSW_20_1_X_2026-09-07-1100`.",
            "The query returned an empty list, indicating that all unit tests passed.",
            "I couldn't find any IB records for the requested tag.",
            "No unit tests failed in the build CMSSW_20_1_X_2026-09-20-2300.",
        ):
            self.assertTrue(_llm_answer_is_absence_claim(text), text)
        for text in (
            "There are 5 failing workflows on CMSSW_20_1_X_2026-08-17-2300.",
            "The `build-any-ib` job is the entry point for creating an Integration Build.",
            "Exit code 16640 means OtherCMS.",
        ):
            self.assertFalse(_llm_answer_is_absence_claim(text), text)

    def test_verification_session_context_is_none_without_an_established_tag(self):
        self.assertIsNone(_verification_session_context("what about unit tests", "no-context-session"))

    def test_verification_session_context_is_none_when_tag_already_in_question(self):
        tag = "CMSSW_20_1_X_2026-08-17-2300"
        session_id = "tag-repeated-session"
        _latest_ib_contexts[session_id] = tag
        try:
            self.assertIsNone(_verification_session_context(f"unit tests for {tag}", session_id))
        finally:
            _latest_ib_contexts.pop(session_id, None)

    def test_verification_session_context_describes_carried_tag_and_arch(self):
        # Real false positive found live 2026-09-22: without this note, the
        # verifier has no memory of the conversation and flags a legitimate
        # tag-carried-from-earlier-turn answer ("what about unit tests" ->
        # an answer naming the tag) as a mismatch.
        tag = "CMSSW_20_1_X_2026-08-17-2300"
        arch = "el9_amd64_gcc14"
        session_id = "tag-carried-session"
        _latest_ib_contexts[session_id] = tag
        _latest_ib_arch_contexts[session_id] = arch
        try:
            note = _verification_session_context("what about unit tests", session_id)
        finally:
            _latest_ib_contexts.pop(session_id, None)
            _latest_ib_arch_contexts.pop(session_id, None)

        self.assertIn(tag, note)
        self.assertIn(arch, note)

    @patch("server.api_server._retriever", None)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.verify_answer_matches_question", return_value=True)
    @patch(
        "server.api_server._execute_planned_capability",
        return_value=("Unit-test results for CMSSW_20_1_X_2026-08-17-2300 are architecture-specific.", []),
    )
    @patch("server.api_server._latest_ib_records")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.DETERMINISTIC_ANSWER_VERIFICATION_ENABLED", True)
    def test_deterministic_verification_passes_session_tag_context_for_tagless_followup(
        self, latest_ib_records, execute_capability, verify_answer, store_chat_turns, restore, store_decision,
        clear_pending
    ):
        # Confirms the fix for the false positive above end to end: a real
        # tag-carried-from-earlier-turn follow-up must not be rerouted just
        # because the verifier lacks conversation memory -- the session tag
        # is threaded into the verifier call so it can judge fairly.
        tag = "CMSSW_20_1_X_2026-08-17-2300"
        session_id = "verification-context-followup"
        _latest_ib_contexts[session_id] = tag
        try:
            result = chat(ChatRequest(question="what about unit tests, only the failing ones", session_id=session_id))
        finally:
            _latest_ib_contexts.pop(session_id, None)

        self.assertEqual(result.route, "capability:unittest")
        verify_answer.assert_called_once()
        self.assertIn(tag, verify_answer.call_args.kwargs["session_context"])

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server._execute_planned_capability", return_value=("relval evidence", []))
    @patch("server.api_server._ROUTER_SHADOW_EXECUTOR")
    def test_router_shadow_logging_is_dispatched_off_the_response_path(
        self, executor, execute_capability, store_chat_turns, restore, store_decision, clear_pending
    ):
        # Found live via scripts/stress_test_bot.py, 2026-09-22: this call
        # used to run inline and block chat() before the real answer logic
        # even started, stacking with the answer-verification/investigation
        # calls to inflate total response time under gateway contention. It
        # never affects what's shown to the user, so it must be submitted to
        # the background executor, never called (and waited on) inline.
        result = chat(ChatRequest(
            question="how many relvals are failing in CMSSW_20_1_X_2026-08-17-2300",
            session_id="shadow-non-blocking",
        ))

        self.assertEqual(result.answer, "relval evidence")
        executor.submit.assert_called_once()
        self.assertIs(executor.submit.call_args.args[0], _log_router_shadow_decision)

    @patch("server.api_server._newest_complete_primary_tag", new=lambda *a, **k: (None, ""))  # no live DB
    @patch("server.api_server.DETERMINISTIC_ANSWER_VERIFICATION_ENABLED", False)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.verify_answer_matches_question")
    @patch("server.api_server._execute_planned_capability", return_value=("No failing RelVals were found.", []))
    @patch("server.api_server._latest_ib_records")
    def test_deterministic_answer_skips_verification_when_flag_disabled(
        self, latest_ib_records, execute_capability, verify_answer,
        store_chat_turns, restore, store_decision, clear_pending
    ):
        # Rollback-path guard: with the flag at its default (off), the
        # verifier must never even be called -- the deterministic answer is
        # returned exactly as today, unchanged.
        latest_ib_records.return_value = [{"tag": "CMSSW_20_1_X_2026-08-17-2300"}]

        result = chat(ChatRequest(
            question="how many relvals failing in the latest IB", session_id="semantic-mismatch-flag-off",
        ))

        self.assertEqual(result.answer, "No failing RelVals were found.")
        self.assertEqual(result.route, "capability:relval")
        verify_answer.assert_not_called()

    @patch("server.api_server._newest_complete_primary_tag", new=lambda *a, **k: (None, ""))  # no live DB
    @patch("server.api_server.DETERMINISTIC_ANSWER_VERIFICATION_ENABLED", True)
    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server.verify_answer_matches_question", side_effect=RuntimeError("boom"))
    @patch("server.api_server._execute_planned_capability", return_value=("No failing RelVals were found.", []))
    @patch("server.api_server._latest_ib_records")
    @patch("server.api_server._llm_client", object())
    def test_deterministic_answer_verification_fails_open_on_verifier_error(
        self, latest_ib_records, execute_capability, verify_answer,
        store_chat_turns, restore, store_decision, clear_pending
    ):
        # A verifier-layer failure (network, timeout, malformed client
        # response, ...) must never be worse than not having verification at
        # all -- the call site must catch it and trust the deterministic
        # answer, not let it propagate into a 500.
        latest_ib_records.return_value = [{"tag": "CMSSW_20_1_X_2026-08-17-2300"}]

        result = chat(ChatRequest(
            question="how many relvals failing in the latest IB", session_id="semantic-mismatch-verifier-error",
        ))

        self.assertEqual(result.answer, "No failing RelVals were found.")
        self.assertEqual(result.route, "capability:relval")

    def test_evidence_grounded_accepts_numbers_present_in_evidence(self):
        evidence = [{"tool": "t", "arguments": {}, "result": {"failing_count": 5, "tag": "CMSSW_20_1_X"}}]
        self.assertTrue(_evidence_grounded("There are 5 failures in CMSSW_20_1_X.", evidence))

    def test_evidence_grounded_rejects_fabricated_number(self):
        evidence = [{"tool": "t", "arguments": {}, "result": {"failing_count": 5}}]
        self.assertFalse(_evidence_grounded("There are 42 failures.", evidence))

    def test_format_raw_evidence_includes_tool_name_and_result(self):
        evidence = [{"tool": "get_ib_summary", "arguments": {"tag": "CMSSW_20_1_X"}, "result": {"failing_count": 5}}]
        rendered = _format_raw_evidence(evidence)
        self.assertIn("get_ib_summary", rendered)
        self.assertIn("failing_count", rendered)
        self.assertIn("5", rendered)

    def _gpu_relvals_chunk(self):
        return {
            "heading": "[ib-run-gpu-relvals](https://cmssdt.cern.ch/jenkins/job/ib-run-gpu-relvals)",
            "text": "Downstream projects: [ib-run-relvals](#ib-run-relvals) Sub-projects: [ib-run-relvals](#ib-run-relvals)",
            "source_path": "jenkins_reports/CMSSDT.md",
        }

    def _unrelated_job_chunk(self):
        return {
            "heading": "[process-relval-logs](https://cmssdt.cern.ch/jenkins/job/process-relval-logs)",
            "text": "Upstream projects: [ib-validation](#ib-validation) Downstream: [update-das-queries](#update-das-queries)",
            "source_path": "jenkins_reports/CMSSDT.md",
        }

    def test_shifter_doc_claims_rejects_cross_entity_attribution(self):
        # Regression for a real live finding: question asked about
        # ib-run-gpu-relvals, whose own retrieved chunk only mentions
        # ib-run-relvals -- but the synthesized answer reported
        # process-relval-logs's real upstream/downstream names
        # (ib-validation, update-das-queries) as if they belonged to
        # ib-run-gpu-relvals. Correct retrieval, wrong attribution.
        evidence = [{
            "tool": "search_shifter_docs", "arguments": {"query": "x"},
            "result": [self._gpu_relvals_chunk(), self._unrelated_job_chunk()],
        }]
        answer = (
            "ib-run-gpu-relvals' upstream projects are ib-validation, and its "
            "downstream project is update-das-queries."
        )

        self.assertFalse(_shifter_doc_claims_grounded(
            "what does ib-run-gpu-relvals depend on", answer, evidence,
        ))

    def test_shifter_doc_claims_accepts_facts_from_the_subjects_own_chunk(self):
        evidence = [{
            "tool": "search_shifter_docs", "arguments": {"query": "x"},
            "result": [self._gpu_relvals_chunk(), self._unrelated_job_chunk()],
        }]
        answer = "ib-run-gpu-relvals is a sub-project of ib-run-relvals."

        self.assertTrue(_shifter_doc_claims_grounded(
            "what does ib-run-gpu-relvals depend on", answer, evidence,
        ))

    def test_shifter_doc_claims_ignores_prose_docs_with_no_job_headings(self):
        # The forward-port/release-cycle style pages have no Jenkins-job-URL
        # headings at all -- this guard must not engage for them, since
        # there's nothing job-shaped to cross-check (verified correct live).
        evidence = [{
            "tool": "search_shifter_docs", "arguments": {"query": "x"},
            "result": [{"heading": "Forward ports", "text": "auto-forward-port is a placeholder job.",
                        "source_path": "forward-ports.md"}],
        }]

        self.assertTrue(_shifter_doc_claims_grounded(
            "what is a forward port in CMSSW", "It's a placeholder job.", evidence,
        ))

    def test_shifter_doc_claims_ignores_non_doc_evidence(self):
        # Ordinary graph-tool evidence (no search_shifter_docs call at all)
        # must be completely unaffected by this guard.
        evidence = [{"tool": "get_failing_relvals", "arguments": {}, "result": {"failed": 3}}]

        self.assertTrue(_shifter_doc_claims_grounded("how many relvals are failing", "3 are failing.", evidence))

    # -- Regression tests for bugs found and fixed this session (Phase 0 golden set) --

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._chat_response", side_effect=lambda *args, **kwargs: args[2])
    @patch("server.api_server._execute_planned_capability", return_value=("relval evidence", []))
    @patch("server.api_server._latest_ib_records")
    @patch("server.api_server._with_results_or_newest_complete", side_effect=lambda tag, *args: tag)  # R1-5: no graph in this test
    def test_latest_ib_phrase_resolves_to_a_tag_before_routing(
        self, _with_results, latest_ib_records, execute, chat_response, restore, store_decision, clear_pending
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
    @patch("server.api_server._store_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server._failure_overview_answer", return_value=("overview", []))
    @patch("server.api_server._latest_ib_records")
    @patch("server.api_server._with_results_or_newest_complete", side_effect=lambda tag, *args: tag)  # R1-2: no graph in this test
    def test_bare_failure_request_resolves_to_latest_tag_before_routing(
        self, _with_results, latest_ib_records, overview, store_chat_turns, restore, store_decision, store_pending, clear_pending
    ):
        # "what are failing" names no tag, no category, and doesn't even say
        # "latest"/"IB" -- it must still resolve to the latest IB and answer.
        latest_ib_records.return_value = [{"tag": "CMSSW_20_1_X_2026-08-17-2300"}]
        session_id = "bare-failure-resolve"
        try:
            result = chat(ChatRequest(question="what are failing", session_id=session_id))
        finally:
            _latest_ib_contexts.pop(session_id, None)

        self.assertEqual(result.route, "capability:failure_overview")
        overview.assert_called_once_with("CMSSW_20_1_X_2026-08-17-2300", None)

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._store_chat_turns")
    @patch("server.api_server._failure_overview_answer", return_value=("overview", []))
    @patch("server.api_server._latest_ib_records")
    def test_bare_failure_request_reuses_an_existing_session_tag_over_latest(
        self, latest_ib_records, overview, store_chat_turns, restore, store_decision, store_pending, clear_pending
    ):
        # A bare follow-up in a session that already has a tag must use THAT
        # tag, never silently jump to the latest IB instead.
        session_id = "bare-failure-existing-context"
        established_tag = "CMSSW_20_1_X_2026-07-01-1100"
        _latest_ib_contexts[session_id] = established_tag
        try:
            result = chat(ChatRequest(question="what are failing", session_id=session_id))
        finally:
            _latest_ib_contexts.pop(session_id, None)

        self.assertEqual(result.route, "capability:failure_overview")
        latest_ib_records.assert_not_called()
        overview.assert_called_once_with(established_tag, None)

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

    def test_is_latest_ib_question_recognizes_release_as_a_synonym(self):
        # Regression: found live 2026-09-18 -- "what relvals are failing in latest
        # release" never matched at all, because this function required the literal
        # word "ib"/"integration build" alongside "latest" and didn't know "release"
        # is the same thing in CMSSW terminology (the tags themselves are named after
        # a release series, e.g. CMSSW_20_1_X). A fresh session with no tag yet had
        # no way to resolve "latest" into an actual tag and fell through to the flat
        # "No imported local evidence" message even though real data was imported.
        self.assertTrue(_is_latest_ib_question("what relvals are failing in latest release"))

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

    # -- Bare "last" as a synonym for "latest" -- found live, 2026-10-02:
    # "last" had the exact same drift as the "lastest" typo above, except in
    # reverse -- only _is_latest_ib_question had its own local `|last`
    # addition, so _latest_issue_answer, _latest_clang_warning_answer, and
    # _is_latest_comparison_question never recognized this extremely
    # ordinary synonym at all. Folded into the shared _LATEST_WORD_RE itself
    # this time instead of patched into one more call site.

    def test_is_latest_ib_question_recognizes_bare_last(self):
        self.assertTrue(_is_latest_ib_question("what workflows are failing in the last IB"))

    def test_is_latest_comparison_question_recognizes_bare_last(self):
        self.assertTrue(_is_latest_comparison_question("compare the last IB with the previous one"))

    @patch("server.api_server.get_driver")
    def test_latest_clang_warning_answer_recognizes_bare_last(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"tag": "CMSSW_20_1_X_2026-08-25-2300", "release": "CMSSW_20_1_X",
             "arch": "el9_amd64_gcc14", "locations": 3, "occurrences": 7},
        ]

        answer = _latest_clang_warning_answer("what are the last clang warnings")

        self.assertIsNotNone(answer)

    @patch("server.api_server.get_driver")
    def test_latest_issue_answer_recognizes_bare_last(self, get_driver):
        # Real bug found live, 2026-10-02: "what is the last issue number you
        # have?" fell all the way through to the generic "I couldn't match
        # that" fallback, even though _latest_issue_answer already does
        # exactly this lookup -- its trigger just never recognized "last".
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "number": 51644, "title": "x", "state": "open", "created_at": None,
        }

        answer = _latest_issue_answer("what is the last issue number you have")

        self.assertIsNotNone(answer)
        self.assertIn("51644", answer)

    def test_clang_warning_answer_still_excludes_bare_last(self):
        self.assertIsNone(_clang_warning_answer("what are the last clang warnings"))


class ShiftSummaryCapabilityTests(unittest.TestCase):
    def setUp(self):
        # kept off the live graph: R1-15 (2026-11-11) checks whether the newest build has results before using the newest pair (tested in test_ib_health_questions)
        patcher = patch("server.api_server._shift_pair_with_results", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

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
            # Regression: found live -- "new <noun> [are/is] failing" (not
            # just the exact-noun "new failures/errors" form above) also
            # asks for the previous-vs-latest delta ("only those absent in
            # the previous release"), the same generate_shift_summary
            # request, just phrased around a different noun (workflows).
            "what new workflows are failing in the latest ib",
            "what new relvals are failing",
            # Regression: found live -- naming two IBs directly and asking
            # for one category's new failures fell through to the two-IB
            # comparison-or-single-build clarification instead of routing to
            # shift_summary, because the old pattern required "new" and the
            # failure noun to be adjacent with nothing (like a category
            # noun) in between, and had no reversed-order form at all.
            "what are the new relval failures between CMSSW_20_1_X_2026-08-17-1100 "
            "and CMSSW_20_1_X_2026-08-17-2300",
            "new unit test failures since the last build",
            "what relval failures are new between CMSSW_20_1_X_2026-08-17-1100 "
            "and CMSSW_20_1_X_2026-08-17-2300",
            "any unit test errors are new in the latest ib",
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

    @patch("server.api_server.generate_shift_summary", return_value="named-pair digest")
    def test_two_explicit_ibs_route_directly_to_generate_shift_summary(self, generate):
        # A shifter naming two IBs directly wants the diff for exactly those
        # builds, not the session's latest-vs-previous default -- and the
        # cache lookup (keyed only on latest-vs-previous) must not be
        # consulted for an arbitrary named pair.
        answer, actions = _execute_planned_capability(
            "shift_summary",
            "what are the new relval failures between CMSSW_20_1_X_2026-08-17-2300 "
            "and CMSSW_20_1_X_2026-08-17-1100",
            "s1",
        )

        self.assertEqual(answer, "named-pair digest")
        self.assertEqual(actions, [])
        # Ordered by _ib_sort_key, not question word order: the 2300 tag was
        # typed first but is the newer build, so it must land as latest_tag.
        generate.assert_called_once_with(
            "CMSSW_20_1_X_2026-08-17-2300", "CMSSW_20_1_X_2026-08-17-1100", None,
        )

    @patch("server.api_server._latest_and_previous_ib_tags", return_value=None)
    def test_not_enough_history_returns_graceful_message(self, latest_and_previous):
        answer, actions = _execute_planned_capability("shift_summary", "give me a shift summary", "s1")

        self.assertIn("Not enough Integration Build history", answer)
        self.assertEqual(actions, [])


class RecencyWindowParsingTests(unittest.TestCase):
    def test_digit_count_accepts_both_last_and_past_prefixes(self):
        for prefix in ["last", "past", "previous", "within the", "over the"]:
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    _parse_recency_window_days(f"new failing workflows {prefix} 15 days"), 15,
                )

    def test_bare_digit_count_needs_no_prefix(self):
        self.assertEqual(_parse_recency_window_days("new failing workflows 30 days"), 30)

    def test_word_number_week_and_month_forms(self):
        self.assertEqual(_parse_recency_window_days("what's new in the past one week"), 7)
        self.assertEqual(_parse_recency_window_days("what's new in the past one month"), 30)
        self.assertEqual(_parse_recency_window_days("failures over the last month"), 30)
        self.assertEqual(_parse_recency_window_days("failures in the last a week"), 7)

    def test_day_and_24_hour_forms(self):
        self.assertEqual(_parse_recency_window_days("new failures in the last day"), 1)
        self.assertEqual(_parse_recency_window_days("new failures within the last 24 hours"), 1)

    def test_all_time_phrasing_returns_none_regardless_of_default(self):
        self.assertIsNone(_parse_recency_window_days("failures of all time", default=7))
        self.assertIsNone(_parse_recency_window_days("has this ever failed", default=7))

    def test_bare_all_as_a_determiner_is_not_mistaken_for_all_time(self):
        # Regression: found live -- "list all the new failing workflows for
        # the past 15 days" tripped the all-time check on the plain
        # determiner "all", discarding the explicit 15-day window entirely.
        self.assertEqual(
            _parse_recency_window_days(
                "list all the new failing workflows for the past 15 days"
            ),
            15,
        )

    def test_no_window_mentioned_uses_caller_supplied_default(self):
        # Frequent-failures keeps its long-standing 7-day default...
        self.assertEqual(_parse_recency_window_days("what fails most often"), 7)
        # ...but shift-summary passes default=None so a plain "what changed"
        # falls back to "immediately previous build," not a 7-day window.
        self.assertIsNone(_parse_recency_window_days("what changed", default=None))


class LatestAndDaysAgoIbTagsTests(unittest.TestCase):
    @patch("server.api_server.get_driver")
    def test_anchors_to_the_build_closest_to_the_requested_window(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"tag": "CMSSW_20_1_X_2026-08-25-2300"},
            {"tag": "CMSSW_20_1_X_2026-08-20-2300"},
            {"tag": "CMSSW_20_1_X_2026-08-10-2300"},
            {"tag": "CMSSW_20_1_X_2026-08-01-2300"},
        ]

        with patch("server.api_server.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 8, 25, 23, 0, tzinfo=timezone.utc)
            result = _latest_and_days_ago_ib_tags(15)

        self.assertEqual(
            result, ("CMSSW_20_1_X_2026-08-25-2300", "CMSSW_20_1_X_2026-08-10-2300"),
        )
        get_driver.return_value.close.assert_called_once()

    @patch("server.api_server.get_driver")
    def test_falls_back_to_oldest_build_when_none_is_old_enough(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"tag": "CMSSW_20_1_X_2026-08-25-2300"},
            {"tag": "CMSSW_20_1_X_2026-08-24-2300"},
        ]

        with patch("server.api_server.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 8, 25, 23, 0, tzinfo=timezone.utc)
            result = _latest_and_days_ago_ib_tags(30)

        self.assertEqual(
            result, ("CMSSW_20_1_X_2026-08-25-2300", "CMSSW_20_1_X_2026-08-24-2300"),
        )

    @patch("server.api_server.get_driver")
    def test_only_one_build_in_the_newest_series_returns_none(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"tag": "CMSSW_20_1_X_2026-08-25-2300"}]

        result = _latest_and_days_ago_ib_tags(15)

        self.assertIsNone(result)

    @patch("server.api_server.get_driver")
    def test_picks_the_newest_release_series_not_just_the_newest_timestamp(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"tag": "CMSSW_20_1_X_2026-08-25-2300"},
            {"tag": "CMSSW_20_1_X_2026-08-10-2300"},
            {"tag": "CMSSW_20_0_X_2026-08-24-2300"},
        ]

        with patch("server.api_server.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 8, 25, 23, 0, tzinfo=timezone.utc)
            result = _latest_and_days_ago_ib_tags(30)

        self.assertEqual(
            result, ("CMSSW_20_1_X_2026-08-25-2300", "CMSSW_20_1_X_2026-08-10-2300"),
        )


class ShiftSummaryDayWindowTests(unittest.TestCase):
    def setUp(self):
        # kept off the live graph: R1-15 (2026-11-11) checks whether the newest build has results before using the newest pair (tested in test_ib_health_questions)
        patcher = patch("server.api_server._shift_pair_with_results", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_trigger_regex_recognizes_day_window_phrasings(self):
        for question in [
            "list all the new failing workflows for the past one week",
            "list all the new failing workflows for the past 15 days",
            "list all the new failing workflows for the past 30 days",
            "list all the new failing workflows for the past one month",
        ]:
            with self.subTest(question=question):
                self.assertTrue(RequestContext.extract(question).has_shift_summary_request)

    @patch("server.api_server.generate_shift_summary", return_value="15-day digest")
    @patch("server.api_server._latest_and_days_ago_ib_tags")
    def test_execute_planned_capability_threads_the_day_window_through(
        self, days_ago, generate,
    ):
        days_ago.return_value = ("CMSSW_20_1_X_2026-08-25-2300", "CMSSW_20_1_X_2026-08-10-2300")

        answer, actions = _execute_planned_capability(
            "shift_summary",
            "list all the new failing workflows for the past 15 days",
            "s1",
        )

        days_ago.assert_called_once_with(15, None)
        generate.assert_called_once_with(
            "CMSSW_20_1_X_2026-08-25-2300", "CMSSW_20_1_X_2026-08-10-2300", None,
        )
        self.assertIn("15-day digest", answer)
        self.assertIn("CMSSW_20_1_X_2026-08-10-2300", answer)
        self.assertEqual(actions, [])

    @patch("server.api_server._latest_and_previous_ib_tags")
    def test_plain_shift_summary_question_still_uses_immediately_previous_build(
        self, latest_and_previous,
    ):
        # Regression guard: default=None must mean "no window mentioned falls
        # back to the immediately-previous-build comparison," not "7 days."
        latest_and_previous.return_value = None

        _execute_planned_capability("shift_summary", "give me a shift summary", "s1")

        latest_and_previous.assert_called_once_with(None)

    @patch("server.api_server._latest_and_days_ago_ib_tags", return_value=None)
    def test_not_enough_history_for_the_window_returns_graceful_message(self, days_ago):
        answer, actions = _shift_summary_answer(None, days=15), []

        self.assertIn("Not enough Integration Build history", answer)
        self.assertIn("15-day window", answer)


class RenderMetricDeltasTests(unittest.TestCase):
    """A real "all architectures" comparison (2026-09-09, CLANG_X) came back with 5 of
    11 metrics reading "0 | 0 | +0" -- pure noise -- and 2 more ("RelVals run",
    "Unit tests run") unchanged but nonzero, repeating the same number twice for no
    reason. User feedback: don't show zero-to-zero at all, and for an unchanged
    nonzero metric just say the old value and that nothing new happened, instead of a
    redundant three-column row."""

    METRICS = [("Failed RelVals", "failed"), ("RelVal errors", "errors"), ("RelVals run", "total")]

    def test_omits_zero_to_zero_metrics_entirely(self):
        before = {"failed": 0, "errors": 5, "total": 10}
        after = {"failed": 0, "errors": 6, "total": 10}
        lines = _render_metric_deltas(before, after, self.METRICS, "tagA", "tagB")
        joined = "\n".join(lines)

        self.assertNotIn("Failed RelVals", joined)

    def test_changed_metric_gets_a_real_table_row(self):
        before = {"failed": 1, "errors": 5, "total": 10}
        after = {"failed": 2, "errors": 5, "total": 10}
        lines = _render_metric_deltas(before, after, self.METRICS, "tagA", "tagB")
        joined = "\n".join(lines)

        self.assertIn("| Failed RelVals | 1 | 2 | +1 |", joined)

    def test_unchanged_nonzero_metric_becomes_a_compact_note_not_a_row(self):
        before = {"failed": 1, "errors": 5, "total": 10}
        after = {"failed": 1, "errors": 6, "total": 10}
        lines = _render_metric_deltas(before, after, self.METRICS, "tagA", "tagB")
        joined = "\n".join(lines)

        self.assertNotIn("| Failed RelVals |", joined)
        # RelVal errors moved, so this is not "no new failures" (found 2026-11-04)
        self.assertIn("Unchanged: Failed RelVals still 1", joined)
        self.assertIn("| RelVal errors | 5 | 6 | +1 |", joined)

    def test_all_metrics_unchanged_or_zero_reports_no_changes(self):
        before = {"failed": 0, "errors": 0, "total": 10}
        after = {"failed": 0, "errors": 0, "total": 10}
        lines = _render_metric_deltas(before, after, self.METRICS, "tagA", "tagB")

        self.assertEqual(
            lines,
            ["No new failures/errors -- RelVals run still 10."],
        )

    def test_completely_unchanged_and_zero_reports_no_changes_between_builds(self):
        before = {"failed": 0, "errors": 0, "total": 0}
        after = {"failed": 0, "errors": 0, "total": 0}
        lines = _render_metric_deltas(before, after, self.METRICS, "tagA", "tagB")

        self.assertEqual(lines, ["No changes between these builds."])


class IbComparisonAnswerTests(unittest.TestCase):
    """Regression: comparing two sub-IB tags directly (e.g. two dated builds of the
    same CLANG_X series) used to silently report "neither requested IB is present"
    even when both were fully imported -- the grouping keyed rows by their *resolved*
    primary tag, which never equals a sub-IB tag typed by the caller. Found live
    (2026-09-09) comparing CMSSW_20_1_CLANG_X_2026-09-04-2300 vs ...-09-06-2300."""

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_driver")
    def test_failing_workflows_section_lists_every_category_by_name(self, get_driver, get_exit_code_names):
        # A shifter needs the full picture: which workflows are brand new, which were
        # already failing before this window (so a persistent failure doesn't look
        # like something new), and which just got fixed -- not only the ones whose
        # error/warning counts happened to move. Found live (2026-09-09, EVTGEN3):
        # 9 of 12 failing workflows were carried over with byte-identical stats and
        # used to vanish entirely from "Changed failing workflows".
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {
                "tag": "CMSSW_20_1_CLANG_X_2026-09-04-2300", "primary_tag": "CMSSW_20_1_X_2026-09-04-2300",
                "variant": "CLANG", "relval_total": 10, "relval_failed": 3, "relval_errors": 5,
                "relval_warnings": 2, "unit_test_total": 3, "unit_test_failed": 0, "unit_test_errors": 0,
                "failing_relvals": [
                    {"workflow_id": "100.0", "name": "StillUnchanged", "errors": 1, "warnings": 1, "exitcode": 1},
                    {"workflow_id": "200.0", "name": "StillChanged", "errors": 2, "warnings": 2, "exitcode": 2},
                    {"workflow_id": "300.0", "name": "WillResolve", "errors": 3, "warnings": 3, "exitcode": 3},
                ],
            },
            {
                "tag": "CMSSW_20_1_CLANG_X_2026-09-06-2300", "primary_tag": "CMSSW_20_1_X_2026-09-06-2300",
                "variant": "CLANG", "relval_total": 10, "relval_failed": 3, "relval_errors": 6,
                "relval_warnings": 3, "unit_test_total": 3, "unit_test_failed": 0, "unit_test_errors": 0,
                "failing_relvals": [
                    {"workflow_id": "100.0", "name": "StillUnchanged", "errors": 1, "warnings": 1, "exitcode": 1},
                    {"workflow_id": "200.0", "name": "StillChanged", "errors": 5, "warnings": 2, "exitcode": 2},
                    {"workflow_id": "400.0", "name": "NewlyFailing", "errors": 4, "warnings": 4, "exitcode": 4},
                ],
            },
        ]

        answer, actions = _ib_comparison_answer(
            "compare CMSSW_20_1_CLANG_X_2026-09-04-2300 and CMSSW_20_1_CLANG_X_2026-09-06-2300 el9_amd64_gcc14"
        )

        self.assertIn(
            "| `100.0` | StillUnchanged | 1 | 1 | Still failing (already failing in "
            "`CMSSW_20_1_CLANG_X_2026-09-04-2300`) | - | - |",
            answer,
        )
        self.assertIn(
            "| `200.0` | StillChanged | 5 | 2 | Still failing (already failing in "
            "`CMSSW_20_1_CLANG_X_2026-09-04-2300`) | - | - |",
            answer,
        )
        self.assertIn("| `300.0` | WillResolve | 3 | 3 | Resolved | - | - |", answer)
        self.assertIn("| `400.0` | NewlyFailing | 4 | 4 | Newly failing | - | - |", answer)

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_driver")
    def test_same_workflow_id_different_exit_code_is_resolved_plus_newly_failing(
        self, get_driver, get_exit_code_names
    ):
        # _render_failing_workflows now keys rows by the same (workflow_id, step,
        # exitcode) identity as _diff_failing_relvals/generate_shift_summary (see
        # _relval_identity) instead of workflow_id alone -- a workflow failing with a
        # different exit code between builds must show as its old identity resolved
        # and a new identity newly failing, not as one "still failing" row that
        # silently absorbs the exit-code change.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {
                "tag": "CMSSW_20_1_CLANG_X_2026-09-04-2300", "primary_tag": "CMSSW_20_1_X_2026-09-04-2300",
                "variant": "CLANG", "relval_total": 10, "relval_failed": 1, "relval_errors": 1,
                "relval_warnings": 0, "unit_test_total": 0, "unit_test_failed": 0, "unit_test_errors": 0,
                "failing_relvals": [
                    {"workflow_id": "100.0", "name": "FlipsExitCode", "errors": 1, "warnings": 0, "exitcode": 1},
                ],
            },
            {
                "tag": "CMSSW_20_1_CLANG_X_2026-09-06-2300", "primary_tag": "CMSSW_20_1_X_2026-09-06-2300",
                "variant": "CLANG", "relval_total": 10, "relval_failed": 1, "relval_errors": 1,
                "relval_warnings": 0, "unit_test_total": 0, "unit_test_failed": 0, "unit_test_errors": 0,
                "failing_relvals": [
                    {"workflow_id": "100.0", "name": "FlipsExitCode", "errors": 1, "warnings": 0, "exitcode": 2},
                ],
            },
        ]

        answer, actions = _ib_comparison_answer(
            "compare CMSSW_20_1_CLANG_X_2026-09-04-2300 and CMSSW_20_1_CLANG_X_2026-09-06-2300 el9_amd64_gcc14"
        )

        self.assertIn("| `100.0` | FlipsExitCode | 1 | 1 | Resolved | - | - |", answer)
        self.assertIn("| `100.0` | FlipsExitCode | 1 | 2 | Newly failing | - | - |", answer)

    @staticmethod
    def _sub_ib_rows(before_failed: int, after_failed: int, before_ut_failed: int = 0, after_ut_failed: int = 0):
        return [
            {
                "tag": "CMSSW_20_1_CLANG_X_2026-09-04-2300", "primary_tag": "CMSSW_20_1_X_2026-09-04-2300",
                "variant": "CLANG", "relval_total": 10, "relval_failed": before_failed,
                "relval_errors": 5, "relval_warnings": 2, "unit_test_total": 3,
                "unit_test_failed": before_ut_failed, "unit_test_errors": 0,
                "failing_relvals": [
                    {"workflow_id": "140.051", "name": "RunMuonEG2022C", "errors": 1, "warnings": 1, "exitcode": 16640},
                ][:before_failed],
            },
            {
                "tag": "CMSSW_20_1_CLANG_X_2026-09-06-2300", "primary_tag": "CMSSW_20_1_X_2026-09-06-2300",
                "variant": "CLANG", "relval_total": 10, "relval_failed": after_failed,
                "relval_errors": 6, "relval_warnings": 3, "unit_test_total": 3,
                "unit_test_failed": after_ut_failed, "unit_test_errors": 0,
                "failing_relvals": [
                    {"workflow_id": "140.051", "name": "RunMuonEG2022C", "errors": 1, "warnings": 1, "exitcode": 16640},
                    {"workflow_id": "37634.911", "name": "TTbar", "errors": 2, "warnings": 0, "exitcode": 1},
                ][:after_failed],
            },
        ]

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_driver")
    def test_matches_when_both_requested_tags_are_sub_ib_tags(self, get_driver, get_exit_code_names):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = self._sub_ib_rows(before_failed=1, after_failed=1)

        answer, actions = _ib_comparison_answer(
            "compare CMSSW_20_1_CLANG_X_2026-09-04-2300 and CMSSW_20_1_CLANG_X_2026-09-06-2300 el9_amd64_gcc14"
        )

        self.assertNotIn("neither requested IB", answer)
        self.assertIn("### `CLANG`", answer)
        # Failed RelVals is unchanged and nonzero (1 -> 1), so it's folded into the
        # compact note rather than a redundant "1 | 1 | +0" row -- see _render_metric_deltas.
        self.assertIn("Failed RelVals still 1", answer)
        self.assertNotIn("| Failed RelVals |", answer)

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.find_prs_merged_between", return_value=[])
    @patch("server.api_server._single_ib_comparison_row")
    @patch("server.api_server._variant_previous_tag")
    @patch("server.api_server.get_driver")
    def test_falls_back_to_own_previous_build_and_cross_references_it(
        self, get_driver, variant_previous_tag, single_ib_row, find_prs_merged_between, get_exit_code_names,
    ):
        # Part 1: a variant with no exact match at before_tag (but one at after_tag)
        # falls back to its own true previous build, same policy
        # generate_shift_summary's _variant_diff_baseline already uses for the shift
        # digest -- applied here to the chat comparison too. Part 2: cross-variant
        # "Also failing in" references must reflect that SUBSTITUTED tag/row, not
        # pretend the literal before_tag applies to a variant that never had a build
        # there.
        before_tag, after_tag = "CMSSW_20_1_X_2026-08-31-1100", "CMSSW_20_1_X_2026-09-01-1100"
        substitute_tag = "CMSSW_20_1_ASAN_X_2026-08-27-1100"

        def row(tag, primary_tag, variant, failed, failing_relvals):
            return {
                "tag": tag, "primary_tag": primary_tag, "variant": variant,
                "relval_total": 10, "relval_failed": failed, "relval_errors": failed,
                "relval_warnings": failed, "unit_test_total": 1, "unit_test_failed": 0,
                "unit_test_errors": 0, "failing_relvals": failing_relvals,
            }

        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            row(before_tag, before_tag, "primary", 0, []),
            row(after_tag, after_tag, "primary", 1, [
                {"workflow_id": "700.0", "name": "PrimaryNew", "errors": 1, "warnings": 1, "exitcode": 1},
            ]),
            # No ASAN row at before_tag at all -- the "no exact match" case.
            row("CMSSW_20_1_ASAN_X_2026-09-01-1100", after_tag, "ASAN", 1, [
                {"workflow_id": "700.0", "name": "PrimaryNew", "errors": 2, "warnings": 2, "exitcode": 2},
            ]),
        ]
        variant_previous_tag.return_value = substitute_tag
        single_ib_row.return_value = row(substitute_tag, "CMSSW_20_1_X_2026-08-27-1100", "ASAN", 1, [
            {"workflow_id": "700.0", "name": "PrimaryNew", "errors": 2, "warnings": 2, "exitcode": 2},
        ])

        answer, actions = _ib_comparison_answer(f"compare {before_tag} and {after_tag} el9_amd64_gcc14")

        variant_previous_tag.assert_called_once_with(
            "CMSSW_20_1_ASAN_X_2026-09-01-1100", "ASAN", "el9_amd64_gcc14"
        )
        single_ib_row.assert_called_once_with(substitute_tag, "el9_amd64_gcc14")
        self.assertIn(
            f"`ASAN` has no build at `{before_tag}` -- compared against its own "
            f"previous build `{substitute_tag}` instead.",
            answer,
        )

        primary_block = answer.split("### `ASAN`")[0]
        asan_block = answer.split("### `ASAN`")[1]
        # Primary's newly-failing 700.0 must cross-reference ASAN's SUBSTITUTED
        # before-side tag, not a literal (nonexistent) before_tag-dated ASAN build.
        self.assertIn(
            f"| `700.0` | PrimaryNew | 1 | 1 | Newly failing | ASAN (`{substitute_tag}`) | "
            f"ASAN (`CMSSW_20_1_ASAN_X_2026-09-01-1100`) |",
            primary_block,
        )
        # ASAN's own row, diffed against its substituted baseline, correctly shows
        # "still failing" (already failing in the substitute) and cross-references
        # primary only on the after side (primary didn't have it failing before).
        self.assertIn(
            f"| `700.0` | PrimaryNew | 2 | 2 | Still failing (already failing in `{before_tag}`) | "
            f"- | primary (`{after_tag}`) |",
            asan_block,
        )

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server._single_ib_comparison_row")
    @patch("server.api_server._variant_previous_tag", return_value=None)
    @patch("server.api_server.get_driver")
    def test_no_earlier_build_at_all_keeps_existing_no_record_message(
        self, get_driver, variant_previous_tag, single_ib_row, get_exit_code_names,
    ):
        # When _variant_previous_tag finds nothing (first-ever import of this
        # sub-IB, or genuinely no earlier build), behavior must stay exactly what it
        # was before Part 1 existed: the plain "no imported record" message, no
        # substitution note, and no wasted row fetch.
        before_tag, after_tag = "CMSSW_20_1_X_2026-08-31-1100", "CMSSW_20_1_X_2026-09-01-1100"

        def row(tag, primary_tag, variant, failed, failing_relvals):
            return {
                "tag": tag, "primary_tag": primary_tag, "variant": variant,
                "relval_total": 10, "relval_failed": failed, "relval_errors": failed,
                "relval_warnings": failed, "unit_test_total": 1, "unit_test_failed": 0,
                "unit_test_errors": 0, "failing_relvals": failing_relvals,
            }

        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            row(before_tag, before_tag, "primary", 0, []),
            row(after_tag, after_tag, "primary", 0, []),
            row("CMSSW_20_1_G4ADEPT_X_2026-09-01-1100", after_tag, "G4ADEPT", 0, []),
        ]

        answer, actions = _ib_comparison_answer(f"compare {before_tag} and {after_tag} el9_amd64_gcc14")

        single_ib_row.assert_not_called()
        self.assertIn(
            f"`{before_tag}` has no imported record for this sub-IB on `el9_amd64_gcc14`.",
            answer,
        )
        self.assertNotIn("has no build at", answer)

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.find_prs_merged_between")
    @patch("server.api_server.get_driver")
    def test_lists_merged_prs_when_relval_failures_increased(self, get_driver, find_prs_merged_between, get_exit_code_names):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = self._sub_ib_rows(before_failed=1, after_failed=2)
        find_prs_merged_between.return_value = [
            {"repo": "cmssw", "pr_number": 51659, "title": "TICLGeom: parallelize the geometry SoA build",
             "merged_at": "2026-09-05T10:00:00Z", "author": "felicepantaleo"},
        ]

        answer, actions = _ib_comparison_answer(
            "compare CMSSW_20_1_CLANG_X_2026-09-04-2300 and CMSSW_20_1_CLANG_X_2026-09-06-2300 el9_amd64_gcc14"
        )

        # Must be looked up against this block's own exact tags (the sub-IB's tags),
        # not the primary tags -- a sub-IB's regression must never be attributed to
        # PRs merged into a different variant.
        find_prs_merged_between.assert_called_once_with(
            "CMSSW_20_1_CLANG_X_2026-09-04-2300", "CMSSW_20_1_CLANG_X_2026-09-06-2300"
        )
        self.assertIn("PRs merged between", answer)
        self.assertIn("PR #51659", answer)
        self.assertIn("do not prove a PR or Issue caused or fixed a failure", answer)

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.find_prs_merged_between")
    @patch("server.api_server.get_driver")
    def test_skips_merged_pr_lookup_when_nothing_regressed(self, get_driver, find_prs_merged_between, get_exit_code_names):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = self._sub_ib_rows(before_failed=1, after_failed=1)

        answer, actions = _ib_comparison_answer(
            "compare CMSSW_20_1_CLANG_X_2026-09-04-2300 and CMSSW_20_1_CLANG_X_2026-09-06-2300 el9_amd64_gcc14"
        )

        find_prs_merged_between.assert_not_called()
        self.assertNotIn("PRs merged between", answer)

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.find_prs_merged_between", return_value=[])
    @patch("server.api_server.get_driver")
    def test_omits_merged_pr_section_when_regressed_but_no_pr_recorded(
        self, get_driver, find_prs_merged_between, get_exit_code_names
    ):
        # Different code path than "nothing regressed": a real regression happened
        # (relval_failed went up), so the lookup IS made, but the graph has no PR
        # recorded as merged for this exact tag pair (verified live -- 81 of 98 real
        # regressions found in the imported graph on 2026-09-09 have an empty merged-PR
        # window, e.g. CMSSW_20_1_X_2026-08-20-2300 -> ...-08-25-2300 on el10_amd64_gcc15).
        # An empty result must not render as an empty/misleading "PRs merged between..."
        # header with nothing under it.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = self._sub_ib_rows(before_failed=1, after_failed=2)

        answer, actions = _ib_comparison_answer(
            "compare CMSSW_20_1_CLANG_X_2026-09-04-2300 and CMSSW_20_1_CLANG_X_2026-09-06-2300 el9_amd64_gcc14"
        )

        find_prs_merged_between.assert_called_once_with(
            "CMSSW_20_1_CLANG_X_2026-09-04-2300", "CMSSW_20_1_CLANG_X_2026-09-06-2300"
        )
        self.assertNotIn("PRs merged between", answer)

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.find_prs_merged_between", return_value=[])
    @patch("server.api_server.get_driver")
    def test_failing_workflows_table_shows_cross_variant_visibility(
        self, get_driver, find_prs_merged_between, get_exit_code_names
    ):
        # Explicit design (agreed with the user): the failing-workflow table stays
        # per-variant for classification (Newly failing/Still failing/Resolved never
        # changes based on what other variants show), but two extra columns list
        # every OTHER variant at this architecture also failing on the same
        # workflow_id, named by that variant's own full exact tag -- purely
        # additive visibility, not reclassification.
        before_tag, after_tag = "CMSSW_20_1_X_2026-09-04-2300", "CMSSW_20_1_X_2026-09-06-2300"
        asan_before, asan_after = "CMSSW_20_1_ASAN_X_2026-09-04-2300", "CMSSW_20_1_ASAN_X_2026-09-06-2300"
        clang_before, clang_after = "CMSSW_20_1_CLANG_X_2026-09-04-2300", "CMSSW_20_1_CLANG_X_2026-09-06-2300"

        def row(tag, primary_tag, variant, failed, failing_relvals):
            return {
                "tag": tag, "primary_tag": primary_tag, "variant": variant,
                "relval_total": 10, "relval_failed": failed, "relval_errors": failed,
                "relval_warnings": failed, "unit_test_total": 1, "unit_test_failed": 0,
                "unit_test_errors": 0, "failing_relvals": failing_relvals,
            }

        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            row(before_tag, before_tag, "primary", 1, [
                {"workflow_id": "600.0", "name": "NoOverlap", "errors": 1, "warnings": 1, "exitcode": 1},
            ]),
            row(after_tag, after_tag, "primary", 2, [
                {"workflow_id": "600.0", "name": "NoOverlap", "errors": 1, "warnings": 1, "exitcode": 1},
                {"workflow_id": "500.0", "name": "CrossVariant", "errors": 5, "warnings": 5, "exitcode": 5},
            ]),
            # ASAN already had 500.0 failing on BOTH sides -- unrelated to primary's
            # regression, but visible cross-reference material for primary's row.
            row(asan_before, before_tag, "ASAN", 1, [
                {"workflow_id": "500.0", "name": "CrossVariant", "errors": 3, "warnings": 3, "exitcode": 3},
            ]),
            row(asan_after, after_tag, "ASAN", 1, [
                {"workflow_id": "500.0", "name": "CrossVariant", "errors": 3, "warnings": 3, "exitcode": 3},
            ]),
            # CLANG never has either workflow failing -- the "-" / no-overlap case.
            row(clang_before, before_tag, "CLANG", 0, []),
            row(clang_after, after_tag, "CLANG", 0, []),
        ]

        answer, actions = _ib_comparison_answer(
            f"compare {before_tag} and {after_tag} el9_amd64_gcc14"
        )

        primary_block = answer.split("### `ASAN`")[0]
        asan_block = answer.split("### `ASAN`")[1].split("### `CLANG`")[0]
        clang_block = answer.split("### `CLANG`")[1]

        # Primary: 500.0 is newly failing, and ASAN already had it failing on both
        # sides -- both "Also failing in" columns must name ASAN's own exact tag.
        self.assertIn(
            f"| `500.0` | CrossVariant | 5 | 5 | Newly failing | ASAN (`{asan_before}`) | ASAN (`{asan_after}`) |",
            primary_block,
        )
        # 600.0 has no cross-variant overlap anywhere -- both cells read "-".
        self.assertIn(
            f"| `600.0` | NoOverlap | 1 | 1 | Still failing (already failing in `{before_tag}`) | - | - |",
            primary_block,
        )
        # ASAN's own row for 500.0: primary only had it failing from after_tag
        # onward, so primary appears in the after-side column only, not before --
        # the symmetric reference is real, not a mirrored copy of primary's row.
        self.assertIn(
            f"| `500.0` | CrossVariant | 3 | 3 | Still failing (already failing in `{before_tag}`) | "
            f"- | primary (`{after_tag}`) |",
            asan_block,
        )
        self.assertIn("No RelVal workflow is failing in either build.", clang_block)

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_driver")
    def test_all_arch_comparison_matches_when_both_requested_tags_are_sub_ib_tags(self, get_driver, get_exit_code_names):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {
                "tag": "CMSSW_20_1_CLANG_X_2026-09-04-2300", "primary_tag": "CMSSW_20_1_X_2026-09-04-2300",
                "arch": "el9_amd64_gcc14", "variant": "CLANG", "build_warnings": 0, "ignored_warnings": 0,
                "clang_locations": 0, "clang_occurrences": 0, "relval_total": 10, "relval_failed": 1,
                "relval_errors": 5, "relval_warnings": 2, "unit_test_total": 3, "unit_test_failed": 0,
                "unit_test_errors": 0,
                "failing_relvals": [
                    {"workflow_id": "140.051", "name": "RunMuonEG2022C", "errors": 1, "warnings": 1, "exitcode": 16640},
                ],
            },
            {
                "tag": "CMSSW_20_1_CLANG_X_2026-09-06-2300", "primary_tag": "CMSSW_20_1_X_2026-09-06-2300",
                "arch": "el9_amd64_gcc14", "variant": "CLANG", "build_warnings": 0, "ignored_warnings": 0,
                "clang_locations": 0, "clang_occurrences": 0, "relval_total": 10, "relval_failed": 2,
                "relval_errors": 6, "relval_warnings": 3, "unit_test_total": 3, "unit_test_failed": 0,
                "unit_test_errors": 0,
                "failing_relvals": [
                    {"workflow_id": "140.051", "name": "RunMuonEG2022C", "errors": 1, "warnings": 1, "exitcode": 16640},
                    {"workflow_id": "200.0", "name": "NewWorkflow", "errors": 2, "warnings": 2, "exitcode": 1},
                ],
            },
        ]

        answer = _all_arch_ib_comparison_answer(
            "compare CMSSW_20_1_CLANG_X_2026-09-04-2300 and CMSSW_20_1_CLANG_X_2026-09-06-2300 all architectures"
        )

        self.assertNotIn("has no imported record here", answer.split("#### `CLANG`")[1])
        self.assertIn("| Failed RelVals | 1 | 2 | +1 |", answer)
        # The "all architectures" comparison used to have no per-workflow breakdown at
        # all -- only aggregate metric counts -- so a shifter had no way to see which
        # workflows those counts referred to. Same _render_failing_workflows helper as
        # the single-architecture path now covers this comparison too.
        self.assertIn(
            "| `140.051` | RunMuonEG2022C | 1 | 16640 | Still failing (already failing in "
            "`CMSSW_20_1_CLANG_X_2026-09-04-2300`) | - | - |",
            answer,
        )
        self.assertIn("| `200.0` | NewWorkflow | 2 | 1 | Newly failing | - | - |", answer)
        # The consolidated cross-architecture summary (added so a shifter sees what
        # actually changed before scrolling past per-arch metric-delta detail) must
        # also list this newly-failing workflow, and must appear before the detail.
        self.assertIn("**Newly failing:**", answer)
        self.assertIn("`200.0` (`NewWorkflow`) on `el9_amd64_gcc14` (CLANG)", answer)
        self.assertLess(answer.index("**Newly failing:**"), answer.index("### `el9_amd64_gcc14`"))

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_driver")
    def test_all_arch_comparison_leads_with_cross_architecture_change_summary(self, get_driver, get_exit_code_names):
        # Regression: found live 2026-09-18 -- a real "all architectures" comparison
        # buried the one thing that mattered (one newly-failing workflow) under six
        # architectures of RelVal-warning-count churn and repeated sub-IB
        # substitution notes. User's own words: "the shifter not only needs the
        # interesting facts not noise" -- confirmed recommendation: lead with a
        # consolidated Newly failing/Resolved section across ALL architectures
        # before the per-architecture detail.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        base = {
            "primary_tag": "CMSSW_20_1_X_2026-09-04-2300", "variant": "primary", "build_warnings": 0,
            "ignored_warnings": 0, "clang_locations": 0, "clang_occurrences": 0, "relval_total": 10,
            "relval_failed": 0, "relval_errors": 0, "relval_warnings": 0, "unit_test_total": 3,
            "unit_test_failed": 0, "unit_test_errors": 0,
        }
        session.run.return_value = [
            {**base, "tag": "CMSSW_20_1_X_2026-09-04-2300", "arch": "arch_a", "relval_failed": 0,
             "failing_relvals": []},
            {**base, "tag": "CMSSW_20_1_X_2026-09-06-2300", "arch": "arch_a", "relval_failed": 1,
             "failing_relvals": [{"workflow_id": "537.0", "name": "SomeWorkflow", "errors": 1, "warnings": 0, "exitcode": 62720}]},
            {**base, "tag": "CMSSW_20_1_X_2026-09-04-2300", "arch": "arch_b", "relval_failed": 1,
             "failing_relvals": [{"workflow_id": "10805.31", "name": "OtherWorkflow", "errors": 7, "warnings": 0, "exitcode": 25600}]},
            {**base, "tag": "CMSSW_20_1_X_2026-09-06-2300", "arch": "arch_b", "relval_failed": 0,
             "failing_relvals": []},
        ]

        answer = _all_arch_ib_comparison_answer(
            "compare CMSSW_20_1_X_2026-09-04-2300 and CMSSW_20_1_X_2026-09-06-2300 all architectures"
        )

        summary = answer.split("---")[0]
        self.assertIn("`537.0` (`SomeWorkflow`) on `arch_a` (primary) -- 62720", summary)
        self.assertIn("`10805.31` (`OtherWorkflow`) on `arch_b` (primary) -- was 25600", summary)
        self.assertLess(summary.index("**Newly failing:**"), summary.index("**Resolved:**"))

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_driver")
    def test_all_arch_comparison_summary_says_so_when_nothing_changed(self, get_driver, get_exit_code_names):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        base = {
            "primary_tag": "CMSSW_20_1_X_2026-09-04-2300", "variant": "primary", "build_warnings": 0,
            "ignored_warnings": 0, "clang_locations": 0, "clang_occurrences": 0, "relval_total": 10,
            "relval_failed": 0, "relval_errors": 0, "relval_warnings": 0, "unit_test_total": 3,
            "unit_test_failed": 0, "unit_test_errors": 0, "failing_relvals": [],
        }
        session.run.return_value = [
            {**base, "tag": "CMSSW_20_1_X_2026-09-04-2300", "arch": "arch_a"},
            {**base, "tag": "CMSSW_20_1_X_2026-09-06-2300", "arch": "arch_a"},
        ]

        answer = _all_arch_ib_comparison_answer(
            "compare CMSSW_20_1_X_2026-09-04-2300 and CMSSW_20_1_X_2026-09-06-2300 all architectures"
        )

        self.assertIn(
            "No RelVal workflow is newly failing or resolved across any compared architecture.", answer
        )


if __name__ == "__main__":
    unittest.main()