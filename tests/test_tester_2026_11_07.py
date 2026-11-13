"""Wrong/dead-end answers found by reading the 2,803 earlier questions on the data of 2026-11-06 (weekend test, item C3)."""
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from server import api_server
from server.request_planner import RequestContext


def _rec(tag, variant, arch, failed=(), units=()):
    return {"tag": tag, "arch": arch, "variant": variant, "build": "passed", "relvals": 954, "failed": list(failed), "units": 380, "unit_failed": list(units)}


class FailureSetsTests(unittest.TestCase):
    def test_only_in_a_flavour_compares_with_the_other_flavours_of_the_build(self):
        primary = "CMSSW_20_1_X_2026-11-03-2300"
        rows = [_rec(primary, "primary", "el9_amd64_gcc14", ["539.0"]), _rec("CMSSW_20_1_ROOT6_X_2026-11-03-2300", "ROOT6", "el9_amd64_gcc14", ["539.0", "1.0"])]
        with patch.object(api_server, "_ib_build_records", side_effect=lambda t, *a, **k: rows if t == primary else rows[1:]) as records:
            answer = api_server._failure_sets_answer("how many workflows fail only in ROOT6 in CMSSW_20_1_ROOT6_X_2026-11-03-2300?")
        records.assert_called_with(primary)
        self.assertIn("`1.0`", answer)
        self.assertNotIn("`539.0`", answer)

    def test_a_named_sub_ib_stays_when_the_word_is_an_architecture(self):
        tag = "CMSSW_20_1_ROOT6_X_2026-11-03-2300"
        rows = [_rec(tag, "ROOT6", "el9_amd64_gcc14", ["1.0"]), _rec(tag, "ROOT6", "el9_aarch64_gcc14", ["1.0", "2.0"])]
        with patch.object(api_server, "_ib_build_records", return_value=rows) as records:
            answer = api_server._failure_sets_answer(f"what fails only on aarch64 in {tag}?")
        records.assert_called_with(tag)
        self.assertIn("`2.0`", answer)


class BuildMeansTheIbTests(unittest.TestCase):
    def test_what_failed_in_a_build_has_no_build_category(self):
        for q in ("what failed on el8_amd64_gcc14 in CMSSW_20_1_X_2026-10-30-1100 20_1 build?",
                  "what's failing on el10_amd64_gcc15 in the last 20_1 build it had?",
                  "what is failing in that build?"):
            self.assertNotIn("build", RequestContext.extract(q).categories, q)

    def test_questions_about_compilation_keep_it(self):
        for q in ("did the build fail in CMSSW_20_1_X_2026-10-30-1100?", "what failed in the build status of CMSSW_20_1_X_2026-10-30-1100?",
                  "build warnings in CMSSW_20_1_X_2026-10-30-1100", "what failed with warnings in the build?"):
            self.assertIn("build", RequestContext.extract(q).categories, q)


class BuildStatusKeptTests(unittest.TestCase):
    def test_import_clears_only_the_tags_in_the_file(self):
        import json, tempfile
        from pathlib import Path
        import build_graph
        session = MagicMock()
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CMSSW_20_1_X.json"
            path.write_text(json.dumps({"release_name": "CMSSW_20_1_X", "comparisons": [
                {"release_name": "CMSSW_20_1_X_2026-11-05-2300", "builds": [{"arch": "el9_amd64_gcc14", "passed": "passed", "details": {}}]}]}))
            with patch.object(build_graph, "source_needs_import", return_value=True), patch.object(build_graph, "mark_source_imported"):
                build_graph._ingest_build_status_file(driver, path, False)
        clear = session.run.call_args_list[0]
        self.assertIn("ib.tag IN $tags", clear.args[0])
        self.assertEqual(clear.kwargs["tags"], ["CMSSW_20_1_X_2026-11-05-2300"])


class OverviewWithoutBuildStatusTests(unittest.TestCase):
    def _answer(self, status):
        row = {"tag": "CMSSW_20_1_X_2026-10-30-1100", "arch": "el8_amd64_gcc14", "variant": "primary", "build_status": status, "failed_relvals": [],
               "known_errors": 0, "failed_unittests": [], "relval_total": 954, "das_errors": 0, "unittest_total": 377}
        session = MagicMock()
        session.run.return_value = [row]
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "get_exit_code_names", return_value={}):
            return api_server._failure_overview_answer(row["tag"], row["arch"])[0]

    def test_unset_status_is_not_called_clean(self):
        self.assertIn("no build status is imported", self._answer(None))
        self.assertNotIn("no build errors", self._answer(None))

    def test_a_status_is_still_reported(self):
        self.assertIn("no build errors", self._answer("passed"))


class BuildWarningsOfADayTests(unittest.TestCase):
    def test_day_wide_question_lists_the_builds_of_that_day(self):
        rows = [{"tag": "CMSSW_20_1_ROOT6_X_2026-11-04-2300", "variant": "ROOT6", "arch": "el9_amd64_gcc14", "status": "warning", "comp_warnings": 5},
                {"tag": "CMSSW_20_1_X_2026-11-04-2300", "variant": "primary", "arch": "el9_amd64_gcc14", "status": "passed", "comp_warnings": 0}]
        session = MagicMock()
        session.run.return_value = rows
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver):
            answer = api_server._build_warning_answer("are there builds with compilation warnings on 2026-11-04?", "CMSSW_20_1_X_2026-11-03-2300")
        self.assertIn("1 more built with compilation warnings", answer)
        self.assertIn("`CMSSW_20_1_ROOT6_X_2026-11-04-2300` on `el9_amd64_gcc14`: 5", answer)

    def test_the_date_is_not_turned_into_one_tag(self):
        question = "are there builds with compilation warnings on 2026-11-04?"
        self.assertEqual(api_server._date_tag_context(question), (question, None))


class TwoFlavoursTests(unittest.TestCase):
    TAGS = ["CMSSW_20_1_X_2026-11-03-2300", "CMSSW_20_1_X_2026-11-05-2300", "CMSSW_20_1_CLANG_X_2026-11-03-2300", "CMSSW_20_1_CLANG_X_2026-11-05-2300",
            "CMSSW_20_1_ASAN_X_2026-11-04-2300", "CMSSW_20_1_UBSAN_X_2026-11-04-2300"]

    def _rewrite(self, question):
        session = MagicMock()
        session.run.return_value = [{"tag": t} for t in self.TAGS]
        return api_server._with_newest_flavour_compare_tags(question, session, "CMSSW_20_1")

    def test_a_written_date_picks_that_build(self):
        self.assertEqual(self._rewrite("is CLANG better or worse than the primary in 2026-11-03 23:00?"),
                         "compare CMSSW_20_1_CLANG_X_2026-11-03-2300 and CMSSW_20_1_X_2026-11-03-2300")

    def test_two_flavours_same_relvals(self):
        self.assertEqual(self._rewrite("do UBSAN and ASAN fail the same relvals in 2026-11-04 23:00 on el9_amd64_gcc14?"),
                         "compare CMSSW_20_1_ASAN_X_2026-11-04-2300 and CMSSW_20_1_UBSAN_X_2026-11-04-2300 el9_amd64_gcc14")

    def test_no_build_of_that_date_leaves_the_question(self):
        question = "is CLANG better than the primary on 2026-11-04?"
        self.assertEqual(self._rewrite(question), question)

    def test_the_dated_flavour_step_leaves_two_builds_alone(self):
        self.assertIsNone(api_server._dated_flavour_question("do UBSAN and ASAN fail the same relvals in 2026-11-04 23:00?"))
        self.assertIsNone(api_server._dated_flavour_question("is CLANG better than the primary in 2026-11-03 23:00?"))

    def test_no_comparison_word_leaves_the_question(self):
        question = "what fails in CLANG on 2026-11-03?"
        self.assertEqual(self._rewrite(question), question)


class ContextAfterAiAnswerTests(unittest.TestCase):
    def _answer(self, sid, text, route="ai:llm_validated"):
        with patch.object(api_server, "_store_chat_turns"):
            api_server._chat_response(sid, "q", text, route)

    def setUp(self):
        self.sid = "ctx-test"
        api_server._latest_ib_contexts[self.sid] = "CMSSW_20_1_X_2026-11-03-1100"
        api_server._last_workflow_contexts[self.sid] = "537.0"
        self.addCleanup(api_server._latest_ib_contexts.pop, self.sid, None)
        self.addCleanup(api_server._last_workflow_contexts.pop, self.sid, None)
        self.addCleanup(api_server._latest_ib_arch_contexts.pop, self.sid, None)

    def test_read_as_moves_build_and_workflow(self):
        self._answer(self.sid, "_I read this as: why is workflow 539.0 failing in CMSSW_20_1_X_2026-11-02-2300?_\n\nBecause.")
        self.assertEqual(api_server._latest_ib_contexts[self.sid], "CMSSW_20_1_X_2026-11-02-2300")
        self.assertEqual(api_server._last_workflow_contexts[self.sid], "539.0")

    def test_a_build_in_the_first_paragraph_counts_when_the_question_named_none(self):
        self._answer(self.sid, "_I read this as: which build was PR 52375 first in?_\n\nIt first appears in CMSSW_20_1_X_2026-11-05-1100.\n\nAlso in CMSSW_20_1_X_2026-11-05-2300.")
        self.assertEqual(api_server._latest_ib_contexts[self.sid], "CMSSW_20_1_X_2026-11-05-1100")
        self.assertEqual(api_server._last_workflow_contexts[self.sid], "537.0")

    def test_two_builds_or_another_route_leave_it(self):
        self._answer(self.sid, "_I read this as: compare CMSSW_20_1_X_2026-11-04-2300 and CMSSW_20_1_X_2026-11-05-2300_\n\nText.")
        self.assertEqual(api_server._latest_ib_contexts[self.sid], "CMSSW_20_1_X_2026-11-03-1100")
        self._answer(self.sid, "_I read this as: CMSSW_20_1_X_2026-11-04-2300_\n\nText.", route="ai:llm_ungrounded_fallback")
        self.assertEqual(api_server._latest_ib_contexts[self.sid], "CMSSW_20_1_X_2026-11-03-1100")


class ThatBuildInAShiftSummaryTests(unittest.TestCase):
    def test_that_build_is_the_one_in_the_conversation(self):
        sid = "that-build"
        api_server._latest_ib_contexts[sid] = "CMSSW_20_1_X_2026-11-05-1100"
        self.addCleanup(api_server._latest_ib_contexts.pop, sid, None)
        with patch.object(api_server, "_previous_primary_tag", return_value="CMSSW_20_1_X_2026-11-04-2300"), \
                patch.object(api_server, "generate_shift_summary", return_value="digest") as digest, \
                patch.object(api_server, "_condense_shift_summary_for_chat", side_effect=lambda x: x):
            api_server._execute_planned_capability("shift_summary", "did that build get any new failures?", sid)
        digest.assert_called_once_with("CMSSW_20_1_X_2026-11-05-1100", "CMSSW_20_1_X_2026-11-04-2300", None)


class NoLiteralNoneTests(unittest.TestCase):
    def test_chat_shift_summary_has_no_none(self):
        digest = ("## Shift summary: A -> B\n\n### `el9_amd64_gcc14`\n\n#### Results not imported yet\n- RelVal results: none imported yet for `B` in the primary build "
                  "(the previous build had 3), so \"None\" below means not checked yet, not passing.\n\n#### Newly failing RelVal workflows\nNone.\n")
        answer = api_server._condense_shift_summary_for_chat(digest)
        self.assertNotIn("None", answer)
        self.assertIn("not checked yet", answer)

    def test_related_items_name_the_item_a_comment_is_on(self):
        session = MagicMock()
        session.run.return_value = [{"kind": "PR", "n": 52357, "title": "Hw double init fix"}]
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        api_server._last_issue_contexts["rel"] = "52323"
        api_server._turn_memory["rel"] = {"question": "Issue 52323", "route": "deterministic:issue_number", "answer": "## Issue #52323 - Periodic failure"}
        self.addCleanup(api_server._last_issue_contexts.pop, "rel", None)
        self.addCleanup(api_server._turn_memory.pop, "rel", None)
        with patch.object(api_server, "get_driver", return_value=driver):
            answer = api_server._listed_items_follow_up("related PRs?", "rel")
        self.assertIn("HAS_COMMENT", session.run.call_args.args[0])
        self.assertIn("**PR #52357**", answer)
        self.assertNotIn("None", answer)


class RulesAfterAFailedInvestigationTests(unittest.TestCase):
    def _run(self, investigated_route, rule_route):
        def response(route, answer="text"):
            return api_server.ChatResponse(answer=answer, session_id="s", route=route, citations=[], github_matches=None, actions=[])
        first = response(api_server._NEEDS_AI_ROUTE, "")
        req = api_server.ChatRequest(question="why?", session_id="s")
        with patch.object(api_server, "_ai_routed", return_value={"standalone": "why is 537.0 failing in CMSSW_20_1_X_2026-11-05-2300 on el9_amd64_gcc16?", "known": ""}), \
                patch.object(api_server, "_ai_investigated", return_value=response(investigated_route)), \
                patch.object(api_server, "_chat", return_value=response(rule_route, "Workflow `537.0` failed")) as chat, \
                patch.object(api_server, "_ai_judged", return_value={"fits": True, "problem": ""}):
            final, note = api_server._ai_supervise(req, first, "s", [])
        return final, chat

    def test_the_rules_answer_when_the_lookups_came_back_empty(self):
        final, chat = self._run("ai:llm_unverified_absence", "deterministic:workflow_why")
        self.assertEqual(final.route, "ai_known:deterministic:workflow_why")
        self.assertIn("537.0", chat.call_args.args[0].question)

    def test_a_validated_investigation_stands(self):
        final, chat = self._run("ai:llm_validated", "deterministic:workflow_why")
        self.assertEqual(final.route, "ai:llm_validated")
        chat.assert_not_called()

    def test_no_rule_keeps_the_investigation(self):
        final, _ = self._run("ai:llm_unverified_absence", api_server._NEEDS_AI_ROUTE)
        self.assertEqual(final.route, "ai:llm_unverified_absence")


class LlmAnswerGuardsTests(unittest.TestCase):
    def test_self_correction_is_recognised(self):
        self.assertTrue(api_server._SELF_CORRECTION_RE.search("8. el9_amd64_gcc15 \u2014 wait, that's the same as #4"))
        self.assertTrue(api_server._SELF_CORRECTION_RE.search("Sorry, I miscounted."))
        for text in ("the workflow is waiting for input", "That's the same build twice", "wait times are long", "await, result"):
            self.assertFalse(api_server._SELF_CORRECTION_RE.search(text), text)

    def test_stop_sentence_is_no_rewrite(self):
        self.assertFalse(api_server._usable_retry("(stopped after 8 tool calls)"))
        self.assertFalse(api_server._usable_retry("API error: x"))
        self.assertFalse(api_server._usable_retry(""))
        self.assertTrue(api_server._usable_retry("A rewrite."))


class WorkflowFixedTests(unittest.TestCase):
    def test_fixed_in_a_build_is_a_workflow_status_question(self):
        for q in ("did 539.0 get fixed in CMSSW_20_1_X_2026-11-05-1100?", "was 537.0 fixed in CMSSW_20_1_X_2026-11-05-1100", "has 539.0 recovered?"):
            self.assertTrue(api_server._WORKFLOW_WHY_RE.search(q), q)

    def test_other_questions_keep_their_routes(self):
        self.assertFalse(api_server._WORKFLOW_WHY_RE.search("how many relvals failed in CMSSW_20_1_X_2026-11-05-1100"))


class ArchShorthandAndSlotTests(unittest.TestCase):
    def test_short_arch_names(self):
        self.assertEqual(api_server._with_full_arch_name("relval failures yesterday 23:00 build el9 gcc14?"), "relval failures yesterday 23:00 build el9_amd64_gcc14?")
        self.assertEqual(api_server._with_full_arch_name("wf 537 on EL8 gcc 13 last result"), "wf 537 on el8_amd64_gcc13 last result".replace("el8", "EL8"))
        self.assertEqual(api_server._with_full_arch_name("el9_aarch64_gcc14 and gcc14 only"), "el9_aarch64_gcc14 and gcc14 only")

    def test_a_time_keeps_the_answer_to_that_slot_and_names_the_workflows(self):
        session = MagicMock()
        newest = MagicMock()
        newest.single.return_value = {"t": "CMSSW_20_1_X_2026-11-05-2300"}
        session.run.side_effect = [[{"tag": "CMSSW_20_1_X_2026-11-03-1100"}, {"tag": "CMSSW_20_1_X_2026-11-03-2300"}], newest]
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), \
                patch.object(api_server, "_ib_build_records", return_value=[_rec("CMSSW_20_1_X_2026-11-03-2300", "primary", "el9_amd64_gcc14", ["539.0", "537.0"])]) as records:
            answer = api_server._arch_on_day_answer("which relvals fail in the 2026-11-03 23:00 20_1 IB on el9_amd64_gcc14?")
        records.assert_called_once_with("CMSSW_20_1_X_2026-11-03-2300", None, "el9_amd64_gcc14")
        self.assertIn("`537.0`, `539.0`", answer)


class NewProblemsInLogsTests(unittest.TestCase):
    def test_unit_logs_are_a_test_log_question(self):
        for q in ("is there any problem new this week in the unit logs for 20_1?", "anything new in the addon logs?"):
            self.assertTrue(api_server._TEST_LOG_QUESTION_RE.search(q), q)

    def test_a_question_about_logs_is_not_about_dashboard_failures(self):
        self.assertIsNone(api_server._persistent_failures_answer("any new problems in the test logs of CMSSW_20_1_X_2026-11-05-2300 compared with the previous build?", "x"))


class SlotOfAnotherSeriesTests(unittest.TestCase):
    def test_other_series_of_that_slot_are_named(self):
        session = MagicMock()
        session.run.return_value = [{"tag": "CMSSW_16_1_X_2026-11-06-1100"}, {"tag": "CMSSW_17_0_X_2026-11-06-1100"}, {"tag": "CMSSW_16_1_X_2026-11-06-0900"}]
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        from datetime import datetime
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-05-2300"}]), \
                patch.object(api_server, "_question_day", return_value="2026-11-06"):
            _, message = api_server._date_tag_context("what failed in the 11:00 IB today?")
        self.assertIn("No `CMSSW_20_1` IB built on 2026-11-06 11:00", message)
        self.assertIn("`CMSSW_17_0_X_2026-11-06-1100`, `CMSSW_16_1_X_2026-11-06-1100`", message)
        self.assertNotIn("0900", message)


class FixDraftContextTests(unittest.TestCase):
    def test_package_of_the_last_answer(self):
        api_server._turn_memory["fx"] = {"question": "is SimG4Core/Geometry failing?", "route": "deterministic:package_status", "answer": "`SimG4Core/Geometry` unit tests, judged by"}
        self.addCleanup(api_server._turn_memory.pop, "fx", None)
        self.assertEqual(api_server._fix_draft_subject("suggest a fix", "fx"), ("SimG4Core/Geometry", None))

    def test_workflow_why_remembers_the_workflow(self):
        with patch.object(api_server, "_latest_ib_records", return_value=[]), patch.object(api_server, "get_driver", side_effect=RuntimeError("no db")):
            try:
                api_server._workflow_why_failing_answer("why is 539.0 timing out?", "fy")
            except RuntimeError:
                pass
        self.addCleanup(api_server._last_workflow_contexts.pop, "fy", None)
        self.assertEqual(api_server._last_workflow_contexts["fy"], "539.0")
        self.assertEqual(api_server._fix_draft_subject("what could the fix be?", "fy"), ("RelVal workflow 539.0", "539.0"))


class HowManyWasThatTests(unittest.TestCase):
    def test_counts_of_an_overview(self):
        answer = ("\U0001F534 What's failing in `CMSSW_20_1_X_2026-11-03-2300` on `el10_amd64_gcc15` -- primary IB:\n\n**Build** -- failed: `el10_amd64_gcc15`\n\n"
                  "**RelVals** -- 2 failing:\n- x\n\n**Unit tests** -- 5 failing:\n- y")
        out = api_server._counts_in_answer(answer)
        self.assertIn("**2** failing RelVal(s)", out)
        self.assertIn("**5** failing unit-test package(s)", out)
        self.assertIn("**1** failed build(s)", out)
        self.assertIn("`el10_amd64_gcc15`", out)

    def test_counts_of_a_list(self):
        out = api_server._counts_in_answer("**100 known-error RelVal(s)** in `CMSSW_20_1_X_2026-11-04-2300` on `el9_aarch64_gcc14` and its sub-IBs (excluded from failing counts), by reason:\n- a")
        self.assertTrue(out.startswith("**100 known-error RelVal(s)** in `CMSSW_20_1_X_2026-11-04-2300`"))

    def test_nothing_to_count(self):
        self.assertIsNone(api_server._counts_in_answer("Issue #1 is open."))


class ItemsMentioningPluralTests(unittest.TestCase):
    def test_words_are_also_tried_singular(self):
        session = MagicMock()
        session.run.side_effect = [[], [{"n": 52407, "title": "[RelVal] 539.0 timing out", "in_title": False, "state": "open", "pr_state": None, "merged": None}]]
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_github_export_note", return_value="note"):
            answer = api_server._items_mentioning_answer("which open issues mention RelVal timeouts?")
        self.assertEqual(session.run.call_args.kwargs["words"], ["relval", "timeout"])
        self.assertIn("52407", answer)


class DateSpanOfOneThingTests(unittest.TestCase):
    def test_workflow_between_dates_is_not_the_per_build_table(self):
        with patch.object(api_server, "get_driver", side_effect=AssertionError("no query expected")):
            self.assertIsNone(api_server._failures_between_answer("between 2026-10-27 and 2026-11-04, how often did workflow 539.0 fail on el9_amd64_gcc14?"))

    def test_workflow_count_over_a_span_is_not_the_last_30_days(self):
        with patch.object(api_server, "get_driver", side_effect=AssertionError("no query expected")):
            self.assertIsNone(api_server._workflow_failure_count_answer("between 2026-10-27 and 2026-11-04, how often did workflow 539.0 fail?"))

    def test_package_over_a_span_is_not_one_build(self):
        question = "how many primary CMSSW_20_1_X IBs from 2026-10-26 to 2026-11-04 had FWCore/Integration failing?"
        self.assertEqual(api_server._date_tag_context(question), (question, None))
        with patch.object(api_server, "get_driver", side_effect=AssertionError("no query expected")):
            self.assertIsNone(api_server._package_status_answer(question, "x"))


class FlavoursOfTheLastAnswerTests(unittest.TestCase):
    def test_read_from_its_table(self):
        api_server._turn_memory["ft"] = {"question": "was 573.0 failing on 2026-11-04?", "route": "deterministic:workflow_on_day", "answer":
                                         "**Yes.**\n\n| Build | Flavour | Architecture | Exit |\n| --- | --- | --- | --- |\n| `X` | ASAN | `el9_amd64_gcc15` | 1 |\n| `X` | ASAN | `el9_amd64_gcc14` | 1 |\n"}
        self.addCleanup(api_server._turn_memory.pop, "ft", None)
        self.assertEqual(api_server._listed_items_follow_up("in which flavours?", "ft"), "In that answer: **ASAN** on `el9_amd64_gcc15`, `el9_amd64_gcc14`.")


class LastGreenIsNotAboutFlavoursTests(unittest.TestCase):
    def test_flavours_of_a_build(self):
        with patch.object(api_server, "get_driver", side_effect=AssertionError("no query expected")):
            self.assertIsNone(api_server._last_green_answer("which flavours of the last complete 20_1 IB have no failing relvals?"))


class RelvalVersusUnitTestTests(unittest.TestCase):
    def test_the_difference_is_explained(self):
        for q in ("what's the difference between a RelVal failure and a unit test failure?", "what is the difference between a relval and a unit test?"):
            self.assertIn("RelVal", api_server._basic_term_answer(q), q)
            self.assertIn("Unit tests", api_server._basic_term_answer(q), q)


class CrashSignalPromptTests(unittest.TestCase):
    def test_the_prompt_says_where_signals_are(self):
        self.assertIn("62720 SIGSEGV", api_server.SYSTEM_PROMPT)


class LogOverviewRepeatedLinesTests(unittest.TestCase):
    def test_same_problem_in_several_signatures_is_one_line(self):
        rows = [{"summary": "Python IndexError: list index out of range", "owner": "513.0_W:step1", "tests": n, "arch": "el9_amd64_gcc14", "suite": "relval"} for n in (7, 5, 2)]
        session = MagicMock()
        session.run.side_effect = lambda *a, **k: rows
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-05-2300"}]), \
                patch.object(api_server, "_newest_log_stamp", return_value="2026-11-05-2300"):
            answer = api_server._log_overview_question_answer("any tracebacks in RelVal logs?")
        self.assertEqual(answer.count("IndexError"), 1, answer)
        self.assertIn("14 test(s)", answer)


class PointingBackTests(unittest.TestCase):
    def test_words(self):
        for q in ("did that build get any new failures?", "what changed between them?"):
            self.assertTrue(api_server._POINTS_BACK_RE.search(q), q)
        self.assertFalse(api_server._POINTS_BACK_RE.search("what changed between the last two builds?"))

    def test_no_context_is_not_the_newest_pair(self):
        self.assertIsNone(api_server._execute_planned_capability("shift_summary", "what changed between them?", "nobody"))


class FlavoursWhichFailedTests(unittest.TestCase):
    def test_listed_when_asked(self):
        records = [_rec("T", "primary", "el8_amd64_gcc14"), _rec("T", "ASAN", "el9_amd64_gcc14", ["1.0"], ["A/B"])]
        with patch.object(api_server, "_ib_build_records", return_value=records):
            answer = api_server._flavours_of_build_answer("how many sub-IBs does CMSSW_20_1_X_2026-11-04-2300 have and which failed something?")
            plain = api_server._flavours_of_build_answer("how many sub-IBs does CMSSW_20_1_X_2026-11-04-2300 have?")
        self.assertIn("**1 sub-IB flavour(s)**", answer)
        self.assertIn("- ASAN `el9_amd64_gcc14`: 1 RelVal(s), 1 unit test(s)", answer)
        self.assertNotIn("Failing something", plain)


class AfterAnAiAnswerTests(unittest.TestCase):
    def test_how_many_does_not_repeat_a_superseded_answer(self):
        sid = "sup"
        api_server._turn_memory[sid] = {"question": "list open issues by X", "route": "deterministic:issues_by_author", "answer": "**517 Issue(s)** by `X`, newest first:\n- a"}
        self.addCleanup(api_server._turn_memory.pop, sid, None)
        self.assertIn("517", api_server._listed_items_follow_up("how many?", sid) or "")
        with patch.object(api_server, "_store_chat_turns"):
            api_server._chat_response(sid, "list open issues by X", "X has 140 open issues.", "ai:llm_validated")
        self.assertNotIn("517", api_server._listed_items_follow_up("how many?", sid) or "")

    def test_raw_records_without_a_number_show_no_none(self):
        text = api_server._format_raw_evidence([{"tool": "query_graph", "arguments": {}, "result": [{"number": None, "title": None, "state": "open"}]}])
        self.assertNotIn("None", text)


if __name__ == "__main__":
    unittest.main()


class PrsMergedOnDayCountTests(unittest.TestCase):
    """D1 (2026-11-09): "how many PRs were merged on <day>?" gave the length of a list capped at 30 as the count."""

    def _driver(self, listed, per_repo):
        session = MagicMock()
        session.run.side_effect = lambda cypher, **kw: per_repo if "count(*)" in cypher else listed
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        return driver

    def test_count_is_the_full_count_not_the_capped_list(self):
        listed = [{"repo": "cmssw", "n": 52000 + i, "title": "t", "merged": "2026-11-02T08:00:00Z", "who": "x"} for i in range(30)]
        per_repo = [{"repo": "cmssw", "n": 39}, {"repo": "cmsdist", "n": 9}]
        with patch.object(api_server, "get_driver", return_value=self._driver(listed, per_repo)), patch.object(api_server, "_github_export_note", return_value="note"):
            answer = api_server._prs_merged_on_day_answer("how many PRs were merged on 2026-11-02?")
        self.assertIn("**48 PR(s)**", answer)
        self.assertIn("cmssw 39, cmsdist 9", answer)
        self.assertIn("the 30 most recent", answer)
        self.assertNotIn("**30 PR(s)**", answer)

    def test_a_short_day_lists_all_without_a_cap_note(self):
        listed = [{"repo": "cmssw", "n": 52001, "title": "t", "merged": "2026-10-25T08:00:00Z", "who": "x"}]
        with patch.object(api_server, "get_driver", return_value=self._driver(listed, [{"repo": "cmssw", "n": 1}])), patch.object(api_server, "_github_export_note", return_value="note"):
            answer = api_server._prs_merged_on_day_answer("number of PRs merged on 2026-10-25")
        self.assertIn("**1 PR(s)** merged on 2026-10-25 (UTC):", answer)
        self.assertNotIn("most recent", answer)


class SpanFailureCountTests(unittest.TestCase):
    """D2 (2026-11-09): a written date span about one package or workflow got one build's overview ("doesn't cover the period") or the default window."""

    def _driver(self, rows, seen):
        session = MagicMock()
        def run(cypher, **kw):
            seen.append((cypher, kw))
            return rows
        session.run.side_effect = run
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        return driver

    def test_package_span_counts_builds_in_the_span_with_filters(self):
        rows = [{"release": "CMSSW_20_1_X", "variant": "primary", "arch": "fc42_riscv64_gcc15", "tag": f"CMSSW_20_1_X_{s}", "state": st}
                for s, st in (("2026-10-29-1100", 2), ("2026-10-29-2300", 2), ("2026-11-03-1100", 0))]
        seen = []
        with patch.object(api_server, "get_driver", return_value=self._driver(rows, seen)):
            answer = api_server._span_failure_count_answer(
                "on fc42_riscv64_gcc15, how many primary CMSSW_20_1_X IBs from 2026-10-26 to 2026-11-05 had RecoTracker/LSTGeometry failing?")
        cypher, kw = seen[0]
        self.assertIn("RAN_UNITTEST", cypher)
        self.assertEqual((kw["subject"], kw["start"], kw["end"], kw["arch"], kw["variant"], kw["prefix"]),
                         ("RecoTracker/LSTGeometry", "2026-10-26", "2026-11-05", "fc42_riscv64_gcc15", "primary", "CMSSW_20_1_"))
        self.assertIn("**2 of 3**", answer)
        self.assertIn("2026-10-29-1100, 2026-10-29-2300", answer)
        self.assertNotIn("None", answer)

    def test_workflow_span_uses_the_written_dates_and_keeps_known_errors_apart(self):
        rows = [{"release": "CMSSW_20_1_X", "variant": "primary", "arch": "el8_amd64_gcc14", "tag": f"CMSSW_20_1_X_2026-11-0{d}-2300", "state": st}
                for d, st in ((2, 2), (3, 1), (4, 0))]
        seen = []
        with patch.object(api_server, "get_driver", return_value=self._driver(rows, seen)):
            answer = api_server._span_failure_count_answer(
                "between 2026-11-05 and 2026-10-27, how often did workflow 539.0 fail on el8_amd64_gcc14 in the primary 20_1 IBs?")
        cypher, kw = seen[0]
        self.assertIn("RAN_RELVAL", cypher)
        self.assertEqual((kw["subject"], kw["start"], kw["end"]), ("539.0", "2026-10-27", "2026-11-05"))
        self.assertIn("**1 of 3**", answer)
        self.assertIn("plus 1 as a known error", answer)

    def test_leaves_other_questions_alone(self):
        with patch.object(api_server, "get_driver", side_effect=AssertionError("no query expected")):
            for q in ("how often did workflow 539.0 fail this week?",  # no written span: the 30/7-day answer
                      "how many PRs from 2026-10-26 to 2026-11-05 mention 539.0 failing?",
                      "from 2026-10-26 to 2026-11-05, how many builds had 539.0 and 537.0 failing?",
                      "which builds from 2026-10-26 to 2026-11-05 had 539.0 failing?",
                      "how many ASAN and UBSAN builds from 2026-10-26 to 2026-11-05 had 573.0 failing?"):
                self.assertIsNone(api_server._span_failure_count_answer(q), q)

    def test_empty_scope_says_so_without_zero(self):
        with patch.object(api_server, "get_driver", return_value=self._driver([], [])):
            answer = api_server._span_failure_count_answer("how many times did 1.0 fail on el9_amd64_gcc14 between 2026-09-01 and 2026-09-05?")
        self.assertIn("No imported result", answer)
        self.assertNotIn("0 of", answer)


class ExitCodeOfWorkflowInTagTests(unittest.TestCase):
    """D2 (2026-11-09): "exit code of workflow X in <tag> <arch>" was rewritten without the tag, so it got the workflow's history."""

    def test_the_named_build_stays(self):
        rewritten = api_server._everyday_wording_rewrite("exit code of workflow 4.17 in CMSSW_13_2_X_2026-11-02-2300 el9_amd64_gcc11", "s")
        self.assertEqual(rewritten, "why is workflow 4.17 failing in CMSSW_13_2_X_2026-11-02-2300 on el9_amd64_gcc11")

    def test_without_a_tag_unchanged(self):
        self.assertEqual(api_server._everyday_wording_rewrite("exit code of 539.0 on el9_amd64_gcc14?", "s"), "why is workflow 539.0 failing on el9_amd64_gcc14")
        self.assertEqual(api_server._everyday_wording_rewrite("exit codes for workflow 539.0", "s"), "why is workflow 539.0 failing")

    def test_a_build_that_does_not_exist_is_said_so(self):
        session = MagicMock()
        tags = [{"t": "CMSSW_20_1_X_2026-11-07-1100"}, {"t": "CMSSW_20_1_X_2026-11-08-0000"}, {"t": "CMSSW_20_1_X_2026-11-06-2300"}]
        session.run.side_effect = lambda cypher, **kw: tags if "DISTINCT ib.tag AS t" in cypher else []
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver):
            answer = api_server._workflow_why_failing_answer("why is workflow 539.0 failing in CMSSW_20_1_X_2026-11-07-2300 on el9_aarch64_gcc14", "s-nx")
        self.assertTrue(answer.startswith("No IB `CMSSW_20_1_X_2026-11-07-2300` is imported"), answer)
        self.assertIn("`CMSSW_20_1_X_2026-11-07-1100`, `CMSSW_20_1_X_2026-11-08-0000`", answer)


class SpanFailureCountScopeTests(unittest.TestCase):
    """D2 (2026-11-09): with one failed build the span answer named one date, and the scope check sent it to the AI as "one build's answer"."""

    def test_one_failure_is_not_read_as_one_build(self):
        rows = [{"release": "CMSSW_20_1_X", "variant": "primary", "arch": "el8_amd64_gcc14", "tag": f"CMSSW_20_1_X_{s}", "state": st}
                for s, st in (("2026-10-27-1100", 0), ("2026-11-02-1100", 2), ("2026-11-05-2300", 0))]
        session = MagicMock()
        session.run.return_value = rows
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        q = "how many times did 140.111 fail on el8_amd64_gcc14 in CMSSW_20_1_X primary IBs between 2026-10-27 and 2026-11-05?"
        with patch.object(api_server, "get_driver", return_value=driver):
            answer = api_server._span_failure_count_answer(q)
        self.assertIn("**1 of 3**", answer)
        self.assertIn("builds 2026-10-27-1100 to 2026-11-05-2300", answer)
        self.assertIsNone(api_server._scope_mismatch(q, answer))


class GithubMilestoneReplacesOlderTests(unittest.TestCase):
    """E1-1: PR 51516 kept release CMSSW_17_0_X from an older import while its milestone is CMSSW_17_1_X."""
    def test_newer_milestone_replaces_an_older_github_value(self):
        import build_graph
        session = MagicMock()
        session.run.return_value = []
        rec = {"source_id": "s", "content_hash": "h", "number": 51516, "release": "CMSSW_17_1_X", "label_names": []}
        build_graph._import_raw_pr_batch(session, [rec], True)
        query = " ".join(c.args[0] for c in session.run.call_args_list)
        self.assertIn("pr.release=CASE WHEN pr.source_id IS NULL THEN coalesce(raw.release, pr.release) "
                      "ELSE coalesce(pr.release, raw.release) END", query)
        self.assertNotIn("pr.release=coalesce(pr.release, raw.release)", query)


class PrOriginOfAnIssueTests(unittest.TestCase):
    """E1-2: "who opened 52093 and what is it about" -> the AI's pr_origin said not found, answer "no data at all"; 52093 is an Issue."""
    def test_issue_number_is_reported_as_an_issue(self):
        from graph import analytics
        issue = {"title": "SQL thing", "state": "open", "created_at": "2026-10-06", "closed_at": None, "labels": "[]",
                 "opened_by": ["someone"], "comments_imported": 4}
        with patch.object(analytics, "_rows", side_effect=[[], [issue]]):
            out = analytics.pr_origin(52093)
        self.assertFalse(out["found"])
        self.assertTrue(out["is_issue"])
        self.assertEqual(out["issue"]["opened_by"], ["someone"])
        self.assertIn("is an Issue, not a PR", out["note"])

    def test_unknown_number_stays_not_found(self):
        from graph import analytics
        with patch.object(analytics, "_rows", side_effect=[[], []]):
            self.assertEqual(analytics.pr_origin(99999), {"found": False, "repo": "cmssw", "number": 99999})


class CommentCapTests(unittest.TestCase):
    """E1-3: threads cut at 100 comments by the backup: "PR #47447 has 100 imported comments" (153 on GitHub)."""
    def test_note_only_at_the_cap(self):
        self.assertEqual(api_server._comment_cap_note(99), "")
        self.assertIn("first 100 comments", api_server._comment_cap_note(100))

    def test_issue_comments_answer_says_newest_imported(self):
        session = MagicMock()
        issue = MagicMock(); issue.single.return_value = {"title": "T", "via": None}
        rows = [{"author": f"u{i % 3}", "at": f"2026-10-{1 + i % 28:02d}", "body": "x"} for i in range(100)]
        session.run.side_effect = [issue, rows]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver):
            answer = api_server._issue_comments_answer("who commented last on issue 51060?")
        self.assertIn("100 imported comment(s)", answer)
        self.assertIn("may have more", answer)
        self.assertIn("Newest imported first (not the newest on GitHub)", answer)

    def test_schema_tells_the_ai(self):
        from graph import tools
        self.assertIn("first 100 comments of a thread", tools.GRAPH_SCHEMA)


class CoverageNoteCountsPrsAndCommentsTests(unittest.TestCase):
    """E1-4: the note said "full export (with comments) reaches 2026-11-05" for the 2026-11-06 backup (Issues updated last were web copies)."""
    def test_query_uses_prs_and_comments_and_the_date_is_shown(self):
        session = MagicMock()
        first = MagicMock(); first.single.return_value = {"n": 52402, "t": "2026-11-06T08:55:12Z"}
        web = MagicMock(); web.single.return_value = {"n": None, "t": None, "via": None}
        session.run.side_effect = [first, web]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        api_server._GITHUB_COVERAGE_CACHE.clear()
        try:
            with patch.object(api_server, "get_driver", return_value=driver):
                note = api_server._github_export_note()
        finally:
            api_server._GITHUB_COVERAGE_CACHE.clear()
        query = session.run.call_args_list[0].args[0]
        self.assertIn("MATCH (p:PR {repo:'cmssw'}) WHERE p.fetched_via IS NULL", query)
        self.assertIn("max(c.created_at)", query)
        self.assertIn("2026-11-06", note)


class GithubWindowPastReachTests(unittest.TestCase):
    """E1-5: "any PRs opened today?" -> "No PRs were opened today" with GitHub data ending 2026-11-06."""
    def setUp(self):
        from graph import analytics
        analytics._GITHUB_REACH.clear()

    def test_window_after_the_backup_gets_the_note(self):
        from graph import analytics
        with patch.object(analytics, "_rows", side_effect=[[], [{"tc": "2026-11-06T08:55:12Z", "tp": "2026-11-06T08:24:13Z"}]]):
            out = analytics.search_github(kind="pr", since="2026-11-09", count_only="true")
        self.assertEqual(out["total"], 0)
        self.assertIn("reaches 2026-11-06", out["note"])
        self.assertIn("does not mean none", out["note"])

    def test_window_inside_the_backup_has_no_note(self):
        from graph import analytics
        with patch.object(analytics, "_rows", side_effect=[[], [{"tc": "2026-11-06T08:55:12Z", "tp": None}]]):
            out = analytics.search_github(kind="pr", since="2026-10-01", until="2026-10-31", count_only="true")
        self.assertNotIn("reaches", out["note"])


class PrReleaseTwoFormsTests(unittest.TestCase):
    """E1-6: "PRs merged for CMSSW_15_0_X since September" -> 4, truth 5: #52309 has release 'CMSSW_15_0' (IB pages), the others 'CMSSW_15_0_X'."""
    def test_schema_tells_the_ai_both_forms(self):
        from graph import tools
        self.assertIn("p.release IN ['CMSSW_15_0', 'CMSSW_15_0_X']", tools.GRAPH_SCHEMA)

    def test_merged_window_matches_both_forms(self):
        session = MagicMock(); session.run.return_value = []
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver):
            api_server._prs_merged_window_answer("PRs merged into 15_0 this week")
        query = session.run.call_args_list[0].args[0]
        self.assertIn("pr.release IN [$release, $release + '_X']", query)
        self.assertEqual(session.run.call_args_list[0].kwargs["release"], "CMSSW_15_0")


class SignaturesFromCommentsTests(unittest.TestCase):
    """E1-7: "who signed 51979 for dqm and pdmv?" -> "the imported data doesn't include who reviewed a PR"; the +dqm/+pdmv comments are imported."""
    def _driver(self, item, rows):
        first = MagicMock(); first.single.return_value = item
        session = MagicMock(); session.run.side_effect = [first, rows]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        return driver

    def test_lists_people_signatures_not_bot_or_plain_comments(self):
        item = {"kind": "PR", "title": "PF DQM", "labels": '["hlt-pending", "dqm-approved", "ngt"]', "comments": 49, "via": None}
        rows = [{"author": "gabrielmscampos", "at": "2026-10-16T12:30:18Z", "body": "+dqm"},
                {"author": "someone", "at": "2026-10-17T00:00:00Z", "body": "please test"},
                {"author": "teteumac", "at": "2026-10-27T08:51:45Z", "body": "+pdmv\\n\\nthanks"},
                {"author": "x", "at": "2026-10-28T00:00:00Z", "body": "+code-checks"}]
        with patch.object(api_server, "get_driver", return_value=self._driver(item, rows)):
            answer = api_server._signatures_answer("who signed 51979 for dqm and pdmv?")
        self.assertIn("`+dqm` by `gabrielmscampos`, 2026-10-16", answer)
        self.assertIn("`+pdmv` by `teteumac`, 2026-10-27", answer)
        self.assertNotIn("please test", answer)
        self.assertNotIn("code-checks", answer)
        self.assertIn("hlt-pending, dqm-approved", answer)
        self.assertIn("Still pending (by the labels): hlt.", answer)

    def test_without_a_number_the_not_imported_answer_stays(self):
        self.assertIsNone(api_server._signatures_answer("who reviewed it?"))
        self.assertIsNotNone(api_server._not_imported_answer("who reviewed it?"))


class BareNumberMayBeAnIssueSchemaTests(unittest.TestCase):
    """E1-8: "did 52357 fix the herwig problem in 52323?" -> "PR 52323 has no record in the graph at all" (52323 is an Issue)."""
    def test_schema_says_numbers_are_shared(self):
        from graph import tools
        self.assertIn("a bare number ('52323') may be either", tools.GRAPH_SCHEMA)


class QuestionAboutAnIssueIsNotAPasteTests(unittest.TestCase):
    """E1-9: "issue 47194 heap-buffer-overflow, how was it fixed?" went to the pasted-error search ("Not found on GitHub")."""
    def test_named_issue_question_is_left_to_routing(self):
        self.assertIsNone(api_server._pasted_exception_response("s", "q", "issue 47194 heap-buffer-overflow, how was it fixed?"))

    def test_a_real_sanitizer_paste_is_still_a_paste(self):
        with patch.object(api_server, "_github_reports_section", return_value=None), \
             patch.object(api_server, "_test_log_problem_answer", return_value="log answer"), \
             patch.object(api_server, "_referenced_items_note", return_value=None), \
             patch.object(api_server, "_pasted_error_evidence_answer", return_value=None), \
             patch.object(api_server, "_chat_response", side_effect=lambda sid, q, text, route: (text, route)):
            out = api_server._pasted_exception_response("s", "q", "==123==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1 in Foo::bar")
        self.assertEqual(out, ("log answer", "deterministic:test_log_problem"))


class BackportOfWhichTests(unittest.TestCase):
    """E1-10: "PR 36091 is a backport of which PR?" -> "I found no backport" (title "(packport of #35984)")."""
    def _driver(self, source, rows):
        first = MagicMock(); first.single.return_value = source
        session = MagicMock(); session.run.side_effect = [first, rows]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        return driver

    def test_names_the_originals_from_title_and_body(self):
        source = {"title": "[14_0_X] Backport of 44970 and 45820 (ScoutingNano event content changes for 2024)", "body": "backport of #44970 and https://github.com/cms-sw/cmssw/pull/45820",
                  "release": "CMSSW_14_0_X", "merged": "2024-09-16T09:14:06Z"}
        rows = [{"n": 44970, "title": "Scouting A", "release": "CMSSW_14_1_X", "merged": "2024-05-01T00:00:00Z", "state": "closed"},
                {"n": 45820, "title": "Scouting B", "release": "CMSSW_14_2_X", "merged": None, "state": "closed"}]
        with patch.object(api_server, "get_driver", return_value=self._driver(source, rows)):
            answer = api_server._backport_answer("PR 45950 backports which PRs?")
        self.assertIn("**PR #44970** `CMSSW_14_1_X` - Scouting A (merged 2024-05-01)", answer)
        self.assertIn("**PR #45820**", answer)
        self.assertNotIn("#2024", answer)
        self.assertNotIn("no backport", answer)

    def test_typo_packport_in_title(self):
        source = {"title": "fix memory bug in DNN-based ID (packport of #35984)", "body": "", "release": "CMSSW_12_1_X", "merged": "2021-11-12T13:26:54Z"}
        rows = [{"n": 35984, "title": "fix memory bug", "release": "CMSSW_12_2_X", "merged": "2021-11-11T00:00:00Z", "state": "closed"}]
        with patch.object(api_server, "get_driver", return_value=self._driver(source, rows)):
            answer = api_server._backport_answer("PR 36091 is a backport of which PR?")
        self.assertIn("**PR #35984**", answer)

    def test_other_direction_unchanged(self):
        self.assertIsNone(api_server._BACKPORT_OF_WHICH_RE.search("is there a backport of 52372?"))

    def test_original_pr_wording_reaches_it(self):
        source = {"title": "[13_1_X] XrdAdaptor: Use inactive sources first", "body": "backport of https://github.com/cms-sw/cmssw/pull/47593\nand #47960",
                  "release": "CMSSW_13_1_X", "merged": None}
        with patch.object(api_server, "get_driver", return_value=self._driver(source, [])):
            answer = api_server._backport_answer("what is the original PR of 52314?")
        self.assertIn("**#47593**", answer)
        self.assertIn("**#47960**", answer)

    def test_is_it_a_backport_with_a_typo_in_the_title(self):
        title = {"title": "fix memory bug in DNN-based ID (packport of #35984)", "release": "CMSSW_12_1_X", "merged": "2021-11-12T13:26:54Z"}
        first = MagicMock(); first.single.return_value = title
        session = MagicMock(); session.run.side_effect = [first, []]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), \
             patch.object(api_server, "_backport_source_answer", return_value="names #35984") as src:
            answer = api_server._backport_answer("is 36091 a backport?")
        self.assertEqual(answer, "**Yes.** names #35984")
        src.assert_called_once_with(36091)


class IssueTopicTests(unittest.TestCase):
    """E1-11: "is there an issue about python 3.14?" (read as workflow 3.14), "any issue about GCC15 build errors? is it fixed?" (build status),
    "were there issues about DAS errors in relvals recently?" (one IB's DAS errors)."""
    def _run(self, question, first_rows, second_rows=None):
        session = MagicMock(); session.run.side_effect = [first_rows, second_rows or []]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_github_export_note", return_value="note"):
            return api_server._issue_topic_answer(question), session

    def test_version_number_is_text(self):
        row = {"n": 52376, "title": "Moving to Python 3.14 as default Python in 20.1.X", "state": "open", "created": "2026-11-03T10:00:00Z", "closed": None, "in_title": True}
        answer, session = self._run("is there an issue about python 3.14?", [row])
        self.assertIn("**Issue #52376**", answer)
        self.assertEqual(session.run.call_args_list[0].kwargs["t"], "python 3.14")

    def test_words_and_time_word_and_state(self):
        row = {"n": 51970, "title": "GCC15 and GCC16 build errors", "state": "closed", "created": "2026-09-25T00:00:00Z", "closed": "2026-11-03T09:52:48Z", "in_title": True}
        answer, session = self._run("any issue about GCC15 build errors? is it fixed?", [], [row])
        self.assertEqual(session.run.call_args_list[1].kwargs["words"], ["gcc15", "build", "err"])
        self.assertIn("(`closed`; opened 2026-09-25, closed 2026-11-03)", answer)
        body_only = {"n": 50000, "title": "Other", "state": "open", "created": "2026-01-01T00:00:00Z", "closed": None, "in_title": False}
        answer, _ = self._run("any issue about GCC15 build errors?", [], [row, body_only])
        self.assertIn("(1 more only mention it in the description)", answer)
        self.assertNotIn("#50000", answer)
        answer, session = self._run("were there issues about DAS errors in relvals recently?", [], [])
        self.assertEqual(session.run.call_args_list[1].kwargs["words"], ["das", "err", "relval"])
        self.assertIn("updated_at, x.created_at) >= '", session.run.call_args_list[1].args[0])

    def test_workflow_number_left_to_the_reported_search(self):
        self.assertIsNone(api_server._issue_topic_answer("is there an issue about 539.0 timing out?"))
        self.assertIsNone(api_server._issue_topic_answer("which relvals fail in 539.0?"))


class FixForTopicTests(unittest.TestCase):
    """E1-12: "is there a PR for the TICLCandidateProducer vector bug?" listed every PR naming TICLCandidateProducer and missed Issue #51952,
    whose comments say "#51916 fixed the issue"; "is there a ROOT fix for the PoolSource slowdown with ROOT 6.40?" missed the ROOT PR a
    comment links."""
    def _run(self, question, issues, items=(), cmsdist=()):
        session = MagicMock(); session.run.side_effect = [issues, list(cmsdist), list(items)]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_github_export_note", return_value="note"):
            return api_server._fix_for_topic_answer(question), session

    def test_pr_named_in_issue_comments(self):
        issue = {"n": 51952, "title": "TICLCandidateProducer vector access/alloc bug", "state": "closed", "created": "2026-09-23T10:00:00Z",
                 "closed": "2026-09-30T09:41:09Z", "body": "crash in the producer", "total": 9,
                 "comments": [{"author": "Moanwar", "at": "2026-09-23T14:00:00Z", "body": "I think this solved here : https://github.com/cms-sw/cmssw/pull/51916 "},
                              {"author": "gabrielribcesario", "at": "2026-09-30T09:00:00Z", "body": "Yes, #51916 fixed the issue."}]}
        pr = {"n": 51916, "title": "Keep linked tracksters parallel", "pr": True, "merged": "2026-09-23T19:07:34Z", "state": "closed"}
        session = MagicMock(); session.run.side_effect = [[issue], [pr]]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_github_export_note", return_value="note"):
            answer = api_server._fix_for_topic_answer("is there a PR for the TICLCandidateProducer vector bug?")
        self.assertEqual(session.run.call_args_list[0].kwargs["words"], ["ticlcandidateproducer", "vector"])
        self.assertIn("**Issue #51952**", answer)
        self.assertIn("**PR #51916** - Keep linked tracksters parallel (merged 2026-09-23)", answer)
        self.assertIn("`gabrielribcesario` (2026-09-30)", answer)
        self.assertEqual(answer.count("**PR #51916**"), 1)

    def test_problem_word_dropped_and_external_pr(self):
        issue = {"n": 52394, "title": "regression in PoolSource performance with ROOT 6.40", "state": "open", "created": "2026-11-04T10:00:00Z",
                 "closed": None, "body": "", "total": 100,
                 "comments": [{"author": "pcanal", "at": "2026-11-05T19:25:00Z", "body": "The solution PR is available at https://github.com/root-project/root/pull/24009"}]}
        session = MagicMock(); session.run.side_effect = [[issue]]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_github_export_note", return_value="note"):
            answer = api_server._fix_for_topic_answer("is there a ROOT fix for the PoolSource slowdown with ROOT 6.40?")
        self.assertEqual(session.run.call_args_list[0].kwargs["words"], ["poolsource", "root", "6.40"])
        self.assertIn("`root-project/root#24009` (not in the imported data), linked by `pcanal` (2026-11-05)", answer)
        self.assertIn("only the first 100 comments", answer)

    def test_left_to_other_answers(self):
        for q in ("is there a fix for memory?", "which PR fixed issue 52086?", "is there a PR for 539.0 failing?", "what is the latest IB?"):
            self.assertIsNone(api_server._fix_for_topic_answer(q), q)

    def test_no_issue_falls_through(self):
        session = MagicMock(); session.run.side_effect = [[]]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver):
            self.assertIsNone(api_server._fix_for_topic_answer("is there a PR for the HGCal geometry crash?"))


class PrMergedNowNoteTests(unittest.TestCase):
    """E1-13: "is PR 51979 merged now?" -> "Open, not merged (per the imported GitHub data)" with no date, asked three days after the backup."""
    def _card(self, question, merged_at=None):
        row = {"repo": "cmssw", "pr_number": 51979, "title": "Particle Flow DQM at HLT.", "ib_tags": [], "body": "text", "merged_at": merged_at,
               "state": "open" if not merged_at else "closed", "release": "CMSSW_20_1_X", "labels": "[]", "additions": None, "deletions": None,
               "changed_files": None, "fetched_via": None, "author": "bfonta", "comment_count": 3, "comments": [], "references": [], "comment_references": []}
        session = MagicMock(); session.run.return_value = [row]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_github_export_note", return_value="GitHub data: reaches 2026-11-06"):
            return api_server._pr_number_answer(question)[0]

    def test_now_gets_the_data_date(self):
        answer = self._card("is PR 51979 merged now?")
        self.assertIn("- **Data as of:** GitHub data: reaches 2026-11-06; its state on GitHub today may differ.", answer)

    def test_plain_status_and_merged_unchanged(self):
        self.assertNotIn("Data as of", self._card("status of PR 51979"))
        self.assertNotIn("Data as of", self._card("is PR 51979 merged now?", merged_at="2026-11-07T10:00:00Z"))


class IssueLinksTests(unittest.TestCase):
    """E1-14: "what PRs are linked to issue 52323?" -> only #52357 (which points at the Issue); the PRs the Issue's own comments name were missed."""
    def _run(self, question, issue, links):
        result = MagicMock(); result.single.return_value = issue
        session = MagicMock(); session.run.side_effect = [result, links]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_github_export_note", return_value="note"):
            return api_server._issue_links_answer(question)

    def test_both_directions(self):
        issue = {"title": "Periodic failure of Herwig7GeneratorFilter in RelVal 537.0", "state": "open", "created": "2026-10-28T20:47:20Z", "closed": None}
        base = {"repo": "cmssw", "pr": True, "state": "closed"}
        links = [
            {**base, "n": 52303, "title": "Propagate exception", "merged": "2026-11-02T08:13:36Z", "how": "its comments", "who": "Dr15Jones",
             "at": "2026-10-28T21:00:00Z", "body": "The pull request #52303 turns the segmentation fault into an exception"},
            {**base, "n": 52357, "title": "Hw double init fix", "merged": None, "state": "open", "how": "mentions it in its description", "who": None, "at": None, "body": None},
            {**base, "n": 52357, "title": "Hw double init fix", "merged": None, "state": "open", "how": "mentions it in its comments", "who": "Dr15Jones", "at": "2026-10-30T00:00:00Z", "body": None},
            {"repo": "cmssw", "pr": False, "n": 52354, "title": "Memory problem", "merged": None, "state": "open", "how": "mentions it in its description", "who": None, "at": None, "body": None},
        ]
        answer = self._run("what PRs are linked to issue 52323?", issue, links)
        self.assertIn("**2 linked PR(s)**", answer)
        self.assertIn("**PR #52303** - Propagate exception (merged 2026-11-02): named in this Issue's comments by `Dr15Jones`", answer)
        self.assertIn("**PR #52357** - Hw double init fix (open, not merged): mentions it in its description; mentions it in its comments (`Dr15Jones`)", answer)
        self.assertIn("`Dr15Jones` (2026-10-28): \"...The pull request #52303", answer)
        self.assertIn("**Issue #52354**", answer)
        self.assertLess(answer.index("#52357"), answer.index("#52303"))

    def test_other_questions_left_alone(self):
        for q in ("who mentioned issue 52323?", "is issue 52323 still open?", "what did makortel comment on issue 52323?", "what PRs are linked to PR 52357?"):
            self.assertIsNone(api_server._issue_links_answer(q), q)


class IssueAskedAsPrNowNoteTests(unittest.TestCase):
    """E1-13: "is PR 52360 still open?" for an Issue: its card without the data date."""
    def test_note_added(self):
        session = MagicMock(); session.run.return_value = []
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_issue_summary", return_value="## Issue #52360 - x\\n- **State:** open"), \
                patch.object(api_server, "_github_export_note", return_value="GitHub data: reaches 2026-11-06"):
            answer = api_server._pr_number_answer("is PR 52360 still open?")[0]
            normalized = api_server._pr_number_answer("is PR 52360 open?")[0]
            plain = api_server._pr_number_answer("tell me about PR 52360")[0]
        self.assertIn("_GitHub data: reaches 2026-11-06; its state on GitHub today may differ._", answer)
        self.assertNotIn("may differ", plain)
        self.assertIn("may differ", normalized)


class BareNumberStateTests(unittest.TestCase):
    """E1-13: "is 52357 still open?" without the word PR: answered by the AI with no date of the data."""
    def test_bare_number_state_question_is_a_number_lookup(self):
        session = MagicMock(); session.run.return_value = []
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_issue_summary", return_value=None), \
                patch.object(api_server, "_github_export_note", return_value="note"):
            answer = api_server._pr_number_answer("is 52357 still open?")[0]
        self.assertIn("#52357 is not in the imported GitHub data", answer)
        self.assertEqual(session.run.call_args.kwargs["number"], 52357)
        self.assertIsNone(api_server._pr_number_answer("is 539.0 still failing?"))
        self.assertIsNone(api_server._pr_number_answer("is 12345 failing in the latest IB?"))


class TerseStateQuestionTests(unittest.TestCase):
    """E1-13: "is 52323 closed?" was rewritten to "PR 52323": the state question and the data date were lost."""
    def test_state_word_kept(self):
        with patch.object(api_server, "get_exit_code_names", return_value={}):
            self.assertEqual(api_server._terse_question("is 52323 closed?"), "is PR 52323 closed?")
            self.assertEqual(api_server._terse_question("52357 merged"), "is PR 52357 merged?")
            self.assertEqual(api_server._terse_question("#52357"), "PR 52357")
            self.assertEqual(api_server._terse_question("52357 target"), "PR 52357")


class NewCommentsSinceTests(unittest.TestCase):
    """E1-15: "any new comments on 52394 since friday?" asked on Monday 11-09: "Nothing after that ... no activity since", while the GitHub
    data ends with the 2026-11-06 backup (GitHub had two newer comments)."""
    def _run(self, question, comments, now):
        item = MagicMock(); item.single.return_value = {"pr": False, "title": "regression in PoolSource", "repo": "cmssw", "via": None, "total": len(comments)}
        session = MagicMock(); session.run.side_effect = [item, comments]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session

        class Now(datetime):
            @classmethod
            def now(cls, tz=None):
                return now
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_github_data_reach", return_value="2026-11-06"), \
                patch.object(api_server, "datetime", Now):
            return api_server._new_comments_answer(question)

    def test_since_weekday_and_reach(self):
        comments = [{"author": "fwyzard", "at": "2026-11-06T07:57:14Z", "body": "backport to 6.40?"},
                    {"author": "pcanal", "at": "2026-11-05T19:25:00Z", "body": "solution PR"}]
        answer = self._run("any new comments on 52394 since friday?", comments, datetime(2026, 11, 9, 14, 0, tzinfo=timezone.utc))
        self.assertIn("**1 imported comment(s) since 2026-11-06**", answer)
        self.assertIn("`fwyzard`", answer)
        self.assertNotIn("pcanal", answer)
        self.assertIn("**Our GitHub data reaches 2026-11-06**", answer)

    def test_none_since_says_where_data_ends(self):
        comments = [{"author": "pcanal", "at": "2026-11-05T19:25:00Z", "body": "solution PR"}]
        answer = self._run("any comments on 52394 since 2026-11-07?", comments, datetime(2026, 11, 9, tzinfo=timezone.utc))
        self.assertIn("**0 imported comment(s) since 2026-11-07**.", answer)
        self.assertIn("The newest imported comment is from 2026-11-05", answer)
        self.assertIn("can't be told from here", answer)

    def test_other_questions_left_alone(self):
        for q in ("what is the latest comment on 52323?", "who commented on 52376?", "any new failures since friday?", "any new comments on 539.0?"):
            self.assertIsNone(api_server._new_comments_answer(q), q)


class PersonCommentsTests(unittest.TestCase):
    """E1-16: "what did civanch say in issue 51505?" re-read by the AI supervisor as "issue 51505" (the card, no comment); "what did makortel
    comment on 52402?" came back as raw records."""
    def _run(self, question, comments, total=None):
        item = MagicMock(); item.single.return_value = {"pr": False, "title": "Keep Test Beam 2006", "repo": "cmssw", "total": total or len(comments)}
        session = MagicMock(); session.run.side_effect = [item, comments]
        driver = MagicMock(); driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_github_export_note", return_value="note"):
            return api_server._person_comments_answer(question)

    def test_person_and_topic(self):
        comments = [{"author": "makortel", "at": "2026-07-15T10:00:00Z", "body": "opening"},
                    {"author": "civanch", "at": "2026-09-14T21:14:00Z", "body": "we are not ready to drop 17_X validation of Geant4"},
                    {"author": "Civanch", "at": "2026-09-20T08:00:00Z", "body": "files in cfipython/<SubSystem>/<Package> <br> ok"}]
        answer = self._run("what did civanch say in issue 51505?", comments)
        self.assertIn("`civanch` wrote 2 imported comment(s), oldest first:", answer)
        self.assertIn("2026-09-14 21:14: we are not ready", answer)
        self.assertIn("cfipython/‹SubSystem›/‹Package›", answer)
        self.assertNotIn("opening", answer)
        answer = self._run("what did civanch explain about geant4 in 51505?", comments)
        self.assertIn("1 of them about \"geant4\"", answer)

    def test_not_a_commenter_and_cap(self):
        comments = [{"author": "Dr15Jones", "at": "2026-10-28T00:00:00Z", "body": "x"}]
        answer = self._run("what did fwyzard say on 52323?", comments, total=100)
        self.assertIn("no imported comment by `fwyzard`", answer)
        self.assertIn("only the first 100 comments", answer)

    def test_left_alone(self):
        for q in ("what did it say on 52323?", "what did the last comment on 52323 say?", "who commented on 52376?"):
            self.assertIsNone(api_server._person_comments_answer(q), q)
