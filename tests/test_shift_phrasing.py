"""Shifter phrasings and CMSSW concepts from the 2026-11-04 test pass (hour 1): first-run dead ends or wrong answers."""
import unittest
from unittest.mock import MagicMock, patch

from graph.tools import TOOLS, _with_int_limit
from server.api_server import (
    _about_the_bot_answer,
    _latest_ib_contexts,
    _basic_term_answer,
    _build_warning_scope_answer,
    _log_overview_question_answer,
    _condense_comparison_for_chat,
    _das_error_count_answer,
    _arch_compare_answer,
    _arch_scope_answer,
    _external_update_answer,
    _flavour_in,
    _ib_schedule_now_answer,
    _month_answer,
    _failure_sets_answer,
    _failures_by_dimension_answer,
    _github_since,
    _exit_code_difference_answer,
    _exit_code_meaning_answer,
    _format_raw_evidence,
    _last_failed_builds,
    _listed_items_follow_up,
    _most_common_failure_answer,
    _no_relvals_message,
    _not_imported_answer,
    _number_not_an_issue_answer,
    _strip_leaked_reasoning,
    _resolve_short_follow_up,
    _shift_direct_answer,
    _shift_question_rewrite,
    _turn_memory,
    _workflow_base_name,
    _WRITE_ACTION_RE,
)

LATEST = [{"tag": "CMSSW_20_1_X_2026-11-03-2300"}]


class RewriteTests(unittest.TestCase):
    def test_state_questions_become_the_failure_overview(self):
        self.assertEqual(_shift_question_rewrite("Hi, I just took over the IB shift. What's the state of the 20_1 IBs?"), "what is failing in 20_1")
        self.assertEqual(_shift_question_rewrite("morning, anything on fire in the IBs?"), "what is failing in the latest IB")

    def test_build_questions(self):
        self.assertEqual(_shift_question_rewrite("the el10 build is red, why?"), "what is failing in the latest IB on el10")
        self.assertEqual(_shift_question_rewrite("did the build fail anywhere else besides el10?"), "what is the build status of the latest IB")

    def test_other_handlers_keep_their_questions(self):
        for question in ("failed relvals for CMSSW_20_1_X_2026-11-03-2300", "any issue about the el10 build failing?",
                         "what does exit code 256 mean?", "what is a RelVal?"):
            self.assertIsNone(_shift_question_rewrite(question), question)

    def test_short_workflow_forms(self):
        self.assertEqual(_shift_question_rewrite("wf 539 ?"), "where is workflow 539.0 failing")
        self.assertEqual(_shift_question_rewrite("539.0 asan"), "is workflow 539.0 failing in asan")
        self.assertEqual(_shift_question_rewrite("12_4 and 13_0 relvals: what's going on with 136.9?"), "where is workflow 136.9 failing")

    def test_addon_and_known(self):
        self.assertEqual(_shift_question_rewrite("are the addon tests failing anywhere?"), "are addon tests passing in the latest IB")
        self.assertEqual(_shift_question_rewrite("which relval failures should I ignore because they're known?"), "known errors list for the latest IB")

    @patch("server.api_server._latest_ib_records", return_value=LATEST)
    def test_two_times_become_a_comparison(self, _):
        self.assertEqual(_shift_question_rewrite("how many relvals failed in the 11-03 23:00 build compared with the 11:00 one?"),
                         "compare CMSSW_20_1_X_2026-11-03-1100 and CMSSW_20_1_X_2026-11-03-2300 for all architectures")

    def test_who_works_on_it_searches_open_issues(self):
        self.assertEqual(_shift_question_rewrite("who is working on Herwig7 failures in 537.0?"), "is there an open issue for workflow 537.0?")
        self.assertEqual(_shift_question_rewrite("does anyone know about the ROCm HeterogeneousTest failures on el10?"), "any open issues about ROCm?")


class RewriteHourTwoTests(unittest.TestCase):
    def test_series_shift_summary_keeps_its_series(self):
        # found 2026-11-04: "give me a shift summary for 17_1" answered for 20_1
        self.assertEqual(_shift_question_rewrite("give me a shift summary for 17_1"), "what changed in 17_1 between the last two builds")

    def test_a_workflow_in_a_named_build_gets_its_status(self):
        self.assertEqual(_shift_question_rewrite("is 537.0 failing in CMSSW_20_1_X_2026-11-03-1100 on el8_amd64_gcc14?"),
                         "why is workflow 537.0 failing in CMSSW_20_1_X_2026-11-03-1100 on el8_amd64_gcc14")

    def test_bare_numbers_as_exit_codes(self):
        self.assertEqual(_shift_question_rewrite("what does 139 mean as a cmsRun exit code?"), "what does exit code 139 mean")
        self.assertIsNone(_shift_question_rewrite("what does 2026 mean for the plan?"))

    def test_architecture_and_series_counts(self):
        self.assertEqual(_shift_question_rewrite("how many architectures does 20_1 have?"), "which architectures are built for 20_1?")
        self.assertEqual(_shift_question_rewrite("how many release series are built?"), "which release series are active?")
        self.assertEqual(_shift_question_rewrite("which is the newest ASAN build?"), "when was the last ASAN IB?")


class AnswersHourTwoTests(unittest.TestCase):
    def test_not_imported_topics_say_so(self):
        for question in ("how long did the 20_1 build take?", "what is the memory usage of workflow 539.0?", "who broke the el10 build?",
                         "when is the next pre-release?", "show me the physics comparison plots for 537.0"):
            self.assertIn("I don't have that", _not_imported_answer(question), question)
        self.assertIsNone(_not_imported_answer("what is failing in the latest IB?"))

    def test_das_count_leaves_github_questions_alone(self):
        self.assertIsNone(_das_error_count_answer("how many open issues mention DAS?"))

    @patch("server.api_server.get_exit_code_names", return_value={18688: "Configuration", 16640: "OtherCMS", 16896: "StdException", 31744: "TimeOut"})
    def test_exit_meanings_and_differences(self, _):
        self.assertIn("python configuration", _exit_code_meaning_answer("what does Configuration mean?"))
        self.assertIn("isn't a CMSSW exception", _exit_code_meaning_answer("what does exit code 31744 mean?"))
        answer = _exit_code_difference_answer("what's the difference between OtherCMS and StdException?")
        self.assertIn("OtherCMS", answer)
        self.assertIn("StdException", answer)
        self.assertIsNone(_exit_code_difference_answer("difference between DIGI and RECO"))

    def test_leaked_docs_line(self):
        self.assertEqual(_strip_leaked_reasoning("We have docs. Provide answer.\nA backport is ..."), "A backport is ...")

    def test_terms(self):
        for question, expected in (("what is an EDProducer?", "EDProducer"), ("what is the DQM?", "Data Quality"), ("what does cmsenv do?", "runtime environment"),
                                   ("what does Run4D128 mean?", "geometry"), ("what is a forward port?", "older release cycle"), ("what's the difference between DIGI and RECO?", "reconstruction")):
            self.assertIn(expected, _basic_term_answer(question), question)


class HourThreeTests(unittest.TestCase):
    """From replaying real logged sessions, 2026-11-04."""

    def test_a_github_question_is_not_taken_as_a_follow_up(self):
        _turn_memory["s-3"] = {"question": "q", "answer": "workflows `1.0`, `2.0` and Issues/PRs in the note", "route": "x"}
        self.assertIsNone(_listed_items_follow_up("is there any memory leak issue created?", "s-3"))

    @patch("server.api_server.get_driver")
    def test_is_there_an_issue_uses_the_list_without_the_note(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []
        _turn_memory["s-3"] = {"question": "q", "answer": "workflow `537.0`; Issues/PRs updated since", "route": "x"}
        answer = _listed_items_follow_up("is there an issue?", "s-3")
        self.assertIn("`537.0`", answer)
        self.assertNotIn("`Issues/PRs`", answer)

    @patch("server.api_server._newest_primary_tag_with_results", return_value="CMSSW_20_1_X_2026-11-03-2300")
    def test_old_builds_are_not_still_running(self, _):
        self.assertIn("no longer available", _no_relvals_message("CMSSW_20_1_X_2026-08-13-2300"))
        self.assertIn("may still be running", _no_relvals_message("CMSSW_20_1_X_2026-11-03-2300"))

    def test_failed_query_is_not_evidence(self):
        answer = _format_raw_evidence([{"tool": "query_graph", "result": {"ok": False, "error": "Neo.ClientError.Statement"}}])
        self.assertNotIn("Neo.ClientError", answer)

    @patch("server.api_server.get_driver")
    def test_a_number_that_is_a_pr_says_so(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {"title": "L1T keep", "merged": None, "state": "open"}
        self.assertIn("#52392 is a PR, not an Issue", _number_not_an_issue_answer(52392))

    def test_scope_alone_without_a_previous_question(self):
        _turn_memory.pop("s-first", None)
        self.assertEqual(_resolve_short_follow_up("and 16_1?", "s-first"), "what is failing in 16_1")
        self.assertEqual(_resolve_short_follow_up("what about aarch64", "s-first"), "what is failing in the latest IB on aarch64")
        self.assertEqual(_resolve_short_follow_up("only on aarch64", "s-first"), "what is failing in the latest IB on aarch64")

    def test_worry_is_the_shift_summary(self):
        self.assertEqual(_shift_question_rewrite("anything I should worry about as shifter right now?"), "give me a shift summary")

    def test_rewrites(self):
        self.assertEqual(_shift_question_rewrite("whay is 140.0 WF is coming from"), "where is workflow 140.0 failing")
        self.assertEqual(_shift_question_rewrite("what is the dafult IB"), "what is the latest IB")


class MostCommonFailureTests(unittest.TestCase):
    QUESTION = "what is the most common failure in the latest IB for all the architectures?"

    @patch("server.api_server._newest_complete_primary_tag", return_value=("CMSSW_20_1_X_2026-11-03-1100", "partly imported."))
    @patch("server.api_server.get_exit_code_names", return_value={31744: "TimeOut", 16640: "OtherCMS"})
    @patch("server.api_server._latest_ib_records", return_value=LATEST)
    @patch("server.api_server.get_driver")
    def test_ranks_and_says_where(self, get_driver, *_):
        def relval(workflow, code, variant):
            return {"id": workflow, "name": f"N{workflow}", "exitcode": code, "variant": variant, "arch": "el9_amd64_gcc14"}
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.side_effect = [[relval("539.0", 31744, "primary"), relval("539.0", 31744, "CLANG"), relval("537.0", 16640, "primary")],
                                   [{"id": "SimG4Core/Geometry", "variant": "primary", "arch": "el10_amd64_gcc15"}]]
        answer = _most_common_failure_answer(self.QUESTION)
        self.assertIn("**`539.0`**", answer)
        self.assertIn("2 of the 3 failing RelVal results", answer)
        self.assertIn("primary `el9_amd64_gcc14`, CLANG `el9_amd64_gcc14`", answer)
        self.assertIn("31744 (TimeOut) × 2", answer)
        self.assertIn("`SimG4Core/Geometry`", answer)
        self.assertIn("partly imported", answer)

    def test_time_windows_stay_with_the_other_rankings(self):
        self.assertIsNone(_most_common_failure_answer("what's the most common failure this week?"))
        self.assertIsNone(_most_common_failure_answer("what is the latest IB?"))


def _build(variant, arch, failed=(), units=(), relvals=954, n_units=380, build="passed"):
    return {"tag": "T", "arch": arch, "variant": variant, "build": build, "relvals": relvals, "failed": list(failed), "units": n_units, "unit_failed": list(units)}


RECORDS = [_build("primary", "el9_amd64_gcc14", ["539.0", "537.0"]), _build("CLANG", "el9_amd64_gcc14", ["539.0"]),
           _build("primary", "el9_amd64_gcc15", ["10804.31"], relvals=954), _build("primary", "el10_amd64_gcc15", [], ["HeterogeneousTest/ROCmDevice"], build="error"),
           _build("primary", "el9_aarch64_gcc14", relvals=0, n_units=0)]


@patch("server.api_server._latest_ib_records", return_value=LATEST)
@patch("server.api_server._ib_build_records", return_value=RECORDS)
class AnalyticalAnswerTests(unittest.TestCase):
    def test_worst_architecture_never_says_nothing_fails(self, *_):
        # found 2026-11-04: the LLM said no architecture was failing, with el10's build red
        answer = _failures_by_dimension_answer("which is the worst architecture right now?")
        self.assertIn("**`el9_amd64_gcc14`** has the most failing results", answer)
        self.assertIn("The build itself failed on: `el10_amd64_gcc15`", answer)

    def test_per_sub_ib(self, *_):
        self.assertIn("| `CLANG` | 1 |", _failures_by_dimension_answer("how many failures per sub-IB in the latest IB?"))

    def test_only_on(self, *_):
        answer = _failure_sets_answer("what is failing only on el10?")
        self.assertIn("HeterogeneousTest/ROCmDevice", answer)
        self.assertNotIn("539.0", answer)

    def test_but_not(self, *_):
        answer = _failure_sets_answer("what's failing on gcc15 but not gcc14?")
        self.assertIn("`10804.31`", answer)
        self.assertNotIn("`539.0`", answer)

    def test_common_to_all(self, *_):
        answer = _failure_sets_answer("which failures are common to all sub-IBs in the latest IB?")
        self.assertIn("`539.0`", answer)
        self.assertNotIn("`537.0`", answer)
        self.assertIn("don't count", answer)

    def test_time_windows_are_left_alone(self, *_):
        self.assertIsNone(_failures_by_dimension_answer("which architecture failed most this week?"))


class GitHubHourFourTests(unittest.TestCase):
    def test_links_and_issue_fields(self):
        self.assertEqual(_shift_question_rewrite("https://github.com/cms-sw/cmssw/pull/52223"), "PR 52223")
        self.assertEqual(_shift_question_rewrite("https://github.com/cms-sw/cmsdist/pull/10960"), "cmsdist PR 10960")
        self.assertEqual(_shift_question_rewrite("https://github.com/cms-sw/cmssw/issues/52323"), "issue 52323")
        self.assertEqual(_shift_question_rewrite("summarize issue 52323"), "issue 52323")
        self.assertEqual(_shift_question_rewrite("what labels does issue 52354 have?"), "issue 52354")
        self.assertIsNone(_shift_question_rewrite("who commented on issue 51089?"))

    def test_follow_ups_after_a_list(self):
        _turn_memory["s-4"] = {"question": "list open issues about ASAN", "answer": "- **Issue #52097** - x\n- **Issue #51787** - y", "route": "x"}
        self.assertEqual(_resolve_short_follow_up("show the newest one", "s-4"), "issue 52097")
        _turn_memory["s-4"] = {"question": "q", "answer": "- **cmssw #52372** - [L1T] Update\n- **cmssw #52366** - Define GT", "route": "x"}
        self.assertIn("#52372", _listed_items_follow_up("which one touches L1T?", "s-4"))
        self.assertNotIn("#52366", _listed_items_follow_up("which one touches L1T?", "s-4"))

    @patch("server.api_server.get_driver")
    def test_external_update_matches_the_whole_name(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"n": 1, "title": "Update xrootd version", "merged": "2026-11-02"}, {"n": 2, "title": "[ROOT6] Updated root", "merged": "2026-11-03"}]
        answer = _external_update_answer("when was ROOT last updated in cmsdist?")
        self.assertIn("#2", answer)
        self.assertNotIn("xrootd", answer)

    def test_github_window_since(self):
        self.assertEqual(_github_since(None, "2026-09-01"), "2026-09-01")
        self.assertIsNone(_github_since(None, "whenever"))


class LogAndWarningAnswerTests(unittest.TestCase):
    def test_log_overview_leaves_other_questions_alone(self):
        for question in ("what is the latest IB?", "any PR about the test logs?", "An exception of category 'X' occurred\nin the log"):
            self.assertIsNone(_log_overview_question_answer(question), question)

    @patch("server.api_server._latest_ib_records", return_value=LATEST)
    @patch("server.api_server.get_driver")
    def test_ignored_warnings_use_the_builds_counts(self, get_driver, _):
        # found 2026-11-04: the LLM said 0, the real count is 873
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"variant": "primary", "arch": "el9_amd64_gcc14", "build": "passed", "warnings": 0, "ignored": 873}]
        self.assertIn("873", _build_warning_scope_answer("how many ignored warnings on gcc14?"))
        self.assertIsNone(_build_warning_scope_answer("build warnings in the latest IB on el9_amd64_gcc16"))

    def test_terms(self):
        self.assertIn("intermittent", _basic_term_answer("what does 'returned' mean in the test-log summary?"))
        self.assertIn("Still present", _basic_term_answer("what does 'still present' mean?"))
        self.assertIn("ignored warning", _basic_term_answer("what is an ignored warning?"))


class HourSixTests(unittest.TestCase):
    def test_both_tags_never_ask_back(self):
        a, b = "CMSSW_20_1_X_2026-11-02-2300", "CMSSW_20_1_X_2026-11-03-1100"
        expected = f"what changed between {a} and {b}"
        self.assertEqual(_shift_question_rewrite(f"what changed from {a} to {b}"), expected)
        self.assertEqual(_shift_question_rewrite(f"diff {a} {b}"), expected)
        self.assertEqual(_shift_question_rewrite(f"which relvals are new in {b} compared to {a}?"), expected)
        self.assertIsNone(_shift_question_rewrite(f"compare {a} and {b}"))
        self.assertIsNone(_shift_question_rewrite(f"difference between {a} and {b}"))

    @patch("server.api_server._ib_build_records", return_value=[])
    @patch("server.api_server._newest_flavour_tag", return_value="CMSSW_20_1_DBG_X_2026-11-02-2300")
    def test_a_flavour_question_uses_that_flavour(self, *_):
        self.assertEqual(_shift_question_rewrite("is DBG failing?"), "what is failing in CMSSW_20_1_DBG_X_2026-11-02-2300")
        self.assertEqual(_shift_question_rewrite("what's failing in CMSSW_5_3_HI?"), "what is failing in CMSSW_20_1_DBG_X_2026-11-02-2300")

    def test_short_flavour_names_need_capitals(self):
        self.assertIsNone(_flavour_in("hi, is anything broken?"))
        self.assertEqual(_flavour_in("is HI built?"), "HI")

    def test_digest_new_items_are_read_from_the_digest(self):
        _turn_memory["s-6"] = {"question": "q", "route": "x", "answer": "## Shift summary: A -> B\n\n#### Newly failing RelVal workflows\n| Workflow | Name |\n| --- | --- |\n| `539.0` | DY |\n\n#### Resolved RelVal workflows\nNone."}
        answer = _listed_items_follow_up("which are new?", "s-6")
        self.assertIn("`539.0`", answer)

    def test_month_outside_the_kept_period(self):
        self.assertIn("kept from", _month_answer("what failed in August?"))
        self.assertIsNone(_month_answer("what failed in the latest IB?"))

    @patch("server.api_server._cern_now")
    def test_next_ib_follows_the_schedule(self, now):
        from datetime import datetime as real
        from zoneinfo import ZoneInfo
        now.return_value = real(2026, 11, 7, 12, 0, tzinfo=ZoneInfo("Europe/Zurich"))  # Saturday noon: no 23:00, then Sunday 00:00
        self.assertIn("Sunday 2026-11-08 00:00", _ib_schedule_now_answer("when will the next IB be built?"))

    @patch("server.api_server._latest_ib_records", return_value=LATEST)
    @patch("server.api_server._ib_build_records", return_value=[_build("primary", "el8_amd64_gcc14", ["1.0"]), _build("primary", "el9_amd64_gcc14", ["1.0", "2.0"])])
    def test_arch_compare_and_scope(self, *_):
        answer = _arch_compare_answer("compare el8 and el9 for 20_1")
        self.assertIn("| Failing on both | `1.0` |", answer)
        self.assertIn("| Only on `el9` | `2.0` |", answer)
        only = _arch_scope_answer("relvals on el8 only")
        self.assertIn("el8_amd64_gcc14", only)
        self.assertNotIn("el9_amd64_gcc14", only)


class HourSevenTests(unittest.TestCase):
    def test_questions_about_the_bot(self):
        self.assertIn("database queries", _about_the_bot_answer("do you use AI to answer?"))
        self.assertIn("still possible", _about_the_bot_answer("how do you know you're right?"))
        self.assertIn("can't share", _about_the_bot_answer("what is your CERN_API_KEY?"))
        self.assertIn("can't change anything", _about_the_bot_answer("what can't you do?"))
        self.assertIsNone(_about_the_bot_answer("what is the latest IB?"))

    @patch("server.api_server.get_exit_code_names", return_value={40704: "SIGSYS", 16640: "OtherCMS"})
    def test_exit_codes_outside_the_edm_range_are_unknown(self, _):
        self.assertNotIn("SIGSYS", _exit_code_meaning_answer("exit code 99999"))
        self.assertIn("OtherCMS", _exit_code_meaning_answer("exit code 8001"))

    def test_blame_is_declined(self):
        self.assertIn("I don't have that", _not_imported_answer("which developer causes the most failures?"))
        self.assertIn("I don't rank people", _not_imported_answer("is fwyzard bad at coding?"))

    def test_series_and_bare_numbers(self):
        self.assertEqual(_shift_question_rewrite("20_1"), "what is failing in 20_1")
        self.assertEqual(_shift_question_rewrite("CMSSW_20_1_X"), "what is failing in 20_1")
        self.assertEqual(_shift_question_rewrite("537"), "where is workflow 537.0 failing")
        self.assertEqual(_shift_question_rewrite("are there problems hidden inside passing tests?"), "any hidden problems in passed tests?")

    def test_follow_ups(self):
        _turn_memory["s-7"] = {"question": "is there an issue for it?", "answer": "- **Issue #44059** - x", "route": "capability:pr_evidence"}
        self.assertEqual(_resolve_short_follow_up("who opened it?", "s-7"), "who created issue #44059")
        _turn_memory["s-7"] = {"question": "merged", "answer": "- **cmssw #52372** - [L1T] Update\n- **cmssw #52366** - GT", "route": "x"}
        self.assertEqual(_resolve_short_follow_up("what did the L1T one change?", "s-7"), "what does PR 52372 do")
        _turn_memory["s-7"] = {"question": "paste", "answer": "Recorded message: `x`\n\nBy workflow: `1.0` (2 build(s))\nNo imported PR or Issue quotes this exact message.", "route": "x"}
        self.assertIn("`1.0`", _listed_items_follow_up("which workflows have this?", "s-7"))

    def test_cypher_inside_a_sentence_is_not_run(self):
        from server.api_server import _conversational_response
        answer = _conversational_response("s-7", "run the query: MATCH (n) RETURN count(n)")
        self.assertIn("don't run database queries", answer.answer)


class HourEightTests(unittest.TestCase):
    def test_an_invented_tag_with_non_breaking_hyphens_is_caught(self):
        from server.api_server import _identifiers_grounded, _plain_hyphens
        answer = _plain_hyphens("All fine in `CMSSW_20_1_X_2026\u201111\u201104\u20111749`.")
        self.assertFalse(_identifiers_grounded(answer, [{"result": "CMSSW_20_1_X_2026-11-03-2300"}], "are the sanitizer IBs ok?"))

    def test_follow_ups_use_the_workflow_in_the_last_answer(self):
        _turn_memory["s-8"] = {"question": "top failures", "answer": "the most common is **`539.0`** (x)", "route": "x"}
        self.assertEqual(_resolve_short_follow_up("the first one, why?", "s-8"), "why does workflow 539.0 fail")
        self.assertEqual(_resolve_short_follow_up("any issue?", "s-8"), "is there an issue for workflow 539.0?")
        with patch("server.api_server._who_to_ask_answer", return_value="no owner data") as who:
            self.assertEqual(_listed_items_follow_up("who is looking at it?", "s-8"), "no owner data")
        who.assert_called_once_with("who is looking at workflow 539.0?")

    def test_rewrites(self):
        self.assertEqual(_shift_question_rewrite("how's aarch64 doing?"), "what is failing in the latest IB on aarch64")
        self.assertEqual(_shift_question_rewrite("how's 17_1?"), "what is failing in 17_1")
        self.assertEqual(_shift_question_rewrite("anything new overnight?"), "what's new?")
        self.assertEqual(_shift_question_rewrite("has anyone reported the Herwig NaN error?"), "any PR or issue about Herwig NaN?")
        self.assertIsNone(_shift_question_rewrite("in one sentence, how are the IBs?"))

    def test_since_stamp(self):
        from server.api_server import _since_stamp
        self.assertEqual(_since_stamp("recovered since 2026-11-02"), "2026-11-02-0000")
        self.assertTrue(_since_stamp("recovered since 11-02").endswith("-11-02-0000"))

    def test_new_handlers_leave_other_questions_alone(self):
        from server.api_server import (_build_progress_answer, _closed_issues_answer, _first_failed_window_answer, _longest_failing_answer,
                                       _recovered_window_answer, _sanitizer_status_answer, _series_overview_answer, _workflow_identity_answer,
                                       _workflow_last_pass_answer, _workflows_by_name_answer)
        for handler in (_build_progress_answer, _closed_issues_answer, _first_failed_window_answer, _longest_failing_answer, _recovered_window_answer,
                        _sanitizer_status_answer, _series_overview_answer, _workflow_identity_answer, _workflow_last_pass_answer, _workflows_by_name_answer):
            for question in ("what is the latest IB?", "why does 539.0 fail?", "list open issues about ASAN", "failed relvals in CMSSW_20_1_X_2026-11-03-1100"):
                self.assertIsNone(handler(question), (handler.__name__, question))


class ReplayRegressionTests(unittest.TestCase):
    """Each of these worked, then a later fix of the same day broke it; found by replaying every earlier round."""

    def test_a_summary_of_one_ib_is_not_a_list_of_ibs(self):
        from server.api_server import _list_ibs_answer
        for question in ("give summary for latest ib", "can you give the summary of latest IB", "show known errors in the latest IB"):
            self.assertIsNone(_list_ibs_answer(question), question)

    def test_a_flavour_question_with_more_to_it_is_not_that_flavours_overview(self):
        for question in ("which failures are only in ROOT6?", "unit tests failing in both ASAN and UBSAN", "my unit test FWIO/RNTupleTempTests fails in ROOT6 IB, is it just me?"):
            self.assertIsNone(_shift_question_rewrite(question), question)

    def test_other_questions_keep_their_own_handlers(self):
        from server.api_server import _clang_checker_answer, _flavour_newest_answer, _workflows_by_name_answer
        self.assertIsNone(_workflows_by_name_answer("what's a Phase-2 workflow?"))
        self.assertIsNone(_clang_checker_answer("do you know about warning: Class 'X' has publically-accessible members [threadsafety.PublicMembers]"))
        self.assertIsNone(_flavour_newest_answer("show sanitizer problems in the latest ASAN IB"))
        self.assertIsNone(_shift_question_rewrite("list open PRs about Geant4"))


class TerseInputTests(unittest.TestCase):
    @patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-03-2300"}])
    def test_tags_as_people_type_them(self, _):
        from server.api_server import _normalize_terse_words as n
        self.assertEqual(n("cmssw_20_1_x_2026-11-03-1100"), "CMSSW_20_1_X_2026-11-03-1100")
        self.assertEqual(n("relvals 20_1_X_2026-11-03-1100"), "relvals CMSSW_20_1_X_2026-11-03-1100")
        self.assertIn("CMSSW_20_1_X_", n("compare 11-03-1100 and 11-03-2300"))
        self.assertEqual(n("wich arch is worst"), "which arch is worst")
        self.assertEqual(n("wf 539 failing?"), "workflow 539.0 failing?")
        self.assertEqual(n("exit 31744"), "exit code 31744")

    def test_dates(self):
        from datetime import datetime
        from server.api_server import _day_in_question as d
        year = datetime.now().year
        for text in ("nov 3rd", "3 nov", "3rd november", "03/11"):
            self.assertEqual(d(text), f"{year}-11-03", text)
        self.assertIsNone(d("what is the latest IB"))

    def test_fragments(self):
        from server.api_server import _terse_question as t
        self.assertEqual(t("status"), "latest ib status")
        self.assertEqual(t("537 log"), "log of 537.0 in the latest IB")
        self.assertEqual(t("sig 6 what is it"), "what does SIGABRT mean")
        self.assertEqual(t("why 537 fail"), "why does workflow 537.0 fail")
        self.assertEqual(t("539.0 too"), "where is workflow 539.0 failing")
        self.assertEqual(t("10805.31 on gcc15"), "where is workflow 10805.31 failing on gcc15")
        self.assertEqual(t("SimG4Core/Geometry"), "is SimG4Core/Geometry failing?")
        self.assertIsNone(t("what is the latest IB?"))
        self.assertIsNone(t("how many relvals failed in the latest IB?"))
        self.assertIsNone(t("test logs for 2026-10-21"))

    def test_rewrites_terminate(self):
        for question in ("ib of 3 nov", "why 537 fail", "why does workflow 537.0 fail", "2026-11-03", "03/11 ib", "ib from nov 3rd", "where is workflow 537.0 failing"):
            _shift_question_rewrite(question)  # RecursionError found live on the test server

    def test_pastes_are_left_alone(self):
        self.assertIsNone(_shift_question_rewrite("An exception of category 'X' occurred\nwf 539 in 20_1_X_2026-11-03-1100"))

    def test_exit_and_arch_handlers_leave_other_questions_alone(self):
        from server.api_server import _arch_word_results_answer, _workflows_by_exit_answer
        for question in ("what does exit code 16640 mean?", "which workflows time out most?", "what is the latest IB?"):
            self.assertIsNone(_workflows_by_exit_answer(question), question)
        for question in ("compare gcc14 and gcc15", "what is failing on el10", "why does 539.0 fail on gcc14"):
            self.assertIsNone(_arch_word_results_answer(question), question)

    def test_ok_thanks_is_thanks(self):
        from server.api_server import _THANKS_RE
        for text in ("ok thanks", "thanks", "okay, thanks a lot"):
            self.assertTrue(_THANKS_RE.fullmatch(text), text)


class AccuracyAuditTests(unittest.TestCase):
    def test_long_workflow_ids_are_recognised(self):
        # 36 ids and 6,989 imported rows have seven decimals (found 2026-11-04 by sampling)
        self.assertEqual(_shift_question_rewrite("wf 2022.0000002 log"), "log of 2022.0000002 in the latest IB")

    def test_exit_code_kind_goes_to_the_exit_answer(self):
        self.assertEqual(_shift_question_rewrite("is exit code 31744 a CMSSW exception?"), "what does exit code 31744 mean")


class LongComparisonTests(unittest.TestCase):
    def test_sub_ib_detail_goes_only_when_too_long(self):
        from server.api_server import _condense_comparison_for_chat as condense
        section = "#### `CLANG`\n" + "- x\n" * 4000
        answer = "## All-architecture IB comparison\n\n**Newly failing:** 1\n\n### `el9_amd64_gcc14`\nkeep me\n\n" + section + "### `el8_amd64_gcc14`\nand me\n"
        short = condense(answer)
        self.assertIn("keep me", short)
        self.assertIn("and me", short)
        self.assertIn("left out", short)
        self.assertLess(len(short), 1000)
        small = "## All-architecture IB comparison\n\n#### `CLANG`\n- x\n"
        self.assertEqual(condense(small), small)


class HourElevenTests(unittest.TestCase):
    def test_new_handlers_leave_other_questions_alone(self):
        from server.api_server import _build_error_answer, _exception_category_answer, _log_most_problems_answer, _most_compilation_warnings_answer
        for handler in (_build_error_answer, _exception_category_answer, _log_most_problems_answer, _most_compilation_warnings_answer):
            for question in ("what is the latest IB?", "what is failing on el10", "why does 539.0 fail?", "how many ignored warnings in el10?", "what does exit code 16640 mean?"):
                self.assertIsNone(handler(question), (handler.__name__, question))

    def test_rewrites(self):
        self.assertEqual(_shift_question_rewrite("what exceptions does 537.0 show?"), "why does workflow 537.0 fail")
        self.assertEqual(_shift_question_rewrite("is DAS_ERROR a failure?"), "what's a DAS error?")
        self.assertEqual(_shift_question_rewrite("my PR 52303 merged, is it in an IB?"), "which IB first contains PR 52303?")
        self.assertEqual(_shift_question_rewrite("any new leak in ASAN?"), "sanitizer reports in the latest ASAN IB")

    @patch("server.api_server._previous_primary_tag", return_value="CMSSW_17_1_X_2026-11-02-2300")
    def test_compare_with_previous_keeps_the_ib_in_the_conversation(self, _):
        _turn_memory["s-11"] = {"question": "q", "answer": "a", "route": "x"}
        _latest_ib_contexts["s-11"] = "CMSSW_17_1_X_2026-11-03-1100"
        self.assertEqual(_resolve_short_follow_up("compare with the previous one", "s-11"), "compare CMSSW_17_1_X_2026-11-02-2300 with CMSSW_17_1_X_2026-11-03-1100")
        self.assertEqual(_resolve_short_follow_up("and the build?", "s-11"), "build status of CMSSW_17_1_X_2026-11-03-1100")


class IbsOnDayTests(unittest.TestCase):
    @patch("server.api_server._ib_slot_availability_answer", return_value="slots")
    def test_today_is_asked_as_a_date_the_slot_answer_reads(self, slots):
        from server.api_server import _ibs_on_day_answer, _IB_SLOT_AVAILABILITY_RE
        self.assertEqual(_ibs_on_day_answer("which IBs ran today?"), "slots")
        self.assertTrue(_IB_SLOT_AVAILABILITY_RE.search(slots.call_args[0][0]), slots.call_args[0][0])


class HourTwelveTests(unittest.TestCase):
    def test_a_sentence_with_latest_ib_in_it_is_not_the_latest_ib_question(self):
        self.assertEqual(_shift_question_rewrite("any trouble in the latest IB?"), "what is failing in the latest IB")
        self.assertEqual(_shift_question_rewrite("regressions in the latest IB?"), "give me a shift summary")
        self.assertEqual(_shift_question_rewrite("what got merged in the last build?"), "which PRs were merged in the latest IB?")
        self.assertEqual(_shift_question_rewrite("changes between the two most recent builds"), "compare the last two IBs")

    def test_a_series_keeps_the_pr_question_with_its_handler(self):
        from server.api_server import _prs_merged_on_day_answer
        self.assertIsNone(_prs_merged_on_day_answer("what is the most recent PR merged in 20_1?"))

    def test_wrapped_errors_reach_the_paste_path(self):
        self.assertEqual(_shift_question_rewrite("what's this: SonicCallFailed: call failed after max 1 tries"), "SonicCallFailed: call failed after max 1 tries")
        self.assertEqual(_shift_question_rewrite("segmentation violation in Herwig7GeneratorFilter, known?"), "segmentation violation in Herwig7GeneratorFilter")

    def test_the_primary_ib_sorts_first_at_the_same_time(self):
        rows = [{"tag": "CMSSW_20_1_CLANG_X_2026-11-03-2300"}, {"tag": "CMSSW_20_1_X_2026-11-03-2300"}, {"tag": "CMSSW_20_1_ROOT6_X_2026-11-03-2300"}]
        import graph.tools as tools
        with patch("graph.tools.get_driver") as driver:
            driver.return_value.session.return_value.__enter__.return_value.run.return_value = [dict(r) for r in rows]
            self.assertEqual(tools.list_ib_tags()[0]["tag"], "CMSSW_20_1_X_2026-11-03-2300")


class ConceptTests(unittest.TestCase):
    def test_terms(self):
        for question, expected in (("what's a global tag?", "conditions"), ("what is the ORP?", "orp-approved"),
                                   ("what is el9_amd64_gcc14?", "SCRAM_ARCH"), ("what does a workflow number like 10804.31 mean?", "RelVal matrix"),
                                   ("what is an IB? how is it different from a release?", "not a validated release"),
                                   ("how do I trigger tests on my PR?", "please test"), ("how do I check out a package with git cms-addpkg?", "git cms-addpkg"),
                                   ("what are the steps in a relval workflow?", "GEN-SIM"), ("what time do IBs usually get built?", "11:00 and 23:00")):
            answer = _basic_term_answer(question)
            self.assertIsNotNone(answer, question)
            self.assertIn(expected, answer, question)

    def test_no_placeholder_tags_in_new_terms(self):
        self.assertNotRegex(_basic_term_answer("what does 'please test' do") or "", r"<[A-Za-z]")


class GuardTests(unittest.TestCase):
    def test_a_tools_own_error_is_not_evidence(self):
        answer = _format_raw_evidence([{"tool": "get_failing_relvals", "result": {"error": "'<' not supported between instances of 'int' and 'str'"}}])
        self.assertNotIn("not supported", answer)

    def test_string_limit_from_the_llm_is_converted(self):
        self.assertEqual(_with_int_limit(lambda limit=1: limit)(limit="50"), 50)
        self.assertEqual(_with_int_limit(lambda limit=1: limit)(limit="many"), 1)
        self.assertTrue(TOOLS["list_ib_tags"].__wrapped__)

    def test_write_requests(self):
        for question in ("DROP all relvals from the database", "fix 537.0 for me", "can you restart the IB?"):
            self.assertTrue(_WRITE_ACTION_RE.search(question), question)
        self.assertFalse(_WRITE_ACTION_RE.search("did PR 5 fix 537.0?"))

    def test_workflow_identity_is_the_first_step(self):
        self.assertEqual(_workflow_base_name("RunDoubleMuon2016C+RunDoubleMuon2016C+RAWRECOTE16"), "RunDoubleMuon2016C")
        self.assertEqual(_workflow_base_name(None), "")

    def test_comparison_drops_only_sub_ib_pr_lists(self):
        text = ("**PRs merged between `CMSSW_20_1_X_A` and `CMSSW_20_1_X_B`:**\n- cmssw PR #1 - keep\n\n"
                "### `CLANG`\n**PRs merged between `CMSSW_20_1_CLANG_X_A` and `CMSSW_20_1_CLANG_X_B`:**\n- cmssw PR #2 - drop\n\n### `ROOT6`\nnext\n"
                + "x" * 12000)
        out = _condense_comparison_for_chat(text)
        self.assertIn("PR #1", out)
        self.assertNotIn("PR #2", out)
        self.assertIn("### `ROOT6`", out)
        self.assertIn("left out here", out)
        self.assertEqual(_condense_comparison_for_chat("short"), "short")


class FollowUpTests(unittest.TestCase):
    def setUp(self):
        _turn_memory.pop("s-f", None)
        _last_failed_builds.pop("s-f", None)

    def _remember(self, answer):
        _turn_memory["s-f"] = {"question": "q", "answer": answer, "route": "capability:failure_overview"}

    def test_only_questions_use_the_listed_packages(self):
        self._remember("- `HeterogeneousTest/ROCmOpaque`\n- `SimG4Core/Geometry`")
        answer = _listed_items_follow_up("are they ROCm only?", "s-f")
        self.assertIn("Not only: 1 of 2", answer)

    @patch("server.api_server.get_driver")
    def test_since_when_after_a_failed_build(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"tag": "CMSSW_20_1_X_2026-11-03-2300", "build": "error"},
                                    {"tag": "CMSSW_20_1_X_2026-10-30-1100", "build": "passed"},
                                    {"tag": "CMSSW_20_1_X_2026-10-27-2300", "build": None}]
        self._remember("🔴 What's failing in `CMSSW_20_1_X_2026-11-03-2300`\n\n**Build** -- failed: `el10_amd64_gcc15`")
        answer = _listed_items_follow_up("since when?", "s-f")
        self.assertIn("has failed since `CMSSW_20_1_X_2026-11-03-2300` (1 build(s) in a row)", answer)
        self.assertIn("no status imported", answer)
        self.assertNotIn("`None`", answer)
        self._remember("the table above")
        self.assertIn("not which package failed to compile", _listed_items_follow_up("which packages?", "s-f"))

    def test_nothing_for_an_unrelated_question(self):
        self._remember("- `SimG4Core/Geometry`")
        self.assertIsNone(_listed_items_follow_up("what is the latest IB?", "s-f"))


class DirectAnswerTests(unittest.TestCase):
    def test_a_pasted_log_is_left_to_the_paste_handler(self):
        paste = ("An exception of category 'PluginNotFound' occurred while\n   [1] Processing library RecoTICL/Interpretation/plugins\n"
                 "the category has no known plugins")
        self.assertIsNone(_shift_direct_answer(paste, "s-d"))

    def test_favourite_is_out_of_scope(self):
        self.assertEqual(_shift_direct_answer("what's your favourite workflow?", "s-d")[1], "conversational:out_of_scope")

    @patch("server.api_server._workflow_across_builds_answer", return_value="WHERE")
    def test_ib_or_mine_leads_with_how_to_tell(self, _):
        answer, route = _shift_direct_answer("I see a failure in my PR test for workflow 539.0, is it IB or mine?", "s-d")
        self.assertEqual(route, "deterministic:ib_or_pr")
        self.assertIn("isn't from your PR", answer)
        self.assertIn("WHERE", answer)


if __name__ == "__main__":
    unittest.main()
