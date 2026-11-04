"""Wrong/dead-end answers found 2026-11-03 by a keyword-trap probe of the live bot
(one keyword in a question picking a deterministic answer for something else)."""
import ast
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

from server.api_server import (
    _ABOUT_DATA_RE,
    _addon_status_answer,
    _architectures_answer,
    _ib_variant_glossary_answer,
    _known_error_list_answer,
    _latest_ib_answer,
    _newest_complete_primary_tag,
    _not_failing_answer,
    _package_status_answer,
    _previous_primary_tag,
    _prs_merged_window_answer,
    _tracked_failures_answer,
    _workflow_log_answer,
    _issue_state_answer,
    _GITHUB_COVERAGE_CACHE,
    _github_export_note,
    _variant_diff_baseline,
    _variant_tag,
    _known_error_list_answer,
    _render_pr_evidence_rows,
    _strip_leaked_reasoning,
    _exception_ranking_answer,
    _release_series_answer,
    _normalize_deterministic_question,
    _conversational_response,
    _MISSING_RESULTS_RE,
    _ARCH_LIST_RE,
    _test_log_question_answer,
    _with_variant_ib_context,
    _pr_author_ranking_answer,
    _relval_by_exit_code_answer,
    _basic_term_answer,
    _clang_package_warning_answer,
    _issue_search_term,
    _latest_ib_contexts,
    _resolve_short_follow_up,
    _turn_memory,
    _unittest_summary_answer,
)
from server.request_planner import RequestContext


def _driver_with(*results):
    """A get_driver() mock whose session.run returns each result in turn."""
    session = MagicMock()
    session.run.side_effect = list(results)
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    return driver, session


class ShortFollowUpTests(unittest.TestCase):
    def setUp(self):
        self.sid = "demo-fixes"
        _turn_memory.pop(self.sid, None)
        _latest_ib_contexts.pop(self.sid, None)

    def remember(self, question, answer):
        _turn_memory[self.sid] = {"question": question, "answer": answer, "route": "deterministic"}

    def test_architecture_follow_up_repeats_previous_question(self):
        # was a PR/Issue text search for "el8_amd64_gcc14"
        self.remember("what is the latest IB?", "Newest imported ... `CMSSW_20_1_X_2026-11-02-2300`")
        self.assertEqual(_resolve_short_follow_up("what about el8?", self.sid), "what is the latest IB on el8")

    def test_architecture_follow_up_replaces_previous_architecture(self):
        self.remember("unit tests for CMSSW_20_1_X_2026-11-02-2300 on el9_amd64_gcc14", "...")
        self.assertEqual(_resolve_short_follow_up("and on aarch64 only", self.sid),
                         "unit tests for CMSSW_20_1_X_2026-11-02-2300 on aarch64")

    def test_which_ones_after_how_many(self):
        self.remember("how many relvals failed on el9_amd64_gcc14 in the latest IB?", "2 failing")
        self.assertEqual(_resolve_short_follow_up("which ones?", self.sid),
                         "which relvals failed on el9_amd64_gcc14 in the latest IB?")

    def test_is_it_good_to_use_after_latest_ib(self):
        self.remember("what is the latest IB?", "- **IB tag:** `CMSSW_20_1_X_2026-11-02-2300`")
        self.assertEqual(_resolve_short_follow_up("and is it good to use?", self.sid),
                         "what is failing in CMSSW_20_1_X_2026-11-02-2300")

    def test_why_prefers_failing_workflow_over_das_error_one(self):
        self.remember(
            "how many relvals failed on el9_amd64_gcc14 in CMSSW_20_1_X_2026-11-02-1100?",
            "DAS errors: `38551.85`\nFailing workflows:\n- SingleGammaPt10 (wf 10804.31, exit 25600)")
        self.assertEqual(_resolve_short_follow_up("why?", self.sid),
                         "why is workflow 10804.31 failing in CMSSW_20_1_X_2026-11-02-1100")

    def test_flavour_follow_up_is_not_taken_as_architecture(self):
        self.remember("relval failures in CMSSW_20_1_X_2026-11-02-2300", "...")
        self.assertEqual(_resolve_short_follow_up("what about ASAN?", self.sid),
                         "relval failures in CMSSW_20_1_ASAN_X_2026-11-02-2300")


class DefinitionTests(unittest.TestCase):
    def test_das_error_with_underscore(self):
        self.assertIn("DAS error", _basic_term_answer("what does DAS_ERROR mean?"))

    def test_difference_between_two_known_terms(self):
        answer = _basic_term_answer("what is the difference between a relval and a unit test?")
        self.assertIn("RelVal", answer)
        self.assertIn("Unit tests", answer)

    def test_difference_with_unknown_term_is_not_answered(self):
        self.assertIsNone(_basic_term_answer("what is the difference between a relval and a banana?"))

    def test_ib_schedule(self):
        for question in ("how often are IBs built?", "when are IBs built", "IB schedule"):
            self.assertIn("11:00 and 23:00", _basic_term_answer(question), question)

    def test_failure_questions_are_not_schedule_questions(self):
        self.assertIsNone(_basic_term_answer("when did the latest IB build fail?"))
        self.assertIsNone(_basic_term_answer("how often does 140.111 fail?"))

    def test_cmssw(self):
        self.assertIn("software framework", _basic_term_answer("what is CMSSW?"))


class RankingIsNotAShiftSummaryTests(unittest.TestCase):
    def test_most_often_over_a_window(self):
        context = RequestContext.extract("which workflow failed most often in the last 14 days?")
        self.assertFalse(context.has_shift_summary_request)

    def test_newly_failing_over_a_window_still_is(self):
        self.assertTrue(RequestContext.extract("what's newly failing in the last 3 days").has_shift_summary_request)


class IssueSearchTermTests(unittest.TestCase):
    def test_short_term_is_whole_word(self):
        term = _issue_search_term("DAS")
        self.assertRegex("DAS_ERROR at step1", term["re"].replace("(?is)", ""))
        self.assertNotRegex("New IB dashboard", "(?i)" + term["re"].replace("(?is)", ""))

    def test_long_term_keeps_substring_match(self):
        self.assertIsNone(_issue_search_term("FileInPath")["re"])


class UnitTestArchitectureTests(unittest.TestCase):
    @patch("server.api_server._unittest_scope_records")
    def test_architecture_without_results_says_so(self, records):
        records.return_value = [
            {"arch": "el8_amd64_gcc14", "is_default": False, "total": 379, "failed": 0, "passed": 379, "errors": 0},
            {"arch": "el9_amd64_gcc14", "is_default": True, "total": 379, "failed": 0, "passed": 379, "errors": 0},
        ]
        answer = _unittest_summary_answer(
            "unit tests for CMSSW_20_1_X_2026-11-02-2300 on el9_aarch64_gcc14", None, "demo-ut")
        self.assertIn("No unit-test results are imported", answer)
        self.assertIn("el9_aarch64_gcc14", answer)
        self.assertNotIn("Choose one", answer)


class AddOnStatusTests(unittest.TestCase):
    @patch("server.api_server.get_driver")
    def test_named_ib_addon_status(self, get_driver):
        driver, _ = _driver_with([
            {"tag": "CMSSW_20_1_X_2026-11-02-2300", "variant": None, "arch": "el9_amd64_gcc14", "status": "passed"},
            {"tag": "CMSSW_20_1_ASAN_X_2026-11-02-2300", "variant": "ASAN", "arch": "el9_amd64_gcc14",
             "status": "error"},
        ])
        get_driver.return_value = driver
        answer = _addon_status_answer("are addon tests passing in CMSSW_20_1_X_2026-11-02-2300?", "demo-addon")
        self.assertIn("**1 failing** of 2", answer)
        self.assertIn("| ASAN | `el9_amd64_gcc14` | `error` |", answer)
        self.assertIn("and its sub-IBs", answer)

    def test_definition_and_unrelated_questions_are_left_alone(self):
        self.assertIsNone(_addon_status_answer("what is the latest IB?", "demo-addon"))
        self.assertIsNone(_addon_status_answer("any PR about addon tests failing?", "demo-addon"))


class ClangPackageWarningTests(unittest.TestCase):
    @patch("server.api_server.get_driver")
    def test_subsystem_counts_come_from_named_build(self, get_driver):
        exists = MagicMock()
        exists.single.return_value = {"w": 1}
        driver, _ = _driver_with(
            exists,
            [{"tag": "CMSSW_20_1_X_2026-08-12-2300", "arch": "el9_amd64_gcc13", "key": "k1"}],
            [{"package": "FWCore/Services", "locations": 30, "occurrences": 30},
             {"package": "FWCore/Framework", "locations": 12, "occurrences": 12}],
        )
        get_driver.return_value = driver
        answer = _clang_package_warning_answer("are there any clang warnings in FWCore?")
        self.assertIn("**42** distinct", answer)
        self.assertIn("`CMSSW_20_1_X_2026-08-12-2300` on `el9_amd64_gcc13`", answer)
        self.assertIn("not necessarily the latest IB", answer)

    @patch("server.api_server.get_driver")
    def test_unknown_package_defers(self, get_driver):
        missing = MagicMock()
        missing.single.return_value = None
        driver, _ = _driver_with(missing)
        get_driver.return_value = driver
        self.assertIsNone(_clang_package_warning_answer("clang warnings in Nonexistent/Thing"))

    def test_tag_only_question_is_not_a_package_question(self):
        self.assertIsNone(_clang_package_warning_answer("clang warnings for CMSSW_20_1_X_2026-08-12-2300"))


class RoundTwoFollowUpTests(ShortFollowUpTests):
    def test_release_series_follow_up(self):
        self.remember("status of the latest 17_1 IB", "...")
        self.assertEqual(_resolve_short_follow_up("and 16_1?", self.sid), "status of the latest IB in 16_1")

    def test_release_series_follow_up_replaces_a_tag(self):
        self.remember("what failed in CMSSW_17_1_X_2026-11-02-2300?", "...")
        self.assertEqual(_resolve_short_follow_up("what about 16_1", self.sid),
                         "what failed in the latest IB in 16_1")

    def test_why_after_package_only_overview(self):
        self.remember("what failed in CMSSW_17_1_X_2026-11-02-2300 on el8_aarch64_gcc13?",
                      "**Unit tests** -- 2 failing:\n- `el8_aarch64_gcc13`: `CondTools/SiStrip`, `SimG4Core/Geometry`")
        self.assertEqual(_resolve_short_follow_up("why?", self.sid),
                         "why is CondTools/SiStrip failing in CMSSW_17_1_X_2026-11-02-2300")


class RoundTwoRoutingTests(unittest.TestCase):
    def test_status_of_latest_ib_is_not_just_its_name(self):
        self.assertIsNone(_latest_ib_answer("status of the latest 17_1 IB"))
        self.assertIsNone(_latest_ib_answer("is the latest IB good?"))

    def test_variant_failure_question_is_not_a_definition(self):
        self.assertIsNone(_ib_variant_glossary_answer("what's wrong with the ROOT642 IB?"))
        self.assertIn("ROOT 6.42", _ib_variant_glossary_answer("what does ROOT642 mean?"))

    def test_data_freshness_question(self):
        self.assertTrue(_ABOUT_DATA_RE.search("is the data up to date?"))

    def test_shifter_opener_is_a_shift_summary(self):
        self.assertTrue(RequestContext.extract("what should I check first as a shifter today?").has_shift_summary_request)

    @patch("server.api_server.get_exit_code_names", return_value={31744: "TimeOut", 16640: "OtherCMS"})
    @patch("server.api_server.get_driver")
    def test_timed_out_maps_to_exit_code(self, get_driver, _names):
        driver, session = _driver_with([
            {"tag": "CMSSW_20_1_ASAN_X_2026-11-02-2300", "arch": "el9_amd64_gcc14", "n": 1, "examples": ["539.0"]}],
            MagicMock(single=MagicMock(return_value={"n": 8})))
        get_driver.return_value = driver
        answer = _relval_by_exit_code_answer("which workflows timed out in CMSSW_20_1_X_2026-11-02-2300?")
        self.assertIsNotNone(answer)
        self.assertEqual(session.run.call_args_list[0].kwargs["code"], 31744)


class CompletenessTests(unittest.TestCase):
    @patch("server.api_server.get_driver")
    def test_partly_imported_newest_build_is_skipped(self, get_driver):
        driver, _ = _driver_with([
            {"tag": "CMSSW_20_1_X_2026-11-02-2300", "arch": "el8_amd64_gcc14", "n": 15},
            {"tag": "CMSSW_20_1_X_2026-11-02-2300", "arch": "el9_amd64_gcc14", "n": 30},
            {"tag": "CMSSW_20_1_X_2026-11-02-1100", "arch": "el8_amd64_gcc14", "n": 954},
            {"tag": "CMSSW_20_1_X_2026-11-02-1100", "arch": "el9_amd64_gcc14", "n": 954},
            # the 23:00 build before also had aarch64: a missing architecture is not "partial"
            {"tag": "CMSSW_20_1_X_2026-11-01-2300", "arch": "el8_amd64_gcc14", "n": 954},
            {"tag": "CMSSW_20_1_X_2026-11-01-2300", "arch": "el9_aarch64_gcc14", "n": 954},
            {"tag": "CMSSW_20_1_X_2026-11-01-2300", "arch": "el9_amd64_gcc14", "n": 954},
        ])
        get_driver.return_value = driver
        tag, note = _newest_complete_primary_tag("relval")
        self.assertEqual(tag, "CMSSW_20_1_X_2026-11-02-1100")
        self.assertIn("partly imported", note)


class NewHandlerTests(unittest.TestCase):
    @patch("server.api_server.get_driver")
    def test_known_error_list(self, get_driver):
        driver, _ = _driver_with([
            {"tag": "CMSSW_20_1_X_2026-11-02-2300", "arch": "el9_aarch64_gcc14", "wf": "10804.31", "name": "x",
             "reason": "Triton Server Instance for non-x86_64"}])
        get_driver.return_value = driver
        answer = _known_error_list_answer("list known errors in CMSSW_20_1_X_2026-11-02-2300", "demo-ke")
        self.assertIn("**1 known-error RelVal(s)**", answer)
        self.assertIn("Triton", answer)
        self.assertIsNone(_known_error_list_answer("what is a known error?", "demo-ke"))

    @patch("server.api_server.get_driver")
    def test_architectures_marks_stopped_ones(self, get_driver):
        driver, _ = _driver_with([
            {"series": "20_1", "arch": "el9_amd64_gcc14", "newest": "2026-11-02-2300", "builds": 150},
            {"series": "20_1", "arch": "el9_amd64_gcc13", "newest": "2026-08-12-2300", "builds": 41},
        ])
        get_driver.return_value = driver
        answer = _architectures_answer("which architectures are built for 20_1?")
        self.assertIn("**1 architecture(s)** are currently built", answer)
        self.assertIn("| `el9_amd64_gcc13` | `CMSSW_20_1_X_2026-08-12-2300` | **no** (stopped?)", answer)

    def test_architecture_failure_question_is_left_alone(self):
        self.assertIsNone(_architectures_answer("which architectures failed in the latest IB?"))

    @patch("server.api_server.get_driver")
    def test_pr_author_ranking(self, get_driver):
        driver, _ = _driver_with(
            [{"author": "felicepantaleo", "n": 9, "repos": ["cmssw"], "examples": [52346]}],
            MagicMock(single=MagicMock(return_value={"n": 60})))
        get_driver.return_value = driver
        answer = _pr_author_ranking_answer("who merged the most PRs this week?")
        self.assertIn("| `felicepantaleo` | 9 |", answer)
        self.assertIn("who pressed merge is not in the imported data", answer)
        self.assertIsNone(_pr_author_ranking_answer("who authored PR 52346"))


class VariantAndNotFailingTests(unittest.TestCase):
    def test_workflow_question_naming_a_variant_keeps_the_workflow(self):
        # "is 539.0 failing only in ASAN?" is about the workflow everywhere, not the ASAN IB
        self.assertEqual(_with_variant_ib_context("is 539.0 failing only in ASAN?"),
                         "is 539.0 failing only in ASAN?")

    @patch("server.api_server._newest_complete_primary_tag", return_value=("CMSSW_20_1_X_2026-11-02-1100", ""))
    @patch("server.api_server.get_driver")
    def test_not_failing_without_a_tag_uses_newest_complete_ib_and_the_arch(self, get_driver, _complete):
        driver, session = _driver_with([
            {"tag": "CMSSW_20_1_X_2026-11-02-1100", "arch": "el8_amd64_gcc14", "total": 954, "passed": 952}])
        get_driver.return_value = driver
        answer = _not_failing_answer("which relvals are not failing on el8_amd64_gcc14?")
        self.assertIn("**952 of 954**", answer)
        self.assertEqual(session.run.call_args.kwargs["arch"], "el8_amd64_gcc14")


class TestLogDateTests(unittest.TestCase):
    @patch("server.api_server.get_driver")
    def test_day_without_logs_is_said_with_nearest_days(self, get_driver):
        # found 2026-11-03: "test logs for 2026-10-21" (outage gap) answered for the newest build
        driver, _ = _driver_with([{"tag": t} for t in (
            "CMSSW_20_1_X_2026-10-16-2300", "CMSSW_20_1_X_2026-10-27-1100", "CMSSW_20_1_X_2026-11-02-2300")])
        get_driver.return_value = driver
        answer = _test_log_question_answer("test logs for 2026-10-21", "demo-tl")
        self.assertIn("No raw test logs are imported for `CMSSW_20_1_X` builds of 2026-10-21", answer)
        self.assertIn("2026-10-16 and 2026-10-27", answer)


class PackageSessionScopeTests(unittest.TestCase):
    @patch("server.api_server._package_newest_status_answer", return_value="**Failing now in 4 of 26** ...")
    @patch("server.api_server.get_driver")
    def test_package_passing_in_session_ib_shows_where_it_fails(self, get_driver, newest):
        history = MagicMock(single=MagicMock(return_value={"total": 300, "failed": 40, "first_failed": "2026-10-26-2300"}))
        driver, _ = _driver_with(
            [{"variant": "primary", "arch": "el8_amd64_gcc14", "status": "PASSED", "errors": 0}], history)
        get_driver.return_value = driver
        _latest_ib_contexts["demo-pkg"] = "CMSSW_20_1_X_2026-11-03-1100"
        answer = _package_status_answer("why does SimG4Core/Geometry fail?", "demo-pkg")
        self.assertIn("passes in every record of `CMSSW_20_1_X_2026-11-03-1100`", answer)
        self.assertIn("Failing now in 4 of 26", answer)


class RoundThreeTests(unittest.TestCase):
    def setUp(self):
        _turn_memory.pop("demo-r3", None)

    def test_since_when_after_an_overview(self):
        _turn_memory["demo-r3"] = {"question": "why is 17_1 red?", "route": "capability:failure_overview",
                                   "answer": "- `el8_amd64_gcc13`: **11 failing** -- e.g. `7.4` CosmicsPEAK2018"}
        self.assertEqual(_resolve_short_follow_up("since when?", "demo-r3"), "since when is workflow 7.4 failing")

    def test_shift_summary_phrasings(self):
        for q in ("give me an overview of IB health this week", "did anything get fixed today?",
                  "which unit tests started failing this week?", "summary for the shift meeting"):
            self.assertTrue(RequestContext.extract(q).has_shift_summary_request, q)

    def test_missing_results_captures_the_kind(self):
        self.assertEqual(_MISSING_RESULTS_RE.search("which IBs have no relval results yet?").group(1).lower(), "relval")

    def test_architecture_status_phrasings(self):
        self.assertTrue(_ARCH_LIST_RE.search("what is the status of riscv?"))
        self.assertTrue(_ARCH_LIST_RE.search("is el10 ready?"))

    def test_tracked_question_with_its_own_topic_defers(self):
        self.assertIsNone(_tracked_failures_answer("is there an open issue about DAS errors?", "demo-r3"))
        self.assertIsNone(_tracked_failures_answer("what failed in 17_1?", "demo-r3"))
        self.assertIsNone(_tracked_failures_answer("latest open issue", "demo-r3"))

    @patch("server.api_server.get_driver")
    def test_sub_ib_previous_build_is_its_own_flavour(self, get_driver):
        driver, session = _driver_with(MagicMock(single=MagicMock(return_value={"tag": "CMSSW_20_1_ASAN_X_2026-10-30-2300"})))
        get_driver.return_value = driver
        self.assertEqual(_previous_primary_tag("CMSSW_20_1_ASAN_X_2026-11-02-2300", same_flavour=True),
                         "CMSSW_20_1_ASAN_X_2026-10-30-2300")
        self.assertEqual(session.run.call_args.kwargs["prefix"], "CMSSW_20_1_ASAN_X_")

    @patch("server.api_server.get_driver")
    def test_merged_window_filters_by_topic(self, get_driver):
        driver, _ = _driver_with([
            {"repo": "cmssw", "number": 52316, "title": "[12_4_X] XrdAdaptor: Use inactive sources first",
             "merged_at": "2026-11-03", "author": "makortel", "body": ""},
            {"repo": "cmsdist", "number": 10979, "title": "Update virtualenv", "merged_at": "2026-11-03",
             "author": "cms-sw", "body": ""}])
        get_driver.return_value = driver
        answer = _prs_merged_window_answer("PRs about XrdAdaptor merged this week")
        self.assertIn("#52316", answer)
        self.assertNotIn("#10979", answer)

    @patch("server.api_server.get_driver")
    def test_workflow_log_links(self, get_driver):
        url = ("https://cmssdt.cern.ch/SDT/cgi-bin/logreader/el8_amd64_gcc13/CMSSW_17_1_X_2026-11-03-1100/"
               "pyRelValMatrixLogs/run/140.5611_RunHI2018reMINIAOD/step3_RunHI2018reMINIAOD.log")
        driver, _ = _driver_with([{"tag": "CMSSW_17_1_X_2026-11-03-1100", "arch": "el8_amd64_gcc13", "step": "step3",
                                   "url": url, "category": "ConditionDatabase"}])
        get_driver.return_value = driver
        self.assertIn(url, _workflow_log_answer("show me the log of 140.5611"))
        self.assertIsNone(_workflow_log_answer("why is 140.5611 failing?"))


class RoundFourTests(unittest.TestCase):
    def test_full_date_with_time_becomes_a_timestamp(self):
        self.assertIn("2026-11-02-2300", _normalize_deterministic_question("what changed in the IB of 2026-11-02 at 23:00?"))

    def test_about_data_phrasings(self):
        for q in ("how much data do you have?", "how reliable is this bot?", "can I trust these answers?"):
            self.assertTrue(_ABOUT_DATA_RE.search(q), q)

    def test_better_than_yesterday_is_a_comparison(self):
        self.assertTrue(RequestContext.extract("is the latest IB better than yesterday's?").has_shift_summary_request)

    def test_sanitizer_question_is_not_a_shift_summary(self):
        self.assertFalse(RequestContext.extract("any new sanitizer errors?").has_shift_summary_request)
        self.assertTrue(RequestContext.extract("is the ASAN build getting better?").has_shift_summary_request)

    @patch("server.api_server._store_chat_turns")
    def test_bye(self, _store):
        self.assertEqual(_conversational_response("demo-bye", "bye").answer, "Bye! Ask any time.")

    @patch("server.api_server._newest_series_prefix", return_value="CMSSW_20_1")
    @patch("server.api_server.get_driver")
    def test_exception_ranking(self, get_driver, _prefix):
        driver, _ = _driver_with([{"category": "PluginNotFound", "message": "Unable to find plugin 'HLTTriggerTypeFilter'",
                                   "workflows": 57, "builds": 2, "archs": ["fc42_riscv64_gcc15"]}])
        get_driver.return_value = driver
        answer = _exception_ranking_answer("what's the most common exception this week?")
        self.assertIn("| PluginNotFound |", answer)
        self.assertIn("| 57 | 2 |", answer)
        self.assertIsNone(_exception_ranking_answer("is there a PR for the most common exception?"))

    @patch("server.api_server.get_driver")
    def test_release_series(self, get_driver):
        driver, _ = _driver_with([
            {"series": "20_1", "newest": "2026-11-03-1100", "stamps": ["2026-11-03-1100", "2026-11-02-2300"], "archs": ["a", "b"]},
            {"series": "12_4", "newest": "2026-09-01-0000", "stamps": ["2026-09-01-0000"], "archs": ["a"]}])
        get_driver.return_value = driver
        answer = _release_series_answer("which release series are active?")
        self.assertIn("**1 release series** are being built", answer)
        self.assertIn("| `CMSSW_12_4_X` | `2026-09-01-0000` | 0 | 1 | no |", answer)


class GitHubStalenessTests(unittest.TestCase):
    def setUp(self):
        _GITHUB_COVERAGE_CACHE.clear()

    @patch("server.api_server.get_driver")
    def test_issue_state_is_dated_never_current(self, get_driver):
        # found 2026-11-04: "is issue 51354 still open?" -> "currently open"; closed on GitHub since
        driver, _ = _driver_with(
            MagicMock(single=MagicMock(return_value={"title": "LogicError in EvaluateMuonMVAID", "state": "open",
                                                     "created": "2026-06-29T09:04:55Z", "closed": None,
                                                     "updated": "2026-08-31T10:00:00Z"})),
            MagicMock(single=MagicMock(return_value={"n": 51822, "t": "2026-09-03T21:31:41Z"})),
            MagicMock(single=MagicMock(return_value={"n": None, "t": None, "via": None})))
        get_driver.return_value = driver
        answer = _issue_state_answer("is issue 51354 still open?")
        self.assertIn("`open` in the imported data", answer)
        self.assertIn("reaches #51822 (2026-09-03)", answer)
        self.assertNotIn("currently open", answer)

    @patch("server.api_server.get_driver")
    def test_note_separates_the_export_from_the_web_fetch(self, get_driver):
        # after the 2026-11-04 web fetch a plain max() would claim the *export* reaches today
        driver, _ = _driver_with(
            MagicMock(single=MagicMock(return_value={"n": 51806, "t": "2026-09-03T21:31:41Z"})),
            MagicMock(single=MagicMock(return_value={"n": 52376, "t": "2026-11-04T10:10:51Z",
                                                     "via": "github-web-fetch 2026-11-04"})))
        get_driver.return_value = driver
        note = _github_export_note()
        self.assertIn("full export (with comments) reaches 2026-09-03", note)
        self.assertIn("fetched directly from GitHub on 2026-11-04", note)
        self.assertIn("no comments", note)
        self.assertIn("newest Issue #52376, last update 2026-11-04", note)
        self.assertNotIn("export that reaches #52376", note)

    def test_non_state_questions_are_left_alone(self):
        self.assertIsNone(_issue_state_answer("who opened issue 52274?"))

    def test_unmerged_pr_is_not_shown_as_merged(self):
        # found 2026-11-04: open PR #52384 from the web fetch rendered as "(merged unknown)"
        row = {"repo": "cmssw", "pr_number": 52384, "title": "Fix", "merged_at": None, "state": "open",
               "evidence": ["pr_title"], "matched_files": [], "matched_unit_tests": [], "matched_warnings": 0}
        answer = _render_pr_evidence_rows("ROOT642", [row])
        self.assertIn("not merged, state `open`", answer)
        self.assertNotIn("merged unknown", answer)
        self.assertIn("(merged 2026-11-02)", _render_pr_evidence_rows("X", [dict(row, merged_at="2026-11-02")]))


class KnownErrorListPartialBuildTests(unittest.TestCase):
    @patch("server.api_server._newest_tag_for_arch_word",
           return_value=("CMSSW_20_1_X_2026-11-03-2300", ["el9_aarch64_gcc14"]))
    @patch("server.api_server.get_driver")
    def test_newest_build_without_relvals_on_the_arch_is_not_reported_as_none(self, get_driver, _newest):
        known = {"tag": "CMSSW_20_1_X_2026-11-02-2300", "arch": "el9_aarch64_gcc14", "wf": "1.0", "name": "W",
                 "reason": "tracked"}
        driver, _ = _driver_with(
            [],
            MagicMock(single=MagicMock(return_value={"tag": "CMSSW_20_1_X_2026-11-02-2300"})),
            [known])
        get_driver.return_value = driver
        answer = _known_error_list_answer("known errors list for aarch64 20_1", "s-known-partial")
        self.assertIn("has no RelVal results imported on `el9_aarch64_gcc14` yet", answer)
        self.assertIn("**1 known-error RelVal(s)** in `CMSSW_20_1_X_2026-11-02-2300`", answer)
        self.assertNotIn("No known-error", answer)


class ShiftDigestPrimaryBaselineTests(unittest.TestCase):
    """Found 2026-11-04: 13_0 on el8_amd64_gcc11 listed one HLT unit test as both newly failing and resolved."""

    def test_variant_tag_of_the_primary_is_the_primary_tag(self):
        # a second definition of _variant_tag replaced this one and gave "CMSSW_13_0__X_...", which sorts
        # after every primary tag, so "the previous build" was the latest build itself
        self.assertEqual(_variant_tag("CMSSW_13_0_X_2026-11-03-2300", ""), "CMSSW_13_0_X_2026-11-03-2300")
        self.assertEqual(_variant_tag("CMSSW_13_0_X_2026-11-03-2300", "HLT"), "CMSSW_13_0_HLT_X_2026-11-03-2300")

    @patch("server.api_server._variant_previous_tag", return_value="CMSSW_13_0_X_2026-11-01-0000")
    def test_primary_baseline_excludes_its_sub_ib_rows(self, previous):
        rows = [{"name": "Pkg", "variant": "HLT"}, {"name": "Own", "variant": None}]
        before, note, has_baseline = _variant_diff_baseline(
            "", "el8_amd64_gcc11", "CMSSW_13_0_X_2026-11-01-2300", {}, {}, set(), lambda tag, arch: rows,
            "UnitTest", latest_tag="CMSSW_13_0_X_2026-11-03-2300")
        self.assertEqual(before, [{"name": "Own", "variant": None}])
        self.assertTrue(has_baseline)
        self.assertIn("CMSSW_13_0_X_2026-11-01-0000", note)
        previous.assert_called_once_with("CMSSW_13_0_X_2026-11-03-2300", None, "el8_amd64_gcc11")


class NoDuplicateDefinitionsTests(unittest.TestCase):
    def test_no_module_level_function_replaces_another(self):
        # a second _variant_tag silently replaced the first (2026-11-04) and broke the shift digest's baselines
        root = Path(__file__).resolve().parent.parent
        for name in ("server/api_server.py", "server/agent.py", "server/request_planner.py", "graph/tools.py"):
            tree = ast.parse((root / name).read_text())
            counts = Counter(node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
            self.assertEqual([def_name for def_name, n in counts.items() if n > 1], [], name)


class LeakedReasoningTests(unittest.TestCase):
    def test_planning_lines_are_dropped(self):
        leaked = ('The user asked: "who won the champions league?" This is unrelated. We must respond that we '
                  "cannot answer.\nI'm sorry, but I don't have information on that.")
        self.assertEqual(_strip_leaked_reasoning(leaked), "I'm sorry, but I don't have information on that.")

    def test_a_real_answer_is_untouched(self):
        for text in ("**Issue 1** is open.\nWe must note the user asked this.", "The IB is fine.",
                     "We must respond."):
            self.assertEqual(_strip_leaked_reasoning(text), text)


if __name__ == "__main__":
    unittest.main()
