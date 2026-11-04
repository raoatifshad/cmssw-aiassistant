"""Wrong/dead-end answers found by the unattended tester run of 2026-11-04 (fresh questions about
that day's data: read in full and checked against the graph)."""
import unittest
from unittest.mock import MagicMock, patch

from server import api_server
from server.api_server import (
    _basic_term_answer, _exception_term_answer, _known_error_list_answer, _relvals_on_both_answer, _with_series_change_tags,
    _build_warning_answer, _cross_flavour_comparison_answer, _ib_comparison_answer, _newest_tag_for_arch_word,
    _package_failing_tests_note, _prs_merged_window_answer, _with_full_year_dates, _with_newest_arch_ib,
    _with_newest_flavour_compare_tags, _named_test_answer, _sanitizer_problems_answer, _with_relative_compare_tags,
    _TEST_LOG_QUESTION_RE, _WORKFLOW_WHERE_RE,
    _architectures_answer, _not_failing_answer, _pr_search_terms, _resolve_short_follow_up, _turn_memory,
    _LOG_PROBLEM_PASTE_RE, _ib_failure_ranking_answer, _normalize_deterministic_question,
    _relval_exception_list_answer, _is_latest_ib_question, _loose_arch_choice, _render_metric_deltas,
    _today_falls_back_to_latest,
)


def _driver_with(*results):
    """A get_driver() mock whose session.run returns each result in turn."""
    session = MagicMock()
    session.run.side_effect = list(results)
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    return driver, session


class TodayFallbackTests(unittest.TestCase):
    @patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_17_1_X_2026-11-03-1100"}])
    def test_today_without_a_build_is_the_newest_ib_of_the_named_series(self, latest):
        question, note = _today_falls_back_to_latest("what's wrong with 17_1 today?")
        self.assertIn("CMSSW_17_1_X_2026-11-03-1100", question)
        self.assertNotIn("today", question)
        self.assertEqual(latest.call_args.args[1], (17, 1))
        self.assertIn("No IB from today", note)

    @patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-03-1100"}])
    def test_a_named_date_or_hour_or_yesterday_stays_a_no(self, _):
        self.assertIsNone(_today_falls_back_to_latest("what failed on 2026-11-04?"))
        self.assertIsNone(_today_falls_back_to_latest("failures in the 11:00 IB today"))
        self.assertIsNone(_today_falls_back_to_latest("what failed yesterday?"))


class MetricDeltaWordingTests(unittest.TestCase):
    METRICS = [("Failed RelVals", "f"), ("RelVals run", "n")]

    def test_no_failure_claim_when_a_failure_count_moved(self):
        text = "\n".join(_render_metric_deltas({"f": 1, "n": 954}, {"f": 2, "n": 954}, self.METRICS, "A", "B"))
        self.assertNotIn("No new failures", text)
        self.assertIn("Unchanged: RelVals run still 954", text)

    def test_claim_kept_when_nothing_moved(self):
        text = "\n".join(_render_metric_deltas({"f": 1, "n": 954}, {"f": 1, "n": 954}, self.METRICS, "A", "B"))
        self.assertIn("No new failures/errors", text)


class LooseArchChoiceTests(unittest.TestCase):
    ARCHS = ["el9_amd64_gcc14", "el9_amd64_gcc15", "el9_amd64_gcc16"]

    def test_compiler_picks_the_one_architecture(self):
        self.assertEqual(_loose_arch_choice("and only gcc16?", self.ARCHS), "el9_amd64_gcc16")

    def test_several_or_no_matches_stay_unresolved(self):
        self.assertIsNone(_loose_arch_choice("only el9", self.ARCHS))
        self.assertIsNone(_loose_arch_choice("only gcc12", self.ARCHS))
        self.assertIsNone(_loose_arch_choice("which one is best?", self.ARCHS))


class IbFailureRankingTests(unittest.TestCase):
    @patch("server.api_server._newest_series_prefix", return_value="CMSSW_20_1")
    @patch("server.api_server.get_driver")
    def test_ranks_per_tag_and_architecture(self, get_driver, _):
        driver, session = _driver_with([
            {"tag": "CMSSW_20_1_X_2026-11-02-2300", "arch": "el9_aarch64_gcc14", "failed": 51},
            {"tag": "CMSSW_20_1_ASAN_X_2026-11-02-2300", "arch": "el9_amd64_gcc14", "failed": 3}])
        get_driver.return_value = driver
        text = _ib_failure_ranking_answer("which IBs failed the most relvals this week?")
        self.assertIn("| `CMSSW_20_1_X_2026-11-02-2300` | `el9_aarch64_gcc14` | 51 |", text)
        self.assertIn("the last 7 days", text)
        self.assertIn("known_error", session.run.call_args.args[0])

    def test_other_questions_are_not_taken(self):
        self.assertIsNone(_ib_failure_ranking_answer("which package has the most unit test failures?"))
        self.assertIsNone(_ib_failure_ranking_answer("which IB is the newest?"))
        self.assertIsNone(_ib_failure_ranking_answer("most failing workflows in CMSSW_20_1_X_2026-11-02-2300"))


class LatestSeriesTests(unittest.TestCase):
    def test_latest_with_only_a_release_series(self):
        self.assertTrue(_is_latest_ib_question("build errors in the latest 20_1"))
        self.assertTrue(_is_latest_ib_question("newest CMSSW_17_1_X, what failed"))
        self.assertFalse(_is_latest_ib_question("the latest PR about ROOT"))


class ShortLogLineTests(unittest.TestCase):
    def test_typed_log_lines_are_searched_in_the_test_logs(self):
        for text in ("heap-use-after-free edm::storage::File::syswrite",
                     "global-buffer-overflow in HGCalGeometry::getNewCorners, seen in ASAN logs. known?",
                     "UCX ERROR ibv_create_ah in CMSSW_16_1_X_2026-11-03-1100 -- is it new?",
                     "Warning: Could not download index.php from CERNBox: HTTP Error 429: Too Many Requests",
                     "SLHA::readFile Warning: line 103 - read error or empty line"):
            self.assertTrue(_LOG_PROBLEM_PASTE_RE.search(text), text)

    def test_ordinary_questions_are_not(self):
        for text in ("why is the build failing with errors?", "which warnings are new in the latest IB?",
                     "what is edm::Event used for?", "how many errors in 539.0?"):
            self.assertFalse(_LOG_PROBLEM_PASTE_RE.search(text), text)


class ExceptionListTests(unittest.TestCase):
    @patch("server.api_server.get_driver")
    def test_groups_per_build_and_message_and_names_silent_failures(self, get_driver):
        tag = "CMSSW_20_1_X_2026-11-02-2300"
        sub = "CMSSW_20_1_ROOT6_X_2026-11-02-2300"
        arch = "el9_aarch64_gcc14"
        exc = lambda t, wf: {"tag": t, "arch": arch, "workflow_id": wf, "exception": "ExternalLHEProducer: Child failed",
                             "category": "ExternalLHEProducer", "source_url": "http://x"}
        driver, _ = _driver_with(
            [{"tag": tag, "arch": arch, "workflow_id": "1360.0", "exitcode": 256, "known_error": True},
             {"tag": tag, "arch": arch, "workflow_id": "1361.0", "exitcode": 256, "known_error": True},
             {"tag": tag, "arch": arch, "workflow_id": "539.0", "exitcode": 31744, "known_error": False}],
            [exc(tag, "1360.0"), exc(tag, "1361.0"), exc(sub, "1360.0")])
        get_driver.return_value = driver
        text = _relval_exception_list_answer("why are they failing", {"tag": tag, "arch": arch})
        self.assertIn("and its sub-IBs", text)
        self.assertIn("2 workflows: `1360.0`, `1361.0`", text)
        self.assertIn("2 of them known errors", text)
        self.assertIn(f"`{sub}` `{arch}` workflow `1360.0`", text)
        self.assertIn("`539.0` (exit 31744)", text)
        self.assertEqual(text.count("Child failed"), 2)

    def test_exceptions_in_a_tag_is_a_relval_exception_question(self):
        self.assertIn("relval exceptions in CMSSW_20_1_X_2026-11-03-1100",
                      _normalize_deterministic_question("exceptions in CMSSW_20_1_X_2026-11-03-1100"))


class ShortFollowUpTests(unittest.TestCase):
    def setUp(self):
        self.sid = "tester-1104"
        self.addCleanup(_turn_memory.pop, self.sid, None)

    def _after(self, question, answer="", route="deterministic"):
        _turn_memory[self.sid] = {"question": question, "answer": answer, "route": route}

    def test_what_does_it_do_after_a_pr(self):
        self._after("PR 52366 merged in which IB?")
        self.assertEqual(_resolve_short_follow_up("what does it do?", self.sid), "what does PR 52366 do")

    def test_that_exit_code_after_an_exit_code_question(self):
        self._after("which workflows have exit code 25600 in the latest 20_1 IB?")
        self.assertEqual(_resolve_short_follow_up("what does that exit code mean?", self.sid),
                         "what does exit code 25600 mean")

    def test_which_step_after_a_workflow(self):
        self._after("why?", "Workflow `11024.2` (`TTbar`) in `CMSSW_17_1_X_2026-11-03-1100`:")
        self.assertEqual(_resolve_short_follow_up("which step?", self.sid),
                         "why is workflow 11024.2 failing in CMSSW_17_1_X_2026-11-03-1100")

    def test_any_new_only_after_test_log_problems(self):
        self._after("hidden problems in the newest IB", route="deterministic:test_log_changes")
        self.assertEqual(_resolve_short_follow_up("any of them new?", self.sid), "hidden problems in the newest IB")
        self._after("what is the latest IB?", route="deterministic")
        self.assertIsNone(_resolve_short_follow_up("any of them new?", self.sid))


class PhrasingTests(unittest.TestCase):
    def test_hidden_problems_are_not_the_passing_list(self):
        question = "hidden problems in passed unit tests in the newest IB"
        self.assertTrue(_TEST_LOG_QUESTION_RE.search(question))
        self.assertIsNone(_not_failing_answer(question))

    def test_cmsdist_number_is_a_pr(self):
        self.assertIn("cmsdist PR 10977", _normalize_deterministic_question("cmsdist 10977 status?"))
        self.assertNotIn("PR", _normalize_deterministic_question("CMSSW_20_1 and cmssw 12345.5"))

    def test_worry_is_a_failure_question(self):
        self.assertIn("what is failing", _normalize_deterministic_question("should I worry about something in the latest IB?"))

    def test_have_exit_code_filters(self):
        with patch("server.api_server._is_latest_ib_question", return_value=False):
            self.assertIsNone(api_server._relval_by_exit_code_answer("which workflows have the most failures?"))

    def test_timing_out_is_a_where_question(self):
        self.assertTrue(_WORKFLOW_WHERE_RE.search("is 539.0 timing out everywhere?"))

    def test_global_tag_keeps_its_digits(self):
        terms = _pr_search_terms("are there PRs about the missing global tag 170X_upgrade2018_realistic_HI_v1?")
        self.assertIn("170X_upgrade2018_realistic_HI_v1", terms)
        self.assertNotIn("X_upgrade2018_realistic_HI_v1", terms)

    @patch("server.api_server.get_driver")
    def test_architectures_of_one_tag(self, get_driver):
        driver, _ = _driver_with([{"arch": "el9_amd64_gcc16"}, {"arch": "el9_amd64_gcc14"}])
        get_driver.return_value = driver
        text = _architectures_answer("which architectures does CMSSW_20_1_CPP23_X_2026-11-03-1100 have?")
        self.assertIn("**2 architecture(s)**: `el9_amd64_gcc14`, `el9_amd64_gcc16`", text)


class NamedTestTests(unittest.TestCase):
    FULL = "FWCore/Integration/TestIntegrationEventSetup"

    def _run(self, question, runs, problems):
        driver, session = _driver_with(runs, problems)
        with patch("server.api_server.get_driver", return_value=driver):
            return _named_test_answer(question), session

    def test_failed_run_with_its_log_problem(self):
        tag = "CMSSW_20_1_X_2026-11-02-2300"
        text, session = self._run("why does TestIntegrationEventSetup fail on aarch64?", [
            {"tag": tag, "arch": "el9_aarch64_gcc14", "run_variant": "", "failed": [self.FULL], "passed": [], "other": []},
            {"tag": "CMSSW_20_1_X_2026-11-01-2300", "arch": "el9_aarch64_gcc14", "run_variant": "", "failed": [], "passed": [self.FULL], "other": []},
        ], [{"kind": "fatal_exception", "category": "LogicError", "summary": "Fatal exception 'LogicError': boom"}])
        self.assertIn("not passed in 1 of 2", text)
        self.assertIn(f"`{tag}` | `el9_aarch64_gcc14`", text)
        self.assertIn("boom", text)
        self.assertEqual(session.run.call_args_list[0].kwargs["family"], "aarch64")

    def test_passed_says_so_and_shows_what_its_log_has(self):
        text, _ = self._run("why is TestIntegrationEventSetup failing", [
            {"tag": "CMSSW_20_1_X_2026-11-01-2300", "arch": "el9_amd64_gcc14", "run_variant": "", "failed": [], "passed": [self.FULL], "other": []},
        ], [{"kind": "sanitizer", "category": "x", "summary": "AddressSanitizer: overflow"}])
        self.assertIn("passed in all 1", text)
        self.assertIn("It passed, but its log", text)

    def test_no_such_test_leaves_the_question_to_others(self):
        driver, _ = _driver_with([])
        with patch("server.api_server.get_driver", return_value=driver):
            self.assertIsNone(_named_test_answer("why does TestNothingLikeThis fail?"))

    def test_questions_without_a_test_name_do_no_lookup(self):
        with patch("server.api_server.get_driver") as get_driver:
            self.assertIsNone(_named_test_answer("why does the latest IB fail?"))
            self.assertIsNone(_named_test_answer("what is TestIntegrationEventSetup used for"))
            get_driver.assert_not_called()


class SanitizerListTests(unittest.TestCase):
    def _session(self, *results):
        driver, session = _driver_with(*results)
        return patch("server.api_server.get_driver", return_value=driver), session

    def test_uses_the_sanitizer_ib_of_the_conversation_and_lists_leaks_last(self):
        tag = "CMSSW_20_1_ASAN_X_2026-11-02-2300"
        count = MagicMock()
        count.single.return_value = {"n": 4}
        patcher, session = self._session(count, [
            {"category": "LeakSanitizer: detected memory leaks", "summary": "leak in bash", "arch": "a", "owners": ["x", "y"]},
            {"category": "AddressSanitizer: heap-use-after-free", "summary": "use after free", "arch": "a", "owners": ["z"]},
            {"category": "AddressSanitizer: heap-use-after-free", "summary": "use after free", "arch": "b", "owners": ["z"]},
        ])
        with patch.dict(api_server._latest_ib_contexts, {"s": tag}), patcher:
            text = _sanitizer_problems_answer("what about the sanitizer problems in the test logs?", "s")
        self.assertIn(f"`{tag}`", text)
        self.assertLess(text.index("use after free"), text.index("leak in bash"))
        self.assertIn("(1 besides leak reports)", text)

    def test_passed_tests_filter_and_definitions(self):
        tag = "CMSSW_20_1_ASAN_X_2026-11-02-2300"
        count = MagicMock()
        count.single.return_value = {"n": 4}
        patcher, session = self._session(count, [])
        with patcher:
            text = _sanitizer_problems_answer(f"are there sanitizer errors in passed tests in {tag}?", "s")
        self.assertIn("inside tests reported as PASSED", text)
        self.assertIs(session.run.call_args.kwargs["only_passed"], True)
        self.assertIsNone(_sanitizer_problems_answer("what does the ASAN error mean?", "s"))


class RelativeCompareTests(unittest.TestCase):
    @patch("server.api_server._newest_series_prefix", return_value="CMSSW_20_1")
    @patch("server.api_server.get_driver")
    def test_flavour_not_built_that_day_uses_its_two_newest_builds_up_to_it(self, get_driver, _):
        driver, _session = _driver_with([{"tag": f"CMSSW_20_1_CLANG_X_2020-01-0{d}-2300"} for d in (1, 2, 5)])
        get_driver.return_value = driver
        self.assertEqual(_with_relative_compare_tags("compare CLANG yesterday with the day before"),
                         "compare CMSSW_20_1_CLANG_X_2020-01-02-2300 and CMSSW_20_1_CLANG_X_2020-01-05-2300")


class CompileWarningsTests(unittest.TestCase):
    @patch("server.api_server.get_driver")
    def test_compile_warnings_are_the_build_status_counts(self, get_driver):
        driver, _ = _driver_with([{"arch": "el9_amd64_gcc16", "status": "warning", "comp_warnings": 439,
                                   "ignored_warnings": 882, "is_default": False}])
        get_driver.return_value = driver
        text = _build_warning_answer("how many compile warnings in CMSSW_20_1_CPP23_X_2026-11-03-1100 el9_amd64_gcc16?")
        self.assertIn("**439** compilation warning(s)", text)

    def test_across_flavours_is_left_to_the_build_status_table(self):
        self.assertIsNone(_build_warning_answer("which builds have build warnings in CMSSW_20_1_X_2026-11-02-2300"))


class ComparisonPhrasingTests(unittest.TestCase):
    def test_difference_between_two_tags_is_a_comparison(self):
        text = _normalize_deterministic_question(
            "difference between CMSSW_20_1_X_2026-11-02-1100 and CMSSW_20_1_X_2026-11-02-2300 on el9_amd64_gcc14")
        self.assertTrue(text.startswith("compare CMSSW_20_1_X_2026-11-02-1100"))

    def test_newest_builds_of_two_flavours(self):
        session = MagicMock()
        session.run.return_value = [{"tag": t} for t in (
            "CMSSW_20_1_ROOT6_X_2026-11-02-2300", "CMSSW_20_1_ROOT6_X_2026-11-01-2300", "CMSSW_20_1_X_2026-11-02-2300",
            "CMSSW_20_1_X_2026-11-03-1100", "CMSSW_20_1_ASAN_X_2026-11-02-2300")]
        self.assertEqual(_with_newest_flavour_compare_tags("compare latest ROOT6 with the primary el9_amd64_gcc14", session, "CMSSW_20_1"),
                         "compare CMSSW_20_1_ROOT6_X_2026-11-02-2300 and CMSSW_20_1_X_2026-11-02-2300 el9_amd64_gcc14")
        self.assertIn("CMSSW_20_1_ASAN_X_2026-11-02-2300", _with_newest_flavour_compare_tags("compare ASAN and ROOT6 latest", session, "CMSSW_20_1"))
        self.assertEqual(_with_newest_flavour_compare_tags("what does ROOT6 compare to", session, "CMSSW_20_1"), "what does ROOT6 compare to")

    @patch("server.api_server.get_driver")
    def test_cross_flavour_lists_what_fails_in_each(self, get_driver):
        tags = ["CMSSW_20_1_ROOT6_X_2026-11-02-2300", "CMSSW_20_1_X_2026-11-02-2300"]
        relval = lambda i: {"id": i, "name": "wf" + i, "exitcode": 31744}
        record = lambda ids, units: {"arch": "el9_amd64_gcc14", "relval_total": 954, "relvals": [relval(i) for i in ids],
                                     "unit_total": 379, "units": units}
        driver, _ = _driver_with([record(["539.0", "10805.31"], ["FWIO/RNTupleTempTests"])], [record(["537.0", "539.0"], [])])
        get_driver.return_value = driver
        text, _ = _ib_comparison_answer(f"compare {tags[0]} and {tags[1]} el9_amd64_gcc14")
        self.assertIn("## ROOT6 vs primary on `el9_amd64_gcc14`", text)
        self.assertIn("**RelVals failing only in ROOT6:** `10805.31`", text)
        self.assertIn("**RelVals failing only in primary:** `537.0`", text)
        self.assertIn("**RelVals failing in both:** `539.0`", text)

    @patch("server.api_server.get_driver")
    def test_cross_flavour_asks_only_for_architectures_both_have(self, get_driver):
        record = lambda arch: {"arch": arch, "relval_total": 1, "relvals": [], "unit_total": 1, "units": []}
        driver, _ = _driver_with([record("a"), record("b"), record("c")], [record("b"), record("c")])
        get_driver.return_value = driver
        text, actions = _cross_flavour_comparison_answer(["CMSSW_20_1_ROOT6_X_2026-11-02-2300", "CMSSW_20_1_X_2026-11-02-2300"], None)
        self.assertIn("Both have: `b`, `c`", text)
        self.assertEqual(len(actions), 2)


class MoreRoutingTests(unittest.TestCase):
    def test_short_timestamps_get_the_year(self):
        self.assertRegex(_with_full_year_dates("warnings in 11-02-2300?"), r"20\d\d-11-02-2300\?")
        self.assertRegex(_with_full_year_dates("IB from 11-03 11:00"), r"20\d\d-11-03-1100")
        self.assertEqual(_with_full_year_dates("CMSSW_20_1_X_2026-11-03-1100"), "CMSSW_20_1_X_2026-11-03-1100")

    def test_error_class_in_logs_and_fatal_exception_lines_are_log_searches(self):
        for text in ("Fatal exception 'ConfigFileReadError': unknown python problem occurred.",
                     "ConfigFileReadError KeyError Run4D500", "IndexError in Madgraph relval logs, how often?"):
            self.assertTrue(_LOG_PROBLEM_PASTE_RE.search(text), text)

    @patch("server.api_server.get_driver")
    def test_newest_ib_on_an_architecture_is_not_the_newest_ib(self, get_driver):
        driver, _ = _driver_with([{"tag": "CMSSW_20_1_X_2026-11-02-2300", "arch": "el9_aarch64_gcc14"},
                                  {"tag": "CMSSW_20_1_X_2026-11-01-2300", "arch": "el9_aarch64_gcc14"}])
        get_driver.return_value = driver
        self.assertEqual(_with_newest_arch_ib("known errors in the newest aarch64 IB"),
                         "known errors in the newest aarch64 IB CMSSW_20_1_X_2026-11-02-2300 el9_aarch64_gcc14")

    @patch("server.api_server.get_driver")
    def test_package_note_names_the_failing_tests_and_their_log_problem(self, get_driver):
        driver, _ = _driver_with([{"t": "FWCore/Integration/TestX"}], [{"summary": "Fatal exception 'LogicError': boom"}])
        get_driver.return_value = driver
        text = _package_failing_tests_note("FWCore/Integration", [("CMSSW_20_1_X_2026-11-02-2300", "el9_aarch64_gcc14")])
        self.assertIn("`FWCore/Integration/TestX` -- Fatal exception 'LogicError': boom", text)

    @patch("server.api_server.get_driver")
    def test_package_note_is_none_without_imported_logs(self, get_driver):
        driver, _ = _driver_with([])
        get_driver.return_value = driver
        self.assertIsNone(_package_failing_tests_note("A/B", [("CMSSW_20_1_X_2026-11-02-2300", "x")]))

    @patch("server.api_server.get_driver")
    def test_prs_merged_since_a_weekday(self, get_driver):
        driver, session = _driver_with([{"repo": "cmssw", "number": 52331, "title": "T", "merged_at": "2026-11-03T09:19:04", "author": "a"}])
        get_driver.return_value = driver
        text = _prs_merged_window_answer("PRs merged into 20_1 since monday")
        self.assertIn("**1 PR(s)** merged for `CMSSW_20_1`", text)
        self.assertIn("cmssw #52331", text)
        self.assertEqual(session.run.call_args.kwargs["release"], "CMSSW_20_1")
        self.assertIsNone(_prs_merged_window_answer("PRs merged in CMSSW_20_1_X_2026-11-02-2300 this week"))

    def test_status_of_a_series_and_summary_for_shifters(self):
        self.assertEqual(_normalize_deterministic_question("status of 20_0"), "what is failing in the latest 20_0 IB")
        self.assertIn("shift summary", _normalize_deterministic_question("summary for shifters of the latest 2 IBs"))


class RoundThreeTests(unittest.TestCase):
    def test_sanitizer_list_leaves_github_issue_questions_alone(self):
        with patch("server.api_server.get_driver") as get_driver:
            self.assertIsNone(_sanitizer_problems_answer("is there an issue about IOPool ASAN test failures?", "s"))
            self.assertIsNone(_sanitizer_problems_answer("how many open issues mention ASAN?", "s"))
            get_driver.assert_not_called()

    @patch("server.api_server.get_driver")
    def test_on_both_architectures_is_the_intersection(self, get_driver):
        driver, _ = _driver_with([
            {"arch": "el8_amd64_gcc14", "total": 954, "failed": ["539.0", "10805.31"], "names": []},
            {"arch": "el9_amd64_gcc14", "total": 954, "failed": ["539.0"], "names": []},
            {"arch": "el9_aarch64_gcc14", "total": 0, "failed": [], "names": []}])
        get_driver.return_value = driver
        text = _relvals_on_both_answer("which workflows failed on both el8 and el9 in CMSSW_20_1_X_2026-11-03-1100?")
        self.assertIn("**1 RelVal(s)**", text)
        self.assertIn("`539.0`", text)
        self.assertNotIn("10805.31", text)
        self.assertIsNone(_relvals_on_both_answer("which workflows failed in CMSSW_20_1_X_2026-11-03-1100?"))

    @patch("server.api_server.get_driver")
    def test_exception_word_shows_what_is_recorded_and_says_it_is_no_definition(self, get_driver):
        driver, _ = _driver_with(
            [{"category": "ExternalFailure", "message": "SonicCallFailed: call failed", "n": 33,
              "workflows": ["10805.31"], "wf_count": 1, "last": "2026-11-03-1100"}],
            [{"why": "Triton Server Instance for non-x86_64"}])
        get_driver.return_value = driver
        text = _exception_term_answer("what does SonicCallFailed mean?")
        self.assertIn("not an explanation", text)
        self.assertIn("33 record(s)", text)
        self.assertIn("non-x86_64", text)

    def test_exception_word_does_no_lookup_for_other_questions(self):
        with patch("server.api_server.get_driver") as get_driver:
            self.assertIsNone(_exception_term_answer("which PR fixes SonicCallFailed?"))
            self.assertIsNone(_exception_term_answer("what is an IB?"))
            get_driver.assert_not_called()

    @patch("server.api_server._newest_tag_for_arch_word", return_value=("CMSSW_20_1_X_2026-11-02-2300", ["el9_aarch64_gcc14"]))
    @patch("server.api_server.get_driver")
    def test_known_errors_for_an_architecture_use_its_newest_build(self, get_driver, _):
        driver, session = _driver_with([])
        get_driver.return_value = driver
        _known_error_list_answer("known errors list for aarch64 20_1", "s")
        self.assertEqual(session.run.call_args.kwargs["tag"], "CMSSW_20_1_X_2026-11-02-2300")
        self.assertEqual(session.run.call_args.kwargs["arch"], "el9_aarch64_gcc14")

    @patch("server.api_server._previous_primary_tag", return_value="CMSSW_17_1_X_2026-11-02-2300")
    @patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_17_1_X_2026-11-03-1100"}])
    def test_what_changed_in_a_series_names_its_two_builds(self, *_):
        self.assertEqual(_with_series_change_tags("what changed in 17_1 between the last two builds?"),
                         "what changed between CMSSW_17_1_X_2026-11-02-2300 and CMSSW_17_1_X_2026-11-03-1100")
        self.assertEqual(_with_series_change_tags("what changed in 17_1 this week?"), "what changed in 17_1 this week?")

    def test_difference_between_das_error_and_failed_is_a_definition(self):
        text = _basic_term_answer("what's the difference between DAS_ERROR and FAILED?")
        self.assertIn("DAS error", text)
        self.assertIn("**FAILED**", text)

    def test_timing_out_and_signals_are_about_the_workflow(self):
        self.assertTrue(_WORKFLOW_WHERE_RE.search("537.0 SIGSEGV or SIGABRT, what is the pattern?"))
        self.assertIsNone(api_server._exit_code_meaning_answer("537.0 SIGSEGV or SIGABRT, what is the pattern?"))

    def test_more_normalizations(self):
        self.assertIn("what is newly failing", _normalize_deterministic_question("any regressions in unit tests in the latest IB"))
        self.assertIn("with DAS errors", _normalize_deterministic_question("list DAS errors in latest IB"))
        self.assertIn("global-buffer-overflow", _normalize_deterministic_question("is the global buffer overflow new?"))


if __name__ == "__main__":
    unittest.main()
