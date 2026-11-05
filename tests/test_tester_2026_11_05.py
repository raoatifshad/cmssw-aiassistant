"""Wrong/dead-end answers found by the unattended tester run of 2026-11-05 (fresh questions about
that day's data: read in full and checked against the graph)."""
import unittest
from unittest.mock import MagicMock, patch

from server import api_server
from server.api_server import (
    _EARLY_LOG_LINE_RE, _LOG_PROBLEM_PASTE_RE, _latest_ib_contexts, _package_status_answer, _packages_named_loosely,
    _paste_follow_up_answer, _resolve_short_follow_up, _turn_memory, _default_tag_notes, _with_arch_missing_from_newest,
)
from graph import tools


def _driver_with(*results):
    """A get_driver() mock whose session.run returns each result in turn."""
    session = MagicMock()
    session.run.side_effect = list(results)
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    return driver, session


class PastedLogLineTests(unittest.TestCase):
    LINES = [
        "2026/11/04 17:18:35 ERROR failed to parse X509 proxy: crypto/tls: failed to parse key",
        "Is DAS broken? 2026/11/04 17:18:35 ERROR failed to parse X509 proxy",
        "[2026-11-04 21:31:41,747] ERROR: Caught exception when trying to connect to cms-conddb-dev.cern.ch",
        "[2026-11-04 07:01:41.714696 +0100][Error  ][AsyncSock         ] [st-096-hh151811.cern.ch:1095.0] Socket error",
        "g++: internal compiler error: Bus error signal terminated program cc1plus",
        "Assertion failed: __n < this->size()",
    ]

    def test_recognised_as_pastes_before_the_date_handling(self):
        for line in self.LINES:
            self.assertTrue(_LOG_PROBLEM_PASTE_RE.search(line), line)
            self.assertTrue(_EARLY_LOG_LINE_RE.search(line), line)

    def test_questions_with_a_date_are_not_pastes(self):
        for question in ("what failed on 2026-11-04?", "what failed on 2026-11-04 at 11:00?",
                         "errors in the IB of 2026-11-04", "shift summary for 2026-11-04"):
            self.assertFalse(_EARLY_LOG_LINE_RE.search(question), question)
            self.assertFalse(_LOG_PROBLEM_PASTE_RE.search(question), question)

    def test_timestamp_and_typed_words_are_not_searched_word_for_word(self):
        self.assertEqual(
            tools._LOG_LINE_STAMP_RE.sub("", "is das broken? 2026/11/04 17:18:35 error failed to parse x509 proxy"),
            "error failed to parse x509 proxy")
        self.assertEqual(tools._LOG_LINE_STAMP_RE.sub("", "[2026-11-04 21:31:41,747] error: caught"), "error: caught")
        self.assertEqual(tools._LOG_LINE_STAMP_RE.sub("", "failed to parse x509 proxy"), "failed to parse x509 proxy")

    @patch("graph.tools._log_problem_details", side_effect=lambda chosen, match: [{"sid": s, "match": match[s][0]} for s in chosen])
    @patch("graph.tools._channel")
    @patch("graph.tools.get_driver")
    def test_a_line_without_identifiers_is_still_searched_as_a_phrase(self, _driver, channel, _details):
        channel.return_value = [{"sid": "s1", "text": "2026/11/03 10:00:00 ERROR failed to parse x509 proxy: bad",
                                 "summary": "", "owners": ["A/B/t"], "category": None}]
        report = tools.search_test_log_problems_report("2026/11/04 17:18:35 ERROR failed to parse x509 proxy: bad")
        queries = [call.args[2].get("q", "") for call in channel.call_args_list]
        self.assertTrue(any('"error failed to parse x509 proxy: bad"' in q for q in queries), queries)
        self.assertEqual([p["match"] for p in report["problems"]], ["contains"])


class PasteFollowUpTests(unittest.TestCase):
    PROBLEM = {"match": "contains", "ib_tags": ["CMSSW_20_1_X_2026-11-03-2300"], "architectures": ["el10_amd64_gcc15"],
               "tests": {"HeterogeneousTest/ROCmDevice/testA": {"suite": "unit", "passed": 0, "failed": 1, "other": 0}},
               "series_gone": [], "series_total": 1}

    def setUp(self):
        _turn_memory["s"] = {"question": "Fatal exception 'PluginNotFound': ...", "route": "deterministic:test_log_problem",
                             "answer": "**Found in the imported IB test logs** -- problems ..."}
        self.addCleanup(_turn_memory.pop, "s", None)
        patcher = patch("server.api_server.search_test_log_problems_report", return_value={"problems": [self.PROBLEM]})
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch("server.api_server.test_log_coverage", return_value={"first": "2026-10-01-2300"})
    def test_since_when(self, _):
        answer = _paste_follow_up_answer("since when?", "s")
        self.assertIn("first in `CMSSW_20_1_X_2026-11-03-2300`", answer)
        self.assertIn("start at 2026-10-01,", answer)

    def test_which_tests(self):
        answer = _paste_follow_up_answer("which tests show this?", "s")
        self.assertIn("`HeterogeneousTest/ROCmDevice/testA` (reported failed)", answer)

    def test_architecture_scope(self):
        self.assertIn("only on `el10_amd64_gcc15`", _paste_follow_up_answer("is that on el10 only?", "s"))
        self.assertIn("No: this error is not on `aarch64`", _paste_follow_up_answer("on aarch64 too?", "s"))

    def test_other_questions_and_other_turns_are_left_alone(self):
        self.assertIsNone(_paste_follow_up_answer("is that a known error?", "s"))
        self.assertIsNone(_paste_follow_up_answer("what's the latest IB?", "s"))
        _turn_memory["s"]["answer"] = "RelVal results for ..."
        self.assertIsNone(_paste_follow_up_answer("since when?", "s"))


class PackageInReleaseSeriesTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(_latest_ib_contexts.pop, "p", None)

    def _rows(self):
        return [
            {"tag": "CMSSW_17_0_X_2026-11-04-1100", "variant": "primary", "arch": "el8_amd64_gcc13", "status": "FAILED", "errors": 5},
            {"tag": "CMSSW_17_0_X_2026-11-03-2300", "variant": "primary", "arch": "el8_amd64_gcc13", "status": "PASSED", "errors": 0},
            {"tag": "CMSSW_20_1_X_2026-10-29-2300", "variant": "primary", "arch": "fc42_riscv64_gcc15", "status": "FAILED", "errors": 5},
        ]

    @patch("server.api_server._package_failing_tests_note", return_value=None)
    def test_the_named_series_wins_over_the_session_ib(self, _):
        driver, _session = _driver_with(self._rows())
        _latest_ib_contexts["p"] = "CMSSW_20_1_X_2026-11-04-1100"
        with patch("server.api_server.get_driver", return_value=driver):
            answer = _package_status_answer("why did TauAnalysis/MCEmbeddingTools fail in 17_0?", "p")
        self.assertIn("CMSSW_17_0 IB flavour", answer)
        self.assertIn("failing in `CMSSW_17_0_X_2026-11-04-1100`", answer)
        self.assertNotIn("riscv", answer)

    def test_a_series_without_that_package_says_so(self):
        driver, _session = _driver_with(self._rows())
        with patch("server.api_server.get_driver", return_value=driver):
            answer = _package_status_answer("is TauAnalysis/MCEmbeddingTools failing in 15_1?", "p")
        self.assertIn("no imported unit-test record in `CMSSW_15_1`", answer)

    @patch("server.api_server._package_failing_tests_note", return_value="NOTE")
    def test_which_test_after_it_passes_again_names_the_last_failure(self, note):
        rows = [
            {"tag": "CMSSW_20_1_X_2026-11-03-2300", "variant": "primary", "arch": "el9_aarch64_gcc14", "status": "PASSED", "errors": 0},
            {"tag": "CMSSW_20_1_X_2026-11-02-2300", "variant": "primary", "arch": "el9_aarch64_gcc14", "status": "FAILED", "errors": 1},
        ]
        driver, _session = _driver_with(rows)
        with patch("server.api_server.get_driver", return_value=driver):
            answer = _package_status_answer("FWCore/Integration failing on aarch64, which test?", "p")
        self.assertIn("NOTE", answer)
        self.assertEqual(note.call_args.args[1], [("CMSSW_20_1_X_2026-11-02-2300", "el9_aarch64_gcc14")])


class LooseNamesTests(unittest.TestCase):
    NAMES = ["HeterogeneousTest/ROCmDevice", "HeterogeneousTest/ROCmKernel", "RecoTracker/LSTCore", "FWIO/RNTupleTempTests",
             "SimG4Core/Geometry", "Geometry/CommonDetUnit"]

    def setUp(self):
        patcher = patch("server.api_server._known_package_names", return_value=self.NAMES)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_exact_and_prefix(self):
        self.assertEqual(_packages_named_loosely("did LSTCore pass on gcc16?"), ("LSTCore", ["RecoTracker/LSTCore"]))
        self.assertEqual(_packages_named_loosely("what's up with RNTupleTempTests in ROOT6?"), ("RNTupleTempTests", ["FWIO/RNTupleTempTests"]))
        self.assertEqual(_packages_named_loosely("why are ROCm tests failing on el10?"),
                         ("ROCm", ["HeterogeneousTest/ROCmDevice", "HeterogeneousTest/ROCmKernel"]))

    def test_left_alone(self):
        for question in ("why is Geometry failing?", "any PR about LSTCore failing?", "is SimG4Core/Geometry failing?",
                         "what does LSTCore do?", "why is ROOT6 failing?", "is CMSSW failing?", "why is 537.0 failing in LSTCore?"):
            self.assertIsNone(_packages_named_loosely(question), question)


class PackageFollowUpTests(unittest.TestCase):
    def setUp(self):
        _turn_memory["f"] = {"question": "why did SimG4Core/Geometry fail there?", "route": "deterministic:package_status",
                             "answer": "`SimG4Core/Geometry` unit tests in `CMSSW_17_0_X_2026-11-04-1100`"}
        _latest_ib_contexts["f"] = "CMSSW_17_0_X_2026-11-04-1100"
        self.addCleanup(_turn_memory.pop, "f", None)
        self.addCleanup(_latest_ib_contexts.pop, "f", None)

    def test_rewrites(self):
        self.assertEqual(_resolve_short_follow_up("does it fail anywhere else?", "f"), "is SimG4Core/Geometry failing anywhere?")
        self.assertEqual(_resolve_short_follow_up("since when?", "f"), "is SimG4Core/Geometry failing? since when")
        self.assertEqual(_resolve_short_follow_up("on aarch64 too?", "f"), "is SimG4Core/Geometry failing on aarch64?")

    @patch("server.api_server._previous_primary_tag", return_value="CMSSW_17_0_X_2026-11-03-2300")
    def test_the_build_before(self, _):
        self.assertEqual(_resolve_short_follow_up("and in the build before?", "f"), "is SimG4Core/Geometry failing in CMSSW_17_0_X_2026-11-03-2300")


class FailingUnitTestsDigestTests(unittest.TestCase):
    def test_grouped_by_message_most_tests_first_and_counts_the_rest(self):
        runs = [{"arch": "el8_amd64_gcc13", "failed": ["A/B/t1", "A/B/t2", "C/D/t3", "E/F/t4"]}]
        rows = [
            {"kind": "message_logger", "category": "XrdAdaptor", "summary": "%MSG fatal XrdAdaptor: Opened a file at URL root://a//x.root", "tests": ["A/B/t1"]},
            {"kind": "message_logger", "category": "XrdAdaptor", "summary": "%MSG fatal XrdAdaptor: Opened a file at URL root://b//y.root", "tests": ["A/B/t2", "A/B/passed"]},
            {"kind": "crash", "category": "Bus error", "summary": "Process bus error", "tests": ["C/D/t3"]},
        ]
        driver, _session = _driver_with(runs, rows)
        with patch("server.api_server.get_driver", return_value=driver):
            text = api_server._failing_unit_tests_log_digest("CMSSW_17_0_X_2026-11-04-1100")
        self.assertIn("the 4 failing test(s)", text)
        self.assertLess(text.index("**2 test(s)**: %MSG fatal XrdAdaptor"), text.index("**1 test(s)**: Process bus error"))
        self.assertIn("**1 test(s)**: the log scanner found no error line", text)
        self.assertIn("not why", text)

    def test_plain_error_lines_only_for_tests_with_nothing_more_specific(self):
        runs = [{"arch": "a", "failed": ["A/B/t1", "A/B/t2"]}]
        rows = [{"kind": "crash", "category": "x", "summary": "Process bus error", "tests": ["A/B/t1"]},
                {"kind": "generic_error_line", "category": None, "summary": "make: *** Error 4", "tests": ["A/B/t1"]},
                {"kind": "generic_error_line", "category": None, "summary": "ERROR failed to parse X509 proxy", "tests": ["A/B/t2"]}]
        driver, _session = _driver_with(runs, rows)
        with patch("server.api_server.get_driver", return_value=driver):
            text = api_server._failing_unit_tests_log_digest("T")
        self.assertIn("X509", text)
        self.assertNotIn("make:", text)

    def test_no_failing_tests_or_no_logs(self):
        driver, _session = _driver_with([], [])
        with patch("server.api_server.get_driver", return_value=driver):
            self.assertIsNone(api_server._failing_unit_tests_log_digest("CMSSW_17_0_X_2026-11-04-1100"))


class LogTextInSuiteTests(unittest.TestCase):
    ROWS = [{"tag": "CMSSW_17_0_X_2026-11-04-1100", "arch": "el8_amd64_gcc13", "summary": "Process bus error", "text": "Bus error",
             "owners": ["A/B/t", "C/D/t"], "statuses": ["failed", "passed"]}]

    def _ask(self, question, rows=None):
        driver, session = _driver_with(self.ROWS if rows is None else rows)
        with patch("server.api_server.get_driver", return_value=driver):
            return api_server._log_text_in_suite_answer(question), session

    def test_finds_the_text_in_that_series_and_suite(self):
        answer, session = self._ask("what is Bus error in 17_0 unit logs?")
        self.assertIn("`Bus error` in `CMSSW_17_0_*` unit test logs: 1 build(s)", answer)
        self.assertIn("2 test(s), 1 reported failed", answer)
        self.assertEqual(session.run.call_args.kwargs["series"], "CMSSW_17_0_")

    def test_says_when_it_is_not_there(self):
        answer, _ = self._ask("Transport endpoint is not connected in 17_0 unit tests?", rows=[])
        self.assertIn("is not in any", answer)

    def test_ordinary_questions_are_left_alone(self):
        for question in ("failures in 17_0 unit tests", "Why are there failures in 17_0 unit tests?", "new problems in unit logs",
                         "Errors in unit tests?", "unit test failures in 17_0"):
            self.assertIsNone(api_server._log_text_in_suite_answer(question), question)


class SeriesUnitFailuresTests(unittest.TestCase):
    def test_ranks_by_failing_packages_in_each_newest_build(self):
        rows = [
            {"release": "CMSSW_17_0_X", "arch": "el8_amd64_gcc13", "tag": "CMSSW_17_0_X_2026-11-04-1100", "total": 394, "failed": 57},
            {"release": "CMSSW_17_0_X", "arch": "el8_amd64_gcc13", "tag": "CMSSW_17_0_X_2026-11-03-2300", "total": 394, "failed": 0},
            {"release": "CMSSW_20_1_X", "arch": "el9_amd64_gcc14", "tag": "CMSSW_20_1_X_2026-11-04-1100", "total": 381, "failed": 1},
        ]
        driver, _session = _driver_with(rows)
        with patch("server.api_server.get_driver", return_value=driver):
            answer = api_server._series_stability_answer("which release series has the most failing unit tests right now?")
        self.assertIn("**`CMSSW_17_0_X`** has the most: 57", answer)
        self.assertLess(answer.index("| `CMSSW_17_0_X` | 57 | 394"), answer.index("| `CMSSW_20_1_X` | 1 | 381"))

    def test_most_failing_relvals_is_said_first(self):
        rows = [{"release": "CMSSW_20_1_X", "builds": 12, "total": 35000, "failed": 1909}, {"release": "CMSSW_5_3_X", "builds": 1, "total": 201, "failed": 56}]
        driver, _session = _driver_with(rows)
        with patch("server.api_server.get_driver", return_value=driver):
            answer = api_server._series_stability_answer("which release series has the most failing relvals")
        self.assertTrue(answer.startswith("**`CMSSW_20_1_X`** has the most failing RelVals: 1,909"), answer)
        self.assertIn("the highest rate is `CMSSW_5_3_X`", answer)

    def test_the_follow_up_about_relvals(self):
        _turn_memory["r"] = {"question": "which release series has the most failing unit tests right now?", "route": "deterministic:series_stability", "answer": "x"}
        self.addCleanup(_turn_memory.pop, "r", None)
        self.assertEqual(_resolve_short_follow_up("and most failing relvals?", "r"), "which release series has the most failing relvals")


class WhichOnesAfterAUnitAnswerTests(unittest.TestCase):
    def test_lists_them(self):
        _turn_memory["w"] = {"question": "why are so many unit tests failing in 17_0?", "route": "capability:unittest", "answer": "57 failed"}
        _latest_ib_contexts["w"] = "CMSSW_17_0_X_2026-11-04-1100"
        self.addCleanup(_turn_memory.pop, "w", None)
        self.addCleanup(_latest_ib_contexts.pop, "w", None)
        self.assertEqual(_resolve_short_follow_up("which ones?", "w"), "list failed unit tests in CMSSW_17_0_X_2026-11-04-1100")


class WorkflowInSeriesTests(unittest.TestCase):
    def test_the_newest_failing_primary_of_that_series(self):
        driver, session = _driver_with([{"t": "CMSSW_16_1_X_2026-11-03-1100"}, {"t": "CMSSW_16_1_ROOT6_X_2026-11-04-1100"}])
        with patch("server.api_server.get_driver", return_value=driver):
            tag = api_server._workflow_newest_failing_primary_tag("10805.31", (16, 1))
        self.assertEqual(tag, "CMSSW_16_1_X_2026-11-04-1100")
        self.assertEqual(session.run.call_args.kwargs["prefix"], "CMSSW_16_1_")

    def test_no_failure_there_is_none(self):
        driver, _session = _driver_with([])
        with patch("server.api_server.get_driver", return_value=driver):
            self.assertIsNone(api_server._workflow_newest_failing_primary_tag("10805.31", (16, 1)))


class ShifterSummaryWordingTests(unittest.TestCase):
    def test_shifter_summary_means_shift_summary(self):
        self.assertEqual(api_server._shift_question_rewrite("shifter summary for 11-04"), "shift summary for 11-04")
        self.assertEqual(api_server._shift_question_rewrite("shifter digest for yesterday"), "shift digest for yesterday")

    def test_what_a_series_looks_like(self):
        self.assertEqual(api_server._shift_question_rewrite("what does the 16_1 IB look like?"), "what is failing in 16_1")


class ArchBuildOutTests(unittest.TestCase):
    def _ask(self, question, archs):
        driver, _session = _driver_with([{"a": a} for a in archs])
        with patch("server.api_server.get_driver", return_value=driver), \
                patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-04-1100"}]):
            return api_server._arch_build_out_answer(question)

    def test_not_yet_and_yes(self):
        self.assertIn("Not yet: `CMSSW_20_1_X_2026-11-04-1100` has no imported `el10`", self._ask("is el10 out for the 11-04 11:00 build?", ["el8_amd64_gcc14"]))
        self.assertIn("Not yet", self._ask("is aarch64 of the 11-04-1100 build out yet?", ["el8_amd64_gcc14"]))
        self.assertIn("Yes: `CMSSW_20_1_X_2026-11-03-2300`", self._ask("is aarch64 ready for 11-03 23:00?", ["el9_aarch64_gcc14"]))

    def test_left_alone(self):
        self.assertIsNone(api_server._arch_build_out_answer("is el10 failing on 11-04?"))
        self.assertIsNone(api_server._arch_build_out_answer("is aarch64 failing in CMSSW_20_1_X_2026-11-04-1100?"))

    def test_a_named_ib(self):
        self.assertIn("Not yet: `CMSSW_20_1_X_2026-11-04-1100`", self._ask("is aarch64 of the CMSSW_20_1_X_2026-11-04-1100 build out yet?", ["el8_amd64_gcc14"]))


class UnitSummaryArchScopeTests(unittest.TestCase):
    @patch("server.api_server.get_failing_unittests", return_value=[{"name": "A/B", "errors": 1}])
    @patch("server.api_server._sub_ib_variant_tags", return_value=[{"tag": "CMSSW_20_1_ROOT6_X_2026-11-03-2300", "variant": "ROOT6"}])
    def test_a_sub_ib_section_only_for_the_asked_architecture(self, _variants, _failing):
        primary = [dict(tag="CMSSW_20_1_X_2026-11-03-2300", arch="el9_amd64_gcc16", failed=1, passed=379, total=380, errors=1)]
        root6 = [dict(tag="CMSSW_20_1_ROOT6_X_2026-11-03-2300", arch="el9_aarch64_gcc14", failed=2, passed=371, total=373, errors=3)]
        with patch("server.api_server._unittest_scope_records", side_effect=lambda tag: root6 if "ROOT6" in tag else primary):
            scoped = api_server._unittest_summary_from_records("CMSSW_20_1_X_2026-11-03-2300", primary, arch_scoped=True)
            whole = api_server._unittest_summary_from_records("CMSSW_20_1_X_2026-11-03-2300", primary)
        self.assertNotIn("ROOT6", scoped)
        self.assertIn("ROOT6", whole)


class ThereFollowUpTests(unittest.TestCase):
    def tearDown(self):
        _turn_memory.pop("t", None)

    def test_stays_on_the_ib_and_architecture_of_the_answer(self):
        _turn_memory["t"] = {"question": "RelVal failures on el10 in the newest IB?", "route": "deterministic:arch_word_results",
                             "answer": "`CMSSW_20_1_X_2026-11-04-1100` has no build on `el10` yet; this is `CMSSW_20_1_X_2026-11-03-2300`, the newest IB built on it.\n\n"
                                       "`el10` in it: 1 build(s) on `el10_amd64_gcc15`."}
        self.assertEqual(_resolve_short_follow_up("unit tests there?", "t"), "failed unit tests in CMSSW_20_1_X_2026-11-03-2300 on el10_amd64_gcc15")
        self.assertEqual(_resolve_short_follow_up("how many failed there?", "t"), "failed relvals in CMSSW_20_1_X_2026-11-03-2300 on el10_amd64_gcc15")

    def test_without_an_architecture_in_the_question_nothing_changes(self):
        _turn_memory["t"] = {"question": "RelVal failures in the newest IB?", "route": "capability:relval",
                             "answer": "`CMSSW_20_1_X_2026-11-04-1100` on `el8_amd64_gcc14`"}
        self.assertIsNone(_resolve_short_follow_up("unit tests there?", "t"))


class ArchMissingFromNewestTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(_default_tag_notes.pop, "a", None)
        for name, value in (("_latest_ib_records", [{"tag": "CMSSW_20_1_X_2026-11-04-1100"}]),
                            ("_packages_named_loosely", None),
                            ("_newest_tag_for_arch_word", ("CMSSW_20_1_X_2026-11-03-2300", ["el9_amd64_gcc16"])),
                            ("_arch_scope_matches", ("CMSSW_20_1_X_2026-11-04-1100", "gcc16", [], ["el8_amd64_gcc14", "el9_amd64_gcc14"]))):
            patcher = patch(f"server.api_server.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_uses_the_newest_ib_built_on_it_and_says_so(self):
        question = _with_arch_missing_from_newest("what's broken on gcc16?", "a")
        self.assertEqual(question, "what's broken on gcc16? CMSSW_20_1_X_2026-11-03-2300 el9_amd64_gcc16")
        self.assertIn("has no `gcc16` build yet", _default_tag_notes["a"])

    def test_left_alone(self):
        for question in ("is gcc16 out yet?", "compare gcc15 and gcc16 failures", "failures on gcc16 this week",
                         "why is 537.0 failing on gcc16?", "what failed on gcc16 in 17_1?",
                         "what's broken in CMSSW_20_1_X_2026-11-03-2300 on gcc16?"):
            self.assertEqual(_with_arch_missing_from_newest(question, "a"), question, question)
        self.assertNotIn("a", _default_tag_notes)

    def test_the_shift_rewrite_keeps_the_architecture(self):
        self.assertEqual(api_server._shift_question_rewrite("what's broken on gcc16?"), "what is failing in the latest IB on gcc16")
        self.assertEqual(api_server._shift_question_rewrite("what's broken?"), "what is failing in the latest IB")

    def test_a_build_of_another_series_in_the_session_stays(self):
        with patch("server.api_server._arch_scope_matches", return_value=("CMSSW_17_0_X_2026-11-04-1100", "aarch64", [], ["el8_amd64_gcc13"])):
            self.assertEqual(_with_arch_missing_from_newest("what's failing on aarch64?", "a"), "what's failing on aarch64?")


if __name__ == "__main__":
    unittest.main()
