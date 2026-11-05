"""Wrong/dead-end answers found by the unattended tester run of 2026-11-05 (fresh questions about
that day's data: read in full and checked against the graph)."""
import unittest
from unittest.mock import MagicMock, patch

from server import api_server
from server.api_server import (
    _EARLY_LOG_LINE_RE, _LOG_PROBLEM_PASTE_RE, _latest_ib_contexts, _package_status_answer, _packages_named_loosely,
    _paste_follow_up_answer, _resolve_short_follow_up, _turn_memory,
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


if __name__ == "__main__":
    unittest.main()
