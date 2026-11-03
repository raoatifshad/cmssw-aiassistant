"""Wrong/dead-end answers found by the unattended tester run of 2026-11-04 (fresh questions about
that day's data: read in full and checked against the graph)."""
import unittest
from unittest.mock import MagicMock, patch

from server import api_server
from server.api_server import _LOG_PROBLEM_PASTE_RE, _ib_failure_ranking_answer, _is_latest_ib_question, _loose_arch_choice, _render_metric_deltas, _today_falls_back_to_latest


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


if __name__ == "__main__":
    unittest.main()
