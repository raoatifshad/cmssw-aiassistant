"""Wrong/dead-end answers found by the unattended tester run of 2026-11-04 (fresh questions about
that day's data: read in full and checked against the graph)."""
import unittest
from unittest.mock import MagicMock, patch

from server import api_server
from server.api_server import _loose_arch_choice, _render_metric_deltas, _today_falls_back_to_latest


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


if __name__ == "__main__":
    unittest.main()
