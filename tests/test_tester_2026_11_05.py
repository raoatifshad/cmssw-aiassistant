"""Wrong/dead-end answers found by the unattended tester run of 2026-11-05 (fresh questions about
that day's data: read in full and checked against the graph)."""
import unittest
from unittest.mock import MagicMock, patch

from server import api_server
from server.api_server import (
    _EARLY_LOG_LINE_RE, _LOG_PROBLEM_PASTE_RE,
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


if __name__ == "__main__":
    unittest.main()
