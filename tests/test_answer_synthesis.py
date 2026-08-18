import unittest
from unittest.mock import patch

from server.api_server import _extract_numbers, _synthesize_if_safe


class NumberExtractionTests(unittest.TestCase):
    def test_strips_thousands_separators_for_comparison(self):
        self.assertEqual(_extract_numbers("441 records, 258 failing"), {"441", "258"})
        self.assertEqual(_extract_numbers("1,234 records"), {"1234"})


class AnswerSynthesisSafetyTests(unittest.TestCase):
    def test_disabled_by_default_returns_the_original_text_unchanged(self):
        # ANSWER_SYNTHESIS_ENABLED is false in this test process.
        original = "Workflow `140.051` has failed in **258 of 441** imported record(s)."
        self.assertEqual(_synthesize_if_safe("any question", original), original)

    @patch("server.api_server.synthesize_evidence_answer")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.ANSWER_SYNTHESIS_ENABLED", True)
    def test_accepts_a_rephrase_that_preserves_every_number(self, synthesize):
        original = "Workflow 140.051 has failed in 258 of 441 imported records."
        synthesize.return_value = "This workflow (140.051) failed 258 out of 441 times so far."

        result = _synthesize_if_safe("has it recurred?", original)

        self.assertEqual(result, synthesize.return_value)

    @patch("server.api_server.synthesize_evidence_answer")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.ANSWER_SYNTHESIS_ENABLED", True)
    def test_rejects_a_rephrase_that_drops_or_alters_a_number(self, synthesize):
        original = "Workflow 140.051 has failed in 258 of 441 imported records."
        # The LLM dropped 441 entirely -- a real, known LLM paraphrase failure mode.
        synthesize.return_value = "This workflow has failed 258 times recently."

        result = _synthesize_if_safe("has it recurred?", original)

        self.assertEqual(result, original)

    @patch("server.api_server.synthesize_evidence_answer", return_value=None)
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.ANSWER_SYNTHESIS_ENABLED", True)
    def test_falls_back_when_synthesis_returns_nothing(self, synthesize):
        original = "Workflow 140.051 has failed in 258 of 441 imported records."
        self.assertEqual(_synthesize_if_safe("q", original), original)

    @patch("server.api_server.synthesize_evidence_answer", side_effect=RuntimeError("boom"))
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.ANSWER_SYNTHESIS_ENABLED", True)
    def test_falls_back_and_never_raises_if_synthesis_errors(self, synthesize):
        original = "Workflow 140.051 has failed in 258 of 441 imported records."
        self.assertEqual(_synthesize_if_safe("q", original), original)


if __name__ == "__main__":
    unittest.main()
