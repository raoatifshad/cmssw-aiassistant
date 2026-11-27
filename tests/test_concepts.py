import unittest
from unittest.mock import patch

from graph.tools import search_concepts
from server.api_server import _concept_answer


class ConceptGlossaryTests(unittest.TestCase):
    @patch("graph.tools.get_driver")
    def test_search_concepts_matches_on_significant_words_only(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"question": "what does WrongTrackerSubDet mean", "definition": "A geometry mismatch.", "match_count": 1},
        ]

        results = search_concepts("what does WrongTrackerSubDet mean?")

        self.assertEqual(len(results), 1)
        terms = session.run.call_args.args[1]["terms"]
        self.assertIn("WrongTrackerSubDet", terms)
        self.assertNotIn("what", [t.lower() for t in terms])

    def test_search_concepts_returns_empty_for_only_stopwords(self):
        self.assertEqual(search_concepts("what is this"), [])

    @patch("server.api_server.search_concepts", return_value=[])
    def test_concept_answer_is_none_without_a_stored_match(self, search):
        self.assertIsNone(_concept_answer("what does WrongTrackerSubDet mean?"))

    @patch("server.api_server.search_concepts")
    def test_concept_answer_surfaces_a_stored_correction(self, search):
        search.return_value = [{
            "question": "is yourdb complete?",
            "definition": "PR #51643 has a relval failure.",
        }]

        answer = _concept_answer("is yourdb complete?")

        self.assertIn("PR #51643 has a relval failure.", answer)
        self.assertIn("expert-reviewed chat feedback", answer)


if __name__ == "__main__":
    unittest.main()
