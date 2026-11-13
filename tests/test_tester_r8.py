import unittest
from unittest import mock
from server import api_server as a

ROWS = [{"repo": "cmssw", "pr_number": 52397, "title": "avoid re-classification", "merged_at": "2026-11-06T08:20:15Z", "author": "x"}]
TAG = "CMSSW_20_1_X_2026-11-06-1100"

class CulpritCandidates(unittest.TestCase):
    def ask(self, q, rows=ROWS):
        with mock.patch.object(a, "find_prs_before_ib", return_value=rows):
            return a._ib_culprit_candidates_answer(q)

    def test_lists_prs_merged_into_the_build(self):
        ans, rows = self.ask(f"which PR could have broken {TAG}?")
        self.assertIn("52397", ans); self.assertIn("relevance only", ans); self.assertEqual(rows, ROWS)

    def test_other_wording(self):
        self.assertIsNotNone(self.ask(f"who introduced the failures in {TAG}")); self.assertIsNotNone(self.ask(f"what PR is to blame for {TAG}?"))

    def test_no_break_word_or_two_tags_or_pr_number(self):
        self.assertIsNone(self.ask(f"which PRs are in {TAG}?"))
        self.assertIsNone(self.ask(f"which PR broke {TAG} and CMSSW_20_1_X_2026-11-05-1100?"))
        self.assertIsNone(self.ask(f"did PR 52397 break {TAG}?"))

    def test_no_rows_says_no_list_imported(self):
        ans, rows = self.ask(f"which PR broke {TAG}?", rows=[])
        self.assertIn("No list of the PRs merged into", ans); self.assertEqual(rows, [])

    def test_no_placeholder(self):
        ans, _ = self.ask(f"which PR broke {TAG}?")
        self.assertNotIn("<number>", ans); self.assertNotIn("&lt;", ans)

if __name__ == "__main__":
    unittest.main()
