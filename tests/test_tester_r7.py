import unittest
from server import api_server as a

class KnownQuestionInvents(unittest.TestCase):
    def test_pr_number_inside_a_longer_number_is_invented(self):
        history = [{"question": "which PR broke it?", "answer": "two PRs: **#52375** Phase2 and **#52388** Run3"}]
        self.assertTrue(a._known_question_invents("what does PR 5237 do", "what did it change?", history))

    def test_whole_pr_number_is_said(self):
        history = [{"question": "q", "answer": "see **#52375** and #52388"}]
        self.assertFalse(a._known_question_invents("what does PR 52375 do", "what did it change?", history))

    def test_workflow_prefix_is_invented(self):
        history = [{"question": "q", "answer": "workflow 136.901 fails"}]
        self.assertTrue(a._known_question_invents("why does 136.9 fail", "why?", history))
        self.assertFalse(a._known_question_invents("why does 136.901 fail", "why?", history))

if __name__ == "__main__":
    unittest.main()
