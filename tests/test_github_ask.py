"""PR/Issue asks about a named unit test or quoted text (server/github_ask.py). Found live 2026-11-11 (owner's session):
'is there any issue or PR open for "testVolumes crashes"' got the unit-test history only (GitHub never searched, open Issue #51089
has it in its title), and 'did you find any issue or PR open related to this' got "Choose a scope"."""
import unittest
from unittest.mock import patch

from server import api_server, github_ask

_OPEN = {"kind": "Issue", "number": 51089, "title": "[SimG4Core/Geometry] testVolumes crashes on el10_amd64_gcc15", "state": "open",
         "created_at": "2026-05-29T10:12:35Z", "author": "akritkbehera", "strong": True}
_CLOSED = {"kind": "Issue", "number": 25876, "title": "Migration from CLHEP Units to DD Units", "state": "closed",
           "created_at": "2019-02-06T16:09:18Z", "author": "ianna", "strong": True}
_HISTORY = ("Unit test `SimG4Core/Geometry/testVolumes`, from the raw IB test logs: **not passed in 90 of 995** imported run(s)."
            "\n\n| IB | Architecture |\n| --- | --- |")


class GithubNamedAskTests(unittest.TestCase):
    def setUp(self):
        self.sid = "github-ask-test"
        api_server._turn_memory.pop(self.sid, None)
        github_ask._subjects.pop(self.sid, None)

    def _run(self, question, items=(_OPEN, _CLOSED), history=_HISTORY, follow_up_only=False):
        report = {"items": list(items), "searched": ["x"]} if items else None
        with patch.object(api_server, "_github_mentions_report", return_value=report) as search, \
             patch.object(api_server, "_github_reports_section", return_value="**Found on GitHub** section" if items else None), \
             patch.object(api_server, "_named_test_answer", return_value=history), \
             patch.object(api_server, "_github_export_note", return_value="GitHub data reaches 2026-11-06."):
            return github_ask.github_named_ask_answer(question, self.sid, follow_up_only=follow_up_only), search

    def test_open_ask_names_the_open_item_first_then_the_test_history(self):
        answer, search = self._run('is there any issue or PR open for "testVolumes crashes"')
        self.assertTrue(answer.startswith("**Open: 1**"))
        self.assertIn("#51089", answer.split("\n\n")[0])
        self.assertNotIn("#25876", answer.split("\n\n")[0])
        self.assertIn("**Found on GitHub** section", answer)
        self.assertIn("not passed in 90 of 995", answer)
        self.assertNotIn("| IB |", answer)  # the short history only, not the table
        # the quoted phrase first, then the test name on its own
        self.assertEqual(search.call_args[0][0]["names"], ["testVolumes crashes", "testVolumes"])
        self.assertEqual(search.call_count, 1)  # searched once, the section reuses it

    def test_only_closed_matches_are_said_to_be_closed(self):
        answer, _ = self._run('is there an open issue for "testVolumes crashes"', items=(_CLOSED,))
        self.assertTrue(answer.startswith("**No open PR or Issue**"))

    def test_no_match_says_so_with_the_data_reach(self):
        answer, _ = self._run('any PR about "zzqqx nothingmatches"?', items=(), history=None)
        self.assertIn('No PR or Issue mentions "zzqqx nothingmatches"', answer)
        self.assertIn("2026-11-06", answer)

    def test_without_open_there_is_no_open_lead(self):
        answer, _ = self._run('can you find an issue or pull request about "testVolumes crashes"')
        self.assertFalse(answer.startswith("**Open"))

    def test_related_to_this_after_a_unit_test_answer(self):
        api_server._turn_memory[self.sid] = {"question": "is testVolumes failing?", "answer": _HISTORY, "route": "deterministic:named_test"}
        answer, search = self._run("did you find any issue or PR open related to this", follow_up_only=True)
        self.assertIn("#51089", answer)
        self.assertEqual(search.call_args[0][0]["names"], ["testVolumes"])

    def test_related_to_this_after_this_route_keeps_the_subject(self):
        self._run('is there any issue or PR open for "testVolumes crashes"')
        api_server._turn_memory[self.sid] = {"question": "q", "answer": "a", "route": "deterministic:github_named_ask"}
        _, search = self._run("did you find any issue or PR open related to this", follow_up_only=True)
        self.assertEqual(search.call_args[0][0]["names"], ["testVolumes crashes", "testVolumes"])

    def test_related_to_this_after_a_list_of_failing_packages_searches_each(self):
        api_server._turn_memory[self.sid] = {"question": "which unit tests failed?", "route": "capability:unittest",
                                             "answer": "**el9_aarch64_gcc14** (3 failed):\n- SimG4Core/Geometry (errors: 1)\n- PerfTools/VTune (errors: 1)\n"
                                                       "- DQMServices/Demo (errors: 1)\n### ROOT6\n- SimG4Core/Geometry (errors: 1)"}
        merged = {"kind": "PR", "number": 51864, "title": "Add Perftools/VTune", "state": "closed", "merged_at": "2026-11-01", "created_at": "2026-10-01", "strong": True}
        reports = {"SimG4Core/Geometry": {"items": [_OPEN, _CLOSED]}, "PerfTools/VTune": {"items": [merged]}, "DQMServices/Demo": None}
        with patch.object(api_server, "_github_mentions_report", side_effect=lambda terms: reports[terms["names"][0]]), \
             patch.object(api_server, "_github_reports_section") as section, \
             patch.object(api_server, "_github_export_note", return_value="GitHub data reaches 2026-11-06."):
            answer = github_ask.github_named_ask_answer("any issue or PR related to this?", self.sid, follow_up_only=True)
        section.assert_not_called()  # no AI check per package
        self.assertIn("3 package(s)", answer)
        self.assertIn("`SimG4Core/Geometry`: **open:** Issue #51089", answer)
        self.assertIn("`PerfTools/VTune`: no open item; 1 closed or merged mention it, newest PR #51864 - Add Perftools/VTune (merged)", answer)
        self.assertIn("`DQMServices/Demo`: nothing found", answer)
        self.assertIn("2026-11-06", answer)

    def test_this_after_an_unrelated_or_superseded_answer_is_left_alone(self):
        api_server._turn_memory[self.sid] = {"question": "q", "answer": "Unit test `A/B/c`", "route": "deterministic:relval_summary"}
        self.assertIsNone(self._run("any issue about this?", follow_up_only=True)[0])
        api_server._turn_memory[self.sid] = {"question": "q", "answer": _HISTORY, "route": "deterministic:named_test", "superseded": True}
        self.assertIsNone(self._run("any issue about this?", follow_up_only=True)[0])

    def test_a_named_subject_waits_for_the_later_hook(self):
        # before _reported_answer only follow-ups are taken: a paste or "was X reported?" keeps its richer route
        self.assertIsNone(self._run('is there any issue or PR open for "testVolumes crashes"', follow_up_only=True)[0])

    def test_other_shapes_are_left_to_their_rules(self):
        for question in ['which PRs merged "testVolumes" fixes?', "is there an issue for workflow 1.0 failing?",
                         'is "testVolumes" failing?', "is there any issue open?"]:
            self.assertIsNone(self._run(question)[0], question)

    def test_hooks_run_before_the_unit_test_rule(self):
        with open(api_server.__file__) as handle:
            source = handle.read()
        chat = source[source.index("def _chat(req: ChatRequest):"):]
        self.assertLess(chat.index("github_named_ask_answer(original_question, sid, follow_up_only=True)"), chat.index("reported = _reported_answer("))
        self.assertLess(chat.index("github_named_ask_answer(original_question, sid)"), chat.index("named_test = _named_test_answer("))


if __name__ == "__main__":
    unittest.main()
