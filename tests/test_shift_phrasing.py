"""Shifter phrasings and CMSSW concepts from the 2026-11-04 test pass (hour 1): first-run dead ends or wrong answers."""
import unittest
from unittest.mock import MagicMock, patch

from graph.tools import TOOLS, _with_int_limit
from server.api_server import (
    _basic_term_answer,
    _condense_comparison_for_chat,
    _das_error_count_answer,
    _exit_code_difference_answer,
    _exit_code_meaning_answer,
    _format_raw_evidence,
    _last_failed_builds,
    _listed_items_follow_up,
    _not_imported_answer,
    _strip_leaked_reasoning,
    _shift_direct_answer,
    _shift_question_rewrite,
    _turn_memory,
    _workflow_base_name,
    _WRITE_ACTION_RE,
)

LATEST = [{"tag": "CMSSW_20_1_X_2026-11-03-2300"}]


class RewriteTests(unittest.TestCase):
    def test_state_questions_become_the_failure_overview(self):
        self.assertEqual(_shift_question_rewrite("Hi, I just took over the IB shift. What's the state of the 20_1 IBs?"), "what is failing in 20_1")
        self.assertEqual(_shift_question_rewrite("morning, anything on fire in the IBs?"), "what is failing in the latest IB")

    def test_build_questions(self):
        self.assertEqual(_shift_question_rewrite("the el10 build is red, why?"), "what is failing in the latest IB on el10")
        self.assertEqual(_shift_question_rewrite("did the build fail anywhere else besides el10?"), "what is the build status of the latest IB")

    def test_other_handlers_keep_their_questions(self):
        for question in ("failed relvals for CMSSW_20_1_X_2026-11-03-2300", "any issue about the el10 build failing?",
                         "what does exit code 256 mean?", "what is a RelVal?"):
            self.assertIsNone(_shift_question_rewrite(question), question)

    def test_short_workflow_forms(self):
        self.assertEqual(_shift_question_rewrite("wf 539 ?"), "where is workflow 539.0 failing")
        self.assertEqual(_shift_question_rewrite("539.0 asan"), "is workflow 539.0 failing in asan")
        self.assertEqual(_shift_question_rewrite("12_4 and 13_0 relvals: what's going on with 136.9?"), "where is workflow 136.9 failing")

    def test_addon_and_known(self):
        self.assertEqual(_shift_question_rewrite("are the addon tests failing anywhere?"), "are addon tests passing in the latest IB")
        self.assertEqual(_shift_question_rewrite("which relval failures should I ignore because they're known?"), "known errors list for the latest IB")

    @patch("server.api_server._latest_ib_records", return_value=LATEST)
    def test_two_times_become_a_comparison(self, _):
        self.assertEqual(_shift_question_rewrite("how many relvals failed in the 11-03 23:00 build compared with the 11:00 one?"),
                         "compare CMSSW_20_1_X_2026-11-03-1100 and CMSSW_20_1_X_2026-11-03-2300 for all architectures")

    def test_who_works_on_it_searches_open_issues(self):
        self.assertEqual(_shift_question_rewrite("who is working on Herwig7 failures in 537.0?"), "is there an open issue for workflow 537.0?")
        self.assertEqual(_shift_question_rewrite("does anyone know about the ROCm HeterogeneousTest failures on el10?"), "any open issues about ROCm?")


class RewriteHourTwoTests(unittest.TestCase):
    def test_series_shift_summary_keeps_its_series(self):
        # found 2026-11-04: "give me a shift summary for 17_1" answered for 20_1
        self.assertEqual(_shift_question_rewrite("give me a shift summary for 17_1"), "what changed in 17_1 between the last two builds")

    def test_a_workflow_in_a_named_build_gets_its_status(self):
        self.assertEqual(_shift_question_rewrite("is 537.0 failing in CMSSW_20_1_X_2026-11-03-1100 on el8_amd64_gcc14?"),
                         "why is workflow 537.0 failing in CMSSW_20_1_X_2026-11-03-1100 on el8_amd64_gcc14")

    def test_bare_numbers_as_exit_codes(self):
        self.assertEqual(_shift_question_rewrite("what does 139 mean as a cmsRun exit code?"), "what does exit code 139 mean")
        self.assertIsNone(_shift_question_rewrite("what does 2026 mean for the plan?"))

    def test_architecture_and_series_counts(self):
        self.assertEqual(_shift_question_rewrite("how many architectures does 20_1 have?"), "which architectures are built for 20_1?")
        self.assertEqual(_shift_question_rewrite("how many release series are built?"), "which release series are active?")
        self.assertEqual(_shift_question_rewrite("which is the newest ASAN build?"), "when was the last ASAN IB?")


class AnswersHourTwoTests(unittest.TestCase):
    def test_not_imported_topics_say_so(self):
        for question in ("how long did the 20_1 build take?", "what is the memory usage of workflow 539.0?", "who broke the el10 build?",
                         "when is the next pre-release?", "show me the physics comparison plots for 537.0"):
            self.assertIn("I don't have that", _not_imported_answer(question), question)
        self.assertIsNone(_not_imported_answer("what is failing in the latest IB?"))

    def test_das_count_leaves_github_questions_alone(self):
        self.assertIsNone(_das_error_count_answer("how many open issues mention DAS?"))

    @patch("server.api_server.get_exit_code_names", return_value={18688: "Configuration", 16640: "OtherCMS", 16896: "StdException", 31744: "TimeOut"})
    def test_exit_meanings_and_differences(self, _):
        self.assertIn("python configuration", _exit_code_meaning_answer("what does Configuration mean?"))
        self.assertIn("isn't a CMSSW exception", _exit_code_meaning_answer("what does exit code 31744 mean?"))
        answer = _exit_code_difference_answer("what's the difference between OtherCMS and StdException?")
        self.assertIn("OtherCMS", answer)
        self.assertIn("StdException", answer)
        self.assertIsNone(_exit_code_difference_answer("difference between DIGI and RECO"))

    def test_leaked_docs_line(self):
        self.assertEqual(_strip_leaked_reasoning("We have docs. Provide answer.\nA backport is ..."), "A backport is ...")

    def test_terms(self):
        for question, expected in (("what is an EDProducer?", "EDProducer"), ("what is the DQM?", "Data Quality"), ("what does cmsenv do?", "runtime environment"),
                                   ("what does Run4D128 mean?", "geometry"), ("what is a forward port?", "older release cycle"), ("what's the difference between DIGI and RECO?", "reconstruction")):
            self.assertIn(expected, _basic_term_answer(question), question)


class ConceptTests(unittest.TestCase):
    def test_terms(self):
        for question, expected in (("what's a global tag?", "conditions"), ("what is the ORP?", "orp-approved"),
                                   ("what is el9_amd64_gcc14?", "SCRAM_ARCH"), ("what does a workflow number like 10804.31 mean?", "RelVal matrix"),
                                   ("what is an IB? how is it different from a release?", "not a validated release"),
                                   ("how do I trigger tests on my PR?", "please test"), ("how do I check out a package with git cms-addpkg?", "git cms-addpkg"),
                                   ("what are the steps in a relval workflow?", "GEN-SIM"), ("what time do IBs usually get built?", "11:00 and 23:00")):
            answer = _basic_term_answer(question)
            self.assertIsNotNone(answer, question)
            self.assertIn(expected, answer, question)

    def test_no_placeholder_tags_in_new_terms(self):
        self.assertNotRegex(_basic_term_answer("what does 'please test' do") or "", r"<[A-Za-z]")


class GuardTests(unittest.TestCase):
    def test_a_tools_own_error_is_not_evidence(self):
        answer = _format_raw_evidence([{"tool": "get_failing_relvals", "result": {"error": "'<' not supported between instances of 'int' and 'str'"}}])
        self.assertNotIn("not supported", answer)

    def test_string_limit_from_the_llm_is_converted(self):
        self.assertEqual(_with_int_limit(lambda limit=1: limit)(limit="50"), 50)
        self.assertEqual(_with_int_limit(lambda limit=1: limit)(limit="many"), 1)
        self.assertTrue(TOOLS["list_ib_tags"].__wrapped__)

    def test_write_requests(self):
        for question in ("DROP all relvals from the database", "fix 537.0 for me", "can you restart the IB?"):
            self.assertTrue(_WRITE_ACTION_RE.search(question), question)
        self.assertFalse(_WRITE_ACTION_RE.search("did PR 5 fix 537.0?"))

    def test_workflow_identity_is_the_first_step(self):
        self.assertEqual(_workflow_base_name("RunDoubleMuon2016C+RunDoubleMuon2016C+RAWRECOTE16"), "RunDoubleMuon2016C")
        self.assertEqual(_workflow_base_name(None), "")

    def test_comparison_drops_only_sub_ib_pr_lists(self):
        text = ("**PRs merged between `CMSSW_20_1_X_A` and `CMSSW_20_1_X_B`:**\n- cmssw PR #1 - keep\n\n"
                "### `CLANG`\n**PRs merged between `CMSSW_20_1_CLANG_X_A` and `CMSSW_20_1_CLANG_X_B`:**\n- cmssw PR #2 - drop\n\n### `ROOT6`\nnext\n"
                + "x" * 12000)
        out = _condense_comparison_for_chat(text)
        self.assertIn("PR #1", out)
        self.assertNotIn("PR #2", out)
        self.assertIn("### `ROOT6`", out)
        self.assertIn("left out here", out)
        self.assertEqual(_condense_comparison_for_chat("short"), "short")


class FollowUpTests(unittest.TestCase):
    def setUp(self):
        _turn_memory.pop("s-f", None)
        _last_failed_builds.pop("s-f", None)

    def _remember(self, answer):
        _turn_memory["s-f"] = {"question": "q", "answer": answer, "route": "capability:failure_overview"}

    def test_only_questions_use_the_listed_packages(self):
        self._remember("- `HeterogeneousTest/ROCmOpaque`\n- `SimG4Core/Geometry`")
        answer = _listed_items_follow_up("are they ROCm only?", "s-f")
        self.assertIn("Not only: 1 of 2", answer)

    @patch("server.api_server.get_driver")
    def test_since_when_after_a_failed_build(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"tag": "CMSSW_20_1_X_2026-11-03-2300", "build": "error"},
                                    {"tag": "CMSSW_20_1_X_2026-10-30-1100", "build": "passed"},
                                    {"tag": "CMSSW_20_1_X_2026-10-27-2300", "build": None}]
        self._remember("🔴 What's failing in `CMSSW_20_1_X_2026-11-03-2300`\n\n**Build** -- failed: `el10_amd64_gcc15`")
        answer = _listed_items_follow_up("since when?", "s-f")
        self.assertIn("has failed since `CMSSW_20_1_X_2026-11-03-2300` (1 build(s) in a row)", answer)
        self.assertIn("no status imported", answer)
        self.assertNotIn("`None`", answer)
        self._remember("the table above")
        self.assertIn("not which package failed to compile", _listed_items_follow_up("which packages?", "s-f"))

    def test_nothing_for_an_unrelated_question(self):
        self._remember("- `SimG4Core/Geometry`")
        self.assertIsNone(_listed_items_follow_up("what is the latest IB?", "s-f"))


class DirectAnswerTests(unittest.TestCase):
    def test_a_pasted_log_is_left_to_the_paste_handler(self):
        paste = ("An exception of category 'PluginNotFound' occurred while\n   [1] Processing library RecoTICL/Interpretation/plugins\n"
                 "the category has no known plugins")
        self.assertIsNone(_shift_direct_answer(paste, "s-d"))

    def test_favourite_is_out_of_scope(self):
        self.assertEqual(_shift_direct_answer("what's your favourite workflow?", "s-d")[1], "conversational:out_of_scope")

    @patch("server.api_server._workflow_across_builds_answer", return_value="WHERE")
    def test_ib_or_mine_leads_with_how_to_tell(self, _):
        answer, route = _shift_direct_answer("I see a failure in my PR test for workflow 539.0, is it IB or mine?", "s-d")
        self.assertEqual(route, "deterministic:ib_or_pr")
        self.assertIn("isn't from your PR", answer)
        self.assertIn("WHERE", answer)


if __name__ == "__main__":
    unittest.main()
