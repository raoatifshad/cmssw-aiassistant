"""A pasted error is log text, not keywords, and answers about it say only what the data shows
(found 2026-11-04 on a pasted TritonService exception). Mocked: no database or LLM needed."""
import unittest
from unittest.mock import MagicMock, patch

from graph.tools import _split_by_result_code, result_codes
from server.api_server import (ChatRequest, _chat, _first_last_builds_text, _is_pasted_log, _pasted_error_in_ib_lead,
                               _relval_by_exit_code_answer, _relval_exception_context, _relval_exception_list_answer,
                               _relval_exception_message_answer, _test_log_problem_answer, _pr_not_in_ib_text,
                               _pr_number_answer, _workflow_artifact_answer, _one_line_error_with_ib,
                               _loose_arch_failure_scope, _rewrite_lost_arch_word, _is_plain_arch_overview, _default_tag_notes, _latest_ib_contexts)
from server.request_planner import RequestContext

TAG = "CMSSW_20_1_ROOT6_X_2026-11-03-2300"
ARCH = "el9_amd64_gcc14"
BLOCK = """----- Begin Fatal Exception 04-Nov-2026 05:07:07 CET-----------------------
An exception of category 'ExternalFailure' occurred while
   [0] Processing begin Job
   [1] Handling pre signal, likely in a service function
Exception Message:
TritonService: Starting the fallback server failed with exit code 65280
----- End Fatal Exception -------------------------------------------------"""
PASTE = f"is this exception is in {TAG}` on `{ARCH}`\n{BLOCK}"


def fake_driver(answer):
    """A driver whose session.run(query, **params) returns answer(query, params)."""
    session = MagicMock()

    def run(query, **params):
        result = MagicMock()
        rows = answer(query, params)
        result.__iter__.side_effect = lambda: iter(rows)
        result.single.return_value = rows[0] if rows else None
        return result

    session.run.side_effect = run
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    return driver


def graph(exceptions=(), problems=(), builds=None, log_runs=1):
    builds = [{"tag": TAG, "arch": ARCH, "relvals": 954}] if builds is None else builds

    def answer(query, params):
        if "MATCH (ib:IB)" in query:
            return builds
        if "MATCH (e:RelValException)" in query:
            return list(exceptions)
        if "MATCH (p:LogProblem)" in query:
            return list(problems)
        if "MATCH (r:TestLogRun)" in query:
            return [{"n": log_runs}]
        return []

    return fake_driver(answer)


EXC = [dict(tag=TAG, arch=ARCH, workflow=w, step="step3") for w in ("10804.31", "10805.31")]


class RoutingTests(unittest.TestCase):
    def test_block_and_multiline_log_are_pastes_but_a_question_is_not(self):
        self.assertTrue(_is_pasted_log(PASTE))
        self.assertTrue(_is_pasted_log("look at this\nTraceback (most recent call last):\n  x"))
        self.assertFalse(_is_pasted_log(f"which relvals failed with exit code 256 in {TAG}"))
        self.assertFalse(_is_pasted_log("what does exit code 65280 mean"))

    @patch("server.api_server._chat_response", side_effect=lambda sid, q, answer, route, **kw: (answer, route))
    @patch("server.api_server._test_log_problem_answer", return_value=None)
    @patch("server.api_server._relval_exception_message_answer", return_value="history")
    @patch("server.api_server._pasted_error_in_ib_lead", return_value="**Yes**")
    @patch("server.api_server._relval_by_exit_code_answer", return_value="No RelVal failed with exit code 65280")
    @patch("server.api_server._exit_code_meaning_answer", return_value="exit code meaning")
    def test_paste_never_reaches_the_keyword_handlers_and_leads_with_the_direct_answer(self, meaning, by_exit, *_):
        answer, route = _chat(ChatRequest(question=PASTE, session_id="s"))
        by_exit.assert_not_called()
        meaning.assert_not_called()
        self.assertEqual(route, "deterministic:relval_exception_message")
        self.assertEqual(answer, "**Yes**\n\nhistory")

    @patch("server.api_server.get_exit_code_names", return_value={65280: "SIGHUP"})
    def test_one_line_error_text_is_not_a_relval_exit_code_question(self, _names):
        def answer(query, params):
            return [{"n": 4}] if "RETURN count(r) AS n" in query else [dict(tag=TAG, arch=ARCH, n=2, examples=["10804.31"])]

        with patch("server.api_server.get_driver", return_value=fake_driver(answer)):
            self.assertIsNone(_relval_by_exit_code_answer(
                f"TritonService: Starting the fallback server failed with exit code 65280 in {TAG}"))
            # the question itself still works
            self.assertIn("exit code 65280", _relval_by_exit_code_answer(f"which relvals failed with exit code 65280 in {TAG}"))


class DirectAnswerTests(unittest.TestCase):
    def lead(self, question=PASTE, **kw):
        with patch("server.api_server.get_driver", return_value=graph(**kw)):
            return _pasted_error_in_ib_lead(question)

    def test_yes_names_the_workflows(self):
        lead = self.lead(exceptions=EXC)
        self.assertIn(f"**Yes, this exception is in `{TAG}` on `{ARCH}`:**", lead)
        self.assertIn("`10804.31`, `10805.31` (step3)", lead)

    def test_found_only_in_a_test_log_counts_and_shows_its_status(self):
        lead = self.lead(problems=[dict(tag=TAG, arch=ARCH, owners=["FWCore/x"], statuses=["passed"],
                                        text="TritonService: Starting the fallback server failed with exit code 65280")])
        self.assertIn("**Yes", lead)
        self.assertIn("`FWCore/x` (test log, reported passed)", lead)

    def test_another_exit_code_in_the_same_build_is_not_a_yes(self):
        lead = self.lead(problems=[dict(tag=TAG, arch=ARCH, owners=["w"], statuses=["failed"],
                                        text="Starting the fallback server failed with exit code 256")])
        self.assertTrue(lead.startswith("**No, this exception is not recorded"), lead)

    def test_no_says_what_was_checked(self):
        self.assertIn("RelVal exceptions and test logs", self.lead())
        self.assertIn("test logs are not imported", self.lead(log_runs=0))

    def test_nothing_imported_is_never_a_no(self):
        self.assertIn("I can't say", self.lead(builds=[], log_runs=0))
        self.assertIn("I can't say yet", self.lead(builds=[{"tag": TAG, "arch": ARCH, "relvals": 0}], log_runs=0))

    def test_architecture_without_relvals_is_unknown_not_a_no(self):
        # found replaying it: this IB is built on aarch64 but its RelVals aren't imported there
        other = "el9_aarch64_gcc14"
        lead = self.lead(f"is this exception in {TAG} on {other}\n{BLOCK}", exceptions=EXC, log_runs=0,
                         builds=[{"tag": TAG, "arch": ARCH, "relvals": 954}, {"tag": TAG, "arch": other, "relvals": 0}])
        self.assertTrue(lead.startswith(f"**I can't say yet:** no RelVal results or RelVal test logs are imported for `{TAG}` on `{other}`"), lead)
        self.assertIn(f"`{TAG}` on `{ARCH}`: `10804.31`, `10805.31`", lead)

    def test_only_a_sub_ib_and_only_another_architecture_are_told_apart(self):
        sub = "CMSSW_20_1_X_2026-11-03-2300"
        builds = [{"tag": sub, "arch": ARCH, "relvals": 9}, {"tag": TAG, "arch": ARCH, "relvals": 9}]
        lead = self.lead(f"is this in {sub} on {ARCH}\n{BLOCK}", builds=builds, exceptions=EXC)
        self.assertIn(f"**Not in `{sub}` itself on `{ARCH}`, but in its sub-IBs:**", lead)
        other = "el9_aarch64_gcc14"
        lead = self.lead(builds=[{"tag": TAG, "arch": ARCH, "relvals": 9}, {"tag": TAG, "arch": other, "relvals": 9}],
                         exceptions=[dict(EXC[0], arch=other)])
        self.assertTrue(lead.startswith("**No,"), lead)
        self.assertIn(f"on `{other}`", lead)

    def test_no_ib_named_or_failed_lookup_gives_no_direct_answer(self):
        self.assertIsNone(_pasted_error_in_ib_lead(BLOCK))
        broken = MagicMock()
        broken.session.side_effect = OSError("down")
        with patch("server.api_server.get_driver", return_value=broken):
            self.assertIsNone(_pasted_error_in_ib_lead(PASTE))


class RelvalCoverageTests(unittest.TestCase):
    def test_architecture_without_relvals_is_said_when_listing_exceptions(self):
        other = "el9_aarch64_gcc14"

        def answer(query, params):
            if "NOT (ib)-[:RAN_RELVAL]" in query:
                return [dict(arch=other)]
            if "exception.ib_tag" in query:
                return [dict(tag=TAG, arch=ARCH, workflow_id="10804.31", exception="boom", category="X", source_url=None)]
            return []

        question = f"what are the relval exceptions in {TAG}?"
        with patch("server.api_server.get_driver", return_value=fake_driver(answer)):
            text = _relval_exception_list_answer(question, _relval_exception_context(question))
        self.assertIn(f"Not covered, no RelVal results imported for this IB on: `{other}`.", text)


class FirstLastTests(unittest.TestCase):
    def test_by_date_not_by_release_series(self):
        text = _first_last_builds_text(["CMSSW_12_4_X_2026-11-03-2300", "CMSSW_20_1_TF_X_2026-08-12-1100"], "First in", "last in")
        self.assertEqual(text, "First in `CMSSW_20_1_TF_X_2026-08-12-1100`, last in `CMSSW_12_4_X_2026-11-03-2300`")

    def test_ties_are_said(self):
        text = _first_last_builds_text(["CMSSW_16_1_X_2026-10-04-2300", "CMSSW_17_0_X_2026-11-03-2300", TAG], "first in", "last in")
        self.assertEqual(text, f"first in `CMSSW_16_1_X_2026-10-04-2300`, last in `{TAG}` (and 1 other build(s) from that date)")
        self.assertEqual(_first_last_builds_text(["CMSSW_12_4_X_2026-11-03-2300", "CMSSW_13_0_X_2026-11-03-2300"], "first in", "last in"),
                         "all from the `2026-11-03-2300` builds")


class ResultCodeTests(unittest.TestCase):
    def test_only_stated_result_codes_count(self):
        self.assertEqual(result_codes("failed with exit code 65280"), {"65280"})
        self.assertEqual(result_codes("Exit status: 3 at /cvmfs/x/123 run 42"), {"3"})
        self.assertEqual(result_codes("file 123 at 12:00"), set())

    @patch("graph.tools._run_query_all")
    def test_split_keeps_the_same_code_and_problems_stating_none(self, query):
        query.return_value = [dict(sid="a", key="k1", tag="T1", text="failed with exit code 65280"),
                              dict(sid="a", key="k2", tag="T2", text="failed with exit code 256"),
                              dict(sid="a", key="k3", tag="T3", text="cut before the code")]
        split = _split_by_result_code(["a"], {"65280"})["a"]
        self.assertEqual(split["same"], 2)
        self.assertEqual(split["excluded"], ["k2"])
        self.assertEqual(split["other"], {"256": {"T2"}})

    @patch("server.api_server.test_log_coverage", return_value={"first": "2026-10-01-1100", "last": "2026-11-03-2300", "runs": 9})
    @patch("server.api_server.search_test_log_problems_report")
    def test_answer_names_the_other_codes_outside_the_exact_count(self, search, _cov):
        problem = dict(match="exact", kind="fatal_exception", category="ExternalFailure", summary="s", shared_terms=[], modules=[],
                       tests={}, architectures=[ARCH], ib_count=2, ib_tags=[TAG, "CMSSW_17_0_X_2026-10-04-2300"],
                       series_gone=[], series_total=1, places=[], evidence="e", example_tag=TAG, example_arch=ARCH,
                       example_member="m", example_line=1, example_path="p", origin=None,
                       other_result_codes=[dict(code="256", ib_count=23, example="CMSSW_15_0_X_2026-10-11-0000")])
        search.return_value = dict(problems=[problem], searched=[], failed=[])
        answer = _test_log_problem_answer(BLOCK)
        self.assertIn("Seen in **2 IBs**", answer)
        self.assertIn("code 256 in 23 IB(s), e.g. `CMSSW_15_0_X_2026-10-11-0000`; not counted above", answer)


class ExceptionHistoryTests(unittest.TestCase):
    def test_categories_are_split_and_order_is_labelled_truthfully(self):
        rows = [dict(tag=f"CMSSW_20_1_X_2026-10-{d:02d}-2300", arch=ARCH, category="ExternalFailure", step="step3",
                     workflows=["10804.31"], message="TritonService: Starting the fallback server failed with exit code 65280")
                for d in range(10, 20)]
        rows.append(dict(tag="CMSSW_12_4_X_2026-11-03-2300", arch="el8_amd64_gcc10", category="FallbackFailed", step="step3",
                         workflows=["10804.31"], message=rows[0]["message"]))

        def answer(query, params):
            return rows if "RelValException" in query else []

        with patch("server.api_server.get_driver", return_value=fake_driver(answer)):
            text = _relval_exception_message_answer(BLOCK)
        self.assertIn("categories `ExternalFailure` (10), `FallbackFailed` (1)", text)
        self.assertIn("Newest release series first:", text)
        self.assertIn("... and 3 more build/architecture pair(s)", text)
        self.assertNotIn("older", text)
        self.assertIn("First recorded in `CMSSW_20_1_X_2026-10-10-2300`, most recently in `CMSSW_12_4_X_2026-11-03-2300`.", text)


class OneLineErrorTests(unittest.TestCase):
    MESSAGE = "TritonService: Starting the fallback server failed with exit code 65280"

    def read(self, question, known=True):
        with patch("server.api_server.get_driver", return_value=fake_driver(lambda q, p: [dict(known=known)])):
            return _one_line_error_with_ib(question)

    def test_the_error_is_taken_out_of_the_words_around_it(self):
        for question in (f"{self.MESSAGE} in {TAG}", f"{self.MESSAGE} in `{TAG}` on `{ARCH}`?",
                         f"is this in {TAG} on {ARCH}: {self.MESSAGE}"):
            self.assertEqual(self.read(question), self.MESSAGE, question)

    def test_text_not_on_record_or_no_ib_or_a_failed_lookup_is_not_a_paste(self):
        self.assertIsNone(self.read(f"Note: please compare these two builds carefully in {TAG}", known=False))
        self.assertIsNone(self.read(self.MESSAGE))
        self.assertIsNone(self.read(f"which relvals failed in {TAG}"))
        broken = MagicMock()
        broken.session.side_effect = OSError("down")
        with patch("server.api_server.get_driver", return_value=broken):
            self.assertIsNone(_one_line_error_with_ib(f"{self.MESSAGE} in {TAG}"))

    def test_lead_uses_the_message_given_and_keeps_the_ib_and_architecture(self):
        with patch("server.api_server.get_driver", return_value=graph(exceptions=EXC)):
            lead = _pasted_error_in_ib_lead(f"{self.MESSAGE} in {TAG} on {ARCH}", self.MESSAGE)
        self.assertIn(f"**Yes, this exception is in `{TAG}` on `{ARCH}`:**", lead)

    @patch("server.api_server._chat_response", side_effect=lambda sid, q, answer, route, **kw: (answer, route))
    @patch("server.api_server._test_log_problem_answer", return_value=None)
    @patch("server.api_server._pasted_error_in_ib_lead", return_value="**Yes**")
    @patch("server.api_server._one_line_error_with_ib", return_value=MESSAGE)
    def test_chat_searches_by_the_error_alone(self, _one, lead, _log, _resp):
        with patch("server.api_server._relval_exception_message_answer", return_value="history") as exact:
            answer, route = _chat(ChatRequest(question=f"{self.MESSAGE} in {TAG}", session_id="one-line"))
        exact.assert_called_once_with(self.MESSAGE)
        lead.assert_called_once_with(f"{self.MESSAGE} in {TAG}", self.MESSAGE)
        self.assertEqual(answer, "**Yes**\n\nhistory")


class LooseArchScopeTests(unittest.TestCase):
    LATEST = "CMSSW_20_1_X_2026-11-04-1100"
    BUILT = "CMSSW_20_1_X_2026-11-03-2300"

    def setUp(self):
        _default_tag_notes.pop("loose", None)
        _latest_ib_contexts.pop("loose", None)

    def scope(self, question, words="gcc16", matches=(), found=("CMSSW_20_1_X_2026-11-03-2300", ["el9_amd64_gcc16"])):
        with patch("server.api_server._arch_scope_matches", return_value=(self.LATEST, words, list(matches), ["el9_amd64_gcc14"])), \
                patch("server.api_server._newest_tag_for_arch_word", return_value=found) as newest:
            result = _loose_arch_failure_scope(question, "loose")
        self.newest = newest
        return result

    def test_uses_the_newest_ib_built_there_and_says_so(self):
        question, answer = self.scope("what's broken on gcc16?")
        self.assertIsNone(answer)
        self.assertEqual(question, f"what's broken on gcc16? {self.BUILT} el9_amd64_gcc16")
        self.assertIn(f"`{self.LATEST}`, the newest IB, has no `gcc16` build yet", _default_tag_notes["loose"])

    def test_newest_ib_already_built_there_changes_nothing(self):
        q = "what's broken on gcc16?"
        self.assertEqual(self.scope(q, matches=["el9_amd64_gcc16"]), (q, None))
        self.assertNotIn("loose", _default_tag_notes)

    def test_never_built_there_says_so_instead_of_another_compiler(self):
        q = "what's broken on gcc99?"
        question, answer = self.scope(q, words="gcc99", found=None)
        self.assertEqual(question, q)
        self.assertEqual(answer, "No imported primary IB has RelVal results on `gcc99`.")

    def test_all_words_must_match_and_compiler_is_searched_first(self):
        q = "what's broken on el8 gcc16?"
        _, answer = self.scope(q, words="el8 gcc16")
        self.assertEqual(self.newest.call_args.args[0], "gcc16")
        self.assertIn("`el8 gcc16`", answer)

    def test_only_a_plain_overview_question_qualifies(self):
        for q in ("what's broken on gcc16?", "what\u2019s broken on aarch64", "arm failures", "failures on arm",
                  "what failed in the latest IB on aarch64"):
            self.assertTrue(_is_plain_arch_overview(q), q)
        # all found replaying real questions: each has an answer of its own
        for q in ("why would TritonService fail on aarch64?", "why does TestIntegrationEventSetup fail on aarch64?",
                  "show the error for SimG4Core/Geometry on el10", "did the build fail anywhere else besides el10?",
                  "is there an issue for the el10 ROCm failures?", "how many known errors on aarch64?",
                  "anything newly failing on aarch64?", "10805.31 on gcc15", "on el10", "gcc16"):
            self.assertFalse(_is_plain_arch_overview(q), q)
            self.assertEqual(self.scope(q), (q, None))

    def test_a_named_ib_or_session_ib_or_other_question_is_left_alone(self):
        for q in (f"what's broken on gcc16 in {self.LATEST}", "how many gcc16 builds are there"):
            self.assertEqual(self.scope(q), (q, None))
        _latest_ib_contexts["loose"] = self.LATEST
        self.assertEqual(self.scope("what's broken on gcc16?"), ("what's broken on gcc16?", None))
        _latest_ib_contexts.pop("loose")


class RewriteKeepsArchWordTests(unittest.TestCase):
    def test_rewrite_dropping_an_architecture_word_is_refused(self):
        generic = "what is failing in the latest IB"
        self.assertTrue(_rewrite_lost_arch_word("what's broken on gcc16?", generic))
        self.assertTrue(_rewrite_lost_arch_word("what's broken on aarch64?", generic))
        self.assertTrue(_rewrite_lost_arch_word("what's broken on el8?", generic))
        self.assertFalse(_rewrite_lost_arch_word("what's broken?", generic))
        self.assertFalse(_rewrite_lost_arch_word("what's broken on gcc16?", generic + " on gcc16"))


class WorkflowIdIsNotAPrNumberTests(unittest.TestCase):
    def test_workflow_id_is_never_read_as_a_pr_or_issue_number(self):
        question = "any open issue or PR for 34634.713"
        self.assertIsNone(_pr_number_answer(question))   # returns before any database call
        self.assertIsNone(RequestContext.extract(question).pr_number)
        self.assertIsNone(RequestContext.extract("PR 34634.713").pr_number)
        self.assertEqual(RequestContext.extract("PR 34634").pr_number, 34634)
        self.assertEqual(RequestContext.extract("what about #34634").pr_number, 34634)

    def workflow_answer(self, question, rows):
        def answer(query, params):
            return rows

        with patch("server.api_server.get_driver", return_value=fake_driver(answer)), \
                patch("server.api_server._workflow_error_evidence_lines", return_value=[]), \
                patch("server.api_server._github_export_note", return_value="GitHub data note"):
            return _workflow_artifact_answer(question)

    CLOSED = dict(kind="Issue", number=51512, title="fails", state="closed", merged_at=None, matched_field="title")
    MERGED = dict(kind="PR", number=51527, title="fix", state="closed", merged_at="2026-07-20T15:31:09Z", matched_field="body")

    def test_open_question_says_none_is_open(self):
        text = self.workflow_answer("any open issue or PR for 34634.713", [self.CLOSED, self.MERGED])
        self.assertTrue(text.startswith("**No open PR or Issue names workflow `34634.713`**. GitHub data note."), text)
        self.assertIn("**Issue #51512**", text)

    def test_open_question_says_how_many_are_open(self):
        text = self.workflow_answer("any open issue for 34634.713", [dict(self.CLOSED, state="open"), self.MERGED])
        self.assertTrue(text.startswith("**Yes: 1 open PR(s)/Issue(s) name workflow `34634.713`**"), text)

    def test_merged_pr_whose_state_says_open_is_not_open(self):
        text = self.workflow_answer("any open pr for 34634.713", [dict(self.MERGED, state="open")])
        self.assertTrue(text.startswith("**No open"), text)

    def test_question_not_about_open_has_no_such_line(self):
        text = self.workflow_answer("is there a PR for 34634.713", [self.CLOSED])
        self.assertTrue(text.startswith("PRs/Issues that name workflow"), text)


class PrNotInIbTests(unittest.TestCase):
    def text(self, merged_at, first="2026-07-23-2300", last="2026-11-04-1100"):
        with patch("server.api_server.get_driver", return_value=fake_driver(lambda q, p: [dict(first=first, last=last)])):
            return _pr_not_in_ib_text(merged_at)

    def test_each_case_says_what_is_true(self):
        self.assertEqual(self.text("2021-07-28T09:20:13Z"), "none, it was merged 2021-07-28, before the oldest imported build (2026-07-23)")
        self.assertEqual(self.text("2026-11-05T09:00:00Z"), "not yet, it was merged 2026-11-05, after the newest imported build (2026-11-04)")
        self.assertIn("merged-PR list is probably not imported", self.text("2026-09-01T00:00:00Z"))

    def test_failed_lookup_claims_no_date_order(self):
        broken = MagicMock()
        broken.session.side_effect = OSError("down")
        with patch("server.api_server.get_driver", return_value=broken):
            self.assertEqual(_pr_not_in_ib_text("2021-07-28T09:20:13Z"), "no imported IB lists it as newly merged")


if __name__ == "__main__":
    unittest.main()
