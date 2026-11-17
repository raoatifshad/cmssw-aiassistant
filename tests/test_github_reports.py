"""Has this error or workflow been reported? Found 2026-11-05 with real pastes against a direct search of the graph: 4 of 22 right."""
import json
import unittest
from unittest.mock import MagicMock, patch

from server.agent import draft_fix_suggestion, judge_answer_fit, route_to_known_question, judge_github_matches, parse_json_reply, review_fix_draft, verify_answer_against_evidence
from graph.tools import count_warnings_by_package
from server.cmssw_source import check_names, code_names, source_excerpts
from server.api_server import (
    _ai_checked_lines,
    _fix_draft_answer,
    _fix_draft_checks,
    ChatRequest,
    ChatResponse,
    _PENDING_TURN,
    _ai_supervised_chat,
    _ai_judged,
    Plan,
    PlanKind,
    _NEEDS_AI_ONLY,
    _ai_supervisor_history,
    _evidence_grounded,
    _run_llm_investigation,
    _second_model_checked,
    _chat_response,
    _reviewed_fix_draft,
    _with_second_opinion,
    _merge_readings,
    _github_reports_section,
    _new_exceptions_answer,
    _os_family_compare_answer,
    _package_clang_count_answer,
    _paste_search_terms,
    _reported_answer,
    _session_pastes,
    _split_paste_and_ask,
)

ROCM = ("An exception of category 'PluginNotFound' occurred while\nException Message:\n"
        "Unable to find plugin 'ROCmTestDeviceAdditionModule' in category 'CMS EDM Framework Module'. Please check spelling of name.")
ASAN = ("==735947==ERROR: AddressSanitizer: heap-use-after-free on address 0x115ef326c2d0 at pc 0x148ef707ae43\n"
        "    #0 0x148ef707ae42 in write ../../../../libsanitizer/sanitizer_common/sanitizer_common_interceptors.inc:1132\n"
        "    #1 0x108eeec24535 in edm::storage::File::syswrite(void const*, unsigned long)")


def row(number, kind="Issue", title_hit=False, body=True, human=0, bot=0, state="open"):
    return dict(kind=kind, repo="cmssw", number=number, title=f"item {number}", state=state, merged_at=None, created_at="2026-01-02T00:00:00Z", author="someone",
                in_title=title_hit, in_body=body, human_comments=human, bot_comments=bot)


def llm_reply(content):
    client = MagicMock()
    client.with_options.return_value = client
    client.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=content))]
    return client


class ParseJsonReplyTests(unittest.TestCase):
    def test_reasoning_after_the_json_is_ignored(self):
        self.assertEqual(parse_json_reply('{"verdict":"same"}We need answer only JSON. {maybe} } more'), {"verdict": "same"})

    def test_text_before_and_nested_objects(self):
        self.assertEqual(parse_json_reply('Here: {"a": {"b": 1}} done'), {"a": {"b": 1}})

    def test_no_object_is_none(self):
        for reply in (None, "", "no json", "[1, 2]", "{broken"):
            self.assertIsNone(parse_json_reply(reply))


class PasteTermsTests(unittest.TestCase):
    def test_a_sanitizer_report_is_searched_by_its_functions_not_its_paths(self):
        terms = _paste_search_terms(ASAN)
        self.assertEqual(terms["names"], ["File::syswrite"])
        self.assertEqual(terms["exact"], "")

    def test_a_quoted_name_is_searched_besides_the_whole_sentence(self):
        terms = _paste_search_terms(ROCM)
        self.assertIn("ROCmTestDeviceAdditionModule", terms["names"])
        self.assertIn("Unable to find plugin", terms["exact"])
        self.assertEqual(terms["wording"], "Unable to find plugin")
        self.assertEqual(terms["category"], "PluginNotFound")

    def test_the_question_is_not_part_of_the_error(self):
        for question in (f"has anyone reported this?\n\n{ROCM}", f"{ROCM}\n\nis this reported?", f"{ROCM}\n\ncheck the PR titles, issues and comments, has this been reported?"):
            paste, ask = _split_paste_and_ask(question)
            self.assertEqual(paste, ROCM)
            self.assertTrue(ask)


@patch("server.api_server._llm_client", None)
class SectionWithoutLlmTests(unittest.TestCase):
    TERMS = {"exact": "the whole sentence of it", "wording": "", "names": ["SpecificName"], "category": None}

    def section(self, found, ask=""):
        with patch("server.api_server.search_github_mentions", side_effect=lambda term: found.get(term, [])):
            return _github_reports_section("paste", ask, dict(self.TERMS))

    def test_says_where_it_was_found(self):
        answer = self.section({"SpecificName": [row(38370, title_hit=True, body=False), row(37767, body=False, human=1)]})
        self.assertIn("Found on GitHub: 2 item(s)", answer)
        self.assertIn("in title: `SpecificName`", answer)
        self.assertIn("in comments: `SpecificName`", answer)
        self.assertLess(answer.index("#38370"), answer.index("#37767"))

    def test_the_ci_bot_is_not_a_report(self):
        answer = self.section({"SpecificName": [row(51469, kind="PR", body=False, bot=3)]})
        self.assertIn("No human report found", answer)
        self.assertIn("PR #51469", answer)

    def test_nothing_found_says_what_was_searched(self):
        answer = self.section({})
        self.assertIn("Not found on GitHub", answer)
        self.assertIn("`SpecificName` (0)", answer)
        self.assertIn("export", answer)  # "full export" with the database, "an export" without

    def test_a_failed_search_is_never_not_found(self):
        def failing(term):
            if term == "SpecificName":
                raise RuntimeError("down")
            return []
        with patch("server.api_server.search_github_mentions", side_effect=failing):
            answer = _github_reports_section("paste", "", dict(self.TERMS))
        self.assertIn("This search failed", answer)

    def test_a_common_name_alone_is_not_a_report_but_two_together_are(self):
        many = [row(n) for n in range(1000, 1060)]
        answer = self.section({"SpecificName": many})
        self.assertIn("No PR or Issue found for this specific error", answer)
        answer = self.section({"SpecificName": many, "the whole sentence of it": [row(1000)] + [row(n) for n in range(2000, 2060)]})
        self.assertIn("Found on GitHub: 1 item(s)", answer)

    def test_asked_for_one_kind(self):
        found = {"SpecificName": [row(1), row(2, kind="PR")]}
        issue = self.section(found, "is there a github issue for it?")
        self.assertIn("**Issue #1**", issue)
        self.assertNotIn("**PR #2**", issue)
        self.assertIn("**PR #2**", self.section(found, "and any PR?"))


class AiCheckTests(unittest.TestCase):
    CANDIDATES = [{"id": "Issue #52323", "title": "t"}, {"id": "PR #40939", "title": "t"}]

    def test_either_key_form_is_read(self):
        judged = judge_github_matches(llm_reply(json.dumps({"52323": {"verdict": "same", "reason": "r"}, "PR #40939": {"verdict": "unrelated", "reason": ""}})), "m", "err", self.CANDIDATES)
        self.assertEqual(judged["Issue #52323"]["verdict"], "same")
        self.assertEqual(judged["PR #40939"]["verdict"], "unrelated")

    def test_only_given_candidates_and_known_verdicts_count(self):
        judged = judge_github_matches(llm_reply(json.dumps({"99999": {"verdict": "same"}, "52323": {"verdict": "certainly"}})), "m", "err", self.CANDIDATES)
        self.assertIsNone(judged)

    def test_prose_is_no_verdict(self):
        self.assertIsNone(judge_github_matches(llm_reply("I think they are related"), "m", "err", self.CANDIDATES))

    def test_every_checked_item_stays_visible(self):
        items = [dict(row(1), where={"description": ["x"]}), dict(row(2), where={"description": ["x"]}), dict(row(3), where={"description": ["x"]})]
        lines = "\n".join(_ai_checked_lines(items, {"Issue #1": {"verdict": "unrelated", "reason": ""}, "Issue #2": {"verdict": "related", "reason": "why"}}, "item(s)", ""))
        self.assertIn("No report of this same problem found", lines)
        self.assertIn("read 2 text match", lines)
        self.assertIn("Only share words, by the AI check: Issue #1", lines)
        self.assertIn("**Issue #2**", lines)
        self.assertIn("Not read by the AI check, text matches only: Issue #3", lines)
        self.assertIn("can be wrong", lines)

    @patch("server.api_server.get_github_item_excerpts", return_value=[{"id": "Issue #1", "title": "t", "description_match": ""}])
    @patch("server.api_server._llm_client", new_callable=MagicMock)
    def test_a_quoted_long_message_cant_be_only_shared_words(self, _client, _excerpts):
        terms = {"exact": "Starting the fallback server failed with exit code", "wording": "", "names": [], "category": None}
        with patch("server.api_server.search_github_mentions", return_value=[row(1)]), \
                patch("server.api_server.judge_github_matches", return_value={"Issue #1": {"verdict": "unrelated", "reason": "no"}}):
            answer = _github_reports_section("paste", "", terms)
        self.assertIn("Quotes this exact message", answer)
        self.assertNotIn("Only share words", answer)

    @patch("server.api_server._llm_client", new_callable=MagicMock)
    def test_llm_down_falls_back_to_text_matches(self, _client):
        terms = {"exact": "", "wording": "", "names": ["SpecificName"], "category": None}
        with patch("server.api_server.search_github_mentions", return_value=[row(1)]), \
                patch("server.api_server.get_github_item_excerpts", return_value=[{"id": "Issue #1"}]), \
                patch("server.api_server.judge_github_matches", side_effect=RuntimeError("timeout")):
            answer = _github_reports_section("paste", "", terms)
        self.assertIn("Found on GitHub: 1 item(s)", answer)


class TwoModelTests(unittest.TestCase):
    BATCH = [[{"id": "Issue #1"}, {"id": "Issue #2"}, {"id": "Issue #3"}]]

    def merged(self, first, second):
        with patch("server.api_server._llm_client", new_callable=MagicMock), patch("server.api_server.DEFAULT_MODEL", "a"), \
                patch("server.api_server.REVIEW_MODEL", "b"), patch("server.api_server.judge_github_matches", side_effect=second):
            return _with_second_opinion(first, "paste", self.BATCH)

    def test_same_needs_both_and_a_disagreement_is_shown_as_such(self):
        first = {"Issue #1": {"verdict": "same", "reason": "a"}, "Issue #2": {"verdict": "same", "reason": "b"}, "Issue #3": {"verdict": "related", "reason": "c"}}
        second = [{"Issue #1": {"verdict": "same", "reason": "x"}, "Issue #2": {"verdict": "related", "reason": "y"}, "Issue #3": {"verdict": "same", "reason": "z"}}]
        merged = self.merged(first, second)
        self.assertEqual([merged[k]["verdict"] for k in ("Issue #1", "Issue #2", "Issue #3")], ["same", "possible", "possible"])

    def test_no_second_opinion_means_same_is_not_confirmed(self):
        merged = self.merged({"Issue #1": {"verdict": "same", "reason": "a"}}, RuntimeError("down"))
        self.assertEqual(merged["Issue #1"]["verdict"], "possible")

    def test_possible_headline(self):
        item = dict(row(1), where={"title": ["x"]})
        lines = _ai_checked_lines([item], {"Issue #1": {"verdict": "possible", "reason": "r"}}, "item(s)", "")
        self.assertIn("Possibly reported: 1 item(s)", lines[0])
        self.assertIn("One AI check says the same problem: r", "\n".join(lines))


class ReadingsInTimeTests(unittest.TestCase):
    """2026-11-17: GLM took 45-110 s per batch of 8 and was dropped at its 20 s limit, after which gpt-oss ran: ~20 s for no AI check.
    Now both start at once and the answer waits for the review model plus a short grace only."""
    BATCHES = [[{"id": "Issue #1"}, {"id": "Issue #2"}]]

    def run_with(self, judge, grace=0.2, deadline=2.0):
        import server.api_server as api
        with patch("server.api_server._llm_client", new_callable=MagicMock), patch("server.api_server.DEFAULT_MODEL", "glm"), \
                patch("server.api_server.REVIEW_MODEL", "oss"), patch("server.api_server.judge_github_matches", side_effect=judge), \
                patch("server.api_server._AI_CHECK_GRACE_S", grace), patch("server.api_server._AI_CHECK_DEADLINE_S", deadline):
            import time
            started = time.monotonic()
            first, second = api._readings_in_time("paste", self.BATCHES)
            return first, second, time.monotonic() - started

    def test_a_slow_first_model_is_not_waited_for(self):
        import time

        def judge(_c, model, _p, batch):
            if model == "glm":
                time.sleep(1.5)
                return {"Issue #1": {"verdict": "unrelated", "reason": ""}}
            return {"Issue #1": {"verdict": "same", "reason": "r"}, "Issue #2": {"verdict": "related", "reason": ""}}
        first, second, took = self.run_with(judge)
        self.assertEqual(first, {})
        self.assertEqual(second["Issue #1"]["verdict"], "same")
        self.assertLess(took, 1.0)
        merged = _merge_readings(first, second)
        self.assertEqual(merged["Issue #1"], {"verdict": "possible", "reason": "r", "single": True})
        self.assertEqual(merged["Issue #2"]["verdict"], "related")

    def test_both_in_time_need_to_agree_on_same(self):
        def judge(_c, model, _p, batch):
            return {"Issue #1": {"verdict": "same", "reason": model}, "Issue #2": {"verdict": "same" if model == "glm" else "related", "reason": model}}
        first, second, _ = self.run_with(judge)
        merged = _merge_readings(first, second)
        self.assertEqual(merged["Issue #1"]["verdict"], "same")
        self.assertEqual(merged["Issue #2"]["verdict"], "possible")
        self.assertNotIn("single", merged["Issue #2"])

    def test_review_model_down_gives_the_first_one_the_rest_of_the_time(self):
        import time

        def judge(_c, model, _p, batch):
            if model == "oss":
                raise RuntimeError("502")
            time.sleep(0.5)
            return {"Issue #1": {"verdict": "same", "reason": "g"}}
        first, second, took = self.run_with(judge, grace=0.1)
        self.assertEqual(second, {})
        self.assertEqual(first["Issue #1"]["verdict"], "same")
        self.assertEqual(_merge_readings(first, second)["Issue #1"]["verdict"], "possible")

    def test_one_model_same_on_a_common_word_match_is_only_related(self):
        import server.api_server as api
        items = [dict(row(1), strong=True), dict(row(2), strong=False)]
        reading = {"Issue #1": {"verdict": "possible", "reason": "r", "single": True}, "Issue #2": {"verdict": "possible", "reason": "r", "single": True}}
        with patch("server.api_server._llm_client", new_callable=MagicMock), patch("server.api_server.DEFAULT_MODEL", "glm"), \
                patch("server.api_server.REVIEW_MODEL", "oss"), \
                patch("server.api_server.get_github_item_excerpts", return_value=[{"id": "Issue #1"}, {"id": "Issue #2"}]), \
                patch("server.api_server._readings_in_time", return_value=({}, {})), \
                patch("server.api_server._merge_readings", return_value=reading):
            checked = api._ai_checked_matches("paste", items, [])
        self.assertEqual(checked["Issue #1"]["verdict"], "possible")
        self.assertEqual(checked["Issue #2"], {"verdict": "related", "reason": "r"})

    def test_one_check_headline(self):
        item = dict(row(1), where={"title": ["x"]})
        lines = _ai_checked_lines([item], {"Issue #1": {"verdict": "possible", "reason": "r", "single": True}}, "item(s)", "")
        self.assertIn("only one AI check answered in time", lines[0])
        self.assertNotIn("disagree", lines[0])


class SourceCheckTests(unittest.TestCase):
    def test_names_found_in_a_text(self):
        found = code_names("Run `edmDumpEventContent` on it, see HeterogeneousTest/ROCmDevice/plugins and edm::storage::File::syswrite, in FxFxFileReader. GitHub says so.")
        self.assertIn("edmDumpEventContent", found["names"])
        self.assertIn("syswrite", found["names"])
        self.assertIn("FxFxFileReader", found["names"])
        self.assertNotIn("GitHub", found["names"])
        self.assertIn("HeterogeneousTest/ROCmDevice/plugins", found["paths"])

    def test_missing_only_when_in_neither_source_nor_context(self):
        def hits(name):
            return {"RealName": ["A/b.cc"]}.get(name, [])
        with patch("server.cmssw_source.source_version", return_value={"commit": "a", "date": "d"}), patch("server.cmssw_source.files_with", side_effect=hits):
            result = check_names("Use RealName, FakeToolName and ThePegThing.", "known: ThePegThing")
        self.assertEqual(result["missing"], ["FakeToolName"])
        self.assertIn("RealName", result["in_source"])

    def test_two_package_names_are_not_a_missing_path(self):
        def exists(path):
            return path == "FWCore"
        with patch("server.cmssw_source.source_version", return_value={"commit": "a", "date": "d"}), patch("server.cmssw_source.path_exists", side_effect=exists):
            result = check_names("Check Herwig/ThePEG and FWCore/NoSuchPackage.")
        self.assertEqual(result["missing"], ["FWCore/NoSuchPackage"])

    def test_no_checkout_is_no_verdict(self):
        with patch("server.cmssw_source.SOURCE_DIR", "/nonexistent/path"):
            self.assertIsNone(check_names("FakeToolName"))
            self.assertIsNone(source_excerpts("FakeToolName"))


class AiSupervisorTests(unittest.TestCase):
    FITS = dict(fits=True, problem="", standalone_question="")

    def response(self, answer, route="deterministic", sid="s1"):
        return ChatResponse(answer=answer, session_id=sid, route=route, citations=[], github_matches=None, actions=[])

    def misfit(self, standalone="is 537.0 new?"):
        return dict(fits=False, problem="p", standalone_question=standalone)

    def run_chat(self, chats, verdicts, investigated=None, question="is it new?", routed=None):
        with patch("server.api_server._llm_client", new_callable=MagicMock), patch("server.api_server.DEFAULT_MODEL", "a"), patch("server.api_server.REVIEW_MODEL", "b"), \
                patch("server.api_server.route_to_known_question", return_value=routed or {"standalone": "is 537.0 new?", "known": ""}), \
                patch("server.api_server._chat", side_effect=chats) as inner, patch("server.api_server.judge_answer_fit", side_effect=verdicts) as judge, \
                patch("server.api_server._run_llm_investigation", return_value=investigated) as agent, patch("server.api_server._store_chat_turns") as store:
            final = _ai_supervised_chat(ChatRequest(question=question, session_id="s1"))
        _ai_supervisor_history.pop("s1", None)
        return final, inner, judge, agent, store

    def test_a_fitting_answer_is_untouched(self):
        final, inner, _judge, agent, _store = self.run_chat([self.response("A")], [self.FITS])
        self.assertEqual(final.answer, "A")
        self.assertEqual(inner.call_count, 1)
        agent.assert_not_called()

    def test_a_misfit_goes_to_a_known_question_and_says_how_it_was_read(self):
        final, inner, _judge, agent, _store = self.run_chat([self.response("menu"), self.response("real answer", "capability:relval")], [self.misfit(), self.FITS],
                                                            routed={"standalone": "is 537.0 new?", "known": "what is failing in the latest IB"})
        self.assertEqual(inner.call_args.args[0].question, "what is failing in the latest IB")
        self.assertTrue(final.answer.startswith("_I read this as: what is failing in the latest IB_"))
        self.assertEqual(final.route, "ai_known:capability:relval")
        agent.assert_not_called()

    def test_no_known_question_goes_straight_to_the_investigation(self):
        final, inner, _judge, agent, _store = self.run_chat([self.response("A")], [self.misfit()], self.response("found", "ai:llm_validated"))
        self.assertEqual(inner.call_count, 1)
        self.assertEqual(agent.call_args.args[2], "is it new?")  # no conversation: asked as it was typed
        self.assertIn("found", final.answer)

    def test_a_follow_up_is_investigated_written_out(self):
        _ai_supervisor_history["s1"] = [{"question": "q", "answer": "a", "route": "deterministic"}]
        final, inner, _judge, agent, _store = self.run_chat([self.response("A")], [self.misfit()], self.response("found", "ai:llm_validated"))
        self.assertEqual(agent.call_args.args[2], "is 537.0 new?")

    def test_a_known_answer_that_does_not_fit_is_not_shown(self):
        final, _inner, _judge, agent, _store = self.run_chat([self.response("A"), self.response("B")], [self.misfit(), self.misfit()], self.response("found", "ai:llm_validated"),
                                                             routed={"standalone": "full", "known": "other"})
        self.assertEqual(agent.call_count, 1)
        self.assertIn("found", final.answer)

    def test_nothing_from_the_rules_goes_to_one_investigation(self):
        final, inner, judge, agent, _store = self.run_chat([self.response("", "needs_ai")], [], self.response("found", "ai:llm_validated"))
        self.assertEqual(agent.call_count, 1)
        judge.assert_not_called()
        self.assertTrue(final.answer.endswith("found"))

    def test_the_rules_never_run_the_ai_themselves_while_supervised(self):
        token = _NEEDS_AI_ONLY.set(True)
        try:
            with patch("server.api_server._llm_client", new_callable=MagicMock), patch("server.api_server.KGChatSession") as session:
                response = _run_llm_investigation("s", "q", "q", Plan(kind=PlanKind.LLM))
            session.assert_not_called()
            self.assertEqual(response.route, "needs_ai")
        finally:
            _NEEDS_AI_ONLY.reset(token)

    def test_a_failed_investigation_keeps_the_first_answer(self):
        final, *_ = self.run_chat([self.response("A")], [self.misfit()], self.response("A", "deterministic"))
        self.assertEqual(final.answer, "A")

    def test_second_model_judges_when_the_first_is_down(self):
        final, _inner, judge, *_ = self.run_chat([self.response("A")], [RuntimeError("down"), self.FITS])
        self.assertEqual(judge.call_count, 2)
        self.assertEqual(final.answer, "A")

    def test_two_named_builds_need_no_clarification(self):
        with patch("server.api_server._clear_pending_clarification"):
            final, _inner, judge, agent, _store = self.run_chat([self.response("Do you want a comparison?", "clarification")], [], self.response("found", "ai:llm_validated"),
                                                                question="did CMSSW_20_1_X_2026-11-04-2300 fix anything from CMSSW_20_1_X_2026-11-03-2300?")
        self.assertEqual(agent.call_count, 1)
        self.assertIn("found", final.answer)

    def test_both_down_keeps_the_answer(self):
        final, *_ = self.run_chat([self.response("A")], RuntimeError("down"))
        self.assertEqual(final.answer, "A")

    def test_turn_is_held_back_until_the_final_answer(self):
        pending = []
        token = _PENDING_TURN.set(pending)
        try:
            with patch("server.api_server._store_chat_turns") as inner_store:
                _chat_response("s1", "q", "a", "conversational:x")
            inner_store.assert_not_called()
        finally:
            _PENDING_TURN.reset(token)
        self.assertEqual(pending, ["s1", []])

    def test_answers_the_llm_already_read_and_pastes_are_skipped(self):
        for route in ("conversational:read_only", "llm:fix_draft"):
            final, _inner, judge, *_ = self.run_chat([self.response("A", route)], [])
            judge.assert_not_called()
        final, _inner, judge, *_ = self.run_chat([self.response("A")], [], question="x" * 601)
        judge.assert_not_called()

    def test_second_model_objection_gets_one_rewrite_then_raw_records(self):
        session = MagicMock()
        session.ask.return_value = "without the name"
        no = {"supported": False, "unsupported": ["a real name"]}
        with patch("server.api_server._llm_client", new_callable=MagicMock), patch("server.api_server.AI_SUPERVISOR_ENABLED", True), \
                patch("server.api_server.DEFAULT_MODEL", "a"), patch("server.api_server.REVIEW_MODEL", "b"):
            # the other model agrees with the answer: no rewrite
            with patch("server.api_server.verify_answer_against_evidence", side_effect=[no, {"supported": True, "unsupported": []}]):
                self.assertEqual(_second_model_checked(session, "q", "fine as it is", []), "fine as it is")
            # both object: one rewrite, checked again
            with patch("server.api_server.verify_answer_against_evidence", side_effect=[no, no, {"supported": True, "unsupported": []}]):
                self.assertEqual(_second_model_checked(session, "q", "with a name", []), "without the name")
            with patch("server.api_server.verify_answer_against_evidence", side_effect=[no, no, no, no]):
                self.assertIsNone(_second_model_checked(session, "q", "with a name", []))
            with patch("server.api_server.verify_answer_against_evidence", side_effect=[no, no, no, {"supported": True, "unsupported": []}]):
                self.assertEqual(_second_model_checked(session, "q", "with a name", []), "without the name")
            with patch("server.api_server.verify_answer_against_evidence", side_effect=RuntimeError("down")):
                self.assertEqual(_second_model_checked(session, "q", "with a name", []), "with a name")

    def test_an_absence_claim_needs_results_that_cover_the_scope(self):
        client = llm_reply('{"supported": true, "unsupported": []}')
        verify_answer_against_evidence(client, "m", "q", "none", [], absence=True)
        system = client.with_options.return_value.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("One empty result proves nothing", system)
        verify_answer_against_evidence(client, "m", "q", "some", [])
        system = client.with_options.return_value.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertNotIn("One empty result proves nothing", system)

    def test_a_clock_time_is_not_two_numbers(self):
        self.assertTrue(_evidence_grounded("The 23:00 build", [{"result": "CMSSW_20_1_X_2026-11-04-2300"}]))
        self.assertFalse(_evidence_grounded("It has 23 failures", [{"result": "nothing"}]))

    def test_the_conversation_remembers_the_answer_without_our_notes(self):
        tag = "CMSSW_20_1_X_2026-11-05-1100"
        with patch("server.api_server.partial_builds", return_value=[dict(tag=tag, arch="el9", results=3, usual=954)]):
            final, *_ = self.run_chat([self.response(f"failures in `{tag}`", "capability:relval")], [self.FITS], question="what is failing?")
        self.assertTrue(final.answer.startswith("**Still being imported"))
        # run_chat clears the history afterwards, so read what was kept while it is there
        with patch("server.api_server.partial_builds", return_value=[dict(tag=tag, arch="el9", results=3, usual=954)]), patch("server.api_server._llm_client", new_callable=MagicMock), \
                patch("server.api_server._chat", return_value=self.response(f"failures in `{tag}`", "capability:relval")), patch("server.api_server.judge_answer_fit", return_value=self.FITS), patch("server.api_server._store_chat_turns"):
            _ai_supervised_chat(ChatRequest(question="what is failing?", session_id="s1"))
        self.assertEqual(_ai_supervisor_history.pop("s1")[-1]["answer"], f"failures in `{tag}`")

    def test_router_replies_are_read_safely(self):
        self.assertEqual(route_to_known_question(llm_reply('{"standalone": "s", "known": "k"} We need'), "m", "q", [], "list"), {"standalone": "s", "known": "k"})
        self.assertIsNone(route_to_known_question(llm_reply("no idea"), "m", "q", [], "list"))

    def test_a_dead_end_is_not_shown_as_the_answer_when_the_investigation_fails(self):
        menu = self.response("RelVal failures over this period span several architectures, which I don't combine unless you ask. Choose one", "deterministic")
        final, *_ = self.run_chat([menu], [self.misfit()], self.response("x", "ai:llm_ungrounded_fallback"), question="in which tests?")
        self.assertEqual(final.route, "ai:no_reliable_answer")
        self.assertNotIn("Choose one", final.answer)

    def test_the_second_model_doesnt_bring_back_an_answer_the_judge_rejected(self):
        final, *_ = self.run_chat([self.response("run this command")], [self.misfit()], self.response("not stored", "ai:llm_validated"))
        self.assertEqual(final.answer, "not stored")

    def test_judge_reply_is_read_safely(self):
        self.assertIsNone(judge_answer_fit(llm_reply("I think it fits"), "m", "q", "a"))
        decided = judge_answer_fit(llm_reply('{"fits": false, "problem": "wrong", "standalone_question": "full"}We need answer only JSON'), "m", "q", "a")
        self.assertEqual(decided["standalone_question"], "full")


class SubsystemWarningCountTests(unittest.TestCase):
    def test_a_subsystem_counts_its_packages(self):
        rows = [[{"distinct_locations": 0, "total_occurrences": 0}], [{"package": "HLTrigger/Timer", "distinct_locations": 19, "total_occurrences": 76}, {"package": "HLTrigger/Tools", "distinct_locations": 8, "total_occurrences": 32}]]
        with patch("graph.tools._run_query", side_effect=rows):
            result = count_warnings_by_package("HLTrigger")
        self.assertEqual((result["distinct_locations"], result["total_occurrences"]), (27, 108))
        self.assertIn("2 packages under HLTrigger/", result["scope"])

    def test_a_package_with_its_own_count_is_left_alone(self):
        with patch("graph.tools._run_query", return_value=[{"distinct_locations": 5, "total_occurrences": 6}]) as query:
            self.assertEqual(count_warnings_by_package("FWCore/Services")["distinct_locations"], 5)
        self.assertEqual(query.call_count, 1)


class FollowUpTests(unittest.TestCase):
    def setUp(self):
        _session_pastes["s"] = ROCM

    def tearDown(self):
        _session_pastes.pop("s", None)

    def test_follow_ups_keep_the_paste(self):
        with patch("server.api_server._github_reports_section", return_value="ANSWER") as section:
            for question in ("has anyone reported it?", "is there a github issue for it?", "and any PR?", "who reported it?"):
                self.assertEqual(_reported_answer(question, "s"), "ANSWER", question)
                self.assertEqual(section.call_args.args[0], ROCM)

    def test_other_questions_are_left_alone(self):
        with patch("server.api_server._github_reports_section") as section:
            for question in ("is it new?", "since when?", "what is the latest IB?", "any PR about the LST segfault in the HLT menu?"):
                self.assertIsNone(_reported_answer(question, "s"), question)
            section.assert_not_called()


class WorkflowTests(unittest.TestCase):
    def test_a_workflow_question_goes_to_the_workflow_search(self):
        with patch("server.api_server._workflow_github_reports_answer", return_value="W") as search:
            for question in ("has 537.0 been reported?", "is there an issue for workflow 539.0?", "any PR about 10805.31?"):
                self.assertEqual(_reported_answer(question, "none"), "W", question)
            self.assertEqual(search.call_args.args[0], "10805.31")

    def test_open_asks_keep_their_own_answer(self):
        with patch("server.api_server._workflow_github_reports_answer") as search:
            self.assertIsNone(_reported_answer("any open issue for 539.0?", "none"))
            search.assert_not_called()


class FixDraftTests(unittest.TestCase):
    DRAFT = {"likely_cause": "It might be the module. Tests confirm this.", "suggested_change": "Look at it. It takes 20 minutes.",
             "pr_title": "Register the module", "pr_description": "Registers the module so it is found."}

    def answer(self, draft=None, question=f"draft a PR title and description for this\n\n{ROCM}", names=None, review=None):
        report = {"items": [dict(row(40680), strong=True, where={"comments": ["x"]})], "searched": ["x"]}
        with patch("server.api_server._llm_client", new_callable=MagicMock), patch("server.api_server.DEFAULT_MODEL", "a"), patch("server.api_server.REVIEW_MODEL", "b"), \
                patch("server.api_server._github_mentions_report", return_value=report), \
                patch("server.api_server.get_github_item_excerpts", return_value=[{"id": "Issue #40680", "title": "t"}]), \
                patch("server.api_server.draft_fix_suggestion", return_value=draft or dict(self.DRAFT)), \
                patch("server.api_server.source_excerpts", return_value=[]), \
                patch("server.api_server.source_version", return_value={"commit": "abc", "date": "2026-09-22"}), \
                patch("server.api_server.check_names", return_value=names if names is not None else {"missing": [], "in_source": {}}), \
                patch("server.api_server.review_fix_draft", return_value=review):
            return _fix_draft_answer(question, "none")

    def test_labelled_with_the_project_pr_template_and_validation_left_open(self):
        answer = self.answer()
        self.assertIn("AI draft, not tested", answer)
        self.assertIn("#### PR description:", answer)
        self.assertIn("#### PR validation:", answer)
        self.assertIn("To fill in after running it", answer)
        self.assertIn("Based on: Issue #40680", answer)

    def test_results_nobody_measured_are_removed(self):
        answer = self.answer()
        self.assertNotIn("Tests confirm", answer)
        self.assertNotIn("20 minutes", answer)
        self.assertIn("It might be the module.", answer)

    def test_invented_references_and_cmake_are_not_shown(self):
        self.assertIn("aren't in its evidence", self.answer(dict(self.DRAFT, pr_description="Fixes #99999.")))
        self.assertIn("CMake", self.answer(dict(self.DRAFT, suggested_change="Edit CMakeLists.txt.")))
        self.assertNotIn("PR title", self.answer(dict(self.DRAFT, suggested_change="Edit CMakeLists.txt.")))

    def test_code_that_does_not_exist_is_removed_and_named(self):
        draft = dict(self.DRAFT, suggested_change="Run edmInputCopier on the file. Look at the module.")
        answer = self.answer(draft, names={"missing": ["edmInputCopier"], "in_source": {"edmDumpEventContent": ["x"]}})
        self.assertNotIn("Run edmInputCopier", answer)
        self.assertIn("Look at the module.", answer)
        self.assertIn("Not found in the source", answer)
        self.assertIn("`edmInputCopier`", answer)

    def test_an_unavailable_checkout_is_not_called_clean(self):
        with patch("server.api_server.check_names", return_value=None):
            self.assertEqual(_fix_draft_checks(None, None, []).count("- Code names: not checked, the CMSSW checkout couldn't be read."), 1)

    def test_a_contradicted_draft_is_flagged_first(self):
        review = {"claims": [{"claim": "It is the build", "verdict": "contradicted", "source": "Issue #40680"}], "addresses_failure": "no", "problems": ["wrong package"]}
        answer = self.answer(review=review)
        self.assertTrue(answer.startswith("**Don't rely on this draft"))
        self.assertIn("| It is the build | **contradicted** | Issue #40680 |", answer)
        self.assertIn("Still open: wrong package", answer)

    def test_review_findings_go_back_to_the_author_once(self):
        first = {"claims": [], "addresses_failure": "no", "problems": ["wrong package"]}
        good = {"claims": [{"claim": "c", "verdict": "backed", "source": "failure"}], "addresses_failure": "yes", "problems": []}
        rewrite = dict(self.DRAFT, pr_title="Better")
        with patch("server.api_server._llm_client", new_callable=MagicMock), patch("server.api_server.DEFAULT_MODEL", "a"), patch("server.api_server.REVIEW_MODEL", "b"), \
                patch("server.api_server.check_names", return_value={"missing": [], "in_source": {}}), \
                patch("server.api_server.review_fix_draft", side_effect=[first, good]), \
                patch("server.api_server.draft_fix_suggestion", return_value=rewrite) as author:
            review, _names = _reviewed_fix_draft(dict(self.DRAFT), "f", [], [], "f")
        self.assertEqual(author.call_count, 1)
        self.assertIn("wrong package", author.call_args.kwargs["review_notes"])
        self.assertEqual(review["draft"]["pr_title"], "Better")
        self.assertEqual(review["addresses_failure"], "yes")

    def test_a_reviewer_cannot_cite_what_it_was_not_given(self):
        reply = json.dumps({"claims": [{"claim": "a", "verdict": "backed", "source": "Issue #999"}, {"claim": "b", "verdict": "backed", "source": "A/B/c.cc"},
                                       {"claim": "c", "verdict": "contradicted", "source": "nowhere"}], "addresses_failure": "yes", "problems": []})
        review = review_fix_draft(llm_reply(reply), "m", "f", [{"id": "Issue #1"}], [{"id": "A/B/c.cc"}], dict(self.DRAFT))
        self.assertEqual([c["verdict"] for c in review["claims"]], ["inference", "backed", "inference"])

    def test_a_search_in_pr_descriptions_is_not_a_draft_request(self):
        self.assertIsNone(_fix_draft_answer("does anyone mention 10805.31 in a PR description?", "none"))

    def test_needs_a_subject(self):
        self.assertIn("Which failure", _fix_draft_answer("suggest a fix", "none"))

    def test_other_questions_are_left_alone(self):
        for question in ("what is the latest IB?", "show me PR 52393", "any PR about LST?", "was it fixed?", "fix it for me"):
            self.assertIsNone(_fix_draft_answer(question, "none"), question)

    def test_draft_needs_all_fields(self):
        self.assertIsNone(draft_fix_suggestion(llm_reply(json.dumps({"pr_title": "x"})), "m", "failure", []))


class FalseAnswerTests(unittest.TestCase):
    def test_subsystem_clang_count_is_not_zero(self):
        session = MagicMock()
        session.__enter__.return_value = session
        session.run.side_effect = [[{"package": "FWCore/Services", "locations": 120, "total": 130}], MagicMock(value=lambda: ["CMSSW_20_1_X_2026-08-12-2300"])]
        driver = MagicMock()
        driver.session.return_value = session
        with patch("server.api_server.get_driver", return_value=driver):
            answer = _package_clang_count_answer("how many clang warnings does FWCore have?")
        self.assertIn("120 distinct warning location(s)", answer)
        self.assertIn("2026-08-12-2300", answer)
        self.assertIsNone(_package_clang_count_answer("how many warnings in the latest IB?"))

    def test_os_families_side_by_side(self):
        table = ("In `x`:\n\n| Architecture | Failing RelVals | Failing unit tests | Failed builds | RelVal results imported |\n"
                 "| `el9_amd64_gcc14` | 14 | 5 | 0 of 11 | 7,911 |\n| `el9_aarch64_gcc14` | 1 | 3 | 0 of 2 | 1,017 |\n| `el8_amd64_gcc14` | 1 | 0 | 0 of 1 | 954 |")
        with patch("server.api_server._failures_by_dimension_answer", return_value=table):
            answer = _os_family_compare_answer("are the el8 builds in worse shape than el9 today?")
        self.assertIn("`el8`: 1 failing RelVal(s) of 954", answer)
        self.assertIn("`el9`: 15 failing RelVal(s) of 8,928", answer)

    @patch("server.api_server._cutoff_stamp", return_value="2026-11-03-0000")
    @patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-04-2300"}])
    def test_a_changed_number_is_not_a_new_crash(self, _latest, _cutoff):
        session = MagicMock()
        session.__enter__.return_value = session
        session.run.return_value = [
            dict(category="EventGenerationFailure", message="failed waiting for external process 448905_0.", first="2026-10-01-1100", workflows=1),
            dict(category="EventGenerationFailure", message="failed waiting for external process 2173015_0.", first="2026-11-04-2300", workflows=1),
            dict(category="Other", message="a really new one", first="2026-11-04-2300", workflows=2)]
        driver = MagicMock()
        driver.session.return_value = session
        with patch("server.api_server.get_driver", return_value=driver):
            answer = _new_exceptions_answer("any new crash type in today's builds?")
        self.assertIn("a really new one", answer)
        self.assertNotIn("external process", answer)


if __name__ == "__main__":
    unittest.main()
