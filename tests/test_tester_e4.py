"""Wrong answers found by item E4 of the weekend test (2026-11-10): pasted RelVal exceptions and test-log problems asked as
"has anyone reported this?" / "what is this?"."""
import unittest
from unittest.mock import patch

from server import api_server


class PasteWithoutTheQuestionTests(unittest.TestCase):
    """E4-1: the person's words stayed inside the pasted error, so the exact message searched on GitHub matched nothing."""

    def test_question_glued_to_the_error_is_cut(self):
        cases = {
            "Expected input collection size 36, got 6 -- known?": "Expected input collection size 36, got 6",
            "No dictionary for edm::Wrapper<int> again in 136.901, any github issue?": "No dictionary for edm::Wrapper<int> again in 136.901",
            "ModuleNotFoundError: No module named 'RecoTICL.LayerClustering.ticlLayerTileProducer_cfi'\non riscv, known?":
                "ModuleNotFoundError: No module named 'RecoTICL.LayerClustering.ticlLayerTileProducer_cfi'",
            "seeing this in 7.5 step2 on 20_1, anyone reported it?\nNo \"BTLReadoutMapRcd\" record found in the EventSetup.":
                "No \"BTLReadoutMapRcd\" record found in the EventSetup.",
            "PluginNotFound: Unable to find plugin 'EvtGen130'\nwhat is this? EVTGEN3 IB": "PluginNotFound: Unable to find plugin 'EvtGen130'",
            "known issue? 1048.0 step3 dies with\nIllegal parameter found in configuration.": "Illegal parameter found in configuration.",
        }
        for paste, expected in cases.items():
            self.assertEqual(api_server._strip_paste_ask(paste), expected, paste)

    def test_log_text_is_never_cut(self):
        for paste in ("NameError: name 'foo' is not defined. Did you mean: 'for'?",
                      "got 'An exception of unknown type was thrown.' in 536.0 step1 on 17_1 aarch64",
                      "%MSG-e Foo: bar baz"):
            self.assertEqual(api_server._strip_paste_ask(paste), paste)
        self.assertEqual(api_server._strip_paste_ask("ValueError: bad value -- anyone seen this?"), "ValueError: bad value")

    def test_exact_message_searched_without_the_question(self):
        with patch.object(api_server, "parse_pasted_error", return_value={}):
            terms = api_server._paste_search_terms("10804.31 step3 fails with ExternalFailure: SonicCallFailed: call failed after max 1 tries -- has this been reported?")
        self.assertEqual(terms["exact"], "call failed after max 1 tries")

    def test_one_line_paste_with_the_question_at_its_end_is_split(self):
        paste, ask = api_server._split_paste_and_ask("ExternalFailure: SonicCallFailed: call failed after max 1 tries, anyone seen this before?")
        self.assertEqual(paste, "ExternalFailure: SonicCallFailed: call failed after max 1 tries")
        self.assertEqual(ask, "anyone seen this before?")
        self.assertEqual(api_server._split_paste_and_ask("what is the status of 537.0, anyone know?")[1], "")  # no error text: not a paste


class SameErrorOtherWorkflowTests(unittest.TestCase):
    """E4-2: "No dictionary for edm::Wrapper<int> again in 136.901" -> the AI check called open Issue #49120 (the same error, other RelVals)
    related, "different workflow number", so the answer said no report was found."""

    def test_judge_is_told_another_workflow_is_the_same_problem(self):
        import json
        from unittest.mock import MagicMock
        from server.agent import judge_github_matches
        client = MagicMock()
        client.with_options.return_value.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=json.dumps({"49120": {"verdict": "same", "reason": "r"}})))]
        judge_github_matches(client, "m", "No dictionary for edm::Wrapper<int> in 136.901", [{"id": "Issue #49120", "title": "t"}])
        system = client.with_options.return_value.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("another RelVal workflow", system)
        self.assertIn("recurring in other workflows is same", system)

    def _section(self, reason):
        from unittest.mock import MagicMock
        from tests.test_github_reports import row
        terms = {"exact": "", "wording": "", "names": ["SpecificName"], "category": None}
        with patch("server.api_server._llm_client", new_callable=MagicMock), \
                patch("server.api_server.get_github_item_excerpts", return_value=[{"id": "Issue #1", "title": "t"}]), \
                patch("server.api_server.search_github_mentions", return_value=[row(1)]), \
                patch("server.api_server.REVIEW_MODEL", api_server.DEFAULT_MODEL), \
                patch("server.api_server.judge_github_matches", return_value={"Issue #1": {"verdict": "related", "reason": reason}}):
            return api_server._github_reports_section("paste", "", terms)

    def test_related_whose_reason_says_same_is_not_no_report(self):
        answer = self._section("Same DictionaryNotFound for edm::Wrapper<int> in RelVal, another workflow")
        self.assertIn("Possibly reported", answer)
        self.assertNotIn("No report of this same problem found", answer)

    def test_related_whose_reason_names_a_difference_stays_related(self):
        for reason in ("Same test failing in IBs, but failure mode not shown as segfault", "Same DictionaryNotFound but for reco::BeamSpot"):
            self.assertIn("No report of this same problem found", self._section(reason), reason)


class PastedErrorWithWorkflowTests(unittest.TestCase):
    """E4-3: a pasted error naming its workflow went to "has <workflow> been reported?", which counts the same error elsewhere as only related."""

    def test_error_with_a_workflow_number_is_a_paste(self):
        paste, ask = api_server._split_paste_and_ask("No dictionary for edm::Wrapper<int> again in 136.901, any github issue?")
        self.assertEqual(paste, "No dictionary for edm::Wrapper<int> again in 136.901")
        self.assertEqual(ask, "any github issue?")
        self.assertEqual(api_server._split_paste_and_ask("537.0 is failing again in the ASAN build, anyone know?")[1], "")

    def test_known_workflow_report_question_is_dropped_for_a_pasted_error(self):
        q = ("136.902 step4 in 16_1 last night: DictionaryNotFound: No data dictionary found for the following classes: edm::Wrapper<int> "
             "Most likely each dictionary was never generated -- reported?")
        self.assertTrue(api_server._known_question_changes_ask("has 136.902 been reported?", q))
        self.assertFalse(api_server._known_question_changes_ask("has 537.0 been reported?", "has 537.0 been reported on github?"))
        self.assertFalse(api_server._known_question_changes_ask("has 537.0 been reported?", "537.0 failing in ASAN, is there an issue?"))


class CategoryWordAndCommonNamesTests(unittest.TestCase):
    """E4-4 (no AI check): "OutOfBound: TC X1 ..." searched OutOfBound as a name, so 10 Issues with only that word "mention this error" (R10);
    #47744 (ASAN, shares EDConsumerBase + OfflineValidation) shown as mentioning a UBSan error (T01)."""
    CATEGORIES = {"OutOfBound", "ExternalFailure", "SonicCallFailed", "ThePEG::Exception", "ExternalLHEProducer"}

    def terms(self, paste):
        with patch.object(api_server, "parse_pasted_error", return_value={}), \
                patch.object(api_server, "_known_exception_categories", return_value=self.CATEGORIES):
            return api_server._paste_search_terms(paste)

    def test_the_pastes_own_category_is_not_a_name(self):
        terms = self.terms("OutOfBound: TC X1 = 38212.1 out of the seeding histogram bounds 0.076 - 0.58")
        self.assertEqual(terms["category"], "OutOfBound")
        self.assertNotIn("OutOfBound", terms["names"])

    def test_a_second_category_and_qualified_names_stay_names(self):
        terms = self.terms("ExternalFailure: SonicCallFailed: call failed after max 1 tries")
        self.assertEqual(terms["category"], "ExternalFailure")
        self.assertIn("SonicCallFailed", terms["names"])
        self.assertIn("ThePEG::Exception", self.terms("caught exception of type ThePEG::Exception in FxFxFileReader")["names"])

    def section(self, found, terms):
        from tests.test_github_reports import row
        rows = {t: [row(n, **kw) for n, kw in items] for t, items in found.items()}
        with patch("server.api_server._llm_client", None), \
                patch("server.api_server.search_github_mentions", side_effect=lambda term: rows.get(term, [])):
            return api_server._github_reports_section("paste", "", terms)

    def test_two_common_names_together_are_not_shown_as_mentioning_the_error(self):
        terms = {"exact": "", "wording": "", "names": ["EDConsumerBase", "OfflineValidation"], "category": None}
        common = [(n, {}) for n in range(3000, 3050)]
        answer = self.section({"EDConsumerBase": [(47744, {})] + common, "OfflineValidation": [(47744, {})] + common[:45]}, terms)
        self.assertNotIn("mention this error", answer)
        self.assertIn("No PR or Issue found that names this specific error", answer)
        self.assertIn("#47744", answer)
        answer = self.section({"EDConsumerBase": [(47744, {})] + common, "OfflineValidation": [(47744, {})] + common[:45],
                               "RareName": [(46727, {})]}, dict(terms, names=terms["names"] + ["RareName"]))
        self.assertIn("Found on GitHub: 1 item(s) mention this error", answer)
        self.assertLess(answer.index("#46727"), answer.index("may be a different problem"))
        self.assertIn("may be a different problem: Issue #47744", answer)

    def test_long_value_free_wording_on_few_items_names_the_error(self):
        terms = {"exact": "TC X1 = 38212.1 out of the seeding histogram bounds 0.076 - 0.58", "wording": "out of the seeding histogram bounds",
                 "names": [], "category": None}
        answer = self.section({"out of the seeding histogram bounds": [(49795, {})] + [(n, {}) for n in range(5000, 5011)]}, terms)
        self.assertIn("Found on GitHub", answer)
        self.assertIn("#49795", answer)


class SanitizerHeadlineTests(unittest.TestCase):
    """E4-5 (e4:T09): an ASAN paste with shortened frames ("#0 in operator delete", no address) was read as an exception, its "exact" message
    was "0x5020... in thread T0 #0 in operator delete ...", and the headline quoted in the title of #45670 was never searched."""

    def terms(self, paste):
        with patch.object(api_server, "_known_exception_categories", return_value=set()):
            return api_server._paste_search_terms(paste)

    def test_the_headline_is_the_exact_message(self):
        terms = self.terms("==2324182==ERROR: AddressSanitizer: attempting free on address which was not malloc()-ed: 0x5020010f4250 in thread T0\n"
                           "    #0 in operator delete(void*, unsigned long)\n    #1 in TProtoClass::~TProtoClass() libCore.so\nwhat is this? 537.0 in ASAN")
        self.assertEqual(terms["exact"], "attempting free on address which was not malloc()-ed")
        self.assertIn("~TProtoClass", terms["names"])

    def test_a_headline_every_report_has_is_not_searched_as_the_message(self):
        for paste in ("==7==ERROR: AddressSanitizer: heap-use-after-free on address 0x115ef326c2d0 at pc 0x148ef707ae43\n"
                      "    #1 0x145a5ae1522c in edm::storage::File::syswrite(void const*, unsigned long) (/x/lib.so+0xd722c)",
                      "==1==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000 (pc 0x1 bp 0x2 sp 0x3 T0)\n    #0 0x1 in foo::Bar::baz() (/x/lib.so+0x1)",
                      "==12==ERROR: LeakSanitizer: detected memory leaks\n    #1 0x2 in edm::Foo::make() lib.so"):
            terms = self.terms(paste)
            self.assertEqual(terms["exact"], "", paste)
            self.assertTrue(terms["names"], paste)


class ReviewerNotesAreNotFactsTests(unittest.TestCase):
    """E4-6 (e4:P03 T2): the second model objected "Issue 52323 was opened by Dr15Jones, but the database shows author cmsbuild" (wrong), the
    rewrite prompt passed that on as the correction, and the answer then said the Issue was opened by cmsbuild with no discussion."""

    def test_rewrite_prompt_treats_the_objections_as_doubts_to_check(self):
        from unittest.mock import MagicMock
        session = MagicMock()
        session.ask.return_value = "rewritten"
        no = {"supported": False, "unsupported": ["Issue 52323 was opened by Dr15Jones, but the database shows author cmsbuild."]}
        with patch("server.api_server._llm_client", new_callable=MagicMock), patch("server.api_server.AI_SUPERVISOR_ENABLED", True), \
                patch("server.api_server.DEFAULT_MODEL", "a"), patch("server.api_server.REVIEW_MODEL", "b"), \
                patch("server.api_server.verify_answer_against_evidence", side_effect=[no, no, {"supported": True, "unsupported": []}]):
            self.assertEqual(api_server._second_model_checked(session, "q", "opened by Dr15Jones", []), "rewritten")
        prompt = session.ask.call_args.args[0]
        self.assertIn("The reviewer can be wrong", prompt)
        self.assertIn("Never take a name, number, date or claim from the reviewer's notes", prompt)
        self.assertNotIn("aren't supported by the tool results", prompt)


class CategoryMessagePasteTests(unittest.TestCase):
    """E4-7 (e4:R15, T14, R02): "Category: message" pastes asked as "what is this?" went to the exit-code explainer or the AI."""

    def test_category_message_is_a_paste(self):
        cases = {
            "what is this?\nExternalLHEProducer: Child failed with exit code 1.": "ExternalLHEProducer: Child failed with exit code 1.",
            "UnavailableAccelerator: Module alpakaGlobalProducer has the Alpaka backend set explicitly, but its accelerator is not available. what is this?":
                "UnavailableAccelerator: Module alpakaGlobalProducer has the Alpaka backend set explicitly, but its accelerator is not available",
            "what is this? BTLReadoutMap::getElectronicsId: BTLDetId 1659115401 not found!": "BTLReadoutMap::getElectronicsId: BTLDetId 1659115401 not found!",
        }
        for question, paste in cases.items():
            self.assertEqual(api_server._category_error_paste(question), paste, question)

    def test_context_lines_are_not_part_of_the_message(self):
        paste = "ExternalLHEProducer: Child failed with exit code 1.\n   Calling method for module ExternalLHEProducer/'externalLHEProducer'"
        self.assertEqual(api_server._pasted_error_message(paste), ("ExternalLHEProducer", "Child failed with exit code 1."))
        two_lines = "ThePEG::Exception: caught exception of type ThePEG::Exception.\nWhile reading object number 1 of class ThePEG::EventGenerator"
        self.assertIn("While reading object number 1", api_server._pasted_error_message(two_lines)[1])

    def test_questions_are_not_pastes(self):
        for question in ("what does exit code 65 mean?", "Question: why is 537.0 failing in the newest IB?", "what does ExternalLHEProducer do?",
                         "summary: what failed in CMSSW_20_1_X_2026-11-08-2300 today?", "PluginNotFound: what is it?",
                         "which workflows fail with PluginNotFound: Unable to find plugin in the latest IB?"):
            self.assertIsNone(api_server._category_error_paste(question), question)


class PasteValuesAndAskTests(unittest.TestCase):
    """E4-8 (e4:T14, T13, T10): a module label kept in the searched wording (#43780 and #48743 have the same message for other modules);
    the test-log search got the paste with the question glued on, or with its category in front, and called the exact problem weak."""

    def test_a_module_label_is_not_part_of_the_wording(self):
        with patch.object(api_server, "_known_exception_categories", return_value=set()):
            terms = api_server._paste_search_terms(
                "UnavailableAccelerator: Module alpakaGlobalProducer has the Alpaka backend set explicitly, but its accelerator is not available "
                "for the job because of the combination of the job configuration and accelerator availability on the machine.")
        self.assertTrue(terms["wording"].startswith("has the Alpaka backend set explicitly, but its accelerator is not available"), terms)
        with patch.object(api_server, "_known_exception_categories", return_value=set()):
            self.assertIn("EventSetup", api_server._paste_search_terms('No "BTLReadoutMapRcd" record found in the EventSetup.')["wording"])

    def test_the_test_log_search_gets_the_paste_without_the_question(self):
        from tests.test_test_log_answers import _report
        with patch.object(api_server, "search_test_log_problems_report", return_value=_report([])) as search, \
                patch.object(api_server, "test_log_coverage", return_value={"runs": 1}):
            api_server._test_log_problem_answer("Exception Message:\nExpected input collection size 36, got 6 -- known?")
        self.assertEqual(search.call_args.args[0], "Exception Message:\nExpected input collection size 36, got 6")

    def test_a_paste_with_its_category_in_front_is_contained_in_the_log(self):
        from tests.test_test_log_answers import SearchMatchLevelTests
        message = ("Module alpakaGlobalProducer has the Alpaka backend set explicitly, but its accelerator is not available for the job "
                   "because of the combination of the job configuration and accelerator availability on the machine.")
        hits = [dict(sid="same", category="UnavailableAccelerator", owners=["x"], text=message + " The following Alpaka backends are available")]
        found = SearchMatchLevelTests()._search("UnavailableAccelerator: " + message, hits=hits)
        self.assertEqual(found["same"][0], "contains")


class QuestionInFrontOfContextTests(unittest.TestCase):
    """E4-9 (e4:R04): "known issue? 1048.0 step3 dies with\n<Illegal parameter ... 'deleteAfterCopy' ...>" had no question found, so it was
    answered with 1048.0's history and "No imported CMSSW PR or Issue text contains workflow 1048.0", the error never searched."""
    Q = ("known issue? 1048.0 step3 dies with\nIllegal parameter found in configuration.  The parameter is named: 'deleteAfterCopy' You could be "
         "trying to use a parameter name that is not allowed for this plugin.\nValidating configuration of module: class=MEtoEDMConverter label='x'")

    def test_the_question_is_split_off_and_the_context_line_is_not_searched(self):
        paste, ask = api_server._split_paste_and_ask(self.Q)
        self.assertEqual(ask, "known issue?")
        self.assertTrue(api_server._REPORTED_ASK_RE.search(ask))
        with patch.object(api_server, "_known_exception_categories", return_value=set()):
            terms = api_server._paste_search_terms(paste)
        self.assertEqual(terms["exact"], "Illegal parameter found in configuration")
        self.assertIn("deleteAfterCopy", terms["names"])

    def test_a_question_line_without_a_report_word_stays_as_it_was(self):
        self.assertEqual(api_server._split_paste_and_ask("what is this? 537.0 in ASAN\n==1==ERROR: AddressSanitizer: SEGV")[1], "")


class OneNameIsNotTheSameProblemTests(unittest.TestCase):
    """E4-10 (e4:T09): the AI check called #30359 (one name, ~TProtoClass, in a comment) the same problem, above #45670 whose title is the error."""

    def section(self, rows_by_term, terms):
        from unittest.mock import MagicMock
        from tests.test_github_reports import row
        rows = {t: [row(n, **kw) for n, kw in items] for t, items in rows_by_term.items()}
        verdicts = {"Issue #30359": {"verdict": "same", "reason": "crash in TProtoClass"},
                    "Issue #45670": {"verdict": "same", "reason": "same bad free in 537.0"}}
        with patch("server.api_server._llm_client", new_callable=MagicMock), \
                patch("server.api_server.get_github_item_excerpts", side_effect=lambda items, _t: [{"id": f"Issue #{i['number']}", "title": "t"} for i in items]), \
                patch("server.api_server.search_github_mentions", side_effect=lambda term: rows.get(term, [])), \
                patch("server.api_server.REVIEW_MODEL", api_server.DEFAULT_MODEL), \
                patch("server.api_server.judge_github_matches", side_effect=lambda _c, _m, _p, batch: {c["id"]: verdicts[c["id"]] for c in batch}):
            return api_server._github_reports_section("paste", "", terms)

    def test_a_single_name_in_a_comment_is_possible_not_same(self):
        terms = {"exact": "attempting free on address which was not malloc()-ed", "wording": "", "names": ["~TProtoClass"], "category": None}
        answer = self.section({"attempting free on address which was not malloc()-ed": [(45670, dict(title_hit=True))],
                               "~TProtoClass": [(30359, dict(body=False, human=1))]}, terms)
        self.assertIn("Looks reported: 1 item(s)", answer)
        self.assertLess(answer.index("#45670"), answer.index("#30359"))
        self.assertIn("Possibly the same", answer)


class TagInsideAPathTests(unittest.TestCase):
    """E4-11 (e4:T04): the IB tag in a library path was answered as an asked IB: "No, this exception is not recorded in <tag>"."""

    def test_a_tag_in_a_path_is_not_an_asked_ib(self):
        paste = ("PluginLibraryLoadError: unable to load .../CMSSW_17_1_X_2026-11-08-0000/lib/el8_amd64_gcc13/pluginPhysicsToolsUtilAlgos_plugins.so "
                 "in runtestUtilAlgos")
        with patch.object(api_server, "get_driver") as driver:
            self.assertIsNone(api_server._pasted_error_in_ib_lead(paste, paste))
        driver.assert_not_called()

    def test_asked_tags_leave_out_paths(self):
        self.assertEqual(api_server._asked_ib_tags("in CMSSW_17_1_X_2026-11-09-2300: load .../CMSSW_17_1_X_2026-11-08-0000/lib/x.so"),
                         ["CMSSW_17_1_X_2026-11-09-2300"])

    def test_a_shortened_paste_is_looked_up_by_its_longest_piece(self):
        from unittest.mock import MagicMock
        session = MagicMock()
        session.run.return_value = []
        with patch.object(api_server, "get_driver") as driver, patch.object(api_server, "_ib_presence_text", return_value="x"):
            driver.return_value.session.return_value.__enter__.return_value = session
            api_server._pasted_error_in_ib_lead("is this in CMSSW_17_1_X_2026-11-09-2300? PluginLibraryLoadError: unable to load "
                                                ".../CMSSW_17_1_X_2026-11-08-0000/lib/el8_amd64_gcc13/plugin_plugins.so in runtestUtilAlgos",
                                                "PluginLibraryLoadError: unable to load .../CMSSW_17_1_X_2026-11-08-0000/lib/el8_amd64_gcc13/plugin_plugins.so in runtestUtilAlgos")
            self.assertFalse(api_server._pasted_error_in_ib_lead(
                "is this in CMSSW_17_1_X_2026-11-09-2300? PluginLibraryLoadError: unable to load .../lib/el8_amd64_gcc13/plugin_plugins.so in runtestUtilAlgos",
                "PluginLibraryLoadError: unable to load .../lib/el8_amd64_gcc13/plugin_plugins.so in runtestUtilAlgos"))  # nothing found: no "No"
        needles = {c.kwargs.get("needle") for c in session.run.call_args_list if "needle" in c.kwargs}
        needles.discard("/lib/el8_amd64_gcc13/plugin_plugins.so in runtestutilalgos")
        self.assertEqual(needles, {"/cmssw_17_1_x_2026-11-08-0000/lib/el8_amd64_gcc13/plugin_plugins.so in runtestutilalgos"})


if __name__ == "__main__":
    unittest.main()
