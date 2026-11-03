"""Chat answers from the imported raw IB test logs (LogProblem/TestLogRun):
a pasted error is looked up in the logs, which are the only source that sees
an exception inside a test the dashboard reports as PASSED."""
import unittest
from unittest.mock import patch

from server.api_server import ChatRequest, _pasted_names_github_answer, _test_log_problem_answer, chat
from tools import log_search_terms, parse_pasted_error, search_test_log_problems
def _report(problems, searched=(), failed=()):
    return {"problems": list(problems), "searched": list(searched), "failed": list(failed)}


SUBMITPV_REPORT = (
    "While checking the unit tests in CMSSW_20_1_ASAN_X_2026-10-05-2300, I noticed that the SubmitPVrbr and "
    "SubmitPVsplit tests are reported as successful even though the cmsRun jobs run by the tests fail.\n"
    "cmsRun PVValidation_testingOfflineGT_370604_cfg.py\n"
    "An exception of category 'DictionaryNotFound' occurred while\n"
    "   [0] Constructing the EventProcessor\n"
    "No data dictionary found for the following classes:\n  DCSRecord\n  L1GlobalTriggerReadoutRecord\n"
    "---> test SubmitPVrbr succeeded\n"
    "The tests are using Run 3 data files, which can no longer be read in 20_x after the io_v1 change (#50968)."
)

SUBMITPVRBR_PROBLEM = dict(
    sid="45770e30c3246374", kind="fatal_exception", category="DictionaryNotFound", origin="cmssw",
    summary="Fatal exception 'DictionaryNotFound': No data dictionary found for the following classes: DCSRecord",
    evidence="----- Begin Fatal Exception 12-Oct-2026 12:17:33 CEST---\nAn exception of category 'DictionaryNotFound'",
    example_tag="CMSSW_20_1_X_2026-10-12-1100", example_arch="el8_amd64_gcc14",
    example_path="/data/sdt/buildlogs/el8_amd64_gcc14/www/mon/20.1-mon-11/CMSSW_20_1_X_2026-10-12-1100/unitTestLogs.zip",
    example_member="unitTestLogs/Alignment/OfflineValidation/unitTest.log", example_line=7936,
    intentional_hint=None,
    tests={"Alignment/OfflineValidation/SubmitPVrbr": {"suite": "unit", "passed": 177, "failed": 0, "other": 0}},
    ib_tags=["CMSSW_20_1_X_2026-10-02-2300", "CMSSW_20_1_ASAN_X_2026-10-05-2300", "CMSSW_20_1_X_2026-10-12-1100"],
    ib_count=125, first_seen_tag="CMSSW_20_1_X_2026-10-02-2300", last_seen_tag="CMSSW_20_1_X_2026-10-12-1100",
    architectures=["el8_amd64_gcc14", "el9_amd64_gcc14", "el9_amd64_gcc15"], variants=[],
    places=["CMSSW_20_1_ASAN_X_2026-10-05-2300 el9_amd64_gcc14 (unit)",
            "CMSSW_20_1_ASAN_X_2026-10-05-2300 el9_amd64_gcc15 (unit)"],
    series_total=29, series_gone=[],
)

TRACKSTER_PASTE = (
    "----- Begin Fatal Exception 12-Oct-2026 15:45:18 CEST-----------------------\n"
    "An exception of category 'PluginNotFound' occurred while\n"
    "   [0] Running executable \"edmWriteConfigs\"\n"
    "   [1] Processing library /build/muz/del/CMSSW_20_1_X_2026-10-11-0000/tmp/el9_amd64_gcc14/src/RecoTICL/"
    "Interpretation/plugins/RecoTICLInterpretationPlugins/libRecoTICLInterpretationPlugins.so\n"
    "   [2] Writing cfi files using ParameterSetDescriptions for module of base type EDProducer with plugin name "
    "'TICLCandidateProducer'\n"
    "Exception Message:\n"
    "Unable to find plugin 'TracksterInferenceByPFN' because the category "
    "'PluginDescriptorTracksterInferenceAlgoFactory' has no known plugins\n"
    "----- End Fatal Exception -------------------------------------------------\n"
)


class ParsePastedErrorTests(unittest.TestCase):
    """The paste is read by the log scanner itself, not split into keywords."""

    def test_names_come_from_the_message_and_context_not_single_tokens(self):
        parsed = parse_pasted_error(TRACKSTER_PASTE)
        self.assertEqual(parsed["category"], "PluginNotFound")
        self.assertEqual(parsed["message_names"], ["TracksterInferenceByPFN", "PluginDescriptorTracksterInferenceAlgoFactory"])
        self.assertEqual(parsed["context_names"], ["TICLCandidateProducer"])
        self.assertEqual(parsed["executable"], "edmWriteConfigs")
        self.assertEqual(parsed["libraries"], ["RecoTICLInterpretationPlugins"])
        self.assertEqual(parsed["packages"], ["RecoTICL/Interpretation"])
        self.assertEqual(parsed["ib_tags"], ["CMSSW_20_1_X_2026-10-11-0000"])
        self.assertEqual(parsed["architectures"], ["el9_amd64_gcc14"])
        self.assertEqual(len(parsed["signature_ids"]), 1)

    def test_signature_ignores_the_time_and_missing_banner(self):
        # the same problem pasted from another run, or without the banner lines,
        # is the same stored signature -- an exact lookup, not a word search
        other_run = TRACKSTER_PASTE.replace("12-Oct-2026 15:45:18", "03-Oct-2026 01:02:03")
        no_banner = "\n".join(l for l in TRACKSTER_PASTE.splitlines() if "Fatal Exception" not in l)
        ids = parse_pasted_error(TRACKSTER_PASTE)["signature_ids"]
        self.assertEqual(parse_pasted_error(other_run)["signature_ids"], ids)
        self.assertEqual(parse_pasted_error(no_banner)["signature_ids"], ids)

    def test_partial_paste_quoted_names_are_specific_but_the_category_is_not(self):
        parsed = parse_pasted_error("An exception of category 'PluginNotFound' ... plugin 'TrackstersProducer'")
        self.assertEqual(parsed["message_names"], ["TrackstersProducer"])

    def test_module_and_producer_lines_are_context_names_but_paths_are_not(self):
        parsed = parse_pasted_error(
            "An exception of category 'Test' occurred while\n   [1] Running path 'p1Long'\n"
            "   [2] Calling method for module BusyWaitIntProducer/'busy1'\n")
        self.assertEqual(parsed["modules"], ["BusyWaitIntProducer/'busy1'"])
        self.assertEqual(parsed["context_names"], ["BusyWaitIntProducer", "busy1"])
        self.assertEqual(parsed["message_names"], [])


class _FakeSession:
    def __init__(self, signatures, hits, same_ib, name_hits=()):
        self.signatures, self.hits, self.same_ib, self.name_hits = signatures, hits, same_ib, list(name_hits)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, params=None):
        if "ProblemSignature" in cypher:
            return [{"sid": sid} for sid in params["ids"] if sid in self.signatures]
        if "logProblemNames" in cypher:
            return self.name_hits
        if "logProblemText" in cypher:
            return self.hits
        return [{"sid": sid} for sid in self.same_ib]


def _detail_rows(cypher, params=None):
    if "IN $sids" not in cypher or "UNWIND" in cypher:  # owner / module counts: none here
        return []
    return [dict(sid=sid, kind="fatal_exception", category="PluginNotFound", summary=sid, origin="cmssw",
                 evidence="", example_tag="CMSSW_20_1_X_2026-10-11-0000", example_arch="el9_amd64_gcc14",
                 example_path="", example_member="", example_line=1, places=[], owner_statuses=[],
                 intentional_hint=None, all_modules=[]) for sid in params["sids"]]


class SearchMatchLevelTests(unittest.TestCase):
    def _search(self, text, signatures=(), hits=(), same_ib=(), name_hits=()):
        with patch("tools.get_driver") as get_driver, patch("tools._run_query", side_effect=_detail_rows), \
                patch("tools._run_query_all", side_effect=_detail_rows):
            get_driver.return_value.session.return_value = _FakeSession(
                set(signatures), list(hits), list(same_ib), name_hits)
            return {r["sid"]: (r["match"], r["shared_terms"]) for r in search_test_log_problems(text)}

    def test_a_producer_name_matches_whole_in_the_names_index(self):
        name_hits = [dict(sid="prod", category="PluginNotFound", keys_text="",
                          names_text="ticlcandidateproducer ticlcandidate")]
        self.assertEqual(self._search(TRACKSTER_PASTE, name_hits=name_hits)["prod"],
                         ("related", ["TICLCandidateProducer"]))

    def test_same_producer_with_another_message_is_a_module_match(self):
        paste = "An exception of category 'Test' occurred while\n   [2] Calling method for module AddIntsProducer/'a'"
        name_hits = [dict(sid="other", category="Test", names_text="", keys_text="test|addintsproducer")]
        self.assertEqual(self._search(paste, name_hits=name_hits)["other"], ("module", ["Test|AddIntsProducer"]))

    def test_two_ordinary_identifiers_are_only_weak_and_hidden_behind_stronger(self):
        name_hits = [dict(sid="weak", category="Other", keys_text="", names_text="recoticl"),
                     dict(sid="strong", category="PluginNotFound", keys_text="", names_text="tracksterinferencebypfn")]
        found = self._search(TRACKSTER_PASTE, name_hits=name_hits)
        self.assertNotIn("weak", found)
        self.assertEqual(found["strong"][0], "related")
        only_weak = self._search(TRACKSTER_PASTE, name_hits=name_hits[:1])
        self.assertEqual(only_weak["weak"][0], "weak")
        # a same-category problem in the pasted IB is more useful than a weak match
        self.assertEqual(self._search(TRACKSTER_PASTE, name_hits=name_hits[:1], same_ib=["f27b"]), {"f27b": ("same_ib", [])})

    def test_framework_names_in_every_exception_are_not_search_keys(self):
        paste = ("An exception of category 'X' occurred while\n   [0] Constructing the EventProcessor\n"
                 "Exception Message:\nUnable to find plugin 'TrackstersProducer'")
        name_hits = [dict(sid="noise", category="Y", keys_text="", names_text="eventprocessor")]
        self.assertEqual(self._search(paste, name_hits=name_hits), {})

    def test_exact_signature_is_found_by_lookup(self):
        sid = parse_pasted_error(TRACKSTER_PASTE)["signature_ids"][0]
        self.assertEqual(self._search(TRACKSTER_PASTE, signatures=[sid])[sid][0], "exact")

    def test_sharing_only_the_category_is_never_a_match(self):
        hits = [dict(sid="other", category="PluginNotFound", owners=["x"],
                     text="An exception of category 'PluginNotFound' Unable to find plugin 'MkFitGeometryESProducer'")]
        self.assertEqual(self._search(TRACKSTER_PASTE, hits=hits), {})

    def test_a_shared_specific_name_is_related_and_says_which(self):
        hits = [dict(sid="rel", category="PluginNotFound", owners=["x"],
                     text="Unable to find plugin 'TracksterInferenceByPFN' in category 'X'")]
        self.assertEqual(self._search(TRACKSTER_PASTE, hits=hits)["rel"], ("related", ["TracksterInferenceByPFN"]))

    def test_same_category_in_the_pasted_ib_is_labelled_as_such_when_nothing_closer(self):
        self.assertEqual(self._search(TRACKSTER_PASTE, same_ib=["f27b"])["f27b"][0], "same_ib")

    def test_partial_paste_contained_word_for_word_ranks_first(self):
        part = "Unable to find plugin 'TrackstersProducer' in category 'CMS EDM Framework Module'."
        hits = [dict(sid="cfi", category="ModuleNotFoundError", owners=["x"],
                     text="No module named 'RecoTICL.PatternRecognition.trackstersProducer_cfi'"),
                dict(sid="full", category="PluginNotFound", owners=["x"],
                     text="An exception of category 'PluginNotFound' occurred while " + part)]
        found = self._search(part, hits=hits)
        self.assertEqual(found["full"][0], "contains")
        self.assertEqual(list(found)[0], "full")


class NotFoundAnswerTests(unittest.TestCase):
    @patch("server.api_server.test_log_coverage", return_value={"first": "2026-10-01-1100", "last": "2026-10-12-1100", "runs": 2046})
    @patch("server.api_server.search_test_log_problems_report")
    def test_exact_error_absent_is_said_with_where_it_was_raised_and_same_ib_problems_labelled(self, search, _cov):
        search.return_value = _report(
            [dict(SUBMITPVRBR_PROBLEM, match="same_ib", shared_terms=[],
                  tests={"wf:step2": {"suite": "relval", "passed": 0, "failed": 3, "other": 0}})],
            searched=[dict(by="signature", key="the whole error (category + message)", found=0),
                      dict(by="name", key="TracksterInferenceByPFN", found=0),
                      dict(by="producer", key="PluginNotFound|TICLCandidateProducer", found=0),
                      dict(by="same IB", key="PluginNotFound in CMSSW_20_1_X_2026-10-11-0000", found=1)])
        answer = _test_log_problem_answer(TRACKSTER_PASTE, "PluginNotFound")
        self.assertTrue(answer.startswith("**This exact error is not in the imported IB test logs**"))
        self.assertIn("`TracksterInferenceByPFN`", answer)
        self.assertIn("raised while running `edmWriteConfigs`, which runs during the build itself", answer)
        self.assertIn("**Other `PluginNotFound` problems in `CMSSW_20_1_X_2026-10-11-0000`**", answer)
        self.assertIn("a different error", answer)
        self.assertNotIn("Reported as PASSED", answer)  # nothing shown passed
        # every key searched is listed with its count, so a miss is visible
        self.assertIn("Searched the test logs by: whole error 0; `TracksterInferenceByPFN` 0; "
                      "producer `PluginNotFound|TICLCandidateProducer` 0; same IB 1", answer)

    @patch("server.api_server.test_log_coverage", return_value={"first": "2026-10-01-1100", "last": "2026-11-02-2300", "runs": 2046})
    @patch("server.api_server.search_test_log_problems_report")
    def test_same_ib_match_for_a_paste_without_message_is_not_called_a_different_error(self, search, _cov):
        # found 2026-11-03: the 3-line head of a Fatal Exception block (no "Exception Message:")
        # got its own stored problem labelled "a different error"
        search.return_value = _report(
            [dict(SUBMITPVRBR_PROBLEM, match="same_ib", shared_terms=[],
                  tests={"wf:step2": {"suite": "relval", "passed": 3, "failed": 0, "other": 0}})],
            searched=[dict(by="same IB", key="FileInPathError in CMSSW_13_2_X_2026-11-02-2300", found=1)])
        paste = ("In CMSSW_13_2_X_2026-11-02-2300 this test passed but its log shows:\n"
                 "----- Begin Fatal Exception 03-Nov-2026 03:34:42 CET-----------------------\n"
                 "An exception of category 'FileInPathError' occurred while\n"
                 "   [0] Constructing the EventProcessor")
        answer = _test_log_problem_answer(paste, "FileInPathError")
        self.assertNotIn("a different error", answer)
        self.assertIn("may or may not be the same error", answer)

    @patch("server.api_server.test_log_coverage", return_value={"first": "2026-10-01-1100", "last": "2026-10-12-1100", "runs": 2046})
    @patch("server.api_server.search_test_log_problems_report")
    def test_names_found_in_a_related_problem_are_not_listed_as_absent(self, search, _cov):
        search.return_value = _report(
            [dict(SUBMITPVRBR_PROBLEM, match="related", shared_terms=["TICLCandidateProducer"])],
            searched=[dict(by="name", key=n, found=f) for n, f in (
                ("TracksterInferenceByPFN", 0), ("PluginDescriptorTracksterInferenceAlgoFactory", 0),
                ("TICLCandidateProducer", 1))])
        answer = _test_log_problem_answer(TRACKSTER_PASTE, "PluginNotFound")
        self.assertIn("mentions `TracksterInferenceByPFN`, `PluginDescriptorTracksterInferenceAlgoFactory`.", answer)
        self.assertIn("**Found in the imported IB test logs** by the specific names in your paste", answer)
        self.assertIn("Shares with your paste: `TICLCandidateProducer`", answer)

    @patch("server.api_server.test_log_coverage", return_value={"first": "2026-10-01-1100", "last": "2026-10-12-1100", "runs": 2046})
    @patch("server.api_server.search_test_log_problems_report")
    def test_exact_message_from_another_module_says_so(self, search, _cov):
        search.return_value = _report([dict(SUBMITPVRBR_PROBLEM, match="exact", shared_terms=[],
                                    evidence="   [2] Calling method for module BusyWaitIntProducer/'busy1'")])
        answer = _test_log_problem_answer(
            "An exception of category 'Test' occurred while\n   [2] Calling method for module AddIntsProducer/'add'")
        self.assertIn("your paste's module `AddIntsProducer/'add'` is not in the stored example "
                      "(which is from `BusyWaitIntProducer/'busy1'`)", answer)

    @patch("server.api_server.test_log_coverage", return_value={"first": "2026-10-01-1100", "last": "2026-10-12-1100", "runs": 2046})
    @patch("server.api_server.search_test_log_problems_report")
    def test_same_producer_different_message_and_failed_channels_are_said(self, search, _cov):
        search.return_value = _report(
            [dict(SUBMITPVRBR_PROBLEM, match="module", shared_terms=["PluginNotFound|TICLCandidateProducer"],
                  modules=["TICLCandidateProducer/'ticlCandidate'"])], failed=["names"])
        answer = _test_log_problem_answer(TRACKSTER_PASTE, "PluginNotFound")
        self.assertIn("**Same producer, different message**", answer)
        self.assertIn("Same producer: `TICLCandidateProducer`", answer)
        self.assertIn("Raised in module(s): `TICLCandidateProducer/'ticlCandidate'`", answer)

    @patch("server.api_server.test_log_coverage", side_effect=RuntimeError("down"))
    @patch("server.api_server.search_test_log_problems_report", return_value=_report([]))
    def test_never_says_not_found_when_the_logs_could_not_be_searched(self, _search, _cov):
        self.assertIsNone(_test_log_problem_answer(TRACKSTER_PASTE, "PluginNotFound"))


class PastedNamesGithubTests(unittest.TestCase):
    @patch("server.api_server.search_issues_by_text", return_value={"issues": []})
    @patch("server.api_server.find_prs_for_error")
    def test_items_mentioning_more_specific_names_rank_first(self, find_prs, _issues):
        prs = {
            "TracksterInferenceByPFN": [dict(pr_number=52138, title="write the cfi", merged_at="2026-10-11T06:56:00Z"),
                                        dict(pr_number=47616, title="model loading", merged_at="2025-03-21T00:00:00Z")],
            "RecoTICLInterpretationPlugins": [dict(pr_number=52138, title="write the cfi", merged_at="2026-10-11T06:56:00Z")],
            "TICLCandidateProducer": [dict(pr_number=51919, title="TICL restructure", merged_at="2026-10-09T00:00:00Z")],
        }
        find_prs.side_effect = lambda name, limit: prs.get(name, [])
        answer = _pasted_names_github_answer(parse_pasted_error(TRACKSTER_PASTE))
        self.assertLess(answer.index("PR #52138"), answer.index("PR #47616"))
        self.assertLess(answer.index("PR #47616"), answer.index("PR #51919"))
        self.assertIn("mentions `TracksterInferenceByPFN`, `RecoTICLInterpretationPlugins`", answer)
        self.assertNotIn("edmWriteConfigs", answer)  # the executable says nothing about which problem

    @patch("server.api_server.find_prs_for_error", side_effect=RuntimeError("down"))
    def test_failed_lookups_are_not_reported_as_no_mention(self, _find):
        self.assertIsNone(_pasted_names_github_answer(parse_pasted_error(TRACKSTER_PASTE)))


class LogSearchTermTests(unittest.TestCase):
    def test_identifiers_are_kept_and_english_words_ib_tags_and_numbers_dropped(self):
        terms = log_search_terms(SUBMITPV_REPORT)
        for expected in ("SubmitPVrbr", "SubmitPVsplit", "DictionaryNotFound", "EventProcessor", "DCSRecord",
                         "L1GlobalTriggerReadoutRecord", "PVValidation_testingOfflineGT_370604_cfg.py"):
            self.assertIn(expected, terms)
        self.assertFalse(any("CMSSW_20_1_ASAN" in t for t in terms))
        self.assertNotIn("reported", terms)
        self.assertNotIn("50968", terms)


class TestLogProblemAnswerTests(unittest.TestCase):
    @patch("server.api_server.test_log_coverage", return_value={"first": "2026-10-01-1100"})
    @patch("server.api_server.search_test_log_problems_report", return_value=_report([SUBMITPVRBR_PROBLEM]))
    def test_names_the_passed_test_its_ibs_and_the_ib_asked_about(self, _search, _coverage):
        answer = _test_log_problem_answer(SUBMITPV_REPORT, "DictionaryNotFound")
        self.assertIn("`Alignment/OfflineValidation/SubmitPVrbr` (unit): **reported as PASSED in all 177 runs**", answer)
        self.assertIn("Seen in **125 IBs**", answer)
        self.assertIn("first in `CMSSW_20_1_X_2026-10-02-2300`", answer)
        self.assertIn("In `CMSSW_20_1_ASAN_X_2026-10-05-2300`: present (`el9_amd64_gcc14`, `el9_amd64_gcc15`)", answer)
        self.assertIn("Still present in the newest imported build", answer)
        self.assertIn("unitTestLogs/Alignment/OfflineValidation/unitTest.log` line 7936", answer)
        # what the logs can and cannot establish is said, not implied
        self.assertIn("not why", answer)
        self.assertIn("first seen since then", answer)

    @patch("server.api_server.test_log_coverage", return_value={"first": "2026-10-01-1100"})
    @patch("server.api_server.search_test_log_problems_report")
    def test_newer_builds_without_the_problem_are_reported_as_such(self, search, _coverage):
        search.return_value = _report([dict(SUBMITPVRBR_PROBLEM, series_gone=[
            {"key": "x", "newer_builds": 3, "newest": "CMSSW_20_1_X_2026-10-12-1100"}])])
        answer = _test_log_problem_answer("An exception of category 'DictionaryNotFound' occurred while")
        self.assertIn("Not seen in newer builds of 1 of 29 affected build series", answer)
        self.assertNotIn("Still present", answer)

    @patch("server.api_server.search_test_log_problems_report", return_value=_report([]))
    def test_no_match_returns_none_so_other_evidence_is_shown_alone(self, _search):
        self.assertIsNone(_test_log_problem_answer("An exception of category 'Nope' occurred while"))

    @patch("server.api_server.search_test_log_problems_report", side_effect=RuntimeError("no index"))
    def test_search_failure_degrades_to_none(self, _search):
        self.assertIsNone(_test_log_problem_answer("An exception of category 'X' occurred while"))


class PastedErrorRoutingTests(unittest.TestCase):
    @patch("server.api_server._referenced_items_note",
           return_value="Referenced in your text:\n- **PR #50968** - [20_0_X] Move stored types in io_v1 namespace")
    @patch("server.api_server._pasted_exception_search_answer",
           return_value=("Imported GitHub matches 1-5 of 96 for exception `...`:\n- **PR #51514** - x", []))
    @patch("server.api_server._test_log_problem_answer", return_value="**Found in the imported IB test logs** ...")
    @patch("server.api_server.get_driver")
    def test_log_evidence_comes_first_and_category_only_github_list_is_short_and_labelled(
        self, get_driver, log_answer, github_answer, referenced
    ):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.__iter__.return_value = iter([])
        session.run.return_value.single.return_value = None
        result = chat(ChatRequest(session_id="paste-submitpv", question=SUBMITPV_REPORT))
        log_answer.assert_called_once_with(SUBMITPV_REPORT, "DictionaryNotFound")
        github_answer.assert_called_once_with("An exception of category 'DictionaryNotFound' occurred while", limit=5)
        self.assertTrue(result.answer.startswith("**Found in the imported IB test logs**"))
        self.assertIn("PR #50968", result.answer)
        self.assertIn("GitHub items mentioning only this exception *category*", result.answer)
        self.assertLess(result.answer.index("Found in the imported"), result.answer.index("PR #51514"))

    @patch("server.api_server._pasted_exception_search_answer", return_value=("Imported GitHub matches ...", []))
    @patch("server.api_server._test_log_problem_answer", return_value=None)
    @patch("server.api_server.get_driver")
    def test_without_log_matches_the_github_answer_is_unchanged(self, get_driver, _log, github_answer):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.__iter__.return_value = iter([])
        session.run.return_value.single.return_value = None
        result = chat(ChatRequest(session_id="paste-plain",
                                  question="An exception of category 'LogicError' occurred while ."))
        github_answer.assert_called_once_with("An exception of category 'LogicError' occurred while", limit=20)
        self.assertNotIn("Broader context", result.answer)

    @patch("server.api_server._pasted_error_evidence_answer", return_value="Imported GitHub evidence mentioning `getNewCorners`:")
    @patch("server.api_server._test_log_problem_answer", return_value="**Found in the imported IB test logs** ASAN")
    @patch("server.api_server.get_driver")
    def test_sanitizer_report_without_exception_category_is_searched_in_the_logs(self, get_driver, log_answer, _gh):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.__iter__.return_value = iter([])
        session.run.return_value.single.return_value = None
        paste = ("==12345==ERROR: AddressSanitizer: global-buffer-overflow on address 0x7f\n"
                 "    #0 0x7f in HGCalGeometry::getNewCorners(...) HGCalGeometry.cc:612")
        result = chat(ChatRequest(session_id="paste-asan", question=paste))
        log_answer.assert_called_once_with(paste)
        self.assertIn("Found in the imported IB test logs", result.answer)
        self.assertIn("getNewCorners", result.answer)


    @patch("server.api_server._pasted_names_github_answer", return_value=None)
    @patch("server.api_server._pasted_error_evidence_answer", return_value=None)
    @patch("server.api_server._test_log_problem_answer", return_value="**Found in the imported IB test logs** zombiekiller")
    @patch("server.api_server.get_driver")
    def test_shell_report_of_a_killed_test_is_searched_in_the_logs(self, get_driver, log_answer, _gh, _names):
        # Found live 2026-10-15: this line was answered with an IB summary because the
        # paste gate only knew CMSSW/sanitizer/traceback shapes, not the shell's report.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.__iter__.return_value = iter([])
        session.run.return_value.single.return_value = None
        paste = ("In CMSSW_20_1_X_2026-10-12-1100 this test passed but its log shows:\n"
                 "/data/cmsbld/jenkins/workspace/ib-run-qa/CMSSW_20_1_X_2026-10-12-1100/src/FWCore/Services/test/"
                 "test_zombiekiller.sh: line 10: 2006489 Aborted                 (core dumped) ! ( cmsRun $F2 )")
        result = chat(ChatRequest(session_id="paste-shell-abort", question=paste))
        log_answer.assert_called_once_with(paste)
        self.assertIn("Found in the imported IB test logs", result.answer)


class PasteGateTests(unittest.TestCase):
    def test_shell_crash_lines_are_pastes_but_ordinary_questions_are_not(self):
        from server.api_server import _LOG_PROBLEM_PASTE_RE
        for text in ("x.sh: line 10: 2006489 Aborted                 (core dumped) ! ( cmsRun $F2 )",
                     "run.sh: line 7: 12345 Segmentation fault      cmsRun cfg.py",
                     "output\nSegmentation fault (core dumped)\n", "Aborted (core dumped)",
                     "Floating point exception (core dumped)"):
            self.assertTrue(_LOG_PROBLEM_PASTE_RE.search(text), text)
        for text in ("why do my tests get Killed sometimes", "line 5 is aborted",
                     "what does segmentation fault mean in unit tests", "segmentation fault",
                     "why was the job aborted (core dumped) yesterday"):
            self.assertFalse(_LOG_PROBLEM_PASTE_RE.search(text), text)


class PrStatusTests(unittest.TestCase):
    def test_merge_date_wins_then_the_export_state(self):
        from server.api_server import _pr_status
        self.assertEqual(_pr_status("2026-10-11T06:56:00Z", "open"), "merged 2026-10-11")
        self.assertEqual(_pr_status(None, "open"), "open, not merged (per GitHub export)")
        self.assertEqual(_pr_status(None, "closed"), "closed, no merge recorded")
        self.assertEqual(_pr_status(None, None), "not merged")

    @patch("server.api_server.search_issues_by_text", return_value={"issues": []})
    @patch("server.api_server.find_prs_for_error")
    def test_pasted_names_answer_says_a_pr_is_still_open(self, find_prs, _issues):
        find_prs.side_effect = lambda name, limit: [
            dict(pr_number=52200, title="fix PFN inference", merged_at=None, state="open")]
        answer = _pasted_names_github_answer(parse_pasted_error(TRACKSTER_PASTE))
        self.assertIn("PR #52200", answer)
        self.assertIn("open, not merged", answer)


if __name__ == "__main__":
    unittest.main()
