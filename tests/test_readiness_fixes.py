"""Fixes from the 2026-10-15 readiness pass (held-out questions read by hand before
handing the chatbot to colleagues). Each test names the live answer that was wrong."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
import ib_test_log_scanner as scanner  # noqa: E402
from server.api_server import (  # noqa: E402
    _arch_unavailable_answer,
    _artifact_search_answer,
    _build_status_answer,
    _clang_warning_answer,
    _format_raw_evidence,
    _identifiers_grounded,
    _last_capability_contexts,
    _latest_ib_contexts,
    _persistent_failures_answer,
    _with_arch_family_context,
)
from tools import log_search_terms  # noqa: E402

TAG = "CMSSW_20_1_X_2026-10-12-1100"


def _session(get_driver):
    return get_driver.return_value.session.return_value.__enter__.return_value


class IdentifierGroundingTests(unittest.TestCase):
    EVIDENCE = [{"tool": "get_failing_unittests", "arguments": {"ib_tag": TAG},
                 "result": [{"arch": "el9_amd64_gcc14", "name": "Alignment/OfflineValidation"}]}]

    def test_invented_tag_and_architecture_are_rejected(self):
        # live: "is SubmitPVrbr really passing?" -> CMSSW_20_1_X_2026-10-15-1230 on x86_64
        answer = "No unit test failed for `CMSSW_20_1_X_2026-10-15-1230` on the `x86_64` architecture."
        self.assertFalse(_identifiers_grounded(answer, self.EVIDENCE))
        self.assertFalse(_identifiers_grounded(f"Nothing failed in `{TAG}` on `x86_64`.", self.EVIDENCE))

    def test_a_tag_the_model_passed_as_a_tool_argument_is_not_evidence(self):
        # live (second run): the model called a tool with the invented tag and arch, so
        # they were "in the evidence" -- only tool results count
        evidence = [{"tool": "get_failing_unittests",
                     "arguments": {"ib_tag": "CMSSW_20_1_X_2026-10-15-1200", "arch": "x86_64"}, "result": []}]
        self.assertFalse(_identifiers_grounded(
            "Nothing failed in `CMSSW_20_1_X_2026-10-15-1200` on `x86_64`.", evidence))

    def test_absence_wording_from_the_live_answer_is_an_absence_claim(self):
        from server.api_server import _llm_answer_is_absence_claim
        self.assertTrue(_llm_answer_is_absence_claim(
            "No such `TestLogRun` nodes exist for the IB. There are no tests that passed yet reported errors."))
        self.assertFalse(_llm_answer_is_absence_claim("Two tests passed with errors in their logs."))

    def test_tags_and_architectures_from_the_evidence_or_question_pass(self):
        evidence = [dict(self.EVIDENCE[0], result=[{"ib_tag": TAG, "arch": "el9_amd64_gcc14"}])]
        self.assertTrue(_identifiers_grounded(f"`{TAG}` on `el9_amd64_gcc14`: one failure.", evidence))
        # the tag the user asked about, even when the tool result doesn't repeat it
        self.assertTrue(_identifiers_grounded(f"`{TAG}` on `el9_amd64_gcc14`: one failure.", self.EVIDENCE,
                                              question=f"unit tests in {TAG}"))
        self.assertTrue(_identifiers_grounded("Nothing about any build here.", self.EVIDENCE))


class RawEvidenceFallbackTests(unittest.TestCase):
    def test_database_overview_alone_is_not_shown_as_an_answer(self):
        # live: "anything I should worry about as shifter right now?" -> a page of node counts
        text = _format_raw_evidence([{"tool": "get_database_overview", "arguments": {},
                                      "result": {"node_counts": {"IB": 2622}}}])
        self.assertNotIn("node_counts", text)
        self.assertIn("couldn't find evidence specific to this question", text)

    def test_long_evidence_is_truncated(self):
        text = _format_raw_evidence([{"tool": "get_failing_relvals", "arguments": {}, "result": ["x" * 9000]}])
        self.assertIn("(truncated)", text)
        self.assertLess(len(text), 6000)


class StatusOfTagTests(unittest.TestCase):
    @patch("server.api_server.get_driver")
    def test_ib_tag_alone_is_not_a_github_search_term(self, get_driver):
        # live: "status of <tag>" -> "No imported Issues or PRs match `<tag>`"
        self.assertIsNone(_artifact_search_answer(f"status of {TAG}"))
        get_driver.assert_not_called()


class BuildStatusTests(unittest.TestCase):
    @patch("server.api_server.get_driver")
    def test_build_errors_for_a_tag_lists_every_build(self, get_driver):
        # live: "build errors in <tag>" -> "My lookups for this came back empty"
        _session(get_driver).run.return_value = [
            {"tag": TAG, "variant": "primary", "arch": "el9_amd64_gcc14", "status": "error", "comp_warnings": 0},
            {"tag": TAG, "variant": "primary", "arch": "el8_amd64_gcc14", "status": "passed", "comp_warnings": 0},
        ]
        answer = _build_status_answer(f"build errors in {TAG}")
        self.assertIn("**1 of 2** IB/architecture build(s) failed", answer)
        self.assertIn("| primary | `el9_amd64_gcc14` | `error` | 0 |", answer)

    @patch("server.api_server.get_driver")
    def test_uses_the_session_tag_and_says_when_nothing_is_imported(self, get_driver):
        _session(get_driver).run.return_value = []
        self.assertIn("No CMS build-status record", _build_status_answer("did the build fail?", TAG))

    def test_clang_and_unrelated_questions_are_left_alone(self):
        self.assertIsNone(_build_status_answer(f"clang build errors in {TAG}"))
        self.assertIsNone(_build_status_answer(f"relval errors in {TAG}"))


class ClangNotParsedTests(unittest.TestCase):
    @patch("server.api_server.get_driver")
    def test_unparsed_ib_is_unknown_not_zero(self, get_driver):
        # live: "clang warnings <tag>" -> a table of zeros for an IB whose logs were never parsed
        session = _session(get_driver)
        newest = MagicMock()
        newest.single.return_value = {"tag": "CMSSW_20_1_X_2026-08-12-2300"}
        session.run.side_effect = [[{"arch": "el9_amd64_gcc14", "is_default": True, "locations": 0,
                                     "occurrences": 0, "parsed": False}], newest]
        answer = _clang_warning_answer(f"clang warnings for {TAG}")
        self.assertIn("unknown (not zero)", answer)
        self.assertIn("CMSSW_20_1_X_2026-08-12-2300", answer)


class ScopeFollowUpTests(unittest.TestCase):
    def setUp(self):
        self.sid = "readiness-scope"
        _latest_ib_contexts[self.sid] = TAG
        _last_capability_contexts[self.sid] = "relval"

    def tearDown(self):
        _latest_ib_contexts.pop(self.sid, None)
        _last_capability_contexts.pop(self.sid, None)

    @patch("server.api_server.get_driver")
    def test_os_family_resolves_to_the_one_matching_architecture(self, get_driver):
        # live: "only el9" after a unit-test answer -> "My lookups for this came back empty"
        _session(get_driver).run.return_value = [{"arch": "el9_amd64_gcc14"}, {"arch": "el8_amd64_gcc14"}]
        self.assertEqual(_with_arch_family_context("only el9", self.sid), "only el9 el9_amd64_gcc14")

    @patch("server.api_server.get_driver")
    def test_family_the_ib_does_not_have_is_said_with_what_it_has(self, get_driver):
        # live: "what about aarch64" -> a list of PRs/Issues mentioning "aarch64"
        _session(get_driver).run.return_value = [{"arch": "el9_amd64_gcc14"}, {"arch": "el8_amd64_gcc14"}]
        answer = _arch_unavailable_answer("what about aarch64", self.sid)
        self.assertIn(f"`{TAG}` has no imported `aarch64` build", answer)
        self.assertIn("`el8_amd64_gcc14`, `el9_amd64_gcc14`", answer)

    @patch("server.api_server.get_driver")
    def test_not_a_scope_follow_up_is_left_alone(self, get_driver):
        _session(get_driver).run.return_value = [{"arch": "el9_amd64_gcc14"}]
        self.assertIsNone(_arch_unavailable_answer("why is workflow 512.0 failing on aarch64", self.sid))
        _last_capability_contexts.pop(self.sid)
        self.assertIsNone(_arch_unavailable_answer("what about aarch64", self.sid))


class NewSincePreviousTests(unittest.TestCase):
    def test_new_counts_only_when_the_question_is_about_failures(self):
        _latest_ib_contexts["readiness-new"] = TAG
        try:
            with patch("server.api_server.get_driver") as get_driver:
                _session(get_driver).run.return_value = []
                self.assertIsNone(_persistent_failures_answer("is there a new IB since yesterday", "readiness-new"))
                _persistent_failures_answer("which of these are new compared to the previous IB?", "readiness-new")
                get_driver.assert_called()
        finally:
            _latest_ib_contexts.pop("readiness-new", None)


class HexAddressNameTests(unittest.TestCase):
    PASTE = ("==4242==ERROR: AddressSanitizer: heap-use-after-free on address 0x50d0005dfb30\n"
             "    #0 0x14c9588 in edm::storage::File::syswrite libFWStorageStorageFactory.so")

    def test_addresses_are_not_names_or_search_terms(self):
        # live: the answer said the paste "shares" x50d0005dfb30 with a stored problem
        names = scanner.problem_details(self.PASTE.splitlines())["names"]
        self.assertIn("edm::storage::File::syswrite", names)
        self.assertFalse([n for n in names if n.startswith("x") and any(c.isdigit() for c in n)])
        self.assertNotIn("x50d0005dfb30", log_search_terms(self.PASTE))


if __name__ == "__main__":
    unittest.main()
