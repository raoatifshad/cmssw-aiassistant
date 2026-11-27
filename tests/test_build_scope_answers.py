"""Answers that used the wrong build, architecture or flavour (found in testing, 2026-11-06):
an OS word, a two-build comparison on one architecture, a flavour of a series, an older series."""
import unittest
from unittest.mock import MagicMock, patch

from server import api_server
from server.api_server import _basic_term_answer


def _driver_with(*results):
    """A get_driver() mock whose session.run returns each result in turn."""
    session = MagicMock()
    session.run.side_effect = list(results)
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    return driver, session


class ArchGlossaryTests(unittest.TestCase):
    def test_a_question_that_only_ends_in_an_os_name_is_not_a_definition(self):
        for question in ("what did change between the last two 20_1 IBs on el8?", "any failures in 16_0 on el9?",
                         "which relvals failed on el8"):
            self.assertIsNone(_basic_term_answer(question), question)

    def test_asking_what_the_os_name_is_still_works(self):
        for question in ("what is el9?", "what does fc42 mean?"):
            self.assertIn("Enterprise Linux", _basic_term_answer(question), question)


class CompareOnArchWordTests(unittest.TestCase):
    A, B = "CMSSW_20_1_X_2026-11-04-2300", "CMSSW_20_1_X_2026-11-05-1100"

    def _answer(self, word):
        all_archs = [{"arch": a} for a in ("el8_amd64_gcc14", "el9_aarch64_gcc14", "el9_amd64_gcc14")]
        in_both = [{"arch": "el8_amd64_gcc14"}, {"arch": "el9_amd64_gcc14"}]
        driver, _ = _driver_with(all_archs, in_both)
        seen = []
        real = api_server._ib_comparison_answer

        def compare(question):
            if len(seen) == 0 and not question.endswith("gcc14"):
                seen.append(question)
                return real(question)
            seen.append(question)
            return ("ok", [])
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "_ib_comparison_answer", compare):
            return compare(f"compare {self.A} and {self.B} on {word}"), seen

    def test_the_one_architecture_both_builds_have_is_used(self):
        answer, seen = self._answer("el9")
        self.assertEqual(answer[0], "ok")
        self.assertTrue(seen[-1].endswith("el9_amd64_gcc14"), seen)

    def test_a_family_neither_has_is_said(self):
        answer, _ = self._answer("gcc16")
        self.assertIn("no `gcc16` architecture in common", answer[0])

    def test_change_between_last_two_keeps_the_architecture(self):
        with patch.object(api_server, "_latest_ib_records", return_value=[{"tag": self.B}]), \
                patch.object(api_server, "_previous_primary_tag", return_value=self.A):
            self.assertEqual(api_server._with_series_change_tags("what did change between the last two 20_1 IBs on el8?"),
                             f"compare {self.A} and {self.B} on el8")

class SeriesFlavourAndArchTests(unittest.TestCase):
    def test_series_and_flavour_resolve_to_that_flavours_newest_build(self):
        with patch.object(api_server, "_newest_flavour_tag", return_value="CMSSW_20_1_G4ADEPT_X_2026-11-25-2300") as newest:
            out = api_server._with_variant_ib_context("what about 20_1 G4ADEPT unit tests?")
        self.assertTrue(out.endswith("CMSSW_20_1_G4ADEPT_X_2026-11-25-2300"))
        self.assertEqual(newest.call_args.args[1], (20, 1))

    def test_a_build_without_results_uses_the_newest_earlier_one(self):
        empty = [dict(variant="primary", arch="slc7_amd64_gcc10", relval_total=0, unittest_total=0)]
        real = api_server._failure_overview_answer
        with patch.object(api_server, "get_driver", return_value=_driver_with(empty)[0]), \
             patch.object(api_server, "_newest_older_tag_with_results", return_value="CMSSW_12_4_X_2026-11-22-1100"), \
             patch.object(api_server, "_failure_overview_answer", return_value=("OLDER ANSWER", [])):
            text, _ = real("CMSSW_12_4_X_2026-11-22-2300", None)
        self.assertIn("has no RelVal or unit-test results imported yet", text)
        self.assertIn("`CMSSW_12_4_X_2026-11-22-1100`", text)
        self.assertTrue(text.endswith("OLDER ANSWER"))

    def test_an_exception_with_a_series_goes_to_that_series_build_on_the_architecture(self):
        sid = "series-arch-exception"
        with patch.object(api_server, "_requested_release_series", return_value=(17, 0)), \
             patch.object(api_server, "_packages_named_loosely", return_value=False), \
             patch.object(api_server, "_newest_tag_for_arch_word", return_value=("CMSSW_17_0_X_2026-11-22-2300", ["el8_aarch64_gcc13"])), \
             patch.object(api_server, "_latest_ib_records", return_value=[{"tag": "CMSSW_17_0_X_2026-11-26-2300"}]), \
             patch.object(api_server, "_category_in_build_note", return_value="CATEGORY NOTE") as note:
            out = api_server._with_arch_missing_from_newest("is aarch64 still failing with ExternalLHEProducer on 17_0?", sid)
        self.assertIn("CMSSW_17_0_X_2026-11-22-2300 el8_aarch64_gcc13", out)
        self.assertEqual(note.call_args.args, ("ExternalLHEProducer", "CMSSW_17_0_X_2026-11-22-2300", "el8_aarch64_gcc13"))
        said = api_server._default_tag_notes.pop(sid)
        self.assertIn("has no `aarch64` build yet", said)
        self.assertIn("CATEGORY NOTE", said)


class CategoryInBuildNoteTests(unittest.TestCase):
    def note(self, n, known):
        driver, _ = _driver_with(MagicMock(single=MagicMock(return_value={"n": n, "known": known})))
        with patch.object(api_server, "get_driver", return_value=driver):
            return api_server._category_in_build_note("ExternalLHEProducer", "CMSSW_17_0_X_2026-11-22-2300", "el8_aarch64_gcc13")

    def test_all_known_errors(self):
        text = self.note(55, 55)
        self.assertIn("55 RelVal(s)", text)
        self.assertIn("all of them known errors", text)

    def test_some_known_errors(self):
        self.assertIn("3 of them known errors", self.note(5, 3))

    def test_none_known(self):
        text = self.note(5, 0)
        self.assertIn("5 RelVal(s)", text)
        self.assertNotIn("known", text)

    def test_not_in_the_build(self):
        self.assertIn("no RelVal in", self.note(0, 0))


class AllArchComparisonLabelTests(unittest.TestCase):
    """A section compared against its own previous build is labelled with that build (found 2026-11-27:
    el9_aarch64 compared against 11-24-2300 under a 11-26-1100 header)."""
    BEFORE, AFTER, OWN = "CMSSW_20_1_X_2026-11-26-1100", "CMSSW_20_1_X_2026-11-26-2300", "CMSSW_20_1_X_2026-11-24-2300"

    def row(self, tag, failing):
        return {"tag": tag, "primary_tag": tag, "arch": "el9_aarch64_gcc14", "variant": "primary", "build_warnings": 0,
                "ignored_warnings": 0, "clang_locations": 0, "clang_occurrences": 0, "relval_total": 956,
                "relval_failed": len(failing), "relval_errors": 5 + len(failing), "relval_warnings": 2, "unit_test_total": 381,
                "unit_test_failed": 0, "unit_test_errors": 0,
                "failing_relvals": [{"workflow_id": w, "name": "W", "errors": 1, "warnings": 1, "exitcode": 1} for w in failing]}

    def test_header_and_status_name_the_build_really_compared(self):
        after = dict(self.row(self.AFTER, ["1.0"]), relval_warnings=9)   # a change, so a table is drawn
        driver, _ = _driver_with([after])
        with patch.object(api_server, "get_driver", return_value=driver), \
             patch.object(api_server, "get_exit_code_names", return_value={}), \
             patch.object(api_server, "_variant_previous_tag", return_value=self.OWN), \
             patch.object(api_server, "_single_ib_comparison_row", return_value=self.row(self.OWN, ["1.0"])):
            answer = api_server._all_arch_ib_comparison_answer(f"compare {self.BEFORE} and {self.AFTER} all architectures")
        self.assertIn(f"compared against its own previous build `{self.OWN}`", answer)
        self.assertIn(f"| Metric | {self.OWN} | {self.AFTER} | Delta |", answer)
        self.assertNotIn(f"| Metric | {self.BEFORE} |", answer)
        self.assertIn(f"already failing in `{self.OWN}`", answer)


class ArchitecturesOfBuildTests(unittest.TestCase):
    """ "how many el9 in latest IB?" (found 2026-11-27: only el9_amd64 named, no count)."""
    TAG = "CMSSW_20_1_X_2026-11-26-2300"
    ROWS = [{"tag": TAG, "arch": "el8_amd64_gcc14"}, {"tag": TAG, "arch": "el9_amd64_gcc14"},
            {"tag": TAG, "arch": "el9_aarch64_gcc14"}, {"tag": "CMSSW_20_1_ROOT6_X_2026-11-26-2300", "arch": "el9_aarch64_gcc14"}]

    def ask(self, question, rows=None):
        driver, _ = _driver_with(rows if rows is not None else self.ROWS)
        with patch.object(api_server, "get_driver", return_value=driver), \
             patch.object(api_server, "_latest_ib_records", return_value=[{"tag": self.TAG}]):
            return api_server._architectures_of_build_answer(question)

    def test_counts_one_family(self):
        answer = self.ask("how many el9 in latest IB?")
        self.assertIn("**2 el9 architecture(s)**", answer)
        self.assertIn("`el9_aarch64_gcc14`: primary, ROOT6", answer)
        self.assertIn("also `el8_amd64_gcc14`", answer)

    def test_all_architectures(self):
        self.assertIn("**3 architecture(s)**", self.ask("how many architectures in the latest IB?"))

    def test_a_family_it_lacks_names_the_newest_build_with_it(self):
        with patch.object(api_server, "_newest_build_with_arch", return_value=("CMSSW_20_1_X_2026-11-24-2300", ["el10_amd64_gcc15"])):
            answer = self.ask("how many el10 in latest IB?")
        self.assertIn("no el10 architecture", answer)
        self.assertIn("`CMSSW_20_1_X_2026-11-24-2300`", answer)

    def test_other_questions_are_left_alone(self):
        for question in ("what is the latest el9 IB?", "how many relvals failed in the latest IB?", "how many el9 builds failed?"):
            self.assertIsNone(self.ask(question), question)
