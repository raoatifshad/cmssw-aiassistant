"""Wrong answers found by item R2 of the weekend test (2026-11-11): management-style questions, round 2."""
import unittest
from unittest.mock import patch

from server import api_server


def _rec(tag, arch, failed, relvals=900, variant="primary"):
    return {"tag": tag, "arch": arch, "variant": variant, "build": "passed", "relvals": relvals, "failed": failed, "units": 300, "unit_failed": []}


class _FakeDriver:
    def __init__(self, rows):
        self.rows = rows
    def session(self):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def run(self, *a, **k):
        return self.rows
    def close(self):
        pass


class ArchVersusTests(unittest.TestCase):
    """R2-1: "slc7 vs el8 in the latest 14_1 IB" and "gcc15 vs gcc14 unit tests" were answered "Do you want a comparison of these
    Integration Builds, or results for one specific build?"; the newest build with nothing on one side is replaced and said."""

    NEW, OLD = "CMSSW_14_1_X_2026-11-10-2300", "CMSSW_14_1_X_2026-11-10-1100"

    def _records(self, tag, *a, **k):
        if tag == self.NEW:
            return [_rec(tag, "el8_amd64_gcc12", ["4.17"])]
        return [_rec(tag, "el8_amd64_gcc12", ["4.17"]), _rec(tag, "slc7_amd64_gcc12", ["4.17", "4.29"])]

    def test_vs_between_two_architectures_is_a_comparison(self):
        with patch.object(api_server, "_latest_ib_records", return_value=[{"tag": self.OLD}]), \
                patch.object(api_server, "_ib_build_records", side_effect=self._records):
            answer = api_server._arch_compare_answer("slc7 vs el8 in the latest 14_1 IB")
        self.assertIn("| Failing on both | `4.17` |", answer)
        self.assertIn("| Only on `slc7` | `4.29` |", answer)
        self.assertNotIn("Do you want", answer)

    def test_newest_build_without_one_side_is_replaced_and_said(self):
        with patch.object(api_server, "_latest_ib_records", return_value=[{"tag": self.NEW}]), \
                patch.object(api_server, "_ib_build_records", side_effect=self._records), \
                patch.object(api_server, "get_driver", return_value=_FakeDriver([{"tag": self.NEW}, {"tag": self.OLD}, {"tag": "CMSSW_14_1_X_2026-11-09-2300"}])):
            answer = api_server._arch_compare_answer("slc7 vs el8 in the latest 14_1 IB")
        self.assertIn(f"`{self.NEW}` has no results on both yet", answer)
        self.assertIn(f"in `{self.OLD}`", answer)
        self.assertIn("4.29", answer)

    def test_versus_without_the_two_codes_is_not_taken(self):
        self.assertIsNone(api_server._arch_compare_answer("20_1 vs 20_0 unit tests in the latest build"))
        self.assertIsNone(api_server._arch_compare_answer("el8 vs el9 in CMSSW_14_1_X_2026-11-10-1100"))


class TwoBuildsWithoutArchitectureTests(unittest.TestCase):
    """R2-2: "compare <tag A> and <tag B>" asked "Which architecture should I compare?" with every architecture available."""

    def test_two_builds_are_compared_on_all_architectures(self):
        class Driver:
            def session(self):
                return self
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def run(self, *a, **k):
                return [{"arch": "el8_amd64_gcc14"}, {"arch": "el9_amd64_gcc14"}]
            def close(self):
                pass
        seen = []
        def every(question):
            seen.append(question)
            return "ALL ARCHITECTURES TABLE"
        question = "compare CMSSW_20_1_X_2026-11-09-2300 and CMSSW_20_1_X_2026-11-10-1100"
        with patch.object(api_server, "get_driver", return_value=Driver()), patch.object(api_server, "_all_arch_ib_comparison_answer", side_effect=every):
            answer, actions = api_server._ib_comparison_answer(question)
        self.assertEqual(answer, "ALL ARCHITECTURES TABLE")
        self.assertIn("all architectures", seen[0])
        self.assertEqual(actions, [])


if __name__ == "__main__":
    unittest.main()


class BareStampAndPreviousBuildTests(unittest.TestCase):
    """R2-3: "relval failures in the IB before 11-10-1100" was answered for 11-10-1100 itself, and "so what's different in 11-10-1100?"
    in a 16_0 conversation for the newest series' build."""

    def test_bare_stamp_takes_the_series_of_the_conversation(self):
        with patch.dict(api_server._latest_ib_contexts, {"r2-3": "CMSSW_16_0_X_2026-11-10-1100"}):
            out = api_server._bare_stamp_in_session_series("so what's different in 11-10-1100?", "r2-3")
        self.assertIn("CMSSW_16_0_X_", out)
        self.assertTrue(out.endswith("-11-10-1100?"))

    def test_series_in_the_question_or_no_session_keeps_it(self):
        with patch.dict(api_server._latest_ib_contexts, {"r2-3": "CMSSW_16_0_X_2026-11-10-1100"}):
            self.assertEqual(api_server._bare_stamp_in_session_series("what about 17_1 11-09-2300?", "r2-3"), "what about 17_1 11-09-2300?")
            self.assertEqual(api_server._bare_stamp_in_session_series("is CMSSW_20_1_X_2026-11-10-1100 ok?", "r2-3"), "is CMSSW_20_1_X_2026-11-10-1100 ok?")
        self.assertEqual(api_server._bare_stamp_in_session_series("different in 11-10-1100?", "no-such-session"), "different in 11-10-1100?")

    def test_the_ib_before_a_tag_is_the_previous_build(self):
        with patch.object(api_server, "neighbour_build", return_value="CMSSW_16_0_X_2026-11-08-2300") as nb:
            out = api_server._resolve_relative_builds("how many relval failures did 16_0 el8 have in the IB before CMSSW_16_0_X_2026-11-10-1100?")
        self.assertIn("`CMSSW_16_0_X_2026-11-08-2300`", out)
        self.assertNotIn("before CMSSW", out)
        self.assertEqual(nb.call_args[0][:2], ("CMSSW_16_0_X_2026-11-10-1100", "before"))


class AllFlavoursCountTests(unittest.TestCase):
    """R2-4: "how many relval workflows failed in total across all 20_1 flavours in the 11-08-2300 build?" counted the primary build only (37 of 193)."""

    def _answer(self, question):
        seen = {}
        def records(tag, include_sub_ibs):
            seen["sub"] = include_sub_ibs
            return [{"tag": tag, "arch": "el9_amd64_gcc14", "variant": None, "relval_total": 10, "failed_relvals": 1, "passed_relvals": 9,
                     "known_error_relvals": 0, "das_error_relvals": 0, "failed_workflows": []}]
        with patch.object(api_server, "_relval_scope_records", side_effect=records), \
                patch.object(api_server, "_relval_scope_summary", side_effect=lambda recs, heading, variants=None: heading), \
                patch.object(api_server, "_sub_ib_variant_names", return_value=[]):
            return api_server._relval_failure_count_answer(question, "r2-4"), seen["sub"]

    def test_all_flavours_includes_the_sub_ibs(self):
        answer, sub = self._answer("how many relval workflows failed in total across all 20_1 flavours in CMSSW_20_1_X_2026-11-08-2300?")
        self.assertTrue(sub)
        self.assertIn("and its sub-IBs", answer)

    def test_plain_count_stays_on_the_primary_build(self):
        answer, sub = self._answer("how many relval workflows failed in CMSSW_20_1_X_2026-11-08-2300?")
        self.assertFalse(sub)
        self.assertNotIn("sub-IBs", answer)


class InventedArchitectureTests(unittest.TestCase):
    """R2-5: "and on gcc16?" after an el10 question became "...on el10_amd64_gcc16" (no such architecture): "no data for el10_amd64_gcc16"."""

    KNOWN = [{"arch": "el10_amd64_gcc15"}, {"arch": "el9_amd64_gcc16"}]

    def test_made_up_architecture_goes_back_to_the_compiler(self):
        with patch.object(api_server, "get_driver", return_value=_FakeDriver(self.KNOWN)):
            out = api_server._keeps_real_architecture("and on gcc16?", "how is 20_1 doing on el10_amd64_gcc16?")
        self.assertEqual(out, "how is 20_1 doing on gcc16?")

    def test_real_or_named_architecture_is_kept(self):
        with patch.object(api_server, "get_driver", return_value=_FakeDriver(self.KNOWN)):
            self.assertEqual(api_server._keeps_real_architecture("and on gcc16?", "how is 20_1 on el9_amd64_gcc16?"), "how is 20_1 on el9_amd64_gcc16?")
            self.assertEqual(api_server._keeps_real_architecture("and on el10_amd64_gcc16?", "on el10_amd64_gcc16?"), "on el10_amd64_gcc16?")
        with patch.object(api_server, "get_driver", side_effect=AssertionError("no lookup needed")):
            self.assertEqual(api_server._keeps_real_architecture("and gcc16?", "how is 20_1 doing on gcc16?"), "how is 20_1 doing on gcc16?")


class ExceptionSummaryTests(unittest.TestCase):
    """R2-6: "summary of exceptions in 20_1 this week" went to the shift summary (RelVal and unit-test diffs, no exception in it)."""

    def test_summary_of_exceptions_is_the_exception_ranking(self):
        for text in ("summary of exceptions in 20_1 this week", "give me an overview of the exceptions in 16_0 this week", "exceptions summary for 20_1"):
            self.assertTrue(api_server._EXCEPTION_RANKING_RE.search(text), text)
        for text in ("summary of the failures in 20_1 this week", "shift summary this week", "what is an exception?"):
            self.assertFalse(api_server._EXCEPTION_RANKING_RE.search(text), text)


class TiedHealthiestTests(unittest.TestCase):
    """R2-8: "which architecture of 20_1 is the healthiest?" named el8 as the one with the fewest failures; el9_amd64_gcc15 had the same counts."""

    def test_a_tie_is_said(self):
        records = [_rec("T", "el8_amd64_gcc14", ["1.0", "2.0"]), _rec("T", "el9_amd64_gcc15", ["1.0", "2.0"]), _rec("T", "el9_amd64_gcc14", ["1.0", "2.0", "3.0"])]
        with patch.object(api_server, "_latest_or_named_tag", return_value="T"), patch.object(api_server, "_ib_build_records", return_value=records):
            answer = api_server._failures_by_dimension_answer("which architecture of 20_1 is the healthiest?")
        self.assertIn("tie for the fewest", answer)
        self.assertIn("**`el8_amd64_gcc14`**", answer)
        self.assertIn("**`el9_amd64_gcc15`**", answer)

    def test_a_single_winner_is_named_as_before(self):
        records = [_rec("T", "el8_amd64_gcc14", ["1.0"]), _rec("T", "el9_amd64_gcc15", ["1.0", "2.0"])]
        with patch.object(api_server, "_latest_or_named_tag", return_value="T"), patch.object(api_server, "_ib_build_records", return_value=records):
            answer = api_server._failures_by_dimension_answer("which architecture of 20_1 is the healthiest?")
        self.assertIn("**`el8_amd64_gcc14`** has the fewest", answer)
