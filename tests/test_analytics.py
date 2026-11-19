"""graph/analytics.py: what the raw queries the LLM used to write got wrong. Fake rows, no database."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from graph import analytics
from graph.tools import TOOLS, list_ib_tags
from server.agent import _function_to_openai_schema
from server.api_server import _existing_builds_denied, _flavours_of_build_answer, _resolve_relative_builds, _PER_FLAVOUR_RE, _architectures_answer, _identifiers_grounded, _relval_by_exit_code_answer, _keeps_known_error_condition, _pr_opener_sentence, _WORD_SEARCH_DEAD_END_RE, _DATE_RANGE_RE, _with_partial_build_note, _known_question_invents, _scope_mismatch, _without_invented_names, _clang_package_in_tag_answer, _named_month_window


def relval(tag, arch, workflow, status="FAILED", known=False, code=256):
    return dict(tag=tag, arch=arch, workflow=workflow, name=f"wf{workflow}", status=status, exit_code=code, known=known)


class BuildsAreFactsTests(unittest.TestCase):
    TAGS = [dict(tag=f"CMSSW_20_1_X_2026-11-0{d}") for d in ("2-2300", "3-1100", "3-2300")]

    def test_neighbours_come_from_the_data(self):
        with patch.object(analytics, "_rows", return_value=self.TAGS):
            self.assertEqual(analytics.neighbour_build("CMSSW_20_1_X_2026-11-02-2300", "after"), "CMSSW_20_1_X_2026-11-03-1100")
            self.assertEqual(analytics.neighbour_build("CMSSW_20_1_X_2026-11-03-1100", "before"), "CMSSW_20_1_X_2026-11-02-2300")
            self.assertIsNone(analytics.neighbour_build("CMSSW_20_1_X_2026-11-03-2300", "after"))

    def test_a_relative_build_is_written_out(self):
        with patch("server.api_server.neighbour_build", return_value="CMSSW_20_1_X_2026-11-03-1100") as look:
            self.assertEqual(_resolve_relative_builds("what failed in the build after CMSSW_20_1_X_2026-11-02-2300 on el9_amd64_gcc14?"), "what failed in `CMSSW_20_1_X_2026-11-03-1100` on el9_amd64_gcc14?")
        look.assert_called_once_with("CMSSW_20_1_X_2026-11-02-2300", "after", "el9_amd64_gcc14")
        self.assertEqual(_resolve_relative_builds("what failed?"), "what failed?")

    def test_a_build_that_exists_cant_be_called_missing(self):
        answer = "In the next primary build after CMSSW_20_1_X_2026-11-02-2300, the 11-03-1100 primary slot isn't in the graph."
        with patch("server.api_server._identifiers_exist", return_value=True):
            self.assertEqual(_existing_builds_denied(answer), ["CMSSW_20_1_X_2026-11-03-1100"])
        with patch("server.api_server._identifiers_exist", return_value=False):
            self.assertEqual(_existing_builds_denied(answer), [])

    def test_a_statement_about_one_architecture_can_be_true(self):
        with patch("server.api_server._identifiers_exist", return_value=True) as check:
            self.assertEqual(_existing_builds_denied("CMSSW_20_1_X_2026-11-03-1100 isn't imported on el9_aarch64_gcc14."), [])
        check.assert_not_called()

    def test_flavours_of_a_build(self):
        records = [dict(variant="primary", arch="el8"), dict(variant="primary", arch="el9"), dict(variant="ROOT6", arch="el9"), dict(variant="CLANG", arch="el9")]
        with patch("server.api_server._dated_primary_question", return_value="how many CMSSW_20_1_X_2026-11-03-2300"), patch("server.api_server._latest_or_named_tag", return_value="CMSSW_20_1_X_2026-11-03-2300"), patch("server.api_server._ib_build_records", return_value=records):
            answer = _flavours_of_build_answer("how many sub-IB flavours does the 2026-11-03-2300 20_1 build have?")
        self.assertIn("**2 sub-IB flavour(s)**", answer)
        self.assertIn("`CLANG` (`el9`), `ROOT6` (`el9`)", answer)
        self.assertIsNone(_flavours_of_build_answer("which flavour failed most?"))

    def test_sub_ibs_in_a_count_are_named(self):
        rows = [relval("CMSSW_20_1_X_2026-11-03-2300", "el9", "1.0"), relval("CMSSW_20_1_ASAN_X_2026-11-03-2300", "el9", "2.0")]
        with patch.object(analytics, "_rows", side_effect=[rows, []]):
            self.assertEqual(analytics.relval_counts("workflow", tag="CMSSW_20_1_X_2026-11-03-2300")["sub_ibs_included"], ["ASAN"])


class RuleGapsFoundInConversationsTests(unittest.TestCase):
    COUNTS = dict(distinct=1, total_results=58, primary_builds_only=20, rows=[dict(workflow="539.0", name="DY", results=58, builds=30, archs=["el9_amd64_gcc14"], last="2026-11-05-1100")])

    def test_a_timeout_over_days_is_counted_over_those_days(self):
        with patch("server.api_server.get_exit_code_names", return_value={31744: "TimeOut"}), patch("server.api_server.relval_counts", return_value=self.COUNTS) as counts:
            answer = _relval_by_exit_code_answer("which workflows timed out in 20_1 in the last three days?")
        self.assertIn("`539.0`", answer)
        self.assertIn("31744 (TimeOut)", answer)
        self.assertEqual((counts.call_args.kwargs["release"], counts.call_args.kwargs["exit_code"]), ("20_1", "31744"))
        self.assertEqual(counts.call_args.kwargs["since"], (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d"))

    def test_none_found_says_what_was_searched(self):
        with patch("server.api_server.get_exit_code_names", return_value={}), patch("server.api_server.relval_counts", return_value=dict(rows=[])):
            self.assertIn("No RelVal failed with exit code 62720 in `CMSSW_20_1` builds", _relval_by_exit_code_answer("which workflows failed with exit code 62720 this week in 20_1?"))

    def test_on_which_architectures_is_a_follow_up(self):
        for question in ("on which architectures?", "the first one, on which architecture?"):
            self.assertIsNone(_architectures_answer(question))

    def test_per_flavour_is_recognised(self):
        self.assertTrue(_PER_FLAVOUR_RE.search("how many relvals failed per flavour in the 2026-11-04-2300 20_1 build?"))
        self.assertFalse(_PER_FLAVOUR_RE.search("which flavour is worst?"))

    def test_sub_ib_failures_are_reported_next_to_primary_ones(self):
        rows = [dict(tag="CMSSW_20_1_ROOT6_X_2026-11-04-2300", arch="el9", package="FWIO/RNTupleTempTests")] * 5
        with patch.object(analytics, "_rows", side_effect=[[], rows]):
            result = analytics.unit_test_changes("20_1", package="FWIO/RNTupleTempTests", since="2026-10-30")
        self.assertEqual(result["packages_failed"], 0)
        self.assertEqual(result["also_failed_in_sub_ibs"], [dict(package="FWIO/RNTupleTempTests", flavour="ROOT6", arch="el9", builds_failed=5)])

    def test_a_build_that_was_looked_up_and_exists_is_not_invented(self):
        tag = "CMSSW_20_1_X_2026-11-04-1100"
        evidence = [dict(arguments=dict(tag=tag), result=[dict(pr=1)])]
        with patch("server.api_server._identifiers_exist", return_value=True):
            self.assertTrue(_identifiers_grounded(f"PRs in `{tag}`", evidence, "which PRs in 2026-11-04-1100?"))
        with patch("server.api_server._identifiers_exist", return_value=False):
            self.assertFalse(_identifiers_grounded(f"PRs in `{tag}`", evidence, "which PRs?"))
        self.assertFalse(_identifiers_grounded(f"PRs in `{tag}`", [dict(arguments={}, result=[])], "which PRs?"))


class CompareAcrossFlavoursTests(unittest.TestCase):
    def rows(self, tag, archs, workflows):
        return [dict(workflow=w, name="n", status="FAILED", exit_code=1, known=False, arch=a, tag=tag) for a in archs for w in workflows]

    def compare(self, a_rows, b_rows, **kwargs):
        with patch.object(analytics, "_rows", side_effect=[a_rows, b_rows]), patch.object(analytics, "_usual_relval_counts", return_value={}):
            return analytics.compare_relvals(kwargs.pop("a", "CMSSW_20_1_X_2026-11-03-2300"), kwargs.pop("b", "CMSSW_20_1_ASAN_X_2026-11-04-2300"), **kwargs)

    def test_architectures_dont_overwrite_each_other_across_flavours(self):
        result = self.compare(self.rows("CMSSW_20_1_X_2026-11-03-2300", ["a1", "a2", "a3"], ["1.0", "2.0"]), self.rows("CMSSW_20_1_ASAN_X_2026-11-04-2300", ["a1"], ["1.0"]))
        self.assertEqual((result["earlier"]["failed"], result["later"]["failed"]), (6, 1))
        self.assertEqual(len(result["failed_in_both"]), 1)

    def test_an_architecture_family_doesnt_collapse_either(self):
        tag = "CMSSW_20_1_X_2026-11-03-2300"
        result = self.compare(self.rows(tag, ["el9_a", "el9_b"], ["1.0"]), self.rows("CMSSW_20_1_X_2026-11-04-2300", ["el9_a", "el9_b"], ["1.0"]), a=tag, b="CMSSW_20_1_X_2026-11-04-2300", arch_a="el9", arch_b="el9")
        self.assertEqual(result["earlier"]["failed"], 2)

    def test_two_architectures_still_compare_the_same_workflow(self):
        tag = "CMSSW_20_1_X_2026-11-03-2300"
        result = self.compare(self.rows(tag, ["el8"], ["1.0"]), self.rows(tag, ["el9"], ["1.0"]), a=tag, b=tag, arch_a="el8", arch_b="el9")
        self.assertEqual(len(result["failed_in_both"]), 1)


class KeepsTheConditionTests(unittest.TestCase):
    def test_without_stays_without(self):
        q = "no, on all architectures but without known errors"
        turned = "how many relvals failed in CMSSW_20_1_X_2026-11-02-2300 on all architectures, counting known errors as failures (i.e. not excluding known errors)?"
        fixed = _keeps_known_error_condition(q, turned)
        self.assertIn("without known errors", fixed)
        self.assertNotIn("counting known", fixed)

    def test_nothing_else_is_touched(self):
        self.assertEqual(_keeps_known_error_condition("list the known errors", "list the known errors in X"), "list the known errors in X")
        self.assertEqual(_keeps_known_error_condition("without known errors", "failures in X without known errors"), "failures in X without known errors")


class ReplyRepeatsTests(unittest.TestCase):
    def test_every_model_call_has_a_seed(self):
        source = open("server/agent.py").read()
        self.assertEqual(source.count("temperature=0,"), source.count("seed=LLM_SEED,"))
        self.assertIn('request["seed"] = LLM_SEED', source)


class WhatCountsLeaveOutTests(unittest.TestCase):
    TAG = "CMSSW_20_1_X_2026-11-03-2300"

    def test_das_errors_and_the_primary_share_are_reported(self):
        rows = [relval(self.TAG, "el9", "537.0"), relval("CMSSW_20_1_ASAN_X_2026-11-03-2300", "el9", "539.0")]
        with patch.object(analytics, "_rows", side_effect=[rows, [dict(workflow="38551.85"), dict(workflow="38551.85")]]):
            result = analytics.relval_counts("workflow", tag=self.TAG)
        self.assertEqual((result["total_results"], result["primary_builds_only"]), (2, 1))
        self.assertEqual(result["das_errors_left_out"], {"results": 2, "workflows": ["38551.85"]})

    def test_merge_times_cover_what_the_list_per_build_doesnt(self):
        listed = [dict(repo="cmssw", pr_number=1, title="a", merged_at="2026-11-02T12:00:00Z", author="x")]
        timed = listed + [dict(repo="cmsdist", pr_number=10974, title="b", merged_at="2026-11-02T21:30:39+00:00", author="y")]
        with patch("graph.tools.find_prs_merged_between", return_value=listed), patch.object(analytics, "_rows", return_value=timed) as rows:
            result = analytics.find_prs_merged_between("CMSSW_20_1_X_2026-11-02-1100", "CMSSW_20_1_X_2026-11-03-1100")
        self.assertEqual([r["pr_number"] for r in result["by_merge_time"]], [10974])
        self.assertEqual(rows.call_args.kwargs["start"], "2026-11-02T10:00")  # 11:00 at CERN, in winter
        self.assertEqual(rows.call_args.kwargs["head"], "CMSSW_20_1")

    def test_the_tool_the_ai_gets_uses_merge_times(self):
        self.assertIn("by_merge_time", TOOLS["find_prs_merged_between"].__doc__)


class PrOriginTests(unittest.TestCase):
    """Fake PRs by number; what each says about who opened it."""

    PRS = {
        1: dict(body="", names=["alice"]),
        2: dict(body="Automatically ported from CMSSW_8_0_X #1 (original by @some-group).", names=["cmsbuild", "some-group"]),
        3: dict(body="Automatically ported from CMSSW_8_0_X #2 (original by @some-group).", names=["cmsbuild"]),
        4: dict(body="Running code-format for dqm", names=["cmsbuild"]),
        5: dict(body="", names=["fork-owner", "ghost"]),
        6: dict(body="Automatically ported from CMSSW_7_2_X #5", names=["cmsbuild"]),
        7: dict(body="backport of #999", names=["cmsbuild"]),
        8: dict(body="Automatically ported from CMSSW_7_2_X #8", names=["cmsbuild"]),
        9: dict(body="backport of #1", names=["alice"]),
    }

    def origin(self, number):
        # _rows: the "is it an Issue?" lookup for a number that is no imported PR (no database in CI)
        with patch.object(analytics, "_pr_authors", side_effect=lambda repo, n: self.PRS.get(int(n))), \
                patch.object(analytics, "_rows", return_value=[]):
            return analytics.pr_origin(number)

    def test_a_person_is_just_the_opener(self):
        origin = self.origin(1)
        self.assertEqual(origin["opened_by"], "alice")
        self.assertFalse(origin["opened_automatically"])

    def test_the_person_comes_from_the_original_not_the_group_in_the_description(self):
        origin = self.origin(2)
        self.assertEqual((origin["opened_by"], origin["branch_owner"], origin["ported_from"], origin["original_opened_by"]), ("cmsbuild", "some-group", [1], "alice"))

    def test_a_port_of_a_port_is_followed(self):
        origin = self.origin(3)
        self.assertEqual((origin["ported_from"], origin["original_opened_by"]), ([2, 1], "alice"))

    def test_no_link_names_nobody(self):
        origin = self.origin(4)
        self.assertEqual((origin["opened_by"], origin["ported_from"], origin["original_opened_by"]), ("cmsbuild", [], None))

    def test_a_deleted_account_is_not_a_person(self):
        origin = self.origin(5)
        self.assertEqual((origin["opener_account_deleted"], origin["branch_owner"]), (True, "fork-owner"))
        self.assertTrue(self.origin(6)["original_account_deleted"])
        self.assertIsNone(self.origin(6)["original_opened_by"])

    def test_an_original_not_imported_or_a_loop_names_nobody(self):
        self.assertEqual((self.origin(7)["ported_from"], self.origin(7)["original_opened_by"]), ([999], None))
        self.assertEqual((self.origin(8)["ported_from"], self.origin(8)["original_opened_by"]), ([], None))

    def test_a_person_who_opened_a_backport_stays_its_opener(self):
        origin = self.origin(9)
        self.assertEqual((origin["opened_by"], origin["ported_from"]), ("alice", []))

    def test_missing_pr(self):
        self.assertFalse(self.origin(12345)["found"])

    def test_sentences(self):
        with patch.object(analytics, "_pr_authors", side_effect=lambda repo, n: self.PRS.get(int(n))):
            self.assertEqual(_pr_opener_sentence("cmssw", 2, "cmsbuild"), "PR #2 was opened automatically by `cmsbuild`. It is a port of #1, which was opened by **alice**.")
            self.assertIn("doesn't name a person", _pr_opener_sentence("cmssw", 4, "cmsbuild"))
            self.assertIn("`fork-owner`, which is a clue", _pr_opener_sentence("cmssw", 5, "ghost"))
            self.assertIn("doesn't say who opened", _pr_opener_sentence("cmssw", 7, "cmsbuild"))
            self.assertEqual(_pr_opener_sentence("cmssw", 1, "alice"), "PR #1 was created by **alice**.")


class RelvalCountsTests(unittest.TestCase):
    def test_known_errors_are_left_out_unless_asked_for(self):
        rows = [relval("CMSSW_20_1_X_2026-11-04-2300", "el9", "1.0"), relval("CMSSW_20_1_X_2026-11-04-2300", "el9", "2.0", known=True)]
        with patch.object(analytics, "_rows", return_value=rows):
            result = analytics.relval_counts("workflow", tag="CMSSW_20_1_X_2026-11-04-2300")
            self.assertEqual((result["total_results"], result["known_errors_left_out"]), (1, 1))
            self.assertEqual(analytics.relval_counts("workflow", tag="x", include_known_errors="true")["total_results"], 2)

    def test_group_by_ranks_and_a_bad_group_falls_back(self):
        rows = [relval("t_2026-11-04-2300", "a", "1.0", code=256), relval("t_2026-11-04-2300", "b", "2.0", code=256), relval("t_2026-11-04-2300", "a", "3.0", code=134)]
        with patch.object(analytics, "_rows", return_value=rows):
            self.assertEqual(analytics.relval_counts("exit_code")["rows"][0]["exit_code"], 256)
            self.assertEqual(analytics.relval_counts("nonsense")["distinct"], 3)


class CompareTests(unittest.TestCase):
    def side(self, status_by_workflow, tag):
        return [dict(workflow=w, name=w, status=s, exit_code=1, known=False, arch="el9", tag=tag) for w, s in status_by_workflow.items()]

    EARLY, LATE = "CMSSW_20_1_X_2026-11-04-2300", "CMSSW_20_1_X_2026-11-05-1100"

    def compare(self, first, second, *rows, usual=None):
        with patch.object(analytics, "_rows", side_effect=list(rows)), patch.object(analytics, "_usual_relval_counts", return_value=usual or {}):
            return analytics.compare_relvals(first, second, "el9", "el9")

    def test_no_row_is_not_a_pass(self):
        result = self.compare(self.EARLY, self.LATE, self.side({"539.0": "FAILED", "1.0": "FAILED", "2.0": "PASSED"}, self.EARLY), self.side({"1.0": "PASSED"}, self.LATE), usual={("CMSSW_20_1_X_", "el9"): 954})
        self.assertEqual([r["workflow"] for r in result["fixed_in_later"]], ["1.0"])
        self.assertEqual([r["workflow"] for r in result["failed_earlier_no_result_later"]], ["539.0"])
        self.assertIn("still being imported", result["partial_warning"])

    def test_builds_given_backwards_still_read_forwards(self):
        # 10804.31 passed, then failed: a reversed call must not call that a fix
        result = self.compare(self.LATE, self.EARLY, self.side({"10804.31": "PASSED"}, self.EARLY), self.side({"10804.31": "FAILED"}, self.LATE))
        self.assertEqual(result["earlier"]["tag"], self.EARLY)
        self.assertEqual([r["workflow"] for r in result["new_failure_in_later"]], ["10804.31"])
        self.assertEqual(result["fixed_in_later"], [])

    def test_known_errors_are_not_failures(self):
        rows = [dict(workflow="1.0", name="x", status="FAILED", exit_code=1, known=True, arch="el9", tag=self.EARLY)]
        result = self.compare(self.EARLY, self.EARLY, rows, rows)
        self.assertEqual(result["failed_in_both"], [])
        self.assertEqual(result["known_errors"], {"first": 1, "second": 1})


class PartialNoteTests(unittest.TestCase):
    TAG = "CMSSW_20_1_X_2026-11-05-1100"

    def test_said_once_whatever_the_route(self):
        with patch("server.api_server.partial_builds", return_value=[dict(tag=self.TAG, arch="el9", results=3, usual=954)]):
            answer = _with_partial_build_note(f"**8 of 11** passed in `{self.TAG}`", "how many passed?")
        self.assertTrue(answer.startswith("**Still being imported"))
        self.assertIn("3 of about 954", answer)

    def test_only_the_builds_the_answer_is_about(self):
        answer = "Workflow history:\n" + "x" * 500 + f"\n`{self.TAG}` failed"
        with patch("server.api_server.partial_builds") as check:
            self.assertEqual(_with_partial_build_note(answer, "q"), answer)
        check.assert_not_called()

    def test_unit_test_questions_get_no_relval_note(self):
        with patch("server.api_server.partial_builds") as check:
            answer = f"Failed unit-test packages for `{self.TAG}`: none"
            self.assertEqual(_with_partial_build_note(answer, "what are the unit test failures right now?"), answer)
        check.assert_not_called()

    def test_not_repeated_or_invented(self):
        with patch("server.api_server.partial_builds", return_value=[]):
            self.assertEqual(_with_partial_build_note(f"`{self.TAG}` is clean", "q"), f"`{self.TAG}` is clean")
        with patch("server.api_server.partial_builds") as check:
            self.assertIn("partly imported", _with_partial_build_note(f"`{self.TAG}` is partly imported", "q"))
        check.assert_not_called()


class CompletenessTests(unittest.TestCase):
    def test_a_build_is_not_compared_with_itself(self):
        rows = [dict(tag="CMSSW_20_1_X_2026-11-05-1100", arch="el8", total=291)]
        with patch.object(analytics, "_usual_relval_counts", return_value={("CMSSW_20_1_X_", "el8"): 954}):
            marked = analytics._with_completeness(rows)[0]
        self.assertFalse(marked["complete"])
        self.assertEqual(marked["usual_relvals"], 954)

    def test_a_build_far_below_its_usual_count_is_partial(self):
        rows = [dict(tag="CMSSW_20_1_X_2026-11-04-2300", arch="el8", total=954), dict(tag="CMSSW_20_1_X_2026-11-05-1100", arch="el8", total=291), dict(tag="CMSSW_20_1_X_2026-10-31-1100", arch="el8", total=947)]
        with patch.object(analytics, "_usual_relval_counts", return_value={("CMSSW_20_1_X_", "el8"): 954}):
            marked = {r["tag"][-15:]: r["complete"] for r in analytics._with_completeness(rows)}
        self.assertEqual(marked, {"2026-11-04-2300": True, "2026-11-05-1100": False, "2026-10-31-1100": True})

    def test_health_ignores_partial_builds(self):
        rows = [dict(tag="CMSSW_20_1_X_2026-11-04-2300", arch="el8", total=954, failed=1, known=0, das=0, unit_tests=1, unit_failed=0, build_status="passed"),
                dict(tag="CMSSW_20_1_X_2026-11-05-1100", arch="el8", total=291, failed=9, known=0, das=0, unit_tests=1, unit_failed=0, build_status="passed")]
        with patch.object(analytics, "_ib_counts", return_value=rows), patch.object(analytics, "_usual_relval_counts", return_value={("CMSSW_20_1_X_", "el8"): 954}):
            result = analytics.release_health()
        self.assertEqual(result["series"][0]["relval_failures"], 1)
        self.assertEqual(result["still_partial_not_counted"], ["CMSSW_20_1_X_2026-11-05-1100 el8"])


class ParsingTests(unittest.TestCase):
    def test_series_and_dates(self):
        for given in ("20_1", "CMSSW_20_1", "CMSSW_20_1_X", "20.1"):
            self.assertEqual(analytics._series_prefix(given), "CMSSW_20_1_")
        self.assertEqual(analytics._when("2026-11-03"), "2026-11-03-0000")
        self.assertEqual(analytics._when("2026-11-03", end=True), "2026-11-03-9999")
        self.assertEqual(analytics._when("CMSSW_20_1_X_2026-11-04-2300"), "2026-11-04-2300")
        self.assertIsNone(analytics._when("yesterday"))

    def test_flavour_of_a_tag(self):
        self.assertEqual(analytics._variant_of("CMSSW_20_1_X_2026-11-04-2300"), "primary")
        self.assertEqual(analytics._variant_of("CMSSW_20_1_ASAN_X_2026-11-04-2300"), "ASAN")

    def test_every_tool_is_registered_with_a_usable_description(self):
        for name in analytics.ANALYTICS_TOOLS:
            self.assertIn(name, TOOLS)
            description = _function_to_openai_schema(name, TOOLS[name])["function"]["description"]
            self.assertGreater(len(description), 80, name)


class ClangPackageInTagTests(unittest.TestCase):
    def fake(self, rows, parsed=5):
        session = MagicMock()
        session.run.side_effect = [rows, MagicMock(single=lambda: {"n": parsed})]
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        return patch("server.api_server.get_driver", return_value=driver)

    def test_a_package_is_not_answered_with_the_whole_build(self):
        with self.fake([dict(arch="el9_amd64_gcc13", locations=5, occurrences=6, packages=1)]):
            answer = _clang_package_in_tag_answer("how many clang warnings does RecoTracker/LSTCore have in CMSSW_20_1_X_2026-08-12-2300 on el9_amd64_gcc13?", "CMSSW_20_1_X_2026-08-12-2300")
        self.assertIn("**`RecoTracker/LSTCore`**", answer)
        self.assertIn("| 5 | 6 | 1 |", answer)

    def test_no_package_named_is_left_to_the_general_answer(self):
        self.assertIsNone(_clang_package_in_tag_answer("clang warnings for CMSSW_20_1_X_2026-08-12-2300", "CMSSW_20_1_X_2026-08-12-2300"))

    def test_a_build_never_parsed_is_not_zero(self):
        with self.fake([], parsed=0):
            self.assertIsNone(_clang_package_in_tag_answer("clang warnings in FWCore for CMSSW_20_1_X_2026-11-04-2300", "CMSSW_20_1_X_2026-11-04-2300"))


class WindowTests(unittest.TestCase):
    def test_named_month(self):
        self.assertEqual(_named_month_window("merged in October 2026?"), ("2026-10-01", "2026-11-01"))
        self.assertEqual(_named_month_window("merged in 2026-12"), ("2026-12-01", "2027-01-01"))
        self.assertEqual(_named_month_window("merged recently"), (None, None))

    def test_a_date_range_is_not_a_day(self):
        self.assertTrue(_DATE_RANGE_RE.search("how many IBs from 2026-10-26 to 2026-11-04 had X failing?"))
        self.assertIsNone(_DATE_RANGE_RE.search("did aarch64 pass on 2026-10-26?"))


class WhatTheModelMustNotAddTests(unittest.TestCase):
    def test_an_example_tag_is_not_a_known_question(self):
        self.assertTrue(_known_question_invents("any hidden problems in passed tests in CMSSW_20_1_X_2026-10-12-2300?", "which unit tests have hidden problems in the latest IB?", []))
        self.assertTrue(_known_question_invents("has 539.0 been reported?", "has it been reported?", []))
        self.assertFalse(_known_question_invents("has 539.0 been reported?", "has it been reported?", [{"question": "how is 539.0?", "answer": "failing"}]))
        self.assertFalse(_known_question_invents("what is failing in the latest IB", "whats broken", []))

    def test_real_names_are_taken_out_unless_the_data_has_them(self):
        self.assertNotIn("Andrea Bocci", _without_invented_names("**fwyzard (Andrea Bocci)** is the author", []))
        self.assertNotIn("Andrea Bocci", _without_invented_names("fwyzard (Andrea Bocci) is the author", []))
        self.assertIn("Andrea Bocci", _without_invented_names("fwyzard (Andrea Bocci) is the author", [{"result": "Andrea Bocci"}]))
        self.assertEqual(_without_invented_names("the job (Run Three Test) ran", [{"result": "x"}]), "the job ran")


class DashboardTimeTests(unittest.TestCase):
    def test_pr_52269_reads_as_on_github(self):
        from build_graph import dashboard_time_to_iso
        self.assertEqual(dashboard_time_to_iso("1793857384"), "2026-11-05T06:43:04+00:00")  # GitHub: 2026-11-05T06:43:04Z
        self.assertIsNone(dashboard_time_to_iso(None))


class NewestCompleteTests(unittest.TestCase):
    def test_a_partial_newest_build_is_passed_over(self):
        rows = [dict(tag="CMSSW_20_1_X_2026-11-05-1100", arch="el8", n=291), dict(tag="CMSSW_20_1_X_2026-11-04-2300", arch="el8", n=954)]
        with patch.object(analytics, "_rows", return_value=rows), patch.object(analytics, "_usual_relval_counts", return_value={("CMSSW_20_1_X_", "el8"): 954}):
            self.assertEqual(analytics.newest_complete_primary("CMSSW_20_1_"), ["CMSSW_20_1_X_2026-11-04-2300"])


class UsualCountTests(unittest.TestCase):
    def test_the_usual_count_is_the_middle_build_and_a_one_off_big_build_does_not_shift_it(self):
        rows = [dict(tag=f"CMSSW_17_1_X_2026-11-0{d}-2300", arch="el8_aarch64", n=n) for d, n in enumerate([939, 939, 1252, 939, 4], 1)]
        analytics._USUAL_CACHE.clear()
        with patch.object(analytics, "_rows", return_value=rows):
            self.assertEqual(analytics._usual_relval_counts()[("CMSSW_17_1_X_", "el8_aarch64")], 939)
        analytics._USUAL_CACHE.clear()

    def test_an_old_build_is_not_still_arriving(self):
        with patch.object(analytics, "_usual_relval_counts", return_value={("CMSSW_20_1_X_", "el9"): 954}), patch.object(analytics, "_rows") as query:
            self.assertEqual(analytics.partial_builds(["CMSSW_20_1_X_2026-10-14-2300"]), [])
        query.assert_not_called()


class ReleaseSpellingTests(unittest.TestCase):
    def test_any_spelling_finds_the_series(self):
        session = MagicMock()
        session.run.return_value = []
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch("graph.tools.get_driver", return_value=driver):
            list_ib_tags(release="17_1")
        self.assertEqual(session.run.call_args.args[1]["prefix"], "CMSSW_17_1_")


class WordSearchTests(unittest.TestCase):
    def test_a_text_search_for_ordinary_words_is_recognised(self):
        self.assertTrue(_WORD_SEARCH_DEAD_END_RE.match('No imported CMSSW Issue or PR title/description contains "updated most".'))
        self.assertTrue(_WORD_SEARCH_DEAD_END_RE.match('No Issue or PR contains the exact phrase "20_1 ib".'))
        self.assertIsNone(_WORD_SEARCH_DEAD_END_RE.match("**3 open Issues** mention ROCm"))


class ScopeGuardTests(unittest.TestCase):
    def test_another_days_build_is_caught(self):
        self.assertEqual(_scope_mismatch("how many failed in the CLANG IB of 2026-11-03 23:00?", "What's failing in `CMSSW_20_1_CLANG_X_2026-11-04-2300`"), "builds of 2026-11-03")
        self.assertIsNone(_scope_mismatch("what failed on 2026-11-03?", "`CMSSW_20_1_X_2026-11-03-2300`: 2"))

    def test_another_hour_of_the_day_is_caught(self):
        self.assertIn("11:00", _scope_mismatch("failures in the CUDART IB of 2026-11-04 11:00", "`CMSSW_20_1_CUDART_X_2026-11-04-2300`: 1"))
        self.assertIsNone(_scope_mismatch("failures in the CUDART IB of 2026-11-04 11:00", "`CMSSW_20_1_CUDART_X_2026-11-04-1100`: 2"))

    def test_one_build_for_a_period_is_caught(self):
        self.assertIn("whole period", _scope_mismatch("how many IBs from 2026-10-26 to 2026-11-04 had X failing?", "X in `CMSSW_20_1_X_2026-10-26-2300`: failed"))
        self.assertIsNone(_scope_mismatch("how many IBs from 2026-10-26 to 2026-11-04 had X failing?", "2: `CMSSW_20_1_X_2026-10-29-1100`, `CMSSW_20_1_X_2026-10-29-2300`", investigated=True))

    def test_a_week_from_one_build_is_caught_for_rules_only(self):
        answer = "**15 DAS error(s)** in `CMSSW_20_1_X_2026-11-04-2300`"
        self.assertIn("whole period", _scope_mismatch("which workflows had a DAS error this week?", answer))
        self.assertIsNone(_scope_mismatch("which workflows had a DAS error this week?", answer, investigated=True))

    def test_yesterday_is_a_date(self):
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        other = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d")
        self.assertIn(yesterday, _scope_mismatch("did yesterday's 23:00 build fail?", f"`CMSSW_20_1_X_{other}-1100` passed"))
        self.assertIsNone(_scope_mismatch("did yesterday's 23:00 build fail?", f"`CMSSW_20_1_X_{yesterday}-2300` passed"))

    def test_a_window_is_not_one_day(self):
        self.assertIsNone(_scope_mismatch("what failed since 2026-11-03?", "`CMSSW_20_1_X_2026-11-04-2300`: 2"))

    def test_another_architecture_or_build_or_workflow(self):
        self.assertIn("el8_amd64_gcc14", _scope_mismatch("failures on el8_amd64_gcc14", "on `el9_amd64_gcc14`: 2"))
        self.assertIn("CMSSW_20_1_X_2026-11-03-2300", _scope_mismatch("status of CMSSW_20_1_X_2026-11-03-2300", "`CMSSW_20_1_X_2026-11-04-2300` passed"))
        self.assertIn("537.0", _scope_mismatch("why does 537.0 fail?", "`539.0` timed out"))

    def test_sub_ibs_of_the_build_count_and_silence_about_builds_is_no_mismatch(self):
        self.assertIsNone(_scope_mismatch("status of CMSSW_20_1_X_2026-11-03-2300", "`CMSSW_20_1_ASAN_X_2026-11-03-2300`: 1 failing"))
        self.assertIsNone(_scope_mismatch("failures on el8_amd64_gcc14", "No failures."))


class HistoryTests(unittest.TestCase):
    def test_flips_and_first_failure_per_track(self):
        tag = "CMSSW_20_1_X_2026-11-0{}-2300"
        rows = [dict(tag=tag.format(day), arch="el9", name="n", status=status, exit_code=1 if status == "FAILED" else 0, known=False) for day, status in enumerate(["PASSED", "FAILED", "PASSED", "FAILED"], 1)]
        with patch.object(analytics, "_rows", return_value=rows):
            result = analytics.workflow_history("1.0")
        track = result["tracks"][0]
        self.assertEqual((track["flips"], track["failed"], track["passed"]), (3, 2, 2))
        self.assertEqual(track["first_failed"], tag.format(2))
        self.assertEqual(result["failing_now"], 1)


if __name__ == "__main__":
    unittest.main()
