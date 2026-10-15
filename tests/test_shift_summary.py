import re
import re
import unittest
from unittest.mock import patch

from server.api_server import (
    _addon_shift_table,
    _build_shift_table,
    _diff_failing_relvals,
    _diff_failing_unittests,
    _plain_shift_table,
    _split_relval_diff,
    _relval_exit_code_cell,
    _split_unittest_diff,
    _status_diff_rows,
    _unittest_shift_table,
    generate_shift_summary,
)



_PREFETCH_PATCH = patch("server.api_server.prefetch_relval_exception_evidence")


def setUpModule():
    # generate_shift_summary prefetches RelVal evidence with one shared scan; these tests
    # mock get_relval_exception_evidence itself, so the prefetch must not reach a database
    _PREFETCH_PATCH.start()


def tearDownModule():
    _PREFETCH_PATCH.stop()


class DiffFailingRelvalsTests(unittest.TestCase):
    def test_detects_newly_failing_and_resolved(self):
        before = [{"workflow_id": "1.0", "name": "A", "errors": 1, "exitcode": 1}]
        after = [
            {"workflow_id": "1.0", "name": "A", "errors": 1, "exitcode": 1},
            {"workflow_id": "2.0", "name": "B", "errors": 2, "exitcode": 2},
        ]
        diff = _diff_failing_relvals(before, after)
        self.assertEqual([identity[0] for identity, _, _ in diff], ["2.0"])

    def test_unchanged_workflow_is_excluded(self):
        row = {"workflow_id": "1.0", "name": "A", "errors": 1, "exitcode": 1}
        diff = _diff_failing_relvals([row], [dict(row)])
        self.assertEqual(diff, [])

    def test_changed_detail_still_failing_on_both_is_reported(self):
        before = [{"workflow_id": "1.0", "name": "A", "errors": 1, "exitcode": 1}]
        after = [{"workflow_id": "1.0", "name": "A", "errors": 5, "exitcode": 1}]
        diff = _diff_failing_relvals(before, after)
        self.assertEqual(len(diff), 1)
        identity, before_row, after_row = diff[0]
        self.assertEqual(identity[0], "1.0")
        self.assertIsNotNone(before_row)
        self.assertIsNotNone(after_row)


class SplitRelvalDiffTests(unittest.TestCase):
    def test_partitions_newly_failing_and_resolved(self):
        before = [{"workflow_id": "1.0", "name": "A", "errors": 1, "exitcode": 1}]
        after = [{"workflow_id": "2.0", "name": "B", "errors": 2, "exitcode": 2}]
        diff = _diff_failing_relvals(before, after)

        newly_failing, resolved = _split_relval_diff(diff)

        self.assertEqual([row["workflow_id"] for row in newly_failing], ["2.0"])
        self.assertEqual([row["workflow_id"] for row in resolved], ["1.0"])

    def test_changed_exit_code_is_one_newly_failing_row_with_the_old_failure(self):
        # Found live 2026-10-15: 1001.2 (ASAN) exit 256 -> 62720 was listed both as
        # newly failing and as resolved, which read as a self-contradiction.
        before = [{"workflow_id": "1001.2", "name": "A", "errors": 0, "exitcode": 256, "step": "step2"}]
        after = [{"workflow_id": "1001.2", "name": "A", "errors": 0, "exitcode": 62720, "step": "step2"},
                 {"workflow_id": "2.0", "name": "B", "errors": 1, "exitcode": 1}]
        newly_failing, resolved = _split_relval_diff(_diff_failing_relvals(before, after))
        self.assertEqual(resolved, [])
        self.assertEqual([row["workflow_id"] for row in newly_failing], ["1001.2", "2.0"])
        self.assertEqual(newly_failing[0]["previous_failure"], {"step": "step2", "exitcode": 256})
        self.assertNotIn("previous_failure", newly_failing[1])
        self.assertEqual(_relval_exit_code_cell(newly_failing[0], {62720: "SIGSEGV"}), "62720 (SIGSEGV); was 256")

    def test_changed_step_with_same_exit_code_says_so(self):
        row = {"workflow_id": "1.0", "exitcode": 65, "step": "step3",
               "previous_failure": {"step": "step2", "exitcode": 65}}
        self.assertEqual(_relval_exit_code_cell(row, {}), "65; was same exit code at step2")

    def test_changed_detail_still_failing_appears_in_neither_list(self):
        before = [{"workflow_id": "1.0", "name": "A", "errors": 1, "exitcode": 1}]
        after = [{"workflow_id": "1.0", "name": "A", "errors": 5, "exitcode": 1}]
        diff = _diff_failing_relvals(before, after)

        newly_failing, resolved = _split_relval_diff(diff)

        self.assertEqual(newly_failing, [])
        self.assertEqual(resolved, [])


class DiffFailingUnittestsTests(unittest.TestCase):
    def test_detects_newly_failing_and_resolved(self):
        before = [{"name": "test_a", "errors": 1}]
        after = [{"name": "test_a", "errors": 1}, {"name": "test_b", "errors": 2}]
        diff = _diff_failing_unittests(before, after)
        self.assertEqual([name for name, _, _ in diff], ["test_b"])

    def test_unchanged_test_is_excluded(self):
        row = {"name": "test_a", "errors": 1}
        diff = _diff_failing_unittests([row], [dict(row)])
        self.assertEqual(diff, [])


class SplitUnittestDiffTests(unittest.TestCase):
    def test_partitions_newly_failing_and_resolved(self):
        before = [{"name": "test_a", "errors": 1}]
        after = [{"name": "test_b", "errors": 2}]
        diff = _diff_failing_unittests(before, after)

        newly_failing, resolved = _split_unittest_diff(diff)

        self.assertEqual([row["name"] for row in newly_failing], ["test_b"])
        self.assertEqual([row["name"] for row in resolved], ["test_a"])


class PlainShiftTableTests(unittest.TestCase):
    def test_empty_rows_render_header_and_separator_only_no_none_text(self):
        # Unlike _relval_table/_warning_list's "None." fallback, the new
        # Build/Unit-Test/AddOn digest sections use a bare header+separator
        # table for "nothing this shift" -- the frontend's row-count-based
        # stat/alert detection expects a real (zero-row) table, not text.
        lines = _plain_shift_table(["Name", "Errors"], [])
        self.assertEqual(lines, ["| Name | Errors |", "| --- | --- |"])
        self.assertNotIn("None.", lines)

    def test_rows_are_rendered_as_table_cells(self):
        lines = _plain_shift_table(["Name", "Errors"], [["test_something_LST", "Segmentation fault"]])
        self.assertIn("| test_something_LST | Segmentation fault |", lines)

    def test_unittest_shift_table_defaults_missing_errors_to_zero(self):
        lines = _unittest_shift_table([{"name": "test_a"}])
        self.assertIn("| test_a | 0 |", lines)


class StatusDiffRowsTests(unittest.TestCase):
    """Build and AddOn are both a single whole-architecture flag transition,
    not a per-item list -- see _build_and_addon_status's docstring."""

    def test_transition_into_error_is_newly_failing(self):
        newly_failing, resolved = _status_diff_rows("passed", "error", "el9_amd64_gcc13")
        self.assertEqual(newly_failing, {"arch": "el9_amd64_gcc13", "status": "error"})
        self.assertIsNone(resolved)

    def test_transition_out_of_error_is_resolved(self):
        newly_failing, resolved = _status_diff_rows("error", "passed", "el9_amd64_gcc13")
        self.assertIsNone(newly_failing)
        self.assertEqual(resolved, {"arch": "el9_amd64_gcc13", "status": "passed"})

    def test_staying_in_error_is_neither(self):
        # Still broken on both sides isn't a "this shift" event for either table.
        newly_failing, resolved = _status_diff_rows("error", "error", "el9_amd64_gcc13")
        self.assertIsNone(newly_failing)
        self.assertIsNone(resolved)

    def test_missing_status_on_both_sides_triggers_nothing(self):
        # Neither tag ever recorded a status for this arch (e.g. addon tests
        # never ran there) -- None must not be misread as "not failing" in a
        # way that fabricates a resolved/newly-failing row.
        newly_failing, resolved = _status_diff_rows(None, None, "el9_amd64_gcc13")
        self.assertIsNone(newly_failing)
        self.assertIsNone(resolved)


class BuildAndAddonShiftTableTests(unittest.TestCase):
    def test_build_shift_table_names_the_architecture(self):
        lines = _build_shift_table([{"arch": "el9_amd64_gcc13", "status": "error"}])
        self.assertIn("| el9_amd64_gcc13 | build status: error |", lines)

    def test_addon_shift_table_has_three_columns_with_na_exit_code(self):
        lines = _addon_shift_table([{"arch": "el9_amd64_gcc13", "status": "error"}])
        self.assertEqual(lines[0], "| Name | Errors | Exit code |")
        self.assertIn("| el9_amd64_gcc13 | addon status: error | n/a |", lines)


EMPTY_WARNINGS_DIFF = {"new": [], "resolved": [], "changed": []}


def _failing_row(workflow_id="140.51", name="RunMuonEG2022C", variant=None, errors=3, exitcode=16640):
    return {
        "workflow_id": workflow_id, "name": name, "variant": variant,
        "errors": errors, "exitcode": exitcode, "ib_tag": "CMSSW_20_1_X_2026-08-19-2300",
    }


class GenerateShiftSummaryTests(unittest.TestCase):
    def test_recurrence_counts_only_the_same_variant_and_arch(self):
        """Found live 2026-10-15: CLANG workflow 1.0 newly failed with a new LogicError
        (exit 18944) and was shown as "Failed in 7 prior build(s) this series" -- those
        were other variants (MULTIARCHS, NONLTO, primary) failing with exit 256."""
        from server.api_server import _workflow_evidence_cells
        recurrence = {
            "total_appearances": 437, "failed_appearances": 31, "failed_tags": [],
            "failure_history": [
                {"tag": "CMSSW_20_1_CLANG_X_2026-10-14-2300", "arch": "el9_amd64_gcc14", "variant": "CLANG", "exitcode": 18944},
                {"tag": "CMSSW_20_1_MULTIARCHS_X_2026-10-11-0000", "arch": "el9_amd64_gcc14", "variant": "MULTIARCHS", "exitcode": 256},
                {"tag": "CMSSW_20_1_X_2026-10-10-1100", "arch": "el9_amd64_gcc14", "variant": "primary", "exitcode": 256},
                {"tag": "CMSSW_20_1_CLANG_X_2026-10-09-2300", "arch": "el8_amd64_gcc14", "variant": "CLANG", "exitcode": 256},
                {"tag": "CMSSW_20_1_CLANG_X_2026-10-09-2300", "arch": "el9_amd64_gcc14", "variant": "CLANG", "exitcode": 256},
            ],
        }
        row = {"workflow_id": "1.0", "variant": "CLANG", "exitcode": 18944,
               "ib_tag": "CMSSW_20_1_CLANG_X_2026-10-14-2300"}
        with patch("server.api_server.get_relval_exception_evidence", return_value=[]):
            cell, _ = _workflow_evidence_cells("1.0", row, "CMSSW_20_1_X_2026-10-14-2300", "el9_amd64_gcc14", recurrence)
        self.assertEqual(cell, "Failed in 1 prior CLANG build(s) this series on el9_amd64_gcc14 "
                               "(most recent CMSSW_20_1_CLANG_X_2026-10-09-2300), 0 with the same exit code; 31/437 overall.")

        recurrence["failure_history"] = recurrence["failure_history"][:3]
        with patch("server.api_server.get_relval_exception_evidence", return_value=[]):
            cell, _ = _workflow_evidence_cells("1.0", row, "CMSSW_20_1_X_2026-10-14-2300", "el9_amd64_gcc14", recurrence)
        self.assertEqual(cell, "First failure in CLANG builds this series on el9_amd64_gcc14; 31/437 overall.")

    def test_workflow_id_only_match_is_labelled_as_a_mention(self):
        from server.api_server import _workflow_evidence_cells
        evidence = [
            {"category": None, "exception": "boom", "matched_by": "workflow_id", "module": None,
             "matches": [{"kind": "PR", "number": 51225, "state": "closed", "merged_at": "2026-06-17"}]},
            {"category": None, "exception": "bang", "matched_by": "module", "module": "EvaluateMuonMVAID",
             "matches": [{"kind": "Issue", "number": 7, "state": "open", "merged_at": None}]},
        ]
        with patch("server.api_server.get_relval_exception_evidence", return_value=evidence):
            _, pr_cell = _workflow_evidence_cells(
                "10001.0", {"ib_tag": "T_X_1"}, "T_X_1", "el9_amd64_gcc14", {"total_appearances": 0})
        # no badge-shaped "[cmssw PR #N](url) (state)" mention, so no PR badge on the card
        self.assertNotIn("[cmssw", pr_cell)
        self.assertIn("Names workflow 10001.0 only, not this error: "
                      "[PR #51225](https://github.com/cms-sw/cmssw/pull/51225)", pr_cell)
        self.assertIn("Names module EvaluateMuonMVAID only, not this error: "
                      "[Issue #7](https://github.com/cms-sw/cmssw/issues/7)", pr_cell)

        evidence.insert(0, {"category": "LogicError", "exception": "boom", "matched_by": "exception", "module": None,
                            "matches": [{"kind": "PR", "number": 9, "state": "open", "merged_at": None}]})
        with patch("server.api_server.get_relval_exception_evidence", return_value=evidence):
            _, pr_cell = _workflow_evidence_cells(
                "10001.0", {"ib_tag": "T_X_1"}, "T_X_1", "el9_amd64_gcc14", {"total_appearances": 0})
        self.assertTrue(pr_cell.startswith("[LogicError] [cmssw PR #9](https://github.com/cms-sw/cmssw/pull/9) (open) -- "))
        self.assertIn("Also names workflow 10001.0 only, not this error: PR #51225", pr_cell)

    @patch("server.api_server.get_unittest_names", return_value=[])
    @patch("server.api_server.get_relval_workflow_ids", return_value=[])
    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.get_relval_exception_evidence", return_value=[])
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server._build_and_addon_status", return_value={})
    def test_no_new_failures_renders_none_sections_and_no_evidence_block(self, *_mocks):
        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        self.assertIn("Newly failing RelVal workflows", markdown)
        self.assertNotIn("Newly failing workflow evidence", markdown)
        self.assertIn("relevance and historical pattern only", markdown)

    @patch("server.api_server.get_unittest_names", return_value=[])
    @patch("server.api_server.get_relval_workflow_ids", return_value=[])
    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.get_relval_exception_evidence")
    @patch("server.api_server.get_relval_recurrence")
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._build_and_addon_status", return_value={})
    def test_new_failure_with_recurrence_and_pr_evidence(
        self, build_and_addon_status, get_failing_unittests, get_exit_code_names, arch, get_failing, diff_w,
        recurrence, find_prs, search_c, get_relval_ids, get_ut_names
    ):
        new_row = _failing_row()
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [] if tag == "CMSSW_20_1_X_2026-08-19-1100" else [new_row]
        )
        # Positive proof the primary variant existed (ran, any status) at BOTH tags,
        # so _variant_diff_baseline takes the real exact-match path -- a zero-failure
        # baseline -- instead of falling back to a substituted earlier build.
        get_relval_ids.return_value = [{"variant": None, "workflow_id": "0.0"}]
        recurrence.return_value = {
            "workflow_id": "140.51", "name": "RunMuonEG2022C",
            "total_appearances": 5, "failed_appearances": 3,
            "failed_tags": [
                "CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-18-2300", "CMSSW_19_0_X_2026-08-01-0000",
            ],
            "failure_history": [
                {"tag": "CMSSW_20_1_X_2026-08-19-2300", "arch": "el9_amd64_gcc14", "variant": "primary", "exitcode": 16640},
                {"tag": "CMSSW_20_1_X_2026-08-18-2300", "arch": "el9_amd64_gcc14", "variant": "primary", "exitcode": 16640},
                {"tag": "CMSSW_19_0_X_2026-08-01-0000", "arch": "el9_amd64_gcc14", "variant": "primary", "exitcode": 16640},
            ],
        }
        find_prs.return_value = [
            {
                "category": "LogicError",
                "exception": "A muon candidate track has more than 5 stations with matching segments.",
                "matches": [{"kind": "PR", "number": 45213, "title": "fix", "state": "closed", "merged_at": "2026-08-01T00:00:00Z"}],
            },
        ]

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        # Evidence is embedded as extra columns on the newly-failing table (the
        # frontend auto-hides anything past the first four columns behind a
        # per-row toggle), not a separate always-open section.
        self.assertIn("Recurrence | PR/Issue evidence |", markdown)
        self.assertNotIn("Stored knowledge", markdown)
        self.assertIn("`140.51` | RunMuonEG2022C", markdown)
        self.assertIn("cmssw PR #45213", markdown)
        self.assertIn("(https://github.com/cms-sw/cmssw/pull/45213)", markdown)
        self.assertIn("relevance only, do not prove a PR or Issue caused or fixed a failure", markdown)
        # the out-of-series tag must not be counted as an in-series recurrence, and the
        # build being reported (08-19-2300) is not a "prior" build of itself
        self.assertIn("Failed in 1 prior primary build(s) this series on el9_amd64_gcc14 "
                      "(most recent CMSSW_20_1_X_2026-08-18-2300), 1 with the same exit code; 3/5 overall.", markdown)
        # Regression (found live 2026-09-18, workflow 537.0 vs closed Issue #49849):
        # the evidence lookup must be scoped to THIS row's own ib_tag -- the build
        # that's actually newly failing -- not left unscoped to search the workflow's
        # entire cross-release-series exception history and surface a stale,
        # unrelated signature as if it were evidence for the current failure.
        self.assertEqual(find_prs.call_args.kwargs.get("ib_tag"), new_row["ib_tag"])

    @patch("server.api_server.get_unittest_names", return_value=[])
    @patch("server.api_server.get_relval_workflow_ids", return_value=[])
    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.get_relval_exception_evidence", return_value=[])
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._build_and_addon_status", return_value={})
    def test_new_failure_with_no_recurrence_and_no_pr_evidence(
        self, build_and_addon_status, get_failing_unittests, get_exit_code_names, arch, get_failing, diff_w,
        recurrence, find_prs, search_c, get_relval_ids, get_ut_names
    ):
        new_row = _failing_row(workflow_id="999.0", name="BrandNewWorkflow")
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [] if tag == "CMSSW_20_1_X_2026-08-19-1100" else [new_row]
        )
        get_relval_ids.return_value = [{"variant": None, "workflow_id": "0.0"}]

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        self.assertIn("No prior appearance of this workflow ID in the imported graph", markdown)
        # No PR/Issue match -> the evidence cell is empty, not a placeholder sentence --
        # a shifter should see nothing rather than text implying a search came up empty
        # (see _workflow_evidence_cells; this used to render the placeholder text below).
        self.assertNotIn("No PR/Issue in the imported graph matches this workflow's exception signature", markdown)
        self.assertIn("| `999.0` | BrandNewWorkflow", markdown)
        row_line = next(line for line in markdown.splitlines() if line.startswith("| `999.0`"))
        self.assertTrue(row_line.rstrip().endswith("|  |"), row_line)

    @patch("server.api_server.get_unittest_names", return_value=[])
    @patch("server.api_server.get_relval_workflow_ids")
    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.get_relval_exception_evidence", return_value=[])
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._build_and_addon_status", return_value={})
    def test_resolved_workflow_appears_only_in_resolved_table(
        self, build_and_addon_status, get_failing_unittests, get_exit_code_names, arch, get_failing, diff_w,
        recurrence, find_prs, search_c, get_relval_ids, get_ut_names
    ):
        resolved_row = _failing_row(workflow_id="1.0", name="OldFlakyWorkflow")
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [resolved_row] if tag == "CMSSW_20_1_X_2026-08-19-1100" else []
        )
        # "Resolved" now requires positive proof the workflow actually ran at latest_tag
        # (see get_relval_workflow_ids) - absence from the failing list alone isn't
        # enough, since it could also mean latest_tag hasn't been ingested yet.
        get_relval_ids.return_value = [{"variant": None, "workflow_id": "1.0"}]

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        self.assertIn("OldFlakyWorkflow", markdown)
        self.assertNotIn("Newly failing workflow evidence", markdown)

    @patch("server.api_server.get_unittest_names", return_value=[])
    @patch("server.api_server.get_relval_workflow_ids")
    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.get_relval_exception_evidence", return_value=[])
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._build_and_addon_status", return_value={})
    def test_same_workflow_resolved_on_two_sub_ib_variants_is_labeled_per_variant(
        self, build_and_addon_status, get_failing_unittests, get_exit_code_names, arch, get_failing, diff_w,
        recurrence, find_prs, search_c, get_relval_ids, get_ut_names
    ):
        # Regression: get_failing_relvals() includes sub-IB rows alongside the
        # primary run (WHERE ib.tag = $t OR ib.parent_tag = $t), and
        # generate_shift_summary diffs each sub-IB variant's track separately
        # so an unrelated workflow_id collision across variants isn't
        # misread as "unchanged" -- but the rendered table used to drop the
        # variant field entirely, so the same workflow_id resolving
        # independently on two variants rendered as two identical-looking
        # "duplicate" rows with no way to tell them apart (found live against
        # a real deployment: 140.051 appeared 6 times in one table).
        primary_row = _failing_row(workflow_id="140.051", variant=None, errors=120)
        nonlto_row = _failing_row(workflow_id="140.051", variant="NONLTO", errors=121)
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [primary_row, nonlto_row] if tag == "CMSSW_20_1_X_2026-08-19-1100" else []
        )
        # Both variants' tracks genuinely ran (and passed) at latest_tag.
        get_relval_ids.return_value = [
            {"variant": None, "workflow_id": "140.051"},
            {"variant": "NONLTO", "workflow_id": "140.051"},
        ]

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        self.assertIn("| `140.051` | RunMuonEG2022C | 120 | 16640 | primary |", markdown)
        self.assertIn("| `140.051` | RunMuonEG2022C | 121 | 16640 | NONLTO |", markdown)

    @patch("server.api_server.get_unittest_names", return_value=[])
    @patch("server.api_server.get_relval_workflow_ids", return_value=[])
    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.get_relval_exception_evidence")
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._build_and_addon_status", return_value={})
    def test_never_asserts_causality_for_pr_evidence(
        self, build_and_addon_status, get_failing_unittests, get_exit_code_names, arch, get_failing, diff_w,
        recurrence, find_prs, search_c, get_relval_ids, get_ut_names
    ):
        new_row = _failing_row()
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [] if tag == "CMSSW_20_1_X_2026-08-19-1100" else [new_row]
        )
        get_relval_ids.return_value = [{"variant": None, "workflow_id": "0.0"}]
        find_prs.return_value = [
            {
                "category": "LogicError",
                "exception": "boom",
                "matches": [{"kind": "Issue", "number": 1, "title": "t", "state": "open", "merged_at": None}],
            },
        ]

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        # every "PR #"/"Issue #" mention must sit in a sentence hedged with "not prove"/
        # "relevance only", never an unhedged "caused"/"fixed by" claim.
        for line in markdown.splitlines():
            if re.search(r"\bcmssw (?:PR|Issue) #\d+\b", line):
                self.assertTrue("relevance only" in line or "do not prove" in line)
                if "caused" in line:
                    self.assertIn("not prove", line)

    @patch("server.api_server.get_unittest_recurrence", return_value={})
    @patch("server.api_server.get_unittest_log_details", return_value={})
    @patch("server.api_server.get_unittest_names")
    @patch("server.api_server.get_relval_workflow_ids", return_value=[])
    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.get_relval_exception_evidence", return_value=[])
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_unittests")
    @patch("server.api_server._build_and_addon_status", return_value={})
    def test_unit_test_sections_show_newly_failing_and_resolved_rows(
        self, build_and_addon_status, get_failing_unittests, get_exit_code_names, arch, get_failing, diff_w,
        recurrence, find_prs, search_c, get_relval_ids, get_ut_names, _log_details, _ut_recurrence
    ):
        get_failing_unittests.side_effect = lambda tag, arch=None, limit=None, include_sub_ibs=False: (
            [{"name": "OldFlakyTest", "errors": 1}] if tag == "CMSSW_20_1_X_2026-08-19-1100"
            else [{"name": "test_something_LST", "errors": "Segmentation fault"}]
        )
        # "Resolved" now requires positive proof the test actually ran at latest_tag.
        get_ut_names.return_value = [{"name": "OldFlakyTest"}, {"name": "test_something_LST"}]

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        self.assertIn("#### Newly failing Unit Tests", markdown)
        self.assertIn("| Name | Errors | Tests | Failed tests | Exception | Recurrence |", markdown)
        self.assertIn("| test_something_LST | Segmentation fault |  | no logs imported for this build yet |  | "
                      "First failure in this series; 0/0 overall. |", markdown)
        self.assertIn("#### Resolved Unit Tests\n| Name | Errors |", markdown)
        self.assertIn("| OldFlakyTest | 1 |", markdown)

    @patch("server.api_server.get_unittest_names", return_value=[])
    @patch("server.api_server.get_relval_workflow_ids", return_value=[])
    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.get_relval_exception_evidence", return_value=[])
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._build_and_addon_status", return_value={})
    def test_unit_test_sections_have_no_none_text_when_empty(self, *_mocks):
        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        newly_index = markdown.index("#### Newly failing Unit Tests")
        resolved_index = markdown.index("#### Resolved Unit Tests")
        newly_section = markdown[newly_index:resolved_index]
        self.assertIn("| Name | Errors |", newly_section)
        self.assertNotIn("None.", newly_section)

    @patch("server.api_server.get_unittest_names", return_value=[])
    @patch("server.api_server.get_relval_workflow_ids", return_value=[])
    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.get_relval_exception_evidence", return_value=[])
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._build_and_addon_status")
    def test_build_and_addon_sections_show_status_transitions(self, build_and_addon_status, *_mocks):
        build_and_addon_status.return_value = {
            ("CMSSW_20_1_X_2026-08-19-1100", "el9_amd64_gcc14"): {
                "tag": "CMSSW_20_1_X_2026-08-19-1100", "arch": "el9_amd64_gcc14", "variant": "", "parent_tag": None,
                "build_status": "passed", "addon_status": "passed",
            },
            ("CMSSW_20_1_X_2026-08-19-2300", "el9_amd64_gcc14"): {
                "tag": "CMSSW_20_1_X_2026-08-19-2300", "arch": "el9_amd64_gcc14", "variant": "", "parent_tag": None,
                "build_status": "error", "addon_status": "error",
            },
        }

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        self.assertIn("#### Newly failing Build", markdown)
        self.assertIn("| el9_amd64_gcc14 | build status: error |", markdown)
        self.assertIn("#### Newly failing AddOn", markdown)
        self.assertIn("| el9_amd64_gcc14 | addon status: error | n/a |", markdown)
        # nothing regressed out of error, so both Resolved sections stay empty
        resolved_build_idx = markdown.index("#### Resolved Build")
        newly_unit_idx = markdown.index("#### Newly failing Unit Tests")
        self.assertNotIn("el9_amd64_gcc14", markdown[resolved_build_idx:newly_unit_idx])

    @patch("server.api_server.get_unittest_names", return_value=[])
    @patch("server.api_server.get_relval_workflow_ids", return_value=[])
    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.get_relval_exception_evidence", return_value=[])
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server._variant_previous_tag")
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._build_and_addon_status", return_value={})
    def test_sub_ib_missing_at_previous_tag_falls_back_to_its_own_previous_build(
        self, build_and_addon_status, get_failing_unittests, get_exit_code_names, arch, get_failing,
        variant_previous_tag, diff_w, recurrence, find_prs, search_c, get_relval_ids, get_ut_names,
    ):
        # Real bug (2026-09-09): CPP23 only builds on a subset of primary dates and
        # was compared against an empty "before" whenever previous_tag landed on a
        # date CPP23 skipped, misreporting an already-failing CPP23 workflow as
        # newly failing. CPP23 has no build at previous_tag here; its own true
        # previous build (found via _variant_previous_tag) already had the same
        # failure, so it must show as unchanged, not newly failing.
        cpp23_row = _failing_row(workflow_id="140.051", name="RunMuonEG2022C", variant="CPP23", errors=5)
        cpp23_row["ib_tag"] = "CMSSW_20_1_CPP23_X_2026-08-19-2300"
        # get_failing_relvals(latest_tag, ...) pulls CPP23 in via parent_tag matching
        # (so it's queried with the PRIMARY latest_tag, not CPP23's own tag) -- only
        # the fallback path queries CPP23's own specific tag directly.
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [cpp23_row] if tag in ("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_CPP23_X_2026-08-15-1100")
            else []
        )
        variant_previous_tag.return_value = "CMSSW_20_1_CPP23_X_2026-08-15-1100"

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        variant_previous_tag.assert_any_call("CMSSW_20_1_CPP23_X_2026-08-19-2300", "CPP23", "el9_amd64_gcc14")
        self.assertIn(
            "`CPP23` has no RelVal build at `CMSSW_20_1_X_2026-08-19-1100` -- compared against "
            "its own previous build `CMSSW_20_1_CPP23_X_2026-08-15-1100` instead.",
            markdown,
        )
        newly_idx = markdown.index("#### Newly failing RelVal workflows")
        resolved_idx = markdown.index("#### Resolved RelVal workflows")
        self.assertNotIn("140.051", markdown[newly_idx:resolved_idx])

    @patch("server.api_server.get_unittest_names", return_value=[])
    @patch("server.api_server.get_relval_workflow_ids", return_value=[])
    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.get_relval_exception_evidence", return_value=[])
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server._variant_previous_tag", return_value=None)
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_failing_unittests", return_value=[])
    @patch("server.api_server._build_and_addon_status", return_value={})
    def test_sub_ib_with_no_prior_build_at_all_is_not_reported_as_newly_failing(
        self, build_and_addon_status, get_failing_unittests, get_exit_code_names, arch, get_failing,
        variant_previous_tag, diff_w, recurrence, find_prs, search_c, get_relval_ids, get_ut_names,
    ):
        # A brand-new sub-IB (or its very first ever import) has no earlier build to
        # compare against at all. An empty "before" must NOT be silently treated as
        # "nothing was failing yet" (which would misreport every current failure as
        # new) -- per explicit product decision, this case is simply excluded from
        # newly-failing and named in the note instead.
        new_variant_row = _failing_row(workflow_id="1.0", name="BrandNewSubIbWorkflow", variant="G4ADEPT")
        new_variant_row["ib_tag"] = "CMSSW_20_1_G4ADEPT_X_2026-08-19-2300"
        # G4ADEPT is pulled in via parent_tag matching, so it's returned for the
        # PRIMARY latest_tag query, not its own tag.
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [new_variant_row] if tag == "CMSSW_20_1_X_2026-08-19-2300" else []
        )

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        self.assertIn(
            "`G4ADEPT` has no previous RelVal build imported, so its current failures "
            "aren't shown as newly failing.",
            markdown,
        )
        newly_idx = markdown.index("#### Newly failing RelVal workflows")
        resolved_idx = markdown.index("#### Resolved RelVal workflows")
        self.assertNotIn("BrandNewSubIbWorkflow", markdown[newly_idx:resolved_idx])


if __name__ == "__main__":
    unittest.main()


class NewlyFailingUnittestTableTests(unittest.TestCase):
    """The unit-test columns the Shift Console's unit-test cards read (2026-10-15)."""
    TAG = "CMSSW_20_1_ASAN_X_2026-10-05-2300"

    def _table(self, rows, details, history):
        from server.api_server import _newly_failing_unittest_table
        with patch("server.api_server.get_unittest_log_details", return_value=details), \
                patch("server.api_server.get_unittest_recurrence", return_value=history):
            return _newly_failing_unittest_table(rows, "el9_amd64_gcc14", "CMSSW_20_1_X_2026-10-05-2300")

    def test_failed_tests_without_an_error_in_the_log_get_no_exception(self):
        # live case: PrimaryVertex and testPVPlotting failed, the log has no error for them
        rows = [{"ib_tag": self.TAG, "variant": "ASAN", "name": "Alignment/OfflineValidation", "errors": 2, "ok": 24}]
        details = {(self.TAG, "el9_amd64_gcc14"): dict(imported=True, problems={
            "DQMServices/Demo/TestDQMCompareHistogram": ("python_traceback", "OSError: Failed to open file")},
            failed_tests=["Alignment/OfflineValidation/PrimaryVertex", "Alignment/OfflineValidation/testPVPlotting",
                          "DQMServices/Demo/TestDQMCompareHistogram"])}
        history = {("Alignment/OfflineValidation", "ASAN"): [
            (self.TAG, "FAILED"), ("CMSSW_20_1_ASAN_X_2026-10-02-2300", "FAILED"),
            ("CMSSW_19_0_ASAN_X_2026-09-01-2300", "FAILED"), ("CMSSW_20_1_ASAN_X_2026-09-28-2300", "PASSED")]}
        row = self._table(rows, details, history)[2]
        self.assertEqual(row, "| Alignment/OfflineValidation (ASAN) | 2 | 2 failed / 26 | PrimaryVertex<br>testPVPlotting |  | "
                              "Failed in 1 prior build(s) this series (most recent CMSSW_20_1_ASAN_X_2026-10-02-2300); "
                              "3/4 overall. |")

    def test_fixed_strings_caps_and_escaping(self):
        rows = [{"ib_tag": self.TAG, "variant": None, "name": "A/B", "errors": 10, "ok": None},
                {"ib_tag": "CMSSW_20_1_X_2026-10-05-2300", "variant": None, "name": "C/D", "errors": 1, "ok": 3}]
        tests = [f"A/B/t{i}" for i in range(10)]
        details = {
            (self.TAG, "el9_amd64_gcc14"): dict(imported=True, failed_tests=tests, problems={
                t: ("fatal_exception", "x | y\n" + "z" * 400) for t in tests}),
            ("CMSSW_20_1_X_2026-10-05-2300", "el9_amd64_gcc14"): dict(imported=True, failed_tests=["E/F/t"], problems={}),
        }
        lines = self._table(rows, details, {})
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", lines[2])[1:-1]]
        self.assertEqual(cells[2], "")  # no ok count: Tests left empty
        self.assertEqual(cells[3], "<br>".join([f"t{i}" for i in range(8)] + ["+2 more"]))
        exceptions = cells[4].split("<br>")
        self.assertEqual(len(exceptions), 4)
        self.assertEqual(exceptions[-1], "+7 more")
        self.assertTrue(exceptions[0].startswith("t0: x \\| y z"))
        self.assertTrue(exceptions[0].endswith("…"))
        self.assertLessEqual(len(exceptions[0].replace("\\|", "|")), 300)
        self.assertIn("| C/D | 1 | 1 failed / 4 | not found in the imported log |  | First failure in this series; 0/0 overall. |",
                      lines[3])
