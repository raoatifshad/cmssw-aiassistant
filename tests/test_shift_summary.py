import re
import unittest
from unittest.mock import patch

from server.api_server import (
    _diff_failing_relvals,
    _split_relval_diff,
    generate_shift_summary,
)


class DiffFailingRelvalsTests(unittest.TestCase):
    def test_detects_newly_failing_and_resolved(self):
        before = [{"workflow_id": "1.0", "name": "A", "errors": 1, "exitcode": 1}]
        after = [
            {"workflow_id": "1.0", "name": "A", "errors": 1, "exitcode": 1},
            {"workflow_id": "2.0", "name": "B", "errors": 2, "exitcode": 2},
        ]
        diff = _diff_failing_relvals(before, after)
        self.assertEqual([workflow_id for workflow_id, _, _ in diff], ["2.0"])

    def test_unchanged_workflow_is_excluded(self):
        row = {"workflow_id": "1.0", "name": "A", "errors": 1, "exitcode": 1}
        diff = _diff_failing_relvals([row], [dict(row)])
        self.assertEqual(diff, [])

    def test_changed_detail_still_failing_on_both_is_reported(self):
        before = [{"workflow_id": "1.0", "name": "A", "errors": 1, "exitcode": 1}]
        after = [{"workflow_id": "1.0", "name": "A", "errors": 5, "exitcode": 1}]
        diff = _diff_failing_relvals(before, after)
        self.assertEqual(len(diff), 1)
        workflow_id, before_row, after_row = diff[0]
        self.assertEqual(workflow_id, "1.0")
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

    def test_changed_detail_still_failing_appears_in_neither_list(self):
        before = [{"workflow_id": "1.0", "name": "A", "errors": 1, "exitcode": 1}]
        after = [{"workflow_id": "1.0", "name": "A", "errors": 5, "exitcode": 1}]
        diff = _diff_failing_relvals(before, after)

        newly_failing, resolved = _split_relval_diff(diff)

        self.assertEqual(newly_failing, [])
        self.assertEqual(resolved, [])


EMPTY_WARNINGS_DIFF = {"new": [], "resolved": [], "changed": []}


def _failing_row(workflow_id="140.51", name="RunMuonEG2022C", variant=None, errors=3, exitcode=16640):
    return {
        "workflow_id": workflow_id, "name": name, "variant": variant,
        "errors": errors, "exitcode": exitcode, "ib_tag": "CMSSW_20_1_X_2026-08-19-2300",
    }


class GenerateShiftSummaryTests(unittest.TestCase):
    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.find_prs_for_error", return_value=[])
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals", return_value=[])
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    def test_no_new_failures_renders_none_sections_and_no_evidence_block(self, *_mocks):
        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        self.assertIn("Newly failing RelVal workflows", markdown)
        self.assertNotIn("Newly failing workflow evidence", markdown)
        self.assertIn("relevance and historical pattern only", markdown)

    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.find_prs_for_error")
    @patch("server.api_server.get_relval_recurrence")
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    def test_new_failure_with_recurrence_and_pr_evidence(
        self, get_exit_code_names, arch, get_failing, diff_w, recurrence, find_prs, search_c
    ):
        new_row = _failing_row()
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [] if tag == "CMSSW_20_1_X_2026-08-19-1100" else [new_row]
        )
        recurrence.return_value = {
            "workflow_id": "140.51", "name": "RunMuonEG2022C",
            "total_appearances": 5, "failed_appearances": 3,
            "failed_tags": [
                "CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-18-2300", "CMSSW_19_0_X_2026-08-01-0000",
            ],
        }
        find_prs.return_value = [
            {"repo": "cmssw", "pr_number": 45213, "title": "fix", "evidence": ["warning_changed_file"]},
        ]

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        # Evidence is embedded as extra columns on the newly-failing table (the
        # frontend auto-hides anything past the first four columns behind a
        # per-row toggle), not a separate always-open section.
        self.assertIn("Recurrence | PR evidence | Stored knowledge", markdown)
        self.assertIn("`140.51` | RunMuonEG2022C", markdown)
        self.assertIn("cmssw #45213", markdown)
        self.assertIn("relevance only, do not prove a PR caused or fixed a failure", markdown)
        # the out-of-series tag must not be counted as an in-series recurrence
        self.assertIn("Failed in 2 prior build(s) this series", markdown)

    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.find_prs_for_error", return_value=[])
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    def test_new_failure_with_no_recurrence_and_no_pr_evidence(
        self, get_exit_code_names, arch, get_failing, diff_w, recurrence, find_prs, search_c
    ):
        new_row = _failing_row(workflow_id="999.0", name="BrandNewWorkflow")
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [] if tag == "CMSSW_20_1_X_2026-08-19-1100" else [new_row]
        )

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        self.assertIn("No prior appearance of this workflow ID in the imported graph", markdown)
        self.assertIn("No PR in the imported graph matches this workflow ID", markdown)

    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.find_prs_for_error", return_value=[])
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    def test_resolved_workflow_appears_only_in_resolved_table(
        self, get_exit_code_names, arch, get_failing, diff_w, recurrence, find_prs, search_c
    ):
        resolved_row = _failing_row(workflow_id="1.0", name="OldFlakyWorkflow")
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [resolved_row] if tag == "CMSSW_20_1_X_2026-08-19-1100" else []
        )

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        self.assertIn("OldFlakyWorkflow", markdown)
        self.assertNotIn("Newly failing workflow evidence", markdown)

    @patch("server.api_server.search_concepts", return_value=[])
    @patch("server.api_server.find_prs_for_error")
    @patch("server.api_server.get_relval_recurrence", return_value={"total_appearances": 0})
    @patch("server.api_server.diff_warnings", return_value=EMPTY_WARNINGS_DIFF)
    @patch("server.api_server.get_failing_relvals")
    @patch("server.api_server._ib_pair_architectures", return_value=["el9_amd64_gcc14"])
    @patch("server.api_server.get_exit_code_names", return_value={})
    def test_never_asserts_causality_for_pr_evidence(
        self, get_exit_code_names, arch, get_failing, diff_w, recurrence, find_prs, search_c
    ):
        new_row = _failing_row()
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [] if tag == "CMSSW_20_1_X_2026-08-19-1100" else [new_row]
        )
        find_prs.return_value = [
            {"repo": "cmssw", "pr_number": 1, "title": "t", "evidence": ["pr_title"]},
        ]

        markdown = generate_shift_summary("CMSSW_20_1_X_2026-08-19-2300", "CMSSW_20_1_X_2026-08-19-1100")

        # every "PR #" mention must sit in a sentence hedged with "not prove"/"relevance only",
        # never an unhedged "caused"/"fixed by" claim.
        for line in markdown.splitlines():
            if re.search(r"\bcmssw #\d+\b", line):
                self.assertTrue("relevance only" in line or "do not prove" in line)
                if "caused" in line:
                    self.assertIn("not prove", line)


if __name__ == "__main__":
    unittest.main()
