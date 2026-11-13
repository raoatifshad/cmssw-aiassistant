"""Problems found by item G2 of the weekend test (2026-11-12): pass 1 against pass 2 of the 2,803 earlier items."""
import unittest
from unittest.mock import patch

from server import api_server

NOTE = ("GitHub data: the full export (with comments) reaches 2026-11-06; Issues/PRs updated since were fetched directly from GitHub on 2026-11-06 "
        "(titles, descriptions, labels, states; no comments), newest Issue #52407, last update 2026-11-06. Anything changed on GitHub after that is not imported")


class ReportedAfterGithubDataTests(unittest.TestCase):
    """G2-2: "is there an issue for it?" for a workflow failing on days the GitHub data does not reach."""

    def test_line_only_when_the_failure_is_newer_than_the_data(self):
        line = api_server._failing_after_github_data("exit 16640 (OtherCMS) in 21 run(s) in `CMSSW_20_1_X` on 2026-11-10", NOTE)
        self.assertIn("reaches 2026-11-06", line)
        self.assertIn("failing on 2026-11-10", line)
        self.assertIn("not imported", line)
        self.assertEqual(api_server._failing_after_github_data("exit 31744 (TimeOut) in 11 run(s) in `CMSSW_20_1_X` on 2026-11-06", NOTE), "")
        self.assertEqual(api_server._failing_after_github_data("", NOTE), "")
        self.assertEqual(api_server._failing_after_github_data("exit 1 in 2 run(s) in `CMSSW_20_1_X` on 2026-11-10", "an export that may be weeks old"), "")
        older = "GitHub Issues are imported from an export that reaches #52407 (2026-11-06); anything newer on GitHub is not imported"
        self.assertIn("reaches 2026-11-06", api_server._failing_after_github_data("exit 1 in 2 run(s) in `CMSSW_20_1_X` on 2026-11-09", older))

    def _answer(self, named, checked, exits):
        with patch.object(api_server, "_workflow_named_items", return_value=named), \
             patch.object(api_server, "_workflow_current_errors", return_value=[]), \
             patch.object(api_server, "_workflow_current_exits", return_value=exits), \
             patch.object(api_server, "_github_export_note", return_value=NOTE), \
             patch.object(api_server, "_ai_checked_matches", return_value=checked):
            return api_server._workflow_github_reports_answer("36434.0", "is there an issue for it?")

    def test_no_report_answer_says_the_data_ends_before_the_failure(self):
        item = {"kind": "Issue", "repo": "cmssw", "number": 50665, "title": "[UBSAN] Wrong geometry casting", "state": "closed", "merged_at": None,
                "created_at": "2026-04-05T00:00:00Z", "author": "someone", "place": "comments"}
        new = "exit 16640 (OtherCMS) in 21 run(s) in `CMSSW_20_1_X` on 2026-11-10"
        answer = self._answer([item], {"Issue #50665": {"verdict": "related", "reason": ""}}, new)
        first, second = answer.splitlines()[:2]
        self.assertTrue(first.startswith("**No report of this same problem found.**"))
        self.assertIn("Our GitHub data reaches 2026-11-06 and this workflow is failing on 2026-11-10", second)
        # nothing found at all: the same line
        self.assertIn("Our GitHub data reaches 2026-11-06", self._answer([], None, new))
        # a report was found, or the failure is inside the data: no such line
        self.assertNotIn("Our GitHub data reaches", self._answer([item], {"Issue #50665": {"verdict": "same", "reason": ""}}, new))
        old = "exit 31744 (TimeOut) in 11 run(s) in `CMSSW_20_1_X` on 2026-11-05"
        self.assertNotIn("Our GitHub data reaches", self._answer([item], {"Issue #50665": {"verdict": "related", "reason": ""}}, old))


def _rec(variant, arch, failed=(), unit_failed=(), relvals=954, build="passed"):
    return {"tag": "T", "arch": arch, "variant": variant, "build": build, "relvals": relvals, "failed": list(failed), "units": 390, "unit_failed": list(unit_failed)}


class FlavourRuleScopeTests(unittest.TestCase):
    """G2-3: the flavours-of-a-build rule answered a question about one workflow, and "which flavours are clean?" got the catalogue of the series."""
    RECORDS = [_rec("primary", "el8_amd64_gcc14", ["539.0"]), _rec("primary", "el9_amd64_gcc14", ["537.0", "539.0"]), _rec("ASAN", "el9_amd64_gcc14", ["539.0"], ["A/B"]),
               _rec("CLANG", "el9_amd64_gcc14"), _rec("DBG", "el9_amd64_gcc14", relvals=0), _rec("ROOT6", "el9_amd64_gcc14", ["10804.31"])]

    def _ask(self, question):
        with patch.object(api_server, "_ib_build_records", return_value=self.RECORDS):
            return api_server._flavours_of_build_answer(question)

    def test_workflow_counted_per_flavour(self):
        answer = self._ask("how many flavours does 539.0 fail in in CMSSW_20_1_X_2026-11-04-2300?")
        self.assertIn("Workflow `539.0` fails in **2 flavour(s)** of `CMSSW_20_1_X_2026-11-04-2300`, of 5 imported", answer)
        self.assertIn("primary (`el8_amd64_gcc14`, `el9_amd64_gcc14`), ASAN (`el9_amd64_gcc14`)", answer)
        self.assertNotIn("ROOT6", answer)
        self.assertIn("fails in **none** of the 5", self._ask("how many flavours does 1.0 fail in in CMSSW_20_1_X_2026-11-04-2300?"))
        # nothing to say about a workflow that is not a failure count: left to the other rules
        self.assertIsNone(self._ask("how many flavours does 539.0 run in in CMSSW_20_1_X_2026-11-04-2300?"))

    def test_clean_flavours_of_one_build(self):
        answer = self._ask("which flavours of CMSSW_20_1_X_2026-11-04-2300 are clean?")
        self.assertIn("**1 of the 5 imported flavour(s)** of `CMSSW_20_1_X_2026-11-04-2300`", answer)
        self.assertIn("fail nothing: `CLANG`.", answer)
        self.assertIn("No RelVal results imported for part of: `DBG` (not counted as clean)", answer)
        self.assertIn("- ASAN `el9_amd64_gcc14`: 1 RelVal(s), 1 unit test(s)", answer)
        with patch.object(api_server, "_ib_build_records", return_value=self.RECORDS[:3]):
            self.assertIn("**No flavour of `CMSSW_20_1_X_2026-11-04-2300` is clean**", api_server._flavours_of_build_answer("which sub-IBs are green in CMSSW_20_1_X_2026-11-04-2300?"))

    def test_plain_count_unchanged(self):
        self.assertIn("**4 sub-IB flavour(s)**", self._ask("how many sub-IBs does CMSSW_20_1_X_2026-11-04-2300 have?"))


if __name__ == "__main__":
    unittest.main()
