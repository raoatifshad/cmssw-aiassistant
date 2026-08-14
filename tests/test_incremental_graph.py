import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
from build_graph import (
    build_status_records,
    command_ib_tag,
    command_records,
    command_source_paths,
    discover_clang_builds,
    file_hash,
    github_issue_from_issue,
    github_pr_from_issue,
    github_reference_numbers,
    ib_key,
    is_unchanged,
    pr_cache_records,
    relval_default_records,
    relval_exception_records,
)
from link_pr_warnings import changed_files_from_patch


class IncrementalGraphTests(unittest.TestCase):
    def test_relval_default_records_exclude_specialized_architectures(self):
        records = relval_default_records({
            "CMSSW_20_1_X_2026-08-10-1100": (
                "el8_amd64_gcc13:gpu;amd_mi300x,"
                "el9_amd64_gcc13:gpu;amd_mi300x:other;rntuple,"
                "el9_amd64_gcc14"
            ),
            "CMSSW_13_0_X_2026-08-02-0000": "el8_amd64_gcc11,slc7_amd64_gcc11",
        })

        self.assertEqual(records, [
            {"tag": "CMSSW_20_1_X_2026-08-10-1100", "arch": "el9_amd64_gcc14"},
            {"tag": "CMSSW_13_0_X_2026-08-02-0000", "arch": "el8_amd64_gcc11"},
            {"tag": "CMSSW_13_0_X_2026-08-02-0000", "arch": "slc7_amd64_gcc11"},
        ])

    def test_command_source_paths_select_latest_snapshot_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "el9_amd64_gcc13" / "2026-08-12-1100" / "CMSSW_20_1_X.json"
            older = root / "el9_amd64_gcc13" / "2026-08-11-2300" / "CMSSW_20_1_X.json"
            separate_flavor = root / "el9_amd64_gcc13" / "2026-08-11-2300" / "other" / "rntuple" / "CMSSW_20_1_X.json"
            for path in (current, older, separate_flavor):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")

            self.assertEqual(command_source_paths(root), [separate_flavor, current])
            self.assertEqual(command_source_paths(root, include_history=True), [older, separate_flavor, current])
            self.assertEqual(command_ib_tag(root, current), "CMSSW_20_1_X_2026-08-12-1100")

    def test_domain_command_records_keep_workflow_provenance(self):
        records = command_records(
            {"37634.21-1": "1c345a9ac6bf59e9", "not-a-workflow": "ignored"},
            "CMSSW_20_1_X_2026-08-11-2300",
            "el9_amd64_gcc13",
        )

        self.assertEqual(records, [{
            "key": "CMSSW_20_1_X_2026-08-11-2300:el9_amd64_gcc13:37634.21-1",
            "relval_key": "CMSSW_20_1_X_2026-08-11-2300:el9_amd64_gcc13:37634.21",
            "workflow_id": "37634.21",
            "workflow_step": "37634.21-1",
            "step": 1,
            "command_hash": "1c345a9ac6bf59e9",
            "tag": "CMSSW_20_1_X_2026-08-11-2300",
            "arch": "el9_amd64_gcc13",
        }])

    def test_pr_cache_records_preserve_only_supplied_metadata(self):
        records = pr_cache_records({"51668": {
            "hash": "1234567890abcdef1234567890abcdef12345678",
            "title": "A supplied title",
            "author": "author",
            "branch": "CMSSW_20_1_X",
            "user_ref": "topic",
        }})

        self.assertEqual(records[0]["number"], 51668)
        self.assertEqual(records[0]["sha"], "1234567890abcdef1234567890abcdef12345678")
        self.assertEqual(records[0]["branch"], "CMSSW_20_1_X")

    def test_ib_key_preserves_architecture(self):
        tag = "CMSSW_20_1_X_2026-08-03-2300"
        self.assertEqual(ib_key(tag, "el9_amd64_gcc13"), f"{tag}:el9_amd64_gcc13")
        self.assertNotEqual(
            ib_key(tag, "el9_amd64_gcc13"),
            ib_key(tag, "el9_amd64_gcc14"),
        )

    def test_hash_decision_skips_only_unchanged_source(self):
        self.assertTrue(is_unchanged("same", "same"))
        self.assertFalse(is_unchanged("old", "new"))
        self.assertFalse(is_unchanged("same", "same", force=True))

    def test_file_hash_changes_when_contents_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text('{"value": 1}')
            first = file_hash(path)
            path.write_text('{"value": 2}')
            self.assertNotEqual(first, file_hash(path))

    def test_discovers_clang_builds_from_separate_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "CMSSW_20_1_X_2026-08-03-2300" / "el9_amd64_gcc13" / "build-logs"
            second = root / "CMSSW_20_1_X_2026-08-04-2300" / "el9_amd64_gcc14" / "build-logs"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (root / "not-a-build").mkdir()

            builds = discover_clang_builds(root)

        self.assertEqual(
            builds,
            [
                ("CMSSW_20_1_X_2026-08-03-2300", "el9_amd64_gcc13", first),
                ("CMSSW_20_1_X_2026-08-04-2300", "el9_amd64_gcc14", second),
            ],
        )

    def test_github_issue_normalizes_only_pull_requests(self):
        issue = {
            "number": 51633,
            "title": "Move Run 4 default to D127",
            "body": "description",
            "labels": [{"name": "fully-signed"}],
            "user": {"login": "kpedro88"},
            "milestone": {"title": "CMSSW_20_1_X"},
            "pull_request": {"merged_at": "2026-08-10T06:47:43Z"},
            "state": "closed",
        }

        pr = github_pr_from_issue(issue)

        self.assertEqual(pr["number"], 51633)
        self.assertEqual(pr["labels"], ["fully-signed"])
        self.assertEqual(pr["author"], "kpedro88")
        self.assertEqual(pr["merged_at"], "2026-08-10T06:47:43Z")
        self.assertEqual(pr["state"], "closed")
        self.assertIsNone(github_pr_from_issue({"number": 51634}))

    def test_github_issue_normalizes_only_non_pull_requests(self):
        issue = {
            "number": 51634,
            "title": "Regression in geometry workflow",
            "state": "open",
            "labels": [{"name": "bug"}],
            "user": {"login": "reporter"},
        }

        normalized = github_issue_from_issue(issue)

        self.assertEqual(normalized["number"], 51634)
        self.assertEqual(normalized["state"], "open")
        self.assertEqual(normalized["labels"], ["bug"])
        self.assertIsNone(github_issue_from_issue({**issue, "pull_request": {}}))

    def test_github_reference_numbers_are_unique(self):
        self.assertEqual(
            github_reference_numbers(
                "Followup to #51601; also see #51601, #51620, and "
                "https://github.com/cms-sw/cmssw/pull/51562#issuecomment-1."
            ),
            [51562, 51601, 51620],
        )

    def test_relval_exception_records_preserve_exception_workflow_mapping(self):
        records = relval_exception_records({
            "An exception of category 'NoSecondaryFiles' occurred while": ["37834.0", 140.051],
        }, "CMSSW_20_1_X_2026-08-12-1100", "el9_amd64_gcc13")

        self.assertEqual([record["workflow_id"] for record in records], ["37834.0", "140.051"])
        self.assertTrue(all(record["exception_hash"] for record in records))
        self.assertTrue(all(record["exception"].startswith("An exception") for record in records))

    def test_build_status_records_keep_architecture_and_warning_totals(self):
        records = build_status_records([{
            "release_name": "CMSSW_20_1_DBG_X_2026-08-07-2300",
            "builds": [{
                "arch": "el9_amd64_gcc13",
                "passed": "warning",
                "details": {"compWarning": 1, "ignoreWarning": 2117},
            }],
        }])

        self.assertEqual(records, [{
            "tag": "CMSSW_20_1_DBG_X_2026-08-07-2300",
            "arch": "el9_amd64_gcc13",
            "release": "CMSSW_20_1_DBG_X",
            "status": "warning",
            "comp_warnings": 1,
            "ignored_warnings": 2117,
        }])

    def test_patch_parser_keeps_renamed_and_deleted_paths(self):
        patch = (
            "diff --git a/Foo/old.py b/Foo/new.py\n"
            "diff --git a/Bar/deleted.py b/Bar/deleted.py\n"
        )

        self.assertEqual(changed_files_from_patch(patch), ["Foo/new.py", "Bar/deleted.py"])


if __name__ == "__main__":
    unittest.main()
