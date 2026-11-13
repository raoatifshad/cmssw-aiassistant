"""ingest_github_patches: which files a patch names, and error pages saved as patches. No database."""
import tempfile
import unittest
from pathlib import Path

from build_graph import _import_raw_comment_batch
from ingest_github_patches import source_path, touched_files

PATCH = b"""From abc Mon Sep 17 00:00:00 2001
Subject: [PATCH 1/2] one

diff --git a/FWCore/A.cc b/FWCore/A.cc
+diff --git a/Fake/Inside.cc b/Fake/Inside.cc
From def Mon Sep 17 00:00:00 2001
Subject: [PATCH 2/2] two

diff --git a/FWCore/A.cc b/FWCore/A.cc
diff --git a/Old/B.h b/New/B.h
diff --git a/My Dir/c d.py b/My Dir/c d.py
"""


class TouchedFilesTests(unittest.TestCase):
    def parse(self, content: bytes):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "patch.txt"
            path.write_bytes(content)
            return touched_files(path)

    def test_distinct_new_paths_and_a_quoted_diff_is_not_a_file(self):
        self.assertEqual(self.parse(PATCH), ["FWCore/A.cc", "My Dir/c d.py", "New/B.h"])

    def test_error_pages_and_empty_files_are_unknown_not_zero(self):
        for page in (b"<html>\n<body>504</body>", b"<!DOCTYPE html>\n", b"error: too big or took too long to generate",
                     b"The server is unavailable at this time", b""):
            self.assertIsNone(self.parse(page), page)

    def test_a_real_patch_with_no_files_is_zero(self):
        self.assertEqual(self.parse(b"From abc Mon Sep 17 00:00:00 2001\nSubject: empty\n"), [])

    def test_source_path_matches_the_clang_import(self):
        self.assertEqual(source_path("FWCore/A.cc"), "src/FWCore/A.cc")
        self.assertEqual(source_path("src/FWCore/A.cc"), "src/FWCore/A.cc")


class CommentCapTests(unittest.TestCase):
    """The backup keeps only the first 100 comments of some long threads: such a file must not remove a longer copy."""
    def deleted_sources(self, count: int):
        calls = []

        class Session:
            def run(self, query, **params):
                calls.append((query, params))
                return []

        record = {"source_id": "github-comments:cmssw:PR:1", "source_path": "x", "content_hash": "h",
                  "comments": [{"key": f"cmssw:1:{i}"} for i in range(count)]}
        _import_raw_comment_batch(Session(), [record], False)
        return next(params["source_ids"] for query, params in calls if "DETACH DELETE" in query)

    def test_a_file_of_exactly_100_keeps_what_is_there(self):
        self.assertEqual(self.deleted_sources(100), [])

    def test_other_files_replace_the_old_comments(self):
        for count in (0, 99, 101):
            self.assertEqual(self.deleted_sources(count), ["github-comments:cmssw:PR:1"])


if __name__ == "__main__":
    unittest.main()
