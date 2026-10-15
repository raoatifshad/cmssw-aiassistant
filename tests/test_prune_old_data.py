"""ingestion/prune_old_data.py: the 90-day retention cleanup, and build_graph's refusal
to re-import a pruned source."""
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

import build_graph
import prune_old_data


class _Session:
    """Records every query; counts come from `counts` keyed by a substring of the MATCH."""

    def __init__(self, counts):
        self.counts = counts
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.queries.append((query, params))
        result = MagicMock()
        n = next((count for key, count in self.counts.items() if key in query), 0)
        result.single.return_value = {"n": n}
        return result


def _driver(session):
    driver = MagicMock()
    driver.session.return_value = session
    return driver


NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


class PruneTests(unittest.TestCase):
    def test_cutoff_is_compared_with_the_tag_timestamp(self):
        cutoff, cutoff_iso = prune_old_data.cutoffs(90, NOW)
        self.assertEqual(cutoff, "2026-07-17-1200")
        self.assertEqual(cutoff_iso, "2026-07-17T12:00:00")
        # tag timestamps compare correctly as strings
        self.assertLess("2026-07-16-2300", cutoff)
        self.assertGreater("2026-07-17-2300", cutoff)

    def test_dry_run_only_counts(self):
        session = _Session({"(n:LogProblem)": 120, "(n:TestLogRun)": 3})
        counts = prune_old_data.prune(_driver(session), 90, apply=False, now=NOW)
        self.assertEqual(counts["test-log problems"], 120)
        self.assertEqual(counts["test-log runs"], 3)
        self.assertNotIn("chat turns", counts)  # chat only with chat=True
        for query, _ in session.queries:
            self.assertNotIn("DELETE", query)
            self.assertNotIn("SET", query)

    def test_apply_marks_sources_pruned_before_deleting(self):
        session = _Session({"(n:LogProblem)": 120})
        prune_old_data.prune(_driver(session), 90, apply=True, now=NOW)
        queries = [query for query, _ in session.queries]
        mark = next(i for i, q in enumerate(queries) if "pruned_at" in q and "LogProblem" in q)
        delete = next(i for i, q in enumerate(queries) if "DETACH DELETE" in q and "LogProblem" in q)
        self.assertLess(mark, delete)
        self.assertIn("run.source_id", queries[mark])
        self.assertEqual(session.queries[delete][1]["cutoff"], "2026-07-17-1200")
        # nothing with a zero count is touched
        self.assertFalse(any("DELETE" in q and "RelValException" in q for q in queries))

    def test_never_deletes_what_must_be_kept(self):
        session = _Session({"(n:": 1})
        prune_old_data.prune(_driver(session), 90, apply=True, chat=True, now=NOW)
        deleted = " ".join(query for query, _ in session.queries if "DELETE" in query)
        for kept in ("ProblemFirstSeen", "ProblemSignature", "(n:IB", "(n:PR", "(n:Issue", "Concept",
                     "UserFeedback", "AlertRule", "(n:ImportState", "GitHubComment"):
            self.assertNotIn(kept, deleted)

    def test_shared_es_export_state_is_never_marked_pruned(self):
        session = _Session({"(n:RelValException)": 5})
        prune_old_data.prune(_driver(session), 90, apply=True, now=NOW)
        self.assertFalse(any("pruned_at" in q and "RelValException" in q for q, _ in session.queries))


class PrunedSourceNotReimportedTests(unittest.TestCase):
    def _driver(self, record):
        session = MagicMock()
        session.run.return_value.single.return_value = record
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        return driver

    def test_pruned_source_is_skipped_even_with_force(self):
        driver = self._driver({"content_hash": "old", "pruned": True})
        self.assertFalse(build_graph.source_needs_import(driver, "testlogs:x", "new"))
        self.assertFalse(build_graph.source_needs_import(driver, "testlogs:x", "new", force=True))

    def test_unpruned_source_behaves_as_before(self):
        driver = self._driver({"content_hash": "same", "pruned": False})
        self.assertFalse(build_graph.source_needs_import(driver, "s", "same"))
        self.assertTrue(build_graph.source_needs_import(driver, "s", "same", force=True))
        self.assertTrue(build_graph.source_needs_import(driver, "s", "changed"))
        self.assertTrue(build_graph.source_needs_import(self._driver(None), "s", "any"))


if __name__ == "__main__":
    unittest.main()
