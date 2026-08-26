import unittest
from unittest.mock import patch

from graph.tools import _ib_sort_key, diff_warnings, find_prs_before_ib, list_ib_tags


class FindPrsBeforeIbTests(unittest.TestCase):
    @patch("graph.tools.get_driver")
    def test_query_deduplicates_across_architecture_instances(self, get_driver):
        # ingest_merged_prs attaches MERGED_BEFORE to every architecture's IB
        # node sharing a tag (a PR merge is architecture-independent), so
        # without DISTINCT this query would return the same PR once per
        # imported architecture -- found live: a tag with 4 architectures
        # returned each of its 7 merged PRs 4 times (28 rows).
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"repo": "cmssw", "pr_number": 51750, "title": "x", "merged_at": None, "author": None}]

        find_prs_before_ib("CMSSW_20_1_X_2026-08-25-2300")

        query = session.run.call_args.args[0]
        self.assertIn("RETURN DISTINCT", query)

    @patch("graph.tools.get_driver")
    def test_orders_by_pr_number_ascending(self, get_driver):
        # merged_prs_summary.json carries no per-PR merge timestamp, so
        # p.merged_at is null for most of these rows -- ordering by it
        # produced an effectively arbitrary sequence (found live). PR number
        # ascending is a stable, always-populated substitute.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"repo": "cmssw", "pr_number": 51750, "title": "x", "merged_at": None, "author": None}]

        find_prs_before_ib("CMSSW_20_1_X_2026-08-25-2300")

        query = session.run.call_args.args[0]
        self.assertIn("ORDER BY p.number ASC", query)


class LatestIbRankingTests(unittest.TestCase):
    """Regression coverage for the release-series-vs-timestamp ranking bug:
    a higher release series (20_1) must outrank a lower one (20_0) even when
    the lower series' most recent build has a later wall-clock timestamp."""

    def test_sort_key_ranks_release_series_before_timestamp(self):
        newer_series_older_build = _ib_sort_key("CMSSW_20_1_X_2026-08-17-2300")
        older_series_newer_build = _ib_sort_key("CMSSW_20_0_X_2026-08-18-1100")
        self.assertGreater(newer_series_older_build, older_series_newer_build)

    def test_sort_key_uses_timestamp_as_tiebreak_within_a_series(self):
        older_build = _ib_sort_key("CMSSW_20_1_X_2026-08-17-1100")
        newer_build = _ib_sort_key("CMSSW_20_1_X_2026-08-17-2300")
        self.assertGreater(newer_build, older_build)

    def test_sort_key_handles_double_digit_minor_versions_numerically(self):
        # A plain string compare would rank "20_10" below "20_9" ('1' < '9');
        # the numeric key must not repeat that mistake.
        minor_ten = _ib_sort_key("CMSSW_20_10_X_2026-08-01-0000")
        minor_nine = _ib_sort_key("CMSSW_20_9_X_2026-08-01-0000")
        self.assertGreater(minor_ten, minor_nine)

    def test_sort_key_ignores_variant_suffix(self):
        primary = _ib_sort_key("CMSSW_20_1_X_2026-08-17-2300")
        variant = _ib_sort_key("CMSSW_20_1_UBSAN_X_2026-08-17-2300")
        self.assertEqual(primary, variant)

    @patch("graph.tools.get_driver")
    def test_list_ib_tags_ranks_newer_series_first_end_to_end(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"tag": "CMSSW_20_0_X_2026-08-18-1100", "arch": "el8_amd64_gcc13", "release": "CMSSW_20_0_X"},
            {"tag": "CMSSW_20_1_X_2026-08-17-2300", "arch": "el9_amd64_gcc14", "release": "CMSSW_20_1_X"},
            {"tag": "CMSSW_17_0_X_2026-08-18-1100", "arch": "el8_amd64_gcc13", "release": "CMSSW_17_0_X"},
        ]

        rows = list_ib_tags(limit=10)

        self.assertEqual(rows[0]["tag"], "CMSSW_20_1_X_2026-08-17-2300")
        get_driver.return_value.close.assert_called_once()

    @patch("graph.tools.get_driver")
    def test_list_ib_tags_does_not_truncate_before_sorting(self, get_driver):
        # _run_query's MAX_ROWS slice happens before a caller can sort; list_ib_tags
        # must bypass it or the true top entries can be dropped when there are more
        # rows than MAX_ROWS.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        many_old_rows = [
            {"tag": f"CMSSW_10_{i}_X_2026-01-01-0000", "arch": "el8_amd64_gcc13", "release": f"CMSSW_10_{i}_X"}
            for i in range(150)
        ]
        session.run.return_value = many_old_rows + [
            {"tag": "CMSSW_20_1_X_2026-08-17-2300", "arch": "el9_amd64_gcc14", "release": "CMSSW_20_1_X"},
        ]

        rows = list_ib_tags(limit=1)

        self.assertEqual(rows[0]["tag"], "CMSSW_20_1_X_2026-08-17-2300")


class DiffWarningsTests(unittest.TestCase):
    @patch("graph.tools.get_driver")
    def test_returns_new_resolved_and_changed_buckets_in_call_order(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        new_rows = [{"file": "a.cc", "line": 1, "flag": "f", "message": "m", "occurrences": 3}]
        resolved_rows = [{"file": "b.cc", "line": 2, "flag": "f", "message": "m", "occurrences": 1}]
        changed_rows = [{
            "file": "c.cc", "line": 3, "flag": "f", "message": "m",
            "before_count": 1, "after_count": 4,
        }]
        session.run.side_effect = [new_rows, resolved_rows, changed_rows]

        result = diff_warnings("before_tag", "after_tag", "el9_amd64_gcc14")

        self.assertEqual(result, {"new": new_rows, "resolved": resolved_rows, "changed": changed_rows})
        self.assertEqual(session.run.call_count, 3)


if __name__ == "__main__":
    unittest.main()
