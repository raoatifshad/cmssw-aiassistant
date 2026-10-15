import unittest
from unittest.mock import patch

from graph.tools import (
    _ib_sort_key, diff_warnings, find_prs_before_ib, find_prs_for_error, find_prs_merged_between, list_ib_tags,
    latest_and_previous_primary_tags, variant_previous_tag, find_relval_regressions,
    find_unittest_regressions, find_build_regressions, single_ib_status, search_shifter_docs,
)


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
    def test_orders_newest_merged_at_first_with_undated_prs_last(self, get_driver):
        # Most PRs do have a real merged_at (checked live: 39,005 of 45,904),
        # so ordering must actually use it, newest first -- but a real
        # minority have none (merged_prs_summary.json itself carries no
        # per-PR timestamp, so a PR missing one from its own separate import
        # falls back to null), and a plain `ORDER BY p.merged_at DESC` would
        # put those at the very TOP: openCypher treats null as the largest
        # value, so DESC surfaces it first. `p.merged_at IS NULL` as the
        # primary sort key pushes undated rows to the end instead.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"repo": "cmssw", "pr_number": 51750, "title": "x", "merged_at": None, "author": None}]

        find_prs_before_ib("CMSSW_20_1_X_2026-08-25-2300")

        query = session.run.call_args.args[0]
        self.assertIn("ORDER BY p.merged_at IS NULL, p.merged_at DESC", query)


class LatestAndPreviousPrimaryTagsTests(unittest.TestCase):
    """Moved here from server/api_server.py (server.api_server._latest_and_previous_ib_tags
    now delegates to this) so the standalone alerts module can resolve the same
    "current default comparison window" without depending on the server module."""

    @patch("graph.tools.get_driver")
    def test_picks_the_newest_release_series_two_most_recent_timestamps(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"tag": "CMSSW_20_1_X_2026-09-06-2300"},
            {"tag": "CMSSW_20_1_X_2026-09-07-2300"},
            {"tag": "CMSSW_20_1_X_2026-09-08-2300"},
            {"tag": "CMSSW_20_0_X_2026-09-08-2300"},  # older series, must not win
        ]

        result = latest_and_previous_primary_tags("el9_amd64_gcc14")

        self.assertEqual(result, ("CMSSW_20_1_X_2026-09-08-2300", "CMSSW_20_1_X_2026-09-07-2300"))

    @patch("graph.tools.get_driver")
    def test_returns_none_when_fewer_than_two_builds_in_the_newest_series(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"tag": "CMSSW_20_1_X_2026-09-08-2300"}]

        self.assertIsNone(latest_and_previous_primary_tags("el9_amd64_gcc14"))

    @patch("graph.tools.get_driver")
    def test_returns_none_when_nothing_imported(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = []

        self.assertIsNone(latest_and_previous_primary_tags())


class VariantPreviousTagTests(unittest.TestCase):
    """Moved here from server/api_server.py (server.api_server._variant_previous_tag
    now delegates to this)."""

    @patch("graph.tools.get_driver")
    def test_scopes_to_same_release_series_and_same_variant(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {"tag": "CMSSW_20_1_CPP23_X_2026-08-27-1100"}

        result = variant_previous_tag("CMSSW_20_1_CPP23_X_2026-09-01-1100", "CPP23", "el9_amd64_gcc14")

        self.assertEqual(result, "CMSSW_20_1_CPP23_X_2026-08-27-1100")
        query, params = session.run.call_args.args[0], session.run.call_args.kwargs
        self.assertIn("ib.tag STARTS WITH $series_prefix", query)
        self.assertIn("ib.tag < $current_tag", query)
        self.assertEqual(params["series_prefix"], "CMSSW_20_1_")
        self.assertEqual(params["variant"], "CPP23")

    @patch("graph.tools.get_driver")
    def test_returns_none_when_no_earlier_build_exists(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = None

        self.assertIsNone(variant_previous_tag("CMSSW_20_1_CPP23_X_2026-07-28-1100", "CPP23", "el9_amd64_gcc14"))


class FindRelvalRegressionsTests(unittest.TestCase):
    """find_relval_regressions backs the relval_regression alert rule type -- the
    first alert that fires on an actual regression since the last build, rather
    than a currently-failing count against one fixed, pre-configured tag."""

    @patch("graph.tools.variant_previous_tag")
    @patch("graph.tools.get_failing_relvals")
    @patch("graph.tools.get_driver")
    @patch("graph.tools.latest_and_previous_primary_tags")
    def test_reports_no_baseline_when_fewer_than_two_builds_imported(
        self, latest_and_previous, get_driver, get_failing, variant_previous,
    ):
        latest_and_previous.return_value = None

        result = find_relval_regressions()

        self.assertEqual(result, {"latest_tag": None, "previous_tag": None, "newly_failing": []})
        get_failing.assert_not_called()

    @patch("graph.tools.get_relval_workflow_ids")
    @patch("graph.tools.variant_previous_tag")
    @patch("graph.tools.get_failing_relvals")
    @patch("graph.tools.get_driver")
    @patch("graph.tools.latest_and_previous_primary_tags")
    def test_primary_uses_the_fixed_previous_tag_no_fallback_needed(
        self, latest_and_previous, get_driver, get_failing, variant_previous, get_relval_ids,
    ):
        # Primary exists at previous_tag with real proof (get_relval_workflow_ids,
        # any status) even though it happened to have zero failures there --
        # variant_previous_tag must never be consulted for it. Checking existence
        # via get_failing_relvals's failing-only rows instead would wrongly treat
        # this zero-failure baseline as "no exact match" and trigger a needless
        # fallback (the exact bug this test caught during development).
        latest_and_previous.return_value = ("CMSSW_20_1_X_2026-09-08-2300", "CMSSW_20_1_X_2026-09-07-2300")
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"arch": "el9_amd64_gcc14"}]
        get_relval_ids.return_value = [{"variant": None, "workflow_id": "0.0"}]
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [{"ib_tag": tag, "variant": None, "workflow_id": "999.0", "name": "NewOne"}]
            if tag == "CMSSW_20_1_X_2026-09-08-2300" else []
        )

        result = find_relval_regressions("el9_amd64_gcc14")

        self.assertEqual(result["latest_tag"], "CMSSW_20_1_X_2026-09-08-2300")
        self.assertEqual(
            result["newly_failing"],
            [{"variant": "primary", "arch": "el9_amd64_gcc14", "workflow_id": "999.0", "name": "NewOne"}],
        )
        variant_previous.assert_not_called()

    @patch("graph.tools.get_relval_workflow_ids", return_value=[])
    @patch("graph.tools.variant_previous_tag")
    @patch("graph.tools.get_failing_relvals")
    @patch("graph.tools.get_driver")
    @patch("graph.tools.latest_and_previous_primary_tags")
    def test_sub_ib_missing_at_previous_tag_falls_back_to_its_own_previous_build(
        self, latest_and_previous, get_driver, get_failing, variant_previous, get_relval_ids,
    ):
        latest_and_previous.return_value = ("CMSSW_20_1_X_2026-09-08-1100", "CMSSW_20_1_X_2026-09-07-2300")
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"arch": "el9_amd64_gcc14"}]
        variant_previous.return_value = "CMSSW_20_1_CPP23_X_2026-09-03-1100"

        def get_failing_side_effect(tag, arch=None, limit=None):
            if tag == "CMSSW_20_1_X_2026-09-08-1100":
                return [{
                    "ib_tag": "CMSSW_20_1_CPP23_X_2026-09-08-1100", "variant": "CPP23",
                    "workflow_id": "140.051", "name": "RunMuonEG2022C",
                }]
            if tag == "CMSSW_20_1_CPP23_X_2026-09-03-1100":
                return [{"ib_tag": tag, "variant": "CPP23", "workflow_id": "140.051", "name": "RunMuonEG2022C"}]
            return []  # previous_tag itself has no CPP23 row at all

        get_failing.side_effect = get_failing_side_effect

        result = find_relval_regressions("el9_amd64_gcc14")

        variant_previous.assert_called_once_with(
            "CMSSW_20_1_CPP23_X_2026-09-08-1100", "CPP23", "el9_amd64_gcc14"
        )
        # Already failing in CPP23's own previous build -- not newly failing.
        self.assertEqual(result["newly_failing"], [])

    @patch("graph.tools.get_relval_workflow_ids", return_value=[])
    @patch("graph.tools.variant_previous_tag", return_value=None)
    @patch("graph.tools.get_failing_relvals")
    @patch("graph.tools.get_driver")
    @patch("graph.tools.latest_and_previous_primary_tags")
    def test_sub_ib_with_no_earlier_build_at_all_is_not_reported(
        self, latest_and_previous, get_driver, get_failing, variant_previous, get_relval_ids,
    ):
        latest_and_previous.return_value = ("CMSSW_20_1_X_2026-09-08-1100", "CMSSW_20_1_X_2026-09-07-2300")
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"arch": "el9_amd64_gcc14"}]
        get_failing.side_effect = lambda tag, arch=None, limit=None: (
            [{"ib_tag": "CMSSW_20_1_G4ADEPT_X_2026-09-08-1100", "variant": "G4ADEPT",
              "workflow_id": "1.0", "name": "BrandNew"}]
            if tag == "CMSSW_20_1_X_2026-09-08-1100" else []
        )

        result = find_relval_regressions("el9_amd64_gcc14")

        self.assertEqual(result["newly_failing"], [])


class FindUnittestRegressionsTests(unittest.TestCase):
    """Same policy as FindRelvalRegressionsTests, for UnitTest failures."""

    @patch("graph.tools.variant_previous_tag")
    @patch("graph.tools.get_unittest_names")
    @patch("graph.tools.get_failing_unittests")
    @patch("graph.tools.get_driver")
    @patch("graph.tools.latest_and_previous_primary_tags")
    def test_reports_no_baseline_when_fewer_than_two_builds_imported(
        self, latest_and_previous, get_driver, get_failing, get_ut_names, variant_previous,
    ):
        latest_and_previous.return_value = None

        result = find_unittest_regressions()

        self.assertEqual(result, {"latest_tag": None, "previous_tag": None, "newly_failing": []})
        get_failing.assert_not_called()

    @patch("graph.tools.variant_previous_tag")
    @patch("graph.tools.get_unittest_names")
    @patch("graph.tools.get_failing_unittests")
    @patch("graph.tools.get_driver")
    @patch("graph.tools.latest_and_previous_primary_tags")
    def test_primary_zero_failure_baseline_is_not_a_fallback_case(
        self, latest_and_previous, get_driver, get_failing, get_ut_names, variant_previous,
    ):
        # Same ambiguity find_relval_regressions guards against: primary existing
        # at previous_tag with zero failures must not look like "doesn't exist".
        latest_and_previous.return_value = ("CMSSW_20_1_X_2026-09-08-2300", "CMSSW_20_1_X_2026-09-07-2300")
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"arch": "el9_amd64_gcc14"}]
        get_ut_names.return_value = [{"variant": None, "name": "SomeTest"}]
        get_failing.side_effect = lambda tag, arch=None, limit=None, include_sub_ibs=False: (
            [{"ib_tag": tag, "variant": None, "name": "NewlyFailingTest"}]
            if tag == "CMSSW_20_1_X_2026-09-08-2300" else []
        )

        result = find_unittest_regressions("el9_amd64_gcc14")

        self.assertEqual(
            result["newly_failing"],
            [{"variant": "primary", "arch": "el9_amd64_gcc14", "name": "NewlyFailingTest"}],
        )
        variant_previous.assert_not_called()

    @patch("graph.tools.variant_previous_tag")
    @patch("graph.tools.get_unittest_names", return_value=[])
    @patch("graph.tools.get_failing_unittests")
    @patch("graph.tools.get_driver")
    @patch("graph.tools.latest_and_previous_primary_tags")
    def test_sub_ib_missing_at_previous_tag_falls_back_to_its_own_previous_build(
        self, latest_and_previous, get_driver, get_failing, get_ut_names, variant_previous,
    ):
        latest_and_previous.return_value = ("CMSSW_20_1_X_2026-09-08-1100", "CMSSW_20_1_X_2026-09-07-2300")
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"arch": "el9_amd64_gcc14"}]
        variant_previous.return_value = "CMSSW_20_1_UBSAN_X_2026-09-03-1100"

        def get_failing_side_effect(tag, arch=None, limit=None, include_sub_ibs=False):
            if tag == "CMSSW_20_1_X_2026-09-08-1100":
                return [{"ib_tag": "CMSSW_20_1_UBSAN_X_2026-09-08-1100", "variant": "UBSAN", "name": "FlakyTest"}]
            if tag == "CMSSW_20_1_UBSAN_X_2026-09-03-1100":
                return [{"ib_tag": tag, "variant": "UBSAN", "name": "FlakyTest"}]
            return []

        get_failing.side_effect = get_failing_side_effect

        result = find_unittest_regressions("el9_amd64_gcc14")

        variant_previous.assert_called_once_with(
            "CMSSW_20_1_UBSAN_X_2026-09-08-1100", "UBSAN", "el9_amd64_gcc14"
        )
        self.assertEqual(result["newly_failing"], [])


class SingleIbStatusTests(unittest.TestCase):
    @patch("graph.tools.get_driver")
    def test_returns_build_and_addon_status_for_the_exact_tag(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {"build_status": "error", "addon_status": "passed"}

        result = single_ib_status("CMSSW_20_1_CLANG_X_2026-09-06-2300", "el9_amd64_gcc14")

        self.assertEqual(result, {"build_status": "error", "addon_status": "passed"})

    @patch("graph.tools.get_driver")
    def test_returns_none_when_no_such_ib_node(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = None

        self.assertIsNone(single_ib_status("CMSSW_20_1_CLANG_X_2026-01-01-0000", "el9_amd64_gcc14"))


class FindBuildRegressionsTests(unittest.TestCase):
    """Covers BOTH Build and AddOn transitions in one check, since both are the
    same "single flag per architecture+variant" shape."""

    @patch("graph.tools.variant_previous_tag")
    @patch("graph.tools.single_ib_status")
    @patch("graph.tools.get_driver")
    @patch("graph.tools.latest_and_previous_primary_tags")
    def test_reports_no_baseline_when_fewer_than_two_builds_imported(
        self, latest_and_previous, get_driver, single_ib, variant_previous,
    ):
        latest_and_previous.return_value = None

        result = find_build_regressions()

        self.assertEqual(result, {"latest_tag": None, "previous_tag": None, "newly_failing": []})

    @patch("graph.tools.variant_previous_tag")
    @patch("graph.tools.single_ib_status")
    @patch("graph.tools.get_driver")
    @patch("graph.tools.latest_and_previous_primary_tags")
    def test_primary_build_flip_to_error_is_a_regression(
        self, latest_and_previous, get_driver, single_ib, variant_previous,
    ):
        latest_and_previous.return_value = ("CMSSW_20_1_X_2026-09-08-2300", "CMSSW_20_1_X_2026-09-07-2300")
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"tag": "CMSSW_20_1_X_2026-09-07-2300", "arch": "el9_amd64_gcc14", "variant": "",
             "parent_tag": None, "build_status": "passed", "addon_status": "passed"},
            {"tag": "CMSSW_20_1_X_2026-09-08-2300", "arch": "el9_amd64_gcc14", "variant": "",
             "parent_tag": None, "build_status": "error", "addon_status": "passed"},
        ]

        result = find_build_regressions("el9_amd64_gcc14")

        self.assertEqual(
            result["newly_failing"],
            [{"variant": "primary", "arch": "el9_amd64_gcc14", "kind": "build"}],
        )
        variant_previous.assert_not_called()

    @patch("graph.tools.variant_previous_tag")
    @patch("graph.tools.single_ib_status")
    @patch("graph.tools.get_driver")
    @patch("graph.tools.latest_and_previous_primary_tags")
    def test_staying_in_error_is_not_a_new_regression(
        self, latest_and_previous, get_driver, single_ib, variant_previous,
    ):
        latest_and_previous.return_value = ("CMSSW_20_1_X_2026-09-08-2300", "CMSSW_20_1_X_2026-09-07-2300")
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"tag": "CMSSW_20_1_X_2026-09-07-2300", "arch": "el9_amd64_gcc14", "variant": "",
             "parent_tag": None, "build_status": "error", "addon_status": "passed"},
            {"tag": "CMSSW_20_1_X_2026-09-08-2300", "arch": "el9_amd64_gcc14", "variant": "",
             "parent_tag": None, "build_status": "error", "addon_status": "passed"},
        ]

        result = find_build_regressions("el9_amd64_gcc14")

        self.assertEqual(result["newly_failing"], [])

    @patch("graph.tools.variant_previous_tag")
    @patch("graph.tools.single_ib_status")
    @patch("graph.tools.get_driver")
    @patch("graph.tools.latest_and_previous_primary_tags")
    def test_sub_ib_missing_at_previous_tag_falls_back_to_its_own_previous_build(
        self, latest_and_previous, get_driver, single_ib, variant_previous,
    ):
        latest_and_previous.return_value = ("CMSSW_20_1_X_2026-09-08-1100", "CMSSW_20_1_X_2026-09-07-2300")
        session = get_driver.return_value.session.return_value.__enter__.return_value
        # CLANG has no row at all for previous_tag (CMSSW_20_1_X_2026-09-07-2300).
        session.run.return_value = [
            {"tag": "CMSSW_20_1_CLANG_X_2026-09-08-1100", "arch": "el9_amd64_gcc14", "variant": "CLANG",
             "parent_tag": "CMSSW_20_1_X_2026-09-08-1100", "build_status": "error", "addon_status": "passed"},
        ]
        variant_previous.return_value = "CMSSW_20_1_CLANG_X_2026-09-06-2300"
        single_ib.return_value = {"build_status": "passed", "addon_status": "passed"}

        result = find_build_regressions("el9_amd64_gcc14")

        variant_previous.assert_called_once_with(
            "CMSSW_20_1_CLANG_X_2026-09-08-1100", "CLANG", "el9_amd64_gcc14"
        )
        single_ib.assert_called_once_with("CMSSW_20_1_CLANG_X_2026-09-06-2300", "el9_amd64_gcc14")
        self.assertEqual(
            result["newly_failing"],
            [{"variant": "CLANG", "arch": "el9_amd64_gcc14", "kind": "build"}],
        )

    @patch("graph.tools.variant_previous_tag", return_value=None)
    @patch("graph.tools.single_ib_status")
    @patch("graph.tools.get_driver")
    @patch("graph.tools.latest_and_previous_primary_tags")
    def test_addon_flip_is_detected_independently_of_build(
        self, latest_and_previous, get_driver, single_ib, variant_previous,
    ):
        latest_and_previous.return_value = ("CMSSW_20_1_X_2026-09-08-2300", "CMSSW_20_1_X_2026-09-07-2300")
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"tag": "CMSSW_20_1_X_2026-09-07-2300", "arch": "el9_amd64_gcc14", "variant": "",
             "parent_tag": None, "build_status": "passed", "addon_status": "passed"},
            {"tag": "CMSSW_20_1_X_2026-09-08-2300", "arch": "el9_amd64_gcc14", "variant": "",
             "parent_tag": None, "build_status": "passed", "addon_status": "error"},
        ]

        result = find_build_regressions("el9_amd64_gcc14")

        self.assertEqual(
            result["newly_failing"],
            [{"variant": "primary", "arch": "el9_amd64_gcc14", "kind": "addon"}],
        )


class FindPrsMergedBetweenTests(unittest.TestCase):
    @patch("graph.tools.get_driver")
    def test_excludes_prs_already_merged_before_the_earlier_tag(self, get_driver):
        # The point of this query is "what's new in this window" -- a PR merged
        # before *both* builds (i.e. already merged as of before_tag too) must
        # not show up as if it were introduced between them.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"repo": "cmssw", "pr_number": 51659, "title": "x", "merged_at": None, "author": None}]

        find_prs_merged_between("CMSSW_20_1_CLANG_X_2026-09-04-2300", "CMSSW_20_1_CLANG_X_2026-09-06-2300")

        query, params = session.run.call_args.args
        self.assertIn("MERGED_BEFORE]->(after_ib:IB {tag:$after_tag})", query)
        self.assertIn("NOT EXISTS", query)
        self.assertIn("MERGED_BEFORE]->(:IB {tag:$before_tag})", query)
        self.assertEqual(
            params,
            {
                "before_tag": "CMSSW_20_1_CLANG_X_2026-09-04-2300",
                "after_tag": "CMSSW_20_1_CLANG_X_2026-09-06-2300",
                "lim": 20,
            },
        )

    @patch("graph.tools.get_driver")
    def test_orders_newest_merged_at_first_with_undated_prs_last(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"repo": "cmssw", "pr_number": 51659, "title": "x", "merged_at": None, "author": None}]

        find_prs_merged_between("CMSSW_20_1_CLANG_X_2026-09-04-2300", "CMSSW_20_1_CLANG_X_2026-09-06-2300")

        query = session.run.call_args.args[0]
        self.assertIn("ORDER BY p.merged_at IS NULL, p.merged_at DESC", query)


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


def _empty_pr_match_row(evidence: str, **overrides) -> dict:
    row = {
        "repo": "cmssw", "pr_number": 1, "title": "t", "author": "a", "merged_at": None,
        "matched_warnings": 0, "matched_occurrences": 0, "matched_files": [],
        "matched_unit_tests": 0, "matched_unit_test_errors": 0,
        "matched_unit_test_packages": [], "matched_unit_test_ibs": [], "evidence": evidence,
    }
    row.update(overrides)
    return row


class FindPrsForErrorTests(unittest.TestCase):
    @patch("graph.tools.get_driver")
    def test_matches_a_human_pr_comment(self, get_driver):
        # Regression: find_prs_for_error searched PR title/body/changed-files/
        # warnings/unit-test packages but never comments at all -- a term only
        # ever mentioned in a PR *comment* (not the title/body) silently reported
        # no PR evidence even though it was genuinely there. Confirmed live: "is
        # there any PR/issue open for testAlpakaModelCudaAsync" found nothing
        # because get_relval_exception_evidence/_open_issue_answer already search
        # comments, but the general find_prs_for_error path never did.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        comment_match = _empty_pr_match_row("pr_comment", pr_number=51900, title="Fix SoA round-trip")
        session.run.side_effect = [[], [], [], [], [], [comment_match]]

        results = find_prs_for_error("testSoADataTypesCudaAsync")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["pr_number"], 51900)
        self.assertIn("pr_comment", results[0]["evidence"])
        self.assertEqual(session.run.call_count, 6)

    @patch("graph.tools.get_driver")
    def test_comment_query_excludes_cmsbuild_bot_comments(self, get_driver):
        # cms-bot (author "cmsbuild") posts a routine "Failed Tests" comment
        # dumping the full exception traceback of every currently-broken baseline
        # test on every PR, unrelated to whether that PR actually relates to the
        # search term -- same noise source get_relval_exception_evidence already
        # excludes. The comment-matching query itself must filter it out, not
        # rely on the caller to notice.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.side_effect = [[], [], [], [], [], []]

        find_prs_for_error("some error text")

        comment_query = session.run.call_args_list[5].args[0]
        self.assertIn("c.author <> 'cmsbuild'", comment_query)


class SearchShifterDocsTests(unittest.TestCase):
    @patch("graph.tools.get_driver")
    def test_queries_the_fulltext_index_by_name(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"heading": "ib-run-gpu-relvals", "text": "...", "source_path": "jenkins_reports/CMSSDT.md", "score": 15.4}
        ]

        results = search_shifter_docs("ib-run-gpu-relvals dependency schedule")

        query = session.run.call_args.args[0]
        self.assertIn("db.index.fulltext.queryNodes('shifterDocText'", query)
        self.assertEqual(results[0]["source_path"], "jenkins_reports/CMSSDT.md")

    @patch("graph.tools.get_driver")
    def test_missing_index_degrades_to_empty_list_not_an_exception(self, get_driver):
        # A deployment that never ran ingest_shifter_docs.py has no
        # shifterDocText fulltext index yet -- the tool must behave like "no
        # documentation evidence found", not break the whole tool-calling turn.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.side_effect = Exception("no such fulltext schema index")

        self.assertEqual(search_shifter_docs("anything"), [])


if __name__ == "__main__":
    unittest.main()


class RelvalEvidencePrefetchTests(unittest.TestCase):
    """prefetch_relval_exception_evidence: one artifact scan for many workflows, same
    answers as one query per workflow (found 2026-10-15: 62 scans made a digest take 78-120 s)."""

    def tearDown(self):
        import tools
        tools._EVIDENCE_CACHE.clear()

    @patch("tools.get_driver")
    def test_each_workflow_gets_only_its_own_terms_newest_first(self, get_driver):
        import tools
        session = get_driver.return_value.session.return_value.__enter__.return_value
        exceptions = {
            ("1.0", "T"): [{"exception": "boom one", "category": "LogicError", "context": ""}],
            ("2.0", "T"): [{"exception": "boom two", "category": "ProductNotFound", "context": ""}],
        }
        scans = []

        def run(query, **params):
            if "RelValException" in query:
                return exceptions[(params["workflow_id"], params["ib_tag"])]
            scans.append(sorted(params["terms"]))
            return [  # newest first, the union over both workflows' terms
                {"kind": "PR", "number": 30, "title": "c", "state": "open", "merged_at": None,
                 "matched_terms": ["boom two"]},
                {"kind": "Issue", "number": 20, "title": "b", "state": "open", "merged_at": None,
                 "matched_terms": ["1.0", "boom one"]},
                {"kind": "Issue", "number": 10, "title": "a", "state": "closed", "merged_at": None,
                 "matched_terms": ["2.0"]},
            ]
        session.run.side_effect = run

        tools.prefetch_relval_exception_evidence([("1.0", "T"), ("2.0", "T")], limit=5)
        self.assertEqual(scans, [["1.0", "2.0", "boom one", "boom two"]])  # one scan for both

        get_driver.reset_mock()
        first = tools.get_relval_exception_evidence("1.0", "T", 5)
        second = tools.get_relval_exception_evidence("2.0", "T", 5)
        get_driver.assert_not_called()  # served from the prefetch
        self.assertEqual([m["number"] for m in first[0]["matches"]], [20])
        self.assertEqual(first[0]["matches"][0]["matched_terms"], ["1.0", "boom one"])
        self.assertEqual(first[0]["category"], "LogicError")  # matched via its own exception text
        self.assertEqual([m["number"] for m in second[0]["matches"]], [30])  # exact text beats id-only #10
