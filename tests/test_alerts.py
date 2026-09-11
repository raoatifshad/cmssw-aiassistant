import unittest
from unittest.mock import patch

from graph.alerts import (
    RULE_TYPES,
    _CHECKS,
    _check_build_failure_count,
    _check_build_regression,
    _check_relval_regression,
    _check_unittest_failure_count,
    _check_unittest_regression,
    create_rule,
    evaluate_rules,
)


class RuleTypeRegistrationTests(unittest.TestCase):
    def test_unittest_and_build_failure_rule_types_are_registered(self):
        self.assertIn("unittest_failure_count", RULE_TYPES)
        self.assertIn("build_failure_count", RULE_TYPES)
        self.assertIn("unittest_failure_count", _CHECKS)
        self.assertIn("build_failure_count", _CHECKS)

    def test_relval_regression_rule_type_is_registered(self):
        self.assertIn("relval_regression", RULE_TYPES)
        self.assertIn("relval_regression", _CHECKS)

    def test_unittest_and_build_regression_rule_types_are_registered(self):
        self.assertIn("unittest_regression", RULE_TYPES)
        self.assertIn("unittest_regression", _CHECKS)
        self.assertIn("build_regression", RULE_TYPES)
        self.assertIn("build_regression", _CHECKS)

    @patch("graph.alerts.get_driver")
    def test_create_rule_accepts_relval_regression(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = None

        create_rule("regression-alert", "relval_regression", {}, 1)

        self.assertEqual(session.run.call_args.kwargs["rule_type"], "relval_regression")

    @patch("graph.alerts.get_driver")
    def test_create_rule_accepts_unittest_failure_count(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = None

        create_rule("ut-alert", "unittest_failure_count", {"tag": "CMSSW_20_1_X_2026-08-25-2300"}, 3)

        self.assertEqual(session.run.call_args.kwargs["rule_type"], "unittest_failure_count")

    @patch("graph.alerts.get_driver")
    def test_create_rule_accepts_build_failure_count(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = None

        create_rule("build-alert", "build_failure_count", {"tag": "CMSSW_20_1_X_2026-08-25-2300"}, 1)

        self.assertEqual(session.run.call_args.kwargs["rule_type"], "build_failure_count")

    def test_create_rule_rejects_addon_since_no_addon_data_is_imported(self):
        # graph.tools/ingestion/build_graph.py have no addon-test ingestion at
        # all (confirmed: no "addon" anywhere in either module), so an
        # "addon" rule_type must stay unsupported rather than silently
        # accepted and then never firing for a reason nobody can see.
        with self.assertRaises(ValueError):
            create_rule("addon-alert", "addon_failure_count", {"tag": "x"}, 1)


class CheckUnittestFailureCountTests(unittest.TestCase):
    @patch("graph.alerts.get_failing_unittests")
    def test_triggers_when_failing_count_meets_threshold(self, get_failing_unittests):
        get_failing_unittests.return_value = [{"name": "a"}, {"name": "b"}, {"name": "c"}]

        triggered, message, evidence = _check_unittest_failure_count(
            {"tag": "CMSSW_20_1_X_2026-08-25-2300"}, threshold=3,
        )

        self.assertTrue(triggered)
        self.assertEqual(evidence, {"count": 3})
        self.assertIn("3 failing unit test(s)", message)

    @patch("graph.alerts.get_failing_unittests")
    def test_does_not_trigger_below_threshold(self, get_failing_unittests):
        get_failing_unittests.return_value = [{"name": "a"}]

        triggered, _, evidence = _check_unittest_failure_count(
            {"tag": "CMSSW_20_1_X_2026-08-25-2300"}, threshold=3,
        )

        self.assertFalse(triggered)
        self.assertEqual(evidence, {"count": 1})

    @patch("graph.alerts.get_failing_unittests")
    def test_passes_arch_through_to_the_query(self, get_failing_unittests):
        get_failing_unittests.return_value = []

        _check_unittest_failure_count(
            {"tag": "CMSSW_20_1_X_2026-08-25-2300", "arch": "el9_amd64_gcc13"}, threshold=1,
        )

        get_failing_unittests.assert_called_once_with(
            "CMSSW_20_1_X_2026-08-25-2300", arch="el9_amd64_gcc13", limit=100,
        )


class CheckBuildFailureCountTests(unittest.TestCase):
    @patch("graph.alerts.get_build_failures")
    def test_triggers_and_names_the_failing_architectures(self, get_build_failures):
        get_build_failures.return_value = [
            {"ib_tag": "CMSSW_20_1_X_2026-08-25-2300", "arch": "el9_amd64_gcc13"},
            {"ib_tag": "CMSSW_20_1_X_2026-08-25-2300", "arch": "el8_amd64_gcc14"},
        ]

        triggered, message, evidence = _check_build_failure_count(
            {"tag": "CMSSW_20_1_X_2026-08-25-2300"}, threshold=1,
        )

        self.assertTrue(triggered)
        self.assertEqual(evidence["count"], 2)
        self.assertEqual(evidence["architectures"], ["el9_amd64_gcc13", "el8_amd64_gcc14"])
        self.assertIn("el9_amd64_gcc13", message)
        self.assertIn("el8_amd64_gcc14", message)

    @patch("graph.alerts.get_build_failures")
    def test_does_not_trigger_with_no_build_failures(self, get_build_failures):
        get_build_failures.return_value = []

        triggered, message, evidence = _check_build_failure_count(
            {"tag": "CMSSW_20_1_X_2026-08-25-2300"}, threshold=1,
        )

        self.assertFalse(triggered)
        self.assertEqual(evidence["count"], 0)
        self.assertIn("none", message)


class CheckRelvalRegressionTests(unittest.TestCase):
    """Unlike the other _check_* functions, this one dynamically resolves "latest
    vs previous build" each run (find_relval_regressions) rather than checking a
    currently-failing count against one fixed, pre-configured tag -- so it's
    mocked at that level rather than get_failing_relvals/get_driver directly."""

    @patch("graph.alerts.find_relval_regressions")
    def test_triggers_when_newly_failing_count_meets_threshold(self, find_regressions):
        find_regressions.return_value = {
            "latest_tag": "CMSSW_20_1_X_2026-09-08-2300", "previous_tag": "CMSSW_20_1_X_2026-09-07-2300",
            "newly_failing": [
                {"variant": "primary", "arch": "el9_amd64_gcc14", "workflow_id": "1.0", "name": "A"},
            ],
        }

        triggered, message, evidence = _check_relval_regression({}, threshold=1)

        self.assertTrue(triggered)
        self.assertEqual(evidence["count"], 1)
        self.assertIn("1 RelVal workflow(s) newly failing", message)
        self.assertIn("`1.0`", message)

    @patch("graph.alerts.find_relval_regressions")
    def test_does_not_trigger_with_no_newly_failing_workflows(self, find_regressions):
        find_regressions.return_value = {
            "latest_tag": "CMSSW_20_1_X_2026-09-08-2300", "previous_tag": "CMSSW_20_1_X_2026-09-07-2300",
            "newly_failing": [],
        }

        triggered, message, evidence = _check_relval_regression({}, threshold=1)

        self.assertFalse(triggered)
        self.assertEqual(evidence["count"], 0)

    @patch("graph.alerts.find_relval_regressions")
    def test_reports_gracefully_when_fewer_than_two_builds_imported(self, find_regressions):
        find_regressions.return_value = {"latest_tag": None, "previous_tag": None, "newly_failing": []}

        triggered, message, evidence = _check_relval_regression({}, threshold=1)

        self.assertFalse(triggered)
        self.assertEqual(evidence, {})
        self.assertIn("Fewer than two primary builds", message)

    @patch("graph.alerts.find_relval_regressions")
    def test_passes_arch_through_to_find_relval_regressions(self, find_regressions):
        find_regressions.return_value = {"latest_tag": None, "previous_tag": None, "newly_failing": []}

        _check_relval_regression({"arch": "el9_amd64_gcc14"}, threshold=1)

        find_regressions.assert_called_once_with("el9_amd64_gcc14")


class CheckUnittestRegressionTests(unittest.TestCase):
    @patch("graph.alerts.find_unittest_regressions")
    def test_triggers_when_newly_failing_count_meets_threshold(self, find_regressions):
        find_regressions.return_value = {
            "latest_tag": "CMSSW_20_1_X_2026-09-08-2300", "previous_tag": "CMSSW_20_1_X_2026-09-07-2300",
            "newly_failing": [{"variant": "UBSAN", "arch": "el9_amd64_gcc14", "name": "FlakyTest"}],
        }

        triggered, message, evidence = _check_unittest_regression({}, threshold=1)

        self.assertTrue(triggered)
        self.assertEqual(evidence["count"], 1)
        self.assertIn("1 unit test(s) newly failing", message)
        self.assertIn("`FlakyTest`", message)

    @patch("graph.alerts.find_unittest_regressions")
    def test_does_not_trigger_with_no_newly_failing_tests(self, find_regressions):
        find_regressions.return_value = {
            "latest_tag": "CMSSW_20_1_X_2026-09-08-2300", "previous_tag": "CMSSW_20_1_X_2026-09-07-2300",
            "newly_failing": [],
        }

        triggered, _, evidence = _check_unittest_regression({}, threshold=1)

        self.assertFalse(triggered)
        self.assertEqual(evidence["count"], 0)

    @patch("graph.alerts.find_unittest_regressions")
    def test_reports_gracefully_when_fewer_than_two_builds_imported(self, find_regressions):
        find_regressions.return_value = {"latest_tag": None, "previous_tag": None, "newly_failing": []}

        triggered, message, evidence = _check_unittest_regression({}, threshold=1)

        self.assertFalse(triggered)
        self.assertEqual(evidence, {})
        self.assertIn("Fewer than two primary builds", message)


class CheckBuildRegressionTests(unittest.TestCase):
    @patch("graph.alerts.find_build_regressions")
    def test_triggers_on_a_build_badge_flip(self, find_regressions):
        find_regressions.return_value = {
            "latest_tag": "CMSSW_20_1_X_2026-09-08-2300", "previous_tag": "CMSSW_20_1_X_2026-09-07-2300",
            "newly_failing": [{"variant": "primary", "arch": "el9_amd64_gcc14", "kind": "build"}],
        }

        triggered, message, evidence = _check_build_regression({}, threshold=1)

        self.assertTrue(triggered)
        self.assertEqual(evidence["count"], 1)
        self.assertIn("1 Build/AddOn badge(s) newly in error", message)
        self.assertIn("build on", message)

    @patch("graph.alerts.find_build_regressions")
    def test_does_not_trigger_with_no_badge_flips(self, find_regressions):
        find_regressions.return_value = {
            "latest_tag": "CMSSW_20_1_X_2026-09-08-2300", "previous_tag": "CMSSW_20_1_X_2026-09-07-2300",
            "newly_failing": [],
        }

        triggered, _, evidence = _check_build_regression({}, threshold=1)

        self.assertFalse(triggered)
        self.assertEqual(evidence["count"], 0)

    @patch("graph.alerts.find_build_regressions")
    def test_reports_gracefully_when_fewer_than_two_builds_imported(self, find_regressions):
        find_regressions.return_value = {"latest_tag": None, "previous_tag": None, "newly_failing": []}

        triggered, message, evidence = _check_build_regression({}, threshold=1)

        self.assertFalse(triggered)
        self.assertEqual(evidence, {})
        self.assertIn("Fewer than two primary builds", message)


class EvaluateRulesDispatchTests(unittest.TestCase):
    @patch("graph.alerts.get_build_failures")
    @patch("graph.alerts.list_rules")
    def test_evaluate_rules_dispatches_build_failure_count_and_notifies(self, list_rules, get_build_failures):
        list_rules.return_value = [{
            "rule_id": "r1", "name": "build-alert", "rule_type": "build_failure_count",
            "params": {"tag": "CMSSW_20_1_X_2026-08-25-2300"}, "threshold": 1,
        }]
        get_build_failures.return_value = [{"ib_tag": "x", "arch": "el9_amd64_gcc13"}]
        notifier = unittest.mock.Mock()

        fired = evaluate_rules(notifier)

        self.assertEqual(len(fired), 1)
        notifier.notify.assert_called_once()

    @patch("graph.alerts.get_failing_unittests")
    @patch("graph.alerts.list_rules")
    def test_evaluate_rules_dispatches_unittest_failure_count_and_notifies(self, list_rules, get_failing_unittests):
        list_rules.return_value = [{
            "rule_id": "r2", "name": "ut-alert", "rule_type": "unittest_failure_count",
            "params": {"tag": "CMSSW_20_1_X_2026-08-25-2300"}, "threshold": 1,
        }]
        get_failing_unittests.return_value = [{"name": "a"}]
        notifier = unittest.mock.Mock()

        fired = evaluate_rules(notifier)

        self.assertEqual(len(fired), 1)
        notifier.notify.assert_called_once()

    @patch("graph.alerts.find_relval_regressions")
    @patch("graph.alerts.list_rules")
    def test_evaluate_rules_dispatches_relval_regression_and_notifies(self, list_rules, find_regressions):
        list_rules.return_value = [{
            "rule_id": "r3", "name": "regression-alert", "rule_type": "relval_regression",
            "params": {}, "threshold": 1,
        }]
        find_regressions.return_value = {
            "latest_tag": "CMSSW_20_1_X_2026-09-08-2300", "previous_tag": "CMSSW_20_1_X_2026-09-07-2300",
            "newly_failing": [{"variant": "CLANG", "arch": "el9_amd64_gcc14", "workflow_id": "1.0", "name": "A"}],
        }
        notifier = unittest.mock.Mock()

        fired = evaluate_rules(notifier)

        self.assertEqual(len(fired), 1)
        notifier.notify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
