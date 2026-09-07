import unittest
from unittest.mock import patch

from graph.alerts import (
    RULE_TYPES,
    _CHECKS,
    _check_build_failure_count,
    _check_unittest_failure_count,
    create_rule,
    evaluate_rules,
)


class RuleTypeRegistrationTests(unittest.TestCase):
    def test_unittest_and_build_failure_rule_types_are_registered(self):
        self.assertIn("unittest_failure_count", RULE_TYPES)
        self.assertIn("build_failure_count", RULE_TYPES)
        self.assertIn("unittest_failure_count", _CHECKS)
        self.assertIn("build_failure_count", _CHECKS)

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


if __name__ == "__main__":
    unittest.main()
