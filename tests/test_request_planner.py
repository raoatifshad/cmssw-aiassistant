import unittest

from server.request_planner import (
    Capability,
    PlanKind,
    RequestContext,
    RequestPlanner,
    Scope,
    validate_llm_plan,
)


IB_SUMMARY = Capability(
    name="ib_summary",
    required_tag_count=1,
    requires_summary=True,
    priority=30,
)
UNITTEST = Capability(
    name="unittest_summary",
    required_categories=frozenset({"unittest"}),
    priority=40,
)
RELVAL = Capability(
    name="relval_summary",
    required_categories=frozenset({"relval"}),
    priority=40,
)


class RequestPlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = RequestPlanner((IB_SUMMARY, UNITTEST, RELVAL))

    def test_extracts_exact_ib_architecture_and_scope(self):
        context = RequestContext.extract(
            "summary for CMSSW_20_1_X_2026-08-10-1100 with el9_amd64_gcc14"
        )

        self.assertEqual(context.tag, "CMSSW_20_1_X_2026-08-10-1100")
        self.assertEqual(context.architecture, "el9_amd64_gcc14")
        self.assertEqual(context.scope, Scope.EXPLICIT_ARCHITECTURE)
        self.assertTrue(context.has_summary_request)
        self.assertEqual(self.planner.plan(context).capability, "ib_summary")

    def test_all_scope_is_parsed_without_hardcoded_architectures(self):
        context = RequestContext.extract("summary for CMSSW_20_1_X_2026-08-10-1100 all")

        self.assertEqual(context.scope, Scope.ALL)
        self.assertEqual(self.planner.plan(context).capability, "ib_summary")

    def test_extracts_riscv_architecture_and_explicit_stable_ids(self):
        context = RequestContext.extract(
            "PR #51449 for workflow 34551.85 in fc42_riscv64_gcc15"
        )

        self.assertEqual(context.architecture, "fc42_riscv64_gcc15")
        self.assertEqual(context.pr_number, 51449)
        self.assertEqual(context.workflow_id, "34551.85")

    def test_dates_are_not_workflow_ids(self):
        context = RequestContext.extract("summary for CMSSW_20_1_X_2026-08-10-1100")

        self.assertIsNone(context.workflow_id)

    def test_unit_alone_is_recognized_without_the_word_test(self):
        # Regression: "unit failures in <tag>" has no "test"/"tests" word at
        # all -- the old \bunit[\s-]*tests?\b pattern required it, so this
        # question had no recognized category and fell into the ambiguous
        # "choose a scope" clarification instead of routing directly.
        context = RequestContext.extract("unit failures in CMSSW_20_1_PY314_X_2026-08-19-1100")

        self.assertIn("unittest", context.categories)
        self.assertEqual(self.planner.plan(context).capability, "unittest_summary")

    def test_package_alone_is_recognized_as_a_unittest_synonym(self):
        # Regression: found live -- "which packages are failing in <tag>
        # <arch>" has neither "unit" nor "test" in it at all. UnitTest
        # results are package-level aggregates in this graph (see
        # CLAUDE.md), so "package" is as much a unittest synonym here as
        # "wf"/"workflow" already are for "relval" -- without this, the
        # question had no recognized category and fell into the ambiguous
        # "choose a scope" clarification instead of routing directly.
        context = RequestContext.extract(
            "which packages are failing in CMSSW_20_1_X_2026-08-25-2300 el9_aarch64_gcc14"
        )

        self.assertIn("unittest", context.categories)
        self.assertEqual(self.planner.plan(context).capability, "unittest_summary")

    def test_ambiguous_failures_create_deterministic_clarification(self):
        context = RequestContext.extract("failures for CMSSW_20_1_X_2026-08-10-1100")

        plan = self.planner.plan(context, "session-1")

        self.assertEqual(plan.kind, PlanKind.CLARIFICATION)
        self.assertEqual(plan.clarification.missing_field, "category")
        self.assertIn("RelVals", plan.clarification.question)

    def test_category_reply_fills_pending_tag_without_llm(self):
        tag = "CMSSW_20_1_X_2026-08-10-1100"
        self.planner.plan(RequestContext.extract(f"failures for {tag}"), "session-1")

        plan = self.planner.plan(RequestContext.extract("unit tests"), "session-1")

        self.assertEqual(plan.kind, PlanKind.DETERMINISTIC)
        self.assertEqual(plan.capability, "unittest_summary")
        self.assertNotIn("session-1", self.planner.pending)

    def test_clarification_choices_carry_the_original_architecture(self):
        # Regression: found live 2026-09-21 -- "how many ibs are failing for
        # el9_amd64_gcc14 for release <tag>" triggered this clarification,
        # but every canned choice (e.g. "Combined summary") only ever
        # reconstructed "summary for <tag>", silently dropping the
        # architecture the user had already named. Clicking that button
        # then widened the answer to every imported architecture instead of
        # just the one originally asked about.
        tag = "CMSSW_20_1_X_2026-09-13-2300"
        context = RequestContext.extract(
            f"how many ibs are failing for el9_amd64_gcc14 for release {tag}"
        )

        plan = self.planner.plan(context, "session-arch")

        choices = dict(plan.clarification.choices)
        self.assertEqual(choices["Combined summary"], f"summary for {tag} el9_amd64_gcc14")
        self.assertEqual(choices["Build warnings"], f"build warnings for {tag} el9_amd64_gcc14")
        self.assertEqual(choices["RelVal failures"], f"failed RelVals for {tag} el9_amd64_gcc14")
        self.assertEqual(choices["Unit-test results"], f"unit tests for {tag} el9_amd64_gcc14")

    def test_bare_category_reply_resolves_with_original_architecture_preserved(self):
        # _resolve_pending's own path: a bare category reply that repeats
        # neither the tag nor the architecture ("unit tests" alone) must
        # still resolve scoped to the architecture named in the ORIGINAL
        # ambiguous question, not silently widen to every architecture --
        # same bug as test_clarification_choices_carry_the_original_architecture,
        # different call path (free-text bare reply vs. a canned choice).
        tag = "CMSSW_20_1_X_2026-09-13-2300"
        self.planner.plan(
            RequestContext.extract(f"failures for el9_amd64_gcc14 {tag}"), "session-arch-2",
        )

        plan = self.planner.plan(RequestContext.extract("unit tests"), "session-arch-2")

        self.assertEqual(plan.kind, PlanKind.DETERMINISTIC)
        self.assertEqual(plan.capability, "unittest_summary")
        self.assertEqual(plan.context.architecture, "el9_amd64_gcc14")

    def test_multiple_tags_without_comparison_asks_for_intent(self):
        context = RequestContext.extract(
            "CMSSW_20_1_X_2026-08-10-1100 CMSSW_20_1_X_2026-08-11-2300"
        )

        plan = self.planner.plan(context)

        self.assertEqual(plan.kind, PlanKind.CLARIFICATION)
        self.assertEqual(plan.clarification.missing_field, "intent")

    def test_sub_ib_question_selects_relationship_capability(self):
        relationship = Capability(
            name="ib_relationship",
            required_tag_count=1,
            requires_relationship=True,
            priority=50,
        )
        planner = RequestPlanner((IB_SUMMARY, relationship))

        plan = planner.plan(RequestContext.extract(
            "why is CMSSW_20_1_TF_X_2026-08-14-1100 a sub IB?"
        ))

        self.assertEqual(plan.kind, PlanKind.DETERMINISTIC)
        self.assertEqual(plan.capability, "ib_relationship")

    def test_rejects_unregistered_or_writing_llm_plan(self):
        valid = validate_llm_plan(
            {
                "capability": "relval_summary",
                "tool_calls": [{"name": "get_failing_relvals", "arguments": {"ib_tag": "tag"}}],
                "answer_constraints": ["local_only"],
            },
            (RELVAL,),
            {"get_failing_relvals"},
        )
        invalid = validate_llm_plan(
            {
                "capability": "relval_summary",
                "tool_calls": [{"name": "write_graph", "arguments": {}}],
                "answer_constraints": ["local_only"],
            },
            (RELVAL,),
            {"get_failing_relvals"},
        )

        self.assertIsNotNone(valid)
        self.assertIsNone(invalid)
