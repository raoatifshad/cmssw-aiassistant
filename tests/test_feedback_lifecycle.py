import unittest
from unittest.mock import MagicMock, patch

from server.api_server import _maybe_create_concept, _verify_feedback
from ingestion.revalidate_feedback import revalidate_expiring_concepts, revalidate_pending_feedback


class VerifyFeedbackTests(unittest.TestCase):
    """_verify_feedback is the single evidence gate reused by the live
    /api/feedback endpoint and by the revalidation script -- a correction is
    only ever trusted because the graph confirms it."""

    @patch("server.api_server._llm_feedback_verification")
    def test_falls_back_to_llm_verifier_for_non_latest_ib_questions(self, llm_verification):
        llm_verification.return_value = ("accepted", "Graph facts support this.")

        status, verification, method, tag, tags = _verify_feedback(
            "Is Issue #51449 open?", "It is open.", "correct", None, None,
        )

        self.assertEqual(status, "accepted")
        self.assertEqual(method, "llm_graph_tools")
        llm_verification.assert_called_once()

    @patch("server.api_server._llm_feedback_verification", return_value=None)
    def test_stays_pending_when_llm_verifier_cannot_decide(self, llm_verification):
        status, verification, method, tag, tags = _verify_feedback(
            "Is Issue #51449 open?", "It is open.", "correct", None, None,
        )

        self.assertEqual(status, "pending")
        self.assertEqual(method, "unverified")

    @patch("server.api_server._latest_ib_records")
    def test_deterministic_path_never_calls_the_llm_verifier(self, latest_ib_records):
        latest_ib_records.return_value = [{"tag": "CMSSW_20_1_X_2026-08-10-1100"}]
        with patch("server.api_server._llm_feedback_verification") as llm_verification:
            status, verification, method, tag, tags = _verify_feedback(
                "What is the latest IB?", "CMSSW_20_1_X_2026-08-10-1100 is the latest.", "correct", None, None,
            )

        self.assertEqual(status, "accepted")
        self.assertEqual(method, "deterministic")
        llm_verification.assert_not_called()


class MaybeCreateConceptTests(unittest.TestCase):
    """A Concept is only ever created for a graph-verified, evidence-based
    correction -- never from user agreement alone."""

    def test_creates_an_active_concept_with_an_expiry_when_accepted(self):
        session = MagicMock()

        concept_id = _maybe_create_concept(
            session, "fb-1", "What is X?", "incorrect", "X is actually Y.", "accepted", "llm_graph_tools",
        )

        self.assertIsNotNone(concept_id)
        query, kwargs = session.run.call_args.args[0], session.run.call_args.kwargs
        self.assertIn("status:'active'", query)
        self.assertIn("expires_at:datetime()", query)
        self.assertEqual(kwargs["definition"], "X is actually Y.")

    def test_does_not_create_a_concept_when_rejected(self):
        session = MagicMock()

        concept_id = _maybe_create_concept(
            session, "fb-1", "What is X?", "incorrect", "X is actually Y.", "rejected", "llm_graph_tools",
        )

        self.assertIsNone(concept_id)
        session.run.assert_not_called()

    def test_does_not_create_a_concept_without_a_correction(self):
        concept_id = _maybe_create_concept(
            MagicMock(), "fb-1", "What is X?", "incorrect", None, "accepted", "deterministic",
        )
        self.assertIsNone(concept_id)

    def test_does_not_create_a_concept_for_a_correct_verdict(self):
        # verdict='correct' means the original answer was confirmed, not that
        # corrected_answer (if any leftover value) is a new fact to store.
        concept_id = _maybe_create_concept(
            MagicMock(), "fb-1", "What is X?", "correct", "X is actually Y.", "accepted", "deterministic",
        )
        self.assertIsNone(concept_id)


def _run_side_effect(pending_rows=None, expiring_rows=None):
    def run(query, **kwargs):
        if "UserFeedback {status:'pending'}" in query:
            return list(pending_rows or [])
        if "Concept {status:'active'}" in query:
            return list(expiring_rows or [])
        return []
    return run


class RevalidatePendingFeedbackTests(unittest.TestCase):
    def _driver(self, run_side_effect):
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value.run.side_effect = run_side_effect
        return driver

    @patch("ingestion.revalidate_feedback._maybe_create_concept")
    @patch("ingestion.revalidate_feedback._verify_feedback")
    def test_resolved_pending_feedback_is_updated_and_may_promote_a_concept(self, verify, maybe_create):
        verify.return_value = ("accepted", "Graph facts support this.", "llm_graph_tools", None, [])
        row = {
            "feedback_id": "fb-1", "question": "What is X?", "answer": "wrong",
            "verdict": "incorrect", "corrected_answer": "X is Y.", "ib_tag": None,
        }
        driver = self._driver(_run_side_effect(pending_rows=[row]))

        checked, resolved = revalidate_pending_feedback(driver, limit=10)

        self.assertEqual((checked, resolved), (1, 1))
        verify.assert_called_once_with("What is X?", "wrong", "incorrect", "X is Y.", None)
        maybe_create.assert_called_once()
        session = driver.session.return_value.__enter__.return_value
        set_calls = [c for c in session.run.call_args_list if "SET f.status" in c.args[0]]
        self.assertEqual(len(set_calls), 1)
        self.assertEqual(set_calls[0].kwargs["status"], "accepted")

    @patch("ingestion.revalidate_feedback._maybe_create_concept")
    @patch("ingestion.revalidate_feedback._verify_feedback")
    def test_still_pending_feedback_is_left_untouched(self, verify, maybe_create):
        verify.return_value = ("pending", "Still undecidable.", "unverified", None, [])
        row = {
            "feedback_id": "fb-1", "question": "What is X?", "answer": "wrong",
            "verdict": "incorrect", "corrected_answer": "X is Y.", "ib_tag": None,
        }
        driver = self._driver(_run_side_effect(pending_rows=[row]))

        checked, resolved = revalidate_pending_feedback(driver, limit=10)

        self.assertEqual((checked, resolved), (1, 0))
        maybe_create.assert_not_called()
        session = driver.session.return_value.__enter__.return_value
        set_calls = [c for c in session.run.call_args_list if "SET f.status" in c.args[0]]
        self.assertEqual(len(set_calls), 0)


class RevalidateExpiringConceptsTests(unittest.TestCase):
    def _driver(self, run_side_effect):
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value.run.side_effect = run_side_effect
        return driver

    def _row(self, in_pending_grace=False):
        return {
            "concept_id": "c-1", "question": "What is X?", "definition": "X is Y.",
            "in_pending_grace": in_pending_grace,
        }

    @patch("ingestion.revalidate_feedback._verify_feedback")
    def test_still_accepted_concept_gets_its_expiry_extended(self, verify):
        verify.return_value = ("accepted", "Still supported.", "llm_graph_tools", None, [])
        driver = self._driver(_run_side_effect(expiring_rows=[self._row()]))

        checked, extended, revoked = revalidate_expiring_concepts(driver, limit=10)

        self.assertEqual((checked, extended, revoked), (1, 1, 0))
        session = driver.session.return_value.__enter__.return_value
        update = next(c for c in session.run.call_args_list if "c.expires_at=datetime() + duration" in c.args[0])
        self.assertIn("in_pending_grace=false", update.args[0])

    @patch("ingestion.revalidate_feedback._verify_feedback")
    def test_rejected_concept_is_revoked_immediately(self, verify):
        verify.return_value = ("rejected", "Graph facts contradict this.", "llm_graph_tools", None, [])
        driver = self._driver(_run_side_effect(expiring_rows=[self._row()]))

        checked, extended, revoked = revalidate_expiring_concepts(driver, limit=10)

        self.assertEqual((checked, extended, revoked), (1, 0, 1))
        session = driver.session.return_value.__enter__.return_value
        update = next(c for c in session.run.call_args_list if "status='revoked'" in c.args[0])
        self.assertEqual(update.kwargs["concept_id"], "c-1")

    @patch("ingestion.revalidate_feedback._verify_feedback")
    def test_pending_concept_gets_one_grace_period_before_revocation(self, verify):
        verify.return_value = ("pending", "No longer decidable.", "unverified", None, [])
        driver = self._driver(_run_side_effect(expiring_rows=[self._row(in_pending_grace=False)]))

        checked, extended, revoked = revalidate_expiring_concepts(driver, limit=10)

        self.assertEqual((checked, extended, revoked), (1, 0, 0))
        session = driver.session.return_value.__enter__.return_value
        update = next(c for c in session.run.call_args_list if "in_pending_grace=true" in c.args[0])
        self.assertEqual(update.kwargs["concept_id"], "c-1")

    @patch("ingestion.revalidate_feedback._verify_feedback")
    def test_still_pending_concept_is_revoked_after_grace_is_already_used(self, verify):
        verify.return_value = ("pending", "No longer decidable.", "unverified", None, [])
        driver = self._driver(_run_side_effect(expiring_rows=[self._row(in_pending_grace=True)]))

        checked, extended, revoked = revalidate_expiring_concepts(driver, limit=10)

        self.assertEqual((checked, extended, revoked), (1, 0, 1))


if __name__ == "__main__":
    unittest.main()
