import unittest
from unittest.mock import patch

from server.api_server import ANSWER_SYNTHESIS_ENABLED, ROUTER_SHADOW_ENABLED, _log_router_shadow_decision


class RouterShadowFlagTests(unittest.TestCase):
    def test_both_llm_features_default_off(self):
        # These must default off: each costs a real per-message LLM call with
        # no user-visible benefit until explicitly opted into.
        self.assertFalse(ROUTER_SHADOW_ENABLED)
        self.assertFalse(ANSWER_SYNTHESIS_ENABLED)

    @patch("server.api_server.get_driver")
    @patch("server.api_server.propose_capability_classification")
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.ROUTER_SHADOW_ENABLED", True)
    def test_logs_agreement_when_enabled(self, classify, get_driver):
        classify.return_value = {"capability": "relval", "confidence": 0.9}

        _log_router_shadow_decision("how many relvals failing", "deterministic", "relval", "shadow-sid")

        session = get_driver.return_value.session.return_value.__enter__.return_value
        call = session.run.call_args
        self.assertTrue(call.kwargs["agree"])
        self.assertEqual(call.kwargs["llm_capability"], "relval")
        get_driver.return_value.close.assert_called_once()

    @patch("server.api_server.get_driver")
    def test_is_a_no_op_when_disabled(self, get_driver):
        # ROUTER_SHADOW_ENABLED is false by default in this test process --
        # no LLM call, no graph write, no driver ever opened.
        _log_router_shadow_decision("how many relvals failing", "deterministic", "relval", "shadow-sid-2")

        get_driver.assert_not_called()

    @patch("server.api_server.get_driver")
    @patch("server.api_server.propose_capability_classification", side_effect=RuntimeError("boom"))
    @patch("server.api_server._llm_client", object())
    @patch("server.api_server.ROUTER_SHADOW_ENABLED", True)
    def test_never_raises_even_if_the_llm_call_fails(self, classify, get_driver):
        try:
            _log_router_shadow_decision("anything", "llm", None, "shadow-sid-3")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"_log_router_shadow_decision must never raise, got {exc!r}")


if __name__ == "__main__":
    unittest.main()
