import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from server.api_server import app, require_shifter
from server.shifter_auth import shifter_allowlist


def _request(session=None):
    return SimpleNamespace(session=session if session is not None else {})


class ShifterAllowlistTests(unittest.TestCase):
    @patch.dict(os.environ, {"SHIFTER_CERN_USERNAMES": "alice, Bob ,  "}, clear=False)
    def test_parses_trims_and_lowercases_names(self):
        self.assertEqual(shifter_allowlist(), frozenset({"alice", "bob"}))

    @patch.dict(os.environ, {"SHIFTER_CERN_USERNAMES": ""}, clear=False)
    def test_empty_env_var_yields_empty_allowlist(self):
        self.assertEqual(shifter_allowlist(), frozenset())


class RequireShifterTests(unittest.TestCase):
    def test_no_session_raises_401(self):
        with self.assertRaises(HTTPException) as ctx:
            require_shifter(_request())
        self.assertEqual(ctx.exception.status_code, 401)

    @patch.dict(os.environ, {"SHIFTER_CERN_USERNAMES": "alice"}, clear=False)
    def test_session_username_not_on_allowlist_raises_403(self):
        with self.assertRaises(HTTPException) as ctx:
            require_shifter(_request(session={"shifter_username": "mallory"}))
        self.assertEqual(ctx.exception.status_code, 403)

    @patch.dict(os.environ, {"SHIFTER_CERN_USERNAMES": "alice"}, clear=False)
    def test_session_username_on_allowlist_passes(self):
        username = require_shifter(_request(session={"shifter_username": "alice"}))
        self.assertEqual(username, "alice")

    @patch.dict(os.environ, {"SHIFTER_CERN_USERNAMES": "alice"}, clear=False)
    def test_removal_from_allowlist_revokes_an_existing_session_immediately(self):
        # The allowlist is re-checked on every request, not just at login -- a
        # session that was valid when created must stop working the moment the
        # env var no longer lists that username, with no fresh login required.
        request = _request(session={"shifter_username": "alice"})
        require_shifter(request)  # still allowed
        with patch.dict(os.environ, {"SHIFTER_CERN_USERNAMES": "bob"}, clear=False):
            with self.assertRaises(HTTPException) as ctx:
                require_shifter(request)
            self.assertEqual(ctx.exception.status_code, 403)


class DevShifterLoginTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("server.shifter_auth.DEV_SHIFTER_LOGIN", False)
    def test_dev_login_route_is_404_when_disabled(self):
        # Must be indistinguishable from a route that doesn't exist -- a real
        # deployment that forgot to unset DEV_SHIFTER_LOGIN shouldn't even
        # reveal this bypass is present.
        response = self.client.get("/shift/dev-login", params={"username": "alice"}, follow_redirects=False)
        self.assertEqual(response.status_code, 404)

    @patch.dict(os.environ, {"SHIFTER_CERN_USERNAMES": "alice"}, clear=False)
    @patch("server.shifter_auth.DEV_SHIFTER_LOGIN", True)
    def test_dev_login_sets_a_session_that_still_respects_the_allowlist(self):
        response = self.client.get("/shift/dev-login", params={"username": "mallory"}, follow_redirects=False)
        self.assertIn(response.status_code, (302, 307))

        # The session was set (dev-login faked "who"), but require_shifter still
        # rejects it on the next request since "mallory" isn't on the allowlist.
        follow_up = self.client.get("/api/shift-summary", follow_redirects=False)
        self.assertEqual(follow_up.status_code, 403)

    @patch.dict(os.environ, {"SHIFTER_CERN_USERNAMES": "alice"}, clear=False)
    @patch("server.shifter_auth.DEV_SHIFTER_LOGIN", True)
    @patch("server.api_server._shift_summary_answer", return_value="a digest")
    def test_dev_login_grants_access_for_an_allowlisted_username(self, answer):
        login = self.client.get("/shift/dev-login", params={"username": "Alice"}, follow_redirects=False)
        self.assertIn(login.status_code, (302, 307))

        response = self.client.get("/api/shift-summary")

        self.assertEqual(response.status_code, 200)


class ProtectedEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(app.dependency_overrides.clear)

    def test_shift_page_without_a_session_is_refused(self):
        response = self.client.get("/shift", follow_redirects=False)
        self.assertEqual(response.status_code, 401)

    def test_shift_summary_api_without_a_session_is_refused(self):
        response = self.client.get("/api/shift-summary", follow_redirects=False)
        self.assertEqual(response.status_code, 401)

    def test_alerts_status_api_without_a_session_is_refused(self):
        response = self.client.get("/api/alerts/status", follow_redirects=False)
        self.assertEqual(response.status_code, 401)

    @patch("server.api_server._shift_summary_answer", return_value="a digest")
    def test_shift_summary_api_serves_the_digest_for_an_authorized_shifter(self, answer):
        app.dependency_overrides[require_shifter] = lambda: "alice"

        response = self.client.get("/api/shift-summary")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"markdown": "a digest"})

    @patch("server.api_server.evaluate_rules")
    def test_alerts_status_never_writes_to_the_alert_log(self, evaluate_rules):
        # evaluate_rules() must be called with a notifier that only records in
        # memory (_NullNotifier) -- LogNotifier would append a duplicate entry
        # to alerts.log.jsonl on every /shift page load.
        app.dependency_overrides[require_shifter] = lambda: "alice"

        response = self.client.get("/api/alerts/status")

        self.assertEqual(response.status_code, 200)
        notifier_used = evaluate_rules.call_args.args[0]
        self.assertEqual(notifier_used.__class__.__name__, "_NullNotifier")


if __name__ == "__main__":
    unittest.main()
