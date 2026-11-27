import os
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

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
    # SSO_CONFIGURED/DEV_SHIFTER_LOGIN are module-level constants resolved from
    # live env vars at import time -- explicitly patch both in every case here
    # rather than relying on whatever this test process's actual .env happens
    # to have set, matching this project's own test_router_shadow.py lesson.
    @patch("server.shifter_auth.DEV_SHIFTER_LOGIN", True)
    def test_no_session_raises_401_when_a_login_path_exists(self):
        with self.assertRaises(HTTPException) as ctx:
            require_shifter(_request())
        self.assertEqual(ctx.exception.status_code, 401)

    @patch("server.shifter_auth.SSO_CONFIGURED", False)
    @patch("server.shifter_auth.DEV_SHIFTER_LOGIN", False)
    def test_no_session_raises_503_when_nothing_is_configured(self):
        # /api/shift-whoami must be able to report "not set up" on its own --
        # /shift/login's 503 alone isn't enough, since that's a full-page
        # navigation target the frontend's fetch-based logic can't inspect.
        with self.assertRaises(HTTPException) as ctx:
            require_shifter(_request())
        self.assertEqual(ctx.exception.status_code, 503)

    @patch("server.shifter_auth.SSO_CONFIGURED", True)
    @patch("server.shifter_auth.DEV_SHIFTER_LOGIN", False)
    def test_no_session_raises_401_when_real_sso_is_configured(self):
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

    # DEV_SHIFTER_LOGIN patched True in every case below: these assert 401
    # specifically (a login path exists but no session was presented), which
    # requires *some* login path to be configured -- without patching, this
    # instead depends on whatever this test process's ambient .env happens to
    # have set (401 locally where DEV_SHIFTER_LOGIN=true, but 503 in CI where
    # nothing is configured), the same lesson RequireShifterTests already
    # documents and applies to itself above -- found live in CI, where these
    # three failed with "503 != 401" for exactly that reason.
    @patch("server.shifter_auth.DEV_SHIFTER_LOGIN", True)
    def test_shift_page_without_a_session_is_refused(self):
        response = self.client.get("/shift", follow_redirects=False)
        self.assertEqual(response.status_code, 401)

    @patch("server.shifter_auth.DEV_SHIFTER_LOGIN", True)
    def test_shift_summary_api_without_a_session_is_refused(self):
        response = self.client.get("/api/shift-summary", follow_redirects=False)
        self.assertEqual(response.status_code, 401)

    @patch("server.shifter_auth.DEV_SHIFTER_LOGIN", True)
    def test_alerts_status_api_without_a_session_is_refused(self):
        response = self.client.get("/api/alerts/status", follow_redirects=False)
        self.assertEqual(response.status_code, 401)

    @patch("server.shifter_auth.DEV_SHIFTER_LOGIN", True)
    @patch("server.api_server.evaluate_rules")
    @patch("server.api_server.create_rule")
    @patch("server.api_server.deactivate_rule")
    def test_write_endpoints_without_a_session_are_refused(self, deactivate_rule, create_rule, evaluate_rules):
        rule = {"name": "n", "rule_type": "relval_failure_count", "params": {}, "threshold": 1}
        for method, path, kwargs in (
            ("post", "/api/ingest?source=x", {"content": b"{}"}),
            ("post", "/api/alerts", {"json": rule}),
            ("delete", "/api/alerts/abc", {}),
            ("post", "/api/alerts/evaluate", {}),
        ):
            response = getattr(self.client, method)(path, follow_redirects=False, **kwargs)
            self.assertEqual(response.status_code, 401, path)
        create_rule.assert_not_called()
        deactivate_rule.assert_not_called()
        evaluate_rules.assert_not_called()

    @patch("server.api_server.evaluate_rules", return_value=[])
    def test_alert_evaluate_works_for_a_shifter(self, evaluate_rules):
        app.dependency_overrides[require_shifter] = lambda: "alice"
        response = self.client.post("/api/alerts/evaluate")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"fired": 0, "alerts": []})

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

    @patch("server.shifter_auth.DEV_SHIFTER_LOGIN", True)
    def test_shift_summary_windows_api_without_a_session_is_refused(self):
        response = self.client.get("/api/shift-summary/windows", follow_redirects=False)
        self.assertEqual(response.status_code, 401)

    @patch(
        "server.api_server._list_shift_summary_windows",
        return_value=[{"release_name": "CMSSW_20_1_X_2026-09-03-2300", "label": "2026-09-03 23:00"}],
    )
    def test_shift_summary_windows_api_serves_the_window_list(self, windows):
        app.dependency_overrides[require_shifter] = lambda: "alice"

        response = self.client.get("/api/shift-summary/windows")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"windows": [{"release_name": "CMSSW_20_1_X_2026-09-03-2300", "label": "2026-09-03 23:00"}]},
        )

    @patch("server.api_server.generate_shift_summary", return_value="custom window digest")
    def test_shift_summary_api_uses_explicit_from_to_window(self, generate):
        app.dependency_overrides[require_shifter] = lambda: "alice"

        response = self.client.get(
            "/api/shift-summary",
            params={
                "from": "CMSSW_20_1_X_2026-08-20-2300",
                "to": "CMSSW_20_1_X_2026-08-27-2300",
                "arch": "el10_amd64_gcc15",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"markdown": "custom window digest"})
        # to_tag is latest_tag (first arg), from_tag is previous_tag (second arg) --
        # generate_shift_summary's own title line reads "previous_tag -> latest_tag",
        # so this ordering is what makes the response read "from -> to".
        generate.assert_called_once_with(
            "CMSSW_20_1_X_2026-08-27-2300", "CMSSW_20_1_X_2026-08-20-2300", "el10_amd64_gcc15",
        )

    def test_shift_summary_api_rejects_from_without_to(self):
        app.dependency_overrides[require_shifter] = lambda: "alice"

        response = self.client.get("/api/shift-summary", params={"from": "CMSSW_20_1_X_2026-08-20-2300"})

        self.assertEqual(response.status_code, 422)

    @patch("server.shifter_auth.DEV_SHIFTER_LOGIN", True)
    def test_release_explorer_failing_api_without_a_session_is_refused(self):
        response = self.client.get(
            "/api/release-explorer/failing", params={"tag": "CMSSW_20_1_X_2026-09-08-2300"}, follow_redirects=False,
        )
        self.assertEqual(response.status_code, 401)

    @patch("server.api_server.get_unittest_log_details")
    @patch("server.api_server.get_exit_code_names", return_value={16640: "OtherCMS"})
    @patch("server.api_server.get_failing_unittests")
    @patch("server.api_server.get_failing_relvals")
    def test_release_explorer_failing_api_queries_each_requested_arch_and_tags_variant(
        self, get_relvals, get_unittests, get_exit_codes, get_log_details,
    ):
        # One architecture has a primary-build failure (variant=None -> "primary")
        # plus a sub-IB failure (variant="CLANG"); the other has nothing -- both
        # get_failing_relvals/get_failing_unittests calls happen (one per requested
        # arch, since neither RETURNs its own arch column when arch is omitted), and
        # every row in the response is tagged with the arch it was actually asked for.
        def relvals_side_effect(tag, arch=None, limit=200):
            if arch == "el9_amd64_gcc14":
                return [
                    {"ib_tag": tag, "variant": None, "workflow_id": 140.051, "name": "wf_a", "errors": 2, "exitcode": 16640},
                    {"ib_tag": tag, "variant": "CLANG", "workflow_id": 25.0, "name": "wf_b", "errors": 1, "exitcode": 0},
                ]
            return []

        def unittests_side_effect(tag, arch=None, limit=200, include_sub_ibs=False):
            self.assertTrue(include_sub_ibs)
            if arch == "el9_amd64_gcc14":
                return [{"ib_tag": tag, "variant": None, "name": "FastSimulation/Utilities", "errors": 3, "ok": 9}]
            return []

        tag = "CMSSW_20_1_X_2026-09-08-2300"
        get_log_details.side_effect = lambda builds: {
            (tag, "el9_amd64_gcc14"): dict(
                imported=True,
                failed_tests=["FastSimulation/Utilities/testA", "FastSimulation/Utilities/testB", "Other/Pkg/testC"],
                problems={"FastSimulation/Utilities/testB": ("fatal_exception", "ProductNotFound | in\nmodule x")},
            )} if builds else {}

        get_relvals.side_effect = relvals_side_effect
        get_unittests.side_effect = unittests_side_effect
        app.dependency_overrides[require_shifter] = lambda: "alice"

        response = self.client.get(
            "/api/release-explorer/failing",
            params={"tag": "CMSSW_20_1_X_2026-09-08-2300", "archs": "el9_amd64_gcc14,el9_aarch64_gcc14"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["tag"], "CMSSW_20_1_X_2026-09-08-2300")
        self.assertEqual(
            body["relvals"],
            [
                {
                    "arch": "el9_amd64_gcc14", "variant": "primary", "workflow_id": 140.051,
                    "name": "wf_a", "errors": 2, "exit_code": "16640 (OtherCMS)",
                },
                {
                    "arch": "el9_amd64_gcc14", "variant": "CLANG", "workflow_id": 25.0,
                    "name": "wf_b", "errors": 1, "exit_code": "0",
                },
            ],
        )
        self.assertEqual(
            body["unittests"],
            [{"arch": "el9_amd64_gcc14", "variant": "primary", "name": "FastSimulation/Utilities", "errors": 3,
              "total": 12, "logs_imported": True, "failed_tests": ["testA", "testB"],
              # only testB's own error; testA has none, never another test's
              "exception": "testB: ProductNotFound | in module x"}],
        )
        self.assertEqual(
            get_relvals.call_args_list,
            [
                call("CMSSW_20_1_X_2026-09-08-2300", arch="el9_amd64_gcc14", limit=None),
                call("CMSSW_20_1_X_2026-09-08-2300", arch="el9_aarch64_gcc14", limit=None),
            ],
        )


if __name__ == "__main__":
    unittest.main()
