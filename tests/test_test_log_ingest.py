"""POST /api/test-logs/ingest (server/test_log_ingest.py) and the sender that feeds it
(deploy/cmssdt/send_test_log_scans.py), end to end over a real local HTTP server."""
import gzip
import http.server
import importlib.util
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import test_log_ingest as tli
from server.api_server import app

_spec = importlib.util.spec_from_file_location(
    "send_test_log_scans", Path(__file__).resolve().parent.parent / "deploy" / "cmssdt" / "send_test_log_scans.py")
sender = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sender)

TOKEN = "t0ken-for-tests"


def scan(kind="unit", tag="CMSSW_20_1_X_2026-11-16-2300", arch="el9_amd64_gcc14", variant="",
         scanned_at="2026-11-17T06:40:00", **extra) -> dict:
    items = {"workflows": [{"workflow": "1.0", "status": "PASSED"}]} if kind == "relval" else \
        {"tests": [{"package": "A/B", "test": "t1", "status": "PASSED"}]}
    source = dict(kind=kind, arch=arch, tag=tag, variant=variant, queue="20.1-mon-23", scanner_version=6,
                  scanned_at=scanned_at, scanned_by="ib_test_log_problems")
    return dict(kind=kind, problems=[], source=source, **items, **extra)


def gz(data: dict) -> bytes:
    return gzip.compress(json.dumps(data).encode(), mtime=0)


class ReceiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_saved_where_the_copied_zip_pipeline_puts_the_same_build(self):
        got = tli.receive(gz(scan()), self.root)
        self.assertEqual((got.status, got.path),
                         ("saved", "unit/el9_amd64_gcc14/CMSSW_20_1_X_2026-11-16-2300/_/unitTestLogs.json.gz"))
        got = tli.receive(gz(scan("relval", variant="gpu/nvidia_h100")), self.root)
        self.assertEqual(got.path, "relval/el9_amd64_gcc14/CMSSW_20_1_X_2026-11-16-2300/gpu_nvidia_h100/"
                                   "pyRelValMatrixLogs.json.gz")
        self.assertEqual((self.root / got.path).read_bytes(), gz(scan("relval", variant="gpu/nvidia_h100")))
        self.assertEqual([p.name for p in self.root.rglob(".*")], [])          # no temp files left

    def test_same_file_again_is_unchanged_and_a_newer_scan_replaces(self):
        tli.receive(gz(scan()), self.root)
        self.assertEqual(tli.receive(gz(scan()), self.root).status, "unchanged")
        newer = gz(scan(scanned_at="2026-11-17T09:00:00"))
        self.assertEqual(tli.receive(newer, self.root).status, "replaced")
        older = scan(scanned_at="2026-11-17T05:00:00")
        self.assertEqual(tli.receive(gz(older), self.root).status, "older")    # a resend never undoes a merge
        path = self.root / "unit/el9_amd64_gcc14/CMSSW_20_1_X_2026-11-16-2300/_/unitTestLogs.json.gz"
        self.assertEqual(path.read_bytes(), newer)

    def test_bad_files_are_rejected_with_a_reason_and_nothing_is_written(self):
        bad = {
            "not gzip": b"{}",
            "empty": b"",
            "not JSON": gzip.compress(b"hello"),
            "a list": gz([1]),
            "unknown kind": gz(dict(scan(), kind="clang")),
            "kind mismatch": gz(dict(scan(), kind="addon")),
            "path in arch": gz(scan(arch="../../etc")),
            "path in tag": gz(scan(tag="CMSSW_20_1_X_2026-11-16-2300/../../x")),
            "path in variant": gz(scan(variant="gpu/../../x")),
            "no items": gz({k: v for k, v in scan().items() if k != "tests"}),
            "no version": gz(dict(scan(), source=dict(scan()["source"], scanner_version=None))),
        }
        for why, body in bad.items():
            with self.subTest(why):
                with self.assertRaises(tli.Rejected) as ctx:
                    tli.receive(body, self.root)
                self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(list(self.root.rglob("*")), [])

    def test_size_limits(self):
        with patch.object(tli, "MAX_BYTES", 100):
            with self.assertRaises(tli.Rejected) as ctx:
                tli.receive(os.urandom(101), self.root)
            self.assertEqual(ctx.exception.status, 413)
        bomb = gzip.compress(b" " * 5000)
        with patch.object(tli, "MAX_UNPACKED", 4000):
            with self.assertRaises(tli.Rejected) as ctx:
                tli.receive(bomb, self.root)
            self.assertEqual(ctx.exception.status, 413)

    def test_token(self):
        with patch.dict(os.environ, {tli.TOKEN_ENV: ""}):
            with self.assertRaises(tli.Rejected) as ctx:
                tli.check_token(TOKEN)
            self.assertEqual(ctx.exception.status, 503)                         # off unless configured
        with patch.dict(os.environ, {tli.TOKEN_ENV: TOKEN}):
            tli.check_token(TOKEN)
            for wrong in (None, "", "t0ken", TOKEN + "x"):
                with self.assertRaises(tli.Rejected) as ctx:
                    tli.check_token(wrong)
                self.assertEqual(ctx.exception.status, 401)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {tli.TOKEN_ENV: TOKEN, tli.RESULTS_ENV: self.tmp.name})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(app)

    def post(self, body, token=TOKEN):
        headers = {"Content-Type": "application/gzip"}
        if token:
            headers["X-Test-Logs-Token"] = token
        return self.client.post("/api/test-logs/ingest", content=body, headers=headers)

    def test_route(self):
        r = self.post(gz(scan("addon")))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "saved")
        self.assertTrue((Path(self.tmp.name) / r.json()["path"]).is_file())
        self.assertEqual(self.post(gz(scan("addon"))).json()["status"], "unchanged")
        self.assertEqual(self.post(gz(scan("addon")), token=None).status_code, 401)
        self.assertEqual(self.post(gz(scan("addon")), token="nope").status_code, 401)
        r = self.post(b"junk")
        self.assertEqual(r.status_code, 400)
        self.assertIn("gzip", r.json()["detail"])


class _Handler(http.server.BaseHTTPRequestHandler):
    root = None
    fail_next = 0
    calls = 0

    def do_POST(self):
        type(self).calls += 1
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if type(self).fail_next:
            type(self).fail_next -= 1
            return self._reply(502, {"detail": "proxy hiccup"})
        try:
            tli.check_token(self.headers.get("X-Test-Logs-Token"))
            self._reply(200, tli.receive(body, self.root).as_dict())
        except tli.Rejected as exc:
            self._reply(exc.status, {"detail": exc.reason})

    def _reply(self, code, data):
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class SenderTests(unittest.TestCase):
    """The sender against a real HTTP server running the endpoint's own checks."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.buildlogs, self.results = base / "buildlogs", base / "results"
        env = patch.dict(os.environ, {tli.TOKEN_ENV: TOKEN})
        env.start()
        self.addCleanup(env.stop)
        handler = type("H", (_Handler,), {"root": self.results, "fail_next": 0, "calls": 0})
        self.handler = handler
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.lines = []

    def build(self, rel, data, mtime=None):
        path = self.buildlogs / "el9_amd64_gcc14/www/mon/20.1-mon-23/CMSSW_20_1_X_2026-11-16-2300" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gz(data))
        if mtime:
            os.utime(path, (mtime, mtime))
        return path

    def send(self, token=TOKEN, **kw):
        return sender.run(str(self.buildlogs), self.url, token, sleep=lambda s: None, out=self.lines.append, **kw)

    def test_sends_new_files_once_and_a_rewritten_file_again(self):
        unit = self.build("unitTestLogs-scan.json.gz", scan())
        self.build("gpu/nvidia_h100/pyRelValMatrixLogs-scan.json.gz", scan("relval", variant="gpu/nvidia_h100"))
        self.build("pyRelValPartialLogs/1.0_x/step1.log", {})                  # not a scan file: never sent
        (self.buildlogs / "filtered/x/www/mon/q/CMSSW_20_1_X_2026-11-16-2300").mkdir(parents=True)
        self.assertEqual(self.send(), 0)
        self.assertEqual(self.handler.calls, 2)
        self.assertEqual(len(list(self.results.rglob("*.json.gz"))), 2)
        self.assertTrue(Path(str(unit) + ".sent").read_text().split()[1] == "saved")

        self.assertEqual(self.send(), 0)
        self.assertEqual(self.handler.calls, 2)                                 # nothing new: nothing sent
        self.assertIn("summary: 0 sent, 0 unchanged, 2 skipped, 0 rejected, 0 failed", self.lines[-1])

        self.build("unitTestLogs-scan.json.gz", scan(scanned_at="2026-11-17T09:00:00"))   # rewritten
        self.assertEqual(self.send(), 0)
        self.assertEqual(self.handler.calls, 3)
        self.assertIn("replaced", Path(str(unit) + ".sent").read_text())

    def test_resend_last_days(self):
        self.build("unitTestLogs-scan.json.gz", scan())
        self.build("addOnTests-scan.json.gz", scan("addon"), mtime=1_000_000)  # long ago
        self.send()
        self.send(resend_days=7)
        self.assertEqual(self.handler.calls, 3)                                 # only the recent one again
        self.assertIn("summary: 0 sent, 1 unchanged, 1 skipped", self.lines[-1])

    def test_a_short_outage_is_retried_and_a_bad_file_is_reported_once(self):
        self.build("unitTestLogs-scan.json.gz", scan())
        bad = self.build("addOnTests-scan.json.gz", scan("addon", arch="weird"))
        self.handler.fail_next = 2
        self.assertEqual(self.send(), 1)
        self.assertEqual(len(list(self.results.rglob("*.json.gz"))), 1)
        self.assertTrue(any(l.startswith("REJECTED") and "bad architecture" in l for l in self.lines))
        self.assertIn("rejected", Path(str(bad) + ".sent").read_text())
        calls = self.handler.calls
        self.assertEqual(self.send(), 0)                                        # not retried until it changes
        self.assertEqual(self.handler.calls, calls)

    def test_wrong_token_stops_at_once_and_marks_nothing(self):
        files = [self.build(n, scan(k)) for n, k in (("unitTestLogs-scan.json.gz", "unit"),
                                                     ("addOnTests-scan.json.gz", "addon"))]
        self.assertEqual(self.send(token="wrong"), 2)
        self.assertEqual(self.handler.calls, 1)
        self.assertFalse(any(Path(str(f) + ".sent").exists() for f in files))

    def test_unreachable_vm_gives_up_and_retries_next_run(self):
        for i in range(7):
            self.build(f"other/r{i}/unitTestLogs-scan.json.gz", scan(variant=f"other/r{i}"))
        self.server.shutdown()
        self.server.server_close()
        self.assertEqual(sender.run(str(self.buildlogs), "http://127.0.0.1:9", TOKEN, sleep=lambda s: None,
                                    out=self.lines.append), 1)
        self.assertTrue(any("looks unreachable" in l for l in self.lines))
        self.assertEqual(list(self.buildlogs.rglob("*.sent")), [])

    def test_dry_run_sends_nothing(self):
        self.build("unitTestLogs-scan.json.gz", scan())
        self.assertEqual(sender.run(str(self.buildlogs), "", "", dry_run=True, out=self.lines.append), 0)
        self.assertEqual(self.handler.calls, 0)
        self.assertTrue(self.lines[0].startswith("would send el9_amd64_gcc14/"))


if __name__ == "__main__":
    unittest.main()
