"""POST /api/test-logs/ingest (server/test_log_ingest.py) and the sender that feeds it
(deploy/cmssdt/send_test_log_scans.py), over a real local HTTP server."""
import gzip
import hashlib
import http.server
import importlib.util
import io
import json
import os
import tarfile
import tempfile
import threading
import time
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


def step(workflow="1001.0_RunMinBias2011A", number=1, **kw) -> dict:
    doc = dict(kind="relval_step", arch="el9_amd64_gcc14", tag="CMSSW_20_1_X_2026-11-16-2300", variant="",
               queue="20.1-mon-23", workflow=workflow, step=number, status="PASSED", exit_code=0,
               path="el9_amd64_gcc14/www/mon/20.1-mon-23/CMSSW_20_1_X_2026-11-16-2300/pyRelValMatrixLogs.zip",
               workflow_steps={"1": "PASSED", "2": "PASSED"}, workflow_exit_codes=[0, 0], problems=[],
               log_mtime=1000.0, scanned_at="2026-11-17T06:40:00", scanner_version=6)
    return dict(doc, **kw)


def tar_of(files: dict) -> bytes:
    return sender.make_tar(files)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "received"

    def receive(self, files):
        return tli.receive_batch(tar_of(files), self.root)

    def test_good_files_are_saved_together_and_confirmed_with_their_sha256(self):
        files = {"relval/relval__a__1001.0_X+Y__step1.json.gz": gz(step()), "unit/u.json.gz": gz(scan()),
                 "progress.json": json.dumps({"el9_amd64_gcc14/CMSSW_20_1_X_2026-11-16-2300/gpu/nvidia_t4":
                                              dict(jobs=["done.1of3"], total=3, closed=False)}).encode()}
        answer = self.receive(files)
        self.assertEqual(answer, {"saved": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()},
                                  "rejected": {}, "kept": {}})
        saved = list(self.root.glob("*.tar"))
        self.assertEqual([p.suffix for p in saved], [".tar"])
        self.assertEqual(tli.read_tar(saved[0].read_bytes()), files)               # byte for byte

    def test_bad_files_are_refused_one_by_one_and_the_rest_is_saved(self):
        bad = {
            "relval/kind.json.gz": gz(dict(step(), kind="relval")),
            "relval/arch.json.gz": gz(step(arch="../../etc")),
            "relval/tag.json.gz": gz(step(tag="CMSSW_x")),
            "relval/variant.json.gz": gz(step(variant="gpu/../x")),
            "relval/wf.json.gz": gz(step("../1.0_X")),
            "relval/step.json.gz": gz(step(number=-1)),
            "relval/bool.json.gz": gz(step(number=True)),
            "relval/problems.json.gz": gz(step(problems=["x"])),
            "relval/steps.json.gz": gz(step(workflow_steps={"x": "PASSED"})),
            "relval/path.json.gz": gz(step(path="el9_amd64_gcc14/www/../../CMSSW_20_1_X_2026-11-16-2300")),
            "relval/mtime.json.gz": gz(step(log_mtime=None)),
            "relval/nogzip.json.gz": b"hello",
            "relval/cut.json.gz": gz(step())[:40],
            "unit/addon.json.gz": gz(scan("addon")),
            "other/x.json.gz": gz(scan()),
            "relval/../../x.json.gz": gz(step()),
            "relval/.hidden.json.gz": gz(step()),
        }
        answer = self.receive(dict(bad, **{"relval/good.json.gz": gz(step())}))
        self.assertEqual(sorted(answer["rejected"]), sorted(bad))
        self.assertEqual(list(answer["saved"]), ["relval/good.json.gz"])
        tars = list(self.root.glob("*.tar"))
        self.assertEqual(list(tli.read_tar(tars[0].read_bytes())), ["relval/good.json.gz"])

    def test_refused_files_are_kept_with_the_reason(self):
        files = {"relval/x.json.gz": b"no", "relval/../../etc/pass wd.json.gz": b"no2"}
        answer = self.receive(files)
        self.assertEqual(answer["saved"], {})
        self.assertEqual(answer["kept"], {n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
        self.assertEqual(list(self.root.glob("*.tar")), [])
        names = sorted(p.name for p in (self.root / "rejected").iterdir())
        self.assertEqual(len(names), 4)                                            # 2 files + 2 reasons
        self.assertTrue(all(p.parent == self.root / "rejected" for p in self.root.rglob("*") if p.is_file()))
        self.receive(files)
        self.assertEqual(sorted(p.name for p in (self.root / "rejected").iterdir()), names)   # not twice
        with patch.object(tli, "MAX_REJECTED_BYTES", 20):
            self.assertEqual(self.receive({"relval/y.json.gz": b"z" * 30})["kept"], {})

    def test_an_unchanged_progress_list_is_confirmed_but_not_saved_again(self):
        one = json.dumps({f"el9_amd64_gcc14/CMSSW_20_1_X_2026-11-16-2300/_": dict(jobs=["done.1of3"], total=3, closed=False)}).encode()
        for _ in range(3):
            self.assertIn("progress.json", self.receive({"progress.json": one})["saved"])
        self.assertEqual(len(list(self.root.glob("*.tar"))), 1)
        self.receive({"progress.json": one.replace(b"done.1of3", b"done.2of3")})
        self.assertEqual(len(list(self.root.glob("*.tar"))), 2)

    def test_the_request_as_a_whole(self):
        link = io.BytesIO()
        with tarfile.open(fileobj=link, mode="w") as tar:
            info = tarfile.TarInfo("relval/x.json.gz")
            info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tar.addfile(info)
        for body in (b"", b"junk" * 300, tar_of({}), link.getvalue()):
            with self.assertRaises(tli.Rejected) as ctx:
                tli.receive_batch(body, self.root)
            self.assertEqual(ctx.exception.status, 400)
        with patch.object(tli, "MAX_BYTES", 10), self.assertRaises(tli.Rejected) as ctx:
            self.receive({"relval/x.json.gz": gz(step())})
        self.assertEqual(ctx.exception.status, 413)
        with patch.object(tli, "MIN_FREE_BYTES", 1 << 62), self.assertRaises(tli.Rejected) as ctx:
            self.receive({"relval/x.json.gz": gz(step())})
        self.assertEqual(ctx.exception.status, 503)                                # sender keeps its files

    def test_unpacked_size_is_limited_for_the_whole_batch(self):
        big = gz(step(problems=[{"x": "y" * 3000}]))
        with patch.object(tli, "MAX_UNPACKED", 8000):
            answer = self.receive({f"relval/{i}.json.gz": big for i in range(4)})
        self.assertEqual((len(answer["saved"]), len(answer["rejected"])), (2, 2))

    def test_a_failed_write_confirms_nothing_and_leaves_no_part(self):
        with patch.object(tli.os, "replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.receive({"relval/x.json.gz": gz(step())})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_route(self):
        with patch.dict(os.environ, {tli.TOKEN_ENV: TOKEN, tli.RECEIVED_ENV: str(self.root)}):
            client = TestClient(app)
            body = tar_of({"relval/x.json.gz": gz(step())})
            headers = {"Content-Type": "application/x-tar", "X-Test-Logs-Token": TOKEN}
            r = client.post("/api/test-logs/batch", content=body, headers=headers)
            self.assertEqual((r.status_code, list(r.json()["saved"])), (200, ["relval/x.json.gz"]))
            r = client.post("/api/test-logs/batch", content=body, headers=dict(headers, **{"X-Test-Logs-Token": "no"}))
            self.assertEqual(r.status_code, 401)
            self.assertEqual(len(list(self.root.glob("*.tar"))), 1)


class _Handler(http.server.BaseHTTPRequestHandler):
    """The VM's side of the batch upload, as far as the sender sees it."""
    got: dict = {}
    batches: list = []
    fail_next = 0
    answer_code = 0             # force this HTTP code
    lie_about: set = set()      # names confirmed with a wrong sha256
    drop: set = set()           # names not confirmed at all
    refuse: dict = {}           # name -> reason
    no_room: set = set()        # refused and not kept
    poison: set = set()         # names that make the whole request a 400
    on_request = None

    def do_POST(self):
        cls = type(self)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if cls.on_request:
            cls.on_request()
        if cls.fail_next:
            cls.fail_next -= 1
            return self._reply(502, {"detail": "proxy hiccup"})
        if cls.answer_code:
            return self._reply(cls.answer_code, {"detail": "no"})
        if self.headers.get("X-Test-Logs-Token") != TOKEN:
            return self._reply(401, {"detail": "wrong token"})
        with tarfile.open(fileobj=io.BytesIO(body)) as tar:
            files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        if cls.poison & set(files):
            return self._reply(400, {"detail": "poison"})
        cls.batches.append(sorted(files))
        saved = {}
        for name, data in files.items():
            if name in cls.drop or name in cls.refuse:
                continue
            cls.got[name] = data
            saved[name] = "0" * 64 if name in cls.lie_about else hashlib.sha256(data).hexdigest()
        refused = {n: r for n, r in cls.refuse.items() if n in files}
        kept = {n: hashlib.sha256(files[n]).hexdigest() for n in refused if n not in cls.no_room}
        self._reply(200, {"saved": saved, "rejected": refused, "kept": kept})

    def _reply(self, code, data):
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class SenderTests(unittest.TestCase):
    """The sender against a real HTTP server: only confirmed files are deleted."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.buildlogs, self.outbox = base / "buildlogs", base / "outbox"
        self.h = type("H", (_Handler,), dict(got={}, batches=[], lie_about=set(), drop=set(), refuse={}, poison=set(), no_room=set()))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self.h)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.lines = []

    def put(self, name, data=b"x", age_hours=0):
        path = self.outbox / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if age_hours:
            t = time.time() - age_hours * 3600
            os.utime(path, (t, t))
        return path

    def left(self):
        return sorted(str(p.relative_to(self.outbox)) for p in self.outbox.rglob("*")
                      if p.is_file())

    def send(self, token=TOKEN, **kw):
        return sender.run(str(self.outbox), str(self.buildlogs), kw.pop("url", self.url), token,
                          sleep=lambda s: None, out=self.lines.append, **kw)

    def test_everything_waiting_is_sent_in_batches_and_deleted(self):
        for i in range(7):
            self.put(f"relval/r{i}.json.gz", b"r%d" % i)
        self.put("unit/u.json.gz", b"u")
        self.put("addon/a.json.gz", b"a")
        self.put("relval/half.json.gz.tmp", b"being written")
        with patch.object(sender, "BATCH_FILES", 3):
            self.assertEqual(self.send(), 0)
        self.assertEqual([len(b) for b in self.h.batches], [1, 1, 3, 3, 1])
        self.assertEqual(self.h.batches[0], ["unit/u.json.gz"])                  # unit and addOn first
        self.assertEqual(self.h.got["relval/r6.json.gz"], b"r6")
        self.assertEqual(self.left(), ["relval/half.json.gz.tmp"])
        self.assertIn("summary: 9 sent, 0 refused, 0 unconfirmed", self.lines[-1])
        self.assertFalse((self.outbox / "sending").exists())                     # nothing of ours is left

    def test_batches_are_cut_by_size_too(self):
        for i in range(4):
            self.put(f"relval/r{i}.json.gz", b"x" * 40)
        with patch.object(sender, "BATCH_BYTES", 100):
            self.assertEqual(self.send(), 0)
        self.assertEqual([len(b) for b in self.h.batches], [2, 2])

    def test_first_run_creates_the_outbox_and_an_empty_one_sends_nothing(self):
        self.assertEqual(self.send(), 0)
        self.assertTrue(self.outbox.is_dir())
        self.assertEqual(self.h.batches, [])

    def test_files_arriving_during_the_run_are_sent_in_the_same_run(self):
        self.put("relval/r1.json.gz")
        extra = iter(["relval/r2.json.gz"])
        self.h.on_request = lambda: [self.put(n) for n in [next(extra, None)] if n]
        self.assertEqual(self.send(), 0)
        self.assertEqual(sorted(self.h.got), ["relval/r1.json.gz", "relval/r2.json.gz"])
        self.assertEqual(self.left(), [])

    def test_a_file_rewritten_while_it_is_sent_is_not_lost(self):
        self.put("relval/r1.json.gz", b"old")
        again = iter([b"new"])
        self.h.on_request = lambda: [self.put("relval/r1.json.gz", d) for d in [next(again, None)] if d]
        self.assertEqual(self.send(), 0)
        self.assertEqual(self.h.got["relval/r1.json.gz"], b"new")
        self.assertEqual(len(self.h.batches), 2)
        self.assertEqual(self.left(), [])

    def test_unconfirmed_or_wrongly_confirmed_files_are_kept_and_sent_again(self):
        for n in ("a", "b", "c"):
            self.put(f"relval/{n}.json.gz", n.encode())
        self.h.drop, self.h.lie_about = {"relval/a.json.gz"}, {"relval/b.json.gz"}
        self.assertEqual(self.send(), 0)                                         # only waiting for a moment
        self.assertEqual(self.left(), ["sending/relval/a.json.gz", "sending/relval/b.json.gz"])
        self.h.drop, self.h.lie_about = set(), set()
        self.assertEqual(self.send(), 0)
        self.assertEqual(self.left(), [])
        self.assertEqual(sorted(self.h.got), [f"relval/{n}.json.gz" for n in "abc"])

    def test_vm_down_keeps_everything_and_the_next_run_sends_it(self):
        for i in range(3):
            self.put(f"relval/r{i}.json.gz")
        self.assertEqual(self.send(url="http://127.0.0.1:9"), 0)                 # a short outage stays green
        self.assertTrue(any(l.startswith("NOT SENT") for l in self.lines))
        self.assertEqual(len(self.left()), 3)
        self.assertEqual(self.send(), 0)
        self.assertEqual((len(self.h.got), self.left()), (3, []))

    def test_short_outage_is_retried_in_the_same_run(self):
        self.put("relval/r.json.gz")
        self.h.fail_next = 2
        self.assertEqual(self.send(), 0)
        self.assertEqual(self.left(), [])

    def test_wrong_token_or_upload_off_stops_and_deletes_nothing(self):
        self.put("unit/u.json.gz")
        self.assertEqual(self.send(token="wrong"), 2)
        for code in (503, 404):
            self.h.answer_code = code
            self.assertEqual(self.send(), 2)
        self.assertEqual(self.left(), ["sending/unit/u.json.gz"])
        self.assertFalse(any(TOKEN in l or "wrong " in l.split("(")[0] for l in self.lines))

    def test_refused_files_are_deleted_reported_and_do_not_hold_up_the_others(self):
        for n in "abcde":
            self.put(f"relval/{n}.json.gz")
        self.h.refuse = {"relval/a.json.gz": "bad architecture"}
        self.h.poison = {"relval/d.json.gz"}                  # the request as a whole is refused
        self.assertEqual(self.send(), 1)
        self.assertEqual(self.left(), [])
        self.assertEqual(sorted(self.h.got), [f"relval/{n}.json.gz" for n in "bce"])
        self.assertTrue(any("relval/a.json.gz: bad architecture (kept on the VM)" in l for l in self.lines))
        self.assertTrue(any("relval/d.json.gz" in l and "not kept" in l for l in self.lines))
        self.assertEqual(self.send(), 0)                      # reported once

    def test_a_refused_file_the_vm_has_no_room_for_waits_and_is_tried_once_per_run(self):
        for n in "ab":
            self.put(f"relval/{n}.json.gz")
        self.h.refuse, self.h.no_room = {"relval/a.json.gz": "why"}, {"relval/a.json.gz"}
        self.assertEqual(self.send(), 1)
        self.assertEqual(self.left(), ["sending/relval/a.json.gz"])
        self.assertEqual(len(self.h.batches), 1)
        self.assertIn("relval/b.json.gz", self.h.got)

    def test_nothing_stays_for_good(self):
        self.put("relval/old.json.gz", age_hours=73)
        self.put("sending/unit/old.json.gz", age_hours=73)
        self.put("relval/waiting.json.gz", age_hours=0.5)
        self.put("relval/dead.json.gz.tmp", age_hours=25)
        self.put("unit/writing.json.gz.tmp")
        self.assertEqual(self.send(url="http://127.0.0.1:9"), 0)
        self.assertEqual(self.left(), ["relval/waiting.json.gz", "unit/writing.json.gz.tmp"])
        self.assertTrue(any("older than 3 days" in l for l in self.lines))

    def test_old_files_are_sent_rather_than_deleted_when_the_vm_is_back(self):
        self.put("relval/old.json.gz", age_hours=73)
        self.assertEqual(self.send(), 0)
        self.assertIn("relval/old.json.gz", self.h.got)

    def test_every_run_says_the_sender_is_alive(self):
        self.outbox.mkdir()
        os.utime(self.outbox, (1000, 1000))
        self.send(url="http://127.0.0.1:9")
        self.assertLess(time.time() - self.outbox.stat().st_mtime, 60)

    def test_files_waiting_too_long_make_the_run_unstable(self):
        self.put("relval/r.json.gz", age_hours=3)
        self.h.drop = {"relval/r.json.gz"}
        self.assertEqual(self.send(), 1)
        self.assertTrue(any("waiting for 3.0 hours" in l for l in self.lines))

    def test_time_limit(self):
        for i in range(3):
            self.put(f"relval/r{i}.json.gz")
        with patch.object(sender, "BATCH_FILES", 1):
            self.assertEqual(self.send(max_minutes=-1), 0)
        self.assertEqual(len(self.h.got), 1)
        self.assertEqual(len(self.left()), 2)

    def test_dry_run_changes_nothing(self):
        self.put("relval/r.json.gz")
        self.assertEqual(self.send(dry_run=True), 0)
        self.assertEqual(self.lines, ["would send relval/r.json.gz"])
        self.assertEqual((self.left(), self.h.batches), (["relval/r.json.gz"], []))

    def test_progress_of_recent_relval_builds_is_sent(self):
        build = self.buildlogs / "el9_amd64_gcc14/www/mon/20.1-mon-23/CMSSW_20_1_X_2026-11-16-2300"
        main, gpu = build / "pyRelValPartialLogs", build / "gpu/nvidia_t4/pyRelValPartialLogs"
        dbg = self.buildlogs / "el9_amd64_gcc14/www/mon/q/CMSSW_20_1_DBG_X_2026-11-16-2300/pyRelValPartialLogs"
        for folder in (main, gpu, dbg):
            folder.mkdir(parents=True)
            (folder / "done.1of3").touch()
        key = "el9_amd64_gcc14/CMSSW_20_1_X_2026-11-16-2300/"
        self.assertEqual(self.send(), 0)
        first = json.loads(self.h.got["progress.json"])
        self.assertEqual(sorted(first), [key + "_", key + "gpu/nvidia_t4"])       # not DBG
        self.assertEqual(first[key + "_"], dict(jobs=["done.1of3"], total=3, closed=False))
        self.assertEqual(self.left(), [])                                         # nothing remembered here
        (main / "done.2of3").touch()
        main.rename(main.with_name("pyRelValPartialLogs.tmp"))                    # cms-bot finished it
        main.mkdir()
        (main / "done.3of3").touch()                                              # a late job comes back
        self.assertEqual(self.send(), 0)
        second = json.loads(self.h.got["progress.json"])
        self.assertEqual(second[key + "_"], dict(jobs=["done.1of3", "done.2of3", "done.3of3"], total=3, closed=True))

    def test_unconfirmed_progress_is_sent_again(self):
        folder = self.buildlogs / "el9_amd64_gcc14/www/mon/q/CMSSW_20_1_X_2026-11-16-2300/pyRelValPartialLogs"
        folder.mkdir(parents=True)
        (folder / "done.1of1").touch()
        self.h.drop = {"progress.json"}
        self.assertEqual(self.send(), 0)
        self.assertTrue(any("progress not confirmed" in l for l in self.lines))
        self.h.drop = set()
        self.assertEqual(self.send(), 0)
        self.assertIn("progress.json", self.h.got)


if __name__ == "__main__":
    unittest.main()
