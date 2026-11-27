"""Receiving end of the IB test-log scan files.

cms-bot's jobs on cmssdt9-02 write one small file per unit/addOn zip and per RelVal
step; a sender job there (deploy/cmssdt/send_test_log_scans.py) posts everything
waiting as tars to

  POST /api/test-logs/batch    receive_batch(): every file is checked; the good ones are
      saved together as one tar in <received>/ (TEST_LOG_RECEIVED_DIR) and only then
      confirmed with their sha256 -- the sender deletes what was confirmed. A refused file is
      kept in <received>/rejected/ with the reason (the sender deletes it too). The worker
      merges <received>/ into <results>/ (ingestion/merge_test_log_batches.py) and imports it.

This module only CHECKS and SAVES, so a slow or failing import never keeps the sender
waiting and a received file is never lost.

  POST /api/test-logs/ingest   receive(): ONE whole result of a build (gzip body), saved
      straight into <results>/. Kept for sending a result by hand.

receive() saves where the old copied-zip pipeline (result_path) saved the
same build, so a scan file replaces that build's earlier result instead of
adding a second run:

    <results>/<kind>/<arch>/<IB tag>/<variant or _>/<unitTestLogs|addOnTests|pyRelValMatrixLogs>.json.gz

The path is built only from checked fields of the file's content, never from a
name the sender gives. The bytes are stored exactly as received, so sending the
same file again is answered "unchanged" and the importer skips it (same hash).
A file older than the one already saved (scanned_at) is not stored ("older"):
a "resend the last 7 days" run can never undo a newer (merged) RelVal scan.

Access: header X-Test-Logs-Token must equal TEST_LOGS_INGEST_TOKEN; when that
variable is not set, the endpoint is off (503).
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import tarfile
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path

TOKEN_ENV = "TEST_LOGS_INGEST_TOKEN"
RESULTS_ENV = "TEST_LOG_RESULTS_DIR"
RECEIVED_ENV = "TEST_LOG_RECEIVED_DIR"
MAX_FILES = 2000                        # in one batch; the sender sends 500
MAX_REJECTED_BYTES = 200 * 1024 * 1024  # received/rejected/ altogether; the worker deletes after 30 days
MIN_FREE_BYTES = 5 * 1024 ** 3          # below this nothing is accepted: Neo4j shares the disk
PROGRESS_MEMBER = "progress.json"
PROGRESS_KNOWN = ".progress-known"     # sha256 of the last one saved
MAX_BYTES = 32 * 1024 * 1024            # compressed; real files are KBs .. ~3 MB
MAX_UNPACKED = 512 * 1024 * 1024        # guards against a gzip bomb

FILE_NAMES = {"unit": "unitTestLogs.json.gz", "addon": "addOnTests.json.gz", "relval": "pyRelValMatrixLogs.json.gz"}
ITEMS_KEY = {"unit": "tests", "addon": "tests", "relval": "workflows"}
ARCH_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+_[a-z0-9]+$")          # el9_amd64_gcc14
TAG_RE = re.compile(r"^CMSSW_\d+_\d+_[A-Za-z0-9_]*X_\d{4}-\d{2}-\d{2}-\d{4}$")
WORKFLOW_RE = re.compile(r"^[0-9]+(\.[0-9]+)?_[^/\s\x00-\x1f]{1,200}$")   # 1001.0_RunMinBias2011A
MEMBER_RE = re.compile(r"^(?P<kind>unit|addon|relval)/[^./\x00-\x1f][^/\x00-\x1f]{0,250}\.json\.gz$")
VARIANT_RE = re.compile(r"^(gpu|other)/[A-Za-z0-9][A-Za-z0-9_.-]*$")


class Rejected(Exception):
    """The file is not accepted; the message says why (sent back to the sender)."""

    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass
class Received:
    status: str        # "saved" | "replaced" | "unchanged" | "older"
    path: str          # relative to the results folder
    sha256: str

    def as_dict(self) -> dict:
        return {"status": self.status, "path": self.path, "sha256": self.sha256}


def check_token(given: str | None) -> None:
    expected = os.environ.get(TOKEN_ENV, "")
    if not expected:
        raise Rejected(503, f"test-log ingest is off ({TOKEN_ENV} not set on the server)")
    if not given or not hmac.compare_digest(given.encode(), expected.encode()):
        raise Rejected(401, "missing or wrong X-Test-Logs-Token")


def results_dir() -> Path:
    return Path(os.environ.get(RESULTS_ENV, "/data/results"))


def received_dir() -> Path:
    return Path(os.environ.get(RECEIVED_ENV, "/data/received"))


def _gunzip(payload: bytes, limit: int | None = None) -> bytes:
    limit = MAX_UNPACKED if limit is None else limit
    if not payload:
        raise Rejected(400, "empty body")
    if len(payload) > MAX_BYTES:
        raise Rejected(413, f"file is {len(payload)} bytes, limit {MAX_BYTES}")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as fh:
            raw = fh.read(limit + 1)
    except (OSError, EOFError, zlib.error) as exc:
        raise Rejected(400, f"not a gzip file ({exc})")
    if len(raw) > limit:
        raise Rejected(413, f"unpacks to more than {limit} bytes")
    return raw


def _unpack(payload: bytes) -> dict:
    return _parse(_gunzip(payload))


def _parse(raw: bytes) -> dict:
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise Rejected(400, f"not JSON ({exc})")
    if not isinstance(data, dict):
        raise Rejected(400, "not a scan result (JSON object expected)")
    return data


def target_path(data: dict) -> Path:
    """Checks that this is one of our scan results and returns where it belongs,
    relative to the results folder. Raises Rejected."""
    kind = data.get("kind")
    if kind not in FILE_NAMES:
        raise Rejected(400, f"unknown kind {kind!r} (unit, addon or relval)")
    source = data.get("source")
    if not isinstance(source, dict):
        raise Rejected(400, "no source block")
    if source.get("kind") != kind:
        raise Rejected(400, "source.kind does not match kind")
    folder = build_folder(kind, source)
    if not isinstance(data.get(ITEMS_KEY[kind]), list) or not isinstance(data.get("problems"), list):
        raise Rejected(400, f"a {kind} result needs '{ITEMS_KEY[kind]}' and 'problems' lists")
    if not isinstance(source.get("scanner_version"), int):
        raise Rejected(400, "source.scanner_version missing")
    return folder / FILE_NAMES[kind]


def build_folder(kind: str, source: dict) -> Path:
    """<kind>/<arch>/<tag>/<variant or _>, from checked fields only."""
    arch, tag, variant = source.get("arch"), source.get("tag"), source.get("variant", "")
    if not isinstance(arch, str) or not ARCH_RE.match(arch):
        raise Rejected(400, f"bad architecture {arch!r}")
    if not isinstance(tag, str) or not TAG_RE.match(tag):
        raise Rejected(400, f"bad IB tag {tag!r}")
    if not isinstance(variant, str) or (variant and not VARIANT_RE.match(variant)):
        raise Rejected(400, f"bad variant {variant!r}")
    return Path(kind) / arch / tag / (variant.replace("/", "_") or "_")


def check_step(data: dict) -> Path:
    """Checks one RelVal step file and returns the build result it belongs to."""
    if data.get("kind") != "relval_step":
        raise Rejected(400, f"unknown kind {data.get('kind')!r} (relval_step)")
    folder = build_folder("relval", data)
    wf, step = data.get("workflow"), data.get("step")
    if not isinstance(wf, str) or not WORKFLOW_RE.match(wf):
        raise Rejected(400, f"bad workflow {wf!r}")
    if not isinstance(step, int) or isinstance(step, bool) or not 0 <= step < 1000:
        raise Rejected(400, f"bad step {step!r}")
    problems, steps = data.get("problems"), data.get("workflow_steps")
    if not isinstance(problems, list) or not all(isinstance(p, dict) for p in problems):
        raise Rejected(400, "a step needs a 'problems' list")
    if not isinstance(steps, dict) or not all(k.isdigit() and isinstance(v, str) for k, v in steps.items()):
        raise Rejected(400, "bad workflow_steps")
    if not isinstance(data.get("workflow_exit_codes"), list):
        raise Rejected(400, "bad workflow_exit_codes")
    if not isinstance(data.get("log_mtime"), (int, float)) or not isinstance(data.get("scanned_at"), str):
        raise Rejected(400, "log_mtime / scanned_at missing")
    if not isinstance(data.get("scanner_version"), int):
        raise Rejected(400, "scanner_version missing")
    path = data.get("path")
    if not isinstance(path, str) or ".." in path or not path.startswith(data["arch"] + "/www/") or data["tag"] not in path:
        raise Rejected(400, f"bad path {path!r}")
    return folder / FILE_NAMES["relval"]


def check_progress(payload: bytes) -> dict:
    """{"<arch>/<tag>/<variant or _>": {"jobs": [...], "total": n or null, "closed": bool}}"""
    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise Rejected(400, f"progress is not JSON ({exc})")
    if not isinstance(data, dict):
        raise Rejected(400, "progress is not an object")
    for key, entry in data.items():
        progress_target(key)
        jobs, total = (entry.get("jobs"), entry.get("total")) if isinstance(entry, dict) else (None, None)
        if not isinstance(jobs, list) or not all(isinstance(j, str) and j.startswith("done.") for j in jobs):
            raise Rejected(400, f"bad progress of {key!r}")
        if not (total is None or isinstance(total, int)) or not isinstance(entry.get("closed"), bool):
            raise Rejected(400, f"bad progress of {key!r}")
    return data


def progress_target(key: str) -> Path:
    arch, tag, variant = (key.split("/", 2) + ["", ""])[:3]
    return build_folder("relval", dict(arch=arch, tag=tag, variant="" if variant == "_" else variant)) / \
        FILE_NAMES["relval"]


def check_member(name: str, payload: bytes, limit: int | None = None) -> int:
    """Checks one file of a batch and returns its unpacked size. Raises Rejected."""
    if name == PROGRESS_MEMBER:
        check_progress(payload)
        return len(payload)
    m = MEMBER_RE.match(name)
    if not m:
        raise Rejected(400, "bad file name")
    raw = _gunzip(payload, limit)
    data = _parse(raw)
    if m.group("kind") == "relval":
        check_step(data)
    elif data.get("kind") != m.group("kind"):
        raise Rejected(400, f"kind {data.get('kind')!r} in {m.group('kind')}/")
    else:
        target_path(data)
    return len(raw)


def read_tar(payload: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as tar:
        for info in tar:
            if not info.isfile() or info.name in files:
                raise tarfile.TarError(f"not a plain file or twice: {info.name!r}")
            files[info.name] = tar.extractfile(info).read()
    return files


def receive_batch(payload: bytes, root: Path | None = None) -> dict:
    """Checks every file of a batch, saves the good ones as one tar in received/ and returns
    {"saved": {name: sha256}, "rejected": {name: reason}, "kept": {name: sha256}}: kept =
    rejected, but stored in received/rejected/. Nothing is confirmed before it is on disk.
    Raises Rejected for the request as a whole."""
    root = root or received_dir()
    if len(payload) > MAX_BYTES:
        raise Rejected(413, f"batch is {len(payload)} bytes, limit {MAX_BYTES}")
    try:
        files = read_tar(payload)
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise Rejected(400, f"not a tar of plain files ({exc})")
    if not files or len(files) > MAX_FILES:
        raise Rejected(400, f"{len(files)} files in the batch (1 to {MAX_FILES})")
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < MIN_FREE_BYTES:
        raise Rejected(503, "less than 5 GB free on the server")
    good, rejected, budget = {}, {}, MAX_UNPACKED
    for name, data in files.items():
        try:
            budget -= check_member(name, data, budget)      # the whole batch, not each file
            good[name] = data
        except Rejected as exc:
            rejected[name] = exc.reason
    to_save = {n: d for n, d in good.items() if n != PROGRESS_MEMBER or not _progress_known(d, root)}
    if to_save:
        _save_tar(to_save, root)
        if PROGRESS_MEMBER in to_save:
            (root / PROGRESS_KNOWN).write_text(hashlib.sha256(to_save[PROGRESS_MEMBER]).hexdigest())
    return {"saved": {n: hashlib.sha256(d).hexdigest() for n, d in good.items()}, "rejected": rejected,
            "kept": _keep_rejected({n: files[n] for n in rejected}, rejected, root) if rejected else {}}


def _progress_known(payload: bytes, root: Path) -> bool:
    """The sender sends the whole list every run; only a changed one is saved."""
    try:
        return (root / PROGRESS_KNOWN).read_text() == hashlib.sha256(payload).hexdigest()
    except OSError:
        return False


def _keep_rejected(files: dict[str, bytes], reasons: dict[str, str], root: Path) -> dict[str, str]:
    """Stores refused files so they can be looked at (and imported) after a fix; returns what
    is on disk. The names come from outside: only a cleaned copy is used as file name."""
    folder = root / "rejected"
    folder.mkdir(exist_ok=True)
    used = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
    kept = {}
    for name, data in files.items():
        used += len(data)
        if used > MAX_REJECTED_BYTES:
            break
        digest = hashlib.sha256(data).hexdigest()
        clean = re.sub(r"[^A-Za-z0-9_.+-]", "_", name.rsplit("/", 1)[-1])[-150:]
        path = folder / f"{digest[:16]}__{clean}"
        tmp = folder / f".{path.name}.part"
        tmp.write_bytes(data)
        os.replace(tmp, path)
        (folder / f"{path.name}.reason.txt").write_text(f"{name}\n{reasons[name]}\n")
        kept[name] = digest
    return kept


def _save_tar(files: dict[str, bytes], root: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%f")
    dest = root / f"{stamp}-{uuid.uuid4().hex[:8]}.tar"         # sorts by arrival
    tmp = root / f".{dest.name}.part"                           # never seen by the worker (*.tar)
    try:
        with open(tmp, "wb") as fh:
            with tarfile.open(fileobj=fh, mode="w") as tar:
                for name, data in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
        folder = os.open(root, os.O_RDONLY)
        try:
            os.fsync(folder)
        finally:
            os.close(folder)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _scanned_at(path: Path) -> str:
    try:
        return str(json.load(gzip.open(path)).get("source", {}).get("scanned_at") or "")
    except (OSError, ValueError, EOFError):
        return ""   # unreadable existing file: let the new one replace it


def receive(payload: bytes, root: Path | None = None) -> Received:
    """Checks one scan file and saves it. Raises Rejected."""
    root = root or results_dir()
    data = _unpack(payload)
    rel = target_path(data)
    digest = hashlib.sha256(payload).hexdigest()
    dest = root / rel
    status = "saved"
    if dest.exists():
        if hashlib.sha256(dest.read_bytes()).hexdigest() == digest:
            return Received("unchanged", str(rel), digest)
        new_at = str(data["source"].get("scanned_at") or "")
        if new_at and new_at < _scanned_at(dest):
            return Received("older", str(rel), digest)
        status = "replaced"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.part")   # never seen by the importer (*.json.gz)
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return Received(status, str(rel), digest)
