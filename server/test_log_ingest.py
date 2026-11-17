"""Receiving end of the IB test-log scan files: POST /api/test-logs/ingest.

cms-bot's jobs on cmssdt9-02 (ib-build-logs, process-relval-logs) write one
`<name>-scan.json.gz` per test-log zip; a sender job on the same machine
(deploy/cmssdt/send_test_log_scans.py) posts each new or changed one here.
This module only CHECKS and SAVES the file; the worker imports
results/ into Neo4j every 30 min (ingest_test_logs.py), so a slow or failing
import never keeps the sender waiting and a received file is never lost.

Saved where the copied-zip pipeline (fetch_ib_test_logs.result_path) saves the
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

import gzip
import hashlib
import hmac
import io
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

TOKEN_ENV = "TEST_LOGS_INGEST_TOKEN"
RESULTS_ENV = "TEST_LOG_RESULTS_DIR"
MAX_BYTES = 32 * 1024 * 1024            # compressed; real files are KBs .. ~3 MB
MAX_UNPACKED = 512 * 1024 * 1024        # guards against a gzip bomb

FILE_NAMES = {"unit": "unitTestLogs.json.gz", "addon": "addOnTests.json.gz", "relval": "pyRelValMatrixLogs.json.gz"}
ITEMS_KEY = {"unit": "tests", "addon": "tests", "relval": "workflows"}
ARCH_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+_[a-z0-9]+$")          # el9_amd64_gcc14
TAG_RE = re.compile(r"^CMSSW_\d+_\d+_[A-Za-z0-9_]*X_\d{4}-\d{2}-\d{2}-\d{4}$")
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


def _unpack(payload: bytes) -> dict:
    if not payload:
        raise Rejected(400, "empty body")
    if len(payload) > MAX_BYTES:
        raise Rejected(413, f"file is {len(payload)} bytes, limit {MAX_BYTES}")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as fh:
            raw = fh.read(MAX_UNPACKED + 1)
    except (OSError, EOFError) as exc:
        raise Rejected(400, f"not a gzip file ({exc})")
    if len(raw) > MAX_UNPACKED:
        raise Rejected(413, f"unpacks to more than {MAX_UNPACKED} bytes")
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
    arch, tag, variant = source.get("arch"), source.get("tag"), source.get("variant", "")
    if not isinstance(arch, str) or not ARCH_RE.match(arch):
        raise Rejected(400, f"bad architecture {arch!r}")
    if not isinstance(tag, str) or not TAG_RE.match(tag):
        raise Rejected(400, f"bad IB tag {tag!r}")
    if not isinstance(variant, str) or (variant and not VARIANT_RE.match(variant)):
        raise Rejected(400, f"bad variant {variant!r}")
    if not isinstance(data.get(ITEMS_KEY[kind]), list) or not isinstance(data.get("problems"), list):
        raise Rejected(400, f"a {kind} result needs '{ITEMS_KEY[kind]}' and 'problems' lists")
    if not isinstance(source.get("scanner_version"), int):
        raise Rejected(400, "source.scanner_version missing")
    return Path(kind) / arch / tag / (variant.replace("/", "_") or "_") / FILE_NAMES[kind]


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
