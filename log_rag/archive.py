"""Append-only local archive, manifests, and evidence store."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .models import LogEvent


class Archive:
    """Filesystem-backed immutable archive suitable for local/dev deployment.

    Each source payload is content-addressed; normalized events and evidence are
    JSONL append-only records with enough provenance to replay ingestion.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.manifests = self.root / "manifests"
        self.events_path = self.root / "events.jsonl"
        self.evidence_path = self.root / "evidence.jsonl"
        for path in (self.raw, self.manifests):
            path.mkdir(parents=True, exist_ok=True)

    def archive_payload(self, payload: bytes, source_name: str) -> dict:
        digest = hashlib.sha256(payload).hexdigest()
        ingestion_id = hashlib.sha256(f"{source_name}\0{digest}".encode()).hexdigest()
        suffix = Path(source_name).suffix or ".log"
        raw_path = self.raw / f"{digest}{suffix}"
        if not raw_path.exists():
            raw_path.write_bytes(payload)
        return {
            "source_name": source_name,
            "sha256": digest,
            "ingestion_id": ingestion_id,
            "path": str(raw_path.relative_to(self.root)),
        }

    def append(self, events: Iterable[LogEvent], payload_manifest: dict) -> tuple[int, bool]:
        event_rows = [event.as_dict() for event in events]
        if not event_rows:
            return 0, False
        manifest_path = self.manifests / f"{payload_manifest['ingestion_id']}.json"
        if manifest_path.exists():
            return 0, True
        with self.events_path.open("a", encoding="utf-8") as events_file, self.evidence_path.open("a", encoding="utf-8") as evidence_file:
            for row in event_rows:
                events_file.write(json.dumps(row, default=str, sort_keys=True) + "\n")
                evidence_file.write(json.dumps({"event_id": row["event_id"], "text": LogEvent(**row).evidence_text(), "timestamp": row["timestamp"], "source": row["source"], "build": row["build"]}, default=str) + "\n")
        manifest = {**payload_manifest, "event_count": len(event_rows), "event_ids": [row["event_id"] for row in event_rows]}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return len(event_rows), False

    def evidence(self) -> Iterable[dict]:
        if not self.evidence_path.exists():
            return []
        with self.evidence_path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
