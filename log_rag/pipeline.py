"""Adapters, ingestion orchestration, and Neo4j event projection."""
from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

from .archive import Archive
from .models import LogEvent


def parse_json_payload(payload: bytes, source: str, build: str | None = None) -> list[LogEvent]:
    """Normalize a JSON object, JSON array, or JSONL payload into canonical events."""
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        decoded: Any = json.loads(text)
        records = decoded if isinstance(decoded, list) else [decoded]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [LogEvent.from_record(record if isinstance(record, dict) else {"message": record}, source=source, default_build=build) for record in records]


def parse_clang_html(payload: bytes, source: str, build: str | None = None) -> list[LogEvent]:
    """Adapt a Clang log.html file using the repository's validated extractor."""
    from ingestion.clang_parser import extract_warnings, html_to_text
    warnings = extract_warnings(html_to_text(payload.decode("utf-8", errors="replace")))
    return [LogEvent.from_record({"timestamp": None, "severity": "WARNING", "message": warning["message"], "file": warning["file"], "line": warning["line"], "checker": warning["flag"], "snippet": warning["snippet"]}, source=source, default_build=build) for warning in warnings]


class Neo4jProjector:
    """Idempotently projects log events into the existing Neo4j knowledge graph."""
    def __init__(self, driver):
        self.driver = driver

    def project(self, events: Iterable[LogEvent]) -> int:
        rows = [event.as_dict() for event in events]
        if not rows:
            return 0
        query = """
        UNWIND $events AS event
        MERGE (occ:EventOccurrence {event_id:event.event_id})
        SET occ.timestamp=event.timestamp, occ.source=event.source,
            occ.message=event.message, occ.severity=event.severity,
            occ.source_hash=event.source_hash, occ.metadata_json=event.metadata_json
        FOREACH (_ IN CASE WHEN event.build IS NULL THEN [] ELSE [1] END |
          MERGE (build:Build {tag:event.build})
          MERGE (build)-[:CONTAINS_EVENT]->(occ))
        FOREACH (_ IN CASE WHEN event.service IS NULL THEN [] ELSE [1] END |
          MERGE (service:Service {name:event.service})
          MERGE (occ)-[:EMITTED_BY]->(service))
        FOREACH (_ IN CASE WHEN event.host IS NULL THEN [] ELSE [1] END |
          MERGE (host:Host {name:event.host})
          MERGE (occ)-[:OBSERVED_ON]->(host))
        """
        params = [{**row, "metadata_json": json.dumps(row.pop("metadata"), default=str)} for row in rows]
        with self.driver.session() as session:
            session.run(query, events=params).consume()
        return len(params)


class LogRAGPipeline:
    def __init__(self, archive_root: str | Path, projector: Neo4jProjector | None = None, search_index=None):
        self.archive = Archive(archive_root)
        self.projector = projector
        self.search_index = search_index

    def ingest(self, payload: bytes, *, source_name: str, content_type: str = "application/json", build: str | None = None) -> dict:
        if "html" in content_type or source_name.endswith(".html"):
            events = parse_clang_html(payload, source_name, build)
        else:
            events = parse_json_payload(payload, source_name, build)
        manifest = self.archive.archive_payload(payload, source_name)
        archived, already_ingested = self.archive.append(events, manifest)
        projected = self.projector.project(events) if self.projector else 0
        indexed = self.search_index.index_events(events) if self.search_index else 0
        return {"ok": True, "source": source_name, "sha256": manifest["sha256"], "ingestion_id": manifest["ingestion_id"], "already_ingested": already_ingested, "archived_events": archived, "projected_events": projected, "indexed_events": indexed, "event_ids": [event.event_id for event in events]}
