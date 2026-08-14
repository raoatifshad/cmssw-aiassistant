#!/usr/bin/env python3
"""Stream a JSONL file into the local archive, Neo4j, and optional OpenSearch."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "graph"))

from neo4j_conn import get_driver
from log_rag.opensearch import OpenSearchIndexer
from log_rag.pipeline import LogRAGPipeline, Neo4jProjector


def batches(path: Path, max_bytes: int):
    lines: list[bytes] = []
    size = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(line) > max_bytes:
                raise ValueError(f"Line {line_number} exceeds --batch-bytes; increase the limit")
            if lines and size + len(line) > max_bytes:
                yield b"".join(lines)
                lines, size = [], 0
            lines.append(line)
            size += len(line)
    if lines:
        yield b"".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL input file")
    parser.add_argument("--archive-dir", default=os.environ.get("LOG_RAG_ARCHIVE_DIR", "data/log-rag"))
    parser.add_argument("--build", help="Default build tag for events without one")
    parser.add_argument("--batch-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--project-neo4j", action="store_true", help="Project event correlations to Neo4j")
    parser.add_argument("--opensearch-url", default=os.environ.get("OPENSEARCH_URL"))
    parser.add_argument("--opensearch-index", default=os.environ.get("OPENSEARCH_INDEX", "cmssdt-log-events"))
    args = parser.parse_args()

    if args.batch_bytes < 1024:
        parser.error("--batch-bytes must be at least 1024")
    if not args.path.is_file():
        parser.error(f"Input file does not exist: {args.path}")

    driver = get_driver() if args.project_neo4j else None
    indexer = OpenSearchIndexer(args.opensearch_url, args.opensearch_index) if args.opensearch_url else None
    pipeline = LogRAGPipeline(
        args.archive_dir,
        projector=Neo4jProjector(driver) if driver else None,
        search_index=indexer,
    )
    total = 0
    try:
        for chunk_number, payload in enumerate(batches(args.path, args.batch_bytes), start=1):
            result = pipeline.ingest(payload, source_name=str(args.path), build=args.build)
            total += result["archived_events"]
            print(
                f"chunk={chunk_number} archived={result['archived_events']} "
                f"projected={result['projected_events']} indexed={result['indexed_events']}"
            )
    finally:
        if driver:
            driver.close()
    print(f"Completed JSONL ingestion: {total} newly archived events")


if __name__ == "__main__":
    main()
