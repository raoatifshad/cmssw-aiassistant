"""Hybrid evidence retriever with deterministic, inspectable citations."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .archive import Archive

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_./:-]{2,}")


@dataclass
class Citation:
    event_id: str
    source: str
    timestamp: str
    build: str | None
    excerpt: str
    score: float


class HybridRetriever:
    """Combine local evidence scoring with Neo4j neighbourhood expansion.

    Local scoring is deliberately dependency-free and transparent. It is a
    production-safe fallback when an embedding/vector service is unavailable.
    """
    def __init__(self, archive_root: str | Path, neo4j_driver=None, search_index=None):
        self.archive = Archive(archive_root)
        self.neo4j_driver = neo4j_driver
        self.search_index = search_index

    def retrieve(self, question: str, limit: int = 6) -> list[dict]:
        tokens = set(token.lower() for token in _TOKEN_RE.findall(question))
        if self.search_index:
            try:
                results = [
                    asdict(Citation(
                        event_id=item["event_id"],
                        source=item["source"],
                        timestamp=item["timestamp"],
                        build=item.get("build"),
                        excerpt=f"[{item['timestamp']}] {item['severity']} {item['message']}"[:360],
                        score=round(float(item["score"]), 3),
                    ))
                    for item in self.search_index.search(question, limit)
                ]
            except Exception:
                results = self._local_results(tokens, limit)
        else:
            results = self._local_results(tokens, limit)
        if self.neo4j_driver and results:
            event_ids = [item["event_id"] for item in results]
            try:
                with self.neo4j_driver.session() as session:
                    rows = session.run("MATCH (e:EventOccurrence) WHERE e.event_id IN $ids OPTIONAL MATCH (e)-[:EMITTED_BY]->(s:Service) OPTIONAL MATCH (e)-[:OBSERVED_ON]->(h:Host) RETURN e.event_id AS event_id, collect(DISTINCT s.name) AS services, collect(DISTINCT h.name) AS hosts", ids=event_ids)
                    context = {row["event_id"]: {"services": row["services"], "hosts": row["hosts"]} for row in rows}
                for item in results:
                    item["graph_context"] = context.get(item["event_id"], {})
            except Exception:
                pass
        return results

    def _local_results(self, tokens: set[str], limit: int) -> list[dict]:
        scored = []
        for item in self.archive.evidence():
            text = item["text"].lower()
            overlap = sum(token in text for token in tokens)
            if overlap:
                excerpt = item["text"].replace("\n", " ")[:360]
                scored.append(Citation(event_id=item["event_id"], source=item["source"], timestamp=item["timestamp"], build=item.get("build"), excerpt=excerpt, score=round(overlap / max(len(tokens), 1), 3)))
        scored.sort(key=lambda citation: (citation.score, citation.timestamp), reverse=True)
        return [asdict(citation) for citation in scored[:limit]]
