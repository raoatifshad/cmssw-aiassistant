"""Small OpenSearch adapter for canonical log events.

Uses the standard library so local archive-only mode has no additional Python
runtime dependency. The adapter is intentionally optional.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .models import LogEvent


class OpenSearchIndexer:
    def __init__(self, url: str, index: str = "cmssdt-log-events", timeout: float = 10.0):
        self.url = url.rstrip("/")
        self.index = index
        self.timeout = timeout
        self._index_ready = False

    def _request(self, method: str, path: str, payload: bytes | None = None) -> dict[str, Any]:
        request = Request(
            f"{self.url}{path}",
            data=payload,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            body = response.read()
        return json.loads(body) if body else {}

    def ensure_index(self) -> None:
        body = {
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date", "ignore_malformed": True},
                    "message": {"type": "text"},
                    "severity": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "service": {"type": "keyword"},
                    "host": {"type": "keyword"},
                    "build": {"type": "keyword"},
                    "metadata": {"type": "flattened"},
                }
            }
        }
        try:
            self._request("PUT", f"/{self.index}", json.dumps(body).encode())
        except HTTPError as exc:
            if exc.code != 400:
                raise
        self._index_ready = True

    def index_events(self, events: list[LogEvent]) -> int:
        if not events:
            return 0
        if not self._index_ready:
            self.ensure_index()
        lines: list[str] = []
        for event in events:
            lines.append(json.dumps({"index": {"_index": self.index, "_id": event.event_id}}))
            lines.append(json.dumps(event.as_dict(), default=str))
        result = self._request("POST", "/_bulk", ("\n".join(lines) + "\n").encode())
        if result.get("errors"):
            raise RuntimeError("OpenSearch rejected one or more log events")
        return len(events)

    def search(self, question: str, limit: int) -> list[dict[str, Any]]:
        query = {
            "size": limit,
            "query": {
                "multi_match": {
                    "query": question,
                    "fields": ["message^5", "service^3", "host^2", "build^2", "severity", "metadata.*"],
                    "operator": "or",
                    "minimum_should_match": "50%",
                }
            },
        }
        result = self._request("POST", f"/{self.index}/_search", json.dumps(query).encode())
        return [
            {"score": hit.get("_score", 0.0), **hit.get("_source", {})}
            for hit in result.get("hits", {}).get("hits", [])
        ]
