"""Dependency-free checks for the Log RAG core."""
import json
import tempfile
import unittest
from pathlib import Path

from log_rag.pipeline import LogRAGPipeline, parse_json_payload
from log_rag.opensearch import OpenSearchIndexer
from log_rag.retriever import HybridRetriever


class LogRAGTests(unittest.TestCase):
    def test_event_id_is_stable_and_secret_is_redacted(self):
        record = {"timestamp": "2026-08-06T10:00:00Z", "level": "error", "message": "request failed token=abc123", "service": "builder"}
        first = parse_json_payload(json.dumps(record).encode(), "build.json")[0]
        second = parse_json_payload(json.dumps(record).encode(), "build.json")[0]
        self.assertEqual(first.event_id, second.event_id)
        self.assertNotIn("abc123", first.message)
        self.assertIn("[REDACTED]", first.message)

    def test_archive_and_retrieval_preserve_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = LogRAGPipeline(directory)
            result = pipeline.ingest(b'{"timestamp":"2026-08-06T10:00:00Z","level":"ERROR","message":"database timeout in scheduler","service":"scheduler"}\n', source_name="run.jsonl", build="CMSSW_TEST")
            self.assertTrue(result["ok"])
            self.assertEqual(result["archived_events"], 1)
            self.assertTrue((Path(directory) / "manifests" / f"{result['ingestion_id']}.json").exists())
            evidence = HybridRetriever(directory).retrieve("scheduler timeout")
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0]["event_id"], result["event_ids"][0])
            self.assertEqual(evidence[0]["source"], "run.jsonl")

    def test_replay_does_not_duplicate_evidence(self):
        payload = b'{"message":"worker timeout","service":"worker"}\n'
        with tempfile.TemporaryDirectory() as directory:
            pipeline = LogRAGPipeline(directory)
            first = pipeline.ingest(payload, source_name="replay.jsonl")
            second = pipeline.ingest(payload, source_name="replay.jsonl")
            self.assertFalse(first["already_ingested"])
            self.assertTrue(second["already_ingested"])
            self.assertEqual(second["archived_events"], 0)
            self.assertEqual(len(HybridRetriever(directory).retrieve("worker timeout")), 1)

    def test_optional_search_index_is_used(self):
        class FakeIndex:
            def __init__(self):
                self.indexed = []

            def index_events(self, events):
                self.indexed.extend(events)
                return len(events)

            def search(self, question, limit):
                event = self.indexed[0]
                return [{**event.as_dict(), "score": 7.5}]

        with tempfile.TemporaryDirectory() as directory:
            index = FakeIndex()
            result = LogRAGPipeline(directory, search_index=index).ingest(
                b'{"message":"network connection refused","service":"api"}',
                source_name="index.json",
            )
            evidence = HybridRetriever(directory, search_index=index).retrieve("connection refused")
            self.assertEqual(result["indexed_events"], 1)
            self.assertEqual(evidence[0]["event_id"], result["event_ids"][0])
            self.assertEqual(evidence[0]["score"], 7.5)

    def test_opensearch_creates_mapping_before_bulk_index(self):
        class FakeOpenSearch(OpenSearchIndexer):
            def __init__(self):
                super().__init__("http://example.invalid")
                self.calls = []

            def _request(self, method, path, payload=None):
                self.calls.append((method, path))
                return {"errors": False}

        event = parse_json_payload(b'{"message":"index me"}', "index.json")[0]
        index = FakeOpenSearch()
        self.assertEqual(index.index_events([event]), 1)
        self.assertEqual(index.calls, [("PUT", "/cmssdt-log-events"), ("POST", "/_bulk")])

    def test_opensearch_search_allows_question_words_not_in_event(self):
        class FakeOpenSearch(OpenSearchIndexer):
            def __init__(self):
                super().__init__("http://example.invalid")
                self.request = None

            def _request(self, method, path, payload=None):
                self.request = json.loads(payload)
                return {"hits": {"hits": []}}

        index = FakeOpenSearch()
        index.search("Which service reported the timeout?", 6)
        query = index.request["query"]["multi_match"]
        self.assertEqual(query["operator"], "or")
        self.assertEqual(query["minimum_should_match"], "50%")

    def test_graph_enrichment_failure_keeps_local_evidence(self):
        class BrokenDriver:
            def session(self):
                raise RuntimeError("Neo4j unavailable")

        with tempfile.TemporaryDirectory() as directory:
            LogRAGPipeline(directory).ingest(
                b'{"message":"archive remains searchable","service":"api"}',
                source_name="fallback.json",
            )
            evidence = HybridRetriever(directory, neo4j_driver=BrokenDriver()).retrieve("searchable")
            self.assertEqual(len(evidence), 1)


if __name__ == "__main__":
    unittest.main()
