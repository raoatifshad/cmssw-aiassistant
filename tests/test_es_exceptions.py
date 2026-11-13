import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
from relval_log_parser import extract_es_exception_blocks


class EsExceptionParserTests(unittest.TestCase):
    def test_single_block_without_begin_end_markers(self):
        text = (
            "An exception of category 'LogicError' occurred while\n"
            "[0] Processing  Event run: 1 lumi: 1 event: 1 stream: 0\n"
            "[1] Calling method for module EvaluateMuonMVAID/'muonMVAID'\n"
            "Exception Message:\n"
            "A muon candidate track has more than 5 stations with matching segments.\n"
        )
        blocks = extract_es_exception_blocks(text)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["category"], "LogicError")
        self.assertIn("more than 5 stations", blocks[0]["message"])
        self.assertIn("EvaluateMuonMVAID", blocks[0]["context"])

    def test_multiple_concatenated_blocks_of_the_same_category(self):
        text = (
            "An exception of category 'WrongTrackerSubDet' occurred while\n"
            "[0] Processing  Event run: 1 lumi: 1 event: 1 stream: 0\n"
            "Exception Message:\n"
            "Invalid DetID: raw ID 111\n"
            "An exception of category 'WrongTrackerSubDet' occurred while\n"
            "[0] Processing  Event run: 1 lumi: 1 event: 2 stream: 1\n"
            "Exception Message:\n"
            "Invalid DetID: raw ID 222\n"
        )
        blocks = extract_es_exception_blocks(text)

        self.assertEqual(len(blocks), 2)
        self.assertIn("raw ID 111", blocks[0]["message"])
        self.assertIn("raw ID 222", blocks[1]["message"])

    def test_empty_text_returns_no_blocks(self):
        self.assertEqual(extract_es_exception_blocks(""), [])


class EsExceptionIngestionTests(unittest.TestCase):
    @patch("build_graph.source_needs_import", return_value=True)
    @patch("build_graph.mark_source_imported")
    @patch("build_graph.file_hash", return_value="deadbeef")
    def test_ingest_es_exceptions_parses_every_document_into_candidate_records(self, file_hash, mark, needs_import):
        sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
        import build_graph

        payload = {
            "hits": {
                "hits": [
                    {"_source": {
                        "release": "CMSSW_20_1_X_2026-08-17-2300",
                        "architecture": "el9_amd64_gcc14",
                        "workflow": "140.051",
                        "step": "step3",
                        "exception": (
                            "An exception of category 'LogicError' occurred while\n"
                            "[0] Calling method for module EvaluateMuonMVAID/'muonMVAID'\n"
                            "Exception Message:\n"
                            "A muon candidate track has more than 5 stations.\n"
                        ),
                    }},
                    {"_source": {
                        "release": "CMSSW_20_1_X_2026-08-17-2300",
                        "architecture": "el9_amd64_gcc14",
                        "workflow": "999999.0",  # not a known RelVal -- must be silently skipped
                        "step": "step1",
                        "exception": "An exception of category 'Unknown' occurred while\nException Message:\nx\n",
                    }},
                ]
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            fake_path = Path(tmp) / "exceptions.json"
            fake_path.write_text(json.dumps(payload))
            driver = MagicMock()
            build_graph.ingest_es_exceptions(driver, fake_path, force=True)

        session = driver.session.return_value.__enter__.return_value
        # Both parsed records are sent to Cypher — the MATCH (relval:RelVal
        # {key:...}) clause is what actually drops the unmatched workflow, which
        # a mocked driver can't observe (verified separately against the live
        # graph: 2555 of 4567 candidate records matched a known RelVal).
        unwind_call = [c for c in session.run.call_args_list if "UNWIND $records" in c.args[0]][0]
        records = unwind_call.kwargs["records"]
        self.assertEqual(len(records), 2)
        by_workflow = {r["workflow_id"]: r for r in records}
        self.assertEqual(by_workflow["140.051"]["category"], "LogicError")
        self.assertEqual(by_workflow["999999.0"]["category"], "Unknown")


class EsExceptionReimportTests(unittest.TestCase):
    @patch("build_graph.source_needs_import", return_value=True)
    @patch("build_graph.mark_source_imported")
    @patch("build_graph.file_hash", return_value="deadbeef")
    def test_reimport_of_the_same_path_keeps_exceptions_that_left_the_export_window(self, file_hash, mark, needs_import):
        # G2-1 (2026-11-12): the 12:30 refresh imports the same dated file as the 00:30 one; the
        # delete before the import must only cover RelVals the new export still contains.
        sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
        import build_graph

        doc = {"release": "CMSSW_20_1_X_2026-11-10-2300", "architecture": "el9_amd64_gcc14", "workflow": "537.0",
               "step": "step1", "exception": "An exception of category 'OtherCMS' occurred while\nException Message:\nx\n"}
        with tempfile.TemporaryDirectory() as tmp:
            fake_path = Path(tmp) / "exceptions-2026-11-11.json"
            fake_path.write_text(json.dumps({"hits": {"hits": [{"_source": doc}]}}))
            driver = MagicMock()
            build_graph.ingest_es_exceptions(driver, fake_path, force=True)

        session = driver.session.return_value.__enter__.return_value
        deletes = [c for c in session.run.call_args_list if "DELETE" in c.args[0]]
        self.assertEqual(len(deletes), 1)
        self.assertIn("relval.key IN $relval_keys", deletes[0].args[0])
        self.assertEqual(deletes[0].kwargs["relval_keys"], ["CMSSW_20_1_X_2026-11-10-2300:el9_amd64_gcc14:537.0"])


if __name__ == "__main__":
    unittest.main()
