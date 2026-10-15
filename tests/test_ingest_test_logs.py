import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
import ingest_test_logs as itl

TAG = "CMSSW_20_1_ASAN_X_2026-10-05-2300"
SOURCE = dict(kind="unit", arch="el9_amd64_gcc14", tag=TAG, variant="", queue="20.1.ASAN-mon-23",
              path=f"el9_amd64_gcc14/www/mon/20.1.ASAN-mon-23/{TAG}/unitTestLogs.zip", size=7, scanner_version=2)


def problem(test, status, kind="fatal_exception", sid="aaaa", count=1, **extra):
    base = dict(kind=kind, category="DictionaryNotFound", signature=f"{kind}|x|y", signature_id=sid,
                summary="Fatal exception 'DictionaryNotFound'", message="No data dictionary found",
                block=["----- Begin Fatal Exception", "An exception of category 'DictionaryNotFound'"],
                package="Alignment/OfflineValidation", test=test, test_status=status, count=count,
                member="unitTestLogs/Alignment/OfflineValidation/unitTest.log", line_start=10)
    return dict(base, **extra)


UNIT = dict(kind="unit", source=SOURCE,
            tests=[dict(test="SubmitPVrbr", status="passed", package="Alignment/OfflineValidation"),
                   dict(test="SubmitPVsplit", status="passed", package="Alignment/OfflineValidation"),
                   dict(test="testHltDiff", status="failed", package="HLTrigger/Tools")],
            problems=[problem("SubmitPVrbr", "passed"), problem("SubmitPVsplit", "passed", count=2),
                      problem("SubmitPVrbr", "passed", kind="message_logger", sid="bbbb", count=40),
                      problem("testHltDiff", "failed", kind="generic_error_line", sid="cccc", package="HLTrigger/Tools")])


class RunRecordTests(unittest.TestCase):
    def test_every_test_is_kept_with_its_status(self):
        run = itl.run_record(UNIT)
        self.assertEqual(run["key"], f"{TAG}:el9_amd64_gcc14:unit:-")
        self.assertEqual((run["total"], run["passed"], run["failed"]), (3, 2, 1))
        self.assertEqual(run["failed_tests"], ["HLTrigger/Tools/testHltDiff"])
        self.assertIn("Alignment/OfflineValidation/SubmitPVrbr", run["passed_tests"])
        self.assertEqual(run["server_path"], f"/data/sdt/buildlogs/{SOURCE['path']}")

    def test_relval_failed_workflow_keeps_step_statuses_and_exit_codes(self):
        data = dict(kind="relval", source=dict(SOURCE, kind="relval", variant="gpu/nvidia_t4"), problems=[],
                    workflows=[dict(workflow="140.051_RunMuonEG2022C", steps={"1": "PASSED", "2": "FAILED"},
                                    exit_codes=[0, 256], status="failed")])
        run = itl.run_record(data)
        self.assertTrue(run["key"].endswith(":relval:gpu/nvidia_t4"))
        self.assertEqual(run["failed_details"], ["140.051_RunMuonEG2022C: step1=PASSED step2=FAILED; exit 0 256"])


class ProblemRecordTests(unittest.TestCase):
    def test_one_record_per_distinct_problem_listing_every_test_that_hit_it(self):
        records = {r["kind"]: r for r in itl.problem_records(UNIT, "run")}
        self.assertEqual(len(records), 3)
        dict_nf = records["fatal_exception"]
        self.assertEqual(dict_nf["owners"], ["Alignment/OfflineValidation/SubmitPVrbr",
                                             "Alignment/OfflineValidation/SubmitPVsplit"])
        self.assertEqual(dict_nf["count"], 3)
        self.assertTrue(dict_nf["in_passed"] and dict_nf["serious"])
        self.assertFalse(dict_nf["in_failed"])
        self.assertIn("DictionaryNotFound", dict_nf["text"])

    def test_first_seen_is_tracked_for_serious_and_generic_but_not_messagelogger(self):
        records = itl.problem_records(UNIT, "run")
        sid = {r["kind"]: r["signature_id"] for r in records}
        firsts = itl.first_seen_records(UNIT, records)
        # keyed per release series: CMSSW_20_1 (the ASAN sub-IB shares it)
        self.assertEqual(sorted(f["key"] for f in firsts), sorted([
            f"unit|CMSSW_20_1|Alignment/OfflineValidation/SubmitPVrbr|{sid['fatal_exception']}",
            f"unit|CMSSW_20_1|Alignment/OfflineValidation/SubmitPVsplit|{sid['fatal_exception']}",
            f"unit|CMSSW_20_1|HLTrigger/Tools/testHltDiff|{sid['generic_error_line']}"]))

    def test_older_results_are_re_signed_so_the_same_problem_compares_equal(self):
        # v3 results: the Madgraph worker dir (thread0/thread1) was part of the signature
        def traceback(test, thread):
            sig = (f"python_traceback|IndexError|IndexError:list index out of range @ "
                   f"$RELEASE/pyRelval/<WF>/{thread}/lheevent/bin/internal/lhe_parser.py")
            return problem(test, "passed", kind="python_traceback", signature=sig, signature_id=thread)
        data = dict(UNIT, problems=[traceback("SubmitPVrbr", "thread0"), traceback("SubmitPVsplit", "thread3")])
        records = itl.problem_records(data, "run")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["owners"], ["Alignment/OfflineValidation/SubmitPVrbr",
                                                "Alignment/OfflineValidation/SubmitPVsplit"])
        self.assertIn("thread<N>", records[0]["signature"])

    def test_older_glued_signal_crashes_merge_and_get_a_clean_label(self):
        # v5 results: thread output glued onto the signal line made its own signature
        def crash(test, what):
            return problem(test, "passed", kind="crash", category=what, signature=f"crash|{what}|",
                           signature_id=what, message=f"A fatal system signal has occurred: {what}",
                           summary=f"cmsRun crash: fatal system signal ({what})")
        data = dict(UNIT, problems=[crash("SubmitPVrbr", "abort signal"),
                                    crash("SubmitPVsplit", ": index abort signal-916979453")])
        records = itl.problem_records(data, "run")
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0]["category"], records[0]["signature"]), ("abort signal", "crash|abort signal|"))
        self.assertEqual(records[0]["summary"], "cmsRun crash: fatal system signal (abort signal)")
        self.assertEqual(len(records[0]["owners"]), 2)

    def test_release_series_of_primary_and_sub_ib_tags(self):
        self.assertEqual(itl.release_series("CMSSW_20_1_ASAN_X_2026-10-05-2300"), "CMSSW_20_1")
        self.assertEqual(itl.release_series("CMSSW_13_2_X_2026-10-05-2300"), "CMSSW_13_2")

    def test_relval_owner_includes_the_step(self):
        p = dict(problem(None, "passed"), workflow="1.0_ProdMinBias", step=3)
        self.assertEqual(itl.owner_name("relval", p), "1.0_ProdMinBias:step3")


if __name__ == "__main__":
    unittest.main()


class ProblemDetailTests(unittest.TestCase):
    """The evidence is one example; the modules and names of every record are kept."""

    def test_modules_of_every_test_are_kept_and_indexed_whole(self):
        def raised_in(test, module):
            return problem(test, "passed", block=[
                "----- Begin Fatal Exception", "An exception of category 'DictionaryNotFound' occurred while",
                f"   [1] Calling method for module {module}/'label{test}'", "Exception Message:",
                "No data dictionary found", "----- End Fatal Exception"])
        data = dict(UNIT, problems=[raised_in("SubmitPVrbr", "PrimaryVertexValidation"),
                                    raised_in("SubmitPVsplit", "SplitVertexResolution")])
        rec = itl.problem_records(data, "run")[0]
        self.assertEqual(rec["modules"], ["PrimaryVertexValidation/'labelSubmitPVrbr'",
                                          "SplitVertexResolution/'labelSubmitPVsplit'"])
        self.assertEqual(rec["context_keys"], ["DictionaryNotFound|PrimaryVertexValidation",
                                               "DictionaryNotFound|SplitVertexResolution"])
        self.assertIn("splitvertexresolution", rec["names_text"].split())
        self.assertIn("dictionarynotfound|splitvertexresolution", rec["context_keys_text"].split())
        # the second test's context is searchable text too, not only the first example
        self.assertIn("SplitVertexResolution", rec["text"])
        self.assertNotIn("SplitVertexResolution", rec["evidence"])
