import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
import ib_test_log_scanner as scanner

# Snippets below are trimmed from real IB logs (CMSSW_20_1_ASAN_X_2026-10-05-2300
# unit tests, CMSSW_20_1_X_2026-10-05-1100 RelVals, el9_amd64_gcc14).

SED_LEAK = """\
=================================================================
=={pid}==ERROR: LeakSanitizer: detected memory leaks
Direct leak of 1 byte(s) in 1 object(s) allocated from:
    #0 {a}7757 in malloc ../../../../libsanitizer/asan/asan_malloc_linux.cpp:69
    #1 {a}2414 in compile_program (/usr/bin/sed+0xe414) (BuildId: 8bafddaae5acece3aaa264ea74142ef83b25aca2)
SUMMARY: AddressSanitizer: 1 byte(s) leaked in 1 allocation(s).
"""

HGCAL_OVERFLOW = """\
==837850==ERROR: AddressSanitizer: global-buffer-overflow on address 0x14c2f82dcfd8 at pc 0x14c2f8293bdb bp 0x7ffc8bd5fab0 sp 0x7ffc8bd5faa8
READ of size 4 at 0x14c2f82dcfd8 thread T0
%MSG-e FWRecoGeometry:   FWRecoGeometryESProducer:FWRecoGeometryESProducer@callESModule  06-Oct-2026 04:20:19 CEST Run: 1 Event: 1
 ME0 geometry not found An exception of category 'NoGeometry' occurred.
%MSG
    #0 0x14aa6f8acbda in HGCalGeometry::getNewCorners(DetId const&, bool) const (/cvmfs/cms-ib.cern.ch/sw/x86_64/nweek-02962/el9_amd64_gcc14/cms/cmssw/CMSSW_20_1_ASAN_X_2026-10-05-2300/lib/el9_amd64_gcc14/libGeometryHGCalGeometry.so+0x8bbda)
    #1 0x14aa6baf2635 in FWRecoGeometryESProducer::addCaloGeometry(FWRecoGeometry&) (/cvmfs/cms-ib.cern.ch/sw/x86_64/nweek-02962/el9_amd64_gcc14/cms/cmssw/CMSSW_20_1_ASAN_X_2026-10-05-2300/lib/el9_amd64_gcc14/pluginFireworksGeometry.so+0x29635)
SUMMARY: AddressSanitizer: global-buffer-overflow (/cvmfs/cms-ib.cern.ch/sw/x86_64/nweek-02962/el9_amd64_gcc14/cms/cmssw/CMSSW_20_1_ASAN_X_2026-10-05-2300/lib/el9_amd64_gcc14/libGeometryHGCalGeometry.so+0x8bbda)
"""

DICTIONARY_NOT_FOUND = """\
----- Begin Fatal Exception 06-Oct-2026 03:53:46 CEST-----------------------
An exception of category 'DictionaryNotFound' occurred while
   [0] Constructing the EventProcessor
   [1] Calling ProductRegistry::initializeLookupTables
Exception Message:
No data dictionary found for the following classes:

  DCSRecord
----- End Fatal Exception -------------------------------------------------
"""


def unit_log(*tests):
    """tests: (name, body, status) -> a package unitTest.log in the real layout."""
    out = []
    for name, body, status in tests:
        out.append(f'===== Test "{name}" ====\n')
        out.append(body)
        out.append(f"---> test {name} {status}\nTestTime:8\n^^^^ End Test {name} ^^^^\n")
    out.append(">> Tests for package Alignment/OfflineValidation ran.\n")
    return "".join(out).splitlines(keepends=True)


class UnitTestSpanTests(unittest.TestCase):
    def test_nested_command_headers_stay_inside_the_enclosing_test(self):
        lines = (
            '===== Test "TestDQMOnlineClients-compilation_commissioning_run" ====\n'
            '===== Test "python3 /data/cmsbld/jenkins/workspace/ib-run-qa/CMSSW_X/src/DQM/a_cfg.py" ====\n'
            "running\n"
            '===== Test "python3 /data/cmsbld/jenkins/workspace/ib-run-qa/CMSSW_X/src/DQM/b_cfg.py" ====\n'
            "---> test TestDQMOnlineClients-compilation_commissioning_run succeeded\n"
            "^^^^ End Test TestDQMOnlineClients-compilation_commissioning_run ^^^^\n"
        ).splitlines(keepends=True)
        spans = scanner.unit_test_spans(lines)

        self.assertEqual([(s["test"], s["status"], s["line_start"], s["line_end"]) for s in spans],
                         [("TestDQMOnlineClients-compilation_commissioning_run", "passed", 1, 6)])

    def test_had_errors_is_failed(self):
        spans = scanner.unit_test_spans(unit_log(("PrimaryVertex", "x\n", "had ERRORS")))
        self.assertEqual(spans[0]["status"], "failed")


class ProblemInPassedTestTests(unittest.TestCase):
    def test_fatal_exception_in_a_passed_test_is_attributed_to_that_test(self):
        lines = unit_log(("BeamSpotPlotting", "fine\n", "succeeded"),
                         ("SubmitPVrbr", DICTIONARY_NOT_FOUND, "succeeded"))
        tests, problems = scanner.scan_unit_test_log(lines, "Alignment/OfflineValidation")

        fatal = [p for p in problems if p["kind"] == "fatal_exception"]
        self.assertEqual(len(fatal), 1)
        self.assertEqual(fatal[0]["test"], "SubmitPVrbr")
        self.assertEqual(fatal[0]["test_status"], "passed")
        self.assertEqual(fatal[0]["category"], "DictionaryNotFound")
        self.assertEqual(fatal[0]["origin"], "cmssw")
        self.assertIn("No data dictionary found", fatal[0]["message"])
        self.assertEqual(fatal[0]["block"][0][:30], "----- Begin Fatal Exception 06")

    def test_lines_inside_a_fatal_exception_are_not_also_generic_error_lines(self):
        problems = scanner.scan_lines(DICTIONARY_NOT_FOUND.splitlines())
        self.assertEqual([p["kind"] for p in problems], ["fatal_exception"])


class SanitizerTests(unittest.TestCase):
    def test_leak_in_sed_is_infrastructure_and_identical_across_pids_and_addresses(self):
        text = SED_LEAK.format(pid=735950, a="0x14f101cf") + SED_LEAK.format(pid=735963, a="0x14d1a58f")
        issues = scanner.scan_lines(text.splitlines())

        self.assertEqual(len(issues), 2)
        self.assertTrue(all(i["origin"] == "infrastructure" for i in issues))
        self.assertEqual(issues[0]["signature"], issues[1]["signature"])
        self.assertIn("compile_program in sed", issues[0]["summary"])
        grouped = scanner.aggregate(issues, dict(test="BeamSpotPlotting"))
        self.assertEqual([g["count"] for g in grouped.values()], [2])

    def test_overflow_in_a_cmssw_library_is_cmssw_and_names_the_function(self):
        issues = scanner.scan_lines(HGCAL_OVERFLOW.splitlines())
        sanitizer = [i for i in issues if i["kind"] == "sanitizer"]

        self.assertEqual(len(sanitizer), 1)
        self.assertEqual(sanitizer[0]["origin"], "cmssw")
        self.assertEqual(sanitizer[0]["category"], "AddressSanitizer: global-buffer-overflow")
        self.assertIn("HGCalGeometry::getNewCorners in libGeometryHGCalGeometry.so", sanitizer[0]["summary"])
        # the MessageLogger block printed in the middle of the report is still its own problem
        self.assertIn("message_logger", [i["kind"] for i in issues])

    def test_asan_custom_signal_handler_notice_is_not_a_crash(self):
        issues = scanner.scan_lines([
            "==1==WARNING: Found custom handler for signal 11 (Segmentation fault) preinstalled.\n"])
        self.assertNotIn("crash", [i["kind"] for i in issues])


class MessageLoggerTests(unittest.TestCase):
    def test_messages_group_by_severity_and_category_not_by_text(self):
        text = (
            "%MSG-w SiStripBadChannelPatcher:  SiStripBadChannelPatcher:patcher  06-Oct-2026 04:01:02 CEST Run: 1\n"
            "module 470066180 has 3 bad strips\n%MSG\n"
            "%MSG-w SiStripBadChannelPatcher:  SiStripBadChannelPatcher:patcher  06-Oct-2026 04:01:03 CEST Run: 1\n"
            "module 470066181 has no cabling\n%MSG\n"
        )
        grouped = scanner.aggregate(scanner.scan_lines(text.splitlines()), {})

        self.assertEqual(len(grouped), 1)
        record = next(iter(grouped.values()))
        self.assertEqual((record["severity"], record["category"], record["count"]),
                         ("warning", "SiStripBadChannelPatcher", 2))

    def test_info_messages_are_dropped(self):
        text = "%MSG-i ExceptionError: An error occurred but is harmless\n%MSG\n"
        self.assertEqual(scanner.scan_lines(text.splitlines()), [])

    def test_unterminated_message_at_end_of_log_is_kept_and_marked_truncated(self):
        issues = scanner.scan_lines(["%MSG-e L1T:  L1TStage2:l1t  06-Oct-2026 04:01:02 CEST\n", "bad word\n"])
        self.assertEqual(len(issues), 1)
        self.assertTrue(issues[0]["truncated"])


class OtherDetectorTests(unittest.TestCase):
    def test_minified_javascript_from_a_downloaded_page_is_not_an_error_line(self):
        js = ('_.ha=function(a){var b=a;if((0,_.ca)(b)){if(!/^\\s*$/.test(b))throw Error(String(b));}'
              'else if(da(b));return b};var c=1;var d=2;\n')
        kept = "Error in <TFile::TFile>: file thisFileDoesNotExist.root does not exist\n"
        js2 = "(c[f+b]=h[k],delete h[k]);e=e&-16760833|(g&1023)<<14;break a}}if(b){k=Math.max(b,f);if(k>1024)throw Error(\"G\");\n"
        kinds = [(i["kind"], i["summary"][:5]) for i in scanner.scan_lines([js, js2, kept])]
        self.assertEqual(kinds, [("generic_error_line", "Error")])

    def test_uncaught_exception_picks_up_expected_failure_hint_printed_after_it(self):
        text = (
            "terminate called after throwing an instance of 'cms::Exception'\n"
            "  what():  An exception of category 'MissingStop' occurred.\n"
            "Exception Message:\n"
            "stop requested too soon 0\n"
            "\n"
            "expected failure occurred\n"
        )
        issues = scanner.scan_lines(text.splitlines())
        uncaught = [i for i in issues if i["kind"] == "uncaught_exception"]

        self.assertEqual(len(uncaught), 1)
        self.assertEqual(uncaught[0]["origin"], "cmssw")
        self.assertEqual(uncaught[0]["intentional_hint"], "expected failure occurred")

    def test_shell_reported_abort_is_a_crash(self):
        issues = scanner.scan_lines([
            "/data/cmsbld/jenkins/workspace/ib-run-qa/CMSSW_20_1_ASAN_X_2026-10-05-2300/src/FWCore/Services/test/"
            "test_zombiekiller.sh: line 10: 822400 Aborted                 (core dumped) cmsRun x_cfg.py\n"])
        self.assertEqual(issues[0]["kind"], "crash")
        self.assertEqual(issues[0]["summary"], "Process aborted in test_zombiekiller.sh")

    def test_bare_shell_signal_report_is_a_crash(self):
        # the report alone on its line (no "line N: PID"), as a wrapper prints it
        for line, what in (("Aborted (core dumped)", "Aborted"), ("Aborted", "Aborted"),
                           ("Floating point exception (core dumped)", "Floating point exception"),
                           ("Illegal instruction (core dumped)", "Illegal instruction"),
                           ("Segmentation fault (core dumped)", "Segmentation fault")):
            issues = scanner.scan_lines([line + "\n"])
            self.assertEqual([(i["kind"], i["category"]) for i in issues], [("crash", what)], line)
        self.assertEqual(scanner.scan_lines(["Aborted (core dumped)\n"])[0]["summary"], "Process aborted")

    def test_prose_mentioning_an_abort_is_not_a_crash(self):
        for line in ("Job Aborted by the user", "the test was Aborted (core dumped) earlier, retrying",
                     "Event processing Aborted because of a skipped event"):
            self.assertNotIn("crash", [i["kind"] for i in scanner.scan_lines([line + "\n"])], line)

    def test_glued_thread_output_does_not_change_the_signal_name(self):
        # found in the imported data: output of two threads glued onto the signal line
        lines = ["A fatal system signal has occurred: abort signal\n",
                 "A fatal system signal has occurred: : index abort signal-916979453\n",
                 "A fatal system signal has occurred: abort signalabort signal\n",
                 "A fatal system signal has occurred: : abort signal\n"]
        issues = [scanner.scan_lines([line])[0] for line in lines]
        self.assertEqual({i["category"] for i in issues}, {"abort signal"})
        self.assertEqual(len({i["signature"] for i in issues}), 1)
        self.assertEqual(issues[1]["summary"], "cmsRun crash: fatal system signal (abort signal)")

    def test_old_glued_signal_signature_is_resigned(self):
        old = dict(kind="crash", category=": index abort signal-916979453",
                   message="A fatal system signal has occurred: : index abort signal-916979453",
                   signature="crash|: index abort signal-916979453|")
        self.assertEqual(scanner.current_signature(old), "crash|abort signal|")

    def test_python_traceback_outside_cmssw_is_unclassified(self):
        text = (
            "Traceback (most recent call last):\n"
            '  File "/data/cmsbld/jenkins/workspace/ib-run-relvals/pyRelval/516.0_WTolNuJets/thread0/lheevent/'
            'madevent/bin/internal/lhe_parser.py", line 1234, in unweight\n'
            "    max_wgt = all_wgt[-1]\n"
            "IndexError: list index out of range\n"
        )
        issues = scanner.scan_lines(text.splitlines())

        self.assertEqual(len(issues), 1)
        self.assertEqual((issues[0]["kind"], issues[0]["category"], issues[0]["origin"]),
                         ("python_traceback", "IndexError", "unclassified"))

    def test_ucx_error_is_infrastructure(self):
        issues = scanner.scan_lines([
            "[cmsbuild91.cern.ch:1234] osc_ucx_component.c:369 Error: OSC UCX component priority set inside\n"])
        self.assertEqual((issues[0]["kind"], issues[0]["origin"]), ("infrastructure_error", "infrastructure"))

    def test_generic_error_lines_are_kept_but_labelled_generic(self):
        issues = scanner.scan_lines(["Error in opening file /tmp/x.root\n"])
        self.assertEqual([i["kind"] for i in issues], ["generic_error_line"])

    def test_trigreport_column_header_is_not_a_problem(self):
        # the header of every HLT TrigReport table, seen as "new" in each newly scanned series
        issues = scanner.scan_lines([
            "TrigReport  Trig Bit#   Executed     Passed     Failed      Error Name\n",
            "TrigReport    Visited        Run     Passed     Failed      Error Name\n",
            "TrigReport     1    0        100        100          0           0 HLTriggerFirstPath\n"])
        self.assertEqual(issues, [])


class NormalizationTests(unittest.TestCase):
    def test_madgraph_worker_dir_is_not_part_of_the_signature(self):
        a = scanner.normalize_text("/pyRelval/519.0_X/thread0/lheevent/bin/internal/lhe_parser.py")
        b = scanner.normalize_text("/pyRelval/525.0_Y/thread3/lheevent/bin/internal/lhe_parser.py")
        self.assertEqual(a, b)

    def test_a_messagelogger_marker_glued_onto_a_ubsan_line_is_dropped(self):
        issues = scanner.scan_lines([
            "%MSGsrc/DataFormats/GEMDigi/interface/GEMOptoHybrid.h:151:38: runtime error: shift exponent 3927 "
            "is too large for 32-bit type 'int'\n",
            "src/DataFormats/GEMDigi/interface/GEMOptoHybrid.h:151:38: runtime error: shift exponent 2556 "
            "is too large for 32-bit type 'int'\n"])
        self.assertEqual(len({i["signature"] for i in issues}), 1)

    def test_fatal_exception_signature_does_not_depend_on_release_path_length(self):
        def message(path):
            return ("Module intProducer has the Test variant set explicitly to other, but its accelerator is "
                    "not available for the job At: " + path + "/src/FWCore/Integration/plugins/Thing.cc "
                    + "x" * 200)
        short = scanner.fatal_signature_detail(message("/cvmfs/cms-ib.cern.ch/sw/x/cms/cmssw/CMSSW_20_1_X_2026-10-09-1100"))
        long = scanner.fatal_signature_detail(message(
            "/data/cmsbld/jenkins/workspace/ib-run-qa-long-name/CMSSW_20_1_X_2026-10-09-2300"))
        self.assertEqual(short, long)

    def test_current_signature_is_idempotent_and_keeps_messagelogger_categories(self):
        ml = dict(signature="message_logger|warning:HGCalGeomT3|4|")
        self.assertEqual(scanner.current_signature(ml), ml["signature"])
        tb = dict(signature="python_traceback|IndexError|IndexError:x @ $RELEASE/pyRelval/<WF>/thread2/a.py")
        once = scanner.current_signature(tb)
        self.assertEqual(scanner.current_signature(dict(signature=once)), once)

    def test_release_paths_timestamps_and_numbers_are_stripped(self):
        a = scanner.normalize_text(
            "/data/cmsbld/jenkins/workspace/ib-run-qa/CMSSW_20_1_ASAN_X_2026-10-05-2300/src/X.py line 12 "
            "at 06-Oct-2026 04:20:19 CEST pid 735950")
        b = scanner.normalize_text(
            "/data/cmsbld/jenkins/workspace/ib-run-qa/CMSSW_20_1_X_2026-10-07-1100/src/X.py line 99 "
            "at 08-Oct-2026 11:02:01 CEST pid 1")
        self.assertEqual(a, b)
        self.assertIn("$RELEASE/src/X.py", a)

    def test_relval_workflow_folder_is_stripped(self):
        a = scanner.normalize_text("/data/cmsbld/jenkins/workspace/ib-run-relvals/CMSSW_20_1_X_2026-10-05-1100/"
                                   "pyRelval/516.0_WTolNuJets_LO_Mad_13TeV_py8_taupinu/thread0/lhe_parser.py")
        b = scanner.normalize_text("/data/cmsbld/jenkins/workspace/ib-run-relvals/CMSSW_20_1_X_2026-10-05-1100/"
                                   "pyRelval/519.0_WTolNuJets_LO_Mad_13TeV_py8_taurhonu/thread0/lhe_parser.py")
        self.assertEqual(a, b)

    def test_guids_are_stripped(self):
        a = scanner.normalize_text("GUID 6A1E2B3C-0D4E-11EF-9A2B-3C4D5E6F7A8B is inconsistent")
        b = scanner.normalize_text("GUID 0F0E0D0C-0B0A-0909-0807-060504030201 is inconsistent")
        self.assertEqual(a, b)

    def test_iso_timestamps_and_version_numbers_are_stripped(self):
        a = scanner.normalize_text("ERROR ( process 0 ) : On time 2026-10-06T05:08:49.922407 Target 1; HDF5 (1.14.6)")
        b = scanner.normalize_text("ERROR ( process 0 ) : On time 2026-10-07T18:15:46.867217 Target 1; HDF5 (1.12.2)")
        self.assertEqual(a, b)

    def test_glog_timestamps_and_padding_are_stripped(self):
        a = scanner.normalize_text("[W1007 00:51:16.268732746 CUDAFunctions.cpp:108] Warning: CUDA initialization")
        b = scanner.normalize_text("[W1004 10:02:33.000000001 CUDAFunctions.cpp:108] Warning: CUDA initialization")
        self.assertEqual(a, b)
        a = scanner.normalize_text("[ngt-nvidia-h100-02:8304 :0]  ucp_worker.c:1485 UCX  ERROR failed")
        b = scanner.normalize_text("[ngt-nvidia-h100-02:314154:0] ucp_worker.c:1485 UCX ERROR failed")
        self.assertEqual(a, b)

    def test_build_node_paths_and_host_names_are_stripped(self):
        a = scanner.normalize_text(
            "file /pool/condor/dir_1932328/jenkins/workspace/ib-run-qa/CMSSW_15_1_X_2026-10-04-0000/x.root; "
            "[b9g40p4801:3720950:a] ib_md.c:287 UCX ERROR; Error R__unzip_header: Values: 645c")
        b = scanner.normalize_text(
            "file /data/cmsbld/jenkins/workspace/ib-run-qa/CMSSW_15_1_X_2026-10-05-0000/x.root; "
            "[ngt-nvidia-h100-02:145401:0] ib_md.c:287 UCX ERROR; Error R__unzip_header: Values: b7db")
        self.assertEqual(a, b)
        self.assertIn("$RELEASE/x.root", a)

    def test_identifiers_with_digits_are_kept(self):
        self.assertIn("Run4D500", scanner.normalize_text("KeyError: 'Run4D500'"))


class StreamingTests(unittest.TestCase):
    def test_over_long_line_is_cut_and_the_next_lines_are_still_read(self):
        # CLANG IB: testConvertToRaw prints one 2 GB line "( 0 0 0 ..."; reading it whole was OOM-killed
        data = b"hlt bits=\n( " + b"0 " * 100_000 + b"\n---> test testConvertToRaw succeeded\n"
        lines = list(scanner._bounded_lines(io.BufferedReader(io.BytesIO(data)), limit=1000))
        self.assertEqual(len(lines), 3)
        self.assertLessEqual(len(lines[1]), 1001)
        self.assertEqual(lines[2], "---> test testConvertToRaw succeeded\n")

    def test_line_endings_match_text_mode_reading(self):
        data = b"a\r\nprogress 10%\rprogress 20%\rERROR here\nlast\r"
        expected = list(io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", newline=""))
        self.assertEqual(list(scanner._bounded_lines(io.BufferedReader(io.BytesIO(data)))), expected)

    def test_distinct_generic_lines_per_owner_are_capped(self):
        lines = [f"Error number {chr(65 + i % 26)}{chr(65 + i // 26 % 26)}{chr(65 + i // 676)} here\n"
                 for i in range(scanner.MAX_GENERIC_SIGNATURES + 5)]
        grouped = scanner.aggregate(scanner.iter_issues(lines), {})

        self.assertEqual(len(grouped), scanner.MAX_GENERIC_SIGNATURES + 1)
        other = [r for r in grouped.values() if r["summary"] == "(other generic error lines)"]
        self.assertEqual(other[0]["count"], 5)

    def test_unit_test_zip_is_streamed_per_member(self):
        log = "".join(unit_log(("SubmitPVrbr", DICTIONARY_NOT_FOUND, "succeeded"),
                               ("PrimaryVertex", "x\n", "had ERRORS")))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unitTestLogs.zip"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("unitTestLogs/Alignment/OfflineValidation/unitTest.log", log)
            result = scanner.scan_unit_test_zip(str(path))

        self.assertEqual([(t["test"], t["status"]) for t in result["tests"]],
                         [("SubmitPVrbr", "passed"), ("PrimaryVertex", "failed")])
        fatal = [p for p in result["problems"] if p["kind"] == "fatal_exception"]
        self.assertEqual((fatal[0]["package"], fatal[0]["test"]), ("Alignment/OfflineValidation", "SubmitPVrbr"))


class AddonTests(unittest.TestCase):
    def test_statuses_map_to_step_log_members(self):
        statuses = scanner.parse_addon_statuses(
            "[hlt_mc_GRun:2] cmsRun /cvmfs/x/OnLine_HLT_GRun.py realData=False : PASSED - elapsed time 12s\n"
            "[fastsim:1] cmsDriver.py TTbar_8TeV_TuneCUETP8M1_cfi --fast : FAILED - elapsed time 3s\n")

        self.assertEqual(statuses["cmsDriver-hlt_mc_GRun_step2.log"]["status"], "passed")
        self.assertEqual(statuses["cmsDriver-fastsim_step1.log"]["status"], "failed")

    def test_scan_addon_zip_attributes_problems_to_the_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "addOnTests.zip"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("cmsDriver-hlt_mc_GRun_step2.log", DICTIONARY_NOT_FOUND)
                zf.writestr("cmsDriver-hlt_mc_GRun_step2.log-read_config", '{"show_controls": []}')
            result = scanner.scan_addon_zip(
                str(path), "[hlt_mc_GRun:2] cmsRun x.py : PASSED - elapsed time 1s\n")

        self.assertEqual(len(result["tests"]), 1)
        fatal = [p for p in result["problems"] if p["kind"] == "fatal_exception"]
        self.assertEqual((fatal[0]["test"], fatal[0]["step"], fatal[0]["test_status"]), ("hlt_mc_GRun", 2, "passed"))


class RelValTests(unittest.TestCase):
    def test_workflow_log_step0_is_step1(self):
        parsed = scanner.parse_relval_workflow_log(
            "140.051_RunMuonEG2022C Step0-PASSED Step1-PASSED Step2-FAILED Step3-NOTRUN  - time date Mon Oct  5 "
            "17:15:41 2026-date Mon Oct  5 17:15:40 2026; exit: 0 0 256 0\n1 1 0 0 tests passed, 0 0 1 0 failed\n")

        self.assertEqual(parsed["steps"], {1: "PASSED", 2: "PASSED", 3: "FAILED", 4: "NOTRUN"})
        self.assertEqual(parsed["exit_codes"], [0, 0, 256, 0])

    def test_scan_relval_zip_reads_the_duplicate_step_log_once(self):
        wf = "140.051_RunMuonEG2022C"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pyRelValMatrixLogs.zip"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr(f"{wf}/", "")
                zf.writestr(f"{wf}/workflow.log", f"{wf} Step0-PASSED Step1-PASSED ; exit: 0 0\n")
                zf.writestr(f"{wf}/step2_RunMuonEG2022C.log", DICTIONARY_NOT_FOUND)
                zf.writestr(f"{wf}/step2.log", DICTIONARY_NOT_FOUND)
                zf.writestr(f"{wf}/step2_RunMuonEG2022C.log-read_config", "{}")
            result = scanner.scan_relval_zip(str(path))

        self.assertEqual(result["workflows"][0]["status"], "passed")
        fatal = [p for p in result["problems"] if p["kind"] == "fatal_exception"]
        self.assertEqual(len(fatal), 1)
        self.assertEqual(fatal[0]["count"], 1)
        self.assertEqual((fatal[0]["step"], fatal[0]["test_status"], fatal[0]["member"]),
                         (2, "passed", f"{wf}/step2_RunMuonEG2022C.log"))


if __name__ == "__main__":
    unittest.main()


class ProblemDetailsTests(unittest.TestCase):
    def test_details_of_every_occurrence_are_merged_not_just_the_first(self):
        log = []
        for module in ("AddIntsProducer", "BusyWaitIntProducer"):
            log += ["----- Begin Fatal Exception 12-Oct-2026 12:26:58 CEST-----",
                    "An exception of category 'Test' occurred while",
                    "   [0] Processing  Event run: 1 lumi: 7 event: 7 stream: 3",
                    "   [1] Running path 'p1'",
                    f"   [2] Calling method for module {module}/'busy1'",
                    "----- End Fatal Exception -----", ""]
        (record,) = scanner.aggregate(scanner.iter_issues(log), owner={"test": "t"}).values()
        self.assertEqual(record["count"], 2)
        self.assertEqual(record["modules"], ["AddIntsProducer/'busy1'", "BusyWaitIntProducer/'busy1'"])
        self.assertEqual(record["context_keys"], ["Test|AddIntsProducer", "Test|BusyWaitIntProducer"])
        self.assertIn("Processing Event run:<N> lumi:<N> event:<N> stream:<N>", record["contexts"])
        self.assertNotIn("p1", record["names"])  # path names say nothing about the problem

    def test_scope_parts_and_cfi_plugin_names_are_names(self):
        details = scanner.problem_details([
            "==1==ERROR: AddressSanitizer: global-buffer-overflow",
            "    #0 0x7f in HGCalGeometry::getNewCorners(DetId) HGCalGeometry.cc:612",
            "   [2] Writing cfi files for module of base type EDProducer with plugin name 'TICLCandidateProducer'",
            "   [0] Running executable \"edmWriteConfigs\""], "PluginNotFound")
        for name in ("HGCalGeometry::getNewCorners", "HGCalGeometry", "getNewCorners", "TICLCandidateProducer"):
            self.assertIn(name, details["names"])
        self.assertEqual(details["context_keys"], ["PluginNotFound|TICLCandidateProducer",
                                                   "PluginNotFound|exe:edmWriteConfigs"])
