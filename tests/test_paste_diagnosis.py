"""server/paste_diagnosis.py: the diagnosis put before a pasted CMSSW fatal exception's evidence (user's example 2026-11-09)."""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import paste_diagnosis as pd  # noqa: E402

PASTE = """ALCAOUTPUT:PPSCalMaxTracks,ALCA:PromptCalibProdPPSTimingCalib
entry file:step2.root
Warning: The default GeometryRecoDB_cff is being used; however, the DB geometry is not applied. You may need to verify your cmsDriver.
Step: ALCAOUTPUT Spec: ['PPSCalMaxTracks']
Step: ALCA Spec: ['PromptCalibProdPPSTimingCalib']
Starting python3 /data/cmsbld/jenkins/workspace/ib-run-relvals/cms-bot/monitor_workflow.py timeout --signal SIGTERM 9000  cmsRun -j JobReport3.xml  step3_ALCAOUTPUT_ALCA.py
%MSG-i ThreadStreamSetup:  (NoModuleName) 09-Nov-2026 09:05:41 CET pre-events
setting # threads 4
%MSG
I0000 00:00:1794211544.498549 2498442 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results.
09-Nov-2026 09:05:45 CET  Successfully opened file file:step2.root
----- Begin Fatal Exception 09-Nov-2026 09:05:45 CET-----------------------
An exception of category 'Configuration' occurred while
   [0] Constructing the EventProcessor
   [1] Validating configuration of module: class=MEtoEDMConverter label='MEtoEDMConvertPPSTimingCalib'
Exception Message:
Illegal parameter found in configuration.  The parameter is named:
 'deleteAfterCopy'
You could be trying to use a parameter name that is not
allowed for this plugin, or it could be misspelled, or this parameter
needs to be defined in the fillDescriptions() method of the plugin.
----- End Fatal Exception -------------------------------------------------
whats the issue and is it reported?"""

FIX = {"number": 52449, "title": "remove deprecated `deleteAfterCopy` parameter from `MEtoEDMConverter` instances", "state": "closed",
       "merged_at": "2026-11-09T08:43:37+00:00", "created_at": None, "release": "CMSSW_20_1", "author": "mmusich",
       "body": "#### PR description:\r\n\r\nTitle says it all. PR https://github.com/cms-sw/cmssw/pull/52404 added a `fillDescriptions`.\r\n#### PR validation:",
       "labels": '["bug-fix"]', "changed_files": 3, "files": [], "in_ibs": [], "named": ["deleteAfterCopy", "MEtoEDMConverter"], "exact": True}
ORIGIN = {"number": 52404, "title": "Add fillDescriptions to ME and EDM Converters", "merged_at": "2026-11-07T09:51:59+00:00",
          "author": "Dr15Jones", "files": ["src/DQMServices/Components/python/MEtoEDMConverter_cfi.py", "src/DQMServices/Components/plugins/MEtoEDMConverter.cc"],
          "in_ibs": ["CMSSW_20_1_NONLTO_X_2026-11-07-1100", "CMSSW_20_1_X_2026-11-07-1100", "CMSSW_20_1_X_2026-11-08-2300"], "named_by": 52449}
RECORDED = ["CMSSW_20_1_X_2026-11-07-1100", "CMSSW_20_1_NONLTO_X_2026-11-07-1100", "CMSSW_20_1_X_2026-11-08-2300"]
SOURCE = [{"file": "CalibPPS/TimingCalibration/python/ALCARECOPromptCalibProdPPSTimingCalib_cff.py", "line": "10",
           "text": "deleteAfterCopy = cms.untracked.bool(True),", "defines_module": True},
          {"file": "DQM/Physics/python/x_cff.py", "line": "3", "text": "deleteAfterCopy = True", "defines_module": False}]


def _history(fixes=(FIX,), origins=(ORIGIN,), recorded=RECORDED, issues=()):
    return {"terms": ["deleteAfterCopy", "MEtoEDMConvertPPSTimingCalib", "MEtoEDMConverter"], "fixes": list(fixes),
            "issues": list(issues), "origins": list(origins), "recorded": list(recorded)}


class ParseFatalPasteTests(unittest.TestCase):
    def test_reads_module_parameter_phase_and_timing(self):
        facts = pd.parse_fatal_paste(PASTE)
        self.assertEqual(facts["category"], "Configuration")
        self.assertEqual((facts["module_class"], facts["module_label"], facts["parameter"]),
                         ("MEtoEDMConverter", "MEtoEDMConvertPPSTimingCalib", "deleteAfterCopy"))
        self.assertIn("before any event", facts["phase"])
        self.assertEqual((facts["timeout"], facts["elapsed"], facts["step"]), (9000, 4, "step3"))
        self.assertEqual(facts["family"][0], "illegal configuration parameter")
        self.assertEqual(facts["specs"], [("ALCAOUTPUT", ["PPSCalMaxTracks"]), ("ALCA", ["PromptCalibProdPPSTimingCalib"])])

    def test_noise_lines_are_named(self):
        noise = " ".join(pd.parse_fatal_paste(PASTE)["noise"])
        self.assertIn("GeometryRecoDB_cff", noise)
        self.assertIn("oneDNN", noise)
        self.assertIn("`step2.root` opened successfully", noise)

    def test_not_a_fatal_exception(self):
        self.assertIsNone(pd.parse_fatal_paste("ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1234"))
        self.assertIsNone(pd.diagnosis_answer("what failed in the latest IB?"))

    def test_terms_most_specific_first(self):
        self.assertEqual(pd._specific_terms(pd.parse_fatal_paste(PASTE)),
                         ["deleteAfterCopy", "MEtoEDMConvertPPSTimingCalib", "MEtoEDMConverter"])


@patch.object(pd, "_source_lines", return_value=SOURCE)
class DiagnosisAnswerTests(unittest.TestCase):
    def test_merged_fix_after_newest_failing_build(self, _src):
        with patch.object(pd, "find_fix_history", return_value=_history()):
            result = pd.diagnosis_answer(PASTE, "GitHub data reaches 2026-11-06")
        text = result["text"]
        self.assertEqual(result["exact_items"], ["PR #52449"])
        self.assertIn("not a timeout", text)
        self.assertIn("**Yes: a PR names this exact problem and is merged.**", text)
        self.assertIn("merged 2026-11-09 09:43 CET", text)
        self.assertIn("`CMSSW_20_1_X_2026-11-08-2300`, was built before the merge", text)
        self.assertIn("**PR #52404**", text)
        self.assertIn("**the same build**", text)
        self.assertIn("`CalibPPS/TimingCalibration/python/ALCARECOPromptCalibProdPPSTimingCalib_cff.py:10`", text)
        self.assertIn("`DQM/Physics`", text)
        self.assertIn("| Plugin package | `DQMServices/Components` |", text)
        self.assertIn("| New issue needed? | No: the fix PR covers it |", text)
        self.assertIn("not proof of the cause", text)
        self.assertNotIn("None", text)
        self.assertNotIn("<", text.replace("<!--", ""))

    def test_error_still_recorded_after_fix_reached_the_ibs(self, _src):
        fix = dict(FIX, in_ibs=["CMSSW_20_1_X_2026-11-07-2300"])
        with patch.object(pd, "find_fix_history", return_value=_history(fixes=[fix])):
            text = pd.diagnosis_answer(PASTE)["text"]
        self.assertIn("**The error is still recorded after it**", text)
        self.assertIn("Maybe: the error is still there after the fix", text)

    def test_fix_reached_the_ibs_and_error_gone(self, _src):
        fix = dict(FIX, in_ibs=["CMSSW_20_1_X_2026-11-09-1100"])
        with patch.object(pd, "find_fix_history", return_value=_history(fixes=[fix])):
            text = pd.diagnosis_answer(PASTE)["text"]
        self.assertIn("first in `CMSSW_20_1_X_2026-11-09-1100`; the error is not recorded in any imported build since", text)

    def test_open_fix_pr(self, _src):
        fix = dict(FIX, merged_at=None, state="open")
        with patch.object(pd, "find_fix_history", return_value=_history(fixes=[fix])):
            text = pd.diagnosis_answer(PASTE)["text"]
        self.assertIn("not merged yet.**", text)
        self.assertIn("no new report needed while it is open", text)

    def test_nothing_names_it(self, _src):
        with patch.object(pd, "find_fix_history", return_value=_history(fixes=[], origins=[])):
            result = pd.diagnosis_answer(PASTE)
        self.assertEqual(result["exact_items"], [])
        self.assertIn("**No PR or Issue names this problem in its title**", result["text"])
        self.assertNotIn("How it started", result["text"])

    def test_changes_in_the_first_failing_build_when_no_fix_names_it(self, _src):
        candidates = [{"number": 51861, "title": "MTD SoA-based digitization and local reconstruction", "author": "24LopezR",
                       "merged_at": "2026-11-06T08:15:00+00:00", "hits": ["src/EventFilter/MTDRawToDigi/python/BTLReadoutMapESProducer_cfi.py"],
                       "n_files": 101, "first_ib": "CMSSW_20_1_X_2026-11-06-1100", "entered": 10}]
        history = dict(_history(fixes=[], origins=[]), candidates=candidates)
        with patch.object(pd, "find_fix_history", return_value=history):
            text = pd.diagnosis_answer(NORECORD)["text"]
        self.assertIn("missing EventSetup record", text)
        self.assertIn("`BTLReadoutMapRcd`", text)
        self.assertIn("Of the 10 PR(s) that entered that build, 1 touches the names from your error", text)
        self.assertIn("**PR #51861**", text)
        self.assertIn("not proof of the cause", text)
        self.assertNotIn("s after the job's first log line", text)  # only the banner's time: no duration


NORECORD = """----- Begin Fatal Exception 09-Nov-2026 01:12:44 CET-----------------------
An exception of category 'NoRecord' occurred while
   [0] Processing  Event run: 1 lumi: 1 event: 1 stream: 0
   [2] Calling method for module BTLDigiToRawProducer/'btlRawData'
Exception Message:
No "BTLReadoutMapRcd" record found in the EventSetup.
----- End Fatal Exception -------------------------------------------------"""


class OtherMessageTests(unittest.TestCase):
    def test_plugin_with_backend_suffix(self):
        facts = pd.parse_fatal_paste("An exception of category 'PluginNotFound' occurred while\n   [0] Constructing the EventProcessor\n"
                                     "   [1] Constructing module: class=btlrechit::BTLBaseRecHitSoAProducer@alpaka label='btlRecHits'\n"
                                     "Exception Message:\nUnable to find plugin 'btlrechit::BTLBaseRecHitSoAProducer@alpaka' in category 'X'.\n")
        self.assertEqual((facts["module_class"], facts["module_label"], facts["plugin"]),
                         ("btlrechit::BTLBaseRecHitSoAProducer", "btlRecHits", "btlrechit::BTLBaseRecHitSoAProducer"))
        self.assertEqual(pd._specific_terms(facts)[0], "BTLBaseRecHitSoAProducer")

    def test_record_is_the_most_specific_term(self):
        facts = pd.parse_fatal_paste(NORECORD)
        self.assertEqual(facts["record"], "BTLReadoutMapRcd")
        self.assertEqual(facts["module_class"], "BTLDigiToRawProducer")
        self.assertIn("while processing events", facts["phase"])
        self.assertIsNone(facts["elapsed"])


if __name__ == "__main__":
    unittest.main()
