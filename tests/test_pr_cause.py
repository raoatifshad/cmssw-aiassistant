"""'Which PR caused it?' / 'what code did that PR touch?' by rules (server/pr_cause.py). Found 2026-11-12 (owner's demo chain): workflow 36434.0
fails since CMSSW_20_1_X_2026-11-06-1100 with "Unable to find plugin 'btlrechit::BTLBaseRecHitSoAProducer@alpaka'"; of the 8 PRs merged into
that build only #51861 touched BTLBaseRecHitSoAProducerAlgo.h, but the AI path put #52397 first, followed up on it, 80-95 s a turn."""
import unittest
from unittest.mock import patch

from server import api_server, pr_cause

_ERR = "Unable to find plugin 'btlrechit::BTLBaseRecHitSoAProducer@alpaka' in category 'CMS EDM Framework Module'. Please check spelling of name."
_BAD, _GOOD = "CMSSW_20_1_X_2026-11-06-1100", "CMSSW_20_1_X_2026-11-05-2300"
_PRS = [{"number": 52397, "title": "avoid re-classification", "merged_at": "2026-11-06T08:20:00Z", "patch_status": "ok", "tags": [_BAD],
         "files": ["src/SimG4Core/Application/src/Phase2StackingAction.cc"]},
        {"number": 51861, "title": "MTD SoA-based digitization and local reconstruction", "merged_at": "2026-11-06T08:15:44Z", "patch_status": "ok", "tags": [_BAD],
         "files": ["src/RecoLocalFastTime/FTLRecProducers/plugins/alpaka/BTLBaseRecHitSoAProducerAlgo.h", "src/DataFormats/FTLDigi/src/BTLDigi.cc"]},
        {"number": 52116, "title": "something else", "merged_at": "2026-11-06T07:00:00Z", "patch_status": "unavailable", "tags": [_BAD], "files": []}]


def _runs(statuses, arch="el9_amd64_gcc14"):
    return [{"tag": f"CMSSW_20_1_X_2026-11-0{i + 3}-1100", "arch": arch, "status": s} for i, s in enumerate(statuses)]


class ErrorNamesTests(unittest.TestCase):
    def test_plugin_name_without_namespace_or_backend_and_no_framework_words(self):
        paste = "An exception of category 'PluginNotFound' occurred while\n [0] Constructing the EventProcessor\n" + _ERR
        self.assertEqual(pr_cause.error_names(paste, "PluginNotFound"), ["BTLBaseRecHitSoAProducer"])


class StreakTests(unittest.TestCase):
    def test_newest_failing_streak_with_its_last_pass(self):
        s = pr_cause._streak(_runs(["PASSED", "FAILED", "PASSED", "FAILED", "FAILED"]))
        self.assertEqual((s["first_bad"][-15:], s["last_good"][-15:], s["still_failing"]), ("2026-11-06-1100", "2026-11-05-1100", True))

    def test_an_architecture_with_a_real_start_beats_one_failing_since_the_oldest_build(self):
        rows = _runs(["FAILED"] * 5, arch="fc42_riscv64_gcc15") + _runs(["PASSED", "PASSED", "FAILED", "FAILED", "FAILED"])
        self.assertEqual(pr_cause._pick_track(rows, None)["arch"], "el9_amd64_gcc14")
        self.assertEqual(pr_cause._pick_track(rows, "fc42_riscv64_gcc15")["arch"], "fc42_riscv64_gcc15")


class P1Tests(unittest.TestCase):
    """Found 2026-11-12 (P1 run): partial architecture words, 'what changed before it failed?', streaks without a start."""
    def test_partial_architecture_word_picks_that_track(self):
        rows = _runs(["FAILED"] * 5, arch="fc42_riscv64_gcc15") + _runs(["PASSED", "FAILED", "FAILED"], arch="el9_aarch64_gcc14") + _runs(["PASSED", "PASSED", "FAILED"])
        self.assertEqual(pr_cause._pick_track(rows, "aarch64")["arch"], "el9_aarch64_gcc14")
        self.assertEqual(pr_cause._pick_track(rows, "riscv64")["arch"], "fc42_riscv64_gcc15")
        self.assertEqual(pr_cause._arch_word("which PR broke it on aarch64?"), "aarch64")
        self.assertIsNone(pr_cause._arch_word("which PR broke it?"))

    def test_junk_tokens_dropped_and_record_suffix_stripped(self):
        self.assertEqual(pr_cause.error_names('No "EcalPedestalsRcd" record found. 1a2b3c4d5e6f7890 at 0x7f3a2c1d lxplus123:4567'), ["EcalPedestals"])
        self.assertNotIn("deadbeefcafe1234", pr_cause.error_names("BTLDetId 1659058062 not found! deadbeefcafe1234"))

    def test_ended_streak_with_a_start_beats_a_still_failing_one_without(self):
        rows = _runs(["FAILED"] * 5, arch="fc42_riscv64_gcc15") + _runs(["PASSED", "FAILED", "FAILED", "FAILED", "PASSED"])
        self.assertEqual(pr_cause._pick_track(rows, None)["arch"], "el9_amd64_gcc14")

    def test_what_changed_before_it_failed_is_a_cause_question(self):
        for q in ["what changed before it failed?", "what was merged right before it started failing?", "what changed prior to workflow 36434.0 breaking?"]:
            self.assertTrue(pr_cause.CAUSE_RE.search(q), q)
        self.assertFalse(pr_cause.CAUSE_RE.search("what changed in the latest IB?"))

    def _ask(self, failure, rows):
        sid = "pr-cause-p1"
        with patch.object(pr_cause, "_subject", return_value=failure), patch.object(pr_cause, "_window_prs", return_value=(rows, [_BAD])), \
             patch.object(pr_cause, "_previous_build", return_value=None), patch.object(api_server, "_github_export_note", return_value="GitHub data reaches 2026-11-06."):
            return pr_cause.cause_answer("which PR caused it?", sid)

    def test_start_older_than_our_data_is_no_most_likely_and_an_ended_streak_is_not_failing_in_every_build(self):
        failure = {"first_bad": _GOOD, "last_good": None, "first_run": True, "still_failing": False, "newest": _BAD, "arch": "el9_amd64_gcc14",
                   "subject": "unit tests of `Some/Pkg`", "error": None, "names": ["BTLBaseRecHitSoAProducer"], "packages": []}
        answer = self._ask(failure, _PRS)
        self.assertNotIn("Most likely", answer)
        self.assertIn("Possible match, the real start is older than our data: PR #51861", answer)
        self.assertNotIn("Failing in every imported build", answer)
        self.assertIn("passes again in `" + _BAD + "`", answer)
        self.assertIn("not proof", answer)


class CauseAnswerTests(unittest.TestCase):
    def setUp(self):
        self.sid = "pr-cause-test"
        pr_cause._state.pop(self.sid, None)
        api_server._turn_memory.pop(self.sid, None)
        self.failure = {"first_bad": _BAD, "last_good": _GOOD, "still_failing": True, "newest": _BAD, "arch": "el9_amd64_gcc14",
                        "subject": "workflow `36434.0` (TTbar_14TeV)", "workflow": "36434.0", "first_run": False,
                        "error": {"category": "PluginNotFound", "message": _ERR}, "names": ["BTLBaseRecHitSoAProducer"], "packages": []}

    def _ask(self, question, rows=_PRS):
        with patch.object(pr_cause, "_workflow_failure", return_value=self.failure) as lookup, \
             patch.object(pr_cause, "_window_prs", return_value=(rows, [_BAD])), \
             patch.object(api_server, "_github_export_note", return_value="GitHub data reaches 2026-11-06."):
            return pr_cause.cause_answer(question, self.sid), lookup

    def test_the_pr_whose_files_carry_the_error_name_comes_first_with_the_others_listed(self):
        answer, lookup = self._ask("which PR caused workflow 36434.0 to fail?")
        self.assertEqual(lookup.call_args[0][0], "36434.0")
        self.assertIn("**Most likely: PR #51861**", answer)
        self.assertIn("BTLBaseRecHitSoAProducerAlgo.h", answer)
        self.assertIn("touched nothing that matches: #52397, #52116", answer)
        self.assertIn("Files are unknown for 1", answer)
        self.assertIn("not proof", answer)
        self.assertEqual(pr_cause._state[self.sid]["top"], 51861)

    def test_no_pr_with_matching_files_says_so_and_names_none(self):
        answer, _ = self._ask("which PR broke workflow 36434.0?", rows=[_PRS[0], _PRS[2]])
        self.assertIn("**No clear candidate:**", answer)
        self.assertNotIn("Most likely", answer)
        self.assertIsNone(pr_cause._state[self.sid]["top"])

    def test_it_after_a_workflow_question_uses_that_workflow(self):
        api_server._turn_memory[self.sid] = {"question": "is workflow 36434.0 failing reported?", "answer": "...", "route": "deterministic:github_reported"}
        with patch.dict(api_server._last_workflow_contexts, {}, clear=True), patch.dict(api_server._session_pastes, {}, clear=True):
            answer, lookup = self._ask("which PR caused it?")
        self.assertEqual(lookup.call_args[0][0], "36434.0")
        self.assertIn("PR #51861", answer)

    def test_other_questions_and_ib_tag_only_questions_are_left_alone(self):
        for question in ["is workflow 36434.0 failing reported?", "which PRs were merged in the latest IB?", f"which PR broke {_BAD}?"]:
            self.assertIsNone(self._ask(question)[0], question)


class FilesAnswerTests(unittest.TestCase):
    def setUp(self):
        self.sid = "pr-cause-files"
        pr_cause._state[self.sid] = {"top": 51861, "names": ["BTLBaseRecHitSoAProducer"], "packages": [], "candidates": [51861]}
        api_server._turn_memory[self.sid] = {"question": "which PR caused it?", "answer": "...", "route": "deterministic:pr_cause"}
        self.row = [{"title": "MTD SoA-based digitization", "patch_status": "ok", "changed": 2, "files": _PRS[1]["files"]}]

    def test_matching_files_first_and_a_yes_for_does_it_touch(self):
        with patch.object(pr_cause, "_run", return_value=self.row):
            answer = pr_cause.files_answer("does it touch the code where it fails?", self.sid)
        self.assertIn("**Yes:** 1 of them carry a name from the error", answer)
        self.assertIn("`RecoLocalFastTime/FTLRecProducers/plugins/alpaka/BTLBaseRecHitSoAProducerAlgo.h`", answer)
        self.assertIn("`DataFormats/FTLDigi` (1)", answer)

    def test_only_right_after_a_cause_answer_and_never_for_a_named_pr(self):
        with patch.object(pr_cause, "_run", return_value=self.row):
            self.assertIsNone(pr_cause.files_answer("what files did PR 52397 change?", self.sid))
            api_server._turn_memory[self.sid]["route"] = "deterministic:relval_summary"
            self.assertIsNone(pr_cause.files_answer("what code did that PR touch?", self.sid))

    def test_hooks_run_early_in_chat(self):
        with open(api_server.__file__) as handle:
            chat = handle.read().split("def _chat(req: ChatRequest):", 1)[1]
        self.assertLess(chat.index("pr_cause_answer(original_question, sid)"), chat.index("issue_topic = _issue_topic_answer("))


if __name__ == "__main__":
    unittest.main()
