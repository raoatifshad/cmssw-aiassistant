"""Fixes from test item E3 (2026-11-10): newest IBs, natural wording, short follow-ups."""
import unittest
from unittest.mock import patch

from server.api_server import _last_failed_builds, _listed_items_follow_up, _turn_memory

SID = "s-e3"
OVERVIEW = ("🔴 What's failing in `CMSSW_20_1_X_2026-11-09-2300` -- primary IB\n\n"
            "**Build** -- failed: `el9_aarch64_gcc14`\n\n**RelVals** -- 2 failing:\n- `36434.0`, `36834.0`")
BUILD_ROWS = [{"tag": "CMSSW_20_1_X_2026-11-09-2300", "build": "error"},
              {"tag": "CMSSW_20_1_X_2026-11-05-2300", "build": "passed"}]


class SinceWhenSubjectTests(unittest.TestCase):
    """E3-1: "since when?" answered the failed BUILD of an older or never-shown answer."""

    def setUp(self):
        _turn_memory.pop(SID, None)
        _last_failed_builds.pop(SID, None)

    def _remember(self, answer, superseded=False, route="capability:failure_overview"):
        _turn_memory[SID] = {"question": "q", "answer": answer, "route": route}
        if superseded:
            _turn_memory[SID]["superseded"] = True

    @patch("server.api_server.get_driver")
    def test_shown_failed_build_still_answers_since_when(self, get_driver):
        get_driver.return_value.session.return_value.__enter__.return_value.run.return_value = BUILD_ROWS
        self._remember(OVERVIEW)
        self.assertIn("The build on `el9_aarch64_gcc14` has failed since", _listed_items_follow_up("since when?", SID))

    @patch("server.api_server.get_driver")
    def test_an_answer_the_ai_replaced_is_not_the_subject(self, get_driver):
        get_driver.return_value.session.return_value.__enter__.return_value.run.return_value = BUILD_ROWS
        self._remember(OVERVIEW, superseded=True)  # the user saw the AI's answer, not this one
        answer = _listed_items_follow_up("since when?", SID) or ""
        self.assertNotIn("The build on", answer)
        self.assertNotIn(SID, _last_failed_builds)

    @patch("server.api_server.get_driver")
    def test_after_a_later_answer_the_old_build_is_not_the_subject(self, get_driver):
        get_driver.return_value.session.return_value.__enter__.return_value.run.return_value = BUILD_ROWS
        self._remember(OVERVIEW)
        _listed_items_follow_up("which one is new?", SID)  # records the failed build of the overview
        self._remember("## Shift summary: CMSSW_20_1_X_2026-11-09-1100 -> CMSSW_20_1_X_2026-11-09-2300\n\n"
                       "#### Newly failing RelVal workflows\n| Workflow | Name |\n| --- | --- |\n| `136.745` | RunZeroBias2016C |",
                       route="capability:shift_summary")
        answer = _listed_items_follow_up("since when?", SID) or ""
        self.assertNotIn("The build on", answer)

    @patch("server.api_server.get_driver")
    def test_build_table_follow_up_keeps_the_build(self, get_driver):
        get_driver.return_value.session.return_value.__enter__.return_value.run.return_value = BUILD_ROWS
        self._remember(OVERVIEW)
        _listed_items_follow_up("which packages?", SID)
        self._remember("The build on `el9_aarch64_gcc14` has failed since `X`.\n\n| IB | Build status |\n| --- | --- |",
                       route="deterministic:listed_items_follow_up")
        self.assertIn("The build on", _listed_items_follow_up("how long?", SID))



class PackageStatusPrimaryFirstTests(unittest.TestCase):
    """E3-2: "failing now in 22 of 28" listed 15 sub-IB rows and silently cut the primary builds."""

    @patch("server.api_server._package_failing_tests_note", return_value=None)
    @patch("server.api_server.get_driver")
    def test_primary_rows_lead_and_the_rest_is_counted(self, get_driver, _):
        from server.api_server import _package_newest_status_answer
        flavours = ["ASAN", "CLANG", "CUDART", "DBG", "DEVEL", "EVTGEN3", "G4ADEPT", "G4VECGEOM", "GEANT4",
                    "MULTIARCHS", "MULTIARCHSV4", "NONLTO", "PY314", "ROOT6", "ROOT642", "UBSAN"]
        rows = []
        for f in flavours:
            rows += [{"tag": f"CMSSW_20_1_{f}_X_2026-11-08-2300", "variant": f, "arch": "el9_amd64_gcc14", "status": "PASSED", "errors": 0},
                     {"tag": f"CMSSW_20_1_{f}_X_2026-11-09-2300", "variant": f, "arch": "el9_amd64_gcc14", "status": "FAILED", "errors": 1}]
        for arch in ("el8_amd64_gcc14", "el9_amd64_gcc14"):
            rows += [{"tag": "CMSSW_20_1_X_2026-11-06-2300", "variant": "primary", "arch": arch, "status": "PASSED", "errors": 0},
                     {"tag": "CMSSW_20_1_X_2026-11-07-1100", "variant": "primary", "arch": arch, "status": "FAILED", "errors": 1},
                     {"tag": "CMSSW_20_1_X_2026-11-09-2300", "variant": "primary", "arch": arch, "status": "FAILED", "errors": 1}]
        get_driver.return_value.session.return_value.__enter__.return_value.run.return_value = rows
        answer = _package_newest_status_answer("DQMServices/Demo", "did DQMServices/Demo fail in the last 20_1 build?")
        self.assertIn("**Failing now in 18 of 18**", answer)
        listed = [line for line in answer.splitlines() if line.startswith("- ")]
        self.assertTrue(listed[0].startswith("- primary `el8_amd64_gcc14`: failing in `CMSSW_20_1_X_2026-11-09-2300`, since `CMSSW_20_1_X_2026-11-07-1100`"))
        self.assertTrue(listed[1].startswith("- primary `el9_amd64_gcc14`"))
        self.assertIn("since `CMSSW_20_1_X_2026-11-07-1100` (last passed in `CMSSW_20_1_X_2026-11-06-2300`)", listed[0])
        self.assertIn("- ... and 3 more: ", answer)
        self.assertIn("UBSAN `el9_amd64_gcc14`", answer)  # the last flavour is named, not dropped


class LatestPartialNoteTests(unittest.TestCase):
    """E3-3: "the last 17_1 IB" (complete) was called partly imported, pointing to a 20_1 build."""

    ROWS = [{"tag": "CMSSW_20_1_X_2026-11-09-2300", "arch": "el8_amd64_gcc14", "n": 509},
            {"tag": "CMSSW_20_1_X_2026-11-09-1100", "arch": "el8_amd64_gcc14", "n": 954},
            {"tag": "CMSSW_20_1_X_2026-11-08-2300", "arch": "el8_amd64_gcc14", "n": 954},
            {"tag": "CMSSW_17_1_X_2026-11-09-2300", "arch": "el8_amd64_gcc13", "n": 1252},
            {"tag": "CMSSW_17_1_X_2026-11-09-1100", "arch": "el8_amd64_gcc13", "n": 1252}]

    @patch("server.api_server.get_driver")
    def test_complete_build_of_another_series_gets_no_note(self, get_driver):
        from server.api_server import _latest_partial_import_note
        get_driver.return_value.session.return_value.__enter__.return_value.run.return_value = self.ROWS
        self.assertEqual(_latest_partial_import_note("CMSSW_17_1_X_2026-11-09-2300", "relval"), "")

    @patch("server.api_server.get_driver")
    def test_partial_build_names_the_complete_one_of_its_series(self, get_driver):
        from server.api_server import _latest_partial_import_note
        get_driver.return_value.session.return_value.__enter__.return_value.run.return_value = self.ROWS
        note = _latest_partial_import_note("CMSSW_20_1_X_2026-11-09-2300", "relval")
        self.assertIn("only partly imported", note)
        self.assertIn("`CMSSW_20_1_X_2026-11-09-1100` is the newest IB with complete RelVal results", note)


class IssueTopicFailureWordsTests(unittest.TestCase):
    """E3-4: "is there an open issue about SimG4Core/Geometry failing on aarch64?" said none, with #51089 open."""

    @patch("server.api_server._github_export_note", return_value="GitHub data reaches 2026-11-06")
    @patch("server.api_server.get_driver")
    def test_failure_words_left_out_when_nothing_matches(self, get_driver, _):
        from server.api_server import _issue_topic_answer
        row = {"n": 51089, "title": "[SimG4Core/Geometry] testVolumes crashes on el10_amd64_gcc15 and el8_aarch64_gcc13",
               "state": "open", "created": "2026-05-29", "closed": None, "in_title": True}
        run = get_driver.return_value.session.return_value.__enter__.return_value.run
        run.side_effect = [[], [], [row]]
        answer = _issue_topic_answer("is there an open issue about SimG4Core/Geometry failing on aarch64?")
        self.assertIn("Issue #51089", answer)
        self.assertEqual(run.call_args_list[2].kwargs["words"], ["simg4core", "geometry", "aarch64"])
        self.assertIn("x.state = 'open'", run.call_args_list[2].args[0])

    @patch("server.api_server._github_export_note", return_value="GitHub data reaches 2026-11-06")
    @patch("server.api_server.get_driver")
    def test_only_failure_words_left_is_still_none(self, get_driver, _):
        from server.api_server import _issue_topic_answer
        run = get_driver.return_value.session.return_value.__enter__.return_value.run
        run.side_effect = [[], []]
        answer = _issue_topic_answer("is there an issue about tests failing?")
        self.assertIn("No imported Issue", answer)
        self.assertEqual(run.call_count, 2)


def _wf_rows(pattern: dict, wf_name="TTbar"):
    """{arch: "PPFFF"} -> RelVal rows of consecutive primary builds, oldest first."""
    tags = [f"CMSSW_20_1_X_2026-11-0{d}-{h}" for d in range(1, 10) for h in ("1100", "2300")]
    return [{"tag": tags[i], "arch": arch, "failed": c == "F", "name": wf_name}
            for arch, seq in pattern.items() for i, c in enumerate(seq)]


class WorkflowChangePrsTests(unittest.TestCase):
    """E3-5: "which PR?" / "which PRs came in that build?" after a workflow: text matches from 2013, or a later build's PRs."""

    PRS = [{"repo": "cmssw", "pr_number": 51861, "title": "MTD SoA", "merged_at": "2026-11-06T08:15:44Z", "author": "someone"}]

    def _answer(self, pattern, question):
        from server.api_server import _workflow_change_prs_answer
        with patch("server.api_server.get_driver") as get_driver, \
                patch("server.api_server.find_prs_before_ib", return_value=self.PRS) as prs, \
                patch("server.api_server._github_export_note", return_value="GitHub data reaches 2026-11-06"):
            get_driver.return_value.session.return_value.__enter__.return_value.run.return_value = _wf_rows(pattern)
            return _workflow_change_prs_answer("27234.0", question), prs

    def test_started_failing_build_and_its_prs(self):
        answer, prs = self._answer({"el8_amd64_gcc14": "PPPPPPPPPFFFF", "el9_amd64_gcc14": "PPPPPPPPPFFFF",
                                    "fc42_riscv64_gcc15": "FFFFF"}, "which PRs came in that build?")
        self.assertIn("started failing in the primary `CMSSW_20_1_X_2026-11-05-2300` on `el8_amd64_gcc14`, `el9_amd64_gcc14`", answer)
        self.assertIn("(it passed in `CMSSW_20_1_X_2026-11-05-1100`), and is still failing there.", answer)
        prs.assert_called_once_with("CMSSW_20_1_X_2026-11-05-2300", limit=100)
        self.assertIn("cmssw #51861", answer)
        self.assertIn("never a proven cause", answer)
        self.assertIn("GitHub data reaches 2026-11-06", answer)

    def test_bare_question_after_it_passes_again_is_about_the_fix(self):
        answer, prs = self._answer({"el8_amd64_gcc14": "PPPFFFFPP", "el9_amd64_gcc14": "PPPFFFFPP"}, "which PR?")
        self.assertIn("passed again in the primary `CMSSW_20_1_X_2026-11-04-2300`", answer)
        self.assertIn("(it last failed in `CMSSW_20_1_X_2026-11-04-1100`)", answer)
        answer, _ = self._answer({"el8_amd64_gcc14": "PPPFFFFPP", "el9_amd64_gcc14": "PPPFFFFPP"}, "which PR broke it?")
        self.assertIn("started failing in the primary `CMSSW_20_1_X_2026-11-02-2300`", answer)
        self.assertIn("and passes again since.", answer)

    def test_cmsdist_prs_of_the_window_are_added(self):
        note = ("By merge time instead, **2 `CMSSW_20_1` PR(s)** were merged ...\n- **#52345** - cmssw one (by `x`)\n"
                "- **cmsdist #10989** - Herwig: use configure option (by `y`)")
        with patch("server.api_server._safe_merge_time_note", return_value=note):
            answer, _ = self._answer({"el8_amd64_gcc14": "PPPFFFFPP"}, "which PR fixed it?")
        self.assertIn("cmsdist PRs merged in the same window, by merge time", answer)
        self.assertIn("cmsdist #10989", answer)
        self.assertNotIn("#52345** - cmssw one", answer)

    def test_fix_asked_while_still_failing(self):
        answer, prs = self._answer({"el9_amd64_gcc14": "PPFFF"}, "which PR fixed it?")
        self.assertIn("still failing in the newest primary", answer)
        prs.assert_not_called()

    def test_flapping_workflow_says_so(self):
        answer, _ = self._answer({"el9_amd64_gcc14": "PFPFPFPP"}, "which PR fixed it?")
        self.assertIn("comes and goes", answer)

    @patch("server.api_server._workflow_change_prs_answer", return_value="CHANGE")
    def test_follow_up_uses_the_workflow_just_discussed(self, change):
        from server.api_server import _last_workflow_contexts
        _turn_memory[SID] = {"question": "since when is 27234.0 failing?", "answer": "Workflow `27234.0` ...", "route": "deterministic:workflow_where"}
        _last_workflow_contexts[SID] = "27234.0"
        try:
            self.assertEqual(_listed_items_follow_up("which PRs came in that build?", SID), "CHANGE")
            self.assertEqual(_listed_items_follow_up("which PR?", SID), "CHANGE")
            self.assertEqual(_listed_items_follow_up("who broke it?", SID), "CHANGE")
            _turn_memory[SID]["answer"] = "something about a package"
            _turn_memory[SID]["question"] = "is DQMServices/Demo failing?"
            self.assertNotEqual(_listed_items_follow_up("which PR?", SID), "CHANGE")  # the workflow isn't the subject any more
        finally:
            _last_workflow_contexts.pop(SID, None)
            _turn_memory.pop(SID, None)


class SameOnArchFollowUpTests(unittest.TestCase):
    """E3-6: "same on el8?" after a workflow's exception on el9 -> "I couldn't match that"."""

    def test_same_on_arch_repeats_the_question_there(self):
        from server.api_server import _resolve_short_follow_up
        _turn_memory[SID] = {"question": "were there exceptions in 36834.0 in the latest 20_1 IB?", "answer": "Workflow `36834.0` ...",
                             "route": "deterministic:workflow_why"}
        try:
            for q in ("same on el8?", "is it the same on el8?", "the same on el8 too?"):
                self.assertEqual(_resolve_short_follow_up(q, SID), "were there exceptions in 36834.0 in the latest 20_1 IB on el8")
        finally:
            _turn_memory.pop(SID, None)


class WorkflowHistoryStreakTests(unittest.TestCase):
    """E3-7: "when did they stop?" -> the AI took last_passed (the newest pass) as the build where the workflow recovered."""

    def test_passing_again_since_is_the_first_pass_after_the_last_failure(self):
        from unittest.mock import patch
        from graph import analytics
        tags = ["CMSSW_20_1_X_2026-11-06-2300", "CMSSW_20_1_X_2026-11-07-1100", "CMSSW_20_1_X_2026-11-08-0000", "CMSSW_20_1_X_2026-11-08-2300",
                "CMSSW_20_1_X_2026-11-09-1100", "CMSSW_20_1_X_2026-11-09-2300"]
        statuses = ["PASSED", "FAILED", "FAILED", "FAILED", "PASSED", "PASSED"]
        rows = [dict(tag=t, arch="el9_amd64_gcc14", name="n", status=s, exit_code=18688 if s == "FAILED" else 0, known=False) for t, s in zip(tags, statuses)]
        with patch.object(analytics, "_rows", return_value=rows):
            track = analytics.workflow_history("1047.0")["tracks"][0]
        self.assertEqual(track["passing_again_since"], tags[4])
        self.assertEqual(track["last_passed"], tags[5])
        self.assertIsNone(track["failing_since"])
        self.assertEqual(track["failed_builds"], tags[1:4])

    def test_failing_since_is_the_start_of_the_current_streak(self):
        from unittest.mock import patch
        from graph import analytics
        tag = "CMSSW_20_1_X_2026-11-0{}-2300"
        rows = [dict(tag=tag.format(d), arch="el9", name="n", status=s, exit_code=1, known=False) for d, s in enumerate(["FAILED", "PASSED", "FAILED", "FAILED"], 1)]
        with patch.object(analytics, "_rows", return_value=rows):
            track = analytics.workflow_history("1.0")["tracks"][0]
        self.assertEqual((track["failing_since"], track["first_failed"]), (tag.format(3), tag.format(1)))
        self.assertIsNone(track["passing_again_since"])

    def test_never_failed_has_no_recovery(self):
        from unittest.mock import patch
        from graph import analytics
        rows = [dict(tag="CMSSW_20_1_X_2026-11-0%d-2300" % d, arch="el9", name="n", status="PASSED", exit_code=0, known=False) for d in (1, 2)]
        with patch.object(analytics, "_rows", return_value=rows):
            track = analytics.workflow_history("1.0")["tracks"][0]
        self.assertEqual((track["passing_again_since"], track["failing_since"], track["failed_builds"]), (None, None, []))

class KnownQuestionChangesAskTests(unittest.TestCase):
    """E3-8: "is 136.745 new?" answered as "has 136.745 been reported?" (GitHub text matches from 2019)."""

    def test_newness_is_not_a_report_question(self):
        from server.api_server import _known_question_changes_ask
        for q in ("is 136.745 new?", "is that new?", "since when?", "is this a new failure in ASAN?", "how long has it been failing?"):
            self.assertTrue(_known_question_changes_ask("has 136.745 been reported?", q), q)
        self.assertTrue(_known_question_changes_ask("any open issue about 136.745?", "is it new?"))

    def test_report_questions_and_other_rewrites_are_kept(self):
        from server.api_server import _known_question_changes_ask
        self.assertFalse(_known_question_changes_ask("has 136.745 been reported?", "is there a new issue about 136.745?"))
        self.assertFalse(_known_question_changes_ask("has 136.745 been reported?", "has anyone reported it?"))
        self.assertFalse(_known_question_changes_ask("history of workflow 136.745", "is 136.745 new?"))
        self.assertFalse(_known_question_changes_ask("has 136.745 been reported?", "why does it fail?"))

class OnlyConditionKeptTests(unittest.TestCase):
    """E3-9: "and only on aarch64?" rewritten without "only" -> every aarch64 failure listed (none fail only there)."""

    def test_lost_only_is_put_back(self):
        from server.api_server import _keeps_only_condition, _loses_only_condition
        rewritten = "which workflows fail on el9_aarch64_gcc14 in CMSSW_20_1_X_2026-11-08-2300?"
        for q in ("and only on aarch64?", "which ones fail only there?", "anything on aarch64 only?", "which fail on el8 and nowhere else?"):
            self.assertTrue(_loses_only_condition(q, rewritten), q)
            self.assertTrue(_keeps_only_condition(q, rewritten).startswith(rewritten.rstrip()))
            self.assertIn("only there", _keeps_only_condition(q, rewritten))

    def test_kept_only_or_other_only_is_left_alone(self):
        from server.api_server import _keeps_only_condition, _loses_only_condition
        kept = "which workflows fail only on el9_aarch64_gcc14 and not on amd64 in CMSSW_20_1_X_2026-11-08-2300?"
        self.assertEqual(_keeps_only_condition("and only on aarch64?", kept), kept)
        for q in ("show only the primary", "only 3 failures?", "is it the only one?", "which workflows fail on aarch64?"):
            self.assertFalse(_loses_only_condition(q, "which workflows fail on aarch64?"), q)

if __name__ == "__main__":
    unittest.main()
