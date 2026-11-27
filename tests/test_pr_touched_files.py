"""The PR patch data, (PR)-[:TOUCHED_FILE]->(SourceFile), PR.patch_status, PR.patch_files.
Before these, nothing read it: "which PRs modified FWCore/ParameterSet/src/Entry.cc" was answered "no PR touched it" from the empty
CHANGED_FILE, while 34 PRs did."""
import unittest
import unittest.mock
from unittest.mock import patch

from graph import analytics


class TouchTargetTests(unittest.TestCase):
    def test_paths_as_people_write_them(self):
        self.assertEqual(analytics._touch_target("Configuration/AlCa/python/autoCond.py"), ("file", "src/Configuration/AlCa/python/autoCond.py", "autoCond.py"))
        self.assertEqual(analytics._touch_target("src/FWCore/Framework/"), ("directory", "src/FWCore/Framework/", "Framework"))
        self.assertEqual(analytics._touch_target("`autoCond.py`"), ("file name", "/autoCond.py", "autoCond.py"))
        self.assertEqual(analytics._touch_target("SimG4Core"), ("directory", "src/SimG4Core/", "SimG4Core"))


class PrFilesTests(unittest.TestCase):
    def _pr(self, status, files=(), changed=None, title="t"):
        return [{"title": title, "patch_status": status, "state": "closed", "merged_at": "2026-11-03T10:00:00Z",
                 "dashboard_changed_files": changed, "files": list(files)}]

    def test_unavailable_patch_is_unknown_never_no_files(self):
        with patch.object(analytics, "_rows", return_value=self._pr("unavailable")), patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            out = analytics.pr_files(52373)
        self.assertIsNone(out["files"])
        self.assertTrue(out["files_status"].startswith("unknown"))
        self.assertNotIn("files_touched", out)

    def test_cmsdist_has_no_file_lists(self):
        with patch.object(analytics, "_rows", return_value=self._pr(None)), patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            out = analytics.pr_files(10966, repo="cmsdist")
        self.assertIn("cmssw PRs only", out["files_status"])

    def test_known_files_packages_and_asked_path(self):
        files = ["src/SimG4Core/Application/src/Phase2StackingAction.cc", "src/SimG4Core/Application/interface/A.h"]
        with patch.object(analytics, "_rows", return_value=self._pr("ok", files, changed=2)), patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            out = analytics.pr_files(52397, path="SimG4Core/Geometry")
            hit = analytics.pr_files(52397, path="Phase2StackingAction.cc")
        self.assertEqual(out["files_touched"], 2)
        self.assertEqual(out["packages"], {"SimG4Core/Application": 2})
        self.assertFalse(out["touched_asked_path"])
        self.assertTrue(hit["touched_asked_path"])
        self.assertEqual(hit["matching_files"], ["SimG4Core/Application/src/Phase2StackingAction.cc"])

    def test_issue_number_says_issue(self):
        with patch.object(analytics, "_rows", side_effect=[[], [{"title": "x"}]]), patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            out = analytics.pr_files(52323)
        self.assertTrue(out["is_issue"])


class PrsTouchingPathTests(unittest.TestCase):
    def _row(self, number, names):
        return {"repo": "cmssw", "number": number, "title": "t", "state": "closed", "merged_at": None, "created_at": "2026-10-20",
                "date": "2026-10-20", "files_touched": 1, "example_files": ["src/Utilities/XrdAdaptor/src/X.cc"], "names": names}

    def test_lists_prs_and_counts_unknown_ones_in_the_window(self):
        unknown = {"total": 63, "naming": [{"number": 52314, "title": "[13_1_X] XrdAdaptor: ...", "merged_at": "2026-11-03"}]}
        with patch.object(analytics, "_rows", side_effect=[[self._row(52317, ["makortel"])], [unknown]]), \
                patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            out = analytics.prs_touching_path("Utilities/XrdAdaptor", since="2026-10-09")
        self.assertEqual(out["total_prs"], 1)
        self.assertEqual(out["prs"][0]["author"], "makortel")
        self.assertEqual(out["prs"][0]["example_files"], ["Utilities/XrdAdaptor/src/X.cc"])
        self.assertEqual(out["prs_with_unknown_files_in_window"], 63)
        self.assertEqual(out["unknown_files_but_title_names_it"][0]["number"], 52314)
        self.assertIn("never say a PR caused", out["note"])

    def test_author_ranking_counts_the_github_opener(self):
        rows = [self._row(1, ["perrotta"]), self._row(2, ["perrotta"]), self._row(3, ["cmsbuild", "mmusich"])]
        with patch.object(analytics, "_rows", side_effect=[rows, [{"total": 0, "naming": []}]]), \
                patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            out = analytics.prs_touching_path("autoCond.py", group_by="author")
        self.assertEqual(out["rows"], [{"author": "perrotta", "prs": 2}, {"author": "cmsbuild", "prs": 1}])

    def test_loose_match_only_when_the_path_does_not_exist(self):
        # SiStripChannelGain exists but no PR in the window: must not widen to "anywhere in the path"
        with patch.object(analytics, "_rows", side_effect=[[], [{"x": 1}], [{"total": 81, "naming": []}]]) as rows, \
                patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            out = analytics.prs_touching_path("CalibTracker/SiStripChannelGain", since="2026-10-01", until="2026-11-08", date_field="merged")
        self.assertEqual(out["matched_as"], "directory")
        self.assertEqual(out["total_prs"], 0)
        self.assertEqual(rows.call_count, 3)
        self.assertIn("reaches 2026-11-06", out["note"])

    def test_ranking_says_how_many_prs_it_covers(self):
        with patch.object(analytics, "_rows", side_effect=[[{"key": "HLTrigger/Configuration", "prs": 15}], [{"prs": 216, "known": 164}]]), \
                patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            out = analytics.prs_touching_path(group_by="package", since="2026-10-01", until="2026-10-31", date_field="merged")
        self.assertEqual(out["rows"], [{"package": "HLTrigger/Configuration", "prs": 15}])
        self.assertIn("164 of 216", out["note"])

    def test_tools_are_registered_and_in_the_schema(self):
        from graph import tools
        self.assertIn("pr_files", tools.TOOLS)
        self.assertIn("prs_touching_path", tools.TOOLS)
        self.assertIn("TOUCHED_FILE", tools.GRAPH_SCHEMA)
        self.assertIn("patch_status", tools.GRAPH_SCHEMA)


class PrCardFilesTests(unittest.TestCase):
    """E2-2: the PR card listed only the dashboard's file count ("what did PR 52367 change")."""
    def _info(self, **kw):
        base = {"found": True, "files_status": "known", "files_touched": 2, "packages": {"FWCore/ParameterSet": 2},
                "files": ["FWCore/ParameterSet/interface/Entry.h", "FWCore/ParameterSet/src/Entry.cc"], "touched_asked_path": False, "matching_files": []}
        return {**base, **kw}

    def test_lists_files_when_asked(self):
        from server import api_server
        with patch.object(api_server, "pr_files", return_value=self._info()):
            lines = api_server._pr_files_lines("cmssw", 52367, "what did PR 52367 change")
        self.assertTrue(any("FWCore/ParameterSet/src/Entry.cc" in l for l in lines))

    def test_short_line_only_when_not_asked(self):
        from server import api_server
        with patch.object(api_server, "pr_files", return_value=self._info()):
            lines = api_server._pr_files_lines("cmssw", 52367, "is PR 52367 merged?")
        self.assertEqual(len(lines), 1)
        self.assertIn("2 in 1 package", lines[0])

    def test_unknown_is_never_no_files(self):
        from server import api_server
        info = {"found": True, "files_status": "unknown: the backup saved an error page instead", "files": None, "dashboard_changed_files": 9}
        with patch.object(api_server, "pr_files", return_value=info):
            lines = api_server._pr_files_lines("cmssw", 52353, "did PR 52353 touch RecoHGCal/TICL?")
        self.assertIn("unknown", lines[0])
        self.assertIn("dashboard", lines[0])
        self.assertIn("Unknown", lines[1])

    def test_asked_path_answered_yes_or_no(self):
        from server import api_server
        with patch.object(api_server, "pr_files", return_value=self._info(touched_asked_path=False)) as call:
            lines = api_server._pr_files_lines("cmssw", 52397, "did PR 52397 touch anything in SimG4Core/Geometry?")
        self.assertEqual(call.call_args.kwargs["path"], "SimG4Core/Geometry")
        self.assertTrue(any("**Touched `SimG4Core/Geometry`?** No" in l for l in lines))


class StaleNoFilesStatementsTests(unittest.TestCase):
    """E2-3: three answers still said changed files aren't imported."""
    def test_bare_which_files_is_no_longer_not_imported(self):
        from server import api_server
        self.assertIsNone(api_server._not_imported_answer("which files?"))

    def test_which_files_after_a_pr_card(self):
        from server import api_server
        sid = "e2-files"
        api_server._turn_memory[sid] = {"question": "PR 52381", "answer": "## PR #52381 - x\n\n- **Repository:** `cmssw`"}
        try:
            with patch.object(api_server, "_pr_files_lines", return_value=["- **Files touched:** 3 in 2 package(s)"]) as call:
                out = api_server._listed_items_follow_up("which files did it touch?", sid)
        finally:
            api_server._turn_memory.pop(sid, None)
        self.assertEqual(call.call_args.args[:2], ("cmssw", 52381))
        self.assertIn("Files touched", out)

    def test_window_topic_is_not_the_word_last(self):
        from server import api_server
        rows = [{"repo": "cmssw", "number": 52317, "title": "t", "merged_at": "2026-11-03", "author": "a", "body": ""},
                {"repo": "cmssw", "number": 1, "title": "other", "merged_at": "2026-11-03", "author": "a", "body": ""}]
        session = unittest.mock.MagicMock()
        session.run.return_value = rows
        driver = unittest.mock.MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), \
                patch.object(api_server, "_touched_topic", return_value=({("cmssw", 52317)}, [52314])) as touched:
            out = api_server._prs_merged_window_answer("any PRs merged in the last 10 days touching XrdAdaptor?")
        self.assertEqual(touched.call_args.args[1], "XrdAdaptor")
        self.assertIn("#52317", out)
        self.assertNotIn("#1 ", out)
        self.assertIn("unknown for 1", out)


class TouchedPathRuleTests(unittest.TestCase):
    """E2-4: "which PRs modified FWCore/ParameterSet/src/Entry.cc" said no PR touched it; "how many PRs have ever touched autoCond.py?" said 0 (1,067)."""
    def _found(self, **kw):
        base = {"path": "FWCore/ParameterSet/src/Entry.cc", "matched_as": "file", "total_prs": 1, "prs_with_unknown_files_in_window": 49,
                "unknown_files_but_title_names_it": [{"number": 52371, "title": "Fix the name of a few upgrade GTs in autoCond", "merged_at": None}],
                "prs": [{"number": 52367, "title": "Remove unused Entry constructors", "author": "fwyzard", "created_at": "2026-11-02",
                         "merged_at": "2026-11-04T10:00:00Z", "state": "closed", "files_touched": 1, "example_files": ["FWCore/ParameterSet/src/Entry.cc"]}]}
        return {**base, **kw}

    def test_lists_prs_with_unknown_note_and_caveat(self):
        from server import api_server
        with patch("graph.analytics.prs_touching_path", return_value=self._found()) as call, \
                patch.object(api_server, "_github_export_note", return_value="GitHub data reaches 2026-11-06"):
            out = api_server._touched_path_answer("which PRs modified FWCore/ParameterSet/src/Entry.cc")
        self.assertEqual(call.call_args.args[0], "FWCore/ParameterSet/src/Entry.cc")
        self.assertIn("cmssw #52367", out)
        self.assertIn("unknown for 49", out)
        self.assertIn("#52371", out)
        self.assertIn("never the cause", out)

    def test_who_most_is_a_ranking_and_year_is_a_window(self):
        from server import api_server
        found = self._found(rows=[{"author": "stahlleiton", "prs": 17}], total_prs=63)
        with patch("graph.analytics.prs_touching_path", return_value=found) as call, \
                patch.object(api_server, "_github_export_note", return_value="x"):
            out = api_server._touched_path_answer("who touched relval_steps.py most often in 2026?")
        self.assertEqual(call.call_args.kwargs["group_by"], "author")
        self.assertEqual((call.call_args.kwargs["since"], call.call_args.kwargs["until"]), ("2026-01-01", "2026-12-31"))
        self.assertIn("| `stahlleiton` | 17 |", out)

    def test_open_and_merged_filters(self):
        from server import api_server
        with patch("graph.analytics.prs_touching_path", return_value=self._found()) as call, patch.object(api_server, "_github_export_note", return_value="x"):
            api_server._touched_path_answer("open PRs that touch RecoTracker/LSTCore")
            self.assertEqual(call.call_args.kwargs["state"], "open")
            api_server._touched_path_answer("which PRs merged on 2026-11-03 touched Utilities/XrdAdaptor?")
            self.assertEqual((call.call_args.kwargs["since"], call.call_args.kwargs["until"], call.call_args.kwargs["date_field"]), ("2026-11-03", "2026-11-03", "merged"))

    def test_leaves_other_questions_alone(self):
        from server import api_server
        with patch("graph.analytics.prs_touching_path", side_effect=AssertionError("not called")):
            for q in ("did PR 52397 touch SimG4Core/Geometry?", "is 539.0 failing in CMSSW_20_1_X_2026-11-03-2300?",
                      "which PRs touched autoCond.py since mid october?", "which file was changed by the most PRs in October 2026?",
                      "SimG4Core/Geometry unit test is failing in CMSSW_17_0_X_2026-11-04-1100, which PRs touched it?"):
                self.assertIsNone(api_server._touched_path_answer(q), q)

    def test_who_changed_a_path_is_not_rewritten_to_a_text_search(self):
        from server import api_server
        self.assertTrue(api_server._touched_path_answer_applies("who changed Phase2StackingAction.cc recently?"))
        self.assertFalse(api_server._touched_path_answer_applies("who changed Geant4 recently?"))


if __name__ == "__main__":
    unittest.main()


class UnknownIncludesPrsNewerThanBackupTests(unittest.TestCase):
    """E2-5 (rerun 2026-11-10): "any PRs merged last week touching PhysicsTools/TruthInfo?" said files are unknown for 15 PRs; the 28 cmssw PRs
    merged after the 2026-11-06 backup (known from the IB pages, no patch_status at all) were not counted as unknown."""

    def test_unknown_query_counts_missing_patch_status(self):
        unknown = {"total": 20, "not_in_backup": 5, "naming": []}
        hit = {"repo": "cmssw", "number": 52345, "title": "t", "state": "closed", "merged_at": "2026-11-07", "created_at": "2026-10-30",
               "date": "2026-11-07", "files_touched": 1, "example_files": ["src/PhysicsTools/TruthInfo/interface/Branch.h"], "names": ["x"]}
        with patch.object(analytics, "_rows", side_effect=[[hit], [unknown]]) as rows, \
                patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            out = analytics.prs_touching_path("PhysicsTools/TruthInfo", since="2026-11-03", date_field="merged")
        self.assertIn("coalesce(p.patch_status, '') <> 'ok'", rows.call_args_list[-1].args[0])
        self.assertEqual(out["prs_with_unknown_files_in_window"], 20)
        self.assertEqual(out["of_which_newer_than_backup"], 5)

    def test_pr_newer_than_backup_is_unknown_with_reason(self):
        row = [{"title": "t", "patch_status": None, "state": None, "merged_at": "2026-11-09T13:48:00Z", "dashboard_changed_files": None, "files": []}]
        with patch.object(analytics, "_rows", return_value=row), patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            out = analytics.pr_files(52409)
        self.assertIsNone(out["files"])
        self.assertIn("newer than our GitHub backup", out["files_status"])
        self.assertIn("never say it changed no files", out["files_status"])

    def test_answer_says_why_files_are_unknown(self):
        from server import api_server
        found = TouchedPathRuleTests._found(TouchedPathRuleTests(), total_prs=0, prs=[], prs_with_unknown_files_in_window=20,
                                              of_which_newer_than_backup=5, unknown_files_but_title_names_it=[])
        with patch("graph.analytics.prs_touching_path", return_value=found), \
                patch.object(api_server, "_github_export_note", return_value="GitHub data reaches 2026-11-06"):
            out = api_server._touched_path_answer("any PRs merged last week touching PhysicsTools/TruthInfo?")
        self.assertIn("unknown for 20", out)
        self.assertIn("5 are newer than the backup", out)
        self.assertIn("for 15 the GitHub backup saved an error page", out)


class TouchedPathBeforeBuildTests(unittest.TestCase):
    """E2-6 (rerun 2026-11-10, e2:X02): "CalibTracker/SiStripChannelGain fails in CMSSW_20_1_X_2026-11-08-2300. any PR merged before that build
    touched it?" got the build's merged-PR list, which never looks at files."""

    def test_build_window_from_tag(self):
        from server import api_server
        self.assertEqual(api_server._tag_build_window("CMSSW_20_1_X_2026-11-08-2300", None),
                         ("CMSSW_20_1", "2026-10-09", "2026-11-08T22:00", "2026-11-08 23:00 CERN time"))
        self.assertEqual(api_server._tag_build_window("CMSSW_20_1_NONLTO_X_2026-07-01-1100", "2026-06-01")[:3], ("CMSSW_20_1", "2026-06-01", "2026-07-01T09:00"))

    def test_release_and_exact_time_filters(self):
        hit = {"repo": "cmssw", "number": 1, "title": "t", "state": "closed", "merged_at": "2026-11-01", "created_at": "2026-10-30",
               "date": "2026-11-01", "files_touched": 1, "example_files": [], "names": ["x"]}
        with patch.object(analytics, "_rows", side_effect=[[hit], [{"total": 0, "not_in_backup": 0, "naming": []}]]) as rows, \
                patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            analytics.prs_touching_path("CalibTracker/SiStripChannelGain", since="2026-10-09", until="2026-11-08T22:00", date_field="merged",
                                        state="merged", release="CMSSW_20_1")
        first = rows.call_args_list[0]
        self.assertIn("p.release STARTS WITH 'CMSSW_20_1_'", first.args[0])
        self.assertEqual(first.kwargs["end"], "2026-11-08T22:00")

    def test_answer_for_a_build(self):
        from server import api_server
        empty = TouchedPathRuleTests._found(TouchedPathRuleTests(), total_prs=0, prs=[], prs_with_unknown_files_in_window=45, of_which_newer_than_backup=1,
                                            unknown_files_but_title_names_it=[], path="CalibTracker/SiStripChannelGain/", matched_as="directory")
        older = {**empty, "total_prs": 1, "prs": [{"number": 50991, "title": "cleanup", "author": "smuzaffar", "merged_at": "2026-05-20T10:00:00Z"}]}
        with patch("graph.analytics.prs_touching_path", side_effect=[empty, older]) as call, \
                patch.object(api_server, "_github_export_note", return_value="GitHub data reaches 2026-11-06"):
            out = api_server._touched_path_answer("CalibTracker/SiStripChannelGain fails in CMSSW_20_1_X_2026-11-08-2300. any PR merged before that build touched it?")
        first, second = call.call_args_list
        self.assertEqual((first.kwargs["release"], first.kwargs["until"], first.kwargs["state"]), ("CMSSW_20_1", "2026-11-08T22:00", "merged"))
        self.assertNotIn("release", second.kwargs)  # the newest toucher of any series: master is renamed over time
        self.assertIn("No cmssw PR of `CMSSW_20_1` merged between 2026-10-09 and the build", out)
        self.assertIn("cmssw #50991", out)
        self.assertIn("unknown for 45", out)
        self.assertIn("1 is newer than the backup", out)
        self.assertIn("never the cause", out)

    def test_two_tags_left_to_others(self):
        from server import api_server
        self.assertIsNone(api_server._touched_path_answer("which PRs touched FWCore/Framework between CMSSW_20_1_X_2026-11-07-1100 and CMSSW_20_1_X_2026-11-08-2300?"))


class TouchWindowMonthTests(unittest.TestCase):
    """E2-7 (e2b:B11, 2026-11-10): "show me the PRs that touched RecoTracker/LSTCore in October" -> "Something went wrong": the rule's month
    tuple was shadowed by a later module-level _MONTHS dict, so every month-named question raised AttributeError."""

    def test_month_window_with_the_real_module_globals(self):
        from datetime import datetime, timezone
        from server import api_server
        with patch.object(api_server, "_cern_now", return_value=datetime(2026, 11, 10, 10, 0, tzinfo=timezone.utc)):
            self.assertEqual(api_server._touch_window("show me the PRs that touched RecoTracker/LSTCore in October"), ("2026-10-01", "2026-10-31"))
            self.assertEqual(api_server._touch_window("which PRs changed autoCond.py during december 2025"), ("2025-12-01", "2025-12-31"))

    def test_month_question_reaches_the_lookup(self):
        from server import api_server
        found = TouchedPathRuleTests._found(TouchedPathRuleTests(), path="RecoTracker/LSTCore/", matched_as="directory")
        with patch("graph.analytics.prs_touching_path", return_value=found) as call, \
                patch.object(api_server, "_github_export_note", return_value="GitHub data reaches 2026-11-06"):
            out = api_server._touched_path_answer("show me the PRs that touched RecoTracker/LSTCore in October")
        self.assertIsNotNone(out)
        self.assertTrue(call.call_args.kwargs["since"].endswith("-10-01"))
        self.assertTrue(call.call_args.kwargs["until"].endswith("-10-31"))
