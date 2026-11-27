"""Wrong answers found in testing (2026-11-11): management-style questions on IB health, what broke
and since when, PRs, counts, trends and comparisons across releases and architectures."""
import re
import unittest
import unittest.mock
from unittest.mock import patch

from server import api_server


class BuildFailedAgainTests(unittest.TestCase):
    """R1-1: "did the 20_1 aarch64 build fail again last night?" got the RelVal/unit-test still/new/resolved diff, never the build status."""

    def test_build_itself_failing_is_left_to_the_build_answer(self):
        with patch.dict(api_server._latest_ib_contexts, {"r1-1": "CMSSW_20_1_X_2026-11-09-2300"}), \
                patch.object(api_server, "get_driver", side_effect=AssertionError("must not query")):
            for question in ("did the 20_1 aarch64 build fail again last night?",
                             "did the build break again compared to the previous one?",
                             "is it still failing to compile like in the last IB?",
                             "build errors again in the last IB?"):
                self.assertIsNone(api_server._persistent_failures_answer(question, "r1-1"), question)

    def test_test_failures_in_a_build_still_get_the_diff(self):
        for question in ("which unit tests failed again in the last build?", "did the relvals fail again compared to the previous build?",
                         "which of those also failed in the previous build?"):
            guarded = re.search(r"\b(?:relvals?|workflows?|wfs?|unit|tests?|packages?|these|those|them)\b", question, re.IGNORECASE)
            self.assertTrue(not api_server._BUILD_ITSELF_FAILED_RE.search(question) or guarded, question)

    def test_build_regex_shapes(self):
        for text in ("the aarch64 build failed", "build fail again", "build is broken", "the arm build of 20_1 still broken", "fails to compile", "compilation errors"):
            self.assertTrue(api_server._BUILD_ITSELF_FAILED_RE.search(text), text)
        for text in ("which failures are new in the last build", "the build before"):
            self.assertFalse(api_server._BUILD_ITSELF_FAILED_RE.search(text), text)


class NewestBuildWithoutResultsTests(unittest.TestCase):
    """R1-2: "compare the latest 20_0 and 20_1 IBs" showed 20_1 as 0 / 0 (11-10-1600, nothing imported yet); "3 bullet points on 20_1
    IB health" answered "not imported yet" for that build; partly imported builds in a series comparison had no note."""

    def test_empty_newest_build_is_replaced_and_said(self):
        with patch.object(api_server, "_ib_has_no_results", return_value=True), \
                patch.object(api_server, "_newest_complete_primary_tag", return_value=("CMSSW_20_1_X_2026-11-09-2300", "")):
            picked = api_server._with_results_or_newest_complete("CMSSW_20_1_X_2026-11-10-1600", None, (20, 1), "r1-2")
        self.assertEqual(picked, "CMSSW_20_1_X_2026-11-09-2300")
        note = api_server._default_tag_notes.pop("r1-2")
        self.assertIn("CMSSW_20_1_X_2026-11-10-1600", note)
        self.assertIn("no RelVal or unit-test results", note)

    def test_build_with_some_results_stays(self):
        with patch.object(api_server, "_ib_has_no_results", return_value=False), \
                patch.object(api_server, "_newest_complete_primary_tag", side_effect=AssertionError("not needed")):
            self.assertEqual(api_server._with_results_or_newest_complete("CMSSW_20_1_X_2026-11-10-1100", None, (20, 1), "r1-2b"),
                             "CMSSW_20_1_X_2026-11-10-1100")
        self.assertNotIn("r1-2b", api_server._default_tag_notes)

    def test_unit_test_question_takes_the_newest_complete_unit_test_build(self):
        with patch.object(api_server, "_ib_has_no_results", return_value=True), \
                patch.object(api_server, "_newest_complete_primary_tag", return_value=("CMSSW_20_1_X_2026-11-10-1100", "")) as newest:
            picked = api_server._with_results_or_newest_complete("CMSSW_20_1_X_2026-11-10-1600", None, (20, 1), "r1-2c", "unittest")
        self.assertEqual(picked, "CMSSW_20_1_X_2026-11-10-1100")
        self.assertEqual(newest.call_args.args[0], "unittest")
        self.assertIn("complete unit-test results", api_server._default_tag_notes.pop("r1-2c"))

    def test_series_compare_uses_a_build_with_results_and_notes_partial_ones(self):
        records = {"CMSSW_20_0_X_2026-11-10-1100": [dict(variant="primary", arch="el8_amd64_gcc13", build="passed", failed=[], relvals=951, unit_failed=[], units=368)],
                   "CMSSW_20_1_X_2026-11-09-2300": [dict(variant="primary", arch="el9_amd64_gcc14", build="warning", failed=["a", "b"], relvals=955, unit_failed=["x"], units=384)]}
        latest = {(20, 0): [{"tag": "CMSSW_20_0_X_2026-11-10-1100"}], (20, 1): [{"tag": "CMSSW_20_1_X_2026-11-10-1600"}]}
        swap = {"CMSSW_20_1_X_2026-11-10-1600": "CMSSW_20_1_X_2026-11-09-2300"}
        with patch.object(api_server, "_latest_ib_records", side_effect=lambda arch, series: latest[series]), \
                patch.object(api_server, "_ib_build_records", side_effect=lambda tag: records[tag]), \
                patch.object(api_server, "_with_results_or_newest_complete", side_effect=lambda tag, *a: swap.get(tag, tag)), \
                patch.object(api_server, "partial_builds", return_value=[dict(tag="CMSSW_20_0_X_2026-11-10-1100", arch="el8_amd64_gcc13", results=500, usual=951)]):
            answer = api_server._series_compare_answer("compare relval failures of the latest 20_0 and 20_1 IBs")
        self.assertIn("| `CMSSW_20_1_X` | `2026-11-09-2300` | `el9_amd64_gcc14` |", answer)
        self.assertNotIn("2026-11-10-1600` |", answer)
        self.assertIn("`CMSSW_20_1_X_2026-11-10-1600` has no results imported yet", answer)
        self.assertIn("Still being imported, so not final: `CMSSW_20_0_X_2026-11-10-1100` on `el8_amd64_gcc13` (500 of about 951", answer)


class IssueTopicDescriptionOnlyTests(unittest.TestCase):
    """R1-3: "any open issues about aarch64 relval failures?" -> 1 Issue (all three words in a description), while open titles
    say "[AARCH64] ... in RelVal ..." without "failures"."""

    @patch("server.api_server._github_export_note", return_value="GitHub data reaches 2026-11-06")
    @patch("server.api_server.get_driver")
    def test_title_hits_without_the_failure_word_win_over_a_description_hit(self, get_driver, _):
        body_only = {"n": 49535, "title": "[AARCH64] RelVal 30834.0: Assertion 'iadd == 1' failed", "state": "open", "created": "2025-12-03",
                     "closed": None, "in_title": False}
        titled = [{"n": n, "title": t, "state": "open", "created": "2023-01-01", "closed": None, "in_title": True}
                  for n, t in ((40942, "[AARCH64] MixingModule:mix segfaulted in Relval 250402.181"),
                               (40941, "[AARCH64] Segfault in KFTrajectoryFitter::fitOne in RelVal 140.105"))]
        run = get_driver.return_value.session.return_value.__enter__.return_value.run
        run.side_effect = [[], [body_only], titled + [body_only]]
        answer = api_server._issue_topic_answer("any open issues about aarch64 relval failures?")
        self.assertEqual(run.call_args_list[2].kwargs["words"], ["aarch64", "relval"])
        self.assertIn("Issue #40942", answer)
        self.assertIn("Issue #40941", answer)
        self.assertIn('"aarch64 relval" in the title', answer)

    @patch("server.api_server._github_export_note", return_value="GitHub data reaches 2026-11-06")
    @patch("server.api_server.get_driver")
    def test_a_title_hit_with_every_word_is_kept(self, get_driver, _):
        row = {"n": 52407, "title": "RelVal timeouts in recent IBs", "state": "open", "created": "2026-11-06", "closed": None, "in_title": True}
        run = get_driver.return_value.session.return_value.__enter__.return_value.run
        run.side_effect = [[row]]
        answer = api_server._issue_topic_answer("any open issue about relval timeouts?")
        self.assertIn("Issue #52407", answer)
        self.assertEqual(run.call_count, 1)


class PrCountPastTheBackupTests(unittest.TestCase):
    """R1-4: "how many PRs were merged in november so far?" -> 164 with "the data only reaches 2026-11-06, later merges are not
    included", while the 164 include 75 merged later (from the IB pages' merged-PR lists)."""

    def test_note_counts_the_merges_after_the_backup(self):
        from graph import analytics
        prs = [dict(kind="PR", repo="cmssw", number=n, title="t", state="closed", merged_at=m, created_at=None, date=m, author="a", labels=None,
                    fetched_via=None) for n, m in ((1, "2026-11-03T10:00:00Z"), (2, "2026-11-08T10:00:00Z"), (3, "2026-11-10T07:00:00Z"))]
        with patch.object(analytics, "_rows", return_value=prs), patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            result = analytics.search_github(kind="pr", state="merged", since="2026-11-01", date_field="merged", count_only="true")
        self.assertEqual(result["total"], 3)
        self.assertIn("2 of these 3 were merged after 2026-11-06", result["note"])
        self.assertIn("does include merges after 2026-11-06", result["note"])

    def test_no_note_when_everything_is_before_the_backup(self):
        from graph import analytics
        prs = [dict(kind="PR", repo="cmssw", number=1, title="t", state="closed", merged_at="2026-10-03T10:00:00Z", created_at=None,
                    date="2026-10-03", author="a", labels=None, fetched_via=None)]
        with patch.object(analytics, "_rows", return_value=prs), patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            result = analytics.search_github(kind="pr", state="merged", since="2026-10-01", until="2026-10-31", date_field="merged", count_only="true")
        self.assertNotIn("were merged after", result["note"])


class LatestIbWithoutResultsTests(unittest.TestCase):
    """R1-5: "how many relvals failed in the newest 20_1 IB?" -> "no results for 11-10-1600 ... ask e.g. ..." (no numbers)."""

    @patch("server.api_server._clear_pending_clarification")
    @patch("server.api_server._store_planning_decision")
    @patch("server.api_server._restore_pending_clarification")
    @patch("server.api_server._chat_response", side_effect=lambda *args, **kwargs: args[2])
    @patch("server.api_server._execute_planned_capability", return_value=("relval evidence", []))
    @patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-10-1600"}])
    @patch("server.api_server._latest_partial_import_note", side_effect=AssertionError("the switch note stands"))
    def test_latest_without_results_is_answered_for_the_newest_complete(self, _partial, _latest, execute, *_):
        def swap(tag, arch, series, sid, kind="relval"):
            api_server._default_tag_notes[sid] = "_switched_\n\n"
            return "CMSSW_20_1_X_2026-11-09-2300"
        with patch.object(api_server, "_with_results_or_newest_complete", side_effect=swap):
            api_server.chat(api_server.ChatRequest(question="how many relvals failed in the newest 20_1 IB?", session_id="r1-5"))
        self.assertIn("CMSSW_20_1_X_2026-11-09-2300", execute.call_args.args[1])
        self.assertNotIn("CMSSW_20_1_X_2026-11-10-1600", execute.call_args.args[1])
        self.assertEqual(api_server._latest_ib_contexts.pop("r1-5"), "CMSSW_20_1_X_2026-11-09-2300")
        api_server._default_tag_notes.pop("r1-5", None)


class StillBrokenInTheNewerBuildTests(unittest.TestCase):
    """R1-6: "what broke in CMSSW_17_1_X_2026-11-09-2300?" ... "is it still broken?" -> the same build's 22 failures again, while all
    pass in 11-10-1100 (a stale state given as current)."""

    def _run(self, before, now):
        rows = {"CMSSW_17_1_X_2026-11-09-2300": before, "CMSSW_17_1_X_2026-11-10-1100": now}
        session = unittest.mock.MagicMock()
        session.run.side_effect = lambda query, t, a: rows[t]
        driver = unittest.mock.MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "_newest_complete_primary_tag", return_value=("CMSSW_17_1_X_2026-11-10-1100", "")), \
                patch.object(api_server, "get_driver", return_value=driver):
            return api_server._still_failing_in_newer_build_answer("CMSSW_17_1_X_2026-11-09-2300", None)

    def test_all_fixed(self):
        tag, text = self._run([dict(arch="el8_amd64_gcc13", ut=["RecoTracker/MkFit", "RecoVertex/BeamSpotProducer"], rv=[], nut=394, nrv=1252)],
                              [dict(arch="el8_amd64_gcc13", ut=[], rv=[], nut=395, nrv=1232)])
        self.assertEqual(tag, "CMSSW_17_1_X_2026-11-10-1100")
        self.assertTrue(text.startswith("**No: all 2 failures of `CMSSW_17_1_X_2026-11-09-2300` pass in `CMSSW_17_1_X_2026-11-10-1100`.**"))
        self.assertIn("**passing now**", text)

    def test_some_still_failing_and_new_ones(self):
        _, text = self._run([dict(arch="el9_amd64_gcc14", ut=["DQMServices/Demo"], rv=["27234.0", "36434.0"], nut=383, nrv=954)],
                            [dict(arch="el9_amd64_gcc14", ut=["DQMServices/Demo"], rv=["36434.0", "136.905"], nut=384, nrv=955)])
        self.assertIn("**Partly: 2 of the 3 failures", text)
        self.assertIn("RelVals **still failing**: `36434.0`", text)
        self.assertIn("RelVals **passing now** (failed in `CMSSW_17_1_X_2026-11-09-2300`): `27234.0`", text)
        self.assertIn("`136.905`", text)

    def test_wording(self):
        for q in ("is it still broken?", "are they still failing?", "is it fixed now?", "still broken?", "and is the build still red?"):
            self.assertTrue(api_server._STILL_NOW_RE.search(q), q)
        for q in ("is 36434.0 still failing?", "is it still failing on el9?", "is DQMServices/Demo still broken in 20_1?"):
            self.assertFalse(api_server._STILL_NOW_RE.search(q), q)

    def test_newest_build_itself_gets_no_newer_answer(self):
        with patch.object(api_server, "_newest_complete_primary_tag", return_value=("CMSSW_17_1_X_2026-11-09-2300", "")):
            self.assertIsNone(api_server._still_failing_in_newer_build_answer("CMSSW_17_1_X_2026-11-09-2300", None))


class HealthTrendArchAndVerdictTests(unittest.TestCase):
    """R1-7: "trend of relval failures in 20_1 for el9_amd64_gcc14" -> all architectures; "getting better or worse compared to
    last week?" -> the per-day table without a verdict."""

    def _run(self, question, rows, partial=()):
        seen = {}
        session = unittest.mock.MagicMock()

        def run(query, **kw):
            seen.update(kw)
            result = unittest.mock.MagicMock()
            result.__iter__.return_value = iter([r for r in rows if kw.get("arch") in (None, r["arch"])])
            result.single.return_value = None
            return result
        session.run.side_effect = run
        driver = unittest.mock.MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "_latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-10-1100"}]), \
                patch.object(api_server, "get_driver", return_value=driver), \
                patch.object(api_server, "partial_builds", return_value=list(partial)), \
                patch.object(api_server, "_cutoff_stamp", return_value="2026-10-28-0000"):
            return api_server._health_trend_answer(question), seen

    @staticmethod
    def _rows():
        rows = []
        for d in range(1, 15):
            day = f"2026-10-{d + 16:02d}"
            rows.append(dict(day=day, tag=f"CMSSW_20_1_X_{day}-1100", arch="el9_amd64_gcc14", total=1000, failed=10 if d <= 7 else 2))
            rows.append(dict(day=day, tag=f"CMSSW_20_1_X_{day}-1100", arch="fc42_riscv64_gcc15", total=900, failed=850 if d == 12 else 40))
        return rows

    def test_one_architecture_is_kept(self):
        text, seen = self._run("trend of relval failures in 20_1 for el9_amd64_gcc14", self._rows())
        self.assertEqual(seen.get("arch"), "el9_amd64_gcc14")
        self.assertIn("primary IBs on `el9_amd64_gcc14`", text)
        self.assertIn("| 2026-10-17 | 1 | 1,000 | 10 | 1.00% |", text)

    def test_better_or_worse_gets_a_verdict_without_the_mass_failure_build(self):
        text, _ = self._run("is 20_1 health getting better or worse compared to last week?", self._rows())
        self.assertTrue(text.startswith("**Better** by the failure rate"), text[:200])
        self.assertIn("Left out: 1 build(s) where more than half the RelVals failed (`fc42_riscv64_gcc15`)", text)

    def test_peak_follow_up_rewritten_and_last_n_days(self):
        rows = self._rows()
        rows.append(dict(day="2026-10-31", tag="CMSSW_20_1_X_2026-10-31-1100", arch="el9_amd64_gcc14", total=5, failed=4))
        partial = [dict(tag="CMSSW_20_1_X_2026-10-31-1100", arch="el9_amd64_gcc14", results=5, usual=1000)]
        text, _ = self._run("when was the peak of relval failures in 20_1 el9_amd64_gcc14 over the last 10 days?", rows, partial)
        self.assertTrue(text.startswith("**2026-10-21** had the most failures: 10."), text[:120])
        self.assertNotIn("| 2026-10-20 |", text)
        self.assertIn("| 2026-10-21 |", text)
        self.assertNotIn("| 2026-10-31 |", text)
        self.assertIn("Left out, still being imported: `CMSSW_20_1_X_2026-10-31-1100` on `el9_amd64_gcc14` (5 of about 1,000 RelVal results).", text)

    def test_peak_needs_a_relval_subject(self):
        self.assertIsNone(self._run("when was the peak memory of the 20_1 build?", self._rows())[0])

    def test_no_verdict_when_not_asked(self):
        text, _ = self._run("relval trend for 20_1 over time", self._rows())
        self.assertNotIn("by the failure rate", text)


class SeriesWeekStatusTests(unittest.TestCase):
    """R1-12: "give me a short status of the 20_1 IBs this week for management" -> AI: "2 failures each, stable all week",
    "2 build errors" (11-13 a build on 11-06..11-08, 9 build errors)."""

    def test_status_note_lists_build_errors_and_newest_unit_tests(self):
        session = unittest.mock.MagicMock()
        session.run.side_effect = [
            [dict(tag="CMSSW_20_1_X_2026-11-06-2300", arch="el9_aarch64_gcc14"), dict(tag="CMSSW_20_1_ROOT6_X_2026-11-06-2300", arch="el9_aarch64_gcc14"),
             dict(tag="CMSSW_20_1_EVTGEN3_X_2026-11-05-2300", arch="el9_amd64_gcc14")],
            [dict(tag="CMSSW_20_1_X_2026-11-09-2300", arch="el9_amd64_gcc14", total=384, failed=["DQMServices/Demo", None]),
             dict(tag="CMSSW_20_1_X_2026-11-10-1100", arch="el9_amd64_gcc14", total=100, failed=[]),
             dict(tag="CMSSW_20_1_X_2026-11-08-2300", arch="el9_amd64_gcc14", total=384, failed=["A/B", "C/D"]),
             dict(tag="CMSSW_20_1_X_2026-11-09-2300", arch="el9_aarch64_gcc14", total=377, failed=[])]]
        driver = unittest.mock.MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver):
            text = api_server._series_week_status_note("CMSSW_20_1", None, "2026-11-03")
        self.assertIn("**Build errors since 2026-11-03: 3** (primary and sub-IBs):", text)
        self.assertIn("- `el9_aarch64_gcc14`: `CMSSW_20_1_X_2026-11-06-2300`, `CMSSW_20_1_ROOT6_X_2026-11-06-2300`", text)
        self.assertIn("- `el9_amd64_gcc14` `CMSSW_20_1_X_2026-11-09-2300`: 1 of 384 packages failing: `DQMServices/Demo`", text)
        self.assertIn("- `el9_aarch64_gcc14` `CMSSW_20_1_X_2026-11-09-2300`: all 377 packages pass", text)

    def test_window_status_is_not_rewritten_to_the_newest_build(self):
        for q in ("give me a short status of the 20_1 IBs this week for management", "weekly summary for 16_0 please",
                  "20_1 status report for the past 5 days"):
            self.assertTrue(api_server._SERIES_WINDOW_STATUS_RE.search(q), q)
            self.assertIsNone(api_server._shift_question_rewrite(q), q)
            with patch.object(api_server, "_latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-10-1100"}]), \
                    patch.object(api_server, "_status_line", return_value="LINE"):
                self.assertIsNone(api_server._one_line_status_answer(q), q)
        for q in ("short status of 20_1", "what's the status of 20_1?", "status of my PR this week"):
            self.assertFalse(api_server._SERIES_WINDOW_STATUS_RE.search(q) and api_server._requested_release_series(q), q)

    def test_status_lead_line(self):
        rows = [dict(day=f"2026-11-0{d}", tag=f"CMSSW_20_1_X_2026-11-0{d}-1100", arch="el9_amd64_gcc14", total=954, failed=f) for d, f in ((3, 1), (4, 12), (5, 2))]
        session = unittest.mock.MagicMock()
        session.run.return_value = rows
        driver = unittest.mock.MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "_latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-05-1100"}]), \
                patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "partial_builds", return_value=[]), \
                patch.object(api_server, "_series_week_status_note", return_value="\n\nNOTE"):
            text = api_server._health_trend_answer("short status of the 20_1 IBs this week")
        self.assertTrue(text.startswith("**`CMSSW_20_1_X`, 2026-11-03 to 2026-11-05: 15 failed RelVal results in 3 primary build(s)**, worst day 2026-11-04 (12)"), text[:200])
        self.assertTrue(text.endswith("NOTE"))

    def test_short_status_is_compact(self):
        rows = [dict(day=f"2026-11-0{d}", tag=f"CMSSW_20_1_X_2026-11-0{d}-1100", arch="el9_amd64_gcc14", total=954, failed=f) for d, f in ((3, 1), (4, 12), (5, 2))]
        session = unittest.mock.MagicMock()
        session.run.side_effect = [rows, [dict(tag="CMSSW_20_1_ROOT6_X_2026-11-04-1100", arch="el9_aarch64_gcc14")],
                                   [dict(tag="CMSSW_20_1_X_2026-11-05-1100", arch="el9_amd64_gcc14", total=384, failed=["DQMServices/Demo"])]]
        driver = unittest.mock.MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "_latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-05-1100"}]), \
                patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "partial_builds", return_value=[]):
            text = api_server._health_trend_answer("short status of the 20_1 IBs this week for management")
        self.assertNotIn("| Day |", text)
        self.assertIn("- **RelVal failures per day** (known errors excluded): 11-03 1, 11-04 12, 11-05 2.", text)
        self.assertIn("- **Build errors: 1** (primary and sub-IBs): `el9_aarch64_gcc14` ROOT6 2026-11-04.", text)
        self.assertIn("`el9_amd64_gcc14` 1 (`DQMServices/Demo`)", text)

    def test_a_period_answer_with_one_partial_tag_is_not_a_single_build_answer(self):
        q = "give me a short status of the 20_1 IBs this week for management"
        period = ("**`CMSSW_20_1_X`, 2026-11-03 to 2026-11-09: 15 failed** ... Build errors: `el9_aarch64_gcc14` ROOT6 2026-11-06, primary 2026-11-08.\n\n"
                  "_Left out, still being imported: `CMSSW_20_1_X_2026-11-10-1100` on `el9_amd64_gcc14`_")
        self.assertIsNone(api_server._scope_mismatch(q, period))
        self.assertEqual(api_server._scope_mismatch(q, "What's failing in `CMSSW_20_1_X_2026-11-09-2300`: 2 RelVals"), "the whole period you asked about")

    def test_status_words_need_a_series(self):
        with patch.object(api_server, "_latest_ib_records", return_value=[]):
            self.assertIsNone(api_server._health_trend_answer("give me a short status of the 20_1 IBs this week"))
        self.assertIsNone(api_server._health_trend_answer("status of my PR this week"))


class LengthOnlyJudgeComplaintTests(unittest.TestCase):
    """R1-13: the AI judge sent a right weekly status back as "too detailed, not short"; the investigation that replaced it
    said 2 build errors (10)."""

    def test_length_only(self):
        for p in ("Answer too detailed, not short", "Answer is too detailed, not a short status", "Too long for a management summary", "verbose"):
            self.assertTrue(api_server._LENGTH_ONLY_PROBLEM_RE.search(p) and not api_server._NOT_LENGTH_PROBLEM_RE.search(p), p)

    def test_real_problems_still_count(self):
        for p in ("Asked for a short management summary; got an exhaustive raw failure dump of one build",
                  "Too detailed and about the wrong architecture", "answers a different question", "only the latest build, not the week"):
            self.assertFalse(api_server._LENGTH_ONLY_PROBLEM_RE.search(p) and not api_server._NOT_LENGTH_PROBLEM_RE.search(p), p)


class OneLineStatusEmptyNewestTests(unittest.TestCase):
    """R1-14: "short status of 20_1" -> "`CMSSW_20_1_X_2026-11-10-1600`: all builds passed, 0 RelVal and 0 unit-test failures" (nothing imported)."""

    def test_empty_newest_build_is_replaced_and_said(self):
        with patch.object(api_server, "_latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-10-1600"}]), \
                patch.object(api_server, "_with_results_or_newest_complete", return_value="CMSSW_20_1_X_2026-11-09-2300") as pick, \
                patch.object(api_server, "_status_line", side_effect=lambda tag, q, safe: f"LINE {tag}"):
            text = api_server._one_line_status_answer("short status of 20_1")
        self.assertEqual(pick.call_args[0][:3], ("CMSSW_20_1_X_2026-11-10-1600", None, (20, 1)))
        self.assertTrue(text.startswith("_`CMSSW_20_1_X_2026-11-10-1600` has no RelVal or unit-test results imported yet"))
        self.assertTrue(text.endswith("LINE CMSSW_20_1_X_2026-11-09-2300"))

    def test_build_with_results_stays_without_a_note(self):
        with patch.object(api_server, "_latest_ib_records", return_value=[{"tag": "CMSSW_17_1_X_2026-11-10-1100"}]), \
                patch.object(api_server, "_with_results_or_newest_complete", side_effect=lambda tag, *a: tag), \
                patch.object(api_server, "_status_line", side_effect=lambda tag, q, safe: f"LINE {tag}"):
            self.assertEqual(api_server._one_line_status_answer("short status of 17_1"), "LINE CMSSW_17_1_X_2026-11-10-1100")


class ShiftPairWithResultsTests(unittest.TestCase):
    """R1-15: "what's new since yesterday in the IBs?" -> digest of 11-10-1100 -> 11-10-1600 with every section "not imported yet"."""

    def _run(self, empty):
        with patch.object(api_server, "_latest_and_previous_ib_tags", return_value=("CMSSW_20_1_X_2026-11-10-1600", "CMSSW_20_1_X_2026-11-10-1100")), \
                patch.object(api_server, "_ib_has_no_results", return_value=empty), \
                patch.object(api_server, "_newest_complete_primary_tag", return_value=("CMSSW_20_1_X_2026-11-09-2300", "")), \
                patch.object(api_server, "_previous_primary_tag", return_value="CMSSW_20_1_X_2026-11-09-1100"), \
                patch.object(api_server, "generate_shift_summary", side_effect=lambda latest, previous, arch: f"## Shift summary: {previous} -> {latest}"), \
                patch.object(api_server, "_condense_shift_summary_for_chat", side_effect=lambda md: md):
            return api_server._shift_pair_with_results(None)

    def test_empty_newest_build_moves_to_the_newest_complete_pair(self):
        text = self._run(True)
        self.assertTrue(text.startswith("_`CMSSW_20_1_X_2026-11-10-1600` has no RelVal or unit-test results imported yet"))
        self.assertTrue(text.endswith("## Shift summary: CMSSW_20_1_X_2026-11-09-1100 -> CMSSW_20_1_X_2026-11-09-2300"))

    def test_newest_build_with_results_keeps_the_usual_pair(self):
        self.assertIsNone(self._run(False))


class OldestOpenFailureTests(unittest.TestCase):
    """R1-16: "what's the oldest still-open failure in the 20_1 primary IBs?" -> the newest build's overview; and the longest-failing
    table looked at one build only, so the gcc15/gcc16 builds (every other day) were left out, and it ranked by builds in a row."""

    def test_wording(self):
        for q in ("what's the oldest still-open failure in the 20_1 primary IBs?", "oldest open failure in 16_0", "which workflow has been failing the longest?",
                  "longest-standing failure in 20_1?"):
            with patch.object(api_server, "_series_prefix", return_value=("CMSSW_20_1_X_", (20, 1))), \
                    patch.object(api_server, "_newest_complete_primary_tag", return_value=(None, "")) as newest:
                api_server._longest_failing_answer(q)
                self.assertTrue(newest.called, q)
        for q in ("oldest open issue about aarch64 failures", "oldest failing unit test in 20_1"):
            with patch.object(api_server, "_newest_complete_primary_tag", return_value=(None, "")) as newest:
                self.assertIsNone(api_server._longest_failing_answer(q))
                self.assertFalse(newest.called, q)

    def test_each_architecture_by_its_own_newest_build_oldest_first_and_mass_failures_apart(self):
        newest = {None: "CMSSW_20_1_X_2026-11-09-2300", "el9_amd64_gcc14": "CMSSW_20_1_X_2026-11-09-2300",
                  "el9_amd64_gcc16": "CMSSW_20_1_X_2026-11-05-2300", "fc42_riscv64_gcc15": "CMSSW_20_1_X_2026-11-05-2300"}
        records = {("CMSSW_20_1_X_2026-11-09-2300", "el9_amd64_gcc14"): [dict(variant="primary", arch="el9_amd64_gcc14", relvals=955, failed=["36434.0"])],
                   ("CMSSW_20_1_X_2026-11-05-2300", "el9_amd64_gcc16"): [dict(variant="primary", arch="el9_amd64_gcc16", relvals=954, failed=["537.0"])],
                   ("CMSSW_20_1_X_2026-11-05-2300", "fc42_riscv64_gcc15"): [dict(variant="primary", arch="fc42_riscv64_gcc15", relvals=954, failed=[f"{i}.0" for i in range(900)])]}
        history = [dict(wf="36434.0", arch="el9_amd64_gcc14", tag=f"CMSSW_20_1_X_2026-11-0{d}-2300", status="FAILED", known=False, name="TTbar_14TeV+Run4D110FS") for d in (6, 7, 8, 9)]
        history += [dict(wf="36434.0", arch="el9_amd64_gcc14", tag="CMSSW_20_1_X_2026-11-05-2300", status="PASSED", known=False, name="TTbar_14TeV+Run4D110FS")]
        history += [dict(wf="537.0", arch="el9_amd64_gcc16", tag=f"CMSSW_20_1_X_2026-11-0{d}-2300", status="FAILED", known=False, name="DYToLL") for d in (3, 5)]
        history += [dict(wf="537.0", arch="el9_amd64_gcc16", tag="CMSSW_20_1_X_2026-10-29-2300", status="PASSED", known=False, name="DYToLL")]
        session = unittest.mock.MagicMock()
        session.run.side_effect = [[{"arch": a} for a in ("el9_amd64_gcc14", "el9_amd64_gcc16", "fc42_riscv64_gcc15")], history]
        driver = unittest.mock.MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "_series_prefix", return_value=("CMSSW_20_1_X_", (20, 1))), \
                patch.object(api_server, "_newest_complete_primary_tag", side_effect=lambda kind, arch, series: (newest[arch], "")), \
                patch.object(api_server, "_ib_build_records", side_effect=lambda tag, arch=None: records[(tag, arch)]), \
                patch.object(api_server, "get_driver", return_value=driver):
            text = api_server._longest_failing_answer("what's the oldest still-open failure in the 20_1 primary IBs?")
        rows = [l for l in text.splitlines() if l.startswith("| `")]
        self.assertTrue(rows[0].startswith("| `537.0` | DYToLL | `el9_amd64_gcc16` | 2 | `CMSSW_20_1_X_2026-11-03-2300` | `CMSSW_20_1_X_2026-11-05-2300` |"), rows)
        self.assertTrue(rows[1].startswith("| `36434.0`"), rows)
        self.assertEqual(len(rows), 2)
        self.assertIn("Most of the matrix failing, listed apart: `fc42_riscv64_gcc15`: 900 of 954 RelVals failing in `CMSSW_20_1_X_2026-11-05-2300`.", text)


class PassRateTests(unittest.TestCase):
    """R1-17: "what's the relval pass rate in the latest complete 20_1 IB on el9?" -> failing counts only."""

    def _run(self, q):
        records = [dict(tag="CMSSW_20_1_X_2026-11-09-2300", arch="el9_amd64_gcc14", variant="primary", build="warning", relvals=955, failed=["36434.0", "36834.0"],
                        units=384, unit_failed=["DQMServices/Demo"])]
        with patch.object(api_server, "_latest_or_named_tag", return_value="CMSSW_20_1_X_2026-11-09-2300"), \
                patch.object(api_server, "_ib_build_records", return_value=records):
            return api_server._arch_word_results_answer(q)

    def test_rate_when_asked(self):
        text = self._run("what's the relval pass rate in the latest complete 20_1 IB on el9?")
        self.assertIn("RelVals: 2 failing of 955, 99.79% not failing", text)
        self.assertIn("known errors and DAS errors", text)

    def test_no_rate_otherwise(self):
        self.assertNotIn("not failing", self._run("relvals failing on el9 in the latest 20_1 IB?"))


class BuildFixPrsTests(unittest.TestCase):
    """R1-18: "what broke in 20_1 on 2026-11-06?" ... "which PR fixed it?" -> the AI: "there likely isn't one" (11-09-1100 has #52432)."""

    def _answer(self):
        wfs = ["7.5", "27234.0", "36434.0"]
        rows = []
        for later, statuses in (("CMSSW_20_1_X_2026-11-08-2300", (True, True, True)), ("CMSSW_20_1_X_2026-11-09-1100", (False, False, True)),
                                ("CMSSW_20_1_X_2026-11-09-2300", (False, False, True))):
            rows += [dict(arch="el9_amd64_gcc14", wfs=wfs, later=later, wf=w, failed=f) for w, f in zip(wfs, statuses)]
        session = unittest.mock.MagicMock()
        session.run.return_value = rows
        driver = unittest.mock.MagicMock()
        driver.session.return_value.__enter__.return_value = session
        prs = [dict(repo="cmssw", pr_number=52432, title="MTD digitization: add a protection", merged_at="2026-11-09T06:04:28", author="fabiocos")]
        with patch.object(api_server, "get_driver", return_value=driver), patch.object(api_server, "find_prs_before_ib", return_value=prs) as before:
            text = api_server._build_fix_prs_answer("CMSSW_20_1_X_2026-11-06-2300")
        before.assert_called_once_with("CMSSW_20_1_X_2026-11-09-1100", limit=100)
        return text

    def test_biggest_group_and_its_prs(self):
        text = self._answer()
        self.assertTrue(text.startswith("**2 of the 3 RelVal failures of `CMSSW_20_1_X_2026-11-06-2300`**"), text[:120])
        self.assertIn("passed again in `CMSSW_20_1_X_2026-11-09-1100` and since: `7.5`, `27234.0` on `el9_amd64_gcc14`.", text)
        self.assertIn("Still failing (or failing again) in the newest build: `36434.0` on `el9_amd64_gcc14`.", text)
        self.assertIn("- **cmssw #52432** - MTD digitization: add a protection (merged 2026-11-09 06:04; by fabiocos)", text)
        self.assertIn("never a proven fix", text)

    def test_follow_up_routing(self):
        api_server._turn_memory["r1-18"] = {"question": "what broke in 20_1 on 2026-11-06?", "answer": "Imported results for primary IB", "route": "deterministic"}
        api_server._latest_ib_contexts["r1-18"] = "CMSSW_20_1_X_2026-11-06-2300"
        try:
            with patch.object(api_server, "_build_fix_prs_answer", return_value="FIX") as fix, \
                    patch.object(api_server, "_meta_conversation_answer", return_value=None):
                self.assertEqual(api_server._listed_items_follow_up("which PR fixed it?", "r1-18"), "FIX")
                fix.assert_called_once_with("CMSSW_20_1_X_2026-11-06-2300")
        finally:
            api_server._turn_memory.pop("r1-18", None)
            api_server._latest_ib_contexts.pop("r1-18", None)


class KnownErrorsPerSeriesTests(unittest.TestCase):
    """R1-19: "do the aarch64 known errors have the same count in every release?" -> the 20_1 aarch64 failure overview (98 s)."""

    def _run(self, q, rows):
        session = unittest.mock.MagicMock()
        session.run.return_value = rows
        driver = unittest.mock.MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver):
            return api_server._known_errors_per_series_answer(q)

    def test_per_series_newest_complete_build(self):
        rows = [dict(tag="CMSSW_20_1_X_2026-11-09-2300", arch="el9_aarch64_gcc14", total=955, known=50),
                dict(tag="CMSSW_20_1_X_2026-11-10-2300", arch="el9_aarch64_gcc14", total=100, known=10),
                dict(tag="CMSSW_17_1_X_2026-11-08-2300", arch="el8_aarch64_gcc13", total=1252, known=60),
                dict(tag="CMSSW_17_1_ROOT6_X_2026-11-08-2300", arch="el8_aarch64_gcc13", total=1252, known=60)]
        text = self._run("do the aarch64 known errors have the same count in every release?", rows)
        self.assertTrue(text.startswith("**No: from 50 to 60 known-error RelVals per build on `aarch64`**"), text[:100])
        self.assertIn("| `CMSSW_20_1` | `el9_aarch64_gcc14` | `2026-11-09-2300` | 50 | 955 |", text)
        self.assertNotIn("2026-11-10-2300", text)
        self.assertEqual(text.count("| `CMSSW_17_1` |"), 1)

    def test_same_count_says_yes(self):
        rows = [dict(tag="CMSSW_17_1_X_2026-11-08-2300", arch="el8_aarch64_gcc13", total=1252, known=60),
                dict(tag="CMSSW_17_0_X_2026-11-08-2300", arch="el8_aarch64_gcc13", total=1252, known=60)]
        self.assertTrue(self._run("known errors per release on aarch64?", rows).startswith("**Yes: 60 known-error RelVals in every release series on `aarch64`**"))

    def test_wording(self):
        self.assertIsNone(api_server._known_errors_per_series_answer("how many known errors in the latest 20_1 IB?"))


class BareBuildTimeSeriesTests(unittest.TestCase):
    """R1-20: "unit tst failures 17_1 11-09-2300" -> the CMSSW_20_1_X_2026-11-09-2300 unit tests (the newest series, not the one named)."""

    def test_named_series_is_used(self):
        with patch.object(api_server, "_latest_series_prefix", return_value="CMSSW_20_1_X_"):
            self.assertIn("CMSSW_17_1_X_2026-11-09-2300", api_server._normalize_terse_words("unit tst failures 17_1 11-09-2300"))
            self.assertIn("CMSSW_17_1_X_2026-11-09-2300", api_server._normalize_terse_words("relvals 17_1 2026-11-09-2300"))
            self.assertIn("CMSSW_20_1_X_2026-11-09-2300", api_server._normalize_terse_words("unit test failures in 11-09-2300"))
            self.assertIn("CMSSW_16_0_X_2026-11-10-1100", api_server._normalize_terse_words("compare cmssw_16_0_x_2026-11-10-1100 with 11-08-2300 in 17_1"))


class ListedPrsTouchTests(unittest.TestCase):
    """R1-21: "any of them touch Configuration/PyReleaseValidation?" after 8 listed PRs -> a text search over all PRs (21 unrelated items)."""

    def _files(self, number, repo, path):
        data = {52396: ["DataFormats/Portable/test/a.cc"], 51861: ["RecoLocalFastTime/FTLRecProducers/plugins/x.cc"]}
        if number == 52432:
            return {"found": True, "title": "MTD", "files": None, "files_status": "unknown: this PR is newer than our GitHub backup"}
        files = data[number]
        hits = [f for f in files if f.startswith(path)]
        return {"found": True, "title": f"title {number}", "files": files, "touched_asked_path": bool(hits), "matching_files": hits}

    def test_touch_check_per_listed_pr(self):
        prev = "| [#52396](https://github.com/cms-sw/cmssw/pull/52396) | x |\n- **cmssw #51861** - MTD\n- **cmssw #52432** - BTL"
        with patch.object(api_server, "pr_files", side_effect=self._files):
            text = api_server._listed_prs_touch_answer("any of them touch DataFormats/Portable?", prev)
            self.assertTrue(text.startswith("**Yes: 1 of the 3 listed PR(s) touched `DataFormats/Portable`:**"), text[:100])
            self.assertIn("Not touching it: #51861.", text)
            self.assertIn("cmssw #52432 (newer than our GitHub backup)", text)
            no = api_server._listed_prs_touch_answer("did any of them change Configuration/PyReleaseValidation?", prev)
            self.assertTrue(no.startswith("**No listed PR with known files touched `Configuration/PyReleaseValidation`** (2 checked"), no[:120])

    def test_ai_table_links_without_hash(self):
        prev = ("| [52396](https://github.com/cms-sw/cmssw/pull/52396) | Fix CUDA | raoatifshad |\n| 51861 | MTD |\n"
                "**By merge time (cmsdist):** [#10986](https://github.com/cmsdist/cmsdist/pull/10986) - Update tag")
        with patch.object(api_server, "pr_files", side_effect=self._files):
            text = api_server._listed_prs_touch_answer("did any of them change DataFormats/Portable?", prev)
        self.assertTrue(text.startswith("**Yes: 1 of the 3 listed PR(s) touched `DataFormats/Portable`:**"), text[:100])
        self.assertIn("cmsdist #10986 (file lists are imported for cmssw PRs only)", text)

    def test_reads_the_answer_that_was_shown(self):
        api_server._turn_memory["r1-21"] = {"question": "PRs between A and B", "answer": "rule answer #52046 #50985", "route": "deterministic", "superseded": True}
        api_server._ai_supervisor_history["r1-21"] = [{"question": "PRs between A and B", "answer": "| [52396](https://github.com/cms-sw/cmssw/pull/52396) |", "route": "ai:llm_validated"}]
        try:
            with patch.object(api_server, "_listed_prs_touch_answer", return_value="TOUCH") as touch, \
                    patch.object(api_server, "_meta_conversation_answer", return_value=None):
                self.assertEqual(api_server._listed_items_follow_up("did any of them change DataFormats/Portable?", "r1-21"), "TOUCH")
                self.assertIn("pull/52396", touch.call_args[0][1])
        finally:
            api_server._turn_memory.pop("r1-21", None)
            api_server._ai_supervisor_history.pop("r1-21", None)

    def test_not_without_a_list_or_a_path(self):
        self.assertIsNone(api_server._listed_prs_touch_answer("any of them touch Configuration/PyReleaseValidation?", "no PR here"))
        self.assertIsNone(api_server._listed_prs_touch_answer("any of them touch anything?", "#52396"))


class TrendPeakFollowUpTests(unittest.TestCase):
    """R1-9: "trend of relval failures in 20_1 el9_amd64_gcc14 over the last 10 days" then "when was the peak?" -> the AI, which
    summed the sub-IBs into a table that contradicted the one just shown."""

    def setUp(self):
        api_server._turn_memory["r1-9"] = {"question": "trend of relval failures in 20_1 el9_amd64_gcc14 over the last 10 days",
                                           "answer": "Per day for `CMSSW_20_1_X` primary IBs on `el9_amd64_gcc14` (known errors excluded):\n\n| Day |",
                                           "route": "deterministic:health_trend"}

    def tearDown(self):
        api_server._turn_memory.pop("r1-9", None)

    def test_peak_follow_ups(self):
        for q in ("when was the peak?", "which day was the worst?", "and the peak?", "worst day?"):
            self.assertEqual(api_server._resolve_short_follow_up(q, "r1-9"),
                             "trend of relval failures in 20_1 el9_amd64_gcc14 over the last 10 days, which day was the peak?", q)

    def test_not_after_another_answer(self):
        api_server._turn_memory["r1-9"]["answer"] = "CMSSW_20_1_X_2026-11-09-2300 has 6 failing RelVals"
        self.assertNotEqual(api_server._resolve_short_follow_up("when was the peak?", "r1-9"),
                            "trend of relval failures in 20_1 el9_amd64_gcc14 over the last 10 days, which day was the peak?")


class PackageWhatsWrongTests(unittest.TestCase):
    """R1-11: "what's wrong with RecoTracker/LSTCore on gcc16?" went to the AI ("only CPP23, since 11-05"); and a package failing
    in every imported build had "since <oldest build>" as if it broke there."""

    def test_whats_wrong_is_a_package_status_question(self):
        with patch.object(api_server, "_package_newest_status_answer", return_value="STATUS") as newest:
            self.assertEqual(api_server._package_status_answer("what's wrong with RecoTracker/LSTCore on gcc16?", "r1-11"), "STATUS")
            self.assertEqual(api_server._package_status_answer("what is going on with DQMServices/Demo?", "r1-11"), "STATUS")
            self.assertEqual(newest.call_count, 2)

    def _rows(self, statuses):
        return [dict(tag=f"CMSSW_20_1_X_2026-10-{d}-2300", variant="primary", arch="el9_amd64_gcc16", status=st, errors=1)
                for d, st in statuses]

    def _answer(self, rows):
        session = unittest.mock.MagicMock()
        session.run.return_value = rows
        driver = unittest.mock.MagicMock()
        driver.session.return_value.__enter__.return_value = session
        with patch.object(api_server, "get_driver", return_value=driver), \
                patch.object(api_server, "_package_failing_tests_note", return_value=""):
            return api_server._package_newest_status_answer("RecoTracker/LSTCore", "what's wrong with RecoTracker/LSTCore?")

    def test_failing_in_every_imported_build_says_where_the_data_starts(self):
        text = self._answer(self._rows([("27", "FAILED"), ("29", "FAILED"), ("30", "FAILED")]))
        self.assertIn("since `CMSSW_20_1_X_2026-10-27-2300` (the oldest imported result", text)

    def test_a_pass_before_names_the_last_good_build(self):
        text = self._answer(self._rows([("27", "PASSED"), ("29", "FAILED"), ("30", "FAILED")]))
        self.assertIn("since `CMSSW_20_1_X_2026-10-29-2300` (last passed in `CMSSW_20_1_X_2026-10-27-2300`)", text)
        self.assertNotIn("oldest imported result", text)


if __name__ == "__main__":
    unittest.main()
