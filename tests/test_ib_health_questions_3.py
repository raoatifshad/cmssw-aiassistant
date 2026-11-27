"""Wrong answers found in testing (2026-11-11): management-style questions, round 3."""
import unittest
from unittest.mock import MagicMock, patch

from server import api_server


class _Session:
    def __init__(self, rows, total):
        self.rows, self.total, self.calls = rows, total, []
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def run(self, query, **params):
        self.calls.append(params)
        if "RETURN count(p) AS n" in query and "AS author" not in query:
            return MagicMock(single=MagicMock(return_value={"n": self.total}))
        return self.rows


class _Driver:
    def __init__(self, session):
        self._session = session
    def session(self):
        return self._session
    def close(self):
        pass


class PrAuthorRankingOfADayTests(unittest.TestCase):
    """R3-1: "who merged the most PRs in 2026-11-04?" ranked the last 7 days, the AI then gave other numbers (bsunanda 5 for 3, 20 PRs for 21)."""

    def _ask(self, question):
        session = _Session([{"author": "bsunanda", "n": 3, "repos": ["cmssw"], "examples": [52374]}], 21)
        with patch.object(api_server, "get_driver", return_value=_Driver(session)):
            return api_server._pr_author_ranking_answer(question), session.calls[0]

    def test_a_named_day_is_ranked_on_its_own(self):
        answer, params = self._ask("who merged the most PRs in 2026-11-04?")
        self.assertEqual((params["since"], params["until"]), ("2026-11-04", "2026-11-05"))
        self.assertIn("merged on 2026-11-04", answer)
        self.assertIn("21 merged PR(s)", answer)

    def test_which_author_had_the_most_merged_prs_on_a_short_date(self):
        answer, params = self._ask("which author had the most merged PRs on 11-09?")
        self.assertTrue(params["since"].endswith("-11-09"))
        self.assertIn("| `bsunanda` | 3 |", answer)

    def test_a_rolling_window_is_unchanged(self):
        answer, params = self._ask("who merged the most PRs this week?")
        self.assertIsNone(params["until"])
        self.assertIn("in the last 7 days", answer)


class RecurrenceOfTheShownNameTests(unittest.TestCase):
    """R3-2: "why does 29751.85 fail in 14_1?" ended with "failed in 0 of 31" (the 20_1 name of the number); the 14_1 one failed 15 times."""

    def _rows(self):
        return [{"name": "W_2026D110", "status": "FAILED", "known_error": False, "exitcode": 63232, "tag": "CMSSW_14_1_X_2026-11-10-2300", "arch": "a", "variant": "primary"},
                {"name": "W_Run4D110", "status": "PASSED", "known_error": False, "exitcode": 0, "tag": "CMSSW_20_1_X_2026-11-10-2300", "arch": "a", "variant": "primary"}]

    def _recurrence(self, **kwargs):
        from graph import tools
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value.run.return_value = self._rows()
        with patch.object(tools, "get_driver", return_value=driver):
            return tools.get_relval_recurrence("29751.85", **kwargs)

    def test_default_is_the_newest_name(self):
        self.assertEqual(self._recurrence()["failed_appearances"], 0)

    def test_the_shown_name_is_used(self):
        got = self._recurrence(name="W_2026D110")
        self.assertEqual((got["failed_appearances"], got["total_appearances"]), (1, 1))

    def test_an_unknown_name_falls_back(self):
        self.assertEqual(self._recurrence(name="nothing")["name"], "W_Run4D110")


class WholeWordWithUnderscoreTests(unittest.TestCase):
    """R3-3: "how many open issues mention aarch64?" said 23; 8 more name it inside "el8_aarch64_gcc12" (the whole-word match took "_" as part of the word)."""

    def _pattern(self, text):
        from graph import analytics
        seen = {}
        def fake_rows(query, **params):
            seen.update(params)
            return []
        with patch.object(analytics, "_rows", side_effect=fake_rows):
            analytics.search_github(kind="issue", text=text, count_only="true")
        return seen["pattern"]

    def test_underscore_is_a_separator(self):
        import re
        pattern = self._pattern("aarch64")
        for body in ("fails on el8_aarch64_gcc12", "[AARCH64] crash", "aarch64.", "two aarch64s"):
            self.assertTrue(re.fullmatch(pattern, body), body)

    def test_inside_a_longer_word_is_still_no_match(self):
        import re
        pattern = self._pattern("ucx")
        for body in ("host ucxhost", "xucx", "ucx2"):
            self.assertFalse(re.fullmatch(pattern, body), body)


class PrsBetweenDaysOfASeriesTests(unittest.TestCase):
    """R3-4: "which PRs were merged in 14_1 between 11-05 and 11-09?" listed the 49 PRs of every release."""

    def _ask(self, question, rows):
        session = _Session(rows, 0)
        with patch.object(api_server, "get_driver", return_value=_Driver(session)):
            return api_server._prs_between_days_answer(question), session.calls[0]

    def test_the_series_is_passed_to_the_query(self):
        answer, params = self._ask("which PRs were merged in 14_1 between 11-05 and 11-09?", [])
        self.assertEqual(params["head"], "CMSSW_14_1")
        self.assertIn("No imported PR targeting `CMSSW_14_1_X` was merged", answer)

    def test_no_series_no_filter(self):
        row = {"repo": "cmssw", "n": 1, "title": "T", "merged": "2026-11-06T01:00:00Z", "release": "CMSSW_20_1", "who": "x"}
        answer, params = self._ask("which PRs were merged between 11-05 and 11-09?", [row])
        self.assertIsNone(params["head"])
        self.assertNotIn("targeting", answer)

    def test_listed_prs_say_the_filter(self):
        row = {"repo": "cmssw", "n": 52311, "title": "T", "merged": "2026-11-03T01:00:00Z", "release": "CMSSW_14_1", "who": "makortel"}
        answer, _ = self._ask("PRs merged in 14_1 between 11-02 and 11-04", [row])
        self.assertIn("targeting `CMSSW_14_1_X`", answer)
        self.assertIn("#52311", answer)


class ArchCompareWithOneTagTests(unittest.TestCase):
    """R3-5: "20_1 gcc14 vs gcc15 relval failures in the newest complete IB" (the build's tag is added to the text) ended in "Do you want a comparison of these Integration Builds?"."""

    def _records(self):
        base = {"variant": "primary", "build": "passed", "relvals": 900, "units": 300, "unit_failed": []}
        return [dict(base, tag="CMSSW_20_1_X_2026-11-10-1100", arch="el9_amd64_gcc14", failed=["1.0", "2.0"]),
                dict(base, tag="CMSSW_20_1_X_2026-11-10-1100", arch="el9_amd64_gcc15", failed=["1.0"])]

    def test_one_tag_in_the_text_is_the_build(self):
        question = "20_1 gcc14 vs gcc15 relval failures in the newest complete IB CMSSW_20_1_X_2026-11-10-1100"
        with patch.object(api_server, "_latest_or_named_tag", return_value="CMSSW_20_1_X_2026-11-10-1100"), patch.object(api_server, "_ib_build_records", return_value=self._records()):
            answer = api_server._arch_compare_answer(question)
        self.assertIn("`gcc14` and `gcc15` in `CMSSW_20_1_X_2026-11-10-1100`", answer)
        self.assertIn("Only on `gcc14`", answer)

    def test_a_typed_tag_stays_with_the_other_rules(self):
        self.assertIsNone(api_server._arch_compare_answer("gcc14 vs gcc15 in CMSSW_20_1_X_2026-11-10-1100"))

    def test_two_tags_are_a_comparison_of_builds(self):
        question = "compare gcc14 and gcc15 between CMSSW_20_1_X_2026-11-10-1100 and CMSSW_20_1_X_2026-11-09-2300"
        self.assertIsNone(api_server._arch_compare_answer(question))


class SameInTheMainBuildTests(unittest.TestCase):
    """R3-6: "same in the main 20_1 build?" after the failures of a ROOT6 build: "I couldn't match that to a specific local answer"."""

    def setUp(self):
        self.sid = "tester-r3-6"
        self.addCleanup(api_server._turn_memory.pop, self.sid, None)

    def test_main_build_is_the_primary_tag(self):
        api_server._turn_memory[self.sid] = {"question": "what failed in CMSSW_20_1_ROOT6_X_2026-11-10-2300?", "answer": "", "route": "capability:failure_overview"}
        for text in ("same in the main 20_1 build?", "is it the same in the primary build?", "and the same on the normal one?"):
            self.assertEqual(api_server._resolve_short_follow_up(text, self.sid), "what failed in CMSSW_20_1_X_2026-11-10-2300?", text)

    def test_a_flavour_still_works(self):
        api_server._turn_memory[self.sid] = {"question": "what failed in CMSSW_20_1_X_2026-11-10-2300?", "answer": "", "route": "capability:failure_overview"}
        self.assertEqual(api_server._resolve_short_follow_up("same for ROOT6", self.sid), "what failed in CMSSW_20_1_ROOT6_X_2026-11-10-2300?")


class BuildErrorTrendTests(unittest.TestCase):
    """R3-7: "trend of build errors in 20_1 over the last month" was answered with the RelVal failure-rate table."""

    def _rows(self):
        def row(day, status, arch="el9_aarch64_gcc14"):
            return {"day": day, "tag": f"CMSSW_20_1_X_{day}-2300", "arch": arch, "status": status}
        return [row("2026-11-05", "passed"), row("2026-11-06", "error"), row("2026-11-06", "error", "el9_aarch64_gcc14"), row("2026-11-07", None)]

    def _ask(self, question):
        session = _Session(self._rows(), 0)
        with patch.object(api_server, "get_driver", return_value=_Driver(session)), patch.object(api_server, "_latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-10-2300"}]):
            return api_server._health_trend_answer(question)

    def test_build_errors_are_counted_not_relvals(self):
        answer = self._ask("trend of build errors in 20_1 over the last month")
        self.assertIn("1 build(s) of the `CMSSW_20_1_X` primary IBs have a build error** (2 of 3 build/architecture records", answer)
        self.assertIn("| 2026-11-06 | 2 | 2 | 0 | 0 | `el9_aarch64_gcc14` x2 |", answer)
        self.assertIn("1 build/architecture record(s) in this period have no imported build status", answer)
        self.assertNotIn("RelVal results", answer)

    def test_this_month_starts_on_the_first(self):
        rows = [{"day": "2026-10-29", "tag": "CMSSW_20_1_X_2026-10-29-2300", "arch": "a", "status": "error"}] + self._rows()
        session = _Session(rows, 0)
        with patch.object(api_server, "get_driver", return_value=_Driver(session)), patch.object(api_server, "_latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-10-2300"}]):
            answer = api_server._health_trend_answer("how many 20_1 IBs have a build error this month?")
        self.assertIn("1 build(s)", answer)
        self.assertNotIn("2026-10-29", answer)

    def test_a_relval_trend_is_unchanged(self):
        self.assertFalse(api_server._BUILD_ERROR_TREND_RE.search("trend of relval failures in 20_1 over the last month"))
        self.assertFalse(api_server._BUILD_ERROR_TREND_RE.search("is 14_2 getting better or worse this week?"))


class CmsdistOpenPrsNoteTests(unittest.TestCase):
    """R3-8: "how many PRs are open for cmsdist?" -> "0 open PRs for cmsdist" (no open cmsdist PR is ever imported)."""

    def _note(self, **kwargs):
        from graph import analytics
        with patch.object(analytics, "_rows", return_value=[]), patch.object(analytics, "_github_data_reach", return_value="2026-11-06"):
            return analytics.search_github(count_only="true", **kwargs)["note"]

    def test_open_cmsdist_is_not_zero_but_unknown(self):
        self.assertIn("not in the imported data", self._note(kind="pr", repo="cmsdist", state="open"))

    def test_merged_cmsdist_and_cmssw_have_no_such_note(self):
        self.assertNotIn("open cmsdist PR", self._note(kind="pr", repo="cmsdist", state="merged"))
        self.assertNotIn("open cmsdist PR", self._note(kind="pr", repo="cmssw", state="open"))


class PrSearchTermsNoiseTests(unittest.TestCase):
    """R3 rerun: "any github issue on the failures in <tag> on el9_amd64_gcc12?" searched for "GitHub" and the architecture too."""

    def test_site_name_and_architecture_are_not_terms(self):
        terms = api_server._pr_search_terms("are there any GitHub issues about 10804.31 and 180.1 on el9_amd64_gcc12?")
        self.assertNotIn("GitHub", terms)
        self.assertNotIn("el9_amd64_gcc12", terms)
        self.assertIn("10804.31", terms)

    def test_an_architecture_alone_is_still_a_term(self):
        self.assertIn("el9_amd64_gcc12", api_server._pr_search_terms("issues mentioning `el9_amd64_gcc12`"))


if __name__ == "__main__":
    unittest.main()
