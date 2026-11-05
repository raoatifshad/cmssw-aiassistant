"""Gaps found on 2026-11-05 by re-asking every open item of the 2026-11-04 test pass, and regressions the replay caught in the fixes."""
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from ingestion.build_graph import github_reference_numbers
from server.api_server import (
    _arch_on_day_answer,
    _backport_answer,
    _basic_term_answer,
    _build_error_answer,
    _clang_detail_answer,
    _condense_one_arch_comparison,
    _digest_filter_answer,
    _exit_code_meaning_answer,
    _failure_sets_answer,
    _flavours_by_time_answer,
    _github_small_number_answer,
    _ibs_for_arch_answer,
    _items_mentioning_answer,
    _log_missing_days_answer,
    _log_problem_filter_answer,
    _meta_conversation_answer,
    _most_commented_answer,
    _named_day,
    _non_english_answer,
    _pr_core_title,
    _prs_between_days_answer,
    _recent_prs_answer,
    _same_tag_compare_answer,
    _session_questions,
    _shift_question_rewrite,
    _with_named_flavour_tag,
    _workflow_on_day_answer,
    _workflow_step_answer,
)


def _driver(*results):
    """A driver whose session.run returns each given list of rows in turn (the last one repeats)."""
    session = MagicMock()
    queue = list(results)

    def run(*_args, **_kwargs):
        rows = queue.pop(0) if len(queue) > 1 else queue[0]
        result = MagicMock()
        result.__iter__.side_effect = lambda: iter(rows)
        result.single.return_value = rows[0] if rows else None
        return result

    session.run.side_effect = run
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    return driver


class ReferenceParsingTests(unittest.TestCase):
    def test_stack_frames_are_not_references(self):
        text = ("#0  0x0000150cca11bb51 in poll () from /lib64/libc.so.6\n#3  <signal handler called>\n"
                "    #1 0x153be25d7c3c in BasicSingleTrajectoryState::BasicSingleTrajectoryState()\n#1  edm::Foo::bar (x=1) at file.cc:12")
        self.assertEqual(github_reference_numbers(text), [])

    def test_counters_under_100_are_not_references(self):
        self.assertEqual(github_reference_numbers("Tested with whiteRabbit #3, thread #120 crashed"), [])

    def test_real_references_stay(self):
        self.assertEqual(github_reference_numbers("backport of #52372, see #4532\nhttps://github.com/cms-sw/cmssw/pull/7"), [7, 4532, 52372])


class NonEnglishTests(unittest.TestCase):
    def test_says_so_instead_of_guessing(self):
        for question in ("¿qué está fallando en el último IB?", "आज क्या फेल हुआ", "最新のIBは何ですか", "quels tests échouent aujourd'hui ?", "welche IB ist die neueste?"):
            self.assertIn("only answer in English", _non_english_answer(question), question)

    def test_english_and_pastes_are_left_alone(self):
        for question in ("what is failing in the latest IB", "PR 52372", "Fatal Exception: est la\nline two",
                         "No dictionary for edm::Wrapper<int>", "is there a fix", "why does it fail"):
            self.assertIsNone(_non_english_answer(question), question)


class GlossaryReachTests(unittest.TestCase):
    def test_terms_written_but_never_reached(self):
        for question, expected in (("what does IB stand for?", "Integration Build"), ("IB meaning?", "Integration Build"),
                                   ("what's an architecture string like el9_amd64_gcc14?", "operating system"), ("el8_amd64_gcc13 means what?", "operating system"),
                                   ("what's the difference between step2 and step3?", "cmsRun"), ("what is PileUp in a workflow name?", "Pileup"),
                                   ("what's a clang static analyzer warning?", "static analyzer"), ("what's fc42?", "Fedora"), ("what is a release cycle in cmssw?", "release cycle")):
            self.assertIn(expected, _basic_term_answer(question) or "", question)

    def test_jenkins_job_with_ib_in_its_name_is_not_a_definition(self):
        self.assertIsNone(_basic_term_answer("what does ib-run-gpu-relvals depend on"))

    def test_data_questions_are_not_definitions(self):
        for question in ("what is failing in pileup workflows", "what is the status of step1 failures"):
            self.assertIsNone(_basic_term_answer(question), question)

    @patch("server.api_server.get_exit_code_names", return_value={})
    def test_exit_code_zero(self, _names):
        self.assertIn("finished successfully", _exit_code_meaning_answer("exit code 0?"))


class DayWordTests(unittest.TestCase):
    @patch("server.api_server.datetime")
    def test_named_days(self, clock):
        clock.now.return_value = datetime(2026, 11, 5, 12)
        clock.strptime = datetime.strptime
        for question, day in (("yesterday", "2026-11-04"), ("the day before yesterday", "2026-11-03"), ("on Monday", "2026-11-02"), ("on thursday", "2026-10-29"), ("on 11-02", "2026-11-02"), ("2026-10-30", "2026-10-30")):
            self.assertEqual(_named_day(question), day, question)
        self.assertIsNone(_named_day("in the latest IB"))


class WorkflowOnDayTests(unittest.TestCase):
    @patch("server.api_server.get_exit_code_names", return_value={31744: "TimeOut"})
    @patch("server.api_server.get_driver")
    def test_says_no_with_the_failures(self, get_driver, _names):
        get_driver.return_value = _driver([
            {"tag": "CMSSW_20_1_X_2026-11-02-2300", "arch": "el9_amd64_gcc14", "variant": "primary", "status": "FAILED", "exit": "31744", "known": False},
            {"tag": "CMSSW_20_1_X_2026-11-02-1100", "arch": "el9_amd64_gcc14", "variant": "primary", "status": "PASSED", "exit": "0", "known": False}])
        answer = _workflow_on_day_answer("was 539.0 green on 2026-11-02?")
        self.assertTrue(answer.startswith("**No.**"))
        self.assertIn("31744 (TimeOut)", answer)

    @patch("server.api_server.get_driver")
    def test_a_known_error_is_not_a_failure(self, get_driver):
        get_driver.return_value = _driver([{"tag": "CMSSW_20_1_X_2026-11-02-2300", "arch": "el9_aarch64_gcc14", "variant": "primary", "status": "FAILED", "exit": "1", "known": True}])
        with patch("server.api_server.get_exit_code_names", return_value={}):
            self.assertTrue(_workflow_on_day_answer("did 1.0 pass on 2026-11-02?").startswith("**Yes.**"))

    def test_only_with_a_day_and_a_verdict_word(self):
        self.assertIsNone(_workflow_on_day_answer("where is 537.0 failing"))
        self.assertIsNone(_workflow_on_day_answer("what is 537.0 on 11-02"))


class WorkflowStepTests(unittest.TestCase):
    @patch("server.api_server.get_exit_code_names", return_value={64000: "SIGABRT"})
    @patch("server.api_server.get_driver")
    def test_reads_the_step_from_the_logs(self, get_driver, _names):
        get_driver.return_value = _driver(
            [{"tag": "CMSSW_20_1_X_2026-11-03-2300", "arch": "el10_amd64_gcc15", "run": "", "details": ["537.0_DY: step1=FAILED step2=NORUN; exit 64000 0"]}], [])
        answer = _workflow_step_answer("which step of 537.0 fails?", "s")
        self.assertIn("fails in **step1**", answer)
        self.assertIn("64000 (SIGABRT)", answer)

    @patch("server.api_server.get_exit_code_names", return_value={})
    @patch("server.api_server.get_driver")
    def test_different_steps_are_said(self, get_driver, _names):
        get_driver.return_value = _driver(
            [{"tag": "CMSSW_20_1_X_2026-11-03-2300", "arch": "a", "run": "", "details": ["1.0_x: step1=PASSED step2=FAILED; exit 0 256"]},
             {"tag": "CMSSW_20_1_X_2026-11-02-2300", "arch": "a", "run": "", "details": ["1.0_x: step1=FAILED step2=NORUN; exit 256 0"]}], [])
        self.assertIn("doesn't always fail in the same step", _workflow_step_answer("does 1.0 fail in the same step everywhere?", "s"))

    def test_definitions_are_left_alone(self):
        self.assertIsNone(_workflow_step_answer("what is a step", "s"))
        self.assertIsNone(_workflow_step_answer("where is 537.0 failing", "s"))


class GithubAnswerTests(unittest.TestCase):
    def test_numbers_below_one(self):
        self.assertIn("start at 1", _github_small_number_answer("PR 0"))
        self.assertIn("start at 1", _github_small_number_answer("issue -1"))

    def test_normal_numbers_keep_their_own_answer(self):
        self.assertIsNone(_github_small_number_answer("PR 52372"))
        self.assertIsNone(_github_small_number_answer("Issue 51970"))

    def test_backport_title_core(self):
        self.assertEqual(_pr_core_title("[17_0_X] [L1T] Update P2GT"), "update p2gt")
        self.assertEqual(_pr_core_title("Backport of #123 to 16_0: Fix"), "fix")

    @patch("server.api_server._github_export_note", return_value="note")
    @patch("server.api_server.get_driver")
    def test_finds_backports(self, get_driver, _note):
        get_driver.return_value = _driver(
            [{"title": "[L1T] Update P2GT emulator", "release": "CMSSW_20_1_X", "merged": "2026-11-03T00:00:00Z"}],
            [{"n": 52352, "title": "[17_0_X] [L1T] Update P2GT emulator", "release": "CMSSW_17_0_X", "merged": "2026-11-04T00:00:00Z", "state": "closed", "same_title": True, "says_backport": False},
             {"n": 52000, "title": "Something else", "release": "CMSSW_20_1_X", "merged": None, "state": "open", "same_title": False, "says_backport": False}])
        answer = _backport_answer("is there a backport of 52372?")
        self.assertIn("#52352", answer)
        self.assertNotIn("#52000", answer)

    def test_a_number_alone_is_not_a_backport_question(self):
        self.assertIsNone(_backport_answer("PR 52372"))

    def test_rolling_windows_and_authors_stay_with_their_answers(self):
        for question in ("how many PRs did fwyzard merge this week?", "which cmsdist PRs were merged this week?", "PRs merged last week", "PRs about XrdAdaptor merged between 11-02 and 11-03"):
            self.assertIsNone(_prs_between_days_answer(question), question)

    @patch("server.api_server._github_export_note", return_value="note")
    @patch("server.api_server.get_driver")
    def test_two_named_days(self, get_driver, _note):
        get_driver.return_value = _driver([{"repo": "cmssw", "n": 52316, "title": "t", "merged": "2026-11-03T10:00:00Z", "release": "CMSSW_20_1", "who": "a"}])
        with patch("server.api_server.datetime", wraps=datetime) as clock:
            clock.now.return_value = datetime(2026, 11, 5, 12)
            answer = _prs_between_days_answer("which PRs came in between 11-02 and 11-03?")
        self.assertIn("2026-11-02 to 2026-11-03", answer)
        self.assertIn("#52316", answer)

    @patch("server.api_server._github_export_note", return_value="note")
    @patch("server.api_server.get_driver")
    def test_latest_prs(self, get_driver, _note):
        get_driver.return_value = _driver([{"repo": "cmssw", "n": 1, "title": "t", "merged": "2026-11-05T01:00:00Z", "who": "a"}], [])
        self.assertIn("Most recently merged", _recent_prs_answer("latest PRs"))
        self.assertIsNone(_recent_prs_answer("latest PRs about tracking"))

    @patch("server.api_server._github_export_note", return_value="note")
    @patch("server.api_server.get_driver")
    def test_most_commented(self, get_driver, _note):
        get_driver.return_value = _driver([{"number": 40437, "title": "Failures", "n": 186}])
        self.assertIn("186 comment(s)", _most_commented_answer("show me the issue with the most comments"))
        self.assertIsNone(_most_commented_answer("comments of issue 51970"))

    @patch("server.api_server._github_export_note", return_value="note")
    @patch("server.api_server.get_driver")
    def test_items_mentioning_keeps_state_and_kind(self, get_driver, _note):
        get_driver.return_value = _driver([{"n": 52040, "title": "t", "in_title": False, "state": None, "pr_state": "open", "merged": None}])
        answer = _items_mentioning_answer("show open PRs touching generators")
        self.assertIn("1 open PR(s)", answer)
        self.assertIn("\"generators\"", answer)

    def test_plain_open_issues_about_keep_their_answer(self):
        for question in ("open issues about ASAN", "any open issues about LST?", "issues about Herwig"):
            self.assertIsNone(_items_mentioning_answer(question), question)


class HandlersStayInTheirLaneTests(unittest.TestCase):
    """Each of these was taken by a new handler until the replay of all earlier turns caught it."""

    def test_clang_detail(self):
        for question in ("what is the difference between a build warning and a clang warning?", "does clang have more failures than primary?"):
            self.assertIsNone(_clang_detail_answer(question), question)

    def test_ibs_for_arch(self):
        self.assertIsNone(_ibs_for_arch_answer("which IB was the last one where everything was green on 20_1 el9_amd64_gcc14?"))

    def test_flavours_by_time(self):
        self.assertIsNone(_flavours_by_time_answer("which flavour is best today?"))

    def test_log_missing_days(self):
        self.assertIsNone(_log_missing_days_answer("which test logs are missing for the latest IB?"))

    def test_log_filter_needs_a_filter(self):
        self.assertIsNone(_log_problem_filter_answer("hidden problems in the latest IB"))

    def test_failure_sets_is_not_a_ranking(self):
        self.assertIsNone(_failure_sets_answer("what is the most common unit test failure across all architectures?"))

    def test_arch_on_day(self):
        self.assertIsNone(_arch_on_day_answer("did 537.0 pass on el9 yesterday?"))

    def test_clang_warnings_are_not_the_clang_sub_ib(self):
        self.assertIsNone(_with_named_flavour_tag("new clang warnings in CMSSW_20_1_X_2026-08-12-2300"))


class RewriteOrderTests(unittest.TestCase):
    def test_not_rewritten_away(self):
        for question in ("which files in DataFormats/Common have warnings?", "is the el10 build error new?", "what's the el10 build error?"):
            self.assertIsNone(_shift_question_rewrite(question), question)

    def test_last_known_good_reaches_the_build_answer(self):
        with patch("server.api_server.get_driver") as get_driver, patch("server.api_server._latest_ib_records", return_value=[{"tag": "CMSSW_20_1_X_2026-11-04-1100"}]):
            get_driver.return_value = _driver([{"tag": "CMSSW_20_1_X_2026-11-03-2300", "arch": "el10_amd64_gcc15", "build": "error"}])
            self.assertIn("Last good build", _build_error_answer("last known good for el10"))


class SameBuildTests(unittest.TestCase):
    def test_same_tag_twice(self):
        tag = "CMSSW_20_1_X_2026-11-03-2300"
        self.assertIn("same build twice", _same_tag_compare_answer(f"compare {tag} with {tag}"))
        self.assertIn("same build twice", _same_tag_compare_answer(f"compare {tag} to itself"))
        self.assertIsNone(_same_tag_compare_answer(f"compare {tag} with CMSSW_20_1_X_2026-11-03-1100"))


class MetaFollowUpTests(unittest.TestCase):
    def test_what_did_i_ask(self):
        _session_questions["meta-1"] = ["latest IB", "why"]
        self.assertIn("latest IB", _meta_conversation_answer("what did I ask first?", "meta-1"))
        self.assertIn("why", _meta_conversation_answer("what did I ask?", "meta-1"))
        self.assertIn("haven't asked", _meta_conversation_answer("what did I ask first?", "meta-empty"))

    def test_no_paging_is_said(self):
        self.assertIn("nothing to continue", _meta_conversation_answer("more results", "meta-empty"))

    def test_nothing_to_retry(self):
        self.assertIn("nothing to try again", _meta_conversation_answer("try again", "meta-empty"))


DIGEST = """## Shift summary: A -> B

**In short:** newly failing relval workflows: **2**

### `el10_amd64_gcc15`

#### Results not imported yet
- Build: not here.

### `el9_amd64_gcc14`

#### Newly failing RelVal workflows
| Workflow |
| --- |
| `537.0` |

#### Newly failing Unit Tests
| Name |
| --- |
| `Foo/Bar` |
"""


class DigestFilterTests(unittest.TestCase):
    def test_section(self):
        answer = _digest_filter_answer(DIGEST, "only relvals")
        self.assertIn("537.0", answer)
        self.assertNotIn("Foo/Bar", answer)
        self.assertNotIn("In short", answer)

    def test_architecture(self):
        answer = _digest_filter_answer(DIGEST, "only el10")
        self.assertIn("not here", answer)
        self.assertNotIn("537.0", answer)

    def test_says_when_nothing_matches(self):
        self.assertIn("Nothing about", _digest_filter_answer(DIGEST, "just the clang warnings"))

    def test_other_questions_are_left_alone(self):
        self.assertIsNone(_digest_filter_answer(DIGEST, "what about unit tests"))
        self.assertIsNone(_digest_filter_answer("not a digest", "only relvals"))


COMPARISON = """## IB comparison: el9_amd64_gcc14

### Primary
| Metric | A | B | Delta |
| --- | ---: | ---: | ---: |
| Failed RelVals | 1 | 2 | +1 |

**Failing workflows:**
| Workflow | Name | Errors | Exit code | Status | Also failing in (A) | Also failing in (B) |
| --- | --- | ---: | ---: | --- | --- | --- |
| `537.0` | DY | 0 | 16640 (OtherCMS) | Newly failing | ASAN (`x`) | CLANG (`y`) |

### `ASAN`
`B` has no imported record for this sub-IB on `el9_amd64_gcc14`.

### `CLANG`
| Metric | A | B | Delta |
| --- | ---: | ---: | ---: |
| Failed RelVals | 1 | 2 | +1 |

**Failing workflows:**
| Workflow | Name | Errors | Exit code | Status | Also failing in (A) | Also failing in (B) |
| --- | --- | ---: | ---: | --- | --- | --- |
| `537.0` | DY | 0 | 16640 (OtherCMS) | Newly failing | ASAN (`x`) | |
| `539.0` | DY2 | 0 | 31744 (TimeOut) | Still failing (already failing in `A`) | | |
"""


class ComparisonCondenseTests(unittest.TestCase):
    def test_keeps_every_change_and_drops_the_wide_columns(self):
        answer = _condense_one_arch_comparison(COMPARISON)
        self.assertNotIn("Also failing in", answer)
        self.assertIn("| `537.0` | DY | 0 | 16640 (OtherCMS) | Newly failing |", answer)
        self.assertIn("failed RelVals 1 -> 2. Newly failing: `537.0`; Still failing: `539.0`.", answer)
        self.assertIn("`ASAN`", answer)


if __name__ == "__main__":
    unittest.main()
