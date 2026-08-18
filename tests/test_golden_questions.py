"""Phase 0 golden-question set, mined from real logged usage.

Two independent sources feed this file, each used only as far as its trust
level actually supports:

- `LOGGED_QUESTIONS` — the 56 distinct real questions found in the graph's
  `ChatPlanDecision` log (as of 2026-08-18). These have no correctness verdict
  attached, so they are NOT asserted against expected answers here — that
  would bake in whatever the router happened to do, bugs included. What they
  *do* verify: the planner must never raise on real-world phrasing, including
  the junk/garbled inputs real users actually typed ("erfe", "ss", "hello").

- The `LatestIbTimestampTieIncident` test below encodes an actual resolved
  incident from the graph's `UserFeedback` log: 11 of the 22 feedback rows
  turned out to be human-confirmed-correct answers to "what is the latest
  IB" under the *old* semantics (rank by newest timestamp across every
  release). That behavior was intentionally replaced this session (rank by
  release series first) after the CMSSDT dashboard's real definition was
  confirmed to differ — so those 11 rows are not usable as golden fixtures
  (they'd fail the new, correct behavior on purpose). What survives from
  them is the concrete scenario, replayed here against the new behavior.
"""
import unittest
from unittest.mock import patch

from server.api_server import _latest_ib_answer, _request_planner
from server.request_planner import PlanKind, RequestContext


# Verbatim, deduplicated questions from `MATCH (d:ChatPlanDecision) RETURN DISTINCT d.question`.
LOGGED_QUESTIONS = [
    "CMSSW_20_0_X_?? we have CMSSW_20_1_X_2026-08-17-2300 the latest",
    "CMSSW_20_1_X_2026-08-11-2300 summary",
    "CMSSW_20_1_X_2026-08-13-1100",
    "can you give the list of WF in CMSSW_20_1_MULTIARCHSV4_X_2026-08-13-2300",
    "can you tell what this error is about 140.051?",
    "combined report for CMSSW_20_1_EVTGEN3_X_2026-08-16-2300",
    "compare latest IB's",
    "compare latest Ib",
    "compare latest branches",
    "compare latest ibs",
    "compare the latest IB with the previous one",
    "do you have CMSSW_20_1_X_2026-08-17-2300?",
    "do you have info about this iB CMSSW_20_1_X_2026-08-17-2300?",
    "el9_amd64_gcc14",
    "erfe",
    "failed RelVals for CMSSW_20_1_MULTIARCHSV4_X_2026-08-17-2300",
    "give summary for CMSSW_20_1_X_2026-08-11-2300",
    "give summary for CMSSW_20_1_X_2026-08-14-1100",
    "give the summary for CMSSW_20_1_X_2026-08-14-1100",
    "hello",
    "how many WF are failing in CMSSW_20_1_MULTIARCHSV4_X_2026-08-17-2300?",
    "how many relvals are failing in CMSSW_20_1_MULTIARCHSV4_X_2026-08-13-2300",
    "how many relvals failing in CMSSW_20_1_DEVEL_X_2026-08-13-2300",
    "how many relvals failing in CMSSW_20_1_DEVEL_X_2026-08-17-2300?",
    "how many relvals failing in CMSSW_20_1_MULTIARCHSV4_X_2026-08-17-2300",
    "how many relvals failing in CMSSW_20_1_TF_X_2026-08-14-1100",
    "how many relvals failing in the latest IB",
    "how many relvals in CMSSW_20_1_X_2026-08-13-2300",
    "is this the latest IB CMSSW_20_0_X_2026-08-18-1100?",
    "list relval exceptions for CMSSW_20_1_DEVEL_X_2026-08-17-2300 el9_amd64_gcc14",
    "relval results for CMSSW_20_1_DEVEL_X_2026-08-17-2300",
    "ss",
    "summary for CMSSW_20_1_MULTIARCHSV4_X_2026-08-17-2300",
    "summary for CMSSW_20_1_MULTIARCHSV4_X_2026-08-17-2300 all",
    "summary for CMSSW_20_1_X_2026-08-13-1100",
    "summary for CMSSW_20_1_X_2026-08-13-2300",
    "summary for CMSSW_20_1_X_2026-08-13-2300 all",
    "summary for CMSSW_20_1_X_2026-08-14-1100",
    "sx",
    "what WF are in CMSSW_20_1_DEVEL_X_2026-08-13-2300",
    "what is the latest IB",
    "what is the latest ib",
    "what relval exception does workflow 140.051 have in CMSSW_20_1_DEVEL_X_2026-08-17-2300 el9_amd64_gcc14?",
    "what relval exception does workflow 140.051 have in CMSSW_20_1_DEVEL_X_2026-08-17-2300 el9_amd64_gcc14? "
    "any open issue or merged PR causing it?",
    "what relvals are failing in CMSSW_20_1_CLANG_X_2026-08-16-2300",
    "what relvals are in CMSSW_20_1_CLANG_X_2026-08-16-2300",
    "what workflows are failing?",
    "which wf are failing",
    "why did you not showed the latest IB as CMSSW_20_1_X_2026-08-17-2300?",
    "why did you show CMSSW_20_1_TF_X_2026-08-14-1100 as a sub IB?",
    "why did you showed the CMSSW_20_1_TF_X_2026-08-14-1100 sub ib?",
    "yes el8_amd64_gcc14",
    "yes el9_aarch64_gcc14",
    "yes el9_amd64_gcc13",
    "yes el9_amd64_gcc14",
    "you dont have any data for CMSSW_20_1_X_2026-08-13-2300?",
]


class LoggedQuestionsDoNotCrashTheRouter(unittest.TestCase):
    """The planner must never raise, no matter what a real user typed."""

    def test_every_logged_question_produces_a_valid_plan(self):
        failures = []
        for question in LOGGED_QUESTIONS:
            try:
                context = RequestContext.extract(question)
                plan = _request_planner.plan(context)
            except Exception as exc:  # noqa: BLE001 - this is exactly what we're checking for
                failures.append(f"{question!r} raised {exc!r}")
                continue
            if plan.kind not in (PlanKind.DETERMINISTIC, PlanKind.CLARIFICATION, PlanKind.LLM):
                failures.append(f"{question!r} produced an unrecognized plan kind: {plan.kind!r}")
        if failures:
            self.fail("Router crashed or misclassified " + str(len(failures)) + " logged question(s):\n" + "\n".join(failures))


class LatestIbTimestampTieIncident(unittest.TestCase):
    """Replays the real scenario behind 11 confirmed-good UserFeedback rows
    (all "what is the latest IB[?]" variants, timestamp 2026-08-10-1100),
    against the corrected release-series-first ranking."""

    @patch("server.api_server._latest_ib_records")
    def test_newer_release_series_wins_even_when_an_older_series_ties_the_timestamp(self, latest_ib_records):
        # Real historical data: six release series (20_1, 20_1_NONLTO, 20_1_MULTIARCHS,
        # 20_0, 17_0, 16_1) all imported at the same timestamp. The old behavior
        # (human-confirmed correct at the time) listed every one of them as a tie.
        # _latest_ib_records now performs that release-series filtering itself, so
        # a mock reflecting its corrected contract returns only the 20_1 series.
        latest_ib_records.return_value = [{
            "tag": "CMSSW_20_1_X_2026-08-10-1100",
            "release": "CMSSW_20_1_X",
            "architectures": ["el8_amd64_gcc13", "el9_amd64_gcc13"],
            "sub_ib_tags": ["CMSSW_20_1_NONLTO_X_2026-08-10-1100", "CMSSW_20_1_MULTIARCHS_X_2026-08-10-1100"],
            "build_timestamp": "2026-08-10-1100",
        }]

        answer = _latest_ib_answer("what is the latest IB")

        self.assertIn("CMSSW_20_1_X_2026-08-10-1100", answer)
        self.assertNotIn("CMSSW_20_0_X_2026-08-10-1100", answer)
        self.assertNotIn("CMSSW_17_0_X_2026-08-10-1100", answer)
        self.assertNotIn("CMSSW_16_1_X_2026-08-10-1100", answer)


if __name__ == "__main__":
    unittest.main()
