"""Live end-to-end validation for POST /api/chat against a running server + real graph.

Unlike tests/test_chat_routing.py (mocked Neo4j, fast, run in CI), this hits an
actual server so it exercises real data shapes and catches the class of bug
mocks can hide -- a capability the planner correctly recognizes but whose
handler chain silently returns no evidence (see the relval-workflow-listing
fix in server/api_server.py's "relval" branch of _execute_planned_capability).

Usage:
    venv/bin/python scripts/validate_bot.py [--base-url http://localhost:8001]
    venv/bin/python scripts/validate_bot.py --with-capability-fallback  # server must have
                                                                          # ENABLE_CAPABILITY_FALLBACK_TO_LLM=true
    venv/bin/python scripts/validate_bot.py --with-answer-verification  # server must have
                                                                          # ENABLE_DETERMINISTIC_ANSWER_VERIFICATION=true

Requires the API server to be running (see README.md) against real imported
data. Never invents data; every scenario below is templated against the
IB tag the server itself reports as "latest" at run time, so this stays valid
as the graph is refreshed.
"""
from __future__ import annotations

import os
import argparse
import re
import sys
from dataclasses import dataclass, field

import httpx

# Any of these substrings in an answer marks the turn as a hard failure --
# the deterministic path either gave up on data it should have, or leaked an
# internal error to the user instead of a graceful message.
FAILURE_MARKERS = (
    "No imported local evidence is available",
    "traceback",
    "Traceback",
    "internal server error",
    "The approved local evidence could not be synthesized",
)


@dataclass
class Turn:
    question: str
    # Substrings that MUST appear in the answer for this turn to pass.
    # Matched case-insensitively. Empty means "just don't fail" (still
    # checked against FAILURE_MARKERS).
    expect_contains: list[str] = field(default_factory=list)
    # Alternative substrings that also count as a pass even when
    # expect_contains isn't satisfied -- for a legitimately correct "I don't
    # have that data (yet)" answer that isn't the routing/coverage bug this
    # script hunts for. Concretely: relval scenarios are templated against
    # whatever tag the server currently reports as "latest", but RelVals
    # finish importing later than the build itself, so the newest tag can
    # briefly have no RelVal records at all -- a real, correct, graceful
    # answer, not a dead end. Distinguishing this from FAILURE_MARKERS (a
    # genuine "no evidence" routing dead-end) keeps the script from crying
    # wolf on ordinary import lag while still catching real regressions.
    alt_ok_contains: list[str] = field(default_factory=list)
    # If True, a FAILURE_MARKERS hit or a missing expect_contains is reported
    # as a KNOWN GAP (not a fresh regression) -- used for capabilities already
    # flagged as missing session-tag fallback (clang/build), so the report
    # distinguishes "known, already-flagged" from "new" breakage.
    known_gap: str | None = None
    # Substrings that must NOT appear in the answer. Unlike FAILURE_MARKERS
    # (global, catches a routing dead end / crash), this is scenario-specific:
    # it catches a *wrong-but-plausible* answer that never trips a global
    # marker at all -- e.g. a negated question ("workflows that are NOT
    # failing") getting the same answer as its un-negated opposite. That kind
    # of bug doesn't look broken to a marker scan; it looks like a normal
    # answer, just the wrong one.
    must_not_contain: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    turns: list[Turn]


_NO_RELVAL_DATA_YET = ["No imported RelVal records were found", "No failing RelVals were found", "No RelVal results are imported for"]
# A legitimate, intentional outcome (not a routing dead end): the "latest"
# tag genuinely has >1 architecture simultaneously flagged
# is_relval_default, which _relval_scope_or_choice correctly asks about
# rather than guessing -- see docs/DEVELOPER.md's RelVal-architecture-visibility fix.
# Distinct from _NO_RELVAL_DATA_YET (no data at all) -- this is "data exists
# but which architecture is ambiguous."
_RELVAL_ARCH_CHOICE_PENDING = ["are architecture-specific", "Choose a listed architecture"]
_RELVAL_SCOPE_ALT_OK = _NO_RELVAL_DATA_YET + _RELVAL_ARCH_CHOICE_PENDING


def build_scenarios(tag: str, arch: str) -> list[Scenario]:
    return [
        # Found live 2026-10-15: "where is 38551.85 failing?" got the latest IB's failure
        # overview, a bare "38551.85" the "couldn't match that" menu. Workflow 1.0 runs in
        # every release series, so this holds whatever the graph's current data is.
        Scenario("where is a workflow failing, no tag", [
            Turn("where is workflow 1.0 failing?", expect_contains=["Workflow `1.0`", "Runs:"]),
        ]),
        Scenario("bare workflow ID", [
            Turn("1.0", expect_contains=["Workflow `1.0`", "Runs:"]),
        ]),
        Scenario("relval failure count", [
            Turn(
                f"how many relvals are failing in {tag}",
                expect_contains=["failing"], alt_ok_contains=_RELVAL_SCOPE_ALT_OK,
            ),
        ]),
        Scenario("relval workflow list, tag repeated same turn", [
            Turn(
                f"can you give the workflow that are failing in relvals {tag}",
                expect_contains=["Workflow ID"], alt_ok_contains=_RELVAL_SCOPE_ALT_OK,
            ),
        ]),
        Scenario("relval workflow list, tag carried from earlier turn", [
            Turn(
                f"how many relvals are failing in {tag}",
                expect_contains=["failing"], alt_ok_contains=_RELVAL_SCOPE_ALT_OK,
            ),
            Turn(
                "can you give the workflow that are failing",
                expect_contains=["Workflow ID"], alt_ok_contains=_RELVAL_SCOPE_ALT_OK,
            ),
            Turn("workflow names", expect_contains=[]),
        ]),
        Scenario("unit tests, tag carried from earlier turn", [
            Turn(f"summary for {tag}", expect_contains=[]),
            Turn("what about unit tests", expect_contains=["unit"]),
        ]),
        Scenario("build warnings, tag carried from earlier turn", [
            Turn(f"build warnings for {tag}", expect_contains=[]),
            Turn("what about build warnings", expect_contains=[]),
        ]),
        Scenario("clang warnings, tag repeated same turn", [
            Turn(f"clang warnings for {tag}", expect_contains=["Clang warning"]),
        ]),
        Scenario("clang warnings, tag carried from earlier turn", [
            Turn(f"clang warnings for {tag}", expect_contains=["Clang warning"]),
            Turn("show me the clang warnings again", expect_contains=["Clang warning"]),
        ]),
        Scenario("latest IB phrasing variants", [
            Turn("what is the latest IB", expect_contains=["CMSSW"]),
            Turn("compare latest branches", expect_contains=[]),
        ]),
        # Used to expect the "Do you want build warnings, failed RelVals, ...?"
        # clarification -- replaced 2026-10-06 by a direct every-category answer
        # (126 of 128 clarifications in a replay of real sessions were this one,
        # asked back for a question that already said what it wanted).
        Scenario("category-less failure request answers every category, not a question back", [
            Turn(f"failures for {tag}", expect_contains=["What's failing in", "RelVals", "Unit tests", "Build"]),
            Turn("unit tests", expect_contains=["unit-test results"]),
        ]),
        Scenario("workflow follow-ups keep the workflow", [
            Turn("why is workflow 512.0 failing on aarch64", expect_contains=["512.0", "aarch64"]),
            Turn("since when is it failing", expect_contains=["512.0"]),
            Turn("on which architectures", expect_contains=["512.0"]),
        ]),
        Scenario("scope-only follow-up repeats the previous question", [
            Turn(f"unit tests for {tag}", expect_contains=["unit-test results"]),
            # the latest IB often has no aarch64 build yet (10-15-0900 had only el9_amd64_gcc14):
            # saying so is the correct answer, not a routing dead end
            Turn("and on aarch64 only", expect_contains=["unit-test results", "aarch64"],
                 alt_ok_contains=["has no imported `aarch64` build"]),
        ]),
        Scenario("exit code meaning in all three forms", [
            Turn("what does exit code 65 mean", expect_contains=["OtherCMS"]),
            Turn("what does exit code 8001 mean", expect_contains=["OtherCMS"]),
            Turn("what does exit code 35584 mean", expect_contains=["SIGSEGV"]),
        ]),
        Scenario("small talk and write requests get a direct reply", [
            Turn("hi", expect_contains=["What would you like to check"]),
            Turn("please rerun workflow 512.0", expect_contains=["read-only"]),
        ]),
        Scenario("garbage / non-question input never crashes", [
            Turn("erfe", expect_contains=[]),
            Turn("ss", expect_contains=[]),
            Turn("", expect_contains=[]),
        ]),
        Scenario("architecture-qualified relval follow-up", [
            Turn(
                f"how many relvals are failing in {tag}",
                expect_contains=["failing"], alt_ok_contains=_RELVAL_SCOPE_ALT_OK,
            ),
            Turn(arch, expect_contains=[]),
        ]),
        Scenario("recurrence / known-issue phrasing", [
            Turn(f"is this a known issue in {tag}", expect_contains=[]),
        ]),
        Scenario("PR evidence for failing relvals", [
            Turn(f"failed relvals for {tag}", expect_contains=[]),
            Turn("any PRs related to these failures", expect_contains=[]),
        ]),
        Scenario("general-knowledge question is refused, never answered from the LLM's own knowledge", [
            # Key invariant check: the LLM must never answer from its own
            # knowledge, only from tool evidence. Confirmed live 2026-09-22 --
            # the system correctly refuses rather than naming an actual
            # person, since "president of Pakistan" has no local graph
            # evidence to answer from. Flag-independent (never reaches either
            # DETERMINISTIC branch gate), so this belongs in the base set.
            Turn("who is president of pakistan", expect_contains=["couldn't match"]),
        ]),
        Scenario("PR + open-issue evidence combined, not short-circuited", [
            # Regression for the "pr_evidence" or-chain bug: a question asking
            # for both PRs and open Issues used to only ever get PR evidence
            # back, since _open_issue_answer was never in the capability's
            # handler chain at all. "AssociationMap" is a stable real term
            # (both PR #51649 and several open Issues mention it as of
            # 2026-08-25) rather than one templated on the latest tag, since
            # this is a text search independent of IB identity.
            Turn(
                "any information about ticl::AssociationMap any PR/issue open?",
                expect_contains=["PR evidence matching", "Open CMSSW Issues matching"],
            ),
        ]),
        # A real report (2026-10-12) of tests that pass although the cmsRun job
        # inside fails: only the imported raw test logs (LogProblem) can see that.
        # Fixed facts in the graph, so not templated on the latest tag.
        Scenario("pasted exception from a test reported as passed", [
            Turn(
                "While checking the unit tests in CMSSW_20_1_ASAN_X_2026-10-05-2300, SubmitPVrbr is reported as "
                "successful but cmsRun stops with:\n"
                "An exception of category 'DictionaryNotFound' occurred while\n"
                "   [0] Constructing the EventProcessor\n"
                "No data dictionary found for the following classes:\n  DCSRecord\n  L1GlobalTriggerReadoutRecord",
                expect_contains=["Found in the imported IB test logs", "SubmitPVrbr", "reported as PASSED",
                                 "In `CMSSW_20_1_ASAN_X_2026-10-05-2300`: present"],
            ),
        ]),
        Scenario("pasted sanitizer fragment found in the test logs", [
            Turn(
                "ERROR: AddressSanitizer: global-buffer-overflow in HGCalGeometry::getNewCorners",
                expect_contains=["Found in the imported IB test logs", "test_dumpRecoGeometry_Run4"],
            ),
        ]),
        # A build-time exception (edmWriteConfigs) that is not in the test logs: the answer
        # must say so, search by its specific names (not the bare category) and rank the PR
        # mentioning most of them first -- found live 2026-10-13, it used to list unrelated
        # PluginNotFound problems that shared only the category word.
        Scenario("pasted build-time exception absent from the test logs", [
            Turn(
                "----- Begin Fatal Exception 12-Oct-2026 15:45:18 CEST-----------------------\n"
                "An exception of category 'PluginNotFound' occurred while\n"
                "   [0] Running executable \"edmWriteConfigs\"\n"
                "   [1] Processing library /build/muz/del/CMSSW_20_1_X_2026-10-11-0000/tmp/el9_amd64_gcc14/src/"
                "RecoTICL/Interpretation/plugins/RecoTICLInterpretationPlugins/libRecoTICLInterpretationPlugins.so\n"
                "   [2] Writing cfi files using ParameterSetDescriptions for module of base type EDProducer with "
                "plugin name 'TICLCandidateProducer'\n"
                "Exception Message:\n"
                "Unable to find plugin 'TracksterInferenceByPFN' because the category "
                "'PluginDescriptorTracksterInferenceAlgoFactory' has no known plugins\n"
                "----- End Fatal Exception -------------------------------------------------",
                expect_contains=["This exact error is not in the imported IB test logs", "edmWriteConfigs",
                                 "Searched the test logs by:"],
                # (it also expected PR #52138 by the pasted names: that PR was deleted from the graph
                # by the old cmsrepo --reconcile, see project memory "keep 90 days")
            ),
        ]),
        # Searched by the producer in the "[N]" context lines, not only the message: this
        # 'Test' exception has no message at all, only its module.
        Scenario("pasted exception searched by its producer/module", [
            Turn(
                "An exception of category 'Test' occurred while\n"
                "   [1] Running path 'p1'\n"
                "   [2] Calling method for module BusyWaitIntProducer/'busy1'",
                expect_contains=["imported IB test logs", "BusyWaitIntProducer", "Raised in module(s)",
                                 "producer `Test|BusyWaitIntProducer`"],
            ),
        ]),
        # Found live 2026-11-11 (owner): "open" in the question let the unit-test rule take the test name, GitHub was never
        # searched, and the follow-up got "Choose a scope". Issue #51089 is open and has "testVolumes crashes" in its title.
        # Owner's demo chain, found 2026-11-12: the AI path put an unrelated PR first and followed up on it (80-95 s a turn). Holds while
        # 36434.0 keeps failing since 11-06-1100 with the BTL plugin error; check the window if it stops.
        Scenario("failure -> which PR caused it -> what code did it touch", [
            Turn("which PR caused workflow 36434.0 to fail?", expect_contains=["Most likely: PR #51861", "BTLBaseRecHitSoAProducer"]),
            Turn("what code did that PR touch?", expect_contains=["PR #51861", "Files matching the error"]),
        ]),
        Scenario("PR/Issue ask about a named unit test, then 'related to this'", [
            Turn('is there any issue or PR open for "testVolumes crashes"', expect_contains=["#51089", "Open: "]),
            Turn("did you find any issue or PR open related to this", expect_contains=["#51089"], must_not_contain=["Choose a scope"]),
        ]),
    ]


def build_capability_fallback_scenarios(tag: str) -> list[Scenario]:
    """Only meaningful -- and only expected to pass -- against a server
    running with ENABLE_CAPABILITY_FALLBACK_TO_LLM=true. Both phrasings below
    were confirmed live (2026-09-22) to be real capability-matched-but-no-
    handler dead ends: RequestPlanner correctly picks "relval"/"unittest",
    but no helper in that capability's chain recognizes the exact phrasing,
    so _execute_planned_capability returns None. With the flag off this hits
    the flat "No imported local evidence" message (a FAILURE_MARKERS hit) by
    design -- that's why these are opt-in via --with-capability-fallback
    rather than part of the default scenario list every plain server run is
    expected to pass.

    expect_contains is deliberately empty for both turns: the point isn't
    that a specific answer shape comes back (a real evidence-grounded answer
    and a fallback_clarification action menu are both legitimate non-dead-end
    outcomes -- see server/api_server.py's _run_llm_investigation), only that
    the flat dead-end message and other FAILURE_MARKERS never do.
    """
    return [
        Scenario("capability dead end rescued: relval recurrence, odd phrasing", [
            Turn("tell me the recurrence pattern for workflow 136.899", expect_contains=[]),
        ]),
        Scenario("capability dead end rescued: unittest package-fix phrasing", [
            Turn(f"what packages need fixing in {tag}", expect_contains=[]),
        ]),
    ]


def build_answer_verification_scenarios(tag: str) -> list[Scenario]:
    """Only meaningful -- and only expected to pass -- against a server
    running with ENABLE_DETERMINISTIC_ANSWER_VERIFICATION=true. Confirmed
    live 2026-09-22: deterministic routing only tests keyword/substring
    presence, so "workflows that are NOT failing" and "workflows that are
    failing" both trigger the same "failing" keyword in the relval handler
    and get the IDENTICAL answer -- wrong for the negated question, since it
    actually wants the passing workflows. With the flag off this doesn't
    trip any FAILURE_MARKERS at all (it looks like a normal, non-crashing
    answer, just the wrong one) -- that's why must_not_contain exists and
    why this scenario is opt-in via --with-answer-verification rather than
    part of the default scenario list.
    """
    return [
        Scenario("answer verification catches negation mismatch", [
            Turn(
                f"give me the workflows that are NOT failing in relvals {tag}",
                expect_contains=[],
                must_not_contain=["No failing RelVals were found"],
            ),
        ]),
    ]


def run(base_url: str, with_capability_fallback: bool = False, with_answer_verification: bool = False) -> int:
    client = httpx.Client(base_url=base_url, timeout=float(os.environ.get("VALIDATE_TIMEOUT", "30")))

    health = client.get("/api/health")
    health.raise_for_status()
    print(f"Server reachable: {health.json()}")

    latest = client.post("/api/chat", json={"question": "what is the latest IB"})
    latest.raise_for_status()
    latest_answer = latest.json()["answer"]
    tag_match = re.search(r"CMSSW_[A-Za-z0-9_]+_\d{4}-\d{2}-\d{2}-\d{4}", latest_answer)
    if not tag_match:
        print("Could not resolve a live IB tag from 'what is the latest IB' -- aborting.")
        return 1
    tag = tag_match.group(0)
    arch_match = re.search(r"`(el\d+_(?:amd64|aarch64|riscv64)_gcc\d+)`", latest_answer)
    arch = arch_match.group(1) if arch_match else "el9_amd64_gcc14"
    print(f"Using live tag={tag} arch={arch}\n")

    scenarios = build_scenarios(tag, arch)
    if with_capability_fallback:
        print("Including capability-dead-end-fallback scenarios -- server must be "
              "running with ENABLE_CAPABILITY_FALLBACK_TO_LLM=true or these will fail.\n")
        scenarios += build_capability_fallback_scenarios(tag)
    if with_answer_verification:
        print("Including deterministic-answer-verification scenarios -- server must be "
              "running with ENABLE_DETERMINISTIC_ANSWER_VERIFICATION=true or these will fail.\n")
        scenarios += build_answer_verification_scenarios(tag)
    failures: list[str] = []
    known_gaps: list[str] = []
    passed = 0
    total_turns = 0

    for scenario in scenarios:
        session_id = f"validate-bot-{scenario.name}".replace(" ", "-")[:60]
        print(f"=== {scenario.name} ===")
        for turn in scenario.turns:
            total_turns += 1
            resp = client.post("/api/chat", json={"question": turn.question, "session_id": session_id})
            if resp.status_code != 200:
                failures.append(f"[{scenario.name}] {turn.question!r} -> HTTP {resp.status_code}: {resp.text[:200]}")
                print(f"  FAIL (HTTP {resp.status_code}): {turn.question!r}")
                continue
            answer = resp.json().get("answer", "")
            answer_lower = answer.lower()
            hit_marker = next((m for m in FAILURE_MARKERS if m.lower() in answer_lower), None)
            forbidden_hit = next((n for n in turn.must_not_contain if n.lower() in answer_lower), None)
            missing = [needle for needle in turn.expect_contains if needle.lower() not in answer_lower]
            alt_hit = any(needle.lower() in answer_lower for needle in turn.alt_ok_contains)
            ok = hit_marker is None and forbidden_hit is None and (not missing or alt_hit)
            if ok:
                passed += 1
                label = "ok (no data yet)" if missing and alt_hit else "ok  "
                print(f"  {label} {turn.question!r}")
            else:
                if forbidden_hit:
                    reason = f"forbidden={forbidden_hit!r}"
                elif hit_marker:
                    reason = f"marker={hit_marker!r}"
                else:
                    reason = f"missing={missing!r}"
                detail = f"[{scenario.name}] {turn.question!r} -> {reason}\n    answer: {answer[:200]!r}"
                if turn.known_gap:
                    known_gaps.append(f"{detail}\n    known gap: {turn.known_gap}")
                    print(f"  GAP  {turn.question!r} ({reason}) -- {turn.known_gap}")
                else:
                    failures.append(detail)
                    print(f"  FAIL {turn.question!r} ({reason})")
        print()

    print("=" * 70)
    print(f"{passed}/{total_turns} turns passed")
    if known_gaps:
        print(f"\n{len(known_gaps)} known gap(s) (already flagged, not new regressions):")
        for item in known_gaps:
            print(f"- {item}\n")
    if failures:
        print(f"\n{len(failures)} UNEXPECTED FAILURE(S):")
        for item in failures:
            print(f"- {item}\n")
        return 1
    print("\nNo unexpected failures.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument(
        "--with-capability-fallback", action="store_true",
        help="Also run capability-dead-end-fallback scenarios. Only pass this "
             "when the server is running with ENABLE_CAPABILITY_FALLBACK_TO_LLM=true "
             "-- against a plain server these are expected to fail.",
    )
    parser.add_argument(
        "--with-answer-verification", action="store_true",
        help="Also run deterministic-answer-verification scenarios. Only pass this "
             "when the server is running with ENABLE_DETERMINISTIC_ANSWER_VERIFICATION=true "
             "-- against a plain server these are expected to fail.",
    )
    args = parser.parse_args()
    sys.exit(run(
        args.base_url,
        with_capability_fallback=args.with_capability_fallback,
        with_answer_verification=args.with_answer_verification,
    ))
