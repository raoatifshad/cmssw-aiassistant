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

Requires the API server to be running (see README.md) against real imported
data. Never invents data; every scenario below is templated against the
IB tag the server itself reports as "latest" at run time, so this stays valid
as the graph is refreshed.
"""
from __future__ import annotations

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


@dataclass
class Scenario:
    name: str
    turns: list[Turn]


_NO_RELVAL_DATA_YET = ["No imported RelVal records were found", "No failing RelVals were found"]
# A legitimate, intentional outcome (not a routing dead end): the "latest"
# tag genuinely has >1 architecture simultaneously flagged
# is_relval_default, which _relval_scope_or_choice correctly asks about
# rather than guessing -- see CLAUDE.md's RelVal-architecture-visibility fix.
# Distinct from _NO_RELVAL_DATA_YET (no data at all) -- this is "data exists
# but which architecture is ambiguous."
_RELVAL_ARCH_CHOICE_PENDING = ["are architecture-specific", "Choose a listed architecture"]
_RELVAL_SCOPE_ALT_OK = _NO_RELVAL_DATA_YET + _RELVAL_ARCH_CHOICE_PENDING


def build_scenarios(tag: str, arch: str) -> list[Scenario]:
    return [
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
        Scenario("ambiguous request triggers clarification, not a guess", [
            Turn(f"failures for {tag}", expect_contains=["Do you want"]),
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


def run(base_url: str, with_capability_fallback: bool = False) -> int:
    client = httpx.Client(base_url=base_url, timeout=30.0)

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
            missing = [needle for needle in turn.expect_contains if needle.lower() not in answer_lower]
            alt_hit = any(needle.lower() in answer_lower for needle in turn.alt_ok_contains)
            ok = hit_marker is None and (not missing or alt_hit)
            if ok:
                passed += 1
                label = "ok (no data yet)" if missing and alt_hit else "ok  "
                print(f"  {label} {turn.question!r}")
            else:
                reason = f"marker={hit_marker!r}" if hit_marker else f"missing={missing!r}"
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
    args = parser.parse_args()
    sys.exit(run(args.base_url, with_capability_fallback=args.with_capability_fallback))
