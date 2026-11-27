"""Sample questions for people new to the chat, and "ask next" suggestions under an answer -- one list, used by the "help" reply,
GET /api/sample-questions (the chat widget's empty state) and the suggestions of each answer.

Owner, 2026-11-13: users should see which questions they can ask (no Clang questions: its data is old, the owner knows). The demo chain (a failure -> is it reported? -> which PR caused it?
-> what code did it touch?) is shown as a chain, since that is what people should learn to do. The failure example is a workflow failing in
the newest imported build (looked up, cached), so the example always has an answer; without one, a generic question is used.
"""
from __future__ import annotations

import re
import time

from graph.tools import _ib_sort_key, get_driver

_CACHE: dict = {"at": 0.0, "workflow": None}
_TTL = 600


def _failing_workflow() -> str | None:
    """A workflow failing in the newest primary IB (not a known error), preferring one failing on el9_amd64_gcc14; cached 10 minutes."""
    if time.time() - _CACHE["at"] < _TTL:
        return _CACHE["workflow"]
    workflow = None
    try:
        driver = get_driver()
        try:
            with driver.session() as session:
                tags = [r["tag"] for r in session.run(
                    "MATCH (ib:IB)-[:RAN_RELVAL]->() WHERE ib.variant IS NULL WITH DISTINCT ib.tag AS tag RETURN tag ORDER BY right(tag, 15) DESC LIMIT 40")]
                if tags:
                    newest = max(tags, key=_ib_sort_key)
                    rows = [dict(r) for r in session.run(
                        "MATCH (ib:IB {tag:$tag})-[:RAN_RELVAL]->(r:RelVal) WHERE ib.variant IS NULL AND r.status = 'FAILED' "
                        "AND NOT coalesce(r.known_error, false) RETURN r.workflow_id AS wf, ib.arch AS arch", tag=newest)]
                    rows.sort(key=lambda r: (r["arch"] != "el9_amd64_gcc14", r["wf"] or ""))
                    workflow = rows[0]["wf"] if rows else None
        finally:
            driver.close()
    except Exception as exc:
        print(f"WARNING: sample-question workflow lookup failed: {exc}")
    _CACHE.update(at=time.time(), workflow=workflow)
    return workflow


def sample_groups() -> list[dict]:
    """Groups of {label, question}; a group's 'chain' flag means its questions are asked one after the other in one conversation."""
    wf = _failing_workflow()
    chain_start = f"where is RelVal {wf} failing?" if wf else "what is failing in the latest IB?"
    return [
        {"title": "Start here", "items": [
            {"label": "Latest IB", "question": "what is the latest IB?"},
            {"label": "What is failing?", "question": "what is failing in the latest IB?"},
            {"label": "Shift summary", "question": "give me a shift summary"},
            {"label": "Compare with the previous IB", "question": "compare the latest IB with the previous one"},
        ]},
        {"title": "Follow one failure, step by step", "chain": True, "items": [
            {"label": f"Where is {wf} failing?" if wf else "What is failing?", "question": chain_start},
            {"label": "Is it reported?", "question": "is it reported on GitHub?"},
            {"label": "Which PR caused it?", "question": "which PR could have caused it?"},
            {"label": "What code did it touch?", "question": "what code did that PR touch?"},
        ]},
        {"title": "Tests and logs", "items": [
            {"label": "Failing unit tests", "question": "which unit tests are failing in the latest IB?"},
            {"label": "Failing RelVals on aarch64", "question": "which RelVals fail on el9_aarch64_gcc14 in the latest IB?"},
            {"label": "New problems in the test logs", "question": "any new problems in the unit test logs of the latest IB?"},
        ]},
        {"title": "GitHub and code", "items": [
            {"label": "Open issues about an error", "question": "any open issue about PluginNotFound?"},
            {"label": "PRs that touched a package", "question": "which PRs touched FWCore/Framework in the last month?"},
            {"label": "What an exit code means", "question": "what does exit code 8001 mean?"},
        ]},
    ]


PASTE_HINT = ("Or paste an error: an exception from a RelVal log, or a line from a unit-test log. "
              "The chat finds where it was seen, and which GitHub Issues and PRs mention it.")


def help_markdown() -> str:
    lines = ["I answer from imported CMSSW Integration Build data (RelVals, unit tests, builds, IB test logs, GitHub PRs and Issues), "
             "read-only, never from the web. Questions you can ask:"]
    for group in sample_groups():
        lines.append(f"\n**{group['title']}**" + (" (ask them one after the other)" if group.get("chain") else ""))
        if group.get("chain"):
            lines += [f"{n}. \"{item['question']}\"" for n, item in enumerate(group["items"], 1)]
        else:
            lines += [f"- \"{item['question']}\"" for item in group["items"]]
    lines.append(f"\n{PASTE_HINT}")
    lines.append("\nName an IB tag, an architecture (e.g. `el9_aarch64_gcc14`), a workflow number or a package to be precise; "
                 "follow-ups like \"since when?\" or \"which PR caused it?\" use what the previous answer was about.")
    return "\n".join(lines)


def start_suggestions() -> list[dict]:
    groups = sample_groups()
    return [groups[0]["items"][0], groups[0]["items"][1], groups[1]["items"][0], groups[0]["items"][2]]


_REPORTED = {"label": "Is it reported?", "question": "is it reported on GitHub?"}
_CAUSE = {"label": "Which PR caused it?", "question": "which PR could have caused it?"}
_FILES = {"label": "What code did it touch?", "question": "what code did that PR touch?"}
_FAILING = {"label": "What is failing there?", "question": "what is failing in the latest IB?"}
_COMPARE = {"label": "Compare with the previous IB", "question": "compare the latest IB with the previous one"}

# answers about failures that "is it reported?" / "which PR caused it?" can follow up on
_FAILURE_ROUTES = ("workflow_where", "relval_exception_message", "test_log_problem", "capability:relval", "capability:unittest",
                   "failure_overview", "capability:comparison", "package_status", "workflow_why", "named_test")
_SHOWS_FAILURE_RE = re.compile(r"🔴|\bfailing\b|\bfailed\b|\bnewly\s+failing\b|\bwas recorded in\b", re.IGNORECASE)
_SAYS_CLEAN_RE = re.compile(r"\bno failed\b|\bnot failing\b|\b0 failing\b|\bclean\b|\bnothing\b[^.\n]{0,30}\bfail", re.IGNORECASE)


def next_suggestions(route: str, answer: str) -> list[dict]:
    """'Ask next' buttons under an answer: the next step of the failure chain, or where to go from the latest IB. Empty when unsure."""
    route = route or ""
    if route.startswith(("conversational:help", "conversational:greeting")):
        return start_suggestions()
    if route.startswith(("clarification", "error", "conversational")) or not answer:
        return []
    if route.endswith("deterministic:pr_cause"):
        return [_FILES, _REPORTED] if re.search(r"\*\*(?:Most likely|Possible match)", answer) else [_REPORTED]
    if route.endswith("pr_cause_files"):
        return [_REPORTED]
    if route.endswith("deterministic:github_reported"):
        return [_CAUSE]
    if route.endswith("deterministic") and answer.startswith("Newest imported Integration Build"):
        return [_FAILING, _COMPARE]
    if any(key in route for key in _FAILURE_ROUTES):
        head = answer[:1500]
        if _SHOWS_FAILURE_RE.search(head) and not (_SAYS_CLEAN_RE.search(head[:300]) and not re.search(r"🔴", head)):
            return [_REPORTED, _CAUSE]
    return []
