"""Is there a PR or Issue about <a named unit test, or quoted text>? -- answered by the rules, first time.

Found live 2026-11-11 (owner's session on :8002):
  1. 'is there any issue or PR open for "testVolumes crashes"' -> the unit-test history only. The word "open" made
     `_reported_answer` step aside, and `_named_test_answer` took the test name; GitHub was never searched although
     open Issue #51089 has "testVolumes crashes" in its title.
  2. 'did you find any issue or PR open related to this' -> "Choose a scope": "this" was never resolved.
  3. 'can you find a issue or pull request related to "testVolumes crashes"' -> #51089 found.
So: an explicit PR/Issue ask about quoted text or a real unit-test name searches GitHub first (open items named in
the lead line when "open" is asked), the test's log history follows as a short second part, and "this" in the next
message keeps that subject. Text matches only, never a cause or a fix (the section's own caveat says so).
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

_GITHUB_ASK_RE = re.compile(r"\b(?:issues?|prs?|pull\s+requests?|tickets?|bug\s+reports?|reported|github)\b", re.IGNORECASE)
_THIS_RE = re.compile(r"\b(?:this|it|that|this\s+one|the\s+same)\b", re.IGNORECASE)
_OPEN_RE = re.compile(r"\bopen\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[\"“”`']([^\"“”`']{4,120})[\"“”`']")
_TEST_HEAD_RE = re.compile(r"Unit test `([^`]+)`")
_SUBJECT_ROUTES = ("deterministic:named_test", "deterministic:package_status", "deterministic:github_named_ask")

# session -> {"names": [search terms], "test": unit-test name or None, "label": what the user called it}
_subjects: dict[str, dict] = {}


def _api():
    from server import api_server  # late: api_server imports this module
    return api_server


def _unit_test_named(question: str) -> tuple[str, str] | None:
    """(short test name, the short history paragraph) when the question names a unit test that the imported logs know."""
    s = _api()
    match = s._NAMED_TEST_RE.search(s.IB_TAG_RE.sub(" ", question))
    if not match:
        return None
    name = match.group(1)
    try:
        history = s._named_test_answer(f"is {name} failing?")
    except Exception as exc:
        print(f"WARNING: unit test history for the GitHub ask failed: {exc}")
        return None
    return (name, history.strip().split("\n\n", 1)[0]) if history else None


def _subject_in_question(question: str) -> dict | None:
    quoted = [q.strip() for q in _QUOTED_RE.findall(question) if len(q.strip()) >= 4]
    test = _unit_test_named(question)
    names = quoted + ([test[0]] if test and test[0] not in quoted else [])
    if not names:
        return None
    return {"names": names[:3], "test": test[0] if test else None, "history": test[1] if test else None,
            "label": quoted[0] if quoted else test[0]}


def _remembered_subject(sid: str) -> dict | None:
    """The subject of the previous rules answer, when that answer was about a test, a package or this route."""
    s = _api()
    last = s._turn_memory.get(sid) or {}
    route = last.get("route") or ""
    if last.get("superseded") or not (route in _SUBJECT_ROUTES or "unittest" in route):
        return None
    if last.get("route") == "deterministic:github_named_ask" and sid in _subjects:
        return _subjects[sid]
    head = _TEST_HEAD_RE.search(last.get("answer") or "")
    if head:
        short = head.group(1).rsplit("/", 1)[-1]
        return {"names": [short], "test": short, "history": None, "label": head.group(1)}
    # "which unit tests failed in the latest IB?" then "any issue or PR related to this?": every package listed (found 2026-11-11: the AI path, 69 s)
    packages = list(dict.fromkeys(s._PACKAGE_PATH_RE.findall(last.get("answer") or "")))
    if len(packages) > 1 and "unittest" in route:
        return {"packages": packages[:8], "more": max(0, len(packages) - 8), "names": packages[:8], "test": None, "history": None,
                "label": "the packages listed above"}
    package = s._PACKAGE_PATH_RE.search(last.get("answer") or "")
    if package:
        return {"names": [package.group(1)], "test": None, "history": None, "label": package.group(1)}
    return None


def _open_lead(report: dict | None, label: str) -> str:
    items = [i for i in (report or {}).get("items", []) if i.get("strong")]
    opened = [i for i in items if i.get("state") == "open"]
    if opened:
        shown = "; ".join(f"{i['kind']} #{i['number']} - {i['title']} (opened {str(i.get('created_at') or '')[:10]}"
                          f"{' by ' + i['author'] if i.get('author') else ''})" for i in opened[:5])
        more = f" and {len(opened) - 5} more" if len(opened) > 5 else ""
        return f"**Open: {len(opened)}** mentioning \"{label}\": {shown}{more}."
    if items:
        return f"**No open PR or Issue** mentions \"{label}\"; {len(items)} closed or merged one(s) do, below."
    return ""


def _per_package_answer(subject: dict) -> str:
    """One line per package of a listed failure: the open PRs/Issues mentioning it, and the newest closed one. No AI check, so it stays fast."""
    s = _api()

    def search(package):
        try:
            return package, s._github_mentions_report({"exact": "", "wording": "", "names": [package], "category": None})
        except Exception as exc:
            print(f"WARNING: GitHub search for {package} failed: {exc}")
            return package, "failed"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(search, subject["packages"]))
    lines = [f"**Searched GitHub for each of the {len(results)} package(s) listed in the previous answer** (titles, descriptions and comments):"]
    for package, report in results:
        if report == "failed":
            lines.append(f"- `{package}`: the search failed, so this is unknown")
            continue
        items = [i for i in (report or {}).get("items", []) if i.get("strong")]
        opened = [i for i in items if i.get("state") == "open"]
        closed = sorted((i for i in items if i.get("state") != "open"), key=lambda i: str(i.get("created_at") or ""), reverse=True)
        if not items:
            lines.append(f"- `{package}`: nothing found")
            continue
        parts = []
        if opened:
            parts.append("**open:** " + "; ".join(f"{i['kind']} #{i['number']} - {i['title']}" for i in opened[:3])
                         + (f" (+{len(opened) - 3} more open)" if len(opened) > 3 else ""))
        else:
            parts.append("no open item")
        if closed:
            newest = closed[0]
            state = "merged" if newest.get("merged_at") else "closed"
            parts.append(f"{len(closed)} closed or merged mention it, newest {newest['kind']} #{newest['number']} - {newest['title']} ({state})")
        lines.append(f"- `{package}`: " + "; ".join(parts))
    if subject.get("more"):
        lines.append(f"- {subject['more']} more package(s) in that list were not searched: name one to search it.")
    lines.append("")
    lines.append("Text matches only: a PR or Issue naming a package doesn't prove it is about this failure, or caused or fixed it. "
                 + s._github_export_note())
    return "\n".join(lines).strip()


def github_named_ask_answer(question: str, sid: str, follow_up_only: bool = False) -> str | None:
    """A PR/Issue ask about quoted text or a named unit test, or "... related to this" right after an answer about one."""
    if "\n" in question or len(question) > 300 or not _GITHUB_ASK_RE.search(question):
        return None
    s = _api()
    if re.search(r"\b(?:merged|landed|went\s+in)\b", question, re.IGNORECASE) or s._WORKFLOW_ID_RE.search(question):
        return None  # merged-PR lists and workflow reports have their own rules
    subject = None if follow_up_only else _subject_in_question(question)
    if not subject:
        if not (_THIS_RE.search(question) and len(question) <= 140):
            return None
        subject = _remembered_subject(sid)
        if not subject:
            return None
    elif follow_up_only:
        return None
    if subject.get("packages"):
        _subjects[sid] = subject
        return _per_package_answer(subject)
    terms = {"exact": "", "wording": "", "names": subject["names"], "category": None}
    try:
        report = s._github_mentions_report(terms)
        section = s._github_reports_section("", question, terms, report) if report else None
        lead = _open_lead(report, subject["label"]) if _OPEN_RE.search(question) else ""
    except Exception as exc:
        print(f"WARNING: GitHub search for a named ask failed: {exc}")
        return None
    if not section:
        searched = ", ".join(f"\"{n}\"" for n in subject["names"])
        section = (f"**No PR or Issue mentions {searched}** in the imported titles, descriptions and comments. "
                   f"{s._github_export_note()}").strip()
    parts = [p for p in (lead, section) if p]
    history = subject.get("history")
    if not history and subject.get("test"):
        found = _unit_test_named(f"is {subject['test']} failing?")
        history = found[1] if found else None
    if history:
        parts.append(f"**In the IB test logs:** {history}\n\nAsk \"is {subject['test']} failing?\" for the build-by-build table.")
    _subjects[sid] = subject
    return "\n\n".join(parts)
