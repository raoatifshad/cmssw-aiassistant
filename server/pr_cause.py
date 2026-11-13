"""Which PR may have caused a failure, and what code did it touch? -- by rules, from the graph only.

Owner, 2026-11-12: the demo is mostly "is this failure reported?", "which PR caused it?", "what code did that PR touch?".
Found the same day on :8004: workflow 36434.0 fails since CMSSW_20_1_X_2026-11-06-1100 with "Unable to find plugin
'btlrechit::BTLBaseRecHitSoAProducer@alpaka'"; of the 8 PRs merged into that build only #51861 touched a file carrying
that name (RecoLocalFastTime/FTLRecProducers/plugins/alpaka/BTLBaseRecHitSoAProducerAlgo.h), but the AI path listed the
8 PRs in arbitrary order, put #52397 first, answered the follow-ups about #52397 and took 80-95 s per turn.

Method: the failing streak of the newest release series (last passing build -> first failing build, primary IBs), the PRs
merged into the builds of that window ((PR)-[:MERGED_BEFORE]->(IB)), ranked by whether the files they touched
((PR)-[:TOUCHED_FILE]->(SourceFile), from the GitHub backup's patches) carry names from the failure's error text, or lie in
the failing package. Timing plus file names is evidence to read, never proof of a cause (CLAUDE.md invariant): every answer says so.
"""
from __future__ import annotations

import re

from graph.tools import _ib_sort_key, get_driver

CAUSE_RE = re.compile(
    r"\b(?:which|what)\s+(?:pr|pull\s+request|change|commit|merge)s?\b[^?]{0,60}?\b(?:caus|broke|break|introduc|responsible|start|trigger)"
    r"|\bwhat\s+(?:changed|was\s+merged|went\s+in|got\s+merged)\b[^?]{0,40}?\b(?:before|prior\s+to)\b[^?]{0,30}?\b(?:fail|broke|break|start)"
    r"|\bwhat\s+(?:caused|broke|introduced|triggered)\b|\bwho\s+broke\b|\bculprit\b|\b(?:pr|change)s?\s+(?:that|which)\s+(?:caused|broke|introduced)\b"
    r"|\bcaused\s+by\s+(?:a|which|what)\s+(?:pr|change)\b|\bresponsible\s+(?:pr|change)\b", re.IGNORECASE)
FILES_RE = re.compile(
    r"\b(?:what|which)\s+(?:code|files?|packages?|sources?)\b[^?]{0,50}?\b(?:touch|chang|modif|edit)"
    r"|\b(?:does|did)\s+(?:it|that(?:\s+pr)?|this(?:\s+pr)?|the\s+pr)\s+(?:touch|change|modify)\b"
    r"|\b(?:files?|code)\s+(?:of|in)\s+(?:that|this|the)\s+pr\b", re.IGNORECASE)
_WHERE_RE = re.compile(r"\bwhere\s+(?:it|this)\s+fails\b|\bthe\s+(?:failing|broken)\s+(?:code|package|module|plugin)\b|\brelated\s+to\s+the\s+(?:error|failure)\b", re.IGNORECASE)
_PR_NUMBER_RE = re.compile(r"(?:\bpr\s*#?|#)(\d{3,6})\b", re.IGNORECASE)
_PACKAGE_RE = re.compile(r"\b([A-Z][A-Za-z0-9]+/[A-Z][A-Za-z0-9]+)\b")
_STOP = {"exception", "category", "framework", "module", "producer", "edproducer", "eventsetup", "please", "processing", "calling",
         "running", "message", "unable", "plugin", "spelling", "begin", "fatal", "notfound", "productnotfound", "pluginnotfound",
         "configuration", "edanalyzer", "edfilter", "stream", "global", "oneapi", "alpaka", "serialsync", "cudaasync", "rocmasync",
         # framework words in every exception's context lines: they would match FWCore files of any PR (found 2026-11-12)
         "eventprocessor", "eventprincipal", "principal", "getbytoken", "getbylabel", "esproducer", "edfilter", "outputmodule", "pathstatus",
         "triggerresults", "constructing", "beginjob", "beginrun", "endjob", "endrun", "lumiblock", "luminosityblock", "statusinfo", "modulecallingcontext"}

# session -> {"top": PR number or None, "names": [...], "package": str|None, "subject": str, "candidates": [numbers]}
_state: dict[str, dict] = {}


def _api():
    from server import api_server  # late: api_server imports this module
    return api_server


def error_names(text: str, category: str | None = None) -> list[str]:
    """Specific names in an error message: class, plugin and module names (namespace parts split, '@alpaka' dropped)."""
    names: list[str] = []
    for token in re.findall(r"[A-Za-z_][\w:]*(?:@\w+)?", text or ""):
        for part in token.split("@")[0].split("::"):
            if ":" in part or re.fullmatch(r"(?:0x)?[0-9a-fA-Fx]+", part) or sum(c.isdigit() for c in part) * 2 > len(part):
                continue  # host:pid, hex addresses and ids are no code names (found 2026-11-12)
            if part.endswith("Rcd") and len(part) > 9:
                part = part[:-3]  # EventSetup record EcalPedestalsRcd -> the data class EcalPedestals
            if (len(part) >= 6 and re.search(r"[a-z]", part) and re.search(r"[A-Z0-9_]", part[1:]) and part.lower() not in _STOP
                    and part != category and not part.isupper()):
                names.append(part)
    return list(dict.fromkeys(names))[:8]


def _match(files: list[str], title: str, names: list[str], packages: list[str], package_weight: int = 3) -> dict:
    """How a PR's touched files relate to the failure: files whose name carries an error name (strong), files in the failing package."""
    named, in_package = [], []
    for path in files:
        stem = path.rsplit("/", 1)[-1].split(".")[0].lower()
        lower = path.lower()
        hit = next((n for n in names if n.lower() in stem or (len(stem) >= 8 and stem in n.lower()) or f"/{n.lower()}/" in lower), None)
        if hit:
            named.append((path, hit))
        elif any(path.startswith(f"src/{p}/") for p in packages):
            in_package.append(path)
    in_title = [n for n in names + packages if n.lower() in (title or "").lower()]
    return {"named": named, "in_package": in_package, "in_title": in_title,
            "score": 10 * len(named) + package_weight * len(in_package) + 2 * len(in_title)}


def _short(path: str) -> str:
    return path[4:] if path.startswith("src/") else path


def _run(query: str, **params) -> list[dict]:
    driver = get_driver()
    try:
        with driver.session() as session:
            return [dict(r) for r in session.run(query, **params)]
    finally:
        driver.close()


def _streak(runs: list[dict]) -> dict | None:
    """Newest failing streak in time-ordered runs: first failing tag, last passing tag before it, newest status."""
    if not runs:
        return None
    end = len(runs) - 1
    while end >= 0 and runs[end]["status"] != "FAILED":
        end -= 1
    if end < 0:
        return None
    start = end
    while start > 0 and runs[start - 1]["status"] == "FAILED":
        start -= 1
    return {"first_bad": runs[start]["tag"], "last_good": runs[start - 1]["tag"] if start > 0 else None, "first_run": start == 0,
            "still_failing": end == len(runs) - 1, "newest": runs[-1]["tag"], "arch": runs[start]["arch"]}


def _pick_track(rows: list[dict], arch: str | None) -> dict | None:
    """The newest release series, per architecture; the asked architecture, else the one where the streak started first."""
    if not rows:
        return None
    newest_series = max(rows, key=lambda r: _ib_sort_key(r["tag"]))["tag"].rsplit("_X_", 1)[0]
    per_arch: dict[str, list] = {}
    for row in sorted((r for r in rows if r["tag"].startswith(newest_series + "_X_")), key=lambda r: r["tag"][-15:]):
        per_arch.setdefault(row["arch"], []).append(row)
    streaks = {a: s for a, s in ((a, _streak(r)) for a, r in per_arch.items()) if s}
    if arch:
        wanted = [a for a in streaks if arch in a]  # "aarch64", "riscv", "el8" name a track without the full arch string
        if wanted:
            return streaks[min(wanted, key=lambda a: (streaks[a]["first_bad"][-15:], a))]
    if not streaks:
        return None
    # an architecture where it passed and then started failing has a real start; one failing since our oldest build (riscv fails ~all
    # workflows) has none, and picking it gave "failing since 10-29, cause older than our data" for 36434.0 (found 2026-11-12)
    # (a streak that has since ended still has a start worth tracing: a riscv one failing since the oldest build must not hide it)
    current = [s for s in streaks.values() if s["still_failing"]] or list(streaks.values())
    started = [s for s in current if s["last_good"]] or [s for s in streaks.values() if s["last_good"]] or current
    return min(started, key=lambda s: (s["first_bad"][-15:], s["arch"] != "el9_amd64_gcc14", s["arch"]))


def _previous_build(first_bad: str, arch: str | None, kind: str) -> str | None:
    """The build of the same series and architecture just before first_bad that has results of this kind but not this test: the test is new
    there. One without any results proves nothing (data before 2026-10-26 was lost: SimG4Core/Geometry read as new, found 2026-11-12)."""
    rel = "RAN_UNITTEST" if kind == "unit" else "RAN_RELVAL"
    rows = _run(f"MATCH (ib:IB) WHERE ib.variant IS NULL AND ib.tag STARTS WITH $prefix AND right(ib.tag, 15) < $t AND ($arch IS NULL OR ib.arch = $arch) "
                f"AND EXISTS {{ MATCH (ib)-[:{rel}]->() }} "
                "RETURN ib.tag AS tag ORDER BY right(ib.tag, 15) DESC LIMIT 1", prefix=first_bad.rsplit("_X_", 1)[0] + "_X_", t=first_bad[-15:], arch=arch)
    return rows[0]["tag"] if rows else None


def _window_prs(first_bad: str, last_good: str | None) -> tuple[list[dict], list[str]]:
    prefix = first_bad.rsplit("_X_", 1)[0] + "_X_"
    low = last_good[-15:] if last_good else first_bad[-15:]
    rows = _run(
        "MATCH (p:PR)-[:MERGED_BEFORE]->(ib:IB) WHERE ib.tag STARTS WITH $prefix AND right(ib.tag, 15) <= $high "
        "AND (right(ib.tag, 15) > $low OR ($low = $high AND ib.tag = $first)) AND coalesce(p.repo, 'cmssw') = 'cmssw' "
        "OPTIONAL MATCH (p)-[:TOUCHED_FILE]->(f:SourceFile) "
        "RETURN p.number AS number, p.title AS title, p.merged_at AS merged_at, p.patch_status AS patch_status, "
        "collect(DISTINCT ib.tag) AS tags, collect(DISTINCT f.path) AS files",
        prefix=prefix, high=first_bad[-15:], low=low, first=first_bad)
    tags = sorted({t for r in rows for t in r["tags"]}, key=lambda t: t[-15:])
    return rows, tags


def _workflow_failure(workflow_id: str, arch: str | None) -> dict | None:
    s = _api()
    rows = _run("MATCH (ib:IB)-[:RAN_RELVAL]->(r:RelVal {workflow_id:$wf}) WHERE ib.variant IS NULL "
                "RETURN ib.tag AS tag, ib.arch AS arch, r.name AS name, "
                "CASE WHEN r.status = 'FAILED' AND NOT coalesce(r.known_error, false) THEN 'FAILED' ELSE coalesce(r.status, '?') END AS status",
                wf=workflow_id)
    if not rows:
        return None
    current = s._workflow_base_name(max(rows, key=lambda r: _ib_sort_key(r["tag"]))["name"])
    rows = [r for r in rows if s._workflow_base_name(r["name"]) == current]  # workflow numbers get reused
    track = _pick_track(rows, arch)
    if not track:
        return {"subject": f"workflow `{workflow_id}`", "not_failing": True}
    errors = _run("MATCH (e:RelValException {workflow_id:$wf}) WHERE e.ib_tag = $tag AND e.arch = $arch "
                  "RETURN e.category AS category, e.message AS message LIMIT 3", wf=workflow_id, tag=track["first_bad"], arch=track["arch"])
    if not errors:
        errors = [{"category": e["category"], "message": e["message"]} for e in s._workflow_current_errors(workflow_id)]
    error = next((e for e in errors if e.get("message")), None)
    names = error_names(error["message"], error.get("category")) if error else []
    return {**track, "subject": f"workflow `{workflow_id}` ({current})", "workflow": workflow_id, "error": error, "names": names,
            "packages": list(dict.fromkeys(_PACKAGE_RE.findall(error["message"] if error else "")))}


def _package_failure(package: str, arch: str | None) -> dict | None:
    rows = _run("MATCH (ib:IB)-[:RAN_UNITTEST]->(u:UnitTest {name:$pkg}) WHERE ib.variant IS NULL "
                "RETURN ib.tag AS tag, ib.arch AS arch, CASE WHEN u.status = 'FAILED' THEN 'FAILED' ELSE coalesce(u.status, '?') END AS status",
                pkg=package)
    track = _pick_track(rows, arch)
    if not track:
        return {"subject": f"unit tests of `{package}`", "not_failing": True} if rows else None
    # the package's own files are the evidence; its last word ("Geometry") as a name matched every Geometry/ file (found 2026-11-12)
    return {**track, "subject": f"unit tests of `{package}`", "package": package, "error": None, "names": [], "packages": [package], "package_weight": 10}


def _paste_failure(paste: str, arch: str | None) -> dict | None:
    """A pasted error: the workflow whose stored exception has the same message, else no build window (recent PRs only)."""
    s = _api()
    terms = s._paste_search_terms(paste)
    message = terms.get("exact") or ""
    names = error_names(paste, terms.get("category")) or [n for n in (terms.get("names") or []) if len(n) >= 6][:6]
    if message:
        rows = _run("MATCH (e:RelValException) WHERE e.message CONTAINS $m RETURN e.workflow_id AS wf, count(*) AS n ORDER BY n DESC LIMIT 1",
                    m=message[:150])
        if rows and rows[0]["wf"]:
            found = _workflow_failure(rows[0]["wf"], arch)
            if found and not found.get("not_failing"):
                return {**found, "names": names or found["names"], "subject": f"the pasted error (seen in {found['subject']})"}
    if not names:
        return None
    return {"subject": "the pasted error", "names": names, "packages": list(dict.fromkeys(_PACKAGE_RE.findall(paste))), "error": None, "no_window": True}


def _recent_prs(days: int = 14) -> tuple[list[dict], list[str]]:
    s = _api()
    latest = s._latest_ib_records(None, None)
    if not latest:
        return [], []
    prefix = latest[0]["tag"].rsplit("_X_", 1)[0] + "_X_"
    rows = _run("MATCH (p:PR)-[:MERGED_BEFORE]->(ib:IB) WHERE ib.tag STARTS WITH $prefix AND p.merged_at >= $since "
                "AND coalesce(p.repo, 'cmssw') = 'cmssw' OPTIONAL MATCH (p)-[:TOUCHED_FILE]->(f:SourceFile) "
                "RETURN p.number AS number, p.title AS title, p.merged_at AS merged_at, p.patch_status AS patch_status, "
                "collect(DISTINCT ib.tag) AS tags, collect(DISTINCT f.path) AS files",
                prefix=prefix, since=s._cutoff_iso(days) if hasattr(s, "_cutoff_iso") else _iso_days_ago(days))
    return rows, sorted({t for r in rows for t in r["tags"]}, key=lambda t: t[-15:])


def _iso_days_ago(days: int) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")


_ARCH_WORD_RE = re.compile(r"\b(aarch64|arm64|riscv(?:64)?|ppc64le|amd64|el\d+|fc\d+)\b", re.IGNORECASE)


def _arch_word(question: str) -> str | None:
    word = _ARCH_WORD_RE.search(question)
    return {"arm64": "aarch64", "riscv": "riscv64"}.get(word.group(1).lower(), word.group(1).lower()) if word else None


def _subject(question: str, sid: str) -> dict | None:
    s = _api()
    arch = s._requested_ib_arch(question) or _arch_word(question)
    workflow = s._WORKFLOW_ID_RE.search(s.IB_TAG_RE.sub(" ", question))
    if workflow:
        return _workflow_failure(workflow.group(1), arch)
    package = _PACKAGE_RE.search(question)
    if package:
        return _package_failure(package.group(1), arch)
    previous = _state.get(sid)
    last = s._turn_memory.get(sid) or {}
    if previous and last.get("route") == "deterministic:pr_cause":
        return previous.get("failure")
    paste = s._session_pastes.get(sid)
    if paste:
        return _paste_failure(s._split_paste_and_ask(paste)[0], arch)
    listed = _listed_workflows(last)
    if listed:
        return {"multi": listed, "arch": arch}
    if s._last_workflow_contexts.get(sid):
        return _workflow_failure(s._last_workflow_contexts[sid], arch)
    # "is workflow 36434.0 failing reported?" then "which PR caused it?": the reported-route doesn't set the workflow context
    asked = s._WORKFLOW_ID_RE.search(s.IB_TAG_RE.sub(" ", last.get("question") or "")) if last and not last.get("superseded") else None
    if asked:
        return _workflow_failure(asked.group(1), arch)
    named = _PACKAGE_RE.search(last.get("answer") or "") if last.get("route") in ("deterministic:package_status", "capability:unittest",
                                                                                 "deterministic:named_test", "deterministic:github_named_ask") else None
    return _package_failure(named.group(1), arch) if named else None


_LISTED_WF_RE = re.compile(r"(?:\bwf\s+|`)(\d{1,8}\.\d{1,7})(?:`|\b)")


def _listed_workflows(last: dict) -> list[str]:
    """Failing workflows listed by the previous answer about a whole build ("are the RelVals ok there?", a comparison): 2 or more of them.
    'which PR could have caused it?' after such a list went to the AI (34 s), which only looked at the newest build's PRs and missed
    #51861 for 36434.0/36834.0, failing since 11-06 (found live 2026-11-13)."""
    if not last or last.get("superseded") or last.get("route") in ("deterministic:pr_cause", "deterministic:pr_cause_files"):
        return []
    s = _api()
    if s._WORKFLOW_ID_RE.search(s.IB_TAG_RE.sub(" ", last.get("question") or "")):
        return []  # the previous question named one workflow: that one is the subject
    answer = last.get("answer") or ""
    if not re.search(r"\bfail", answer, re.IGNORECASE):
        return []
    # only the failing part: a comparison's "Resolved" list names workflows that pass now
    answer = re.split(r"\*\*Resolved[^*]*\*\*|^#+\s*Resolved", answer, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)[0]
    ids = list(dict.fromkeys(_LISTED_WF_RE.findall(answer)))
    return ids[:6] if len(ids) >= 2 else []


def _rank(failure: dict) -> tuple[list[dict], list[str]]:
    """The PRs merged in the failure's window (or the last 14 days without one), best file match first."""
    if failure.get("no_window"):
        rows, tags = _recent_prs()
    else:
        if not failure.get("last_good") and failure.get("first_run"):
            failure["before"] = _previous_build(failure["first_bad"], failure.get("arch"), "unit" if failure.get("package") else "relval")
        rows, tags = _window_prs(failure["first_bad"], failure.get("last_good") or failure.get("before"))
    names, packages = failure.get("names") or [], failure.get("packages") or []
    ranked = []
    for row in rows:
        known = row.get("patch_status") == "ok" or bool(row["files"])
        ranked.append({**row, "known": known, **_match(row["files"], row["title"] or "", names, packages, failure.get("package_weight", 3))})
    ranked.sort(key=lambda r: (-r["score"], -(r["number"] or 0)))
    return ranked, tags


def _multi_cause_answer(ids: list[str], arch: str | None, sid: str) -> str | None:
    """One block per failure start: workflows that started failing in the same build with the same top PR are shown together."""
    s = _api()
    groups: dict[tuple, dict] = {}
    for wf in ids:
        failure = _workflow_failure(wf, arch)
        if not failure or failure.get("not_failing"):
            continue
        ranked, _tags = _rank(failure)
        top = ranked[0] if ranked and ranked[0]["score"] > 0 else None
        key = (failure["first_bad"], failure.get("last_good"), top["number"] if top else None)
        group = groups.setdefault(key, {"failures": [], "ranked": ranked, "top": top})
        group["failures"].append(failure)
    if not groups:
        return None
    order = sorted(groups.values(), key=lambda g: (g["top"] is None, g["failures"][0]["first_bad"][-15:]))
    lines = [f"**Which PR may have caused them: the {len(ids)} failing workflow(s) of the previous answer**, each traced from where it started failing"
             + (f" on `{arch}`" if arch else "") + ":"]
    for group in order:
        first = group["failures"][0]
        names = ", ".join(f"`{f['workflow']}`" for f in group["failures"])
        title = first["subject"].split("(", 1)[-1].rstrip(")") if "(" in first["subject"] else ""
        if first.get("last_good"):
            when = f"started failing in `{first['first_bad']}` on `{first['arch']}` (last pass `{first['last_good']}`)"
        elif first.get("before"):
            when = f"new in `{first['first_bad']}` on `{first['arch']}` (not run in `{first['before']}`)"
        else:
            when = f"failing since our oldest build `{first['first_bad']}` on `{first['arch']}`, so its start is older than our data"
        lines.append(f"\n- {names}{f' ({title})' if title else ''}: {when}.")
        if first.get("error"):
            lines.append(f"  Error: `{first['error'].get('category') or 'exception'}`: {first['error']['message'][:160]}")
        top = group["top"]
        if top:
            label = "Possible match" if not first.get("last_good") and not first.get("before") else "Most likely"
            lines.append(f"  **{label}: PR #{top['number']}** - {top['title']} (merged {str(top.get('merged_at') or '')[:10]}): {_evidence(top)}")
        else:
            lines.append(f"  **No clear candidate:** none of the {len(group['ranked'])} PR(s) merged in that window touched a file named in its error.")
    tops = [g["top"]["number"] for g in order if g["top"]]
    lines.append("\n" + _caveat(s) + (f" Ask \"which PR caused {order[0]['failures'][0]['workflow']}?\" for one workflow in detail"
                                       + (f", or \"what code did that PR touch?\" for PR #{tops[0]}." if tops else ".")))
    first_failure = order[0]["failures"][0]
    _state[sid] = {"top": tops[0] if tops else None, "failure": first_failure, "names": first_failure.get("names") or [],
                   "packages": first_failure.get("packages") or [], "candidates": tops}
    return "\n".join(lines)


def cause_answer(question: str, sid: str) -> str | None:
    """'which PR caused it?' about a workflow, a unit-test package, a pasted error, or the subject of the previous answer."""
    if "\n" in question.strip() or not CAUSE_RE.search(question):
        return None
    s = _api()
    if s.IB_TAG_RE.search(question) and not s._WORKFLOW_ID_RE.search(s.IB_TAG_RE.sub(" ", question)) and not _PACKAGE_RE.search(question):
        return None  # "which PR broke CMSSW_..._X_...?" has its own rule (the build's merged PRs)
    try:
        failure = _subject(question, sid)
    except Exception as exc:
        print(f"WARNING: failure lookup for the cause question failed: {exc}")
        return None
    if not failure:
        return None
    if failure.get("multi"):
        try:
            return _multi_cause_answer(failure["multi"], failure.get("arch"), sid)
        except Exception as exc:
            print(f"WARNING: cause lookup for the listed workflows failed: {exc}")
            return None
    if failure.get("not_failing"):
        return (f"{failure['subject'][0].upper()}{failure['subject'][1:]} is not failing in the newest imported builds of its release series, "
                "so there is no failure to trace to a PR. Name the build or architecture where you saw it fail.")
    try:
        ranked, tags = _rank(failure)
    except Exception as exc:
        print(f"WARNING: PR window lookup failed: {exc}")
        return None
    names, packages = failure.get("names") or [], failure.get("packages") or []
    lines = [f"**Which PR may have caused it: {failure['subject']}**"]
    if failure.get("no_window"):
        lines.append("I found no failing build with this error in the imported results, so there is no last-good/first-bad window. "
                     f"Below: PRs merged into the newest release series in the last 14 days ({len(ranked)}), ranked by the files they touched.")
    else:
        where = f" on `{failure['arch']}`" if failure.get("arch") else ""
        if failure.get("last_good"):
            lines.append(f"- Started failing in `{failure['first_bad']}`{where}; last passed in `{failure['last_good']}`."
                         + ("" if failure.get("still_failing") else f" (It passes again in `{failure['newest']}`.)"))
        elif failure.get("before"):
            lines.append(f"- It first ran, and failed, in `{failure['first_bad']}`{where}: the build before, `{failure['before']}`, didn't run it, "
                         "so it is new there (or was renamed), and the PRs merged in between are the ones to look at.")
        else:
            lines.append((f"- Failing in every imported build since `{failure['first_bad']}`{where}, our oldest one" if failure.get("still_failing") else
                          f"- Failed in the oldest imported builds, from `{failure['first_bad']}`{where}, and passes again in `{failure['newest']}`")
                         + ": the start is older than our data, so the PRs below may not include the cause.")
        if len(tags) > 1:
            lines.append(f"- {len(tags)} builds in that window: PRs merged into any of them count.")
    if failure.get("error"):
        lines.append(f"- Its error there: `{failure['error'].get('category') or 'exception'}`: {failure['error']['message'][:220]}")
    if names or packages:
        lines.append("- Looked for in the PRs' files: " + ", ".join(f"`{n}`" for n in (names + packages)[:8]) + ".")
    if not ranked:
        lines.append("\nNo merged PR is recorded for those builds in the imported data.")
        _state[sid] = {"top": None, "failure": failure, "names": names, "packages": packages, "candidates": []}
        return "\n".join(lines) + "\n\n" + _caveat(s)
    top = ranked[0] if ranked[0]["score"] > 0 else None
    old_start = not failure.get("no_window") and not failure.get("last_good") and not failure.get("before")
    lines.append("")
    if top:
        evidence = _evidence(top)
        lines.append(f"**{'Possible match, the real start is older than our data' if old_start else 'Most likely'}: PR #{top['number']}** - {top['title']} (merged {str(top.get('merged_at') or '')[:10]}): {evidence}")
        others = [r for r in ranked[1:] if r["score"] > 0]
        for other in others[:3]:
            lines.append(f"- Also related: PR #{other['number']} - {other['title']}: {_evidence(other)}")
    else:
        lines.append(f"**No clear candidate:** none of the {len(ranked)} PR(s) touched a file named in the error"
                     + (" or in the failing package" if packages else "") + ".")
    rest = [r for r in ranked if r["score"] == 0]
    unknown = [r for r in rest if not r["known"]]
    if rest:
        shown = ", ".join(f"#{r['number']}" for r in rest[:12]) + (f" and {len(rest) - 12} more" if len(rest) > 12 else "")
        lines.append(f"- The other {len(rest)} PR(s) merged then touched nothing that matches: {shown}."
                     + (f" Files are unknown for {len(unknown)} of them (no readable patch in the GitHub backup)." if unknown else ""))
    lines.append("\n" + _caveat(s) + (f" Ask \"what code did that PR touch?\" for PR #{top['number']}'s files." if top else ""))
    _state[sid] = {"top": top["number"] if top else None, "failure": failure, "names": names, "packages": packages,
                   "candidates": [r["number"] for r in ranked]}
    return "\n".join(lines)


def _evidence(row: dict) -> str:
    parts = []
    if row["named"]:
        files = ", ".join(f"`{_short(p)}`" for p, _ in row["named"][:3])
        more = f" and {len(row['named']) - 3} more" if len(row["named"]) > 3 else ""
        parts.append(f"it changed {files}{more}, named like `{row['named'][0][1]}` in the error")
    if row["in_package"]:
        parts.append(f"{len(row['in_package'])} file(s) in the failing package")
    if row["in_title"] and not row["named"]:
        parts.append("its title names " + ", ".join(f"`{n}`" for n in row["in_title"][:2]))
    return "; ".join(parts) + "."


def _caveat(s) -> str:
    return ("_How this was found: the PRs merged into the builds between the last pass and the first failure, ranked by whether the files they "
            "touched carry names from the error or lie in the failing package. Timing and file names are evidence to check, not proof that a PR "
            f"caused the failure. {s._github_export_note()}_")


def files_answer(question: str, sid: str) -> str | None:
    """'what code did that PR touch?' / 'does it touch where it fails?' right after a cause answer: the top candidate's files, matching ones first."""
    if "\n" in question.strip() or not (FILES_RE.search(question) or _WHERE_RE.search(question)) or _PR_NUMBER_RE.search(question):
        return None
    s = _api()
    last = s._turn_memory.get(sid) or {}
    state = _state.get(sid)
    if not state or last.get("route") not in ("deterministic:pr_cause", "deterministic:pr_cause_files"):
        return None
    number = state.get("top")
    if not number:
        return "The previous answer found no PR that touched code named in the error, so there is no single PR to show. Name one (\"what files did PR 12345 change?\")."
    rows = _run("MATCH (p:PR {number:$n}) WHERE coalesce(p.repo, 'cmssw') = 'cmssw' OPTIONAL MATCH (p)-[:TOUCHED_FILE]->(f:SourceFile) "
                "RETURN p.title AS title, p.patch_status AS patch_status, p.changed_files AS changed, collect(DISTINCT f.path) AS files", n=number)
    if not rows:
        return None
    row = rows[0]
    files = sorted(row["files"])
    if not files:
        return f"PR #{number} - {row['title']}: its files are unknown (the GitHub backup has no readable patch for it, status `{row['patch_status']}`)."
    match = _match(files, row["title"] or "", state.get("names") or [], state.get("packages") or [])
    named = [p for p, _ in match["named"]]
    lines = [f"**PR #{number} - {row['title']}** touched {len(files)} file(s)" + (f" (GitHub counts {row['changed']})" if row.get("changed") and row["changed"] != len(files) else "") + "."]
    if _WHERE_RE.search(question) or re.search(r"\b(?:does|did)\s+(?:it|that|this|the)\b[^?]*\b(?:touch|change|modify)", question, re.IGNORECASE):
        lines.append(("**Yes:** " if named or match["in_package"] else "**No:** ")
                     + (f"{len(named)} of them carry a name from the error ({', '.join(f'`{n}`' for n in dict.fromkeys(n for _, n in match['named']))})."
                        if named else f"{len(match['in_package'])} are in the failing package." if match["in_package"]
                        else "none of its files carries a name from the error or lies in the failing package."))
    if named:
        lines.append("\nFiles matching the error:")
        lines += [f"- `{_short(p)}`" for p in named[:15]]
    packages: dict[str, int] = {}
    for path in files:
        parts = _short(path).split("/")
        packages["/".join(parts[:2])] = packages.get("/".join(parts[:2]), 0) + 1
    ranked = sorted(packages.items(), key=lambda kv: -kv[1])
    lines.append(f"\nBy package ({len(ranked)}): " + ", ".join(f"`{p}` ({n})" for p, n in ranked[:12]) + (f" and {len(ranked) - 12} more" if len(ranked) > 12 else "") + ".")
    if len(files) <= 25 and not named:
        lines += [f"- `{_short(p)}`" for p in files]
    lines.append(f"\nAsk \"what files did PR {number} change?\" for the full list. A file being touched shows where the PR worked, not that it broke the test.")
    return "\n".join(lines)


def cause_hint(workflow_id: str) -> dict | None:
    """One line under 'is workflow X reported?': where the failure started and the PR whose files match its error."""
    try:
        failure = _workflow_failure(workflow_id, None)
        if not failure or failure.get("not_failing") or not failure.get("last_good"):
            return None
        rows, _tags = _window_prs(failure["first_bad"], failure["last_good"])
    except Exception as exc:
        print(f"WARNING: cause hint failed: {exc}")
        return None
    ranked = sorted(({**r, **_match(r["files"], r["title"] or "", failure["names"], failure["packages"])} for r in rows), key=lambda r: -r["score"])
    head = f"It started failing in `{failure['first_bad']}` (last pass `{failure['last_good']}`, `{failure['arch']}`)"
    start = failure["first_bad"].rsplit("_X_", 1)[-1][:10]
    if ranked and ranked[0]["score"] > 0:
        top = ranked[0]
        evidence = re.sub(r"^it ", "", _evidence(top))
        return {"start": start, "text": f"{head}. Of the {len(rows)} PR(s) merged then, **PR #{top['number']}** ({top['title']}) {evidence} "
                                        "Ask \"which PR caused it?\" for the ranking (evidence, not proof)."}
    return {"start": start, "text": f"{head}; none of the {len(rows)} PR(s) merged then touched a file named in its error."}
