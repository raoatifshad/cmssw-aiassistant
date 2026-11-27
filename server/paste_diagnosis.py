"""A pasted CMSSW fatal exception, read like a shifter would (user's example, 2026-11-09: MEtoEDMConverter 'deleteAfterCopy'):
what failed and when in the job, which lines of the paste are only noise, whether a PR names this exact problem and is merged,
which IB includes it, and the PR its own description says started it. Facts come from the paste, the graph and the local CMSSW
clone only -- never from the web, never from an LLM. A PR found here is evidence to read, not proof of a cause or a fix: every
claim says where it comes from."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from graph.tools import _ib_sort_key, get_driver

_CET = ZoneInfo("Europe/Zurich")

_CATEGORY_RE = re.compile(r"An exception of category '([^']+)' occurred while")
_FRAME_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*$", re.MULTILINE)
_MODULE_RE = re.compile(r"class=([A-Za-z_][\w:]*)(?:@\w+)?\s+label='([^']*)'")
_MODULE_ALT_RE = re.compile(r"\bmodule\s+([A-Za-z_][\w:]*)/'([^']+)'")
_MESSAGE_RE = re.compile(r"Exception Message:\s*(.+?)(?:-{5}\s*End Fatal Exception|\Z)", re.DOTALL)
_RECORD_RE = re.compile(r'No "(\w+)" record found')
_PLUGIN_RE = re.compile(r"Unable to find plugin '([\w:]+)(?:@\w+)?'")
_PARAMETER_RE = re.compile(r"[Pp]arameter (?:is named|named|)\s*:?\s*'([A-Za-z_]\w*)'")
_FATAL_TIME_RE = re.compile(r"Begin Fatal Exception\s+(\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2}:\d{2})")
_LOG_TIME_RE = re.compile(r"(\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2}:\d{2})")
_TIMEOUT_RE = re.compile(r"\btimeout\s+(?:--signal\s+\w+\s+)?(\d{2,6})\s+cmsRun\b")
_STEP_RE = re.compile(r"\bcmsRun\b.*?\b(step\d+)\w*\.py")
_SPEC_RE = re.compile(r"^Step:\s*(\w+)\s+Spec:\s*\[(.+?)\]\s*$", re.MULTILINE)
_OPENED_RE = re.compile(r"Successfully opened file\s+(\S+)")

# Framework message families: what the message means in CMSSW in general. Only well-known framework wording; an unknown
# message gets no explanation rather than a guess. "fixed": the same job fails the same way every time (not transient).
_FAMILIES = [
    (re.compile(r"Illegal parameter found in configuration", re.IGNORECASE), "illegal configuration parameter", True,
     "The Python configuration of `{label}` sets the parameter `{parameter}`, but the C++ plugin `{cls}` does not declare it in its "
     "`fillDescriptions()`, so the configuration is rejected while it is validated. The configuration and the plugin's code "
     "disagree: usually the parameter was removed or renamed in the C++ code, or a newly added `fillDescriptions()` made an old "
     "parameter illegal, while a configuration still sets it."),
    (re.compile(r"Missing required parameter|MissingParameter|parameter .{0,80} (?:is )?missing", re.IGNORECASE), "missing configuration parameter", True,
     "The plugin `{cls}` requires a parameter{param_named} that the configuration of `{label}` does not set: the plugin's "
     "`fillDescriptions()` and the configuration disagree."),
    (re.compile(r"(?:is|has) type .{0,60} but .{0,40}(?:should|expected)|wrong type", re.IGNORECASE), "configuration parameter of the wrong type", True,
     "A parameter of `{label}` has a different type in the configuration than the plugin `{cls}` declares."),
    (re.compile(r'No "\w+" record found in the EventSetup', re.IGNORECASE), "missing EventSetup record", True,
     "Nothing in this job's configuration delivers the EventSetup record `{record}`: the ESSource or ESProducer that makes it is not "
     "loaded (or, for conditions, not in what the job reads), while `{label}` needs it."),
    (re.compile(r"Unable to find plugin|PluginNotFound", re.IGNORECASE), "plugin not found", True,
     "The configuration asks for a plugin that is not in the release's plugin cache: its name is misspelled, or its "
     "library was removed, renamed or not built."),
    (re.compile(r"Found zero products matching|ProductNotFound|Principal::getByToken", re.IGNORECASE), "product not found", False,
     "A module asked the event for a product that is not there: its producer did not run in this job or an earlier step, "
     "or the input file was written without it."),
    (re.compile(r"FallbackFileOpenError|FileOpenError|Failed to open the file|could not be opened", re.IGNORECASE), "input file could not be opened", None,
     "An input file could not be opened (storage, site or network). This kind of failure is often transient."),
]

_PHASES = [
    (re.compile(r"Constructing the EventProcessor|Validating configuration of module|Constructing module", re.IGNORECASE),
     "while the job was being set up (constructing the EventProcessor), before any event was read"),
    (re.compile(r"Calling beginJob|beginJob", re.IGNORECASE), "at the start of the job (beginJob), before any event"),
    (re.compile(r"beginRun|global begin Run|beginLuminosityBlock|global begin Lumi", re.IGNORECASE), "at the start of a run or luminosity block"),
    (re.compile(r"Processing\s+(?:Event|event)|Calling method for module|stream \d+", re.IGNORECASE), "while processing events"),
    (re.compile(r"endJob|endRun|global end", re.IGNORECASE), "at the end of the job"),
]

# lines of a RelVal step log that are normal and not the error (each with what it is)
_NOISE = [
    (re.compile(r"^Warning: The default (\w+) is being used", re.MULTILINE), "the `{0}` warning: a standard cmsDriver notice, unrelated"),
    (re.compile(r"oneDNN custom operations are on|absl::InitializeLog", re.MULTILINE), "the oneDNN / TensorFlow messages: informational"),
    (re.compile(r"^%MSG-i ThreadStreamSetup", re.MULTILINE), "the thread/stream setup messages: informational"),
]

_COMMON_WORDS = {"step", "cmsrun", "configuration", "process", "module", "parameter", "exception", "event", "file"}


def _parse_time(text: str) -> datetime | None:
    try:
        return datetime.strptime(text, "%d-%b-%Y %H:%M:%S")
    except ValueError:
        return None


def parse_fatal_paste(paste: str) -> dict | None:
    """The facts a CMSSW fatal exception paste states. None when it is not one."""
    category = _CATEGORY_RE.search(paste)
    if not category:
        return None
    message_match = _MESSAGE_RE.search(paste)
    message = " ".join(message_match.group(1).split()) if message_match else ""
    frames = [f for _, f in _FRAME_RE.findall(paste[category.end():message_match.start() if message_match else None])]
    module = _MODULE_RE.search(paste) or _MODULE_ALT_RE.search(paste)
    parameter = _PARAMETER_RE.search(message)
    phase = next((text for rx, text in _PHASES if any(rx.search(f) for f in frames)), None)
    family = next(((name, fixed, explain) for rx, name, fixed, explain in _FAMILIES if rx.search(message)), None)
    timeout = _TIMEOUT_RE.search(paste)
    fatal_at = _FATAL_TIME_RE.search(paste)
    # the job's own log before the fatal banner: the banner's time alone says nothing about how long the job ran
    started = _LOG_TIME_RE.search(paste[:fatal_at.start()]) if fatal_at else None
    elapsed = None
    if fatal_at and started:
        t0, t1 = _parse_time(started.group(1)), _parse_time(fatal_at.group(1))
        if t0 and t1 and t1 >= t0:
            elapsed = int((t1 - t0).total_seconds())
    noise = []
    for rx, text in _NOISE:
        found = rx.search(paste)
        if found:
            noise.append(text.format(*found.groups()))
    opened = sorted(set(_OPENED_RE.findall(paste)))
    if opened:
        noise.append(", ".join(f"`{f.removeprefix('file:')}`" for f in opened) + " opened successfully: the input is readable")
    step = _STEP_RE.search(paste)
    return {
        "category": category.group(1),
        "message": message,
        "frames": frames,
        "module_class": module.group(1) if module else None,
        "module_label": module.group(2) if module else None,
        "parameter": parameter.group(1) if parameter else None,
        "record": (_RECORD_RE.search(message) or [None, None])[1],
        "plugin": (_PLUGIN_RE.search(message) or [None, None])[1],
        "phase": phase,
        "family": family,
        "timeout": int(timeout.group(1)) if timeout else None,
        "elapsed": elapsed,
        "noise": noise,
        "step": step.group(1) if step else None,
        "specs": [(s, [x.strip(" '\"") for x in v.split(",")]) for s, v in _SPEC_RE.findall(paste)],
    }


def _specific_terms(facts: dict) -> list[str]:
    """Names from the paste that identify this problem, most specific first."""
    terms = [facts.get("parameter"), facts.get("record"), (facts.get("plugin") or "").split("::")[-1] or None,
             facts.get("module_label"), (facts.get("module_class") or "").split("::")[-1] or None]
    return [t for t in dict.fromkeys(terms) if t and len(t) >= 6 and t.lower() not in _COMMON_WORDS]


def _cet(iso: str | None) -> str | None:
    if not iso:
        return None
    try:
        when = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso[:16]
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(_CET).strftime("%Y-%m-%d %H:%M CET")


def _tag_time(tag: str) -> datetime | None:
    found = re.search(r"(\d{4}-\d{2}-\d{2}-\d{4})$", tag or "")
    if not found:
        return None
    return datetime.strptime(found.group(1), "%Y-%m-%d-%H%M").replace(tzinfo=_CET)


_PR_FIELDS = ("p.number AS number, p.title AS title, coalesce(p.github_state, p.state) AS state, p.merged_at AS merged_at, "
              "p.created_at AS created_at, p.release AS release, p.body AS body, p.labels AS labels, p.changed_files AS changed_files, "
              "p.patch_status AS patch_status, p.fetched_via AS fetched_via, p.source_id AS source_id, "
              "head([(p)-[:AUTHORED_BY]->(a) | a.name]) AS author, "
              "[(p)-[:TOUCHED_FILE]->(f) | f.path][..12] AS files, "
              "[(p)-[:MERGED_BEFORE]->(ib:IB) | ib.tag] AS in_ibs")


def _labels(value) -> list[str]:
    if isinstance(value, list):
        return value
    try:
        return list(json.loads(value or "[]"))
    except (ValueError, TypeError):
        return []


def _where_recorded(session, facts: dict) -> list[str]:
    """IB tags where this error is recorded: RelVal exceptions and test-log problems with the same specific names."""
    terms = _specific_terms(facts)[:1]
    if not terms:
        return []
    # the most specific name only: a recorded message keeps the parameter but not always the module label
    term = terms[0].lower()
    tags = set(session.run(
        "MATCH (e:RelValException) WHERE toLower(coalesce(e.message, '') + ' ' + coalesce(e.text, '')) CONTAINS $term "
        "RETURN collect(DISTINCT e.ib_tag) AS tags", term=term).single()["tags"] or [])
    # the fulltext index finds the candidates (a scan of every LogProblem takes 4-5 s), the CONTAINS keeps exact ones
    phrase = '"' + re.sub(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)', r"\\\1", terms[0]) + '"'
    tags |= set(session.run(
        "CALL db.index.fulltext.queryNodes('logProblemText', $phrase) YIELD node "
        "WHERE node.ib_tag IS NOT NULL AND toLower(coalesce(node.text, '') + ' ' + coalesce(node.message, '')) CONTAINS $term "
        "RETURN collect(DISTINCT node.ib_tag) AS tags", phrase=phrase, term=term).single()["tags"] or [])
    return sorted(tags, key=lambda t: (_tag_time(t) or datetime.min.replace(tzinfo=_CET), t))


def find_fix_history(facts: dict) -> dict:
    """PRs/Issues whose title names this problem, the items their descriptions point to, and the builds involved."""
    terms = _specific_terms(facts)
    out = {"terms": terms, "fixes": [], "issues": [], "origins": [], "recorded": [], "candidates": []}
    if not terms:
        return out
    driver = get_driver()
    try:
        with driver.session() as session:
            out["recorded"] = _where_recorded(session, facts)
            lowered = [t.lower() for t in terms]
            rows = [dict(r) for r in session.run(
                "MATCH (p:PR {repo:'cmssw'}) WHERE any(t IN $terms WHERE toLower(p.title) CONTAINS t) "
                f"RETURN {_PR_FIELDS} ORDER BY p.number DESC LIMIT 20", terms=lowered)]
            issues = [dict(r) for r in session.run(
                "MATCH (i:Issue {repo:'cmssw'}) WHERE any(t IN $terms WHERE toLower(i.title) CONTAINS t) "
                "RETURN i.number AS number, i.title AS title, i.state AS state, i.created_at AS created_at, "
                "head([(i)-[:AUTHORED_BY]->(a) | a.name]) AS author ORDER BY i.number DESC LIMIT 10", terms=lowered)]
            first_recorded = out["recorded"][0] if out["recorded"] else None
            for row in rows:
                text = f"{row['title']} {row.get('body') or ''}".lower()
                title = row["title"].lower()
                in_title = [t for t in terms if t.lower() in title]
                in_text = [t for t in terms if t.lower() in text]
                # the most specific name in the title and a second name from the paste in the title or description
                row["named"] = in_title
                row["exact"] = bool(in_title) and terms[0].lower() in title and len(in_text) >= 2
                if row["exact"] and first_recorded and row.get("created_at"):
                    # a PR opened long before the error first appeared is not about this occurrence
                    tag_time = _tag_time(first_recorded)
                    try:
                        opened = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                        if opened.tzinfo is None:
                            opened = opened.replace(tzinfo=timezone.utc)
                        row["exact"] = not tag_time or (tag_time - opened).days < 60
                    except ValueError:
                        pass
            out["fixes"] = [r for r in rows if r["exact"]]
            out["issues"] = [i for i in issues if terms[0].lower() in i["title"].lower()]
            refs: list[int] = []
            for fix in out["fixes"][:3]:
                for number in re.findall(r"(?:#|/pull/|/issues/)(\d{4,6})\b", fix.get("body") or ""):
                    if int(number) != fix["number"] and int(number) not in refs:
                        refs.append(int(number))
            if refs:
                out["origins"] = [dict(r) for r in session.run(
                    f"MATCH (p:PR {{repo:'cmssw'}}) WHERE p.number IN $numbers RETURN {_PR_FIELDS}", numbers=refs[:4])]
                for origin in out["origins"]:
                    origin["named_by"] = next(f["number"] for f in out["fixes"] if re.search(rf"(?:#|/pull/){origin['number']}\b", f.get("body") or ""))
            if not out["origins"] and out["recorded"]:
                out["candidates"] = _first_build_changes(session, facts, out["recorded"])
    finally:
        driver.close()
    return out


def _first_build_changes(session, facts: dict, recorded: list[str]) -> list[dict]:
    """PRs that entered the IB(s) where the error is first recorded and touch a file (or have a title) with the failing names in it:
    what a shifter checks by hand. Changed files come from the GitHub backup's patches, so newer PRs match by title only."""
    first = _first_ib(recorded)
    first_time = _tag_time(first) if first else None
    if not first_time:
        return []
    tags = [t for t in recorded if _tag_time(t) == first_time]
    names = []
    for value in (facts.get("record"), facts.get("plugin"), facts.get("module_class"), facts.get("parameter")):
        short = re.sub(r"(?:Rcd|Record)$", "", (value or "").split("::")[-1])
        if len(short) >= 6 and short.lower() not in _COMMON_WORDS:
            names.append(short.lower())
    if not names:
        return []
    rows = [dict(r) for r in session.run(
        "MATCH (p:PR)-[:MERGED_BEFORE]->(ib:IB) WHERE ib.tag IN $tags WITH DISTINCT p "
        "OPTIONAL MATCH (p)-[:TOUCHED_FILE]->(f:SourceFile) WITH p, collect(f.path) AS files "
        "WITH p, files, [x IN files WHERE any(n IN $names WHERE toLower(x) CONTAINS n)] AS hits "
        "WHERE size(hits) > 0 OR any(n IN $names WHERE toLower(p.title) CONTAINS n) "
        "RETURN p.number AS number, p.title AS title, p.merged_at AS merged_at, head([(p)-[:AUTHORED_BY]->(a) | a.name]) AS author, "
        "hits[..4] AS hits, size(files) AS n_files ORDER BY size(hits) DESC, p.number DESC LIMIT 5",
        tags=tags, names=names)]
    total = session.run("MATCH (p:PR)-[:MERGED_BEFORE]->(ib:IB) WHERE ib.tag IN $tags RETURN count(DISTINCT p) AS n", tags=tags).single()["n"]
    for row in rows:
        row["first_ib"], row["entered"] = _primary_first(tags), total
    return rows


def _first_ib(tags: list[str], series: str | None = None) -> str | None:
    pool = [t for t in tags if not series or t.startswith(series)]
    return min(pool, key=lambda t: (_tag_time(t) or datetime.max.replace(tzinfo=_CET), len(t)), default=None)


def _series(tag: str) -> str:
    found = re.match(r"(CMSSW_\d+_\d+)_", tag or "")
    return found.group(1) if found else ""


def _primary_first(tags: list[str]) -> str | None:
    """The earliest tag, preferring the primary IB of that build time."""
    first = _first_ib(tags)
    if not first:
        return None
    same_time = [t for t in tags if _tag_time(t) == _tag_time(first)]
    primary = [t for t in same_time if re.fullmatch(r"CMSSW_\d+_\d+_X_\d{4}-\d{2}-\d{2}-\d{4}", t)]
    return (primary or sorted(same_time, key=len))[0]


def _source_lines(facts: dict) -> list[dict]:
    """Where the local CMSSW clone defines the failing module and sets the named parameter."""
    try:
        from server.cmssw_source import _git, source_version
    except Exception:
        return []
    label, parameter = facts.get("module_label"), facts.get("parameter")
    if not label or not parameter or not source_version():
        return []
    hits = []
    defining = _git("grep", "-n", "-F", "-e", f"{label} = ", "-e", f"{label}=", "HEAD", "--", "*.py")
    files = {line.split(":", 2)[1] for line in (defining or "").splitlines() if line.count(":") >= 2}
    setting = _git("grep", "-n", "-w", "-F", "-e", parameter, "HEAD", "--", "*.py")
    for line in (setting or "").splitlines():
        parts = line.split(":", 3)
        if len(parts) == 4 and "=" in parts[3]:
            hits.append({"file": parts[1], "line": parts[2], "text": parts[3].strip(), "defines_module": parts[1] in files})
    return hits


def _package(path: str) -> str:
    parts = path.removeprefix("src/").split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else path


def diagnosis_answer(paste: str, github_note: str = "") -> dict | None:
    """{"text", "exact_items"}: the diagnosis sections to put before the detailed evidence, and the GitHub items (e.g. "PR #52449")
    found here as naming this exact problem. None when the paste is not a CMSSW fatal exception with a module or a known message."""
    facts = parse_fatal_paste(paste)
    if not facts or not (facts["module_class"] or facts["family"]):
        return None
    history = find_fix_history(facts)
    cls, label, parameter = facts["module_class"], facts["module_label"], facts["parameter"]
    family_name, fixed, explain = facts["family"] or (None, None, None)
    lines = []
    subject = f"`{cls}`" + (f" (label `{label}`)" if label else "") if cls else "this job"
    title_kind = family_name or f"{facts['category']} exception"
    lines.append(f"## Diagnosis: {title_kind} in {subject}")
    kind = f"**{facts['category']}** failure"
    timing = []
    if facts["phase"]:
        timing.append(f"it failed {facts['phase']}")
    if facts["elapsed"] is not None:
        limit = f" (its time limit was {facts['timeout']} s)" if facts["timeout"] else ""
        timing.append(f"{facts['elapsed']} s after the job's first log line{limit}")
    if facts["timeout"] and facts["elapsed"] is not None and facts["elapsed"] < facts["timeout"] / 10:
        kind += ", not a timeout"
    lines.append(f"- **Type:** {kind}" + (f": {', '.join(timing)}." if timing else "."))
    if parameter:
        lines.append(f"- **Error:** `{facts['message'].split('.')[0].strip()}`, parameter **`{parameter}`**")
    elif facts["message"]:
        lines.append(f"- **Error:** `{facts['message'][:200]}`")
    job = []
    if facts["step"]:
        job.append(facts["step"])
    job += [f"{name}: {', '.join(values)}" for name, values in facts["specs"]]
    if job:
        lines.append(f"- **Job:** {'; '.join(job)}")
    source = _source_lines(facts)
    sets_here = [s for s in source if s["defines_module"]]
    if explain:
        lines.append("\n**What it means:** " + explain.format(
            label=label or "the module", parameter=parameter or "?", cls=cls or "the plugin", record=facts.get("record") or "?",
            param_named=f" (`{parameter}`)" if parameter else ""))
    if sets_here:
        version = None
        try:
            from server.cmssw_source import source_version
            version = source_version()
        except Exception:
            pass
        when = f" (CMSSW source of {version['date']})" if version else ""
        lines.append(f"\nIn the CMSSW code{when}, `{label}` sets it here:")
        lines += [f"- `{s['file']}:{s['line']}`: `{s['text'][:120]}`" for s in sets_here[:3]]
        others = sorted({_package(s["file"]) for s in source if not s["defines_module"]})
        if others:
            lines.append(f"Other configurations that set `{parameter}` too: " + ", ".join(f"`{p}`" for p in others[:8])
                         + (f" and {len(others) - 8} more" if len(others) > 8 else "") + ".")
    if facts["noise"] or facts["timeout"]:
        lines.append("\n**Not the cause** (normal lines in this log):")
        lines += [f"- {n}" for n in facts["noise"]]
        if facts["timeout"] and facts["elapsed"] is not None and facts["elapsed"] < facts["timeout"] / 10:
            lines.append(f"- the `timeout ... {facts['timeout']}` wrapper: the job stopped after {facts['elapsed']} s, far below that limit")

    fixes, origins, recorded = history["fixes"], history["origins"], history["recorded"]
    exact_items = [f"PR #{f['number']}" for f in fixes]
    lines.append("\n## Is it reported?")
    newest_recorded = max(recorded, key=lambda t: (_tag_time(t) or datetime.min.replace(tzinfo=_CET), t), default=None)
    fix_status = None
    if fixes:
        fix = fixes[0]
        merged = fix.get("merged_at")
        state = "merged " + (_cet(merged) or "") if merged else f"state `{fix.get('state') or 'unknown'}`, not merged"
        opened = f"opened by `{fix['author']}`" if fix.get("author") else ""
        if fix.get("created_at"):
            opened += f" on {fix['created_at'][:10]}"
        verdict = "**Yes: a PR names this exact problem" + (" and is merged.**" if merged else ", not merged yet.**")
        lines.append(verdict)
        lines.append(f"- **PR #{fix['number']}** - {fix['title']} ({'; '.join(p for p in (state, opened, fix.get('release') and 'for `' + fix['release'] + '`') if p)})")
        lines.append(f"  - its title names " + " and ".join(f"`{t}`" for t in fix["named"]) + ", from your error")
        if str(fix.get("source_id") or "").startswith("prs:") and not fix.get("fetched_via") and not fix.get("created_at"):
            # merged after the GitHub backup: known from the IB dashboard's PR list, refreshed with every IB import
            lines.append("  - known from the IB dashboard's list of merged PRs (refreshed with every IB import), newer than the GitHub backup: no comments imported for it")
        body = " ".join((fix.get("body") or "").split())
        body = re.sub(r"#{2,}\s*PR description:\s*", "", body)
        if body:
            sentence = re.split(r"#{2,}", body)[0][:400].strip()
            if sentence:
                lines.append(f"  - its description: _\"{sentence}\"_")
        labels = _labels(fix.get("labels"))
        if labels:
            lines.append("  - labels: " + ", ".join(f"`{l}`" for l in labels[:8]))
        if fix.get("files"):
            lines.append("  - files changed: " + ", ".join(f"`{f.removeprefix('src/')}`" for f in fix["files"][:6]))
        elif fix.get("changed_files"):
            lines.append(f"  - {fix['changed_files']} file(s) changed (the file list is not imported for this PR yet)")
        series = _series(newest_recorded or fix.get("release") or "")
        in_ibs = sorted(fix.get("in_ibs") or [], key=lambda t: (_tag_time(t) or datetime.min.replace(tzinfo=_CET), t))
        first_with_fix = _primary_first([t for t in in_ibs if not series or t.startswith(series)])
        if merged and first_with_fix:
            fix_time = _tag_time(first_with_fix)
            after = [t for t in recorded if _tag_time(t) and fix_time and _tag_time(t) >= fix_time and t.startswith(series)]
            if after:
                last = max(after, key=lambda t: _tag_time(t))
                fix_status = f"included from `{first_with_fix}`, but the error is still recorded in `{last}`: the fix may be incomplete"
                lines.append(f"- **In the IBs:** first in `{first_with_fix}`. **The error is still recorded after it**, in {len(after)} build(s), last `{last}`.")
            else:
                fix_status = f"included from `{first_with_fix}`; the error is not recorded in any imported build since"
                lines.append(f"- **In the IBs:** first in `{first_with_fix}`; the error is not recorded in any imported build since.")
        elif merged:
            merged_dt = None
            try:
                merged_dt = datetime.fromisoformat(merged.replace("Z", "+00:00"))
                if merged_dt.tzinfo is None:
                    merged_dt = merged_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
            if newest_recorded and merged_dt and _tag_time(newest_recorded) and _tag_time(newest_recorded) < merged_dt:
                fix_status = f"merged after the newest imported failing build; expected in the first IB built after {_cet(merged)}"
                lines.append(f"- **In the IBs:** not in any imported IB yet. The newest build where this error is recorded, `{newest_recorded}`, "
                             f"was built before the merge, so the error is expected there. The first IB built after {_cet(merged)} should include the fix.")
            else:
                fix_status = "merged; not yet seen in an imported IB"
                lines.append("- **In the IBs:** not linked to any imported IB yet.")
        for other in fixes[1:3]:
            lines.append(f"- Also: **PR #{other['number']}** - {other['title']} ("
                         + (f"merged {_cet(other['merged_at'])}" if other.get("merged_at") else f"state `{other.get('state') or 'unknown'}`") + ")")
    elif history["issues"]:
        issue = history["issues"][0]
        lines.append(f"**An Issue names it:** **Issue #{issue['number']}** - {issue['title']} (state `{issue['state']}`"
                     + (f", opened by `{issue['author']}`" if issue.get("author") else "") + "). No PR found that names it in its title.")
    else:
        lines.append("**No PR or Issue names this problem in its title** in the imported GitHub data. The text matches below may still be worth reading.")
    for issue in history["issues"][:2] if fixes else []:
        lines.append(f"- Issue **#{issue['number']}** - {issue['title']} (state `{issue['state']}`)")

    if origins:
        lines.append("\n## How it started")
        for origin in origins[:2]:
            merged = _cet(origin.get("merged_at"))
            who = f" by `{origin['author']}`" if origin.get("author") else ""
            lines.append(f"PR #{origin['named_by']}'s description names **PR #{origin['number']}** - {origin['title']}{who}"
                         + (f", merged {merged}" if merged else f", state `{origin.get('state') or 'unknown'}`") + ".")
            touched = [f.removeprefix("src/") for f in origin.get("files") or []]
            if cls:
                short = cls.split("::")[-1]
                touched = sorted(touched, key=lambda f: short not in f)
            if touched:
                lines.append("- It changed " + ", ".join(f"`{f}`" for f in touched[:5]) + (f" and {len(touched) - 5} more" if len(touched) > 5 else "") + ".")
            series = _series(newest_recorded or "")
            first_in = _primary_first([t for t in origin.get("in_ibs") or [] if not series or t.startswith(series)])
            first_seen = _primary_first(recorded)
            if first_in and first_seen:
                same = _tag_time(first_in) == _tag_time(first_seen)
                lines.append(f"- It is first in `{first_in}`; this error is first recorded in `{first_seen}`"
                             + (": **the same build**." if same else "."))
            elif first_seen:
                lines.append(f"- This error is first recorded in `{first_seen}`.")
        lines.append("_From the PRs' own descriptions and the build timing: evidence to read, not proof of the cause._")

    candidates = history.get("candidates") or []
    if candidates:
        first_ib = candidates[0]["first_ib"]
        lines.append("\n## What changed in the first failing build")
        lines.append(f"This error is first recorded in `{first_ib}`. Of the {candidates[0]['entered']} PR(s) that entered that build, "
                     f"{len(candidates)} {'touches' if len(candidates) == 1 else 'touch'} the names from your error:")
        for c in candidates:
            who = f" by `{c['author']}`" if c.get("author") else ""
            hits = ", ".join(f"`{h.removeprefix('src/')}`" for h in c["hits"]) if c["hits"] else "its title"
            lines.append(f"- **PR #{c['number']}** - {c['title']}{who}" + (f", merged {_cet(c['merged_at'])}" if c.get("merged_at") else "")
                         + f": {hits}" + (f" ({c['n_files']} files changed)" if c.get("n_files") else ""))
        lines.append("_A PR that entered the first failing build and touches these names is a lead to read, not proof of the cause._")

    lines.append("\n## Summary")
    lines.append("| | |\n|---|---|")
    lines.append(f"| Failure type | {facts['category']}" + (f" ({family_name})" if family_name else "") + " |")
    if cls:
        lines.append(f"| Module | `{cls}`" + (f", label `{label}`" if label else "") + " |")
    if sets_here:
        lines.append(f"| Configuration | `{_package(sets_here[0]['file'])}` |")
    plugin_files = [f for o in origins for f in (o.get("files") or []) if cls and cls.split("::")[-1] in f and f.endswith((".cc", ".h"))]
    if plugin_files:
        lines.append(f"| Plugin package | `{_package(plugin_files[0])}` |")
    if recorded:
        lines.append(f"| Seen in | {len(recorded)} imported IB(s), first `{_primary_first(recorded)}`, last `{newest_recorded}` |")
    if fixed is not None:
        lines.append("| Transient? | " + ("No: this kind of error fails the same way on every run" if fixed else "Can be") + " |")
    if fixes:
        lines.append(f"| Fix | PR #{fixes[0]['number']}: {fix_status or ('merged' if fixes[0].get('merged_at') else 'open')} |")
        lines.append("| New issue needed? | " + ("No: the fix PR covers it" if fixes[0].get("merged_at") and (not fix_status or "may be incomplete" not in fix_status)
                                                else "Not before reading the fix PR: it is not merged yet" if not fixes[0].get("merged_at")
                                                else "Maybe: the error is still there after the fix") + " |")
    else:
        lines.append("| Fix | none found in the imported GitHub data |")
    if fixes and fixes[0].get("merged_at") and fix_status and "expected in the first IB" in fix_status:
        lines.append(f"\n**What to do:** don't open a new Issue or PR: PR #{fixes[0]['number']} addresses this. Builds before "
                     f"{_cet(fixes[0]['merged_at'])} fail as expected; check the first IB after that to confirm it recovers.")
    elif fixes and fix_status and "may be incomplete" in fix_status:
        lines.append(f"\n**What to do:** the error continues after PR #{fixes[0]['number']} reached the IBs: comment on that PR or open an Issue, quoting the newest failing build.")
    elif fixes and not fixes[0].get("merged_at"):
        lines.append(f"\n**What to do:** follow PR #{fixes[0]['number']}: no new report needed while it is open.")
    if github_note:
        lines.append(f"\n_{github_note}._")
    return {"text": "\n".join(lines), "exact_items": exact_items}
