"""Filtered, grouped lookups for the LLM's tool calls: rankings, set comparisons, date windows, first and last seen. Found 2026-11-05: 40 of 43 investigated questions had no lookup, so the model
wrote its own Cypher over several rounds (median 30 s), got different answers from run to run, counted known errors as failures ("102 failed" where the dashboard's count is 2) and
called a workflow with no row in a partly imported build "fixed". These encode what those queries got wrong: known errors are left out of failures unless asked for, a missing row is
"no result", a partly imported build is flagged. Keep the first line of each docstring complete: it is the tool description the model sees."""
import re
import time
from datetime import datetime, timedelta, timezone



def get_driver():
    # graph.tools sets up the import path, and tests patch it there; it imports this module at its end
    from graph import tools
    return tools.get_driver()

_MAX_ROWS = 60
_USUAL_CACHE: dict = {}
_PARTIAL_SHARE = 0.9  # fewer results than this share of the architecture's usual count: still being imported


def _rows(cypher: str, **params) -> list[dict]:
    driver = get_driver()
    try:
        with driver.session() as session:
            return [dict(row) for row in session.run(cypher, params)]
    finally:
        driver.close()


def _limit(limit, default: int = 30) -> int:
    try:
        return max(1, min(int(limit), _MAX_ROWS))
    except (TypeError, ValueError):
        return default


def _flag(value) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return True if text in ("true", "yes", "1") else False if text in ("false", "no", "0") else None


def _when(value: str | None, end: bool = False) -> str | None:
    """A date as the end of a tag ('YYYY-MM-DD-HHMM'): '2026-11-03' or a whole tag."""
    if not value:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}(?:-\d{4})?", str(value))
    if not match:
        return None
    found = match.group(0)
    return found if len(found) > 10 else found + ("-9999" if end else "-0000")


def _days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d-0000")


def _series_prefix(release: str | None) -> str | None:
    """'20_1', 'CMSSW_20_1', '20.1' or 'CMSSW_20_1_X' -> 'CMSSW_20_1_'."""
    if not release:
        return None
    match = re.search(r"(\d+)[._](\d+)", str(release))
    return f"CMSSW_{match.group(1)}_{match.group(2)}_" if match else None


_IB_WHERE = ("($tag IS NULL OR ib.tag = $tag OR ib.parent_tag = $tag) AND ($prefix IS NULL OR ib.tag STARTS WITH $prefix) AND ($arch IS NULL OR ib.arch CONTAINS $arch) "
             "AND ($since IS NULL OR right(ib.tag, 15) >= $since) AND ($until IS NULL OR right(ib.tag, 15) <= $until) "
             "AND ($variant IS NULL OR ib.variant = $variant) AND (NOT $primary OR ib.parent_tag IS NULL OR ib.parent_tag = ib.tag OR coalesce(ib.variant, '') = '') ")


def _ib_params(release=None, tag=None, arch=None, since=None, until=None, variant=None, primary_only=None, default_days: int | None = 8) -> dict:
    since_value = _when(since)
    if since_value is None and not tag and default_days:
        since_value = _days_ago(default_days)
    return {"prefix": _series_prefix(release), "tag": tag or None, "arch": arch or None, "since": since_value, "until": _when(until, end=True),
            "variant": variant.upper() if variant else None, "primary": bool(_flag(primary_only))}


def _scope(params: dict) -> dict:
    return {k: v for k, v in {"release": params["prefix"], "tag": params["tag"], "arch_contains": params["arch"], "builds_from": params["since"], "builds_to": params["until"],
                              "variant": params["variant"], "primary_only": params["primary"] or None}.items() if v}


def _variant_of(tag: str) -> str:
    match = re.match(r"CMSSW_\d+_\d+_(.*?)_?X_", tag or "")
    return (match.group(1) if match else "") or "primary"


def relval_counts(group_by: str = "workflow", release: str | None = None, tag: str | None = None, arch: str | None = None, since: str | None = None, until: str | None = None,
                  variant: str | None = None, primary_only: str | None = None, status: str = "FAILED", exit_code: str | None = None, workflow_id: str | None = None,
                  include_known_errors: str | None = None, limit: str | None = None) -> dict:
    """Count or rank RelVal results over any builds. group_by: workflow, exit_code, arch, tag, release, variant, status or day. Filters: release ('20_1'), tag (a primary tag includes its sub-IBs), arch (any part: 'el8', 'aarch64', 'gcc14'), since/until ('YYYY-MM-DD'; default last 8 days), variant ('ASAN'), primary_only, status (FAILED, PASSED, DAS_ERROR, ALL), exit_code, workflow_id, include_known_errors (default no, as on the dashboard)."""
    params = _ib_params(release, tag, arch, since, until, variant, primary_only)
    wanted = None if str(status or "FAILED").upper() == "ALL" else str(status or "FAILED").upper()
    code = int(exit_code) if str(exit_code or "").lstrip("-").isdigit() else None
    rows = _rows(f"MATCH (ib:IB) WHERE {_IB_WHERE} MATCH (ib)-[:RAN_RELVAL]->(r:RelVal) "
                 "WHERE ($wanted IS NULL OR r.status = $wanted) AND ($code IS NULL OR r.exitcode = $code) AND ($workflow IS NULL OR r.workflow_id = $workflow) "
                 "RETURN ib.tag AS tag, ib.arch AS arch, r.workflow_id AS workflow, r.name AS name, r.status AS status, r.exitcode AS exit_code, coalesce(r.known_error, false) AS known",
                 **params, wanted=wanted, code=code, workflow=workflow_id or None)
    known = sum(1 for r in rows if r["known"] and r["status"] == "FAILED")
    das = []
    if wanted == "FAILED":
        # what relval_counts leaves out has to be said (found 2026-11-06: "2 RelVals failed" where the dashboard shows 3)
        das = _rows(f"MATCH (ib:IB) WHERE {_IB_WHERE} MATCH (ib)-[:RAN_RELVAL]->(r:RelVal {{status:'DAS_ERROR'}}) "
                    "WHERE ($code IS NULL OR r.exitcode = $code) AND ($workflow IS NULL OR r.workflow_id = $workflow) RETURN r.workflow_id AS workflow",
                    **params, code=code, workflow=workflow_id or None)
    if not _flag(include_known_errors):
        rows = [r for r in rows if not (r["known"] and r["status"] == "FAILED")]
    keys = {"workflow": lambda r: r["workflow"], "exit_code": lambda r: r["exit_code"], "arch": lambda r: r["arch"], "tag": lambda r: r["tag"], "status": lambda r: r["status"],
            "release": lambda r: "_".join(r["tag"].split("_")[:3]), "variant": lambda r: _variant_of(r["tag"]), "day": lambda r: r["tag"][-15:-5]}
    key = keys.get(group_by, keys["workflow"])
    groups: dict = {}
    for row in rows:
        entry = groups.setdefault(key(row), {group_by if group_by in keys else "workflow": key(row), "results": 0, "builds": set(), "archs": set(), "exit_codes": set(), "name": row["name"], "last": ""})
        entry["results"] += 1
        entry["builds"].add(row["tag"])
        entry["archs"].add(row["arch"])
        entry["exit_codes"].add(row["exit_code"])
        entry["last"] = max(entry["last"], row["tag"][-15:])
    ranked = sorted(groups.values(), key=lambda e: -e["results"])
    output = []
    for entry in ranked[:_limit(limit)]:
        item = {**entry, "builds": len(entry["builds"]), "archs": sorted(entry["archs"]), "exit_codes": sorted(c for c in entry["exit_codes"] if c is not None)}
        if group_by != "workflow":
            item.pop("name")
        output.append(item)
    return {"scope": {**_scope(params), "status": wanted or "ALL"}, "total_results": len(rows), "distinct": len(groups), "known_errors_left_out": 0 if _flag(include_known_errors) else known, "rows": output,
            "primary_builds_only": sum(1 for r in rows if _variant_of(r["tag"]) == "primary"),
            "sub_ibs_included": sorted({_variant_of(r["tag"]) for r in rows} - {"primary"}),  # "12 DAS errors on the primary" was 1 plus 11 in its sub-IBs (found 2026-11-06)
            "das_errors_left_out": {"results": len(das), "workflows": sorted({r["workflow"] for r in das})[:10]},
            "note": "Each build on each architecture counts once; known errors are excluded unless asked for."}


def _ib_counts(params: dict) -> list[dict]:
    return _rows(f"MATCH (ib:IB) WHERE {_IB_WHERE} OPTIONAL MATCH (ib)-[:RAN_RELVAL]->(r:RelVal) "
                 "WITH ib, count(r) AS total, sum(CASE WHEN r.status='FAILED' AND NOT coalesce(r.known_error,false) THEN 1 ELSE 0 END) AS failed, "
                 "sum(CASE WHEN r.status='FAILED' AND coalesce(r.known_error,false) THEN 1 ELSE 0 END) AS known, sum(CASE WHEN r.status='DAS_ERROR' THEN 1 ELSE 0 END) AS das "
                 "OPTIONAL MATCH (ib)-[:RAN_UNITTEST]->(u:UnitTest) "
                 "RETURN ib.tag AS tag, ib.arch AS arch, ib.build_status AS build_status, total, failed, known, das, count(u) AS unit_tests, "
                 "sum(CASE WHEN u.status='FAILED' THEN 1 ELSE 0 END) AS unit_failed ORDER BY right(ib.tag, 15)", **params)


def _usual_relval_counts() -> dict:
    """{(tag without its date, arch): the usual number of RelVal results in the last 3 weeks}. From the database, never from the rows being judged: one build compared with itself read "291 of its usual 291, complete" while 954 is usual,
    and "539.0 got fixed" followed from it (found 2026-11-05 by the newest-data check). Cached for 10 minutes."""
    now = time.time()
    if _USUAL_CACHE.get("at", 0) > now - 600:
        return _USUAL_CACHE["counts"]
    since = _days_ago(21)
    rows = _rows("MATCH (ib:IB) WHERE right(ib.tag, 15) >= $since RETURN ib.tag AS tag, ib.arch AS arch, COUNT { (ib)-[:RAN_RELVAL]->() } AS n", since=since)
    seen: dict = {}
    for row in rows:
        if row["n"]:
            seen.setdefault((row["tag"][:-15], row["arch"]), []).append(row["n"])
    # the middle build, not the fullest: 17_1 on el8_aarch64 runs 939, with one build of 1252 it read as "still importing"
    counts = {key: sorted(values)[len(values) // 2] for key, values in seen.items()}
    _USUAL_CACHE.update(at=now, counts=counts)
    return counts


def _with_completeness(rows: list[dict]) -> list[dict]:
    """A build is partial when it has far fewer RelVals than the fullest build of the same flavour and architecture in recent weeks."""
    usual = _usual_relval_counts()
    for row in rows:
        full = max(usual.get((row["tag"][:-15], row["arch"]), 0), row["total"])
        row["complete"] = bool(full) and row["total"] >= full * _PARTIAL_SHARE
        row["usual_relvals"] = full
    return rows


def ib_coverage(release: str | None = None, tag: str | None = None, arch: str | None = None, since: str | None = None, until: str | None = None, variant: str | None = None,
                primary_only: str | None = None, limit: str | None = None) -> dict:
    """Which builds exist and how complete each is: RelVal results against the usual count for that flavour and architecture, failures without known errors, unit tests, build status, complete yes/no. Use for 'still running?', 'missing results', 'is it finished', or before comparing builds. Default last 8 days, newest first."""
    params = _ib_params(release, tag, arch, since, until, variant, primary_only)
    rows = _with_completeness(_ib_counts(params))
    rows.sort(key=lambda r: r["tag"][-15:], reverse=True)
    return {"scope": _scope(params), "builds": len(rows), "rows": rows[:_limit(limit, 40)], "note": "An architecture with no row was not imported for that build. complete=false: results are still arriving, so unknown is not passing."}


def release_health(releases: str | None = None, arch: str | None = None, primary_only: str | None = "true") -> dict:
    """Compare release series using each series' newest COMPLETE build per architecture: RelVal failures (no known errors), results, unit failures, build errors, and which newer builds are still partial. Use for 'which release is healthiest', 'any series with zero failures', 'is 17_1 failing more than 20_1'. releases: optional comma list like '17_1,20_1'."""
    wanted = [p for p in (_series_prefix(r) for r in re.split(r"[,\s]+", releases or "") if r) if p]
    params = _ib_params(None, None, arch, None, None, None, primary_only, default_days=8)
    rows = _with_completeness(_ib_counts(params))
    series: dict = {}
    for row in rows:
        prefix = "_".join(row["tag"].split("_")[:3]) + "_"
        if wanted and prefix not in wanted:
            continue
        entry = series.setdefault(prefix[:-1], {})
        current = entry.get(row["arch"])
        if row["complete"] and (not current or row["tag"][-15:] > current["tag"][-15:]):
            entry[row["arch"]] = row
    output = []
    for name, archs in series.items():
        builds = list(archs.values())
        output.append({"release": name, "relval_failures": sum(b["failed"] for b in builds), "relvals": sum(b["total"] for b in builds), "known_errors": sum(b["known"] for b in builds),
                       "unit_failures": sum(b["unit_failed"] for b in builds), "build_errors": sum(1 for b in builds if b["build_status"] == "error"),
                       "builds": [{"tag": b["tag"], "arch": b["arch"], "failed": b["failed"], "total": b["total"]} for b in builds]})
    output.sort(key=lambda e: (-e["relval_failures"], e["release"]))
    partial = sorted({r["tag"] + " " + r["arch"] for r in rows if not r["complete"] and r["total"]})[:12]
    return {"series": output, "still_partial_not_counted": partial, "note": "Series run different workflows and architectures, so compare with care; failure counts leave out known errors."}


def compare_relvals(tag_a: str, tag_b: str, arch_a: str | None = None, arch_b: str | None = None, include_sub_ibs: str | None = "true") -> dict:
    """Compare RelVal failures of two builds (by date, whichever order they are given in: new_failure_in_later, fixed_in_later, failed_in_both, and failures with NO RESULT on the other side) or of two architectures or flavours of one build (failed_only_in_first, failed_only_in_second, failed_in_both). Use for 'which failures are new', 'what got fixed', 'is aarch64 failing anything amd64 isn't', 'are those the same workflows'. Sub-IBs count when both tags are primary, each flavour against itself; report primary and sub-IB results separately (each row has flavour)."""
    subs = _flag(include_sub_ibs) is not False and _variant_of(tag_a) == "primary" and _variant_of(tag_b) == "primary"  # ROOT6 against the primary means just those two
    across_archs = bool(arch_a and arch_b and arch_a != arch_b)

    def side(tag, arch):
        rows = _rows("MATCH (ib:IB) WHERE (ib.tag = $tag OR ($subs AND ib.parent_tag = $tag)) AND ($arch IS NULL OR ib.arch CONTAINS $arch) MATCH (ib)-[:RAN_RELVAL]->(r:RelVal) "
                     "RETURN r.workflow_id AS workflow, r.name AS name, r.status AS status, r.exitcode AS exit_code, coalesce(r.known_error,false) AS known, ib.arch AS arch, ib.tag AS tag", tag=tag, arch=arch or None, subs=subs)
        results: dict = {}
        cross = _variant_of(tag_a) != _variant_of(tag_b)
        for row in rows:
            flavour = _variant_of(row["tag"])
            row["flavour"] = flavour
            # the same workflow in the same place on both sides: across flavours only the workflow can match, across architectures the flavour must (539.0 in primary and ROOT6 is two results, found 2026-11-05)
            # Never less than that, or results overwrite each other (found 2026-11-06: primary against ASAN keyed on the workflow alone reported 2 failures where there were 12)
            key = (row["workflow"], None if across_archs else row["arch"], None if cross else flavour)
            results[key] = row
        return results, len(rows)

    def failed(results):
        return {k for k, r in results.items() if r["status"] == "FAILED" and not r["known"]}

    def show(results, keys):
        return sorted(({"workflow": results[k]["workflow"], "name": results[k]["name"], "arch": results[k]["arch"], "flavour": results[k]["flavour"], "exit_code": results[k]["exit_code"]} for k in keys),
                      key=lambda r: (r["flavour"] != "primary", r["flavour"], r["workflow"]))[:_MAX_ROWS]

    usual = _usual_relval_counts()

    def still_arriving(side_rows: dict) -> list[str]:
        per_build: dict = {}
        for row in side_rows.values():
            per_build[(row["tag"], row["arch"])] = per_build.get((row["tag"], row["arch"]), 0) + 1
        return sorted(f"{tag} {arch}: {n} of about {usual.get((tag[:-15], arch), n)}" for (tag, arch), n in per_build.items() if n < usual.get((tag[:-15], arch), n) * _PARTIAL_SHARE)

    over_time = tag_a[-15:] != tag_b[-15:]
    if over_time and tag_a[-15:] > tag_b[-15:]:
        tag_a, tag_b, arch_a, arch_b = tag_b, tag_a, arch_b, arch_a  # the earlier build first, however the question put them: a reversed call reported a new failure as a fix (found 2026-11-05)
    a, count_a = side(tag_a, arch_a)
    b, count_b = side(tag_b, arch_b)
    failed_a, failed_b = failed(a), failed(b)
    partial = still_arriving(a) + still_arriving(b)
    names = ("earlier", "later", "fixed_in_later", "new_failure_in_later", "failed_earlier_no_result_later", "failed_later_no_result_earlier") if over_time else ("first", "second", "failed_only_in_first", "failed_only_in_second", "failed_in_first_no_result_in_second", "failed_in_second_no_result_in_first")
    return {"scope": "primary builds and their sub-IBs, each flavour against itself" if subs else "only these builds, no other sub-IBs",
            names[0]: {"tag": tag_a, "arch": arch_a, "results": count_a, "failed": len(failed_a)}, names[1]: {"tag": tag_b, "arch": arch_b, "results": count_b, "failed": len(failed_b)},
            names[2]: show(a, {k for k in failed_a if k in b and b[k]["status"] == "PASSED"}),
            names[3]: show(b, {k for k in failed_b if k in a and a[k]["status"] == "PASSED"}),
            "failed_in_both": show(b, failed_a & failed_b),
            names[4]: show(a, {k for k in failed_a if k not in b}),
            names[5]: show(b, {k for k in failed_b if k not in a}),
            "known_errors": {names[0]: sum(1 for r in a.values() if r["known"]), names[1]: sum(1 for r in b.values() if r["known"])},
            "partial_warning": ("These builds are still being imported, so a workflow with no result there may yet fail: " + "; ".join(partial[:8]) + ". No result is not a pass.") if partial else None}


def workflow_history(workflow_id: str, release: str | None = None, arch: str | None = None, since: str | None = None, variant: str | None = None, primary_only: str | None = None) -> dict:
    """One workflow over time, per release, flavour and architecture: runs, failed, known errors, passed, first and last failure, last pass, current status, exit codes, how many times it flipped, the build where the current failing streak started (failing_since) or where it passed again after its last failure (passing_again_since; last_passed is only the newest pass, not when it recovered), and the failed builds. Use for 'since when', 'when did it stop', 'has it failed before', 'is it new', 'is it in other releases', 'which architectures'. Data goes back about 2 weeks or more; default all imported."""
    params = _ib_params(release, None, arch, since, None, variant, primary_only, default_days=None)
    rows = _rows(f"MATCH (ib:IB) WHERE {_IB_WHERE} MATCH (ib)-[:RAN_RELVAL]->(r:RelVal {{workflow_id:$workflow}}) "
                 "RETURN ib.tag AS tag, ib.arch AS arch, r.name AS name, r.status AS status, r.exitcode AS exit_code, coalesce(r.known_error,false) AS known ORDER BY right(ib.tag, 15)", **params, workflow=workflow_id)
    tracks: dict = {}
    for row in rows:
        release_name = "_".join(row["tag"].split("_")[:3])
        track = tracks.setdefault((release_name, _variant_of(row["tag"]), row["arch"]), {"release": release_name, "flavour": _variant_of(row["tag"]), "arch": row["arch"], "runs": 0, "failed": 0, "known_error": 0, "passed": 0,
                                                                                       "first_failed": None, "last_failed": None, "last_passed": None, "newest": None, "newest_status": None, "exit_codes": set(), "flips": 0, "_previous": None,
                                                                                       "failing_since": None, "passing_again_since": None, "failed_builds": []})
        track["runs"] += 1
        status = "known_error" if row["known"] and row["status"] == "FAILED" else row["status"]
        if status == "FAILED":
            track["failed"] += 1
            track["first_failed"] = track["first_failed"] or row["tag"]
            track["last_failed"] = row["tag"]
            if row["exit_code"] is not None:
                track["exit_codes"].add(row["exit_code"])
        elif status == "PASSED":
            track["passed"] += 1
            track["last_passed"] = row["tag"]
        elif status == "known_error":
            track["known_error"] += 1
        if track["_previous"] in ("FAILED", "PASSED") and status in ("FAILED", "PASSED") and status != track["_previous"]:
            track["flips"] += 1
        # where the current streak began: "when did it stop failing?" was answered with last_passed (the newest pass),
        # one build late (found 2026-11-10, item E3: 1047.0 passed again in 11-09-1100, the AI said 11-09-2300)
        if status == "FAILED":
            track["failed_builds"].append(row["tag"])
            if track["_previous"] != "FAILED":
                track["failing_since"] = row["tag"]
            track["passing_again_since"] = None
        elif status == "PASSED":
            if track["_previous"] != "PASSED" and track["failed"]:
                track["passing_again_since"] = row["tag"]
            track["failing_since"] = None
        track["_previous"] = status
        track["newest"], track["newest_status"] = row["tag"], status
    for track in tracks.values():
        builds = track["failed_builds"]
        track["failed_builds"] = builds if len(builds) <= 12 else builds[:2] + [f"... {len(builds) - 6} more ..."] + builds[-4:]
    output = [{**{k: v for k, v in t.items() if k != "_previous"}, "exit_codes": sorted(t["exit_codes"])} for t in tracks.values()]
    output.sort(key=lambda t: (-t["failed"], t["release"], t["flavour"]))
    return {"workflow": workflow_id, "name": rows[-1]["name"] if rows else None, "tracks_total": len(output), "failing_now": sum(1 for t in output if t["newest_status"] == "FAILED"), "tracks": output[:_MAX_ROWS],
            "builds_from": rows[0]["tag"][-15:] if rows else None, "builds_to": rows[-1]["tag"][-15:] if rows else None}


def flaky_workflows(release: str | None = "20_1", arch: str | None = None, since: str | None = None, primary_only: str | None = "true", limit: str | None = None) -> dict:
    """Workflows that flip between passing and failing: per workflow and architecture, how many flips, failed and passed runs, newest status. Use for 'flip', 'flaky', 'intermittent', 'comes and goes'. Default release 20_1, primary builds, last 14 days; known errors don't count."""
    params = _ib_params(release, None, arch, since, None, None, primary_only, default_days=14)
    rows = _rows(f"MATCH (ib:IB) WHERE {_IB_WHERE} MATCH (ib)-[:RAN_RELVAL]->(r:RelVal) WHERE r.status IN ['FAILED','PASSED'] AND NOT coalesce(r.known_error,false) "
                 "RETURN ib.tag AS tag, ib.arch AS arch, r.workflow_id AS workflow, r.name AS name, r.status AS status ORDER BY right(ib.tag, 15)", **params)
    tracks: dict = {}
    for row in rows:
        track = tracks.setdefault((row["workflow"], row["arch"]), {"workflow": row["workflow"], "name": row["name"], "arch": row["arch"], "flips": 0, "failed": 0, "passed": 0, "newest_status": None})
        if track["newest_status"] and track["newest_status"] != row["status"]:
            track["flips"] += 1
        track["failed" if row["status"] == "FAILED" else "passed"] += 1
        track["newest_status"] = row["status"]
    flaky = sorted((t for t in tracks.values() if t["flips"] >= 2), key=lambda t: (-t["flips"], t["workflow"]))
    return {"scope": _scope(params), "flipped_at_least_twice": len(flaky), "failed_then_passed_or_reverse_once": sum(1 for t in tracks.values() if t["flips"] == 1), "rows": flaky[:_limit(limit, 20)]}


_GITHUB_REACH: dict = {}


def _github_data_reach() -> str | None:
    """Day the imported GitHub backup reaches (newest comment or PR update from it; cached 10 min)."""
    if _GITHUB_REACH and time.time() - _GITHUB_REACH["at"] < 600:
        return _GITHUB_REACH["day"]
    rows = _rows("OPTIONAL MATCH (c:GitHubComment) WITH max(c.created_at) AS tc "
                 "OPTIONAL MATCH (p:PR {repo:'cmssw'}) WHERE p.fetched_via IS NULL "
                 "RETURN tc, max(p.github_updated_at) AS tp")
    days = [str(v)[:10] for v in (rows[0].values() if rows else []) if v]
    _GITHUB_REACH.update(day=max(days) if days else None, at=time.time())
    return _GITHUB_REACH["day"]


def _window_past_reach_note(begin: str | None, until: str | None) -> str:
    """found 2026-11-09: "any PRs opened today?" -> "No PRs were opened today": the GitHub data ends days earlier."""
    reach = _github_data_reach()
    if not reach or (until and until[:10] <= reach) or not (begin or until):
        return ""
    return (f" Our GitHub data (PRs, Issues, comments) reaches {reach}: nothing after that day is imported, so a count or an empty "
            "result for the days after it does not mean none on GitHub; say so. (PRs merged into imported IBs after it come from the IB data.)")


def search_github(kind: str = "both", text: str | None = None, state: str | None = None, author: str | None = None, label: str | None = None, repo: str | None = None, since: str | None = None,
                  until: str | None = None, date_field: str = "created", order: str = "newest", group_by: str | None = None, count_only: str | None = None, title_only: str | None = None, limit: str | None = None) -> dict:
    """Filter imported PRs and Issues by any mix of: kind (PR, Issue, both), whole-word text in title or description, state (open, closed, merged), author, label part ('hlt'), repo (cmssw, cmsdist), since/until ('YYYY-MM-DD'), date_field (created, merged, updated). title_only=true when they say 'in the title' (otherwise text also matches descriptions, so say which you counted). order: newest, oldest, most_comments. group_by: author or label gives a ranking. count_only=true gives only the number. Comments are not searched: for an error text use find_prs_for_error."""
    want_pr = str(kind).lower() in ("pr", "prs", "both")
    want_issue = str(kind).lower() in ("issue", "issues", "both")
    # whole word, but "_" is a separator: found 2026-11-11 (item R3), "open issues mention aarch64" gave 23, the 8 more said "el8_aarch64_gcc12" / "el9_aarch64_gcc14"
    pattern = "(?is).*(?<![A-Za-z0-9])" + re.escape(text.strip()) + "(?:s|es)?(?![A-Za-z0-9]).*" if text and text.strip() else None
    begin = since[:10] if since else None
    # until: a day ('YYYY-MM-DD', whole day) or an exact UTC time ('YYYY-MM-DDTHH:MM', e.g. an IB's build time)
    end = (until[:16] if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d.*", str(until)) else until[:10] + "T99") if until else None
    wanted_state = str(state).lower() if state else None
    found: list[dict] = []
    common = ("WHERE ($pattern IS NULL OR x.title =~ $pattern OR (NOT $title_only AND x.body =~ $pattern)) AND ($repo IS NULL OR x.repo = $repo) "
              "AND ($author IS NULL OR EXISTS { MATCH (x)-[:AUTHORED_BY]->(a:Author) WHERE toLower(a.name) = $author }) "
              "AND ($label IS NULL OR EXISTS { MATCH (x)-[:HAS_LABEL]->(l:Label) WHERE toLower(l.name) CONTAINS $label }) ")
    base = dict(pattern=pattern, title_only=bool(_flag(title_only)), repo=repo.lower() if repo else None, author=str(author).lower() if author else None, label=str(label).lower() if label else None)
    if want_pr and wanted_state != "merged" or (want_pr and wanted_state == "merged"):
        date = {"merged": "x.merged_at", "updated": "x.github_updated_at"}.get(date_field, "x.github_created_at")
        state_filter = {"open": "x.github_state = 'open'", "closed": "x.github_state = 'closed' AND x.merged_at IS NULL", "merged": "x.merged_at IS NOT NULL"}.get(wanted_state, "true")
        found += _rows(f"MATCH (x:PR) {common} AND {state_filter} AND ($begin IS NULL OR {date} >= $begin) AND ($end IS NULL OR {date} <= $end) "
                       f"OPTIONAL MATCH (x)-[:AUTHORED_BY]->(a2:Author) RETURN 'PR' AS kind, x.repo AS repo, x.number AS number, x.title AS title, x.github_state AS state, x.merged_at AS merged_at, "
                       f"x.github_created_at AS created_at, {date} AS date, a2.name AS author, x.labels AS labels, x.fetched_via AS fetched_via", **base, begin=begin, end=end)
    if want_issue and wanted_state != "merged":
        date = {"updated": "x.updated_at"}.get(date_field, "x.created_at")
        state_filter = {"open": "x.state = 'open'", "closed": "x.state = 'closed'"}.get(wanted_state, "true")
        found += _rows(f"MATCH (x:Issue) {common} AND {state_filter} AND ($begin IS NULL OR {date} >= $begin) AND ($end IS NULL OR {date} <= $end) "
                       f"OPTIONAL MATCH (x)-[:AUTHORED_BY]->(a2:Author) RETURN 'Issue' AS kind, x.repo AS repo, x.number AS number, x.title AS title, x.state AS state, null AS merged_at, "
                       f"x.created_at AS created_at, {date} AS date, a2.name AS author, x.labels AS labels, x.fetched_via AS fetched_via", **base, begin=begin, end=end)
    note = "Comments were not searched." + _window_past_reach_note(begin, until)
    if want_pr and (repo or "").lower() == "cmsdist" and wanted_state in (None, "open"):
        # found 2026-11-11 (item R3): "how many PRs are open for cmsdist?" -> "0 open PRs for cmsdist"; the 36 imported cmsdist PRs all came from the IB pages' merged lists
        note += (" cmsdist PRs are imported only when they appear in an IB's merged-PR list, so every imported cmsdist PR is merged: an open cmsdist PR is never "
                 "imported. Do not answer '0 open'; say the open cmsdist PRs are not in the imported data.")
    reach = _github_data_reach() if want_pr else None
    later = sum(1 for r in found if r["kind"] == "PR" and str(r.get("merged_at") or "")[:10] > reach) if reach else 0
    if later:
        # found 2026-11-11 (item R1): "164 PRs merged in November ... the data only reaches 11-06, later merges are not included",
        # while 75 of the 164 were merged after 11-06 (from the IB pages)
        note += (f" {later} of these {len(found)} were merged after {reach}: they are known from the IB pages' merged-PR lists, so the count "
                 f"does include merges after {reach} (only PRs that reached an imported IB; a later merge in no imported IB yet is missing).")
    if _flag(count_only):
        return {"total": len(found), "note": note}
    if group_by in ("author", "label"):
        counts: dict = {}
        for item in found:
            names = [item["author"]] if group_by == "author" else re.findall(r'"([^"]+)"', str(item["labels"]))
            for name in names:
                if name:
                    counts[name] = counts.get(name, 0) + 1
        return {"total": len(found), "rows": [{group_by: k, "items": v} for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:_limit(limit)]], "note": note}
    if order == "most_comments" and found:
        comments: dict = {}
        for label in ("PR", "Issue"):
            # by label and number: without a label this scanned every node (74 s)
            numbers = [{"number": r["number"], "repo": r["repo"]} for r in found if r["kind"] == label][:600]
            for row in _rows(f"UNWIND $numbers AS item MATCH (x:{label} {{number:item.number, repo:item.repo}}) RETURN x.number AS number, COUNT {{ (x)-[:HAS_COMMENT]->() }} AS n", numbers=numbers):
                comments[(label, row["number"])] = row["n"]
        for item in found:
            item["comments"] = comments.get((item["kind"], item["number"]), 0)
        found.sort(key=lambda r: -r.get("comments", 0))
    else:
        found.sort(key=lambda r: r["date"] or "", reverse=order != "oldest")
    for item in found:
        item.pop("labels", None)
        if not item.get("fetched_via"):
            item.pop("fetched_via", None)
    if want_pr and date_field not in ("merged", "updated") and (begin or until):
        # found 2026-11-06: "smuzaffar opened 19 PRs", and 11 cmsdist ones couldn't be counted
        note += " cmsdist PRs have no creation date imported, so a window on the creation date misses them; use date_field='merged' for those."
    return {"total": len(found), "rows": found[:_limit(limit)], "note": note}


def warning_counts(group_by: str = "package", package: str | None = None, tag: str | None = None, arch: str | None = None, checker: str | None = None, limit: str | None = None) -> dict:
    """Rank or compare Clang static-analyzer warnings. group_by: package, subsystem, checker, file or tag (tag shows a trend). package: a package or a whole subsystem ('DQM', 'RecoTracker/MkFit'). Optional tag, arch, checker. Each row has distinct locations and occurrences. Clang logs are staged by hand, so this stops at the newest parsed build it names."""
    prefix = (package or "").strip("/") or None
    rows = _rows("MATCH (ib:IB)-[:REPORTED]->(w:Warning)-[:IN_PACKAGE]->(p:Package) WHERE ($prefix IS NULL OR p.name = $prefix OR p.name STARTS WITH $prefix + '/') AND ($tag IS NULL OR ib.tag = $tag) AND ($arch IS NULL OR ib.arch CONTAINS $arch) "
                 "OPTIONAL MATCH (w)-[:CHECKED_BY]->(c:Checker) WITH ib, w, p, c WHERE $checker IS NULL OR c.name CONTAINS $checker "
                 "RETURN ib.tag AS tag, ib.arch AS arch, p.name AS package, c.name AS checker, w.file AS file, w.count AS count", prefix=prefix, tag=tag or None, arch=arch or None, checker=checker or None)
    keys = {"package": lambda r: r["package"], "subsystem": lambda r: (r["package"] or "").split("/")[0], "checker": lambda r: r["checker"], "file": lambda r: r["file"], "tag": lambda r: f"{r['tag']} {r['arch']}"}
    key = keys.get(group_by, keys["package"])
    groups: dict = {}
    for row in rows:
        entry = groups.setdefault(key(row), {group_by if group_by in keys else "package": key(row), "locations": 0, "occurrences": 0})
        entry["locations"] += 1
        entry["occurrences"] += row["count"] or 0
    ranked = sorted(groups.values(), key=lambda e: -e["occurrences"])
    if group_by == "tag":
        ranked.sort(key=lambda e: e["tag"][-30:])
    newest = max((r["tag"] for r in rows), key=lambda t: t[-15:], default=None)
    return {"scope": {"package": prefix, "tag": tag, "arch": arch, "checker": checker}, "locations": len(rows), "occurrences": sum(r["count"] or 0 for r in rows), "builds": sorted({f"{r['tag']} {r['arch']}" for r in rows}),
            "newest_parsed_build": newest, "rows": ranked[:_limit(limit)], "note": "Without a tag this sums every parsed build, so one location counts once per build it appears in."}


def log_problem_counts(group_by: str = "summary", text: str | None = None, kind: str | None = None, category: str | None = None, suite: str | None = None, release: str | None = None, arch: str | None = None,
                       since: str | None = None, until: str | None = None, in_passed: str | None = None, serious: str | None = None, limit: str | None = None) -> dict:
    """Problems found in the raw test logs (also inside tests shown as PASSED), over any builds. group_by: summary, kind, category, tag, arch, suite. Filters: text (part of the message), kind (fatal_exception, sanitizer, ubsan, crash, assertion, python_traceback, infrastructure_error), category, suite (unit, addon, relval), release, arch, since/until (default last 14 days), in_passed, serious. Each row has first and last build, so use it for 'did X stop', 'is X still happening', 'when did X start'. newest_log_build says how far the logs reach."""
    since_value = _when(since) or _days_ago(14)
    until_value = _when(until, end=True)
    prefix = _series_prefix(release)
    needle = text.lower() if text else None
    rows = _rows("MATCH (p:LogProblem) WHERE right(p.ib_tag, 15) >= $since AND ($until IS NULL OR right(p.ib_tag, 15) <= $until) AND ($prefix IS NULL OR p.ib_tag STARTS WITH $prefix) AND ($arch IS NULL OR p.arch CONTAINS $arch) "
                 "AND ($suite IS NULL OR p.suite = $suite) AND ($kind IS NULL OR p.kind = $kind) AND ($category IS NULL OR toLower(p.category) CONTAINS $category) AND ($in_passed IS NULL OR p.in_passed = $in_passed) AND ($serious IS NULL OR p.serious = $serious) "
                 "AND ($needle IS NULL OR toLower(coalesce(p.summary,'')) CONTAINS $needle OR toLower(coalesce(p.message,'')) CONTAINS $needle OR toLower(coalesce(p.category,'')) CONTAINS $needle) "
                 "RETURN p.ib_tag AS tag, p.arch AS arch, p.suite AS suite, p.kind AS kind, p.category AS category, p.summary AS summary, left(p.message, 400) AS message, p.count AS count, p.owners[0..3] AS owners, p.in_passed AS in_passed, p.serious AS serious",
                 since=since_value, until=until_value, prefix=prefix, arch=arch or None, suite=suite or None, kind=kind or None, category=category.lower() if category else None, in_passed=_flag(in_passed), serious=_flag(serious), needle=needle)
    if needle:
        # whole words: "ucx" is also in host names
        word = re.compile(r"\b" + re.escape(needle) + r"(?:s|es)?\b", re.IGNORECASE)  # "leak" must find "memory leaks" (a false "none", found 2026-11-05)
        rows = [r for r in rows if word.search(" ".join(str(r[k] or "") for k in ("summary", "category", "message")))]
    keys = {"summary": "summary", "kind": "kind", "category": "category", "tag": "tag", "arch": "arch", "suite": "suite"}
    field = keys.get(group_by, "summary")
    groups: dict = {}
    for row in rows:
        when = row["tag"][-15:]
        entry = groups.setdefault(row[field], {field: row[field], "problems": 0, "occurrences": 0, "builds": set(), "archs": set(), "suites": set(), "first": row["tag"], "last": row["tag"], "example_tests": row["owners"]})
        entry["problems"] += 1
        entry["occurrences"] += row["count"] or 0
        entry["builds"].add(row["tag"])
        entry["archs"].add(row["arch"])
        entry["suites"].add(row["suite"])
        if when < entry["first"][-15:]:
            entry["first"] = row["tag"]
        if when > entry["last"][-15:]:
            entry["last"] = row["tag"]
    ranked = sorted(groups.values(), key=lambda e: -e["occurrences"])
    newest = _rows("MATCH (r:TestLogRun) WHERE ($prefix IS NULL OR r.ib_tag STARTS WITH $prefix) AND ($suite IS NULL OR r.suite = $suite) AND ($arch IS NULL OR r.arch CONTAINS $arch) RETURN right(r.ib_tag, 15) AS w ORDER BY w DESC LIMIT 1", prefix=prefix, suite=suite or None, arch=arch or None)
    return {"scope": {"from": since_value, "to": until_value, "release": prefix, "arch": arch, "suite": suite, "text": text, "kind": kind}, "problems": len(rows), "distinct": len(groups), "newest_log_build": newest[0]["w"] if newest else None,
            "rows": [{**e, "builds": len(e["builds"]), "archs": sorted(e["archs"]), "suites": sorted(e["suites"])} for e in ranked[:_limit(limit, 25)]], "note": "A problem being gone after 'last' means anything only if newest_log_build is later."}


def exception_counts(group_by: str = "message", text: str | None = None, category: str | None = None, release: str | None = None, arch: str | None = None, workflow_id: str | None = None, since: str | None = None, until: str | None = None, limit: str | None = None) -> dict:
    """RelVal exceptions recorded from the Elasticsearch export, over any builds. group_by: message, category, workflow, arch, tag or release. Filters: text (part of the message), category, release, arch, workflow_id, since/until (default all imported, from August 2026). Each row: workflows, builds, architectures, first and last build. Use for 'most widespread', 'only on one architecture', 'when did it start', 'which workflows hit it'."""
    prefix = _series_prefix(release)
    rows = _rows("MATCH (e:RelValException) WHERE ($prefix IS NULL OR e.ib_tag STARTS WITH $prefix) AND ($arch IS NULL OR e.arch CONTAINS $arch) AND ($workflow IS NULL OR e.workflow_id = $workflow) AND ($category IS NULL OR toLower(e.category) CONTAINS $category) "
                 "AND ($needle IS NULL OR toLower(coalesce(e.message,'')) CONTAINS $needle OR toLower(coalesce(e.category,'')) CONTAINS $needle) AND ($since IS NULL OR right(e.ib_tag, 15) >= $since) AND ($until IS NULL OR right(e.ib_tag, 15) <= $until) "
                 "RETURN e.ib_tag AS tag, e.arch AS arch, e.workflow_id AS workflow, e.category AS category, left(e.message, 200) AS message",
                 prefix=prefix, arch=arch or None, workflow=workflow_id or None, category=category.lower() if category else None, needle=text.lower() if text else None, since=_when(since), until=_when(until, end=True))
    keys = {"message": lambda r: (r["category"], r["message"]), "category": lambda r: r["category"], "workflow": lambda r: r["workflow"], "arch": lambda r: r["arch"], "tag": lambda r: r["tag"], "release": lambda r: "_".join(r["tag"].split("_")[:3])}
    key = keys.get(group_by, keys["message"])
    groups: dict = {}
    for row in rows:
        entry = groups.setdefault(key(row), {"key": key(row) if group_by != "message" else row["category"], "message": row["message"] if group_by == "message" else None, "records": 0, "workflows": set(), "builds": set(), "archs": set(), "first": row["tag"], "last": row["tag"]})
        entry["records"] += 1
        entry["workflows"].add(row["workflow"])
        entry["builds"].add(row["tag"])
        entry["archs"].add(row["arch"])
        if row["tag"][-15:] < entry["first"][-15:]:
            entry["first"] = row["tag"]
        if row["tag"][-15:] > entry["last"][-15:]:
            entry["last"] = row["tag"]
    ranked = sorted(groups.values(), key=lambda e: -len(e["workflows"]))
    return {"records": len(rows), "distinct": len(groups), "rows": [{**e, "workflows": len(e["workflows"]), "example_workflows": sorted(e["workflows"])[:8], "builds": len(e["builds"]), "archs": sorted(e["archs"])} for e in ranked[:_limit(limit, 20)]],
            "note": "Coverage is what the export has, from August 2026, and it includes known-error workflows. 'first' can't be earlier than the first imported build, so it doesn't show when something really started. A problem with no record after 'last' may also be an architecture that stopped running: check with ib_coverage."}


def _sub_ib_unit_failures(release, arch, begin, until, package) -> list[dict]:
    """What a primary-only look misses (found 2026-11-06: "no failures for FWIO/RNTupleTempTests", which failed 13 times in ROOT6 and ROOT642)."""
    params = _ib_params(release, None, arch, begin, until, None, None, default_days=None)
    rows = _rows(f"MATCH (ib:IB) WHERE {_IB_WHERE} MATCH (ib)-[:RAN_UNITTEST]->(u:UnitTest {{status:'FAILED'}}) WHERE $package IS NULL OR u.name = $package OR u.name STARTS WITH $package + '/' "
                 "RETURN ib.tag AS tag, ib.arch AS arch, u.name AS package", **params, package=(package or "").strip("/") or None)
    counts: dict = {}
    for row in rows:
        if _variant_of(row["tag"]) == "primary":  # a primary build's parent_tag can be its own tag
            continue
        key = (row["package"], _variant_of(row["tag"]), row["arch"])
        counts[key] = counts.get(key, 0) + 1
    return [{"package": k[0], "flavour": k[1], "arch": k[2], "builds_failed": n} for k, n in sorted(counts.items(), key=lambda kv: -kv[1])[:20]]


def unit_test_changes(release: str | None = "20_1", arch: str | None = None, since: str | None = None, until: str | None = None, variant: str | None = None, primary_only: str | None = "true", package: str | None = None, limit: str | None = None) -> dict:
    """Unit-test packages that failed in a window, per architecture: builds_failed (how many builds) with their tags in failed_builds, first and last failed build, whether they passed before the window (new) and whether they pass in the newest build (fixed). Use for 'started failing', 'broke since', 'still failing', 'since when', 'how many builds had X failing'. package: one package or subsystem. Default release 20_1, primary builds, last 7 days."""
    begin = _when(since) or _days_ago(7)
    params = _ib_params(release, None, arch, None, until, variant, primary_only, default_days=None)
    rows = _rows(f"MATCH (ib:IB) WHERE {_IB_WHERE} MATCH (ib)-[:RAN_UNITTEST]->(u:UnitTest) WHERE $package IS NULL OR u.name = $package OR u.name STARTS WITH $package + '/' "
                 "RETURN ib.tag AS tag, ib.arch AS arch, u.name AS package, u.status AS status ORDER BY right(ib.tag, 15)", **params, package=(package or "").strip("/") or None)
    tracks: dict = {}
    for row in rows:
        track = tracks.setdefault((row["package"], row["arch"]), {"package": row["package"], "arch": row["arch"], "failed_before_window": False, "passed_before_window": False, "builds_failed": 0, "failed_builds": [], "first_failed": None, "last_failed": None, "newest_status": None, "newest": None})
        before = row["tag"][-15:] < begin
        if before:
            track["failed_before_window" if row["status"] == "FAILED" else "passed_before_window"] = True
        elif row["status"] == "FAILED":
            track["builds_failed"] += 1  # one per build: a reviewer read "failed_runs: 2" as one build run twice (found 2026-11-05)
            track["failed_builds"].append(row["tag"])
            track["first_failed"] = track["first_failed"] or row["tag"]
            track["last_failed"] = row["tag"]
        track["newest_status"], track["newest"] = row["status"], row["tag"]
    failed = [t for t in tracks.values() if t["builds_failed"]]
    for track in failed:
        track["new_in_window"] = not track.pop("failed_before_window")
    failed.sort(key=lambda t: (not t["new_in_window"], t["newest_status"] != "FAILED", t["package"]))
    elsewhere = _sub_ib_unit_failures(release, arch, begin, until, package) if params["primary"] else []
    return {"scope": {**_scope(params), "window_from": begin}, "also_failed_in_sub_ibs": elsewhere, "packages_failed": len(failed), "new_in_window": sum(1 for t in failed if t["new_in_window"]), "still_failing": sum(1 for t in failed if t["newest_status"] == "FAILED"), "rows": failed[:_limit(limit)],
            "note": "Unit-test results are per package, not per test. Earlier data decides 'new': nothing before the oldest import is known."}


def build_status_list(release: str | None = None, arch: str | None = None, since: str | None = None, until: str | None = None, variant: str | None = None, status: str = "error", limit: str | None = None) -> dict:
    """Build results over a window: which IB and architecture had this compilation status (default error), plus how many builds were checked. Use for 'did the build fail anywhere this week'. Default last 7 days, all releases, sub-IBs included."""
    params = _ib_params(release, None, arch, since, until, variant, None, default_days=7)
    rows = _rows(f"MATCH (ib:IB) WHERE {_IB_WHERE} RETURN ib.tag AS tag, ib.arch AS arch, ib.build_status AS build_status ORDER BY right(ib.tag, 15) DESC", **params)
    wanted = str(status or "error").lower()
    matched = [r for r in rows if str(r["build_status"]).lower() == wanted]
    counts: dict = {}
    for row in rows:
        counts[str(row["build_status"])] = counts.get(str(row["build_status"]), 0) + 1
    return {"scope": _scope(params), "builds_checked": len(rows), "by_status": counts, "matching": len(matched), "rows": matched[:_limit(limit)]}


def partial_builds(tags: list[str]) -> list[dict]:
    """Of these primary tags and their sub-IBs, the builds with far fewer RelVal results than usual: for a note on any answer that names them."""
    if not tags:
        return []
    usual = _usual_relval_counts()
    # only recent builds can still be arriving: an old build has few or none because its files expired before they were imported (a false note on a glossary answer's example, found 2026-11-05)
    recent = [tag for tag in tags if tag[-15:] >= _days_ago(3)]
    if not recent:
        return []
    rows = _rows("MATCH (ib:IB) WHERE ib.tag IN $tags RETURN ib.tag AS tag, ib.arch AS arch, COUNT { (ib)-[:RAN_RELVAL]->() } AS n", tags=recent)
    return [dict(tag=r["tag"], arch=r["arch"], results=r["n"], usual=usual.get((r["tag"][:-15], r["arch"]), r["n"])) for r in rows if r["n"] < usual.get((r["tag"][:-15], r["arch"]), r["n"]) * _PARTIAL_SHARE]


def newest_complete_primary(prefix: str, count: int = 1) -> list[str]:
    """The newest primary tags of a release series, newest first, whose builds on its usual architectures all have their usual RelVal results."""
    usual = _usual_relval_counts()
    rows = _rows("MATCH (ib:IB) WHERE ib.tag STARTS WITH $p AND right(ib.tag, 15) >= $since RETURN ib.tag AS tag, ib.arch AS arch, COUNT { (ib)-[:RAN_RELVAL]->() } AS n", p=f"{prefix}X_", since=_days_ago(14))
    builds: dict = {}
    for row in rows:
        builds.setdefault(row["tag"], []).append(row)
    complete: list[str] = []
    for tag in sorted(builds, key=lambda t: t[-15:], reverse=True):
        counted = [r for r in builds[tag] if usual.get((tag[:-15], r["arch"]))]
        if counted and all(r["n"] >= usual[(tag[:-15], r["arch"])] * _PARTIAL_SHARE for r in counted):
            complete.append(tag)
            if len(complete) == count:
                break
    return complete


def find_prs_merged_between(before_tag: str, after_tag: str, limit: int = 20) -> dict:
    """PRs merged between two builds of a series: in_dashboard_list from the dashboard's own list per build, and by_merge_time from their merge times, which is the only way cmsdist and builds without a list are covered. Timing only, never proof of a cause."""
    from zoneinfo import ZoneInfo
    from graph import tools
    listed = tools.find_prs_merged_between(before_tag, after_tag, limit)
    parts = [re.fullmatch(r"(CMSSW_\d+_\d+)(?:_[A-Za-z0-9]+)?_X_(\d{4}-\d{2}-\d{2}-\d{4})", t or "") for t in (before_tag, after_tag)]
    if not all(parts):
        return {"in_dashboard_list": listed, "by_merge_time": [], "note": "Merge times could not be used: these aren't IB tags."}
    def utc(stamp):
        return datetime.strptime(stamp, "%Y-%m-%d-%H%M").replace(tzinfo=ZoneInfo("Europe/Zurich")).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    start, end = sorted((utc(parts[0].group(2)), utc(parts[1].group(2))))
    shown = {(r["repo"], r["pr_number"]) for r in listed}
    # found 2026-11-06: "0 cmsdist PRs" where 3 were merged, because the list per build has no cmsdist
    timed = _rows("MATCH (p:PR) WHERE p.merged_at >= $start AND p.merged_at < $end AND p.release STARTS WITH $head OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author) "
                  "RETURN p.repo AS repo, p.number AS pr_number, p.title AS title, p.merged_at AS merged_at, a.name AS author ORDER BY p.merged_at DESC", start=start, end=end, head=parts[1].group(1))
    timed = [r for r in timed if (r["repo"], r["pr_number"]) not in shown]
    return {"in_dashboard_list": listed, "by_merge_time": timed[:_limit(limit)], "by_merge_time_total": len(timed),
            "note": f"by_merge_time: merged between {start} and {end} UTC with release {parts[1].group(1)}; a PR merged in the window can still reach the build later. Count both lists."}


def neighbour_build(tag: str, direction: str, arch: str | None = None) -> str | None:
    """The next or previous build of the same series and flavour, from what is imported; on one architecture if named."""
    stamp = tag[-15:]
    head = tag[:-15]
    tags = sorted({r["tag"] for r in _rows("MATCH (ib:IB) WHERE ib.tag STARTS WITH $head AND ($arch IS NULL OR ib.arch = $arch) RETURN DISTINCT ib.tag AS tag", head=head, arch=arch)}, key=lambda x: x[-15:])
    if direction == "after":
        return next((x for x in tags if x[-15:] > stamp), None)
    return next((x for x in reversed(tags) if x[-15:] < stamp), None)


_AUTOMATIC_OPENER = "cmsbuild"
_DELETED_ACCOUNT = "ghost"  # GitHub's name for a deleted account: never a person
_PORT_OF_RE = re.compile(r"(?:automatically\s+ported\s+from\s+\S+\s+|back-?port(?:ed)?\s+(?:of|from)\s+|forward[\s-]port(?:ed)?\s+(?:of|from)\s+)(?:\S+\s+)?#(\d+)", re.IGNORECASE)


def _pr_authors(repo: str, number: int) -> dict | None:
    rows = _rows("MATCH (p:PR {repo:$repo, number:$number}) OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author) "
                 "RETURN p.body AS body, collect(a.name) AS names", repo=repo, number=int(number))
    return rows[0] if rows else None


def pr_origin(number: int, repo: str = "cmssw") -> dict:
    """Who is behind a PR: who opened it on GitHub, and for one the bot cmsbuild opened, the person who opened the original it was ported from. A deleted account shows as ghost, which is not a person."""
    pr = _pr_authors(repo, number)
    if not pr:
        # people say "PR" for any GitHub number; the AI then answered "no data at all" for Issues it has
        # (found 2026-11-09: Issues #52093, #44968, #47275), so say what the number is
        issue = _rows("MATCH (i:Issue {repo:$repo, number:$number}) OPTIONAL MATCH (i)-[:AUTHORED_BY]->(a:Author) "
                      "RETURN i.title AS title, i.state AS state, i.created_at AS created_at, i.closed_at AS closed_at, "
                      "i.labels AS labels, collect(a.name) AS opened_by, COUNT { (i)-[:HAS_COMMENT]->() } AS comments_imported",
                      repo=repo, number=int(number))
        if issue:
            return {"found": False, "repo": repo, "number": int(number), "is_issue": True, "issue": issue[0],
                    "note": f"#{int(number)} is an Issue, not a PR. Use search_issues_by_text or query_graph on (:Issue) for its comments."}
        return {"found": False, "repo": repo, "number": int(number)}
    names = pr["names"]
    # Two names are the GitHub opener plus the owner of the branch, which the older imports recorded as an author too
    opener = next((n for n in names if n in (_AUTOMATIC_OPENER, _DELETED_ACCOUNT)), names[0] if len(names) == 1 else None)
    result = {"found": True, "repo": repo, "number": int(number), "opened_by": opener, "names": names,
              "opened_automatically": opener == _AUTOMATIC_OPENER, "opener_account_deleted": opener == _DELETED_ACCOUNT,
              "branch_owner": next((n for n in names if n != opener), None) if opener else None,
              "ported_from": [], "original_opened_by": None, "original_account_deleted": False}
    if opener != _AUTOMATIC_OPENER:
        return result
    current, seen = pr, {int(number)}
    while len(seen) < 6:  # a port of a port; the text is free-form, so never follow it for long
        link = _PORT_OF_RE.search(current["body"] or "")
        if not link or int(link.group(1)) in seen:
            break
        original = int(link.group(1))
        seen.add(original)
        result["ported_from"].append(original)
        current = _pr_authors(repo, original)
        if not current:
            break  # only say who if the original was imported
        names = current["names"]
        if _DELETED_ACCOUNT in names:
            result["original_account_deleted"] = True
            break
        if _AUTOMATIC_OPENER not in names:
            result["original_opened_by"] = names[0] if len(names) == 1 else None
            break
    return result


_TOUCH_NOTE = ("Touched = the path is in one of the PR's commits, read from the PR's patch in the GitHub backup (a change reverted later in the same PR "
               "still counts). Relevance evidence only: never say a PR caused, broke or fixed something because it touched a file. "
               "cmssw PRs only: cmsdist PRs have no file lists here.")
_UNKNOWN_FILES = "unknown: the backup saved an error page instead of this PR's patch, so which files it changed is not known (never say it changed no files)"


def _pr_opener(names: list) -> str | None:
    """Same rule as pr_origin: one name is the opener; with two, the bot or deleted account among them opened it."""
    names = [n for n in names or [] if n]
    if len(names) == 1:
        return names[0]
    return next((n for n in names if n in (_AUTOMATIC_OPENER, _DELETED_ACCOUNT)), names[0] if names else None)


def _touch_target(path: str) -> tuple[str, str, str]:
    """(kind, exact Cypher value, name) for a path as people write it: a file path, a package or directory, or a bare file name."""
    text = re.sub(r"^(?:\./)?(?:src/)?", "", str(path or "").strip().strip("`'\" ").lstrip("/")).rstrip("/")
    last = text.rsplit("/", 1)[-1]
    is_file = bool(re.search(r"\.[A-Za-z0-9]{1,5}$", last))
    if "/" not in text and is_file:
        return "file name", "/" + text, last
    return ("file" if is_file else "directory"), "src/" + text + ("" if is_file else "/"), last


def _touched_rows(kind: str, value: str, begin, end, date: str, state_filter: str) -> tuple[list[dict], str]:
    match = {"file": "sf.path = $value", "directory": "sf.path STARTS WITH $value", "file name": "sf.path ENDS WITH $value"}
    loose = {"file": "sf.path ENDS WITH '/' + substring($value, 4)", "directory": "sf.path CONTAINS '/' + substring($value, 4)"}
    query = ("MATCH (sf:SourceFile) WHERE {cond} MATCH (p:PR)-[:TOUCHED_FILE]->(sf) "
             f"WITH p, collect(sf.path) AS hit WHERE {state_filter} AND ($begin IS NULL OR {date} >= $begin) AND ($end IS NULL OR {date} <= $end) "
             "OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author) "
             f"RETURN p.repo AS repo, p.number AS number, p.title AS title, p.github_state AS state, p.merged_at AS merged_at, p.github_created_at AS created_at, "
             f"{date} AS date, size(hit) AS files_touched, hit[..3] AS example_files, collect(a.name) AS names ORDER BY {date} DESC, number DESC")
    rows = _rows(query.format(cond=match[kind]), value=value, begin=begin, end=end)
    how = kind
    if not rows and kind in loose and not _rows(f"MATCH (sf:SourceFile) WHERE {match[kind]} RETURN 1 AS x LIMIT 1", value=value):
        # "L1TParticleFlow/interface/layer1_emulator.h" or "XrdAdaptor" without the package in front
        rows = _rows(query.format(cond=loose[kind]), value=value, begin=begin, end=end)
        how = f"{kind} (matched anywhere in the path)"
    return rows, how


def prs_touching_path(path: str | None = None, since: str | None = None, until: str | None = None, date_field: str = "created", state: str | None = None,
                      group_by: str | None = None, limit: str | None = None, release: str | None = None) -> dict:
    """Which cmssw PRs touched a file, a package or directory ('FWCore/Framework'), or a file name ('autoCond.py'), from the PRs' patches: newest first, or group_by=author ('who changes X most'), package or file ('which files/packages were changed most', path optional then). since/until 'YYYY-MM-DD' on date_field created, merged or any (opened or merged in the window) (until may be an exact UTC time 'YYYY-MM-DDTHH:MM'); state open, closed or merged; release 'CMSSW_20_1' keeps one release series. Also says how many PRs in the same window have unknown files. Relevance only, never a cause."""
    begin = since[:10] if since else None
    # until: a day ('YYYY-MM-DD', whole day) or an exact UTC time ('YYYY-MM-DDTHH:MM', e.g. an IB's build time)
    end = (until[:16] if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d.*", str(until)) else until[:10] + "T99") if until else None
    date = {"merged": "p.merged_at", "updated": "p.github_updated_at"}.get(str(date_field).lower(), "p.github_created_at")
    if str(date_field).lower() == "any":
        # opened OR merged in the window: "which PRs touched X last week?" counted only PRs opened then and missed #51861, merged 11-06
        # after weeks open (found live 2026-11-13); the merge date counts when it falls in the window, else the opening date
        date = ("(CASE WHEN p.merged_at IS NOT NULL AND ($begin IS NULL OR p.merged_at >= $begin) AND ($end IS NULL OR p.merged_at <= $end) "
                "THEN p.merged_at ELSE p.github_created_at END)")
    wanted = str(state).lower() if state else None
    state_filter = {"open": "p.github_state = 'open'", "closed": "p.github_state = 'closed' AND p.merged_at IS NULL",
                    "merged": "p.merged_at IS NOT NULL"}.get(wanted, "true")
    if str(date_field).lower() == "merged" and wanted is None:
        state_filter = "p.merged_at IS NOT NULL"
    series = re.match(r"CMSSW_\d+_\d+", str(release or ""))
    if series:  # PRs of one release series ('CMSSW_20_1'; stored as CMSSW_20_1 or CMSSW_20_1_X)
        state_filter = f"({state_filter}) AND (p.release = '{series.group(0)}' OR p.release STARTS WITH '{series.group(0)}_')"
    grouping = str(group_by).lower() if group_by else None
    reach = _github_data_reach()
    note = _TOUCH_NOTE + _window_past_reach_note(begin, until) + (f" GitHub data reaches {reach}." if reach else "")
    if not path:
        if grouping not in ("package", "file"):
            return {"error": "Give a path (file, package or file name), or group_by=package or file for a ranking."}
        rows = _rows(f"MATCH (p:PR {{repo:'cmssw'}}) WHERE {state_filter} AND ($begin IS NULL OR {date} >= $begin) AND ($end IS NULL OR {date} <= $end) "
                     "MATCH (p)-[:TOUCHED_FILE]->(sf:SourceFile) "
                     + ("WITH p, collect(DISTINCT split(sf.path, '/')[1] + '/' + split(sf.path, '/')[2]) AS keys UNWIND keys AS key "
                        if grouping == "package" else "WITH p, sf.path AS key ")
                     + "RETURN key, count(DISTINCT p) AS prs ORDER BY prs DESC, key LIMIT $limit", begin=begin, end=end, limit=_limit(limit, 15))
        scanned = _rows(f"MATCH (p:PR {{repo:'cmssw'}}) WHERE {state_filter} AND ($begin IS NULL OR {date} >= $begin) AND ($end IS NULL OR {date} <= $end) "
                        "RETURN count(p) AS prs, count(CASE WHEN p.patch_status = 'ok' THEN 1 END) AS known", begin=begin, end=end)[0]
        return {"ranking_by": grouping, "rows": [{grouping: (r["key"] or "").removeprefix("src/"), "prs": r["prs"]} for r in rows],
                "prs_in_window": scanned["prs"], "prs_with_known_files": scanned["known"],
                "note": f"Counted over the {scanned['known']} of {scanned['prs']} cmssw PRs in the window whose files are known; the others' files are unknown. " + note}
    kind, value, name = _touch_target(path)
    rows, how = _touched_rows(kind, value, begin, end, date, state_filter)
    for row in rows:
        row["author"] = _pr_opener(row.pop("names"))
        row["example_files"] = [f.removeprefix("src/") for f in row["example_files"]]
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name)
    if kind == "directory" and not re.search(r"[A-Z].*[A-Z]|\d|_", stem):
        stem = ""  # "Geometry", "Framework": too common in titles to point at this package (XrdAdaptor, TruthInfo, Phase2L1GT are not)
    # unknown = an error page saved as the patch, or no patch at all: PRs newer than the GitHub backup come only from the IB pages' merged lists
    unknown = _rows(f"MATCH (p:PR {{repo:'cmssw'}}) WHERE coalesce(p.patch_status, '') <> 'ok' AND {state_filter} AND ($begin IS NULL OR {date} >= $begin) AND ($end IS NULL OR {date} <= $end) "
                    "WITH p, toLower(p.title) CONTAINS toLower($stem) AS names_it ORDER BY p.number DESC "
                    "RETURN count(p) AS total, count(CASE WHEN p.patch_status IS NULL THEN 1 END) AS not_in_backup, [x IN collect(CASE WHEN names_it THEN {number: p.number, title: p.title, merged_at: p.merged_at} END) WHERE x IS NOT NULL][..10] AS naming",
                    stem=stem if len(stem) >= 4 else "\u0000", begin=begin, end=end)[0]
    result = {"path": value.removeprefix("src/"), "matched_as": how, "total_prs": len(rows),
              "prs_with_unknown_files_in_window": unknown["total"], "of_which_newer_than_backup": unknown.get("not_in_backup") or 0,
              "unknown_files_but_title_names_it": unknown["naming"],
              "note": ("The PRs with unknown files may have touched it too: say how many (of_which_newer_than_backup: merged or opened after our GitHub backup, so no patch at all), and name the ones whose title names it (title text only, not a file match). " if unknown["total"] else "") + note}
    if grouping == "author":
        counts: dict = {}
        for row in rows:
            counts[row["author"]] = counts.get(row["author"], 0) + 1
        result["rows"] = [{"author": a, "prs": n} for a, n in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:_limit(limit, 10)]]
        result["note"] += f" Authors are who opened the PR on GitHub; {_AUTOMATIC_OPENER} is the bot opening backports and forward ports."
    elif grouping in ("file", "package"):
        files = _rows(f"MATCH (sf:SourceFile) WHERE sf.path {'STARTS WITH' if kind == 'directory' else '='} $value MATCH (p:PR)-[:TOUCHED_FILE]->(sf) "
                      f"WHERE {state_filter} AND ($begin IS NULL OR {date} >= $begin) AND ($end IS NULL OR {date} <= $end) "
                      "RETURN sf.path AS file, count(DISTINCT p) AS prs ORDER BY prs DESC, file LIMIT $limit", value=value, begin=begin, end=end, limit=_limit(limit, 15))
        result["rows"] = [{"file": r["file"].removeprefix("src/"), "prs": r["prs"]} for r in files]
    else:
        result["prs"] = rows[:_limit(limit, 30)]
    return result


def pr_files(number: int, repo: str = "cmssw", path: str | None = None) -> dict:
    """Which files and packages one cmssw PR touched, from its patch in the GitHub backup; with path ('SimG4Core/Geometry' or 'autoCond.py'), whether it touched that. Files can be unknown (patch not saved): then say unknown, never 'no files'."""
    rows = _rows("MATCH (p:PR {repo:$repo, number:$number}) OPTIONAL MATCH (p)-[:TOUCHED_FILE]->(sf:SourceFile) "
                 "RETURN p.title AS title, p.patch_status AS patch_status, p.github_state AS state, p.merged_at AS merged_at, "
                 "p.changed_files AS dashboard_changed_files, collect(sf.path) AS files", repo=str(repo or "cmssw").lower(), number=int(number))
    reach = _github_data_reach()
    as_of = f"As of our GitHub data (reaches {reach}): commits pushed later are not in it." if reach else ""
    if not rows:
        issue = _rows("MATCH (i:Issue {repo:$repo, number:$number}) RETURN i.title AS title", repo=str(repo or "cmssw").lower(), number=int(number))
        return {"found": False, "number": int(number), "is_issue": bool(issue),
                "note": "This number is an Issue, which has no files." if issue else f"PR {int(number)} is not imported. {as_of}"}
    pr = rows[0]
    base = {"found": True, "repo": repo, "number": int(number), "title": pr["title"], "state": pr["state"], "merged_at": pr["merged_at"]}
    if pr["patch_status"] != "ok":
        why = (_UNKNOWN_FILES if pr["patch_status"] == "unavailable"
               else "unknown: file lists are imported for cmssw PRs only" if str(repo).lower() != "cmssw"
               else f"unknown: this PR is newer than our GitHub backup{' (reaches ' + reach + ')' if reach else ''}, so its patch is not in it (never say it changed no files)")
        return {**base, "files": None, "files_status": why, "dashboard_changed_files": pr["dashboard_changed_files"],
                "note": "Do not guess files from the title; the title and description are text only. " + as_of}
    files = sorted(f.removeprefix("src/") for f in pr["files"])
    packages: dict = {}
    for f in files:
        key = "/".join(f.split("/")[:2])
        packages[key] = packages.get(key, 0) + 1
    result = {**base, "files_status": "known", "files_touched": len(files), "packages": packages, "files": files[:80],
              "files_listed": min(len(files), 80),
              "note": ("Touched = in any of the PR's commits; the dashboard's changed_files (if set) counts the net change, so it can differ. "
                       "Relevance evidence only, never a cause. " + as_of)}
    if pr["dashboard_changed_files"] is not None:
        result["dashboard_changed_files"] = pr["dashboard_changed_files"]
    if path:
        kind, value, _ = _touch_target(path)
        test = {"file": lambda f: "src/" + f == value or f.endswith("/" + value[4:]), "file name": lambda f: f.endswith(value),
                "directory": lambda f: ("src/" + f).startswith(value) or ("/" + value[4:]) in "/" + f}[kind]
        hits = [f for f in files if test(f)]
        result["asked_path"] = value.removeprefix("src/")
        result["touched_asked_path"] = bool(hits)
        result["matching_files"] = hits[:20]
    return result


ANALYTICS_TOOLS = {
    f.__name__: f for f in (pr_files, prs_touching_path, pr_origin,relval_counts, ib_coverage, release_health, compare_relvals, workflow_history, flaky_workflows, search_github, warning_counts,
                                           log_problem_counts, exception_counts, unit_test_changes, build_status_list)}
