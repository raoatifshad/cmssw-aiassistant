"""Live checks that the imported raw IB test logs (TestLogRun / LogProblem /
ProblemSignature / ProblemFirstSeen, see ingestion/ingest_test_logs.py) are
correct and useful to a shifter. Run it after every fetch + import.

Unlike tests/test_test_log_changes.py (mocked graph, CI), this reads the real
graph and the running server, so it sees what the Shift Console tile and the
chat actually show.

FAIL (exit 1) -- the data or the endpoint is wrong:
  1. Integrity: every TestLogRun is linked to an IB, every LogProblem to a run
     and a signature, no stored signature changes when normalized again (a
     signature that still carries run-specific text makes every build look new).
  2. Endpoint contract (GET /api/test-log-problems over consecutive builds of
     each series): to_imported=false always comes with zero totals and a
     message; the same (track, test, signature) is never both new/returned and
     gone in one answer; totals match the per-architecture lists.
  3. Chat: pasting the log line of a serious problem from the newest covered
     build finds it in the test logs and names the test.

WARN (exit 0) -- the data is right but less useful than it could be:
  - Freshness: the newest build of a series has no logs, so the tile shows
    "no logs yet" for the default window. Run the fetch + import.
  - Fragmentation: one test shows a different signature of the same
    kind/category in most of its builds (normalization gap).
  - Noisy "new": a signature reported as new in many windows (intermittent;
    a shifter learns to ignore it).
  - Junk categories: a traceback category that is not an exception name.
  - Intentional errors without an intentional_hint.

REPORT -- what a shifter gets: per series, the newest covered window's new /
returned / gone items and the serious problems (sanitizers, crashes, fatal
exceptions) inside PASSED tests.

Usage:
    venv/bin/python scripts/validate_test_log_data.py [--base-url http://localhost:8002]
        [--series CMSSW_20_1_X ...] [--windows 20] [--chat-samples 3] [--no-chat]

Needs the API server with DEV_SHIFTER_LOGIN=true and NEO4J_URI/NEO4J_PASSWORD
(.env) pointing at the same graph.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "graph"))
sys.path.insert(0, str(ROOT / "ingestion"))

SERIOUS_KINDS = ("fatal_exception", "uncaught_exception", "crash", "sanitizer", "ubsan",
                 "assertion", "python_traceback", "infrastructure_error")
# A Python exception class name: a (dotted) CamelCase identifier, e.g. IndexError,
# InvalidCmd, Exception, madgraph.MadGraph5Error. Not "P4_gq_qtamvl" or prose.
EXCEPTION_NAME_RE = re.compile(r"^(?:[A-Za-z_]\w*\.)*[A-Z][A-Za-z0-9]*$")
INTENTIONAL_RE = re.compile(r"\bintentional|for testing purposes|\bexpected (?:exception|failure|error)", re.I)


# ---------------------------------------------------------------- pure checks

def signature_leak(signature: str, normalize) -> str | None:
    """The stored detail re-normalized must not change; when it does, run-specific
    text survived into the signature. Returns the re-normalized text, else None.
    message_logger signatures carry no detail (and their category may contain "|")."""
    if signature.startswith("message_logger|") or signature.count("|") < 2:
        return None
    detail = signature.split("|", 2)[2]
    again = normalize(detail)
    return again if again != detail else None


def is_junk_category(kind: str, category: str | None) -> bool:
    """A traceback whose last line is not "SomeError: msg" gets that line as its
    category (Madgraph prints its subprocess name, e.g. "P4_gq_qtamvl")."""
    if kind != "python_traceback" or not category:
        return False
    return not EXCEPTION_NAME_RE.match(category)


def _test_ids(group: dict) -> set[tuple]:
    return {(t.get("variant"), t.get("run_variant"), t.get("suite"), t.get("name"))
            for t in group.get("tests", [])}


def check_changes_payload(payload: dict, label: str) -> list[str]:
    """Contract checks on one /api/test-log-problems answer."""
    errors = []
    coverage, totals = payload.get("coverage") or {}, payload.get("totals") or {}
    archs = payload.get("architectures") or []
    if not coverage.get("to_imported"):
        if any(totals.get(k) for k in totals):
            errors.append(f"{label}: to_imported=false but totals are not zero: {totals}")
        if not coverage.get("message"):
            errors.append(f"{label}: to_imported=false without a message (the tile would show nothing)")
        return errors
    if not coverage.get("runs"):
        errors.append(f"{label}: to_imported=true with runs={coverage.get('runs')}")
    for key, field in (("new_problems", "new"), ("returned_problems", "returned"), ("gone_problems", "gone")):
        listed = sum(len(a.get(field) or []) for a in archs)
        if totals.get(key) is not None and totals[key] != listed:
            errors.append(f"{label}: totals.{key}={totals[key]} but the architectures list {listed}")
    for arch in archs:
        appeared = {}
        for field in ("new", "returned"):
            for group in arch.get(field) or []:
                for test in _test_ids(group):
                    appeared[(group["signature_id"], test)] = field
        for group in arch.get("gone") or []:
            for test in _test_ids(group):
                if (group["signature_id"], test) in appeared:
                    errors.append(
                        f"{label} {arch.get('arch')}: {group.get('summary')!r} in {test[3]} ({test[0]}/{test[1] or '-'}"
                        f"/{test[2]}) is both {appeared[(group['signature_id'], test)]} and gone")
    return errors


def noisy_new_signatures(history: list[dict], min_windows: int) -> list[tuple[str, str, int]]:
    """Signatures reported as new/returned in at least min_windows of the replayed
    windows: (summary, signature_id, windows)."""
    windows, summary = Counter(), {}
    for payload in history:
        seen = set()
        for arch in payload.get("architectures") or []:
            for field in ("new", "returned"):
                for group in arch.get(field) or []:
                    seen.add(group["signature_id"])
                    summary[group["signature_id"]] = group.get("summary") or group["signature_id"]
        windows.update(seen)
    return sorted(((summary[s], s, n) for s, n in windows.items() if n >= min_windows), key=lambda x: -x[2])


# ---------------------------------------------------------------- live layer

class Result:
    def __init__(self):
        self.fails, self.warns = [], []

    def fail(self, msg):
        self.fails.append(msg)
        print(f"  FAIL {msg}")

    def warn(self, msg):
        self.warns.append(msg)
        print(f"  WARN {msg}")


def check_graph(session, result: Result) -> None:
    from ib_test_log_scanner import normalize_text

    print("\n== Graph integrity")
    counts = session.run(
        "MATCH (r:TestLogRun) WITH count(r) AS runs "
        "OPTIONAL MATCH (r:TestLogRun) WHERE NOT (r)<-[]-(:IB) AND NOT (r)-[]->(:IB) "
        "WITH runs, count(r) AS orphan_runs "
        "MATCH (p:LogProblem) WITH runs, orphan_runs, count(p) AS problems "
        "OPTIONAL MATCH (p:LogProblem) WHERE NOT (p)-[:HAS_SIGNATURE]->(:ProblemSignature) "
        "WITH runs, orphan_runs, problems, count(p) AS no_sig "
        "OPTIONAL MATCH (p:LogProblem) WHERE NOT (:TestLogRun)-[:HAS_PROBLEM]->(p) "
        "RETURN runs, orphan_runs, problems, no_sig, count(p) AS no_run").single()
    print(f"  {counts['runs']} runs, {counts['problems']} problems")
    if not counts["runs"]:
        result.fail("no TestLogRun at all: nothing imported")
        return
    for key, what in (("orphan_runs", "TestLogRun not linked to an IB (the IB isn't imported, or the tag differs)"),
                      ("no_sig", "LogProblem without a ProblemSignature"),
                      ("no_run", "LogProblem without a TestLogRun")):
        if counts[key]:
            result.fail(f"{counts[key]} {what}")
    unused = session.run("MATCH (s:ProblemSignature) WHERE NOT (s)<-[:HAS_SIGNATURE]-() RETURN count(s) AS n").single()["n"]
    if unused:
        print(f"  info: {unused} ProblemSignature nodes no problem uses (left over from a re-import; harmless)")

    leaks = []
    for row in session.run("MATCH (s:ProblemSignature)<-[:HAS_SIGNATURE]-() "
                           "RETURN DISTINCT s.signature AS sig"):
        again = signature_leak(row["sig"], normalize_text)
        if again:
            leaks.append((row["sig"], again))
    if leaks:
        result.fail(f"{len(leaks)} signatures change when normalized again (re-import with --force?), e.g. "
                    f"{leaks[0][0][:140]!r}")

    print("\n== Coverage per suite and day")
    rows = session.run(
        "MATCH (r:TestLogRun) WITH r, split(r.ib_tag, '_X_')[1] AS when "
        "RETURN substring(when, 0, 10) AS day, r.suite AS suite, count(*) AS n ORDER BY day").data()
    table = defaultdict(dict)
    for row in rows:
        table[row["day"]][row["suite"]] = row["n"]
    for day, suites in table.items():
        print(f"  {day}  unit {suites.get('unit', 0):4}  addon {suites.get('addon', 0):4}  relval {suites.get('relval', 0):4}")

    print("\n== Fragmentation (signatures of one test seen in a single build only)")
    # One problem split into several signatures shows up as a new signature in each
    # build that is never seen again (new + gone every window). Distinct real problems
    # of one test (a test that throws 11 different exceptions on purpose) recur.
    frag = session.run(
        "MATCH (r:TestLogRun)-[:HAS_PROBLEM]->(p:LogProblem) "
        "WHERE p.kind IN $kinds AND p.in_passed "
        "UNWIND p.owners AS owner "
        "WITH r.suite + '|' + r.arch + '|' + coalesce(r.run_variant,'') + '|' + owner AS test, "
        "     p.kind + '|' + coalesce(p.category,'') AS kc, r.ib_tag AS tag, p.signature_id AS sid "
        "WITH test, kc, sid, count(DISTINCT tag) AS sig_builds, collect(DISTINCT tag) AS tags "
        "WITH test, kc, sum(CASE WHEN sig_builds = 1 THEN 1 ELSE 0 END) AS single, "
        "     reduce(t = [], x IN collect(tags) | t + x) AS all_tags "
        "WITH test, kc, single, size(reduce(u = [], x IN all_tags | CASE WHEN x IN u THEN u ELSE u + x END)) AS builds "
        "WHERE builds >= 4 AND single >= 3 AND single * 2 >= builds "
        "RETURN test, kc, builds, single ORDER BY single DESC LIMIT 10",
        kinds=list(SERIOUS_KINDS)).data()
    for row in frag:
        result.warn(f"fragmented: {row['test']} {row['kc']}: {row['single']} one-build-only signatures "
                    f"over {row['builds']} builds")
    if not frag:
        print("  none")

    print("\n== Junk traceback categories")
    junk = [r for r in session.run(
        "MATCH (p:LogProblem {kind:'python_traceback'}) RETURN p.category AS category, count(*) AS n").data()
        if is_junk_category("python_traceback", r["category"])]
    for row in junk:
        result.warn(f"traceback category is not an exception name: {row['category'][:70]!r} ({row['n']} problems)")
    if not junk:
        print("  none")

    print("\n== Intentional errors without intentional_hint")
    intentional = session.run(
        "MATCH (p:LogProblem) WHERE p.kind IN $kinds AND p.intentional_hint IS NULL "
        "AND (p.message =~ '(?is).*(intentional|for testing purposes).*') "
        "RETURN p.category AS category, p.message AS message, count(*) AS n ORDER BY n DESC LIMIT 5",
        kinds=list(SERIOUS_KINDS)).data()
    for row in intentional:
        result.warn(f"no intentional_hint for {row['category']!r}: {row['message'][:80]!r} ({row['n']} problems)")
    if not intentional:
        print("  none")


def dev_login(client: httpx.Client, username: str) -> None:
    resp = client.get(f"/shift/dev-login?username={username}", follow_redirects=False)
    if resp.status_code >= 400:
        raise SystemExit(f"dev-login failed ({resp.status_code}): is DEV_SHIFTER_LOGIN=true and "
                         f"{username!r} in SHIFTER_CERN_USERNAMES?")


def series_windows(client: httpx.Client, series: str) -> list[str]:
    windows = client.get("/api/shift-summary/windows").json()["windows"]
    return sorted({w["release_name"] for w in windows if w["release_name"].startswith(series + "_")})


def print_shifter_view(payload: dict, label: str) -> None:
    print(f"\n  What a shifter sees for {label}:")
    t = payload["totals"]
    print(f"    new {t['new_problems']}  returned {t['returned_problems']}  gone {t['gone_problems']}  "
          f"still present {t['still_present']}")
    for arch in payload["architectures"]:
        for field in ("new", "returned", "gone"):
            for group in arch.get(field) or []:
                tests = group.get("tests", [])
                names = ", ".join(sorted({f"{x['name']} ({x['variant']}{'/' + x['run_variant'] if x['run_variant'] else ''})"
                                          for x in tests}))
                hint = "  [intentional?]" if group.get("intentional_hint") else ""
                print(f"    {field:8} {arch['arch']:16} {group.get('kind', ''):17} "
                      f"{(group.get('summary') or '')[:70]}{hint}\n{'':36}in {names[:160]}")


def check_endpoint(client: httpx.Client, series: str, max_windows: int, result: Result) -> dict | None:
    """Replays the tile over consecutive builds of a series. Returns the newest covered payload."""
    tags = series_windows(client, series)
    if len(tags) < 2:
        result.warn(f"{series}: fewer than 2 windows, nothing to compare")
        return None
    pairs = list(zip(tags, tags[1:]))[-max_windows:]
    print(f"\n== {series}: {len(pairs)} consecutive windows ({pairs[0][1][-15:]} .. {pairs[-1][1][-15:]})")
    history, newest_covered = [], None
    for from_tag, to_tag in pairs:
        resp = client.get("/api/test-log-problems", params={"from": from_tag, "to": to_tag})
        if resp.status_code != 200:
            result.fail(f"{to_tag}: HTTP {resp.status_code} {resp.text[:120]}")
            continue
        payload = resp.json()
        for error in check_changes_payload(payload, to_tag):
            result.fail(error)
        cov, t = payload["coverage"], payload["totals"]
        if cov.get("to_imported"):
            history.append(payload)
            newest_covered = (to_tag, payload)
            print(f"  {to_tag[-15:]}  runs {cov['runs']:3}  new {t['new_problems']:2}  returned {t['returned_problems']:2}  "
                  f"gone {t['gone_problems']:2}  still {t['still_present']}")
    latest = pairs[-1][1]
    if newest_covered is None:
        result.warn(f"{series}: no window has imported logs at all")
        return None
    if newest_covered[0] != latest:
        result.warn(f"{series}: newest build {latest} has no imported logs (newest with logs: "
                    f"{newest_covered[0]}); the tile's default window says 'no logs yet' -- run the fetch + import")
    for summary, sid, n in noisy_new_signatures(history, max(3, len(history) // 3)):
        result.warn(f"{series}: {summary[:70]!r} is new/returned in {n} of {len(history)} windows (intermittent noise)")
    print_shifter_view(newest_covered[1], newest_covered[0])
    return newest_covered[1]


def check_serious_in_passed(session, release: str, when: str) -> None:
    """The serious problems inside PASSED tests of one build (primary IB + its sub-IBs),
    one line per (sub-IB, kind, signature) however many archs/GPU variants repeat it."""
    rows = session.run(
        "MATCH (r:TestLogRun)-[:HAS_PROBLEM]->(p:LogProblem) "
        "WHERE r.ib_tag STARTS WITH $prefix AND r.ib_tag ENDS WITH $when AND p.in_passed "
        "AND p.kind IN ['sanitizer','ubsan','crash','fatal_exception','assertion','uncaught_exception'] "
        "AND p.intentional_hint IS NULL "
        "RETURN r.ib_tag AS tag, r.arch AS arch, r.suite AS suite, coalesce(r.run_variant, '') AS run_variant, "
        "p.kind AS kind, p.signature_id AS sid, p.summary AS summary, p.owners AS owners",
        prefix=release + "_", when=when).data()
    groups = {}
    for row in rows:
        variant = row["tag"][len(release) + 1:-len(when) - 3] or "primary"
        group = groups.setdefault((variant, row["kind"], row["sid"]), dict(
            summary=row["summary"], suite=row["suite"], archs=set(), runs=set(), owners=set()))
        group["archs"].add(row["arch"])
        group["runs"].add(row["run_variant"] or "-")
        group["owners"].update(row["owners"] or [])
    print(f"\n  Serious problems inside PASSED tests of {release}_*_{when} (no intentional hint): {len(groups)}")
    for (variant, kind, _), g in sorted(groups.items(), key=lambda x: (x[0][1] not in ("sanitizer", "ubsan"), x[0]))[:25]:
        owners = sorted(g["owners"])
        print(f"    {variant:10} {g['suite']:6} {kind:16} {(g['summary'] or '')[:70]}\n"
              f"{'':34}{len(owners)} test(s), e.g. {owners[0] if owners else '?'}; {', '.join(sorted(g['archs']))}; "
              f"runs {', '.join(sorted(g['runs']))[:60]}")
    if len(groups) > 25:
        print(f"    ... {len(groups) - 25} more")


def check_chat(client: httpx.Client, session, samples: int, result: Result) -> None:
    print("\n== Chat: paste a stored problem's log line, expect it found")
    rows = session.run(
        "MATCH (r:TestLogRun)-[:HAS_PROBLEM]->(p:LogProblem) "
        "WHERE p.in_passed AND p.kind IN ['sanitizer','ubsan','fatal_exception','crash'] "
        "AND p.intentional_hint IS NULL AND size(coalesce(p.evidence, '')) > 30 "
        "WITH p.kind AS kind, p ORDER BY split(p.ib_tag, '_X_')[1] DESC, p.ib_tag "
        "WITH kind, collect(p)[0] AS p "
        "RETURN kind, p.ib_tag AS tag, p.evidence AS evidence, p.owners AS owners LIMIT $n",
        n=samples).data()
    for row in rows:
        # what a shifter would copy out of the log: the first lines of the evidence, up to and
        # including the exception message when there is one (3 lines cut a Fatal Exception
        # block before its message, leaving only the category to match on)
        lines = row["evidence"].splitlines()
        upto = next((i + 2 for i, l in enumerate(lines) if l.strip().startswith("Exception Message")), 3)
        row["line"] = "\n".join(lines[:max(3, upto)])[:600]
        question = f"In {row['tag']} this test passed but its log shows:\n{row['line']}"
        try:
            resp = client.post("/api/chat", json={"question": question,
                                                  "session_id": f"validate-test-logs-{row['kind']}"}, timeout=120)
            answer = resp.json().get("answer") or resp.json().get("response") or ""
        except Exception as exc:  # noqa: BLE001
            result.fail(f"chat ({row['kind']}): {exc}")
            continue
        # exact / contains / related headings of _test_log_problem_answer
        found = any(h in answer for h in ("**This exact error is in the imported IB test logs**",
                                          "**Found in the imported IB test logs**"))
        named = any(owner.split(":")[0] in answer for owner in row["owners"])
        status = "ok" if found and named else "FAIL"
        print(f"  {status:4} {row['kind']:15} {row['line'][:80]!r}")
        if not found:
            result.fail(f"chat did not find a stored {row['kind']} line from {row['tag']}: {answer[:200]!r}")
        elif not named:
            result.fail(f"chat found the {row['kind']} line but named none of {row['owners'][:3]}")


def run(args) -> int:
    from neo4j_conn import get_driver

    result = Result()
    driver = get_driver()
    with driver.session() as session, httpx.Client(base_url=args.base_url, timeout=120) as client:
        check_graph(session, result)
        dev_login(client, args.username)
        for series in args.series:
            newest = check_endpoint(client, series, args.windows, result)
            if newest:
                check_serious_in_passed(session, series[:-2], newest["to"][-15:])
        if not args.no_chat:
            check_chat(client, session, args.chat_samples, result)
    driver.close()
    print(f"\n{len(result.fails)} failures, {len(result.warns)} warnings")
    return 1 if result.fails else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--username", default="mrao", help="Must be in SHIFTER_CERN_USERNAMES.")
    parser.add_argument("--series", nargs="+", default=["CMSSW_20_1_X", "CMSSW_20_0_X", "CMSSW_16_1_X"])
    parser.add_argument("--windows", type=int, default=20, help="Consecutive windows to replay per series.")
    parser.add_argument("--chat-samples", type=int, default=3)
    parser.add_argument("--no-chat", action="store_true")
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
