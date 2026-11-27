"""Import the IB test-log scan results (results/, one file per IB, written by merge_test_log_batches.py) into Neo4j.

One results file = one zip of one IB (tag, architecture, GPU/other variant):
unitTestLogs, addOnTests or pyRelValMatrixLogs. It becomes

  (:IB)-[:HAS_TEST_LOG]->(:TestLogRun)        every test/workflow and its status
  (:TestLogRun)-[:HAS_PROBLEM]->(:LogProblem)  one per distinct problem in that zip,
                                               with every test/workflow that hit it
  (:LogProblem)-[:HAS_SIGNATURE]->(:ProblemSignature)   the same problem across IBs
  (:ProblemFirstSeen)                          first time a problem was seen per
                                               test (what "new" means for the shifter)

Same incremental model as build_graph.py: each results file is hashed into an
ImportState node (source_id "testlogs:<path>") and skipped when unchanged. These
sources are never touched by build_graph.py --reconcile (different prefix), and
results/ is kept, so the graph can always be rebuilt from it.

    PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python \\
        ingestion/ingest_test_logs.py --results /home/rao/CHAT-BOT-AI/IB-TEST-LOGS/results
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_graph import file_hash, ib_key, mark_source_imported, source_needs_import  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))
from neo4j_conn import get_driver, setup_schema  # noqa: E402
import ib_test_log_scanner as scanner  # noqa: E402

REMOTE_ROOT = "/data/sdt/buildlogs"
SERIOUS_KINDS = {"fatal_exception", "sanitizer", "ubsan", "crash", "assertion", "uncaught_exception",
                 "python_traceback", "infrastructure_error"}
MAX_EVIDENCE_CHARS = 8000
BATCH = 500


def unit_name(suite: str, unit: dict) -> str:
    """The identity of one test/workflow inside its suite."""
    if suite == "unit":
        return f"{unit.get('package') or ''}/{unit['test']}"
    if suite == "addon":
        return f"{unit['test']}:step{unit['step']}" if unit.get("step") is not None else unit["test"]
    return unit["workflow"]


def owner_name(suite: str, problem: dict) -> str:
    """Which test/workflow (and RelVal step) a problem record belongs to."""
    if suite == "unit":
        return f"{problem.get('package') or ''}/{problem.get('test') or '(outside any test)'}"
    if suite == "addon":
        test = problem.get("test") or "(unknown)"
        return f"{test}:step{problem['step']}" if problem.get("step") is not None else test
    step = f":step{problem['step']}" if problem.get("step") is not None else ""
    return f"{problem.get('workflow')}{step}"


def run_record(data: dict) -> dict:
    source = data["source"]
    suite = data["kind"]
    units = data.get("tests") or data.get("workflows") or []
    passed, failed, other, failed_details = [], [], [], []
    for unit in units:
        name = unit_name(suite, unit)
        status = unit.get("status")
        (passed if status == "passed" else failed if status == "failed" else other).append(name)
        if status == "failed" and suite == "relval":
            steps = " ".join(f"step{k}={v}" for k, v in sorted(unit.get("steps", {}).items(), key=lambda kv: int(kv[0])))
            failed_details.append(f"{name}: {steps}; exit {' '.join(map(str, unit.get('exit_codes') or []))}")
    key = f"{ib_key(source['tag'], source['arch'])}:{suite}:{source.get('variant') or '-'}"
    return dict(
        key=key, suite=suite, ib_tag=source["tag"], arch=source["arch"], run_variant=source.get("variant") or "",
        queue=source.get("queue"), server_path=f"{REMOTE_ROOT}/{source['path']}", zip_size=source.get("size"),
        scanned_at=source.get("scanned_at"), scanner_version=source.get("scanner_version"),
        total=len(units), passed=len(passed), failed=len(failed), other=len(other),
        passed_tests=passed, failed_tests=failed, other_tests=other, failed_details=failed_details,
        # RelVal runs scanned step by step in cms-bot: how many jobs reported back.
        # complete=False: jobs are missing -- still running, or (closed=True) cms-bot stopped waiting
        # after 10 hours: the missing jobs' workflows are unknown.
        # None for unit/addOn and for RelVal zips copied from the server (not known).
        jobs_done=source.get("jobs_done"), jobs_total=source.get("jobs_total"), complete=source.get("complete"),
        closed=source.get("closed"),
        # workflow:stepN that ran but whose log has no good scan (yet): unknown, not clean
        not_scanned=source.get("not_scanned") or [],
    )


def problem_records(data: dict, run_key: str) -> list[dict]:
    """One record per distinct signature in the zip, listing every test or
    workflow that hit it (a RelVal zip has ~23k per-step records but ~250
    distinct problems). The evidence is one example, but the modules, producers,
    names and context lines of every record are kept (search keys: nothing found in
    a second test's block is lost). Results scanned before SCANNER_VERSION 3 have
    no such fields; they are computed here from each record's block."""
    suite = data["kind"]
    grouped: dict[str, dict] = {}
    details: dict[str, dict] = {}
    for p in data["problems"]:
        # older results (their zips are deleted) are re-signed with today's normalization,
        # so the same problem compares equal across every imported build
        signature = scanner.current_signature(p)
        p = dict(p, signature=signature, signature_id=scanner.signature_id(signature))
        if p["kind"] == "crash" and scanner.SIGNAL_RE.search(p.get("message") or ""):
            name = scanner.signal_name(p.get("category"))  # glued thread output, scanner v<6
            p.update(category=name, summary=f"cmsRun crash: fatal system signal ({name})")
        sid = p["signature_id"]
        own = p if "names" in p else scanner.problem_details(p.get("block") or [], p.get("category"))
        scanner.merge_details(details.setdefault(sid, {}), own)
        rec = grouped.get(sid)
        if rec is None:
            evidence = "\n".join(p.get("block") or [])[:MAX_EVIDENCE_CHARS]
            rec = grouped[sid] = dict(
                key=f"{run_key}:{sid}", signature_id=sid, signature=p["signature"], kind=p["kind"],
                category=p.get("category"), severity=p.get("severity"), origin=p.get("origin"),
                serious=p["kind"] in SERIOUS_KINDS, summary=p.get("summary"), message=p.get("message"),
                evidence=evidence, truncated=bool(p.get("truncated")), intentional_hint=p.get("intentional_hint"),
                example_member=p.get("member"), example_line=p.get("line_start"),
                count=0, owners=[], owner_statuses=[], in_passed=False, in_failed=False,
                text="\n".join(x for x in (p.get("summary"), p.get("message"), evidence) if x),
            )
        rec["count"] += p.get("count", 1)
        rec["owners"].append(owner_name(suite, p))
        status = p.get("test_status") or "unknown"
        rec["owner_statuses"].append(status)
        rec["in_passed"] |= status == "passed"
        rec["in_failed"] |= status == "failed"
        if not rec["intentional_hint"] and p.get("intentional_hint"):
            rec["intentional_hint"] = p["intentional_hint"]
    for sid, rec in grouped.items():
        d = details[sid]
        rec.update({field: d.get(field) or [] for field, _ in scanner.DETAIL_FIELDS})
        # lowercased, whitespace-separated: the logProblemNames index (whitespace analyzer)
        # matches each whole name, e.g. "trackstersproducer" or "hgcalgeometry::getnewcorners"
        rec["names_text"] = " ".join(dict.fromkeys(
            n.lower().replace(" ", "_") for n in rec["producers"] + rec["names"]))
        rec["context_keys_text"] = " ".join(k.lower().replace(" ", "_") for k in rec["context_keys"])
        extra = [c for c in rec["contexts"] if c not in rec["text"]]
        if extra:  # context lines of other occurrences, so "contains" and fulltext see them too
            rec["text"] += "\nOther contexts: " + " | ".join(extra)
    return list(grouped.values())


def first_seen_key(suite: str, tag: str, owner: str, signature_id: str) -> str:
    """Per release series: a test of CMSSW_20_1_X must not count as "seen before"
    because the same test hit the same problem in CMSSW_13_2_X."""
    return f"{suite}|{release_series(tag)}|{owner}|{signature_id}"


def release_series(tag: str) -> str:
    """CMSSW_20_1_ASAN_X_2026-10-05-2300 -> CMSSW_20_1 (sub-IBs share their series)."""
    return "_".join(tag.split("_")[:3])


def first_seen_records(data: dict, problems: list[dict]) -> list[dict]:
    """(test, signature) pairs for serious problems and generic error lines:
    the shifter is shown one only the first time it is ever seen for that
    test in that release series. MessageLogger warnings are grouped per IB and
    never 'new'."""
    suite, tag = data["kind"], data["source"]["tag"]
    out = []
    for rec in problems:
        if not (rec["serious"] or rec["kind"] == "generic_error_line"):
            continue
        for owner in sorted(set(rec["owners"])):
            out.append(dict(key=first_seen_key(suite, tag, owner, rec["signature_id"]), suite=suite,
                            series=release_series(tag), owner=owner, signature_id=rec["signature_id"],
                            tag=tag, arch=data["source"]["arch"]))
    return out


def _chunks(items: list, size: int = BATCH):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def import_file(driver, path: Path, root: Path, force: bool = False) -> tuple[int, int] | None:
    source_id = f"testlogs:{path.relative_to(root)}"
    digest = file_hash(path)
    if not source_needs_import(driver, source_id, digest, force):
        return None
    data = json.load(gzip.open(path))
    run = run_record(data)
    problems = problem_records(data, run["key"])
    firsts = first_seen_records(data, problems)
    when = run["ib_tag"][-15:]  # YYYY-MM-DD-HHMM, sortable
    with driver.session() as session:
        # IB nodes come only from the dashboard data (build_graph.py): a log of an IB
        # not imported yet must not create one (it would change what "latest IB" is).
        # link_runs_to_ibs() attaches it once that IB exists.
        current_ib = ib_key(run["ib_tag"], run["arch"])
        session.run("MATCH (n:LogProblem {source_id:$s}) DETACH DELETE n", s=source_id)
        session.run("MATCH (n:TestLogRun {source_id:$s}) DETACH DELETE n", s=source_id)
        session.run(
            "MERGE (run:TestLogRun {key:$run.key}) SET run += $run, run.ib_key=$ib_key, run.source_id=$source_id "
            "WITH run OPTIONAL MATCH (ib:IB {ib_key:$ib_key}) "
            "FOREACH (_ IN CASE WHEN ib IS NULL THEN [] ELSE [1] END | MERGE (ib)-[:HAS_TEST_LOG]->(run))",
            ib_key=current_ib, run=run, source_id=source_id,
        )
        for batch in _chunks(problems):
            session.run(
                "MATCH (run:TestLogRun {key:$run_key}) UNWIND $rows AS row "
                "MERGE (p:LogProblem {key:row.key}) "
                "SET p += row, p.suite=$suite, p.ib_tag=$tag, p.arch=$arch, p.run_variant=$variant, "
                "p.ib_key=$ib_key, p.server_path=$server_path, p.source_id=$source_id "
                "MERGE (run)-[:HAS_PROBLEM]->(p) "
                "MERGE (sig:ProblemSignature {signature_id:row.signature_id}) "
                "ON CREATE SET sig.kind=row.kind, sig.category=row.category, sig.signature=row.signature, "
                "sig.summary=row.summary, sig.first_seen_tag=$tag, sig.first_seen_when=$when "
                "SET sig.last_seen_tag = CASE WHEN coalesce(sig.last_seen_when, '') < $when THEN $tag "
                "ELSE sig.last_seen_tag END, "
                "sig.last_seen_when = CASE WHEN coalesce(sig.last_seen_when, '') < $when THEN $when "
                "ELSE sig.last_seen_when END, "
                "sig.first_seen_tag = CASE WHEN $when < sig.first_seen_when THEN $tag ELSE sig.first_seen_tag END, "
                "sig.first_seen_when = CASE WHEN $when < sig.first_seen_when THEN $when ELSE sig.first_seen_when END "
                "MERGE (p)-[:HAS_SIGNATURE]->(sig)",
                run_key=run["key"], rows=batch, suite=run["suite"], tag=run["ib_tag"], arch=run["arch"],
                variant=run["run_variant"], ib_key=current_ib, server_path=run["server_path"],
                source_id=source_id, when=when,
            )
        for batch in _chunks(firsts, 2000):
            session.run(
                "UNWIND $rows AS row MERGE (f:ProblemFirstSeen {key:row.key}) "
                "ON CREATE SET f += row, f.first_seen_when=$when, f.first_seen_tag=row.tag "
                "SET f.first_seen_tag = CASE WHEN $when < f.first_seen_when THEN row.tag ELSE f.first_seen_tag END, "
                "f.first_seen_when = CASE WHEN $when < f.first_seen_when THEN $when ELSE f.first_seen_when END",
                rows=batch, when=when,
            )
    mark_source_imported(driver, source_id, digest, "testlogs", path)
    return run["total"], len(problems)


def link_runs_to_ibs(driver) -> tuple[int, int]:
    """Attach runs imported before their IB existed; returns (linked now, still without an IB)."""
    with driver.session() as session:
        linked = session.run(
            "MATCH (run:TestLogRun) WHERE NOT ()-[:HAS_TEST_LOG]->(run) "
            "MATCH (ib:IB {ib_key:run.ib_key}) MERGE (ib)-[:HAS_TEST_LOG]->(run) RETURN count(run) AS n").single()["n"]
        orphans = session.run(
            "MATCH (run:TestLogRun) WHERE NOT ()-[:HAS_TEST_LOG]->(run) RETURN count(run) AS n").single()["n"]
    return linked, orphans


def ingest_test_logs(driver, root: Path, force: bool = False) -> int:
    files = sorted(p for kind in ("unit", "addon", "relval") for p in (root / kind).rglob("*.json.gz"))
    imported = skipped = units = problems = 0
    for n, path in enumerate(files, 1):
        result = import_file(driver, path, root, force)
        if result is None:
            skipped += 1
        else:
            imported += 1
            units += result[0]
            problems += result[1]
        if n % 100 == 0:
            print(f"  {n}/{len(files)} files ({imported} imported, {skipped} unchanged)", flush=True)
    print(f"  test-log results: {imported} imported ({units} tests/workflows, {problems} problems), "
          f"{skipped} unchanged")
    linked, orphans = link_runs_to_ibs(driver)
    print(f"  linked {linked} earlier runs to their IB; {orphans} runs have no IB in the dashboard data yet "
          f"(refresh cmsrepo + build_graph.py, then rerun this)")
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", default="/home/rao/CHAT-BOT-AI/IB-TEST-LOGS/results")
    parser.add_argument("--force", action="store_true", help="reimport unchanged results files")
    args = parser.parse_args()
    driver = get_driver()
    setup_schema(driver)
    ingest_test_logs(driver, Path(args.results), args.force)
    driver.close()


if __name__ == "__main__":
    main()
