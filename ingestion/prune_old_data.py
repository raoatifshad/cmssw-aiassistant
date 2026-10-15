"""Delete per-build data older than N days (default 90). Dry run unless --apply.

Run by hand or from cron, same standalone pattern as run_alerts.py:
    NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/prune_old_data.py            # what would go
    NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/prune_old_data.py --apply    # delete it

A build's age is the timestamp in its tag (CMSSW_..._X_<YYYY-MM-DD-HHMM>), not when
it was imported. Deleted, for builds older than the cutoff:
  - test logs: TestLogRun + its LogProblem nodes (the biggest grower, ~6.7 KB each)
  - RelValException rows
  - clang Warning nodes, and the IB's HAS_WARNING_IN edges
  - RelVal / UnitTest / WorkflowCommand rows. cms-sw.github.io keeps ~2 weeks of these
    files and build_graph.py --reconcile already drops them once a file is gone, so
    normally there are none this old; kept here so retention still holds if
    --reconcile is not run.
  - with --chat: chat sessions idle since before the cutoff (turns, plan decisions,
    pending clarifications) and RouterShadowLog rows. Off by default: it is
    conversation history, not build data.
Kept, whatever their age:
  - IB nodes (small; build_graph.py recreates them from the release status JSONs anyway)
  - ProblemFirstSeen: what makes a test-log problem "new, first time ever"; deleting it
    would make every old problem look new again
  - ProblemSignature, PR / Issue / comments, Concept, UserFeedback, AlertRule
  - ImportState: a pruned source's state is marked pruned_at instead of deleted, and
    build_graph.source_needs_import never re-imports a pruned source, not even with
    --force. Otherwise the next import would bring the deleted data straight back from
    the files still on disk (IB-TEST-LOGS/results is kept for good). --reconcile still
    removes the state once the file itself is gone.

Neo4j does not shrink its store files after a delete: the space is reused by new data,
so the database stops growing rather than getting smaller (shrinking needs an offline
neo4j-admin database copy).
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))
from neo4j_conn import get_driver  # noqa: E402

BATCH = 5000

# YYYY-MM-DD-HHMM of a tag, sortable as a string; NULL for a tag without "_X_<when>"
_WHEN = "split({tag}, '_X_')[1]"

# (name, label deleted, MATCH...WHERE selecting the old nodes as `n`, source_id to mark
# pruned or None). Each MATCH binds `n`; the cutoff is $cutoff (YYYY-MM-DD-HHMM).
BUILD_TARGETS = [
    ("test-log problems", "LogProblem",
     f"MATCH (run:TestLogRun)-[:HAS_PROBLEM]->(n:LogProblem) WHERE {_WHEN.format(tag='run.ib_tag')} < $cutoff",
     "run.source_id"),
    # after their problems, so a run interrupted halfway is still found on the next run
    ("test-log runs", "TestLogRun",
     f"MATCH (n:TestLogRun) WHERE {_WHEN.format(tag='n.ib_tag')} < $cutoff",
     "n.source_id"),
    ("RelVal exceptions", "RelValException",
     f"MATCH (n:RelValException) WHERE {_WHEN.format(tag='n.ib_tag')} < $cutoff",
     None),  # one ES export holds every build: its ImportState is shared, never pruned
    ("clang warnings", "Warning",
     f"MATCH (ib:IB)-[:REPORTED]->(n:Warning) WHERE {_WHEN.format(tag='ib.tag')} < $cutoff",
     "n.source_id"),
    ("RelVal results", "RelVal",
     f"MATCH (ib:IB)-[:RAN_RELVAL]->(n:RelVal) WHERE {_WHEN.format(tag='ib.tag')} < $cutoff",
     "n.source_id"),
    ("unit-test results", "UnitTest",
     f"MATCH (ib:IB)-[:RAN_UNITTEST]->(n:UnitTest) WHERE {_WHEN.format(tag='ib.tag')} < $cutoff",
     "n.source_id"),
    ("workflow commands", "WorkflowCommand",
     f"MATCH (n:WorkflowCommand) WHERE {_WHEN.format(tag='n.tag')} < $cutoff",
     "n.source_id"),
]

# chat: whole sessions idle since before the cutoff (an ISO timestamp, $cutoff_iso)
_OLD_SESSION = "MATCH (s:ChatSession) WHERE toString(s.updated_at) < $cutoff_iso "
CHAT_TARGETS = [
    ("chat turns", "ChatTurn", _OLD_SESSION + "MATCH (s)-[:HAS_TURN]->(n:ChatTurn)"),
    ("chat plan decisions", "ChatPlanDecision", _OLD_SESSION + "MATCH (s)-[:HAS_PLAN_DECISION]->(n:ChatPlanDecision)"),
    ("pending clarifications", "ChatPendingPlan", _OLD_SESSION + "MATCH (s)-[:HAS_PENDING_PLAN]->(n:ChatPendingPlan)"),
    ("chat sessions", "ChatSession", "MATCH (n:ChatSession) WHERE toString(n.updated_at) < $cutoff_iso"),
    ("router shadow logs", "RouterShadowLog", "MATCH (n:RouterShadowLog) WHERE toString(n.created_at) < $cutoff_iso"),
]


def cutoffs(days: int, now: datetime | None = None) -> tuple[str, str]:
    """(YYYY-MM-DD-HHMM for comparing with tag timestamps, ISO for chat timestamps)."""
    moment = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    return moment.strftime("%Y-%m-%d-%H%M"), moment.strftime("%Y-%m-%dT%H:%M:%S")


def prune(driver, days: int, apply: bool, chat: bool = False, batch: int = BATCH,
          now: datetime | None = None) -> dict[str, int]:
    """Count (and with apply=True delete) everything older than `days`; returns
    {target name: node count}. Sources are marked pruned before their nodes are
    deleted, so an import running meanwhile, or after an interrupted prune, can't
    bring them back; rerunning finishes an interrupted prune."""
    cutoff, cutoff_iso = cutoffs(days, now)
    params = {"cutoff": cutoff, "cutoff_iso": cutoff_iso}
    targets = [(name, label, match, source) for name, label, match, source in BUILD_TARGETS]
    if chat:
        targets += [(name, label, match, None) for name, label, match in CHAT_TARGETS]
    counts: dict[str, int] = {}
    with driver.session() as session:
        for name, label, match, source in targets:
            counts[name] = session.run(f"{match} RETURN count(DISTINCT n) AS n", **params).single()["n"]
            if not apply or not counts[name]:
                continue
            if source:
                session.run(
                    f"{match} WITH DISTINCT {source} AS source_id WHERE source_id IS NOT NULL "
                    "MATCH (state:ImportState {source_id:source_id}) "
                    "SET state.pruned_at = datetime(), state.pruned_before = $cutoff",
                    **params,
                ).consume()
            if label == "Warning":
                # the IB-level "has warnings in package" edges of those same old builds
                session.run(
                    f"MATCH (ib:IB)-[edge:HAS_WARNING_IN]->(:Package) WHERE {_WHEN.format(tag='ib.tag')} < $cutoff "
                    f"CALL {{ WITH edge DELETE edge }} IN TRANSACTIONS OF {int(batch)} ROWS",
                    **params,
                ).consume()
            session.run(
                f"{match} WITH DISTINCT n CALL {{ WITH n DETACH DELETE n }} IN TRANSACTIONS OF {int(batch)} ROWS",
                **params,
            ).consume()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, default=90, help="keep builds newer than this many days (default 90)")
    parser.add_argument("--apply", action="store_true", help="really delete; without it, only report counts")
    parser.add_argument("--chat", action="store_true",
                        help="also delete chat sessions idle since before the cutoff, and router shadow logs")
    parser.add_argument("--batch", type=int, default=BATCH, help="nodes deleted per transaction")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be at least 1")

    cutoff, cutoff_iso = cutoffs(args.days)
    driver = get_driver()
    try:
        counts = prune(driver, args.days, args.apply, args.chat, args.batch)
    finally:
        driver.close()
    verb = "Deleted" if args.apply else "Would delete (dry run, pass --apply to delete)"
    print(f"{verb}, builds before {cutoff}" + (f" / chat idle since before {cutoff_iso}" if args.chat else "") + ":")
    for name, count in counts.items():
        print(f"  {name:24s} {count:>9,}")
    print(f"  {'total':24s} {sum(counts.values()):>9,}")


if __name__ == "__main__":
    main()
