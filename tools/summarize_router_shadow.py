"""Summarize accumulated RouterShadowLog data: how often an independent LLM
classification agrees with the deterministic router, and where it doesn't.

This reads logging-only data (recorded when ENABLE_ROUTER_SHADOW=true) and
never changes how any question gets answered -- the LLM classification it
summarizes was never used to answer anything, only logged for comparison.
This is the evidence to look at before ever considering promoting an LLM
router to primary for capability *selection* (never for the answer itself,
which must stay Cypher-only).

Invoke from cron / a systemd timer / manually -- this process never
schedules itself, matching how run_alerts.py and build_graph.py are invoked
rather than daemonized.

Example:
    CHATBOT_DIR=/path/to/chatbot NEO4J_URI=bolt://localhost:7690 python tools/summarize_router_shadow.py --days 7
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ.get("CHATBOT_DIR", ".")) / "graph"))   # the chatbot checkout
from neo4j_conn import get_driver  # noqa: E402


def _where(days: int | None, extra: str | None = None) -> str:
    conditions = []
    if days is not None:
        conditions.append("log.created_at >= datetime() - duration({days: $days})")
    if extra:
        conditions.append(extra)
    return f"WHERE {' AND '.join(conditions)}" if conditions else ""


def summarize(driver, days: int | None, sample_size: int) -> None:
    with driver.session() as session:
        totals = session.run(
            f"MATCH (log:RouterShadowLog) {_where(days)} "
            "RETURN count(log) AS total, sum(CASE WHEN log.agree THEN 1 ELSE 0 END) AS agreeing",
            days=days,
        ).single()
        total = totals["total"] if totals else 0
        if not total:
            print(
                "No RouterShadowLog entries recorded yet. Requires ENABLE_ROUTER_SHADOW=true "
                "and real chat traffic to accumulate data -- nothing to summarize until then."
            )
            return
        agreeing = totals["agreeing"] or 0
        print(f"Total logged questions: {total}")
        print(f"Agreement rate: {agreeing}/{total} ({agreeing / total:.0%})")
        print()

        print("By deterministic outcome (capability, or plan kind when no capability matched):")
        by_capability = session.run(
            f"MATCH (log:RouterShadowLog) {_where(days)} "
            "RETURN coalesce(log.deterministic_capability, log.deterministic_kind) AS outcome, "
            "count(log) AS total, sum(CASE WHEN log.agree THEN 1 ELSE 0 END) AS agreeing "
            "ORDER BY total DESC",
            days=days,
        )
        for row in by_capability:
            print(f"  {row['outcome'] or '(none)'}: {row['agreeing']}/{row['total']} agree")
        print()

        print(f"Top disagreement patterns (grouped by deterministic->llm outcome, up to {sample_size}):")
        print("Triage these first -- a repeated pattern is a real routing gap; a one-off may just be a")
        print("garbled or ambiguous question not worth chasing.")
        patterns = session.run(
            f"MATCH (log:RouterShadowLog) {_where(days, 'NOT log.agree')} "
            "WITH coalesce(log.deterministic_capability, log.deterministic_kind) AS deterministic, "
            "log.llm_capability AS llm, count(log) AS total, "
            "collect(log.question)[0] AS sample_question, avg(log.llm_confidence) AS avg_confidence "
            "RETURN deterministic, llm, total, sample_question, avg_confidence "
            "ORDER BY total DESC LIMIT $limit",
            days=days, limit=sample_size,
        )
        pattern_rows = list(patterns)
        if not pattern_rows:
            print("  (none)")
        for row in pattern_rows:
            confidence = f"{row['avg_confidence']:.2f}" if row["avg_confidence"] is not None else "?"
            print(
                f"  x{row['total']}: deterministic={row['deterministic'] or '(none)'}  "
                f"llm={row['llm']}  avg_llm_confidence={confidence}"
            )
            print(f"       e.g. {row['sample_question']!r}")
        print()

        print(f"Most recent disagreements (up to {sample_size}, for spot-checking the latest miss):")
        disagreements = session.run(
            f"MATCH (log:RouterShadowLog) {_where(days, 'NOT log.agree')} "
            "RETURN log.question AS question, log.deterministic_capability AS deterministic, "
            "log.deterministic_kind AS kind, log.llm_capability AS llm, log.llm_confidence AS confidence "
            "ORDER BY log.created_at DESC LIMIT $limit",
            days=days, limit=sample_size,
        )
        rows = list(disagreements)
        if not rows:
            print("  (none)")
        for row in rows:
            deterministic = row["deterministic"] or f"({row['kind']})"
            print(f"  Q: {row['question']!r}")
            print(f"     deterministic={deterministic}  llm={row['llm']}  llm_confidence={row['confidence']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None, help="only consider entries from the last N days")
    parser.add_argument("--sample-size", type=int, default=20, help="max disagreement examples to print")
    args = parser.parse_args()

    driver = get_driver()
    try:
        summarize(driver, args.days, args.sample_size)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
