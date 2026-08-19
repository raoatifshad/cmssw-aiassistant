"""Re-run evidence verification for pending feedback and expiring Concepts.

A correction is only ever trusted because the graph confirms it, never
because a user asserted it and never because other users agreed -- see
_verify_feedback in server/api_server.py, which this script reuses unchanged.
This process closes the two gaps that leaves in a system with no human
review step:

  - A UserFeedback row that was `pending` at submission time (the graph
    couldn't decide either way) is otherwise a dead end forever, even after
    the graph is later updated with data that could decide it. Re-checking
    it here lets a genuine correction eventually get recognized once the
    graph actually covers it -- still purely on graph evidence.
  - An accepted Concept currently has no way to ever be un-trusted. Without
    this, one wrong automated verification stays a permanent fact forever.
    Concepts are re-verified before expires_at and revoked if they no longer
    hold; a Concept that comes back undecidable (`pending`) gets one short
    grace period before being revoked, rather than either being trusted
    forever or being revoked on a single transient "can't tell" result.

Invoke from cron / a systemd timer / manually -- this process never
schedules itself, matching how run_alerts.py and build_graph.py are invoked
rather than daemonized.

Example crontab entry (once a day):
    0 3 * * * cd /path/to/cmssdt-kg-chatbot && \\
        NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/revalidate_feedback.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from server.api_server import (  # noqa: E402
    CONCEPT_PENDING_GRACE_DAYS, CONCEPT_TTL_DAYS, _maybe_create_concept, _verify_feedback, get_driver,
)


def revalidate_pending_feedback(driver, limit: int) -> tuple[int, int]:
    """Re-check UserFeedback rows still `pending` against current graph
    facts. Returns (checked, resolved)."""
    with driver.session() as session:
        rows = [dict(record) for record in session.run(
            "MATCH (f:UserFeedback {status:'pending'}) "
            "RETURN f.feedback_id AS feedback_id, f.question AS question, f.answer AS answer, "
            "f.verdict AS verdict, f.corrected_answer AS corrected_answer, f.ib_tag AS ib_tag "
            "LIMIT $limit",
            limit=limit,
        )]
        resolved = 0
        for row in rows:
            status, verification, verification_method, verified_ib_tag, verified_ib_tags = _verify_feedback(
                row["question"], row["answer"], row["verdict"], row["corrected_answer"], row["ib_tag"]
            )
            if status == "pending":
                continue
            resolved += 1
            session.run(
                "MATCH (f:UserFeedback {feedback_id:$feedback_id}) "
                "SET f.status=$status, f.verification=$verification, f.verification_method=$verification_method, "
                "f.verified_ib_tag=$verified_ib_tag, f.verified_ib_tags=$verified_ib_tags",
                feedback_id=row["feedback_id"],
                status=status,
                verification=verification,
                verification_method=verification_method,
                verified_ib_tag=verified_ib_tag,
                verified_ib_tags=verified_ib_tags,
            )
            _maybe_create_concept(
                session, row["feedback_id"], row["question"], row["verdict"],
                row["corrected_answer"], status, verification_method,
            )
    return len(rows), resolved


def revalidate_expiring_concepts(driver, limit: int) -> tuple[int, int, int]:
    """Re-check active Concepts past their expires_at against current graph
    facts. Returns (checked, extended, revoked)."""
    with driver.session() as session:
        rows = [dict(record) for record in session.run(
            "MATCH (c:Concept {status:'active'}) WHERE c.expires_at <= datetime() "
            "RETURN c.concept_id AS concept_id, c.question AS question, c.definition AS definition, "
            "coalesce(c.in_pending_grace, false) AS in_pending_grace "
            "LIMIT $limit",
            limit=limit,
        )]
        extended = 0
        revoked = 0
        for row in rows:
            status, _verification, _verification_method, _tag, _tags = _verify_feedback(
                row["question"], "", "incorrect", row["definition"], None
            )
            if status == "accepted":
                extended += 1
                session.run(
                    "MATCH (c:Concept {concept_id:$concept_id}) "
                    "SET c.expires_at=datetime() + duration({days:$ttl_days}), "
                    "c.last_revalidated_at=datetime(), c.in_pending_grace=false",
                    concept_id=row["concept_id"], ttl_days=CONCEPT_TTL_DAYS,
                )
            elif status == "rejected" or row["in_pending_grace"]:
                revoked += 1
                session.run(
                    "MATCH (c:Concept {concept_id:$concept_id}) "
                    "SET c.status='revoked', c.last_revalidated_at=datetime()",
                    concept_id=row["concept_id"],
                )
            else:
                session.run(
                    "MATCH (c:Concept {concept_id:$concept_id}) "
                    "SET c.expires_at=datetime() + duration({days:$grace_days}), "
                    "c.in_pending_grace=true, c.last_revalidated_at=datetime()",
                    concept_id=row["concept_id"], grace_days=CONCEPT_PENDING_GRACE_DAYS,
                )
    return len(rows), extended, revoked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=50,
        help="max pending-feedback rows and max expiring Concepts to re-check per run "
        "(each re-check may cost an LLM call)",
    )
    args = parser.parse_args()

    driver = get_driver()
    try:
        checked, resolved = revalidate_pending_feedback(driver, args.limit)
        print(f"Pending feedback: {checked} re-checked, {resolved} resolved.")
        checked, extended, revoked = revalidate_expiring_concepts(driver, args.limit)
        print(f"Expiring Concepts: {checked} re-checked, {extended} extended, {revoked} revoked.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
