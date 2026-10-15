"""Alert rules: reuse the same capability library chat already reuses, on a
schedule instead of a chat turn.

By design this module is standalone (no FastAPI dependency) so it can be
invoked by cron/systemd-timer/manual run via ingestion/run_alerts.py, the same
way build_graph.py is — this codebase never runs its own in-process
scheduler.

Notification delivery is pluggable (see Notifier below). Only a log-based
notifier ships today, since no real channel credentials (Slack/email/etc.)
have been provided; adding a real channel later is a one-class addition, not
a change to rule storage or evaluation.
"""
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).parent))
from neo4j_conn import get_driver  # noqa: E402
from tools import (  # noqa: E402
    get_build_failures, get_failing_relvals, get_failing_unittests, get_relval_recurrence, list_ib_tags,
    find_relval_regressions, find_unittest_regressions, find_build_regressions,
)


RULE_TYPES = {
    "relval_failure_count", "relval_recurrence", "latest_ib_stalled",
    "unittest_failure_count", "build_failure_count",
    "relval_regression", "unittest_regression", "build_regression",
}


def create_rule(
    name: str, rule_type: str, params: dict, threshold: float, channel: str = "log", owner: str | None = None,
) -> str:
    if rule_type not in RULE_TYPES:
        raise ValueError(f"Unknown rule type {rule_type!r}; expected one of {sorted(RULE_TYPES)}")
    rule_id = str(uuid.uuid4())
    driver = get_driver()
    try:
        with driver.session() as session:
            session.run(
                "CREATE (rule:AlertRule {"
                "rule_id:$rule_id, name:$name, rule_type:$rule_type, params:$params, "
                "threshold:$threshold, channel:$channel, owner:$owner, active:true, "
                "created_at:datetime()})",
                rule_id=rule_id, name=name, rule_type=rule_type, params=json.dumps(params),
                threshold=threshold, channel=channel, owner=owner,
            )
    finally:
        driver.close()
    return rule_id


def list_rules(active_only: bool = True) -> list[dict]:
    driver = get_driver()
    try:
        with driver.session() as session:
            query = "MATCH (rule:AlertRule) "
            if active_only:
                query += "WHERE rule.active = true "
            query += (
                "RETURN rule.rule_id AS rule_id, rule.name AS name, rule.rule_type AS rule_type, "
                "rule.params AS params, rule.threshold AS threshold, rule.channel AS channel, "
                "rule.owner AS owner, rule.active AS active"
            )
            rows = [dict(record) for record in session.run(query)]
    finally:
        driver.close()
    for row in rows:
        row["params"] = json.loads(row["params"])
    return rows


def deactivate_rule(rule_id: str) -> bool:
    driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (rule:AlertRule {rule_id:$rule_id}) SET rule.active = false RETURN rule.rule_id AS rule_id",
                rule_id=rule_id,
            ).single()
    finally:
        driver.close()
    return result is not None


def _check_relval_failure_count(params: dict, threshold: float) -> tuple[bool, str, dict]:
    tag = params["tag"]
    arch = params.get("arch")
    failing = get_failing_relvals(tag, arch=arch, limit=None)
    count = len(failing)
    triggered = count >= threshold
    message = f"`{tag}`{f' on `{arch}`' if arch else ''} has {count} failing RelVal(s) (threshold {threshold:g})."
    return triggered, message, {"count": count}


def _check_relval_recurrence(params: dict, threshold: float) -> tuple[bool, str, dict]:
    workflow_id = params["workflow_id"]
    recurrence = get_relval_recurrence(workflow_id)
    total = recurrence.get("total_appearances", 0)
    failed = recurrence.get("failed_appearances", 0)
    rate = (failed / total) if total else 0.0
    triggered = total > 0 and rate >= threshold
    message = (
        f"Workflow `{workflow_id}` has a {rate:.0%} failure rate ({failed}/{total}), "
        f"at or above the {threshold:.0%} threshold."
    )
    return triggered, message, {"rate": rate, "failed": failed, "total": total}


def _check_unittest_failure_count(params: dict, threshold: float) -> tuple[bool, str, dict]:
    tag = params["tag"]
    arch = params.get("arch")
    failing = get_failing_unittests(tag, arch=arch, limit=None)
    count = len(failing)
    triggered = count >= threshold
    message = (
        f"`{tag}`{f' on `{arch}`' if arch else ''} has {count} failing unit test(s) "
        f"(threshold {threshold:g})."
    )
    return triggered, message, {"count": count}


def _check_build_failure_count(params: dict, threshold: float) -> tuple[bool, str, dict]:
    """Queue-summary build_status == 'error' domain, not parsed Clang
    diagnostics -- see the "two warning domains" note in graph/tools.py."""
    tag = params["tag"]
    arch = params.get("arch")
    failing = get_build_failures(tag, arch=arch, limit=100)
    count = len(failing)
    triggered = count >= threshold
    architectures = ", ".join(f"`{row['arch']}`" for row in failing) if failing else "none"
    message = (
        f"`{tag}`{f' on `{arch}`' if arch else ''} has {count} architecture(s) with a failed build "
        f"(threshold {threshold:g}): {architectures}."
    )
    return triggered, message, {"count": count, "architectures": [row["arch"] for row in failing]}


def _check_latest_ib_stalled(params: dict, threshold: float) -> tuple[bool, str, dict]:
    """Threshold is hours since the newest imported IB timestamp."""
    latest = list_ib_tags(limit=1)
    if not latest:
        return False, "No IB records are imported.", {}
    tag = latest[0]["tag"]
    timestamp_text = tag[-15:]
    try:
        ts = datetime.strptime(timestamp_text, "%Y-%m-%d-%H%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return False, f"Could not parse a timestamp from `{tag}`.", {}
    hours_stale = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    triggered = hours_stale >= threshold
    message = (
        f"Latest imported IB `{tag}` is {hours_stale:.1f}h old, "
        f"at or above the {threshold:g}h staleness threshold."
    )
    return triggered, message, {"hours_stale": hours_stale, "tag": tag}


def _check_relval_regression(params: dict, threshold: float) -> tuple[bool, str, dict]:
    """Unlike every other rule type here, this doesn't check a currently-failing
    count against one fixed, pre-configured tag -- it dynamically resolves
    "latest vs previous build" each time it runs (find_relval_regressions) and
    fires on any workflow that's genuinely NEW since then. DEFAULT and every
    sub-IB present at the latest build are covered in one combined check: DEFAULT
    diffs against the fixed previous_tag, each sub-IB diffs against its own true
    previous build (same fallback policy the chat comparison and shift digest use)
    since sub-IB build cadence rarely lines up with DEFAULT's. threshold=1 means
    "fire on any newly-failing workflow at all" -- set higher only to tolerate a
    small amount of regression noise before alerting.

    params: {"arch": "<arch>"} optional -- omit to check every architecture that
    has an imported primary record at the latest build (DEFAULT is not a fixed
    architecture list; it can be 2 architectures on one build and 5 on the next)."""
    arch = params.get("arch")
    result = find_relval_regressions(arch)
    if result["latest_tag"] is None:
        return False, "Fewer than two primary builds are imported; nothing to compare yet.", {}
    newly_failing = result["newly_failing"]
    count = len(newly_failing)
    triggered = count >= threshold
    lines = [
        f"{count} RelVal workflow(s) newly failing since the previous build "
        f"(`{result['previous_tag']}` -> `{result['latest_tag']}`)"
        f"{f' on `{arch}`' if arch else ''} (threshold {threshold:g}):"
    ]
    for item in newly_failing[:20]:
        lines.append(f"- `{item['workflow_id']}` ({item['name']}) on `{item['arch']}` [{item['variant']}]")
    if count > 20:
        lines.append(f"... and {count - 20} more.")
    message = "\n".join(lines)
    return triggered, message, {
        "count": count, "latest_tag": result["latest_tag"], "previous_tag": result["previous_tag"],
        "newly_failing": newly_failing,
    }


def _check_unittest_regression(params: dict, threshold: float) -> tuple[bool, str, dict]:
    """Same shape as _check_relval_regression, for UnitTest failures: fires on any
    unit test newly failing since the previous build, across DEFAULT and every
    sub-IB (each against its own previous build) in one combined check."""
    arch = params.get("arch")
    result = find_unittest_regressions(arch)
    if result["latest_tag"] is None:
        return False, "Fewer than two primary builds are imported; nothing to compare yet.", {}
    newly_failing = result["newly_failing"]
    count = len(newly_failing)
    triggered = count >= threshold
    lines = [
        f"{count} unit test(s) newly failing since the previous build "
        f"(`{result['previous_tag']}` -> `{result['latest_tag']}`)"
        f"{f' on `{arch}`' if arch else ''} (threshold {threshold:g}):"
    ]
    for item in newly_failing[:20]:
        lines.append(f"- `{item['name']}` on `{item['arch']}` [{item['variant']}]")
    if count > 20:
        lines.append(f"... and {count - 20} more.")
    message = "\n".join(lines)
    return triggered, message, {
        "count": count, "latest_tag": result["latest_tag"], "previous_tag": result["previous_tag"],
        "newly_failing": newly_failing,
    }


def _check_build_regression(params: dict, threshold: float) -> tuple[bool, str, dict]:
    """Same shape as _check_relval_regression, for the whole-architecture
    build_status/addon_status badges: fires when any badge (Build or AddOn --
    covered together, see find_build_regressions) flips to 'error' since the
    previous build, across DEFAULT and every sub-IB (each against its own previous
    build). threshold=1 means "fire on any badge flip at all"."""
    arch = params.get("arch")
    result = find_build_regressions(arch)
    if result["latest_tag"] is None:
        return False, "Fewer than two primary builds are imported; nothing to compare yet.", {}
    newly_failing = result["newly_failing"]
    count = len(newly_failing)
    triggered = count >= threshold
    lines = [
        f"{count} Build/AddOn badge(s) newly in error since the previous build "
        f"(`{result['previous_tag']}` -> `{result['latest_tag']}`)"
        f"{f' on `{arch}`' if arch else ''} (threshold {threshold:g}):"
    ]
    for item in newly_failing[:20]:
        lines.append(f"- {item['kind']} on `{item['arch']}` [{item['variant']}]")
    if count > 20:
        lines.append(f"... and {count - 20} more.")
    message = "\n".join(lines)
    return triggered, message, {
        "count": count, "latest_tag": result["latest_tag"], "previous_tag": result["previous_tag"],
        "newly_failing": newly_failing,
    }


_CHECKS = {
    "relval_failure_count": _check_relval_failure_count,
    "relval_recurrence": _check_relval_recurrence,
    "latest_ib_stalled": _check_latest_ib_stalled,
    "unittest_failure_count": _check_unittest_failure_count,
    "build_failure_count": _check_build_failure_count,
    "relval_regression": _check_relval_regression,
    "unittest_regression": _check_unittest_regression,
    "build_regression": _check_build_regression,
}


class Notifier(Protocol):
    def notify(self, rule: dict, message: str, evidence: dict) -> None: ...


class LogNotifier:
    """Default notifier: writes matched alerts to a JSON-lines file. Honest
    about not reaching a real channel yet -- a real Notifier (Slack/email/
    webhook) is a drop-in addition once credentials exist; nothing about rule
    storage or evaluation needs to change."""

    def __init__(self, path: Path | str = "alerts.log.jsonl"):
        self.path = Path(path)

    def notify(self, rule: dict, message: str, evidence: dict) -> None:
        record = {
            "rule_id": rule["rule_id"],
            "rule_name": rule["name"],
            "message": message,
            "evidence": evidence,
            "fired_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  ALERT [{rule['name']}]: {message}")


def evaluate_rules(notifier: Notifier | None = None) -> list[dict]:
    """Run every active rule once and notify on each one that triggers.
    Meant to be invoked periodically by cron/systemd-timer, not by an
    in-process scheduler."""
    notifier = notifier or LogNotifier()
    fired = []
    for rule in list_rules(active_only=True):
        check = _CHECKS.get(rule["rule_type"])
        if not check:
            print(f"  skipping rule {rule['name']!r}: unknown rule_type {rule['rule_type']!r}")
            continue
        triggered, message, evidence = check(rule["params"], rule["threshold"])
        if triggered:
            notifier.notify(rule, message, evidence)
            fired.append({"rule": rule, "message": message, "evidence": evidence})
    return fired
