"""Summary of the IB test-log pipeline (ingestion/fetch_ib_test_logs.py): what has been
scanned, per day and suite, from results/ledger.jsonl, plus what is still waiting in
server-copy/ and what the last scan reported as MISSING. Read-only; touches nothing.

Usage:
    venv/bin/python scripts/test_log_stats.py [--base /home/rao/CHAT-BOT-AI/IB-TEST-LOGS]
        [--days 7] [--by build]

--by scan  (default) groups by the day a zip was scanned;
--by build groups by the IB build date in the zip's path (CMSSW_..._X_<date>-<HHMM>).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

DEFAULT_BASE = "/home/rao/CHAT-BOT-AI/IB-TEST-LOGS"
BUILD_DATE_RE = re.compile(r"CMSSW_[A-Za-z0-9_]*?_X_(\d{4}-\d\d-\d\d)-\d{4}")
SUITE_BY_ZIP = {"unitTestLogs.zip": "unit", "addOnTests.zip": "addon", "pyRelValMatrixLogs.zip": "relval"}


def ledger_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_key(row: dict, by: str) -> str:
    if by == "build":
        match = BUILD_DATE_RE.search(row.get("path") or "")
        return match.group(1) if match else "?"
    return (row.get("finished_at") or "?")[:10]


def print_ledger(rows: list[dict], by: str, days: int | None) -> None:
    totals: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        c = totals[(group_key(row, by), row.get("kind") or "?", row.get("status") or "?")]
        c["zips"] += 1
        c["bytes"] += row.get("size") or 0
        c["units"] += row.get("units") or 0
        c["problems"] += row.get("problems") or 0
        c["secs"] += row.get("seconds") or 0
    dates = sorted({key[0] for key in totals})
    if days:
        dates = dates[-days:]
    label = "build date" if by == "build" else "scanned on"
    print(f"{label:10} {'suite':6} {'status':7} {'zips':>5} {'GB':>7} {'tests/wfs':>10} {'problems':>9} {'scan min':>8}")
    for key in sorted(k for k in totals if k[0] in dates):
        c = totals[key]
        print(f"{key[0]:10} {key[1]:6} {key[2]:7} {c['zips']:5} {c['bytes'] / 1e9:7.1f} {c['units']:10} "
              f"{c['problems']:9} {c['secs'] / 60:8.0f}")
    shown = [row for row in rows if group_key(row, by) in dates]
    print(f"{'total':10} {'':6} {'':7} {len(shown):5} {sum(r.get('size') or 0 for r in shown) / 1e9:7.1f}")
    failed = [row for row in shown if row.get("status") != "done"]
    if failed:
        print(f"\n{len(failed)} zip(s) not scanned successfully (kept in server-copy/, retried on the next run):")
        for row in failed[:20]:
            print(f"  {row.get('status')}  {row.get('path')}")


def print_server_copy(base: str, done_paths: set[str]) -> None:
    root = os.path.join(base, "server-copy")
    waiting = collections.Counter()
    waiting_bytes = collections.Counter()
    for dirpath, _, files in os.walk(root):
        for name in files:
            suite = SUITE_BY_ZIP.get(name)
            if not suite:
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            if rel in done_paths:
                continue
            waiting[suite] += 1
            waiting_bytes[suite] += os.path.getsize(os.path.join(dirpath, name))
    if not waiting:
        print("\nserver-copy/: nothing waiting to be scanned")
        return
    print(f"\nserver-copy/: {sum(waiting.values())} zip(s), {sum(waiting_bytes.values()) / 1e9:.1f} GB waiting to be scanned")
    for suite in sorted(waiting):
        print(f"  {suite:6} {waiting[suite]:5}  {waiting_bytes[suite] / 1e9:7.1f} GB")


def print_missing(base: str) -> None:
    path = os.path.join(base, "results", "missing.txt")
    if not os.path.exists(path):
        return
    missing = [line.strip() for line in open(path) if line.strip() and not line.startswith("#")]
    print(f"\nMISSING at the last scan (results/missing.txt): {len(missing)}")
    if missing:
        by_date = collections.Counter(m.group(1) if (m := BUILD_DATE_RE.search(p)) else "?" for p in missing)
        for date in sorted(by_date):
            print(f"  build {date}: {by_date[date]}")
        print("  (copy these before the server deletes them, ~1 week; rerun the scan to refresh the list)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=DEFAULT_BASE, help="folder with results/ and server-copy/")
    parser.add_argument("--days", type=int, default=None, help="only the last N dates")
    parser.add_argument("--by", choices=("scan", "build"), default="scan")
    args = parser.parse_args()
    ledger = os.path.join(args.base, "results", "ledger.jsonl")
    if not os.path.exists(ledger):
        print(f"No ledger at {ledger} -- nothing scanned yet?", file=sys.stderr)
        return 1
    rows = ledger_rows(ledger)
    print_ledger(rows, args.by, args.days)
    print_server_copy(args.base, {row.get("path") for row in rows if row.get("status") == "done"})
    print_missing(args.base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
