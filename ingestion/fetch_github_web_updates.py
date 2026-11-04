"""Stopgap while the GitHub backup on cmssdt9-02 is stale: fetch the cms-sw/cmssw Issues and PRs
updated since a date straight from the GitHub REST API, and write them as the backup's own
layout (<store>/cmssw/issues/<md5[:2]>/<md5[2:]>/issue.json) so build_graph.py --github-issues
imports them like the real backup.

Added 2026-11-04 (user decision): the backup's checkpoint (issues/latest.txt) has been stuck at
2026-09-04. This is an *offline* fetch run by hand or by the refresh -- the chat bot itself still
never calls GitHub. Differences from the real backup, said in the answers via `fetched_via`:
  - no comments.json (comments cost one API call per item); existing comments are kept,
  - no patches or images.
When the real backup works again, its newer copies replace these (the importer keeps the newer
updated_at) and clear `fetched_via`.

    # fetch everything updated since the newest Issue/PR in the graph, then import (no --reconcile!)
    venv/bin/python ingestion/fetch_github_web_updates.py [--since 2026-09-04T00:00:00Z]
    PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/build_graph.py \\
        --cmsrepo cmsrepo --github-issues /home/rao/CHAT-BOT-AI/GITHUB-ISSUES/web-update/store

Without a token GitHub allows 60 requests an hour (100 items each): enough for this listing.
A token is used if GITHUB_TOKEN_FILE points to a file (never logged, never stored elsewhere).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_OUT = "/home/rao/CHAT-BOT-AI/GITHUB-ISSUES/web-update"
REPO = "cms-sw/cmssw"


def newest_in_graph() -> str | None:
    """The newest GitHub update time already imported (Issues and PRs), as the --since default."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "graph"))
    from neo4j_conn import get_driver  # noqa: E402
    driver = get_driver()
    try:
        with driver.session() as session:
            row = session.run(
                "CALL { MATCH (i:Issue {repo:'cmssw'}) RETURN max(i.updated_at) AS t "
                "UNION MATCH (p:PR {repo:'cmssw'}) RETURN max(p.github_updated_at) AS t } "
                "RETURN max(t) AS t").single()
            return row["t"] if row else None
    finally:
        driver.close()


def fetch(since: str, max_pages: int, token: str | None) -> tuple[list[dict], bool]:
    items, page, complete = [], 1, False
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "cmssdt-kg-stopgap-fetch"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    while page <= max_pages:
        url = (f"https://api.github.com/repos/{REPO}/issues?state=all&since={since}"
               f"&per_page=100&sort=updated&direction=asc&page={page}")
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
            left = response.headers.get("X-RateLimit-Remaining")
            batch = json.load(response)
        items += batch
        print(f"  page {page}: {len(batch)} item(s), {len(items)} in total, rate limit left {left}", flush=True)
        if len(batch) < 100:
            complete = True
            break
        if left is not None and int(left) < 3:
            print("  stopping: rate limit nearly used up -- run again later to continue", flush=True)
            break
        page += 1
        time.sleep(1)
    return items, complete


def stage(items: list[dict], store: Path, label: str) -> int:
    """Write each item where github_backup.py would (md5 of "<number>\\n"), marked as web-fetched."""
    written = 0
    for item in items:
        if not isinstance(item.get("number"), int):
            continue
        digest = hashlib.md5(f"{item['number']}\n".encode()).hexdigest()
        directory = store / "cmssw" / "issues" / digest[:2] / digest[2:]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "issue.json").write_text(json.dumps({**item, "_fetched_via": label}))
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--since", help="ISO time; default: the newest GitHub update already in the graph")
    parser.add_argument("--out", default=DEFAULT_OUT, help="folder for the raw listing and the staged store")
    parser.add_argument("--max-pages", type=int, default=50)
    args = parser.parse_args()
    since = args.since or newest_in_graph() or "2026-09-04T00:00:00Z"
    token = None
    if os.environ.get("GITHUB_TOKEN_FILE"):
        token = Path(os.environ["GITHUB_TOKEN_FILE"]).expanduser().read_text().strip() or None
    today = time.strftime("%Y-%m-%d", time.gmtime())
    print(f"fetching {REPO} Issues/PRs updated since {since} ({'with' if token else 'without'} a token)")
    items, complete = fetch(since, args.max_pages, token)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    listing = out / f"cmssw-issues-updated-since-{since[:10]}-fetched-{today}.json"
    listing.write_text(json.dumps({"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                   "since": since, "complete": complete, "items": items}))
    written = stage(items, out / "store", f"github-web-fetch {today}")
    print(f"saved {len(items)} item(s) to {listing}; staged {written} issue.json under {out / 'store'}"
          + ("" if complete else " -- INCOMPLETE, run again to continue"))
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
