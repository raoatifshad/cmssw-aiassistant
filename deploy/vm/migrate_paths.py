"""One-off, when the database moves to another machine: rewrite the file paths
stored in it to where the same files are on the new machine.

Why: every import remembers the full path of the file it came from (source_id,
e.g. "relvals:<tag>:<arch>:/home/rao/.../cmsrepo/data/relvals/..."). On the VM the
same files are under /data/sources/..., so without this the worker re-imports
everything once, and the wiki docs end up in the graph twice (their clean-up only
looks under the current folder). After this, the worker finds the files already
imported (same path, same content hash) and imports only what really changed.

Run it after restoring the dump on the VM and BEFORE the worker's first run:

  podman compose -p kg stop worker
  podman compose -p kg run --rm --no-deps worker python migrate_paths.py \\
      --map /home/rao/CHAT-BOT-AI/NEO4J/cmssdt-kg-chatbot/cmsrepo=/data/sources/cmsrepo \\
      --map /home/rao/CHAT-BOT-AI/NEO4J/cmssdt-kg-chatbot/shifter-wiki=/data/sources/shifter-wiki \\
      --map /home/rao/CHAT-BOT-AI/CLANG-LOGS=/data/clang-logs          # dry run: only counts
  ... the same with --apply [--check]
  podman compose -p kg start worker

A dry run changes nothing. Running it twice is harmless (nothing left to rewrite).
If the worker already ran on the new paths, the old copy of a file that now exists
under both paths is dropped (its ImportState and DocChunks); every other record
just gets the new path.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from neo4j import GraphDatabase

# every (label, property) that holds a file path, as written by build_graph.py,
# ingest_shifter_docs.py and ingest_github_patches.py (test logs store relative paths)
PATH_FIELDS = [
    ("ImportState", "source_id"), ("ImportState", "source_path"),
    ("WorkflowCommand", "source_id"), ("RelVal", "source_id"), ("UnitTest", "source_id"),
    ("Warning", "source_id"), ("RelValException", "source_id"), ("DocChunk", "source_id"),
    ("PR", "source_id"), ("PR", "cache_source_id"), ("Commit", "source_id"), ("ExitCode", "source_id"),
    ("IB", "build_status_source_id"),
]
BATCH = 5000   # rows per transaction: small enough for a 512 MB heap


def parse_map(items: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for item in items:
        old, sep, new = item.partition("=")
        if not sep or not old.startswith("/") or not new.startswith("/"):
            raise SystemExit(f"--map wants /old/path=/new/path, got {item!r}")
        pairs.append((old.rstrip("/") + "/", new.rstrip("/") + "/"))   # whole folder names only
    return pairs


def count(session, label: str, prop: str, old: str) -> int:
    return session.run(f"MATCH (n:`{label}`) WHERE n.`{prop}` CONTAINS $old RETURN count(n) AS c", old=old).single()["c"]


def drop_collisions(session, old: str, new: str) -> int:
    """Files imported under BOTH paths (the worker ran before this): keep the new copy."""
    rows = session.run(
        "MATCH (o:ImportState) WHERE o.source_id CONTAINS $old "
        "WITH o, replace(o.source_id, $old, $new) AS target "
        "MATCH (:ImportState {source_id: target}) RETURN o.source_id AS sid", old=old, new=new).data()
    for row in rows:
        session.run("MATCH (d:DocChunk {source_id:$sid}) DETACH DELETE d", sid=row["sid"])
        session.run("MATCH (s:ImportState {source_id:$sid}) DELETE s", sid=row["sid"])
    return len(rows)


def rewrite(session, label: str, prop: str, old: str, new: str, batch: int = BATCH) -> int:
    """In batches of BATCH, each its own short transaction (one long statement over
    ~1.9M WorkflowCommand nodes hit the connection timeout). Safe to stop and rerun."""
    done = 0
    while True:
        n = session.run(
            f"MATCH (n:`{label}`) WHERE n.`{prop}` CONTAINS $old WITH n LIMIT {int(batch)} "
            f"SET n.`{prop}` = replace(n.`{prop}`, $old, $new) RETURN count(n) AS c",
            old=old, new=new).single()["c"]
        done += n
        if n < batch:
            return done


def leftovers(session, olds: list[str]) -> list[dict]:
    """Every node and string property in the whole graph still holding an old path (slow: reads all)."""
    return session.run(
        "MATCH (n) UNWIND [k IN keys(n) WHERE n[k] IS :: STRING AND any(o IN $olds WHERE n[k] CONTAINS o)] AS k "
        "RETURN labels(n)[0] AS label, k AS prop, count(*) AS n ORDER BY n DESC", olds=olds).data()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", action="append", required=True, metavar="OLD=NEW")
    ap.add_argument("--apply", action="store_true", help="really rewrite (default: count only)")
    ap.add_argument("--check", action="store_true", help="afterwards, search every node for old paths")
    ap.add_argument("--batch", type=int, default=BATCH, help=f"rows per transaction (default {BATCH})")
    args = ap.parse_args(argv)
    pairs = parse_map(args.map)
    driver = GraphDatabase.driver(os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                                  auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]))
    started = time.time()
    total = 0
    with driver, driver.session() as session:
        for old, new in pairs:
            print(f"{old} -> {new}")
            for label, prop in PATH_FIELDS:
                n = count(session, label, prop, old)
                total += n
                if n:
                    print(f"  {label}.{prop}: {n}")
            if args.apply:
                dropped = drop_collisions(session, old, new)
                if dropped:
                    print(f"  dropped {dropped} old copies of files already imported under the new path")
                for label, prop in PATH_FIELDS:
                    rewrite(session, label, prop, old, new, args.batch)
        if args.apply:
            remaining = sum(count(session, label, prop, old) for old, _ in pairs for label, prop in PATH_FIELDS)
            print(f"rewritten in {time.time() - started:.0f} s; left on the known fields: {remaining}")
            if remaining:
                return 1
        else:
            print(f"dry run: {total} values would be rewritten (add --apply)")
        if args.check:
            rest = leftovers(session, [old for old, _ in pairs])
            print("old paths anywhere in the graph: " + (", ".join(f"{r['label']}.{r['prop']} {r['n']}" for r in rest) or "none"))
            if rest and args.apply:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
