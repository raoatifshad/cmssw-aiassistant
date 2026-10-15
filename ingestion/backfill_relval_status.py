"""One-off: re-derive RelVal.status from the relval files still in cmsrepo and fix
rows stored with an older rule. Writes only rows whose status differs.

    NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/backfill_relval_status.py --cmsrepo cmsrepo          # report
    NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/backfill_relval_status.py --cmsrepo cmsrepo --apply

Added 2026-10-15 for DAS_ERROR (stored as PASSED before build_graph.relval_status):
re-running build_graph.py --force would rewrite every source to fix a few hundred
rows. Uses the same key build_graph.py writes, so it can only touch rows imported
from these exact files; ImportState is left alone, so nothing is re-imported.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))
from build_graph import ib_key, relval_status  # noqa: E402
from neo4j_conn import get_driver  # noqa: E402


def wanted_statuses(relvals_root: Path) -> list[dict]:
    """{key, status} for every non-PASSED RelVal in the files (PASSED never changed)."""
    rows = []
    for path in sorted(relvals_root.glob("*/*/*.json")):
        arch = path.parent.parent.name
        tag = f"{path.stem}_{path.parent.name}"
        for entry in json.loads(path.read_text()):
            status = relval_status(entry.get("steps", []))
            if status != "PASSED":
                rows.append({"key": f"{ib_key(tag, arch)}:{entry.get('id', '')}", "status": status})
    return rows


def backfill(driver, relvals_root: Path, apply: bool) -> dict[str, int]:
    """{"<old>-><new>": count} of rows whose stored status differs from the files."""
    rows = wanted_statuses(relvals_root)
    changes: dict[str, int] = {}
    with driver.session() as session:
        for record in session.run(
            "UNWIND $rows AS row MATCH (r:RelVal {key:row.key}) WHERE r.status <> row.status "
            "RETURN r.status + '->' + row.status AS change, count(*) AS n", rows=rows,
        ):
            changes[record["change"]] = record["n"]
        if apply and changes:
            session.run(
                "UNWIND $rows AS row MATCH (r:RelVal {key:row.key}) WHERE r.status <> row.status "
                "SET r.status = row.status", rows=rows,
            ).consume()
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cmsrepo", required=True)
    parser.add_argument("--apply", action="store_true", help="write the changes; without it, only report")
    args = parser.parse_args()
    driver = get_driver()
    try:
        changes = backfill(driver, Path(args.cmsrepo) / "data" / "relvals", args.apply)
    finally:
        driver.close()
    verb = "Updated" if args.apply else "Would update (dry run, pass --apply to write)"
    print(f"{verb}: " + (", ".join(f"{n} {change}" for change, n in sorted(changes.items())) or "nothing"))


if __name__ == "__main__":
    main()
