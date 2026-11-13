"""Import which files each CMSSW PR touched, from the patch.txt files of the GitHub backup.

    PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/ingest_github_patches.py \
        --github-issues /home/rao/CHAT-BOT-AI/GITHUB-ISSUES/issues [--dry-run out.json] [--force]

Each <dir>/patch.txt sits next to the PR's issue.json. It is `git format-patch` output, so it lists every
commit of the PR: a file changed and then changed back still counts, which is why this is "touched", not the
PR's net diff, and why PR.additions/deletions/changed_files (dashboard values) are left alone.

Writes (:PR)-[:TOUCHED_FILE {evidence:'github_patch'}]->(:SourceFile {path:'src/...'}), same path shape as the
Clang import, PR.patch_files (how many) and PR.patch_status ('ok', or 'unavailable' when the backup saved an error
page instead: unknown, never zero, and patch_files stays unset). Only the paths are stored: the 21 GB of diff text stays on disk.
A deliberately separate relationship from CHANGED_FILE, because graph/tools.py joins CHANGED_FILE against every
failed unit test in find_prs_for_error. Like every changed-file match, it is relevance evidence, never causality.

Incremental: ImportState github-patch:cmssw:<number> keeps size:mtime of the patch (rsync -a keeps mtimes).
A PR whose patch changed loses its old links first (force-pushes drop files). Never calls GitHub.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))

DIFF_LINE = re.compile(rb'^diff --git "?a/(.+?)"? "?b/(.+?)"?\s*$')
BATCH = 200
MAX_EDGES_PER_TX = 20000


def touched_files(patch_path: Path) -> list[str] | None:
    """Distinct paths in a patch, in the backup's own new-path spelling, or None when the file is not a patch.
    Found 2026-11-06: about 7,200 of 46,554 are saved error pages (GitHub HTML, "error: too big or took too long",
    "The server is unavailable") or empty, so "no paths" must not be read as "the PR touched nothing".
    Read line by line: the biggest is 265 MB. Lines of a diff body start with ' ', '+' or '-', so a patch that
    adds a patch file can't match here."""
    paths = set()
    with open(patch_path, "rb") as stream:
        if not stream.read(5) == b"From ":
            return None
        stream.seek(0)
        for line in stream:
            if line.startswith(b"diff --git "):
                match = DIFF_LINE.match(line)
                if match:
                    paths.add(match.group(2).decode("utf-8", errors="replace"))
    return sorted(paths)


def source_path(path: str) -> str:
    return path if path.startswith("src/") else f"src/{path}"


def scan(root: Path):
    """Yield (number, patch path, state value) for every patch.txt beside an issue.json that is a PR."""
    for patch in sorted(root.rglob("patch.txt")):
        issue_path = patch.with_name("issue.json")
        if not issue_path.is_file():
            yield None, patch, None
            continue
        issue = json.loads(issue_path.read_text())
        stat = patch.stat()
        yield (issue.get("number") if "pull_request" in issue else None), patch, f"{stat.st_size}:{int(stat.st_mtime)}"


def write_batch(session, batch: list[dict]) -> int:
    numbers = [item["number"] for item in batch]
    found = {row["n"] for row in session.run(
        "MATCH (p:PR {repo:'cmssw'}) WHERE p.number IN $numbers RETURN p.number AS n", numbers=numbers)}
    present = [item for item in batch if item["number"] in found]
    session.run(
        "UNWIND $numbers AS number MATCH (:PR {repo:'cmssw', number:number})-[r:TOUCHED_FILE]->() DELETE r",
        numbers=[item["number"] for item in present])
    edges = [{"number": item["number"], "path": source_path(path)} for item in present for path in item["paths"] or []]
    for start in range(0, len(edges), MAX_EDGES_PER_TX):
        session.run(
            "UNWIND $edges AS edge MATCH (p:PR {repo:'cmssw', number:edge.number}) "
            "MERGE (sf:SourceFile {path:edge.path}) "
            "MERGE (p)-[:TOUCHED_FILE {evidence:'github_patch'}]->(sf)",
            edges=edges[start:start + MAX_EDGES_PER_TX])
    # the state last: a run cut by a power failure redoes this batch, and only this batch
    session.run(
        "UNWIND $items AS item MATCH (p:PR {repo:'cmssw', number:item.number}) "
        "SET p.patch_files=size(item.paths), p.patch_status=CASE WHEN item.paths IS NULL THEN 'unavailable' ELSE 'ok' END "
        "MERGE (state:ImportState {source_id:item.source_id}) "
        "SET state.content_hash=item.state, state.source_type='github-patch', state.source_path=item.patch, "
        "state.last_ingested_at=datetime()",
        items=present)
    return len(batch) - len(present)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-issues", required=True, help="the backup's issues directory")
    parser.add_argument("--dry-run", metavar="OUT", help="only read the patches and write a size report here")
    parser.add_argument("--force", action="store_true", help="reimport unchanged patches")
    args = parser.parse_args()
    root = Path(args.github_issues)

    if args.dry_run:
        report = {"patches": 0, "not_a_pr": [], "empty": [], "unavailable": [], "edges": 0, "largest": [], "unparsed_examples": []}
        paths_seen, sizes = Counter(), []
        for number, patch, _ in scan(root):
            report["patches"] += 1
            if number is None:
                report["not_a_pr"].append(str(patch))
                continue
            files = touched_files(patch)
            if files is None:
                report["unavailable"].append(number)
                continue
            if not files:
                report["empty"].append(number)
            report["edges"] += len(files)
            sizes.append((len(files), number))
            paths_seen.update(source_path(path) for path in files)
        report["distinct_paths"] = len(paths_seen)
        report["largest"] = sorted(sizes, reverse=True)[:25]
        report["odd_paths"] = [p for p in paths_seen if re.search(r'[\s"\\�]', p)][:50]
        report["median_files"] = sorted(sizes)[len(sizes) // 2][0] if sizes else 0
        json.dump(report, open(args.dry_run, "w"), indent=1)
        print({k: v for k, v in report.items() if k in ("patches", "edges", "distinct_paths", "median_files")},
              "not a PR:", len(report["not_a_pr"]), "no files:", len(report["empty"]),
              "unavailable:", len(report["unavailable"]))
        return

    from neo4j_conn import get_driver, setup_schema
    driver = get_driver()
    setup_schema(driver)
    with driver.session() as session:
        states = {row["id"]: row["hash"] for row in session.run(
            "MATCH (s:ImportState {source_type:'github-patch'}) RETURN s.source_id AS id, s.content_hash AS hash")}
        batch, weight = [], 0
        imported = skipped = no_pr_node = not_a_pr = unavailable = 0

        def flush():
            nonlocal batch, weight, no_pr_node, imported
            if batch:
                no_pr_node += write_batch(session, batch)
                imported += len(batch)
                print(f"  patches imported so far: {imported}", flush=True)
            batch, weight = [], 0

        for number, patch, state in scan(root):
            if number is None:
                not_a_pr += 1
                continue
            source_id = f"github-patch:cmssw:{number}"
            if not args.force and states.get(source_id) == state:
                skipped += 1
                continue
            paths = touched_files(patch)
            batch.append({"number": number, "paths": paths, "source_id": source_id, "state": state, "patch": str(patch)})
            weight += len(paths or []) + 1
            unavailable += paths is None
            if len(batch) >= BATCH or weight >= MAX_EDGES_PER_TX:
                flush()
        flush()
    print(f"  GitHub patches imported: {imported}; unchanged: {skipped}; no PR node: {no_pr_node}; "
          f"not a PR: {not_a_pr}; error page instead of a patch: {unavailable}")


if __name__ == "__main__":
    main()
