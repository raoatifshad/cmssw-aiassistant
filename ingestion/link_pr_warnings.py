"""Link imported PRs to Clang warnings through GitHub changed-file evidence.

Usage:
    NEO4J_URI=bolt://localhost:7688 python ingestion/link_pr_warnings.py \
        --release CMSSW_20_1 --ib-tag CMSSW_20_1_X_2026-08-03-2300

The relationships are evidence of a changed source file, not proof that a PR
introduced a warning.
"""
import argparse
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))
from neo4j_conn import get_driver, setup_schema  # noqa: E402

GITHUB_API = "https://api.github.com"
REPOSITORIES = {"cmssw": "cms-sw/cmssw", "cmsdist": "cms-sw/cmsdist"}


def changed_files_from_patch(patch: str) -> list[str]:
    paths = []
    for old_path, new_path in re.findall(r"^diff --git a/(.+) b/(.+)$", patch, re.MULTILINE):
        path = new_path if new_path != "/dev/null" else old_path
        if path != "/dev/null":
            paths.append(path)
    return paths


def changed_files_from_raw_dump(root: Path, issue_source_path: str) -> list[str]:
    patch_path = f"{Path(issue_source_path).parent.as_posix()}/patch.txt"
    if root.is_file() and tarfile.is_tarfile(root):
        with tarfile.open(root, "r:*") as archive:
            member = archive.extractfile(patch_path)
            return changed_files_from_patch(member.read().decode(errors="replace")) if member else []
    local_path = root / patch_path
    return changed_files_from_patch(local_path.read_text(errors="replace")) if local_path.is_file() else []


def fetch_changed_files_from_diff(repository: str, number: int) -> list[str]:
    """Read changed paths from GitHub's public unified-diff endpoint."""
    request = Request(
        f"https://patch-diff.githubusercontent.com/raw/{repository}/pull/{number}.diff",
        headers={"Accept": "text/plain", "User-Agent": "cmssdt-kg-chatbot"},
    )
    with urlopen(request, timeout=30) as response:
        diff = response.read().decode("utf-8", errors="replace")
    paths = []
    return changed_files_from_patch(diff)


def fetch_changed_files(repository: str, number: int, token: str | None) -> list[str]:
    files = []
    page = 1
    while True:
        query = urlencode({"per_page": 100, "page": page})
        url = f"{GITHUB_API}/repos/{repository}/pulls/{number}/files?{query}"
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "cmssdt-kg-chatbot"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                batch = json.load(response)
        except HTTPError as error:
            if token and error.code in (401, 403):
                try:
                    with urlopen(Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "cmssdt-kg-chatbot"}), timeout=30) as response:
                        batch = json.load(response)
                except HTTPError as anonymous_error:
                    if anonymous_error.code not in (403, 429):
                        raise
                    return fetch_changed_files_from_diff(repository, number)
            elif error.code in (403, 429):
                return fetch_changed_files_from_diff(repository, number)
            else:
                raise
        if not batch:
            return files
        files.extend(item["filename"] for item in batch if item.get("filename"))
        if len(batch) < 100:
            return files
        page += 1


def link_pr_files(driver, pr: dict, files: list[str], ib_tag: str | None, evidence: str) -> tuple[int, int]:
    source_matches = 0
    warning_matches = 0
    with driver.session() as session:
        for file_path in files:
            source_path = file_path if file_path.startswith("src/") else f"src/{file_path}"
            result = session.run(
                "MATCH (p:PR {repo:$repo, number:$number}) "
                "MERGE (sf:SourceFile {path:$source_path}) "
                "MERGE (p)-[:CHANGED_FILE {evidence:$evidence}]->(sf) "
                "WITH p, sf "
                "OPTIONAL MATCH (w:Warning)-[:IN_FILE]->(sf) "
                "WHERE $ib_tag IS NULL OR EXISTS { "
                "  MATCH (:IB {tag:$ib_tag})-[:REPORTED]->(w) "
                "} "
                "FOREACH (warning IN CASE WHEN w IS NULL THEN [] ELSE [w] END | "
                "  MERGE (p)-[:MATCHES_WARNING {evidence:'changed_file'}]->(warning) "
                ") "
                "RETURN count(DISTINCT sf) AS source_matches, count(DISTINCT w) AS warning_matches",
                repo=pr["repo"], number=pr["number"], source_path=source_path, ib_tag=ib_tag,
                evidence=evidence,
            ).single()
            source_matches += result["source_matches"]
            warning_matches += result["warning_matches"]
    return source_matches, warning_matches


def main():
    parser = argparse.ArgumentParser(description="Link PR changed files to Clang warnings")
    parser.add_argument("--release", help="PR release, e.g. CMSSW_20_1")
    parser.add_argument("--github-issues", help="Raw GitHub issues directory or .tar/.tar.gz archive")
    parser.add_argument("--ib-tag", help="Restrict warning links to one integration build")
    parser.add_argument("--limit", type=int, help="Limit PRs for a small test run")
    parser.add_argument("--pr-number", type=int, help="Enrich one PR number for a targeted repair")
    parser.add_argument("--dry-run", action="store_true", help="Fetch files without writing Neo4j links")
    args = parser.parse_args()
    if not args.release and not args.github_issues and not args.pr_number:
        parser.error("one of --release, --github-issues, or --pr-number is required")

    token = os.environ.get("GITHUB_TOKEN")
    driver = get_driver()
    setup_schema(driver)
    with driver.session() as session:
        records = session.run(
            "MATCH (p:PR) WHERE $release IS NULL OR p.release = $release "
            "OPTIONAL MATCH (state:ImportState {source_id:p.raw_source_id}) "
            "RETURN p.repo AS repo, p.number AS number, p.title AS title, "
            "state.source_path AS raw_source_path ORDER BY p.number",
            release=args.release,
        ).data()

    if args.pr_number:
        records = [record for record in records if record["number"] == args.pr_number]
    if args.limit:
        records = records[:args.limit]
    print(f"Enriching {len(records)} PRs")

    total_files = 0
    source_matches = 0
    warning_matches = 0
    for pr in records:
        repository = REPOSITORIES.get(pr["repo"])
        if not repository:
            continue
        try:
            if args.github_issues:
                if not pr["raw_source_path"]:
                    continue
                files = changed_files_from_raw_dump(Path(args.github_issues), pr["raw_source_path"])
                evidence = "raw_patch"
            else:
                files = fetch_changed_files(repository, pr["number"], token)
                evidence = "github_changed_file"
        except HTTPError as error:
            print(f"  PR {pr['repo']}#{pr['number']}: GitHub HTTP {error.code}")
            continue
        except OSError as error:
            print(f"  PR {pr['repo']}#{pr['number']}: {error}")
            continue

        total_files += len(files)
        if not args.dry_run:
            linked_sources, linked_warnings = link_pr_files(driver, pr, files, args.ib_tag, evidence)
            source_matches += linked_sources
            warning_matches += linked_warnings
        print(f"  PR {pr['repo']}#{pr['number']}: {len(files)} changed files")

    driver.close()
    if args.dry_run:
        print(f"Fetched {total_files} changed files; no graph writes made.")
    else:
        print(
            f"Fetched {total_files} changed files; linked {source_matches} source files "
            f"and {warning_matches} warnings."
        )


if __name__ == "__main__":
    main()
