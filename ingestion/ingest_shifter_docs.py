"""Ingest static CMSSW/CMSSDT shifter documentation (Jenkins job runbooks,
release-cycle procedures, CI topology notes) as citable (:DocChunk) nodes.

Sources are local, static git checkouts -- a clone of cms-sw/cmssdt-wiki plus
cmsrepo's own unused top-level *.md pages -- never fetched live during a chat
request, same trust model as everything under cmsrepo/data. Re-running this
script is incremental and safe: unchanged files are skipped via the same
ImportState content-hashing every ingest_* function in build_graph.py uses.
Queried at chat time via graph.tools.search_shifter_docs(), a Neo4j fulltext
index over (:DocChunk), never query_graph directly.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_graph import file_hash, mark_source_imported, sha256_bytes, source_needs_import  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))
from neo4j_conn import get_driver, setup_schema  # noqa: E402

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def split_into_chunks(
    text: str, max_section_chars: int = 4000, window_chars: int = 1500, overlap_chars: int = 200,
) -> list[tuple[str, str]]:
    """Split markdown text into (heading, body) chunks on '#'..'######'
    headers. Any section longer than max_section_chars is further split into
    overlapping windows so a single chunk never swamps a fulltext result."""
    matches = list(_HEADER_RE.finditer(text))
    sections: list[tuple[str, str]] = []
    if not matches:
        if text.strip():
            sections.append(("", text.strip()))
    else:
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(("", preamble))
        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((heading, body))

    chunks: list[tuple[str, str]] = []
    for heading, body in sections:
        if len(body) <= max_section_chars:
            chunks.append((heading, body))
            continue
        start = 0
        while start < len(body):
            end = min(start + window_chars, len(body))
            chunks.append((heading, body[start:end]))
            if end == len(body):
                break
            start = end - overlap_chars
    return chunks


def _ingest_doc_file(driver, path: Path, source_root: Path, force: bool) -> int:
    source_id = f"shifter-doc:{path.resolve()}"
    digest = file_hash(path)
    if not source_needs_import(driver, source_id, digest, force):
        return 0
    chunks = split_into_chunks(path.read_text(errors="ignore"))
    relative_path = path.relative_to(source_root)
    records = [
        {
            "id": sha256_bytes(f"{source_id}:{i}".encode()),
            "source_id": source_id,
            "source_path": str(relative_path),
            "heading": heading,
            "text": body,
            "chunk_index": i,
        }
        for i, (heading, body) in enumerate(chunks)
    ]
    with driver.session() as session:
        session.run("MATCH (d:DocChunk {source_id:$source_id}) DETACH DELETE d", source_id=source_id)
        if records:
            session.run(
                "UNWIND $records AS r CREATE (d:DocChunk {id:r.id, source_id:r.source_id, "
                "source_path:r.source_path, heading:r.heading, text:r.text, chunk_index:r.chunk_index})",
                records=records,
            )
    mark_source_imported(driver, source_id, digest, "shifter-doc", path)
    return len(records)


def _reconcile_missing_docs(driver, prefix: str, active_sources: set[str]):
    with driver.session() as session:
        records = session.run(
            "MATCH (state:ImportState) WHERE state.source_id STARTS WITH $prefix "
            "RETURN state.source_id AS source_id",
            prefix=prefix,
        ).data()
    for record in records:
        source_id = record["source_id"]
        if source_id in active_sources:
            continue
        with driver.session() as session:
            session.run("MATCH (d:DocChunk {source_id:$source_id}) DETACH DELETE d", source_id=source_id)
            session.run("MATCH (state:ImportState {source_id:$source_id}) DELETE state", source_id=source_id)
        print(f"  removed missing source: {source_id}")


def ingest_doc_tree(
    driver, root: Path, force: bool = False, reconcile: bool = False,
    recursive: bool = True, skip_dirs: tuple[str, ...] = (),
) -> int:
    """Ingest every .md file under root as (:DocChunk) nodes. skip_dirs names
    path components to exclude (e.g. an 'old' or '_snippets' subdirectory)."""
    if not root.is_dir():
        return 0
    pattern = "**/*.md" if recursive else "*.md"
    total = 0
    active_sources: set[str] = set()
    for path in sorted(root.glob(pattern)):
        if any(part in skip_dirs for part in path.relative_to(root).parts):
            continue
        active_sources.add(f"shifter-doc:{path.resolve()}")
        total += _ingest_doc_file(driver, path, root, force)
    if reconcile:
        _reconcile_missing_docs(driver, f"shifter-doc:{root.resolve()}", active_sources)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-root", help="Local clone of cms-sw/cmssdt-wiki")
    parser.add_argument("--cmsrepo", help="Local cmsrepo checkout -- ingests its unused top-level *.md pages")
    parser.add_argument("--force", action="store_true", help="reimport unchanged source files")
    parser.add_argument("--reconcile", action="store_true", help="remove DocChunks from sources no longer on disk")
    args = parser.parse_args()
    if not args.wiki_root and not args.cmsrepo:
        parser.error("at least one of --wiki-root / --cmsrepo is required")

    driver = get_driver()
    setup_schema(driver)
    total = 0
    if args.wiki_root:
        total += ingest_doc_tree(
            driver, Path(args.wiki_root), args.force, args.reconcile,
            skip_dirs=("old", "_snippets"),
        )
    if args.cmsrepo:
        # build_graph.py already owns everything under cmsrepo/data as
        # structured IB facts -- this only adds the unused top-level
        # FAQ/troubleshooting/workflow pages sitting alongside it.
        total += ingest_doc_tree(driver, Path(args.cmsrepo), args.force, args.reconcile, recursive=False)
    print(f"\nIngested/updated {total} shifter-doc chunks.")


if __name__ == "__main__":
    main()
