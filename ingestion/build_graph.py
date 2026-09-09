"""Incrementally build the CMSSW knowledge graph in Neo4j.

Structured IB results are read from cms-sw.github.io/data while Clang logs are
discovered independently under <clang-root>/<ib-tag>/<arch>/build-logs.
Every source unit is fingerprinted. Unchanged files are skipped, while changed
files replace only the graph records they own.
"""
import argparse
import hashlib
import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))
from neo4j_conn import clear_graph, get_driver, setup_schema  # noqa: E402

from clang_parser import aggregate_occurrences, extract_warnings, html_to_text  # noqa: E402
from relval_log_parser import extract_es_exception_blocks  # noqa: E402


GITHUB_REFERENCE_PATTERN = re.compile(r"(?<![\w/])#(\d+)\b")
GITHUB_ITEM_URL_PATTERN = re.compile(
    r"https?://github\.com/cms-sw/cmssw/(?:pull|issues)/(\d+)\b", re.IGNORECASE
)
IB_TAG_PATTERN = re.compile(
    r"^(CMSSW_\d+_\d+)(?:_(.+))?_X_(\d{4}-\d{2}-\d{2}-\d{4})$"
)
WORKFLOW_STEP_PATTERN = re.compile(r"^(\d{3,6}\.\d+)-(\d+)$")
COMMAND_IMPORT_VERSION = "v2"


def ib_key(tag: str, arch: str) -> str:
    return f"{tag}:{arch}"


def ib_parent_tag(tag: str) -> tuple[str, str | None]:
    """Return an IB's primary run tag and optional sub-IB variant name."""
    match = IB_TAG_PATTERN.fullmatch(tag)
    if not match:
        return tag, None
    release, variant, timestamp = match.groups()
    return f"{release}_X_{timestamp}", variant


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_hash(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


def command_records(entries: dict, tag: str, arch: str) -> list[dict]:
    """Normalize workflow command hashes without inferring a PR relationship."""
    records = []
    for workflow_step, command_hash in entries.items():
        match = WORKFLOW_STEP_PATTERN.fullmatch(str(workflow_step))
        if not match or not isinstance(command_hash, str) or not command_hash:
            continue
        workflow_id, step = match.groups()
        records.append({
            "key": f"{ib_key(tag, arch)}:{workflow_step}",
            "relval_key": f"{ib_key(tag, arch)}:{workflow_id}",
            "workflow_id": workflow_id,
            "workflow_step": workflow_step,
            "step": int(step),
            "command_hash": command_hash,
            "tag": tag,
            "arch": arch,
        })
    return records


def pr_cache_records(entries: dict) -> list[dict]:
    """Normalize cmsrepo's PR-to-commit cache without inventing missing fields."""
    records = []
    for number_text, entry in entries.items():
        try:
            number = int(number_text)
        except (TypeError, ValueError):
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("hash"), str):
            continue
        records.append({
            "number": number,
            "title": entry.get("title") or "",
            "author": entry.get("author") or "unknown",
            "branch": entry.get("branch") or "",
            "user_ref": entry.get("user_ref") or "",
            "sha": entry["hash"],
        })
    return records


def command_source_paths(root: Path, include_history: bool = False) -> list[Path]:
    """Choose current command snapshots; retain all snapshots only on request."""
    paths = [
        path for path in root.rglob("*.json")
        if len(path.relative_to(root).parts) >= 3
    ]
    if include_history:
        return sorted(paths)
    latest: dict[tuple[str, ...], Path] = {}
    for path in paths:
        relative_path = path.relative_to(root)
        source_key = (relative_path.parts[0], *relative_path.parts[2:-1], path.stem)
        previous = latest.get(source_key)
        if previous is None or relative_path.parts[1] > previous.relative_to(root).parts[1]:
            latest[source_key] = path
    return sorted(latest.values())


def command_ib_tag(root: Path, path: Path) -> str:
    relative_path = path.relative_to(root)
    return f"{path.stem}_{relative_path.parts[1]}"


def is_unchanged(previous_hash: str | None, current_hash: str, force: bool = False) -> bool:
    return not force and previous_hash == current_hash


def to_iso(unix_ts) -> str | None:
    if not unix_ts:
        return None
    try:
        return datetime.fromtimestamp(int(unix_ts), tz=timezone.utc).isoformat()
    except (ValueError, TypeError):
        return None


def _source_hash(driver, source_id: str) -> str | None:
    with driver.session() as session:
        record = session.run(
            "MATCH (state:ImportState {source_id:$source_id}) RETURN state.content_hash AS content_hash",
            source_id=source_id,
        ).single()
    return record["content_hash"] if record else None


def source_needs_import(driver, source_id: str, content_hash: str, force: bool = False) -> bool:
    return not is_unchanged(_source_hash(driver, source_id), content_hash, force)


def mark_source_imported(driver, source_id: str, content_hash: str, source_type: str, source_path: Path):
    with driver.session() as session:
        session.run(
            "MERGE (state:ImportState {source_id:$source_id}) "
            "SET state.content_hash=$content_hash, state.source_type=$source_type, "
            "state.source_path=$source_path, state.last_ingested_at=datetime()",
            source_id=source_id,
            content_hash=content_hash,
            source_type=source_type,
            source_path=str(source_path),
        )


def ensure_ib(session, tag: str, arch: str, release: str):
    key = ib_key(tag, arch)
    parent_tag, variant = ib_parent_tag(tag)
    session.run(
        "MERGE (ib:IB {ib_key:$ib_key}) "
        "SET ib.tag=$tag, ib.arch=$arch, ib.release=$release, "
        "ib.parent_tag=$parent_tag, ib.variant=$variant",
        ib_key=key,
        tag=tag,
        arch=arch,
        release=release,
        parent_tag=parent_tag,
        variant=variant,
    )
    return key


def _reset_source_nodes(session, relation: str, label: str, source_id: str, current_ib_key: str):
    session.run(
        f"MATCH (ib:IB {{ib_key:$ib_key}})-[:{relation}]->(node:{label} {{source_id:$source_id}}) "
        "DETACH DELETE node",
        ib_key=current_ib_key,
        source_id=source_id,
    )


def _reconcile_missing_sources(driver, prefix: str, active_sources: set[str]):
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
            session.run(
                "MATCH (node) WHERE node.source_id=$source_id AND NOT node:ImportState "
                "DETACH DELETE node",
                source_id=source_id,
            )
            session.run("MATCH (state:ImportState {source_id:$source_id}) DELETE state", source_id=source_id)
        print(f"  removed missing source: {source_id}")


def _ingest_relval_file(driver, path: Path, arch: str, force: bool) -> tuple[int, str]:
    tag = f"{path.stem}_{path.parent.name}"
    source_id = f"relvals:{tag}:{arch}:{path.resolve()}"
    digest = file_hash(path)
    if not source_needs_import(driver, source_id, digest, force):
        return 0, source_id
    entries = json.loads(path.read_text())
    with driver.session() as session:
        current_ib_key = ensure_ib(session, tag, arch, path.stem)
        _reset_source_nodes(session, "RAN_RELVAL", "RelVal", source_id, current_ib_key)
        for entry in entries:
            steps = entry.get("steps", [])
            workflow_id = str(entry.get("id", ""))
            session.run(
                "MATCH (ib:IB {ib_key:$ib_key}) "
                "MERGE (relval:RelVal {key:$key}) "
                "SET relval.ib_key=$ib_key, relval.ib_tag=$tag, relval.arch=$arch, "
                "relval.workflow_id=$workflow_id, relval.name=$name, relval.status=$status, "
                "relval.exitcode=$exitcode, relval.errors=$errors, relval.warnings=$warnings, "
                "relval.known_error=$known_error, relval.known_error_message=$known_error_message, "
                "relval.source_id=$source_id "
                "MERGE (ib)-[:RAN_RELVAL]->(relval)",
                ib_key=current_ib_key,
                tag=tag,
                arch=arch,
                key=f"{current_ib_key}:{workflow_id}",
                workflow_id=workflow_id,
                name=entry.get("name", ""),
                status="FAILED" if any(step.get("status") == "FAILED" for step in steps) else "PASSED",
                exitcode=entry.get("exitcode"),
                errors=sum(step.get("errors", 0) for step in steps),
                warnings=sum(step.get("warnings", 0) for step in steps),
                # A workflow can be marked "known_error" for reasons unrelated to the
                # code under test (e.g. missing grid-pack shared libraries for a build
                # variant) - the real CMSDT dashboard excludes these from its failing
                # count/list entirely (verified live: workflow 534.0/536.0 on
                # CMSSW_20_1_ASAN_X_2026-08-31-2300/el9_amd64_gcc14 both have known_error=1,
                # "Grid-packs missing asan and non-asan shared libraries", and don't appear
                # on the real dashboard's 5-row failing list even though our graph's
                # step-level status derivation correctly marks them FAILED). Storing this
                # here lets get_failing_relvals match that same exclusion.
                known_error=bool(entry.get("known_error")),
                known_error_message=entry.get("known_error_message"),
                source_id=source_id,
            )
    mark_source_imported(driver, source_id, digest, "relvals", path)
    return len(entries), source_id


def relval_exception_records(entries: dict, tag: str, arch: str) -> list[dict]:
    """Normalize CMSSDT exception-to-workflow mappings for one IB architecture."""
    records = []
    for exception, workflow_ids in entries.items():
        if not isinstance(exception, str) or not isinstance(workflow_ids, list):
            continue
        exception = " ".join(exception.split())
        if not exception:
            continue
        for workflow_id in workflow_ids:
            workflow_id = str(workflow_id)
            records.append({
                "tag": tag,
                "arch": arch,
                "workflow_id": workflow_id,
                "exception": exception,
                "exception_hash": sha256_bytes(exception.encode())[:16],
            })
    return records


def _ingest_relval_exception_file(
    driver, path: Path, arch: str, date_dir: Path, force: bool
) -> tuple[int, str]:
    tag = f"{path.stem.removesuffix('_EXCEPTIONS')}_{date_dir.name}"
    source_id = f"relval-exceptions:{tag}:{arch}:{path.resolve()}"
    digest = file_hash(path)
    if not source_needs_import(driver, source_id, digest, force):
        return 0, source_id
    entries = json.loads(path.read_text())
    if not isinstance(entries, dict):
        raise ValueError(f"RelVal exceptions file must be a JSON object: {path}")
    records = relval_exception_records(entries, tag, arch)
    current_ib_key = ib_key(tag, arch)
    with driver.session() as session:
        session.run(
            "MATCH (:RelVal)-[:HAS_EXCEPTION]->(exception:RelValException {source_id:$source_id}) "
            "DETACH DELETE exception",
            source_id=source_id,
        )
        session.run(
            "UNWIND $records AS record "
            "MATCH (relval:RelVal {key:$ib_key + ':' + record.workflow_id}) "
            "MERGE (exception:RelValException {key:$ib_key + ':' + record.workflow_id + ':' + record.exception_hash}) "
            "SET exception.text=record.exception, exception.ib_key=$ib_key, exception.ib_tag=record.tag, "
            "exception.arch=record.arch, exception.workflow_id=record.workflow_id, exception.source_id=$source_id "
            "MERGE (relval)-[:HAS_EXCEPTION]->(exception)",
            records=records,
            ib_key=current_ib_key,
            source_id=source_id,
        )
    mark_source_imported(driver, source_id, digest, "relval-exceptions", path)
    return len(records), source_id


def ingest_relval_exceptions(driver, root: Path, force: bool = False, reconcile: bool = False):
    if not root.is_dir():
        print(f"  relval exceptions: path not found {root}")
        return 0
    total = 0
    active_sources = set()
    for arch_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for date_dir in sorted(path for path in arch_dir.iterdir() if path.is_dir()):
            for path in sorted(date_dir.rglob("*_EXCEPTIONS.json")):
                count, source_id = _ingest_relval_exception_file(
                    driver, path, arch_dir.name, date_dir, force
                )
                total += count
                active_sources.add(source_id)
    if reconcile:
        _reconcile_missing_sources(driver, "relval-exceptions:", active_sources)
    print(f"  RelVal exceptions imported: {total}")
    return total


LOGREADER_BASE_URL = "https://cmssdt.cern.ch/SDT/cgi-bin/logreader"


def ingest_es_exceptions(driver, path: Path, force: bool = False) -> int:
    """Import RelVal exceptions from an Elasticsearch export (e.g. the output
    of cms-bot's es-cmssw-exceptions.py, redirected to a JSON file) rather
    than manually-staged step logs. One JSON file is one source: it's a batch
    snapshot, not per-IB, so any change to the file reprocesses the whole
    thing (consistent with how every other single-file source here works).

    No source_url line anchor: an ES document's `exception` field is a
    structured value, not a byte offset into a real log file, so there's no
    line range to cite -- the constructed URL points at the run's log file
    without a `#/start-end` fragment.
    """
    if not path.is_file():
        print(f"  ES exceptions: path not found {path}")
        return 0
    source_id = f"es-exceptions:{path.resolve()}"
    digest = file_hash(path)
    if not source_needs_import(driver, source_id, digest, force):
        return 0

    payload = json.loads(path.read_text())
    docs = [hit["_source"] for hit in payload.get("hits", {}).get("hits", [])]

    records = []
    for doc in docs:
        tag = doc.get("release")
        arch = doc.get("architecture")
        workflow_id = doc.get("workflow")
        step = doc.get("step")
        exception_text = doc.get("exception")
        if not (tag and arch and workflow_id and step and exception_text):
            continue
        current_ib_key = ib_key(tag, arch)
        for block in extract_es_exception_blocks(exception_text):
            text = f"{block['category']}: {block['message']}" if block["message"] else block["category"]
            records.append({
                "tag": tag,
                "arch": arch,
                "workflow_id": workflow_id,
                "step": step,
                "category": block["category"],
                "message": block["message"],
                "context": block["context"],
                "text": text,
                "exception_hash": sha256_bytes(text.encode())[:16],
                "ib_key": current_ib_key,
            })

    with driver.session() as session:
        session.run(
            "MATCH (:RelVal)-[:HAS_EXCEPTION]->(exception:RelValException {source_id:$source_id}) "
            "DETACH DELETE exception",
            source_id=source_id,
        )
        session.run(
            "UNWIND $records AS record "
            "MATCH (relval:RelVal {key:record.ib_key + ':' + record.workflow_id}) "
            "MERGE (exception:RelValException "
            "{key:record.ib_key + ':' + record.workflow_id + ':' + record.exception_hash}) "
            "SET exception.text=record.text, exception.category=record.category, "
            "exception.message=record.message, exception.context=record.context, "
            "exception.ib_key=record.ib_key, exception.ib_tag=record.tag, exception.arch=record.arch, "
            "exception.workflow_id=record.workflow_id, exception.step=record.step, "
            "exception.source_url=$logreader_base + '/' + record.arch + '/' + record.tag + "
            "'/pyRelValMatrixLogs/run/' + record.workflow_id + '_' + relval.name + '/' + "
            "record.step + '_' + relval.name + '.log', "
            "exception.source_id=$source_id "
            "MERGE (relval)-[:HAS_EXCEPTION]->(exception)",
            records=records,
            logreader_base=LOGREADER_BASE_URL,
            source_id=source_id,
        )
    mark_source_imported(driver, source_id, digest, "es-exceptions", path)
    print(f"  ES exceptions imported: {len(records)} (from {len(docs)} documents)")
    return len(records)


def merged_pr_records(payload: list) -> list[dict]:
    """Flatten merged_prs_summary.json into (target_tag, number, author_login,
    commit_hash) records. Each `comparisons` entry's `compared_tags` is
    "SOURCE_TAG-->TARGET_TAG"; the PRs in `merged_prs` are new as of the
    TARGET tag -- the same "PRs merged since the previous build" list the
    CMSSDT dashboard shows under each IB tag's tab. Skips malformed entries
    (missing/unparseable `number`, no "-->" in `compared_tags`) the same way
    every other *_records() parser in this file does."""
    records = []
    for branch_entry in payload:
        for comparison in branch_entry.get("comparisons", []):
            compared_tags = comparison.get("compared_tags") or ""
            if "-->" not in compared_tags:
                continue
            target_tag = compared_tags.split("-->")[-1].strip()
            if not target_tag:
                continue
            for pr in comparison.get("merged_prs", []):
                try:
                    number = int(pr.get("number"))
                except (TypeError, ValueError):
                    continue
                records.append({
                    "target_tag": target_tag,
                    "number": number,
                    "author_login": pr.get("author_login"),
                    "commit_hash": pr.get("hash"),
                })
    return records


def ingest_merged_prs(driver, path: Path, force: bool = False) -> int:
    """Populate (PR)-[:MERGED_BEFORE]->(IB) -- see merged_pr_records() for the
    parsing rules. This relationship and the graph_tools.find_prs_before_ib()
    query against it were already designed and registered as an LLM tool;
    only the ingestion side was never built, so it always returned empty.
    This is that missing half, sourced from cms-sw.github.io's
    merged_prs_summary.json (already present in a normal --cmsrepo checkout,
    no separate flag needed).

    One JSON file is one source (same whole-file-hash pattern as
    ingest_es_exceptions): any change to the file reprocesses the whole
    thing. PRs attach to every imported IB node sharing the target tag
    (architecture-independent source-code fact, not tied to one build's
    architecture). A PR only attaches if it's already imported (from
    cmsrepo/data/prs) and the target tag matches an already-imported IB --
    one without the other is silently skipped, not stored, the same rule
    RelVal exceptions already follow for their own cross-references."""
    if not path.is_file():
        print(f"  merged PRs: path not found {path}")
        return 0
    source_id = f"merged-prs:{path.resolve()}"
    digest = file_hash(path)
    if not source_needs_import(driver, source_id, digest, force):
        return 0

    payload = json.loads(path.read_text())
    records = merged_pr_records(payload)

    with driver.session() as session:
        session.run(
            "MATCH (:PR)-[rel:MERGED_BEFORE {source_id:$source_id}]->(:IB) DELETE rel",
            source_id=source_id,
        )
        session.run(
            "UNWIND $records AS record "
            "MATCH (pr:PR {repo:'cmssw', number:record.number}) "
            "MATCH (ib:IB {tag:record.target_tag}) "
            "MERGE (pr)-[rel:MERGED_BEFORE]->(ib) "
            "SET rel.source_id=$source_id, rel.commit_author_login=record.author_login, "
            "rel.commit_hash=record.commit_hash",
            records=records,
            source_id=source_id,
        )
    mark_source_imported(driver, source_id, digest, "merged-prs", path)
    print(f"  Merged-PR build attachments imported: {len(records)}")
    return len(records)


def ingest_relvals(driver, root: Path, force: bool = False, reconcile: bool = False):
    if not root.is_dir():
        print(f"  relvals: path not found {root}")
        return 0
    total = 0
    active_sources = set()
    for arch_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for date_dir in sorted(path for path in arch_dir.iterdir() if path.is_dir()):
            for path in sorted(date_dir.glob("*.json")):
                if path.name.endswith("_EXCEPTIONS.json"):
                    continue
                count, source_id = _ingest_relval_file(driver, path, arch_dir.name, force)
                total += count
                active_sources.add(source_id)
    if reconcile:
        _reconcile_missing_sources(driver, "relvals:", active_sources)
    print(f"  RelVals imported: {total}")
    return total


def relval_default_records(defaults: dict) -> list[dict]:
    """Extract default architectures from cmsrepo's comma-separated descriptors."""
    records = []
    for tag, descriptor in defaults.items():
        if not isinstance(tag, str) or not isinstance(descriptor, str):
            continue
        for entry in descriptor.split(","):
            arch, separator, _ = entry.strip().partition(":")
            if arch and not separator:
                records.append({"tag": tag, "arch": arch})
    return records


def _ingest_default_architectures(
    driver, path: Path, property_name: str, source_type: str, force: bool = False
) -> int:
    """Mark the default architectures from a locally supplied availability mapping."""
    if not path.is_file():
        print(f"  {source_type} defaults: path not found {path}")
        return 0
    source_id = f"{source_type}-defaults:{path.resolve()}"
    digest = file_hash(path)
    if not source_needs_import(driver, source_id, digest, force):
        return 0
    defaults = json.loads(path.read_text())
    if not isinstance(defaults, dict):
        raise ValueError(f"{source_type} defaults file must be a JSON object: {path}")
    records = relval_default_records(defaults)
    with driver.session() as session:
        session.run(f"MATCH (ib:IB) REMOVE ib.{property_name}")
        session.run(
            "UNWIND $records AS record "
            "MATCH (ib:IB {tag:record.tag, arch:record.arch}) "
            f"SET ib.{property_name}=true",
            records=records,
        )
    mark_source_imported(driver, source_id, digest, f"{source_type}-defaults", path)
    print(f"  {source_type} defaults imported: {len(records)}")
    return len(records)


def ingest_relval_defaults(driver, path: Path, force: bool = False) -> int:
    return _ingest_default_architectures(driver, path, "is_relval_default", "RelVal", force)


def ingest_unittest_defaults(driver, path: Path, force: bool = False) -> int:
    return _ingest_default_architectures(driver, path, "is_unittest_default", "UnitTest", force)


def _ingest_unittest_file(driver, path: Path, arch: str, force: bool) -> tuple[int, str]:
    tag = f"{path.stem}_{path.parent.name}"
    source_id = f"unittests:{tag}:{arch}:{path.resolve()}"
    digest = file_hash(path)
    if not source_needs_import(driver, source_id, digest, force):
        return 0, source_id
    entries = json.loads(path.read_text())
    with driver.session() as session:
        current_ib_key = ensure_ib(session, tag, arch, path.stem)
        _reset_source_nodes(session, "RAN_UNITTEST", "UnitTest", source_id, current_ib_key)
        for entry in entries:
            name = entry.get("name", "")
            errors = entry.get("errors", 0)
            session.run(
                "MATCH (ib:IB {ib_key:$ib_key}) "
                "MERGE (test:UnitTest {key:$key}) "
                "SET test.ib_key=$ib_key, test.ib_tag=$tag, test.arch=$arch, test.name=$name, "
                "test.status=$status, test.errors=$errors, test.ok=$ok, test.source_id=$source_id "
                "MERGE (ib)-[:RAN_UNITTEST]->(test) "
                "MERGE (package:Package {name:$name}) "
                "MERGE (test)-[:IN_PACKAGE]->(package)",
                ib_key=current_ib_key,
                tag=tag,
                arch=arch,
                key=f"{current_ib_key}:{name}",
                name=name,
                status="FAILED" if errors > 0 else "PASSED",
                errors=errors,
                ok=entry.get("ok", 0),
                source_id=source_id,
            )
    mark_source_imported(driver, source_id, digest, "unittests", path)
    return len(entries), source_id


def ingest_unittests(driver, root: Path, force: bool = False, reconcile: bool = False):
    if not root.is_dir():
        print(f"  unitTests: path not found {root}")
        return 0
    total = 0
    active_sources = set()
    for arch_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for date_dir in sorted(path for path in arch_dir.iterdir() if path.is_dir()):
            for path in sorted(date_dir.glob("*.json")):
                count, source_id = _ingest_unittest_file(driver, path, arch_dir.name, force)
                total += count
                active_sources.add(source_id)
    if reconcile:
        _reconcile_missing_sources(driver, "unittests:", active_sources)
    print(f"  UnitTests imported: {total}")
    return total


def build_status_records(entries: list[dict] | dict) -> list[dict]:
    """Extract per-architecture compilation and addon-test status from a CMS
    queue summary. addon_status is a whole-architecture pass/fail flag only
    (from the separate 'addons' list in the same source entry, same shape
    Release Explorer's summarizeArchCell already renders as a plain badge
    with no per-test detail) -- the addOnTests.log files themselves are never
    staged or parsed here, so no per-test name/error/exit-code exists,
    unlike RelVal/UnitTest."""
    release_name = None
    if isinstance(entries, dict):
        release_name = entries.get("release_name")
        entries = entries.get("comparisons", [])
    records = []
    for entry in entries:
        tag = entry.get("release_name") or release_name
        if not tag:
            continue
        addon_by_arch = {
            addon.get("arch"): addon.get("passed")
            for addon in entry.get("addons", []) if addon.get("arch")
        }
        for build in entry.get("builds", []):
            arch = build.get("arch")
            if not arch:
                continue
            details = build.get("details") or {}
            addon_passed = addon_by_arch.get(arch)
            records.append({
                "tag": tag,
                "arch": arch,
                "release": re.sub(r"_\d{4}-\d{2}-\d{2}-\d{4}$", "", tag),
                "parent_tag": ib_parent_tag(tag)[0],
                "variant": ib_parent_tag(tag)[1],
                "status": build.get("passed"),
                "comp_warnings": details.get("compWarning", 0),
                "ignored_warnings": details.get("ignoreWarning", 0),
                "addon_status": None if addon_passed is None else ("passed" if addon_passed else "error"),
            })
    return records


def _ingest_build_status_file(driver, path: Path, force: bool) -> tuple[int, str]:
    source_id = f"build-status:{path.resolve()}"
    digest = file_hash(path)
    if not source_needs_import(driver, source_id, digest, force):
        return 0, source_id
    records = build_status_records(json.loads(path.read_text()))
    with driver.session() as session:
        session.run(
            "MATCH (ib:IB {build_status_source_id:$source_id}) "
            "REMOVE ib.build_status, ib.comp_warnings, ib.ignored_warnings, ib.addon_status, "
            "ib.build_status_source_id",
            source_id=source_id,
        )
        session.run(
            "UNWIND $records AS raw "
            "MERGE (ib:IB {ib_key:raw.tag + ':' + raw.arch}) "
            "SET ib.tag=raw.tag, ib.arch=raw.arch, ib.release=raw.release, "
            "ib.parent_tag=raw.parent_tag, ib.variant=raw.variant, "
            "ib.build_status=raw.status, ib.comp_warnings=raw.comp_warnings, "
            "ib.ignored_warnings=raw.ignored_warnings, ib.addon_status=raw.addon_status, "
            "ib.build_status_source_id=$source_id",
            records=records,
            source_id=source_id,
        )
    mark_source_imported(driver, source_id, digest, "build-status", path)
    return len(records), source_id


def ingest_build_statuses(driver, root: Path, force: bool = False) -> int:
    if not root.is_dir():
        print(f"  build status: path not found {root}")
        return 0
    total = 0
    for path in sorted(root.glob("CMSSW_*.json")):
        count, _ = _ingest_build_status_file(driver, path, force)
        total += count
    print(f"  Build statuses imported: {total}")
    return total


def _ingest_pr_file(driver, path: Path, force: bool, reconcile: bool) -> tuple[int, str]:
    release = path.stem
    source_id = f"prs:{release}:{path.resolve()}"
    digest = file_hash(path)
    if not source_needs_import(driver, source_id, digest, force):
        return 0, source_id
    data = json.loads(path.read_text())
    imported = 0
    current_pr_keys = []
    with driver.session() as session:
        for repo in ("cmssw", "cmsdist"):
            for number_text, pr in data.get(repo, {}).items():
                try:
                    number = int(number_text)
                except ValueError:
                    continue
                pr_key = f"{repo}:{number}"
                current_pr_keys.append(pr_key)
                content_hash = json_hash(pr)
                existing = session.run(
                    "MATCH (pr:PR {repo:$repo, number:$number}) RETURN pr.content_hash AS content_hash",
                    repo=repo,
                    number=number,
                ).single()
                if not force and existing and existing["content_hash"] == content_hash:
                    continue
                session.run(
                    "MERGE (pr:PR {repo:$repo, number:$number}) "
                    "SET pr.pr_key=$pr_key, pr.title=$title, pr.body=$body, pr.merged_at=$merged_at, "
                    "pr.release=$release, pr.changed_files=$changed_files, pr.additions=$additions, "
                    "pr.deletions=$deletions, pr.sha=$sha, pr.labels=$labels, "
                    "pr.github_state=$github_state, "
                    "pr.content_hash=$content_hash, pr.source_id=$source_id "
                    "MERGE (author:Author {name:$author}) "
                    "MERGE (pr)-[:AUTHORED_BY]->(author)",
                    repo=repo,
                    number=number,
                    pr_key=pr_key,
                    title=pr.get("title"),
                    body=pr.get("body") or "",
                    merged_at=to_iso(pr.get("merged_at")),
                    release=release,
                    changed_files=pr.get("changed_files"),
                    additions=pr.get("additions"),
                    deletions=pr.get("deletions"),
                    sha=pr.get("merge_commit_sha"),
                    labels=json.dumps(pr.get("labels", [])),
                    github_state=pr.get("state") or ("closed" if pr.get("merged_at") else "unknown"),
                    content_hash=content_hash,
                    source_id=source_id,
                    author=pr.get("author", "unknown"),
                )
                imported += 1
        if reconcile:
            session.run(
                "MATCH (pr:PR {source_id:$source_id}) WHERE NOT pr.pr_key IN $pr_keys "
                "DETACH DELETE pr",
                source_id=source_id,
                pr_keys=current_pr_keys,
            )
    mark_source_imported(driver, source_id, digest, "prs", path)
    return imported, source_id


def ingest_prs(driver, root: Path, force: bool = False, reconcile: bool = False):
    if not root.is_dir():
        print(f"  prs: path not found {root}")
        return 0
    total = 0
    active_sources = set()
    for path in sorted(root.glob("*.json")):
        count, source_id = _ingest_pr_file(driver, path, force, reconcile)
        total += count
        active_sources.add(source_id)
    if reconcile:
        _reconcile_missing_sources(driver, "prs:", active_sources)
    print(f"  PRs imported: {total}")
    return total


def ingest_pr_cache(driver, path: Path, force: bool = False) -> list[dict]:
    """Import cmsrepo's supplied PR-to-commit mapping without replacing PR text."""
    if not path.is_file():
        print(f"  PR cache: path not found {path}")
        return []
    source_id = f"pr-cache:{path.resolve()}"
    digest = file_hash(path)
    records = pr_cache_records(json.loads(path.read_text()))
    if not source_needs_import(driver, source_id, digest, force):
        return records
    with driver.session() as session:
        for batch in _chunks(records, 500):
            session.run(
                "UNWIND $records AS raw "
                "MERGE (pr:PR {repo:'cmssw', number:raw.number}) "
                "ON CREATE SET pr.pr_key='cmssw:' + toString(raw.number), pr.title=raw.title, "
                "pr.body='', pr.github_state='unknown' "
                "SET pr.cache_branch=raw.branch, pr.cache_user_ref=raw.user_ref, "
                "pr.cache_source_id=$source_id "
                "MERGE (author:Author {name:raw.author}) "
                "MERGE (pr)-[:AUTHORED_BY]->(author) "
                "MERGE (commit:Commit {sha:raw.sha}) "
                "SET commit.source_id=$source_id "
                "MERGE (pr)-[:MERGED_AS {source:'cmsrepo-pr-cache'}]->(commit)",
                records=batch,
                source_id=source_id,
            )
    mark_source_imported(driver, source_id, digest, "pr-cache", path)
    print(f"  PR cache imported: {len(records)}")
    return records


def exit_code_records(entries: dict) -> list[dict]:
    """Normalize cmsrepo's CMSSW exit-code table. A later duplicate key in the
    source file wins, matching how json.loads already resolved it."""
    records = []
    for code_text, name in entries.items():
        try:
            code = int(code_text)
        except (TypeError, ValueError):
            continue
        if not isinstance(name, str) or not name:
            continue
        records.append({"code": code, "name": name})
    return records


def ingest_exit_codes(driver, path: Path, force: bool = False) -> int:
    """Import CMSSW's exit-code-to-category table (e.g. 16640 -> OtherCMS),
    sourced from cms-sw.github.io. Static reference data, not IB-scoped."""
    if not path.is_file():
        print(f"  exit codes: path not found {path}")
        return 0
    source_id = f"exit-codes:{path.resolve()}"
    digest = file_hash(path)
    if not source_needs_import(driver, source_id, digest, force):
        return 0
    records = exit_code_records(json.loads(path.read_text()))
    with driver.session() as session:
        session.run(
            "UNWIND $records AS record "
            "MERGE (code:ExitCode {code:record.code}) "
            "SET code.name=record.name, code.source_id=$source_id",
            records=records,
            source_id=source_id,
        )
    mark_source_imported(driver, source_id, digest, "exit-codes", path)
    print(f"  Exit codes imported: {len(records)}")
    return len(records)


def ingest_commands(
    driver, root: Path, pr_cache: list[dict], force: bool = False,
    reconcile: bool = False, include_history: bool = False,
):
    """Import CMSSDT workflow command hashes and exact cache-commit matches."""
    if not root.is_dir():
        print(f"  commands: path not found {root}")
        return 0
    cache_by_prefix: dict[str, str | None] = {}
    for record in pr_cache:
        prefix = record["sha"][:16]
        cache_by_prefix[prefix] = record["sha"] if prefix not in cache_by_prefix else None
    total = 0
    active_sources = set()
    for path in command_source_paths(root, include_history):
        relative_path = path.relative_to(root)
        if len(relative_path.parts) < 3:
            continue
        arch = relative_path.parts[0]
        tag = command_ib_tag(root, path)
        source_id = f"commands:{path.resolve()}"
        digest = sha256_bytes(f"{COMMAND_IMPORT_VERSION}:{file_hash(path)}".encode())
        active_sources.add(source_id)
        if not source_needs_import(driver, source_id, digest, force):
            continue
        entries = json.loads(path.read_text())
        if not isinstance(entries, dict):
            raise ValueError(f"Commands file must be a JSON object: {path}")
        records = command_records(entries, tag, arch)
        for record in records:
            record["source_id"] = source_id
            record["matched_commit_sha"] = cache_by_prefix.get(record["command_hash"])
        with driver.session() as session:
            session.run(
                "MATCH (command:WorkflowCommand {source_id:$source_id}) DETACH DELETE command",
                source_id=source_id,
            )
            for batch in _chunks(records, 500):
                session.run(
                    "UNWIND $records AS raw "
                    "MERGE (command:WorkflowCommand {key:raw.key}) "
                    "SET command.tag=raw.tag, command.arch=raw.arch, command.workflow_id=raw.workflow_id, "
                    "command.workflow_step=raw.workflow_step, command.step=raw.step, "
                    "command.command_hash=raw.command_hash, command.source_id=raw.source_id "
                    "WITH raw, command "
                    "OPTIONAL MATCH (relval:RelVal {key:raw.relval_key}) "
                    "FOREACH (target IN CASE WHEN relval IS NULL THEN [] ELSE [relval] END | "
                    "MERGE (target)-[:USED_COMMAND {source:'cmsrepo-commands'}]->(command)) "
                    "WITH raw, command "
                    "OPTIONAL MATCH (commit:Commit {sha:raw.matched_commit_sha}) "
                    "FOREACH (target IN CASE WHEN commit IS NULL THEN [] ELSE [commit] END | "
                    "MERGE (command)-[:MATCHES_CACHE_COMMIT {source:'cmsrepo-pr-cache'}]->(target))",
                    records=batch,
                )
        mark_source_imported(driver, source_id, digest, "commands", path)
        total += len(records)
    if reconcile:
        _reconcile_missing_sources(driver, "commands:", active_sources)
    print(f"  Workflow commands imported: {total}")
    return total


def github_pr_from_issue(issue: dict) -> dict | None:
    """Normalize a GitHub issue payload when it represents a pull request."""
    if "pull_request" not in issue or not isinstance(issue.get("number"), int):
        return None
    return {
        "number": issue["number"],
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
        "merged_at": issue.get("pull_request", {}).get("merged_at"),
        "release": (issue.get("milestone") or {}).get("title"),
        "labels": [label["name"] for label in issue.get("labels", []) if label.get("name")],
        "author": (issue.get("user") or {}).get("login") or "unknown",
        "state": issue.get("state"),
        "url": issue.get("html_url"),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
    }


def github_issue_from_issue(issue: dict) -> dict | None:
    """Normalize a GitHub issue payload when it is not a pull request."""
    if "pull_request" in issue or not isinstance(issue.get("number"), int):
        return None
    return {
        "number": issue["number"],
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
        "state": issue.get("state"),
        "url": issue.get("html_url"),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
        "labels": [label["name"] for label in issue.get("labels", []) if label.get("name")],
        "author": (issue.get("user") or {}).get("login") or "unknown",
    }


def github_reference_numbers(text: str) -> list[int]:
    text = text or ""
    return sorted({
        *(int(number) for number in GITHUB_REFERENCE_PATTERN.findall(text)),
        *(int(number) for number in GITHUB_ITEM_URL_PATTERN.findall(text)),
    })


def _raw_github_files(root: Path):
    if root.is_file() and tarfile.is_tarfile(root):
        with tarfile.open(root, "r:*") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(("/issue.json", "/comments.json")):
                    continue
                stream = archive.extractfile(member)
                if stream:
                    yield member.name, json.load(stream)
        return
    if root.is_dir():
        for path in sorted(root.rglob("issue.json")):
            yield str(path), json.loads(path.read_text())
            comments_path = path.with_name("comments.json")
            if comments_path.is_file():
                yield str(comments_path), json.loads(comments_path.read_text())
        return
    raise FileNotFoundError(f"GitHub issues source not found: {root}")


def _chunks(items, size: int):
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def _import_raw_pr_batch(session, records: list[dict], force: bool) -> int:
    source_ids = [record["source_id"] for record in records]
    hashes = {
        record["source_id"]: record["content_hash"]
        for record in session.run(
            "MATCH (state:ImportState) WHERE state.source_id IN $source_ids "
            "RETURN state.source_id AS source_id, state.content_hash AS content_hash",
            source_ids=source_ids,
        )
    }
    changed = [record for record in records if force or hashes.get(record["source_id"]) != record["content_hash"]]
    if not changed:
        return 0
    session.run(
        "UNWIND $records AS raw "
        "MERGE (pr:PR {repo:'cmssw', number:raw.number}) "
        "SET pr.pr_key='cmssw:' + toString(raw.number), pr.title=raw.title, pr.body=raw.body, "
        "pr.merged_at=raw.merged_at, pr.release=coalesce(pr.release, raw.release), "
        "pr.labels=raw.labels, pr.github_state=raw.state, pr.github_url=raw.url, "
        "pr.github_created_at=raw.created_at, pr.github_updated_at=raw.updated_at, "
        "pr.raw_content_hash=raw.content_hash, pr.raw_source_id=raw.source_id "
        "MERGE (author:Author {name:raw.author}) "
        "MERGE (pr)-[:AUTHORED_BY]->(author) "
        "WITH pr, raw UNWIND raw.label_names AS label_name "
        "MERGE (label:Label {key:'cmssw:' + label_name}) SET label.name=label_name, label.repo='cmssw' "
        "MERGE (pr)-[:HAS_LABEL]->(label)",
        records=changed,
    )
    session.run(
        "UNWIND $records AS raw "
        "MERGE (state:ImportState {source_id:raw.source_id}) "
        "SET state.content_hash=raw.content_hash, state.source_type='github-pr', "
        "state.source_path=raw.source_path, state.last_ingested_at=datetime()",
        records=changed,
    )
    return len(changed)


def _import_raw_issue_batch(session, records: list[dict], force: bool) -> int:
    source_ids = [record["source_id"] for record in records]
    hashes = {
        record["source_id"]: record["content_hash"]
        for record in session.run(
            "MATCH (state:ImportState) WHERE state.source_id IN $source_ids "
            "RETURN state.source_id AS source_id, state.content_hash AS content_hash",
            source_ids=source_ids,
        )
    }
    changed = [record for record in records if force or hashes.get(record["source_id"]) != record["content_hash"]]
    if not changed:
        return 0
    session.run(
        "UNWIND $records AS raw "
        "MERGE (issue:Issue {issue_key:'cmssw:' + toString(raw.number)}) "
        "SET issue.repo='cmssw', issue.number=raw.number, issue.title=raw.title, issue.body=raw.body, "
        "issue.state=raw.state, issue.github_url=raw.url, issue.created_at=raw.created_at, "
        "issue.updated_at=raw.updated_at, issue.closed_at=raw.closed_at, issue.labels=raw.labels, "
        "issue.raw_content_hash=raw.content_hash, issue.raw_source_id=raw.source_id "
        "MERGE (author:Author {name:raw.author}) "
        "MERGE (issue)-[:AUTHORED_BY]->(author) "
        "WITH issue, raw UNWIND raw.label_names AS label_name "
        "MERGE (label:Label {key:'cmssw:' + label_name}) SET label.name=label_name, label.repo='cmssw' "
        "MERGE (issue)-[:HAS_LABEL]->(label)",
        records=changed,
    )
    session.run(
        "UNWIND $records AS raw "
        "MERGE (state:ImportState {source_id:raw.source_id}) "
        "SET state.content_hash=raw.content_hash, state.source_type='github-issue', "
        "state.source_path=raw.source_path, state.last_ingested_at=datetime()",
        records=changed,
    )
    return len(changed)


def _import_raw_comment_batch(session, records: list[dict], force: bool) -> int:
    source_ids = [record["source_id"] for record in records]
    hashes = {
        record["source_id"]: record["content_hash"]
        for record in session.run(
            "MATCH (state:ImportState) WHERE state.source_id IN $source_ids "
            "RETURN state.source_id AS source_id, state.content_hash AS content_hash",
            source_ids=source_ids,
        )
    }
    changed = [record for record in records if force or hashes.get(record["source_id"]) != record["content_hash"]]
    if not changed:
        return 0
    session.run(
        "MATCH (comment:GitHubComment) WHERE comment.source_id IN $source_ids DETACH DELETE comment",
        source_ids=[record["source_id"] for record in changed],
    )
    comments = [comment for record in changed for comment in record["comments"]]
    if comments:
        session.run(
            "UNWIND $comments AS raw "
            "MATCH (item) WHERE item.repo='cmssw' AND item.number=raw.item_number "
            "AND ((raw.item_kind='PR' AND item:PR) OR (raw.item_kind='Issue' AND item:Issue)) "
            "MERGE (comment:GitHubComment {key:raw.key}) "
            "SET comment.body=raw.body, comment.created_at=raw.created_at, "
            "comment.updated_at=raw.updated_at, comment.author=raw.author, comment.source_id=raw.source_id "
            "MERGE (author:Author {name:raw.author}) "
            "MERGE (item)-[:HAS_COMMENT]->(comment) "
            "MERGE (comment)-[:AUTHORED_BY]->(author)",
            comments=comments,
        )
    session.run(
        "UNWIND $records AS raw "
        "MERGE (state:ImportState {source_id:raw.source_id}) "
        "SET state.content_hash=raw.content_hash, state.source_type='github-comments', "
        "state.source_path=raw.source_path, state.last_ingested_at=datetime()",
        records=changed,
    )
    return len(changed)


def refresh_github_body_references(driver):
    """Rebuild explicit #number references after all PRs and issues are present."""
    with driver.session() as session:
        session.run(
            "MATCH (source)-[reference:REFERENCES {source:'github_body'}]->() "
            "WHERE source:PR OR source:Issue DELETE reference"
        )
        items = session.run(
            "MATCH (item) WHERE item:PR OR item:Issue "
            "RETURN CASE WHEN item:PR THEN 'PR' ELSE 'Issue' END AS kind, "
            "item.repo AS repo, item.number AS number, item.body AS body"
        ).data()
        records = [
            {
                "source_kind": item["kind"],
                "repo": item["repo"],
                "number": item["number"],
                "references": [number for number in github_reference_numbers(item["body"]) if number != item["number"]],
            }
            for item in items
        ]
        for batch in _chunks(records, 250):
            session.run(
                "UNWIND $records AS raw "
                "MATCH (source) WHERE source.repo=raw.repo AND source.number=raw.number "
                "AND ((raw.source_kind='PR' AND source:PR) OR (raw.source_kind='Issue' AND source:Issue)) "
                "UNWIND raw.references AS reference_number "
                "MATCH (target) WHERE target.repo=raw.repo AND target.number=reference_number "
                "AND (target:PR OR target:Issue) "
                "MERGE (source)-[:REFERENCES {source:'github_body'}]->(target)",
                records=batch,
            )


def refresh_github_comment_references(driver):
    """Rebuild explicit PR and Issue references found in GitHub comment text."""
    with driver.session() as session:
        session.run(
            "MATCH (:GitHubComment)-[reference:REFERENCES {source:'github_comment'}]->() DELETE reference"
        )
        comments = session.run(
            "MATCH (comment:GitHubComment) RETURN comment.key AS key, comment.body AS body"
        ).data()
        records = []
        for comment in comments:
            references = github_reference_numbers(comment["body"])
            if references:
                records.append({"key": comment["key"], "references": references})
        for batch in _chunks(records, 250):
            session.run(
                "UNWIND $records AS raw "
                "MATCH (source:GitHubComment {key:raw.key}) "
                "UNWIND raw.references AS reference_number "
                "MATCH (target) WHERE target.repo='cmssw' AND target.number=reference_number "
                "AND (target:PR OR target:Issue) "
                "MERGE (source)-[:REFERENCES {source:'github_comment'}]->(target)",
                records=batch,
            )


def ingest_github_issues(driver, root: Path, force: bool = False):
    """Import issue/PR metadata and comments from a raw GitHub dump or tar archive."""
    imported_prs = imported_issues = imported_comments = 0
    pr_records: list[dict] = []
    issue_records: list[dict] = []
    comment_records: list[dict] = []
    items_by_directory: dict[str, tuple[str, int]] = {}
    pending_comments: dict[str, tuple[str, list]] = {}

    def queue_comments(directory: str, source_path: str, comments: list):
        item = items_by_directory.get(directory)
        if item is None:
            pending_comments[directory] = (source_path, comments)
            return
        item_kind, number = item
        source_id = f"github-comments:cmssw:{item_kind}:{number}"
        comment_records.append({
            "source_id": source_id,
            "source_path": source_path,
            "content_hash": json_hash(comments),
            "comments": [
                {
                    "key": f"cmssw:{number}:{comment['id']}",
                    "item_kind": item_kind,
                    "item_number": number,
                    "body": comment.get("body") or "",
                    "created_at": comment.get("created_at"),
                    "updated_at": comment.get("updated_at"),
                    "author": (comment.get("user") or {}).get("login") or "unknown",
                    "source_id": source_id,
                }
                for comment in comments if comment.get("id") is not None
            ],
        })

    with driver.session() as session:
        for source_path, payload in _raw_github_files(root):
            directory = str(Path(source_path).parent)
            if source_path.endswith("/issue.json") or source_path.endswith("issue.json"):
                pr = github_pr_from_issue(payload)
                issue = github_issue_from_issue(payload)
                if pr:
                    items_by_directory[directory] = ("PR", pr["number"])
                    pr_records.append({
                        **pr,
                        "source_id": f"github-pr:cmssw:{pr['number']}",
                        "source_path": source_path,
                        "content_hash": json_hash(payload),
                        "label_names": pr["labels"],
                        "labels": json.dumps(pr["labels"]),
                    })
                elif issue:
                    items_by_directory[directory] = ("Issue", issue["number"])
                    issue_records.append({
                        **issue,
                        "source_id": f"github-issue:cmssw:{issue['number']}",
                        "source_path": source_path,
                        "content_hash": json_hash(payload),
                        "label_names": issue["labels"],
                        "labels": json.dumps(issue["labels"]),
                    })
                else:
                    continue
                if directory in pending_comments:
                    comment_path, comments = pending_comments.pop(directory)
                    queue_comments(directory, comment_path, comments)
            elif isinstance(payload, list):
                queue_comments(directory, source_path, payload)

            if len(pr_records) >= 250:
                imported_prs += _import_raw_pr_batch(session, pr_records, force)
                pr_records.clear()
            if len(issue_records) >= 250:
                imported_issues += _import_raw_issue_batch(session, issue_records, force)
                issue_records.clear()
            if len(comment_records) >= 250:
                if pr_records:
                    imported_prs += _import_raw_pr_batch(session, pr_records, force)
                    pr_records.clear()
                if issue_records:
                    imported_issues += _import_raw_issue_batch(session, issue_records, force)
                    issue_records.clear()
                imported_comments += _import_raw_comment_batch(session, comment_records, force)
                comment_records.clear()

        for directory, (source_path, comments) in pending_comments.items():
            queue_comments(directory, source_path, comments)
        for batch in _chunks(pr_records, 250):
            imported_prs += _import_raw_pr_batch(session, batch, force)
        for batch in _chunks(issue_records, 250):
            imported_issues += _import_raw_issue_batch(session, batch, force)
        for batch in _chunks(comment_records, 250):
            imported_comments += _import_raw_comment_batch(session, batch, force)
    if imported_prs or imported_issues or force:
        refresh_github_body_references(driver)
    if imported_comments or force:
        refresh_github_comment_references(driver)
    print(
        f"  GitHub raw PRs imported: {imported_prs}; issues imported: {imported_issues}; "
        f"comment files imported: {imported_comments}"
    )
    return imported_prs, imported_issues, imported_comments


def discover_clang_builds(root: Path) -> list[tuple[str, str, Path]]:
    if not root.is_dir():
        return []
    builds = []
    for tag_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for arch_dir in sorted(path for path in tag_dir.iterdir() if path.is_dir()):
            build_logs = arch_dir / "build-logs"
            if build_logs.is_dir():
                builds.append((tag_dir.name, arch_dir.name, build_logs))
    return builds


def _ingest_clang_log(driver, path: Path, build_logs: Path, tag: str, arch: str, force: bool) -> tuple[int, str]:
    source_id = f"clang:{tag}:{arch}:{path.resolve()}"
    digest = file_hash(path)
    if not source_needs_import(driver, source_id, digest, force):
        return 0, source_id
    package = path.parent.relative_to(build_logs).as_posix()
    warnings = aggregate_occurrences(extract_warnings(html_to_text(path.read_text(errors="replace"))))
    with driver.session() as session:
        current_ib_key = ensure_ib(session, tag, arch, "_".join(tag.split("_")[:4]))
        _reset_source_nodes(session, "REPORTED", "Warning", source_id, current_ib_key)
        session.run("MERGE (package:Package {name:$package})", package=package)
        for warning in warnings:
            base_hash = sha256_bytes(
                f"{package}||{warning['file']}||{warning['line']}||{warning['message']}".encode()
            )[:16]
            session.run(
                "MATCH (ib:IB {ib_key:$ib_key}) "
                "MATCH (package:Package {name:$package}) "
                "MERGE (source:SourceFile {path:$file}) "
                "MERGE (checker:Checker {name:$flag}) "
                "MERGE (warning:Warning {key:$key}) "
                "SET warning.hash=$base_hash, warning.file=$file, warning.line=$line, "
                "warning.message=$message, warning.flag=$flag, warning.count=$count, "
                "warning.package=$package, warning.ib_key=$ib_key, warning.source_id=$source_id "
                "MERGE (package)-[:CONTAINS]->(source) "
                "MERGE (source)-[:FLAGGED_BY]->(checker) "
                "MERGE (ib)-[:HAS_WARNING_IN]->(package) "
                "MERGE (ib)-[:REPORTED]->(warning) "
                "MERGE (warning)-[:IN_PACKAGE]->(package) "
                "MERGE (warning)-[:IN_FILE]->(source) "
                "MERGE (warning)-[:CHECKED_BY]->(checker)",
                ib_key=current_ib_key,
                package=package,
                file=warning["file"],
                line=warning["line"],
                message=warning["message"],
                flag=warning["flag"],
                count=warning["count"],
                base_hash=base_hash,
                key=f"{current_ib_key}:{base_hash}",
                source_id=source_id,
            )
    mark_source_imported(driver, source_id, digest, "clang", path)
    return sum(warning["count"] for warning in warnings), source_id


def ingest_clang(driver, build_logs: Path, tag: str, arch: str, force: bool = False, reconcile: bool = False):
    if not build_logs.is_dir():
        print(f"  clang: path not found {build_logs}")
        return 0
    total = 0
    active_sources = set()
    logs = sorted(build_logs.rglob("log.html"))
    for path in logs:
        count, source_id = _ingest_clang_log(driver, path, build_logs, tag, arch, force)
        total += count
        active_sources.add(source_id)
    if reconcile:
        _reconcile_missing_sources(driver, f"clang:{tag}:{arch}:", active_sources)
    print(f"  Clang {tag} {arch}: {len(logs)} logs scanned, {total} occurrences imported")
    return total


def ingest_clang_root(driver, root: Path, force: bool = False, reconcile: bool = False):
    builds = discover_clang_builds(root)
    total = 0
    for tag, arch, build_logs in builds:
        total += ingest_clang(driver, build_logs, tag, arch, force, reconcile)
    print(f"  Clang builds discovered: {len(builds)}")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmsrepo", required=True)
    parser.add_argument("--github-issues", help="Raw GitHub issues directory or .tar/.tar.gz archive")
    parser.add_argument("--clang-root", help="Root containing <ib-tag>/<arch>/build-logs")
    parser.add_argument("--clang-logs", help="One build's build-logs directory (legacy targeted mode)")
    parser.add_argument("--ib-tag", help="Required with --clang-logs")
    parser.add_argument("--arch", default="el9_amd64_gcc13")
    parser.add_argument("--clear", action="store_true", help="wipe the graph first")
    parser.add_argument("--force", action="store_true", help="reimport unchanged source files")
    parser.add_argument("--reconcile", action="store_true", help="remove records from sources no longer on disk")
    parser.add_argument("--command-history", action="store_true", help="import all historical command snapshots")
    parser.add_argument(
        "--es-exceptions-json",
        help="JSON file from cms-bot's es-cmssw-exceptions.py (e.g. `es-cmssw-exceptions.py > exceptions.JSON`), "
        "imported as RelVal exceptions -- no CERN SSO or manual log staging needed",
    )
    args = parser.parse_args()
    if args.clang_logs and not args.ib_tag:
        parser.error("--clang-logs requires --ib-tag")

    driver = get_driver()
    if args.clear:
        clear_graph(driver)
    setup_schema(driver)
    cmsrepo = Path(args.cmsrepo)
    print("\nIncremental ingest into Neo4j...")
    ingest_build_statuses(driver, cmsrepo / "data", args.force)
    ingest_relvals(driver, cmsrepo / "data" / "relvals", args.force, args.reconcile)
    ingest_relval_defaults(driver, cmsrepo / "data" / "RelvalsAvailableResults.json", args.force)
    ingest_unittest_defaults(driver, cmsrepo / "data" / "UnitTestsAvailableResults.json", args.force)
    ingest_relval_exceptions(driver, cmsrepo / "data" / "relvals", args.force, args.reconcile)
    if args.es_exceptions_json:
        ingest_es_exceptions(driver, Path(args.es_exceptions_json), args.force)
    ingest_unittests(driver, cmsrepo / "data" / "unitTests", args.force, args.reconcile)
    ingest_prs(driver, cmsrepo / "data" / "prs", args.force, args.reconcile)
    pr_cache = ingest_pr_cache(driver, cmsrepo / "cmssw-prs-cache.json", args.force)
    ingest_merged_prs(driver, cmsrepo / "data" / "merged_prs_summary.json", args.force)
    ingest_exit_codes(driver, cmsrepo / "exitcodes.json", args.force)
    ingest_commands(
        driver, cmsrepo / "data" / "commands", pr_cache, args.force,
        args.reconcile, args.command_history,
    )
    if args.github_issues:
        ingest_github_issues(driver, Path(args.github_issues), args.force)
    if args.clang_root:
        ingest_clang_root(driver, Path(args.clang_root), args.force, args.reconcile)
    if args.clang_logs:
        ingest_clang(driver, Path(args.clang_logs), args.ib_tag, args.arch, args.force, args.reconcile)
    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
