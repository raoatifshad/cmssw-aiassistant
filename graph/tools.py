"""Cypher query tools for the CMSSW knowledge graph.

These are the functions the LLM calls (via the tool-calling protocol) to
answer questions. Each one runs a Cypher query against Neo4j and returns
structured results. Read-only — no MERGE/CREATE/DELETE/SET in any query
the LLM can trigger.

Two kinds, same rationale as the old SQL system:
  - Named tools for common questions (fast, predictable)
  - A general query_graph tool for everything else (the model writes Cypher)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from neo4j_conn import get_driver  # noqa: E402

GRAPH_SCHEMA = """
NODES:
    (:IB {ib_key, tag, release, arch, parent_tag, variant})
  (:Package {name})
  (:Warning {hash, file, line, message, flag, count, package})
  (:SourceFile {path})
  (:Checker {name})           — e.g. 'threadsafety.ConstCast', 'cms.FunctionChecker'
    (:PR {repo, number, title, body, merged_at, release, changed_files, additions, deletions, sha, labels})
    (:Issue {issue_key, repo, number, title, body, state, github_url, created_at, updated_at, closed_at, labels})
    (:GitHubComment {key, body, created_at, updated_at, author, source_id})
    (:Label {key, name})
  (:Author {name})
    (:RelVal {key, ib_key, ib_tag, arch, workflow_id, name, status, errors, warnings, exitcode})
    (:RelValException {key, ib_key, ib_tag, arch, workflow_id, step, category, message, context, text, source_url})
    (:ExitCode {code, name}) — static CMSSW exit-code table, e.g. 16640 -> 'OtherCMS'; not IB-scoped
    (:Concept {concept_id, question, definition, source_feedback_id, verification_method,
        status, expires_at, last_revalidated_at, created_at})
        — a correction that passed the same graph-evidence verification gate as an
        accepted UserFeedback row, stored as a durable fact. status is 'active' or
        'revoked'; an active Concept is periodically re-verified against current graph
        facts before expires_at (see ingestion/revalidate_feedback.py) and revoked if it
        no longer holds -- never trusted indefinitely from a single automated check, and
        never promoted by user agreement alone.
    (:AlertRule {rule_id, name, capability, params, condition, channel, owner, active, created_at})
    (:UnitTest {key, ib_key, ib_tag, arch, name, status, errors, ok})
    (:ImportState {source_id, content_hash, source_type, source_path, last_ingested_at})

EDGES:
  (IB)-[:REPORTED]->(Warning)
  (IB)-[:HAS_WARNING_IN]->(Package)
  (Warning)-[:IN_PACKAGE]->(Package)
  (Warning)-[:IN_FILE]->(SourceFile)
  (Warning)-[:CHECKED_BY]->(Checker)
  (Package)-[:CONTAINS]->(SourceFile)
  (SourceFile)-[:FLAGGED_BY]->(Checker)
  (IB)-[:RAN_RELVAL]->(RelVal)
  (RelVal)-[:HAS_EXCEPTION]->(RelValException)
  (IB)-[:RAN_UNITTEST]->(UnitTest)
  (UnitTest)-[:IN_PACKAGE]->(Package)
  (PR)-[:AUTHORED_BY]->(Author)
    (Issue)-[:AUTHORED_BY]->(Author)
    ((PR|Issue))-[:HAS_COMMENT]->(GitHubComment)
    ((PR|Issue))-[:HAS_LABEL]->(Label)
    ((PR|Issue))-[:REFERENCES {source:'github_body'}]->((PR|Issue))
    (GitHubComment)-[:REFERENCES {source:'github_comment'}]->((PR|Issue))
    (PR)-[:MERGED_BEFORE]->(IB)
    (PR)-[:CHANGED_FILE {evidence:'github_changed_file'}]->(SourceFile)
    (PR)-[:MATCHES_WARNING {evidence:'changed_file'}]->(Warning)

MATCHES_WARNING means a PR changed the warning's source file. It is a
potential attribution signal, not proof that the PR introduced the warning.
CHANGED_FILE and PR title matches are weaker discovery signals when no warning
edge is available; callers receive their evidence types explicitly.
PR descriptions are searchable as pr_description evidence.
GitHub REFERENCES edges represent an explicit #number or CMSSW GitHub URL in a
body or comment. They do not establish causality, a dependency, or a fix.
UnitTest results are package-level aggregates. A PR test-file match in a failed
package is evidence of relevance, not a failure for a specific test name.
An IB with no variant is a primary run. An IB with a variant (for example,
NONLTO or CLANG) is a sub-IB whose parent_tag identifies its primary run.
RelValException category/message/context come from EDM Fatal Exception text
in a cms-bot es-cmssw-exceptions.py Elasticsearch export (--es-exceptions-json;
the CMSSDT logreader itself requires CERN SSO, so exceptions are sourced from
ES instead). An exception only exists here if both its RelVal and its
release/architecture/workflow were present in that export -- coverage is
whatever the export covered, not automatically comprehensive. source_url
points at the run's log file with no line-range anchor, since ES stores a
structured field, not a byte offset into a real file.
"""

MAX_ROWS = 100
_WRITE_KEYWORDS = re.compile(
    r"\b(MERGE|CREATE|DELETE|SET|REMOVE|DROP|DETACH|CALL\s+db\.)\b", re.IGNORECASE
)


def _enforce_read_only(cypher: str):
    if _WRITE_KEYWORDS.search(cypher):
        raise ValueError("Only read queries (MATCH/RETURN) are allowed.")


def _run_query(cypher: str, params: dict | None = None) -> list[dict]:
    _enforce_read_only(cypher)
    driver = get_driver()
    try:
        with driver.session() as s:
            result = s.run(cypher, params or {})
            records = [dict(r) for r in result]
            return records[:MAX_ROWS]
    finally:
        driver.close()


# ── Named tools ──────────────────────────────────────────────────────

def get_graph_schema() -> str:
    """Return the knowledge graph schema (node labels, properties, edge types).
    Call this before writing Cypher for query_graph."""
    return GRAPH_SCHEMA


def get_database_overview() -> dict:
    """Return live graph coverage, including node counts and newest RelVal IB records."""
    node_counts = _run_query(
        "MATCH (node) UNWIND labels(node) AS label "
        "RETURN label, count(*) AS count ORDER BY label"
    )
    latest_relval_records = _run_query(
        "MATCH (ib:IB)-[:RAN_RELVAL]->(relval:RelVal) "
        "WITH ib, count(relval) AS relval_total "
        "RETURN ib.tag AS tag, ib.parent_tag AS parent_tag, ib.arch AS arch, relval_total "
        "ORDER BY right(ib.tag, 15) DESC, tag, arch LIMIT 20"
    )
    architectures = _run_query(
        "MATCH (ib:IB)-[:RAN_RELVAL]->(:RelVal) "
        "RETURN DISTINCT ib.arch AS arch ORDER BY arch"
    )
    return {
        "node_counts": {row["label"]: row["count"] for row in node_counts},
        "latest_relval_records": latest_relval_records,
        "relval_architectures": [row["arch"] for row in architectures],
        "limits": {
            "relval_errors": "aggregate workflow counters, not detailed diagnostics",
            "unit_tests": "package aggregates, not individual test-case results",
            "github_evidence": "local textual or provenance evidence only; not causality",
        },
    }


_IB_TAG_SORT_PATTERN = re.compile(
    r"^CMSSW_(\d+)_(\d+)(?:_.+)?_X_(\d{4}-\d{2}-\d{2}-\d{4})$"
)


def _ib_sort_key(tag: str) -> tuple:
    """Rank by release series (major, minor) first, then by build timestamp.

    A higher release series (e.g. 20_1) is "later" than a lower one (e.g. 20_0)
    even if that series' most recent build is older in wall-clock time — this
    matches how the IB dashboard orders releases.
    """
    match = _IB_TAG_SORT_PATTERN.match(tag)
    if not match:
        return (-1, -1, tag)
    major, minor, timestamp = match.groups()
    return (int(major), int(minor), timestamp)


def list_ib_tags(release: str | None = None, arch: str | None = None, limit: int = 20) -> list[dict]:
    """List known IB tags, most recent release series first, newest build within a series first."""
    limit = min(limit, MAX_ROWS)
    cypher = (
        "MATCH (ib:IB) WHERE ($rel IS NULL OR ib.release = $rel) "
        "AND ($arch IS NULL OR ib.arch = $arch) "
        "RETURN ib.tag AS tag, ib.arch AS arch, ib.release AS release"
    )
    # Fetch unsliced: _run_query truncates to MAX_ROWS before we can sort, which
    # would drop the actual top entries when there are more IBs than MAX_ROWS.
    driver = get_driver()
    try:
        with driver.session() as s:
            rows = [dict(r) for r in s.run(cypher, {"rel": release, "arch": arch})]
    finally:
        driver.close()
    rows.sort(key=lambda row: _ib_sort_key(row["tag"]), reverse=True)
    return rows[:limit]


_CONCEPT_STOPWORDS = {
    "what", "this", "that", "have", "does", "with", "from", "about", "which",
    "there", "would", "could", "should", "your", "when", "where", "will",
}


def search_concepts(question: str, limit: int = 5) -> list[dict]:
    """Search stored expert-corrected facts (:Concept) for ones relevant to a
    question. These are corrections that already passed the same verification
    gate as an accepted UserFeedback row -- durable graph facts, not just
    conversation history."""
    limit = min(limit, MAX_ROWS)
    terms = [
        word for word in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", question)
        if word.lower() not in _CONCEPT_STOPWORDS
    ]
    if not terms:
        return []
    return _run_query(
        "MATCH (c:Concept) WHERE coalesce(c.status, 'active') = 'active' "
        "WITH c, [term IN $terms WHERE toLower(c.question) CONTAINS toLower(term) "
        "OR toLower(c.definition) CONTAINS toLower(term)] AS matched_terms "
        "WHERE size(matched_terms) > 0 "
        "RETURN c.question AS question, c.definition AS definition, "
        "size(matched_terms) AS match_count "
        "ORDER BY match_count DESC, c.created_at DESC LIMIT $limit",
        {"terms": terms, "limit": limit},
    )


def get_relval_recurrence(workflow_id: str) -> dict:
    """How often has this workflow failed across every imported IB, not just
    the one currently being discussed? Computed at query time from data
    already ingested -- no scheduled job needed, always current."""
    rows = _run_query(
        "MATCH (ib:IB)-[:RAN_RELVAL]->(relval:RelVal {workflow_id:$workflow_id}) "
        "WITH relval.name AS name, ib, relval "
        "ORDER BY ib.tag DESC "
        "RETURN name, count(relval) AS total_appearances, "
        "sum(CASE WHEN relval.status = 'FAILED' THEN 1 ELSE 0 END) AS failed_appearances, "
        "collect(CASE WHEN relval.status = 'FAILED' THEN ib.tag ELSE null END)[0..10] AS failed_tags",
        {"workflow_id": workflow_id},
    )
    if not rows or not rows[0]["total_appearances"]:
        return {"workflow_id": workflow_id, "total_appearances": 0}
    row = rows[0]
    return {
        "workflow_id": workflow_id,
        "name": row["name"],
        "total_appearances": row["total_appearances"],
        "failed_appearances": row["failed_appearances"],
        "failed_tags": [tag for tag in row["failed_tags"] if tag],
    }


def get_exit_code_names() -> dict[int, str]:
    """Return the full CMSSW exit-code -> category-name table (e.g. 16640 ->
    'OtherCMS'), imported from cms-sw.github.io's exitcodes.json. Small static
    reference data — one query fetches the whole table rather than one lookup
    per row. Bypasses _run_query's MAX_ROWS cap: the table has ~160 entries,
    already over that cap, and silently dropping some would misreport exit
    codes as unknown rather than fail loudly.
    """
    driver = get_driver()
    try:
        with driver.session() as s:
            rows = [dict(r) for r in s.run("MATCH (code:ExitCode) RETURN code.code AS code, code.name AS name")]
    finally:
        driver.close()
    return {row["code"]: row["name"] for row in rows}


def count_warnings_by_package(package: str, ib_tag: str | None = None, arch: str | None = None) -> dict:
    """How many warnings does a package have — both distinct locations and
    total occurrences (these are different numbers, confirmed concretely)."""
    if ib_tag:
        result = _run_query(
            "MATCH (ib:IB {tag:$t})-[:REPORTED]->(w:Warning)-[:IN_PACKAGE]->(pkg:Package {name:$pkg}) "
            "WHERE $arch IS NULL OR ib.arch = $arch "
            "RETURN count(w) AS distinct_locations, sum(w.count) AS total_occurrences",
            {"t": ib_tag, "pkg": package, "arch": arch},
        )
    else:
        result = _run_query(
            "MATCH (w:Warning)-[:IN_PACKAGE]->(pkg:Package {name:$pkg}) "
            "RETURN count(w) AS distinct_locations, sum(w.count) AS total_occurrences",
            {"pkg": package},
        )
    row = result[0] if result else {"distinct_locations": 0, "total_occurrences": 0}
    row["package"] = package
    return row


def get_warnings_by_package(package: str, ib_tag: str | None = None, arch: str | None = None, limit: int = 50) -> list[dict]:
    """Get individual warning details for a package. Use count_warnings_by_package
    for counting questions — do NOT count these rows."""
    limit = min(limit, 50)
    if ib_tag:
        return _run_query(
            "MATCH (ib:IB {tag:$t})-[:REPORTED]->(w:Warning)-[:IN_PACKAGE]->(pkg:Package {name:$pkg}) "
            "WHERE $arch IS NULL OR ib.arch = $arch "
            "RETURN w.file AS file, w.line AS line, w.flag AS flag, w.message AS message, "
            "       w.count AS occurrences "
            "ORDER BY w.count DESC LIMIT $lim",
            {"t": ib_tag, "pkg": package, "arch": arch, "lim": limit},
        )
    return _run_query(
        "MATCH (w:Warning)-[:IN_PACKAGE]->(pkg:Package {name:$pkg}) "
        "RETURN w.file AS file, w.line AS line, w.flag AS flag, w.message AS message, "
        "       w.count AS occurrences "
        "ORDER BY w.count DESC LIMIT $lim",
        {"pkg": package, "lim": limit},
    )


def top_packages_by_warnings(ib_tag: str | None = None, arch: str | None = None, limit: int = 20) -> list[dict]:
    """Packages ranked by total warning occurrences."""
    limit = min(limit, MAX_ROWS)
    if ib_tag:
        return _run_query(
            "MATCH (ib:IB {tag:$t})-[:REPORTED]->(w:Warning)-[:IN_PACKAGE]->(pkg:Package) "
            "WHERE $arch IS NULL OR ib.arch = $arch "
            "RETURN pkg.name AS package, count(w) AS distinct_locations, sum(w.count) AS total "
            "ORDER BY total DESC LIMIT $lim",
            {"t": ib_tag, "arch": arch, "lim": limit},
        )
    return _run_query(
        "MATCH (w:Warning)-[:IN_PACKAGE]->(pkg:Package) "
        "RETURN pkg.name AS package, count(w) AS distinct_locations, sum(w.count) AS total "
        "ORDER BY total DESC LIMIT $lim",
        {"lim": limit},
    )


def get_failing_relvals(ib_tag: str, arch: str | None = None, limit: int = 50) -> list[dict]:
    """Failed RelVals for an IB, including sub-IBs when given a primary tag."""
    limit = min(limit, MAX_ROWS)
    return _run_query(
        "MATCH (ib:IB)-[:RAN_RELVAL]->(rv:RelVal {status:'FAILED'}) "
        "WHERE (ib.tag = $t OR ib.parent_tag = $t) "
        "AND ($arch IS NULL OR ib.arch = $arch) "
        "RETURN ib.tag AS ib_tag, ib.variant AS variant, rv.name AS name, rv.workflow_id AS workflow_id, "
        "rv.errors AS errors, rv.exitcode AS exitcode "
        "ORDER BY ib.tag, rv.workflow_id LIMIT $lim",
        {"t": ib_tag, "arch": arch, "lim": limit},
    )


def get_failing_unittests(ib_tag: str, arch: str | None = None, limit: int = 50) -> list[dict]:
    """Unit tests with status=FAILED for a given IB."""
    limit = min(limit, MAX_ROWS)
    return _run_query(
        "MATCH (ib:IB {tag:$t})-[:RAN_UNITTEST]->(ut:UnitTest {status:'FAILED'}) "
        "WHERE $arch IS NULL OR ib.arch = $arch "
        "RETURN ut.name AS name, ut.errors AS errors "
        "LIMIT $lim",
        {"t": ib_tag, "arch": arch, "lim": limit},
    )


def find_prs_before_ib(ib_tag: str, arch: str | None = None, limit: int = 20) -> list[dict]:
    """PRs that merged right before this IB — the first-pass attribution."""
    limit = min(limit, MAX_ROWS)
    return _run_query(
        "MATCH (p:PR)-[:MERGED_BEFORE]->(ib:IB {tag:$t}) "
        "WHERE $arch IS NULL OR ib.arch = $arch "
        "OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author) "
        "RETURN p.repo AS repo, p.number AS pr_number, p.title AS title, "
        "       p.merged_at AS merged_at, a.name AS author "
        "ORDER BY p.merged_at DESC LIMIT $lim",
        {"t": ib_tag, "arch": arch, "lim": limit},
    )


def search_issues_by_text(term: str, include_comments: bool = False, limit: int = 20) -> dict:
    """Search CMSSW Issue titles/bodies, optionally comments; return total and open counts."""
    limit = min(limit, MAX_ROWS)
    rows = _run_query(
        "MATCH (issue:Issue {repo:'cmssw'}) "
        "WHERE toLower(coalesce(issue.title, '')) CONTAINS toLower($term) "
        "OR toLower(coalesce(issue.body, '')) CONTAINS toLower($term) "
        "OR ($include_comments AND EXISTS { MATCH (issue)-[:HAS_COMMENT]->(comment:GitHubComment) "
        "WHERE toLower(coalesce(comment.body, '')) CONTAINS toLower($term) } "
        ") "
        "WITH issue, "
        "CASE WHEN toLower(coalesce(issue.title, '')) CONTAINS toLower($term) THEN ['title'] ELSE [] END "
        "+ CASE WHEN toLower(coalesce(issue.body, '')) CONTAINS toLower($term) THEN ['body'] ELSE [] END "
        "+ CASE WHEN $include_comments AND EXISTS { MATCH (issue)-[:HAS_COMMENT]->(comment:GitHubComment) "
        "WHERE toLower(coalesce(comment.body, '')) CONTAINS toLower($term) } THEN ['comment'] ELSE [] END "
        "AS matched_fields "
        "ORDER BY issue.updated_at DESC, issue.number DESC "
        "WITH collect({number:issue.number, title:issue.title, state:issue.state, "
        "updated_at:issue.updated_at, matched_fields:matched_fields}) AS issues, "
        "count(issue) AS total_matches, "
        "sum(CASE WHEN toLower(coalesce(issue.state, '')) = 'open' THEN 1 ELSE 0 END) AS open_matches "
        "RETURN total_matches, open_matches, issues[0..$limit] AS issues",
        {"term": term, "include_comments": include_comments, "limit": limit},
    )
    return rows[0] if rows else {"total_matches": 0, "open_matches": 0, "issues": []}


def find_prs_for_error(error: str, ib_tag: str | None = None, limit: int = 20) -> list[dict]:
    """Find PRs from warnings, test packages, files, titles, and descriptions.
    The evidence field distinguishes source-warning matches from weaker discovery
    signals. None of the results prove that a PR caused an error."""
    limit = min(limit, MAX_ROWS)
    warning_match = (
        "MATCH (ib:IB {tag:$ib_tag})-[:REPORTED]->(w:Warning)"
        "<-[link:MATCHES_WARNING]-(p:PR) "
        if ib_tag
        else "MATCH (p:PR)-[link:MATCHES_WARNING]->(w:Warning) "
    )
    warning_rows = _run_query(
        warning_match
        + "WHERE toLower(w.message) CONTAINS toLower($error) "
        "OR toLower(w.flag) CONTAINS toLower($error) "
        "OR toLower(w.file) CONTAINS toLower($error) "
        "OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author) "
        "RETURN p.repo AS repo, p.number AS pr_number, p.title AS title, "
        "a.name AS author, p.merged_at AS merged_at, count(DISTINCT w) AS matched_warnings, "
        "sum(w.count) AS matched_occurrences, "
        "collect(DISTINCT w.file)[0..5] AS matched_files, "
        "0 AS matched_unit_tests, 0 AS matched_unit_test_errors, [] AS matched_unit_test_packages, "
        "[] AS matched_unit_test_ibs, "
        "'warning_changed_file' AS evidence "
        "ORDER BY matched_occurrences DESC, matched_warnings DESC LIMIT $limit",
        {"error": error, "ib_tag": ib_tag, "limit": limit},
    )
    title_rows = _run_query(
        "MATCH (p:PR) WHERE toLower(p.title) CONTAINS toLower($error) "
        "OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author) "
        "RETURN p.repo AS repo, p.number AS pr_number, p.title AS title, "
        "a.name AS author, p.merged_at AS merged_at, 0 AS matched_warnings, "
        "0 AS matched_occurrences, [] AS matched_files, 0 AS matched_unit_tests, "
        "0 AS matched_unit_test_errors, [] AS matched_unit_test_packages, [] AS matched_unit_test_ibs, "
        "'pr_title' AS evidence "
        "LIMIT $limit",
        {"error": error, "limit": limit},
    )
    description_rows = _run_query(
        "MATCH (p:PR) WHERE toLower(coalesce(p.body, '')) CONTAINS toLower($error) "
        "OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author) "
        "RETURN p.repo AS repo, p.number AS pr_number, p.title AS title, "
        "a.name AS author, p.merged_at AS merged_at, 0 AS matched_warnings, "
        "0 AS matched_occurrences, [] AS matched_files, 0 AS matched_unit_tests, "
        "0 AS matched_unit_test_errors, [] AS matched_unit_test_packages, [] AS matched_unit_test_ibs, "
        "'pr_description' AS evidence LIMIT $limit",
        {"error": error, "limit": limit},
    )
    file_rows = _run_query(
        "MATCH (p:PR)-[:CHANGED_FILE]->(sf:SourceFile) "
        "WHERE toLower(sf.path) CONTAINS toLower($error) "
        "OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author) "
        "RETURN p.repo AS repo, p.number AS pr_number, p.title AS title, "
        "a.name AS author, p.merged_at AS merged_at, 0 AS matched_warnings, "
        "0 AS matched_occurrences, collect(DISTINCT sf.path)[0..5] AS matched_files, "
        "0 AS matched_unit_tests, 0 AS matched_unit_test_errors, "
        "[] AS matched_unit_test_packages, [] AS matched_unit_test_ibs, "
        "'changed_file_path' AS evidence LIMIT $limit",
        {"error": error, "limit": limit},
    )
    test_rows = _run_query(
        "MATCH (ib:IB)-[:RAN_UNITTEST]->(ut:UnitTest {status:'FAILED'})-[:IN_PACKAGE]->(pkg:Package) "
        "MATCH (p:PR)-[:CHANGED_FILE]->(sf:SourceFile) "
        "WHERE ($ib_tag IS NULL OR ib.tag = $ib_tag) "
        "AND (toLower(p.title) CONTAINS toLower($error) "
        "OR toLower(sf.path) CONTAINS toLower($error)) "
        "AND replace(sf.path, 'src/', '') STARTS WITH pkg.name + '/' "
        "WITH DISTINCT p, ut, pkg, ib "
        "OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author) "
        "RETURN p.repo AS repo, p.number AS pr_number, p.title AS title, "
        "a.name AS author, p.merged_at AS merged_at, 0 AS matched_warnings, "
        "0 AS matched_occurrences, [] AS matched_files, count(ut) AS matched_unit_tests, "
        "sum(ut.errors) AS matched_unit_test_errors, "
        "collect(DISTINCT pkg.name)[0..5] AS matched_unit_test_packages, "
        "collect(DISTINCT ib.tag)[0..5] AS matched_unit_test_ibs, "
        "'failed_unit_test_package' AS evidence LIMIT $limit",
        {"error": error, "ib_tag": ib_tag, "limit": limit},
    )

    merged = {}
    for row in warning_rows + test_rows + file_rows + title_rows + description_rows:
        key = (row["repo"], row["pr_number"])
        existing = merged.get(key)
        if not existing:
            row["evidence"] = [row["evidence"]]
            merged[key] = row
            continue
        existing["evidence"].append(row["evidence"])
        existing["matched_warnings"] = max(existing["matched_warnings"], row["matched_warnings"])
        existing["matched_occurrences"] = max(existing["matched_occurrences"], row["matched_occurrences"])
        existing["matched_files"] = list(dict.fromkeys(existing["matched_files"] + row["matched_files"]))[:5]
        existing["matched_unit_tests"] = max(existing["matched_unit_tests"], row["matched_unit_tests"])
        existing["matched_unit_test_errors"] = max(
            existing["matched_unit_test_errors"], row["matched_unit_test_errors"]
        )
        existing["matched_unit_test_packages"] = list(dict.fromkeys(
            existing["matched_unit_test_packages"] + row["matched_unit_test_packages"]
        ))[:5]
        existing["matched_unit_test_ibs"] = list(dict.fromkeys(
            existing["matched_unit_test_ibs"] + row["matched_unit_test_ibs"]
        ))[:5]

    return sorted(
        merged.values(),
        key=lambda row: (row["matched_occurrences"], row["matched_warnings"], row["merged_at"] or ""),
        reverse=True,
    )[:limit]


def shared_header_packages(file_path: str) -> list[dict]:
    """Which packages are all affected by warnings in the same shared header?
    This is the genuinely graph-shaped question — a natural traversal, not a SQL join."""
    return _run_query(
        "MATCH (pkg:Package)-[:CONTAINS]->(sf:SourceFile {path:$fp}) "
        "RETURN pkg.name AS package ORDER BY pkg.name",
        {"fp": file_path},
    )


def warnings_by_checker(ib_tag: str | None = None, limit: int = 20) -> list[dict]:
    """Warnings grouped by checker type (e.g. threadsafety.ConstCast, cms.FunctionChecker)."""
    limit = min(limit, MAX_ROWS)
    if ib_tag:
        return _run_query(
            "MATCH (ib:IB {tag:$t})-[:REPORTED]->(w:Warning)-[:CHECKED_BY]->(chk:Checker) "
            "RETURN chk.name AS checker, count(w) AS distinct_locations, sum(w.count) AS total "
            "ORDER BY total DESC LIMIT $lim",
            {"t": ib_tag, "lim": limit},
        )
    return _run_query(
        "MATCH (w:Warning)-[:CHECKED_BY]->(chk:Checker) "
        "RETURN chk.name AS checker, count(w) AS distinct_locations, sum(w.count) AS total "
        "ORDER BY total DESC LIMIT $lim",
        {"lim": limit},
    )


def query_graph(cypher: str) -> dict:
    """Run a read-only Cypher query. Call get_graph_schema() first to see
    available node labels, properties, and edge types."""
    try:
        rows = _run_query(cypher)
        return {"ok": True, "rows": rows, "count": len(rows)}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# Tool registry for the agent
TOOLS = {
    "get_graph_schema": get_graph_schema,
    "get_database_overview": get_database_overview,
    "list_ib_tags": list_ib_tags,
    "count_warnings_by_package": count_warnings_by_package,
    "get_warnings_by_package": get_warnings_by_package,
    "top_packages_by_warnings": top_packages_by_warnings,
    "get_failing_relvals": get_failing_relvals,
    "get_failing_unittests": get_failing_unittests,
    "find_prs_before_ib": find_prs_before_ib,
    "search_issues_by_text": search_issues_by_text,
    "find_prs_for_error": find_prs_for_error,
    "shared_header_packages": shared_header_packages,
    "warnings_by_checker": warnings_by_checker,
    "query_graph": query_graph,
}
