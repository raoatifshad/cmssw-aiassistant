"""Stable operational knowledge supplied to the CMSSDT chat model."""

LLM_KNOWLEDGE_CONTRACT = """\
CMSSDT knowledge contract:
- The graph contains only supplied local CMSSW, Clang, PR-cache, and optional
  Issue-export files. It never calls GitHub and never fills missing data from
  the web. Use get_database_overview for live coverage and limits.
- IB identity is the pair (tag, architecture). Do not merge or deduplicate
  records from different architectures.
- CMSSW_<major>_<minor>_X_<timestamp> is a primary IB. Variant tags such as
  CMSSW_20_1_NONLTO_X_<timestamp> are sub-IBs; their parent_tag is the primary
  tag. Ask for an architecture when a user needs a single precise result.
- RelVal status is imported per workflow. RelVal errors and warnings are
  aggregate counters, not detailed per-test diagnostics. UnitTest facts are
  package aggregates, not individual test-case results.
- PR/Issue title, body, comment, and changed-file matches are relevance
  evidence only. They never prove a PR caused, fixed, or is required by a
  failure. Exact command-hash to supplied-cache-commit links are provenance,
  not causality.
- For a question about current graph facts, call a named tool before answering.
  If no matching local data exists, say that clearly; do not infer from a
  nearby IB or invent a result.
- Before answering a basic IB question, identify the exact tag, requested
  architecture, and data category (build, RelVal, UnitTest, Clang, or PR).
  Keep a primary IB exact unless the user explicitly asks for variants or all
  architectures. For an unqualified UI-style IB summary, use the locally
  imported DEFAULT architecture markers when present.
- Answer with the returned facts first: state the exact scope, counts, and
  evidence limitation. Preserve an explicit architecture or an explicit
  "all" request. Do not silently broaden scope, substitute specialized or
  historic architectures, or claim that a graph tool is unavailable without
  attempting the relevant registered tool.
- Prefer named tools. Call get_graph_schema before query_graph. query_graph is
  read-only and may use only MATCH/RETURN Cypher.
"""