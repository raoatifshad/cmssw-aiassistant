# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A standalone, Neo4j-backed chatbot for CMSSW Integration Build (IB) data —
clang build warnings, RelVals, unit tests, and GitHub PRs/Issues. It is
completely independent from an existing SQL-based chatbot (shares no code, no
database, no imports) but exposes the same API shape (`POST /api/chat`,
`GET /api/health`) so the existing React chat widget works by changing one
env var (`VITE_CHAT_API_BASE`).

## Commands

```bash
# setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in CERN_API_BASE, CERN_API_KEY

# Neo4j (standalone docker run, port 7688 bolt via compose or 7687 direct)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/cmssdt2026 -v $HOME/neo4j-data:/data docker.io/library/neo4j:5

# build/refresh the graph (incremental — hashes each source file, skips unchanged)
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/build_graph.py \
  --cmsrepo cmsrepo --clang-root /home/rao/CHAT-BOT-AI/CLANG-LOGS --reconcile
# add --clear only for a one-time full rebuild; --force to reimport unchanged sources

# run the API
uvicorn server.api_server:app --reload --port 8001
curl http://localhost:8001/api/health

# full stack (Neo4j + OpenSearch + Logstash + API) via Compose
./start.sh    # detects Podman or Docker
./stop.sh     # stops services, preserves volumes

# tests (unittest, not pytest — no pytest.ini/conftest.py in this repo)
# PYTHONPATH=ingestion is required: build_graph.py imports sibling modules
# (e.g. clang_parser) by bare name, which only resolves with ingestion/ itself on the path.
PYTHONPATH=ingestion python -m unittest discover -s tests
PYTHONPATH=ingestion python -m unittest tests.test_chat_routing -v
PYTHONPATH=ingestion python -m unittest tests.test_chat_routing.LatestIbRankingTests.test_sort_key_ranks_release_series_before_timestamp
```

Tests mock the Neo4j driver / LLM client rather than requiring a live
database — see the `unittest.mock.patch` usage at the top of
`tests/test_chat_routing.py` and `tests/test_graph_tools.py` for the pattern.

## Architecture

### Two independent halves under one FastAPI app

1. **Knowledge graph chat** (`ingestion/`, `graph/`, most of `server/`) —
   deterministic Q&A over CMSSW IB facts stored in Neo4j.
2. **Log RAG platform** (`log_rag/`, `logstash/`) — a separate durable
   pipeline for ingesting arbitrary JSON/JSONL/Clang logs, archiving them,
   projecting event context into Neo4j, and retrieving cited evidence
   (`/api/ingest`, `/api/search`). Both share the same Neo4j instance and the
   same `/api/chat` response's `citations` field, but are otherwise decoupled.

### Ingestion (`ingestion/build_graph.py`)

Reads raw source trees and incrementally MERGEs them into Neo4j. Every
`ingest_*` function follows the same shape: hash the source file/content,
skip if `source_needs_import` says the hash is unchanged (via `ImportState`
nodes), otherwise MERGE records and call `mark_source_imported`. `--reconcile`
additionally deletes graph records whose backing source file is gone
(`_reconcile_missing_sources`). Sources:

- `cmsrepo/data/{relvals,commands,prs}` — sparse-checkout of
  `cms-sw/cms-sw.github.io`, expected at `--cmsrepo`.
- `--clang-root` — a separately-staged tree of `<ib-tag>/<arch>/build-logs/**/log.html`
  (kept outside `cmsrepo`, parsed by `ingestion/clang_parser.py`).
- `--es-exceptions-json` — RelVal exception text, sourced from an
  Elasticsearch export (cms-bot's `es-cmssw-exceptions.py`), parsed by
  `ingestion/relval_log_parser.py`. This is the **only** supported source for
  `RelValException` records — there is no longer a manually-staged
  `pyRelValMatrixLogs` step-log path (removed; the CMSSDT logreader requires
  CERN SSO so staging individual logs didn't scale). An exception only
  attaches if its release/architecture/workflow already matches an ingested
  `RelVal`.
- `--github-issues` — optional raw CMSSW GitHub Issues export
  (dir or `.tar`/`.tar.gz`), imports PR/Issue bodies, labels, comments, and
  explicit `#number`/URL cross-references. Never calls GitHub live.

**IB identity is always the pair `(tag, architecture)`** — never merge or
dedupe across architectures. `CMSSW_<major>_<minor>_X_<timestamp>` is a
*primary* IB; a tag with an inserted variant (e.g.
`CMSSW_20_1_NONLTO_X_<timestamp>`) is a *sub-IB* whose `parent_tag` points at
the primary tag, tracked via `ib_parent_tag()`.

### Graph schema and tools (`graph/tools.py`)

`GRAPH_SCHEMA` (module-level string) is the single source of truth for node
labels, properties, and edges — it's injected into every LLM system prompt,
so update it whenever the schema changes. `TOOLS` is a name→function registry
of read-only Cypher query functions exposed to the LLM via function-calling;
`query_graph()` is the escape hatch for anything not covered by a named tool,
restricted to `MATCH`/`RETURN` only (`_enforce_read_only` rejects
MERGE/CREATE/DELETE/SET/REMOVE/DROP/`CALL db.`).

### Request routing (`server/api_server.py`, `server/request_planner.py`)

`POST /api/chat` is a long deterministic-first waterfall, in priority order:

1. **`RequestPlanner`** (`server/request_planner.py`) — extracts stable
   entities from the question via regex (`RequestContext.extract`: IB tags,
   architecture, PR/workflow numbers, category keywords) with **no LLM
   involved**, then scores registered `Capability` definitions to pick a
   route, or emits a `Clarification` (e.g. ambiguous multi-tag question) that
   is persisted per-session (`ChatPendingPlan` nodes) and resolved on the next
   turn.
2. If the planner finds no deterministic capability, ~30 more `_*_answer()`
   helper functions in `api_server.py` are tried in a fixed fallback order
   (each returns `None` if it doesn't match) — this is legacy routing
   predating the `RequestPlanner` and still covers many capabilities the
   planner doesn't score yet (PR evidence, exception search, artifact search,
   etc.). Read the big `if`-chain in `chat()` (~line 2757) to see exact order;
   first non-`None` match wins.
3. Only if **nothing** matches does it fall through to the LLM path: a
   constrained, non-executable `propose_investigation_plan` call picks a
   registered capability + tool calls (validated by `validate_llm_plan`
   against real capability/tool names — the LLM can never issue arbitrary
   Cypher or an unvalidated tool), `execute_read_only_plan` runs them, and
   `synthesize_evidence_answer` turns the evidence into prose. The LLM is
   never allowed to answer from its own knowledge — only from tool evidence.

Two experimental phases are gated by env vars, each adding a real per-message
LLM call: `ENABLE_ROUTER_SHADOW` (logs an independent LLM classification
against the deterministic planner's choice for later promotion-gate
comparison, never used to answer — this deployment's `.env` has it **on**,
deliberately, to accumulate real agreement data instead of hand-patching
another regex for every new phrasing forever; summarize with
`ingestion/summarize_router_shadow.py`) and `ENABLE_ANSWER_SYNTHESIS` (LLM
rephrases an already-correct deterministic answer, verified to preserve
every number via `_extract_numbers` before being shown — falls back to the
original text on any mismatch; still off). Don't assume either is off by
default when writing tests — `tests/test_router_shadow.py` learned this the
hard way; test the parsing/behavior explicitly with `@patch(...)`, never by
asserting the live imported constant.

### Agent loop (`server/agent.py`)

`KGChatSession` drives the LLM-fallback and feedback-verification paths. It
prefers native OpenAI-style tool-calling but falls back to parsing a
`<tool_call>{...}</tool_call>` text block for models that don't emit
structured tool calls (`uses_structured_tools` is false only for
`"gpt-5.6-luna"` — check this list before assuming a new model needs the
text-fallback prompt injected).

### Conversation state & feedback

`POST /api/chat` persists user/assistant turns as `ChatSession`/`ChatTurn`
nodes (`_store_chat_turns`) and per-question routing decisions as
`ChatPlanDecision` nodes (`_store_planning_decision`) — both purely additive,
never touching imported CMSSW/Clang/GitHub facts. `POST /api/feedback`
verifies a submitted correct/incorrect verdict via `_verify_feedback`
(`server/api_server.py`): latest-IB questions are checked deterministically
against the current imported tag; everything else goes through an LLM
verifier (`KGChatSession` with read-only graph tools only,
`_llm_feedback_verification`) that must return an evidence-based
`accepted`/`rejected`/`pending` decision — feedback never overwrites imported
facts. An `accepted` correction (`verdict=incorrect` + `corrected_answer`)
is promoted to a durable `Concept` fact by `_maybe_create_concept` — **only**
because the graph confirmed it, never from user agreement/corroboration
alone (a corroboration-based design was deliberately rejected — trusting how
many users repeat a claim is not evidence, only the graph is). That trust
isn't permanent either: `Concept` nodes carry `status`/`expires_at`, and
`ingestion/revalidate_feedback.py` (same standalone/cron pattern as
`run_alerts.py`) periodically re-runs `_verify_feedback` against the
*current* graph — for `pending` feedback (so a claim the graph couldn't
originally decide can still be recognized once more data is ingested) and
for `active` Concepts nearing expiry (extend if still confirmed, revoke if
contradicted, one grace period if newly undecidable). `search_concepts`
(`graph/tools.py`) only ever surfaces `status='active'` Concepts.

### Alerts (`graph/alerts.py`)

`AlertRule` nodes define threshold checks (`relval_failure_count`,
`relval_recurrence`, `latest_ib_stalled`) evaluated on demand via
`evaluate_rules()` (`POST /api/alerts/evaluate`), notifying through a
`Notifier` protocol (currently only `LogNotifier`).

### Log RAG (`log_rag/`)

`pipeline.py` normalizes JSON/JSONL/Clang-HTML payloads into canonical
`LogEvent`s (redacted, content-hashed for dedup), `archive.py` writes an
append-only evidence archive + checksum manifest under
`LOG_RAG_ARCHIVE_DIR` (default `data/log-rag/`), `Neo4jProjector` idempotently
MERGEs event/build/service/host context into the graph, and `retriever.py`
does hybrid retrieval (OpenSearch when `OPENSEARCH_URL` is set, filesystem
evidence scan as the inspectable fallback) surfaced as `citations` in chat
responses. `ingestion/ingest_json_logs.py` is the chunked importer for large
JSONL backfills (avoids sending an entire file through `/api/ingest` in one
request).

## Key invariants (violate these and answers become unsafe/wrong)

- **Never call GitHub or the web to fill in missing data.** The graph only
  ever contains what was explicitly ingested from supplied local files.
- **Textual references, changed-file matches, and comment mentions are
  relevance evidence only** — never causality, never proof a PR fixed/caused
  something. Every `_*_answer()` that surfaces this kind of evidence appends
  an explicit caveat sentence; keep doing that in new code.
- **IB `(tag, architecture)` pairs are never merged.** A "latest IB" or
  "compare IBs" answer must stay explicit about which architecture(s) it
  covers, per `_latest_ib_records`/`_requested_ib_arch`.
- **Release series outranks build recency** when picking "the latest IB":
  `CMSSW_20_1_X_<older-timestamp>` beats `CMSSW_20_0_X_<newer-timestamp>`.
  This ordering lives in `_ib_sort_key` (`graph/tools.py`) and
  `_latest_ib_records`/`_latest_and_previous_ib_tags` (`server/api_server.py`)
  — keep any new "latest" logic consistent with it (see
  `LatestIbRankingTests` in `tests/test_graph_tools.py`).
- **`query_graph` and any LLM-issued Cypher must stay read-only** — don't
  weaken `_enforce_read_only`'s keyword blocklist.
