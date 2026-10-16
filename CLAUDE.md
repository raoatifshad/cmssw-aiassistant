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

# the user's actual daily refresh routine (manual, no cron — see "Shifter
# documentation ingestion" below for why). Always run the cmsrepo half
# FIRST, since --cmsrepo here also re-ingests cmsrepo's own top-level docs.
cd cmsrepo && git fetch --prune origin master && git reset --hard origin/master \
  && git sparse-checkout reapply && cd ..
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/build_graph.py \
  --cmsrepo "$PWD/cmsrepo" --clang-root /home/rao/CHAT-BOT-AI/CLANG-LOGS --reconcile
cd shifter-wiki && git fetch --prune origin master && git reset --hard origin/master && cd ..
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/ingest_shifter_docs.py \
  --wiki-root "$PWD/shifter-wiki" --cmsrepo "$PWD/cmsrepo" --reconcile

# all of the above, unattended, every day at 07:30 (cron; logs + status.txt in
# /home/rao/CHAT-BOT-AI/daily-refresh-logs/). Exceptions go to a dated file per run: importing
# the same path again deletes what that path imported before. cmssdt9-02 is reached with
# your Kerberos ticket only (renewed each run, max 5 days); nothing is stored or installed there.
scripts/daily_refresh.sh [--install-cron|--uninstall-cron]   # STEPS="cmsrepo exceptions testlogs docs checks"

# IB test logs from cmssdt9-02 (unit/addOn/RelVal zips; the server deletes
# them after ~1 week, so run at least every few days). The user copies with
# their own rsync (full command in README "IB test logs"), then:
PYTHONPATH=ingestion venv/bin/python ingestion/fetch_ib_test_logs.py --no-server --delete-local
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/ingest_test_logs.py \
  --results /home/rao/CHAT-BOT-AI/IB-TEST-LOGS/results
# 90-day retention (dry run unless --apply; --chat also drops idle chat sessions +
# router shadow logs). Pruned sources get ImportState.pruned_at and are never
# re-imported, not even with --force. Keeps IB, ProblemFirstSeen, PR/Issue, Concept.
NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/prune_old_data.py [--days 90] [--apply]
# one-off: re-derive RelVal.status from the cmsrepo files (e.g. after a status-rule
# change -- DAS_ERROR was stored as PASSED until 2026-10-15); only differing rows written
NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/backfill_relval_status.py --cmsrepo cmsrepo [--apply]
# scan one zip by hand (read-only report, writes nothing)
PYTHONPATH=ingestion venv/bin/python ingestion/ib_test_log_scanner.py <zip> [--addon-log addOnTests.log] [--json out.json]

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

# live end-to-end validation against a *running* server + real graph data
# (catches things the mocked suite above can't -- see "Capability-handler
# coverage" below). Needs `uvicorn server.api_server:app --port 8001` running.
venv/bin/python scripts/validate_bot.py
venv/bin/python scripts/validate_bot.py --base-url http://localhost:8002  # non-default port

# live structural correctness check for GET /api/shift-summary specifically --
# parses the digest the same way the Shift Console frontend does and asserts
# it isn't internally self-contradictory (e.g. a workflow appearing as both
# newly-failing and resolved). Needs DEV_SHIFTER_LOGIN=true (or --username
# set to a real logged-in identity).
venv/bin/python scripts/validate_shift_summary.py

# live question bank: ~200 questions CMSSW people ask (pasted RelVal exceptions and
# test-log problems, unit tests, workflows, PRs/Issues, IB flavours, comparisons,
# build/Clang, exit codes/terms, follow-ups, typos, misuse), each with checks on its
# answer plus global ones (dead ends, literal None, raw <tags>, 500s, >15k chars, >30 s).
# Its expected facts are pinned to the 2026-10-16 data: after a refresh run
# --check-facts first -- it says which facts (and so which questions) went stale.
venv/bin/python scripts/question_bank.py --base-url http://localhost:8002 [--only PX,UT] [--json out.json]
venv/bin/python scripts/question_bank.py --check-facts

# live check of the imported IB test logs, run after each fetch + import:
# graph integrity, /api/test-log-problems contract over consecutive builds,
# chat finds pasted stored problems (FAIL = exit 1); freshness, noisy "new",
# fragmented signatures, junk categories (WARN); plus what a shifter sees.
venv/bin/python scripts/validate_test_log_data.py --base-url http://localhost:8002 --username testshifter
```

Tests mock the Neo4j driver / LLM client rather than requiring a live
database — see the `unittest.mock.patch` usage at the top of
`tests/test_chat_routing.py` and `tests/test_graph_tools.py` for the pattern.
`scripts/validate_bot.py` is the complementary live layer: it drives real
multi-turn conversations against a running server and real Neo4j data,
templated against whatever tag the server itself reports as "latest" so it
stays valid as the graph is refreshed. `scripts/validate_shift_summary.py` is
the same idea scoped to `GET /api/shift-summary`: it re-implements the
frontend's own markdown parser (`shiftMarkdown.js`'s `##`/`###`/`####`/`|`-table
walk) in Python and checks structural invariants a mocked unit test can't
catch on synthetic data alone — e.g. the "Variant" column being present and
non-empty on every RelVal row (added after a real workflow with multiple
sub-IB variants rendered as unlabeled, seemingly-duplicate/contradictory
rows), and that no identity is listed as both newly-failing and resolved for
the same architecture in the same digest. Coordinate with whoever owns the
Shift Console frontend (a separate repo, `cmssdt-18V` — see
`reference-shift-console-frontend` in project memory) before changing the
digest's column shape, since their parser and this script both hard-code the
same contract.

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
- **IB test logs** — a separate three-step pipeline, not part of
  `build_graph.py`: `ib_test_log_scanner.py` (pure parser, streams zip
  members line by line with a 64 KB per-line cap -- a CLANG unit log has a
  single 2 GB line, and the DBG unit zip is 13 GB unpacked),
  `fetch_ib_test_logs.py` (copy → scan → `results/*.json.gz` + ledger →
  delete the copy; `results/` is the durable copy, never delete it) and
  `ingest_test_logs.py` (results → `TestLogRun`/`LogProblem`/
  `ProblemSignature`/`ProblemFirstSeen`, ImportState prefix `testlogs:`, so
  `build_graph.py --reconcile` never touches it). Its purpose: problems
  inside tests the dashboard shows as PASSED. Invariants: it never creates
  `IB` nodes (MATCH only, relinked later) so "latest IB" can't shift; a test
  is never relabeled failed; nothing is marked "expected" automatically (the
  user confirms each); a detected problem is never silently dropped.
  Chat side: `_pasted_exception_response` (`server/api_server.py`, runs
  before the planner) prepends `_test_log_problem_answer` (backed by
  `search_test_log_problems` in `graph/tools.py`, fulltext index
  `logProblemText`) to the GitHub evidence, and also handles sanitizer /
  traceback / assertion pastes that have no exception category. The paste
  is read by the scanner itself (`parse_pasted_error`, built on the
  scanner's `problem_details`, the same function that produced the stored
  `names`/`modules`/`context_keys`) and searched through parallel channels
  (`search_test_log_problems_report`): signature lookup (exact), text
  (contains the paste), the `logProblemNames` index (specific names matched
  whole; producer keys = same producer, any message) and same-category in
  the pasted IB. Results are merged per problem and labelled exact /
  contains / related / module / weak / same_ib -- the exception category
  alone never counts, weak matches only show when nothing stronger does.
  The answer ends with "Searched the test logs by: ..." (every key and its
  count; a channel that failed is said, not hidden). "Not in the test logs"
  is only said when the logs could actually be searched. GitHub is searched by
  those same names together (`_pasted_names_github_answer`, ranked by how
  many each item mentions), before the category-only list. Tests:
  `tests/test_test_log_answers.py`; live: the three "pasted ..." scenarios in
  `scripts/validate_bot.py`. When
  changing detection or normalization, bump `SCANNER_VERSION`, and measure
  "new" generic lines over a week of data -- run-specific text (timestamps,
  hosts, paths) leaking into a signature makes every build look "new".
  The zips are deleted after scanning, so old results can't be re-scanned:
  the importer re-signs every stored problem with `current_signature`
  (scanner) instead, and a normalization change only takes effect after
  `ingest_test_logs.py --force`. Found live (v5): Madgraph `threadN/` worker
  dirs, a `%MSG` glued onto UBSan lines, and fatal messages cut at 300 *raw*
  chars (release path length moves the cut) each split one problem into
  several signatures, showing it as new + gone in the same window.
  `ProblemFirstSeen` is keyed per release series (`suite|CMSSW_20_1|owner|sid`).
  Shift Console side: `GET /api/test-log-problems?from=&to=[&arch=]`
  (`get_test_log_changes`, `graph/tools.py`) returns JSON, deliberately not a
  `/api/shift-summary` section (that markdown contract is shared with the
  frontend parser). Serious problems in PASSED tests/steps only, per
  (sub-IB variant, arch, suite, run variant) track, each compared with the
  same track at `from` or its own previous build (`baseline_note`), split
  into `new` (first time ever for that test, via `ProblemFirstSeen`),
  `returned` (seen before, absent in the baseline: intermittent -- the
  Madgraph `IndexError` RelVals flap like this), `gone` (test ran again
  without it) and a `still_present` count. `coverage.to_imported=false`
  means no logs yet, never "zero problems". Tests:
  `tests/test_test_log_changes.py`.
- `--github-issues` — optional raw CMSSW GitHub Issues export
  (dir or `.tar`/`.tar.gz`), imports PR/Issue bodies, labels, comments, and
  explicit `#number`/URL cross-references. Never calls GitHub live.

**IB identity is always the pair `(tag, architecture)`** — never merge or
dedupe across architectures. `CMSSW_<major>_<minor>_X_<timestamp>` is a
*primary* IB; a tag with an inserted variant (e.g.
`CMSSW_20_1_NONLTO_X_<timestamp>`) is a *sub-IB* whose `parent_tag` points at
the primary tag, tracked via `ib_parent_tag()`.

### Shifter documentation ingestion (`ingestion/ingest_shifter_docs.py`)

A second, static evidence source added 2026-10-02 so the chat LLM can cite
CMSSW/CMSSDT procedural knowledge (Jenkins job runbooks, release-cycle
steps, CI topology) alongside live IB facts, exposed via the
`search_shifter_docs` tool (`graph/tools.py`) over a `(:DocChunk)` node type
and a Neo4j fulltext index (`shifterDocText`). Sources: a local clone of
`cms-sw/cmssdt-wiki` at `shifter-wiki/`, plus `cmsrepo`'s own unused
top-level `*.md` pages (`build_graph.py` only ever reads `cmsrepo/data/...`).
Same trust model as everything else here — a local git clone, never fetched
live during a chat request — and same incremental hashing
(`source_needs_import`/`mark_source_imported`, reused directly from
`build_graph.py`) so re-running is cheap.

**There is no cron job for this, or for `build_graph.py` itself** — both are
refreshed by hand, daily, as one combined routine (see the Commands section
above). `shifter-wiki`'s own upstream commit history
(`github.com/cms-sw/cmssdt-wiki/commits`) is never polled or watched; a new
commit there only reaches the graph once someone runs `git fetch`/`reset`
on `shifter-wiki/` and re-runs `ingest_shifter_docs.py` — this is a direct
consequence of the "never call GitHub live" invariant below, not an
oversight. If this ever needs to run on its own instead of by reminder, it's
one crontab entry mirroring `ingestion/summarize_router_shadow.py`'s
existing cron pattern — ask before adding it, since it's a durable
system-level change.

A real, found-live failure mode worth knowing when touching this: the
Jenkins-job wiki pages are dozens of near-identical entries back to back
(same Description/Upstream/Downstream/Sub-projects template), and an LLM
synthesizing over several retrieved chunks at once can misattribute one
job's real upstream/downstream names to a different job asked about —
correct retrieval, wrong attribution, not caught by `_evidence_grounded`'s
number-only check. `_shifter_doc_claims_grounded` (`server/api_server.py`)
guards specifically against this (cross-checks any other named Jenkins job
mentioned in the answer against the asked-about job's own retrieved chunk
text) and degrades to raw evidence, same as `_evidence_grounded`, rather
than showing the misattributed prose. A related, upstream bug found while
building this: `_normalize_deterministic_question`'s `relvals`→`relval`
regex used to also fire inside unrelated hyphenated identifiers like the
Jenkins job name `ib-run-gpu-relvals` (`\b` treats a hyphen as a word
boundary) — fixed with a lookbehind/lookahead that excludes hyphen-adjacency
instead.

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

### Capability-handler coverage — a recurring bug class, and how to test for it

`RequestPlanner` correctly recognizing a capability (e.g. `"relval"`,
`"clang"`, `"build"`) does **not** guarantee `_execute_planned_capability`
(`server/api_server.py`) can actually answer it — each capability branch
there is its own short-circuiting `or` chain of `_*_answer()` helpers, and a
real phrasing can fall through all of them even though the router picked the
right capability. When that happens the user sees a flat "No imported local
evidence is available for the selected `X` request" — which reads as "no
data exists" even when the data is there and the router worked correctly;
only that specific helper chain was incomplete. Two concrete ways this has
happened in practice (both found and fixed in the same session, 2026-08-25):

1. **No session-tag fallback.** A capability's helpers only look for an IB
   tag in the *current* question text, so a natural follow-up that doesn't
   repeat the tag ("what about unit tests", "show me the clang warnings
   again") dead-ends even though the tag was already established earlier in
   the same session (`_latest_ib_contexts[sid]`). `_unittest_summary_answer`
   and `_ib_results_summary_answer` already took a `context_tag` fallback
   parameter; `_relval_*`, `_build_warning_answer`, and the new
   `_clang_warning_answer` were retrofitted to do the same — thread
   `_latest_ib_contexts.get(session_id)` through as the tag when
   `IB_TAG_RE.search(question)` finds nothing.
2. **A whole phrasing shape has no handler at all**, not just a missing
   fallback. `"clang warnings for <tag>"` — the same "`X` for `<tag>`" shape
   that already worked for build/relval/unit-test — had zero matching
   function in the `"clang"` branch (only "latest" and two-tag-comparison
   handlers existed) and failed on the very first turn, tag included. This
   one was only found by testing against a live server with real data, not
   by mocked unit tests, since nobody had written a test asserting the gap
   existed to catch it.
3. **A later handler in the chain never runs at all once an earlier one
   matches**, even when the question asked for both. `"pr_evidence"`'s chain
   was `_pr_number_answer or _pr_evidence_answer or _recurring_failure_answer`
   — `_open_issue_answer` wasn't in it. A real question ("any information
   about ticl::AssociationMap any PR/issue open?") explicitly asked for open
   Issues too, but once `_pr_evidence_answer` matched on the PR side, Issues
   were never even searched — the answer looked complete (a real PR list)
   while silently omitting a whole evidence category the graph actually had
   (confirmed live: Issues #42737, #37821, #34000 all mention the term and
   are open). Fixed by running both `_pr_evidence_answer` and
   `_open_issue_answer` and concatenating when both match, instead of a bare
   `or` chain — see `_execute_planned_capability`'s `"pr_evidence"` branch.
   The lesson generalizes beyond this one branch: an `or` chain is fine when
   the helpers are mutually exclusive interpretations of the same question,
   but wrong when a question can legitimately want evidence from more than
   one of them at once.

**Two complementary test layers exist specifically to catch this class of
bug going forward — extend both when adding or changing a capability:**

- **Mocked, fast, CI-safe** (`tests/test_chat_routing.py`): call
  `_execute_planned_capability(capability, question, session_id)` directly.
  For any capability whose helpers accept an IB tag, add a test that sets
  `_latest_ib_contexts[session_id]` and asks a tag-less follow-up, asserting
  a real (non-`None`) answer — see
  `test_relval_capability_lists_failing_workflows_without_repeating_tag`,
  `test_unittest_capability_uses_session_ib_context_when_tag_omitted`,
  `test_clang_capability_uses_session_ib_context_when_tag_omitted`, and
  `test_build_capability_uses_session_ib_context_when_tag_omitted` as the
  pattern to copy. For the "combine, don't short-circuit" gap, see
  `test_pr_evidence_capability_combines_pr_and_open_issue_evidence`.
- **Live, against a running server + real graph** (`scripts/validate_bot.py`):
  add a `Scenario` to `build_scenarios()` for the new phrasing/capability,
  including a multi-turn variant that establishes a tag in one turn and asks
  a tag-less follow-up in the next — this is what actually caught finding
  #2 above. Run it after any change to `_execute_planned_capability` or a
  capability's helper chain, not just the mocked suite; the mocked suite
  proves "the code I wrote does what I mocked it to do," the live script
  proves "a real phrasing against real data actually gets an answer."

If a capability legitimately has no handler for some phrasing yet, prefer
adding the handler (mirroring `_clang_warning_answer` as the template: same
`"X for <tag>"` shape as its siblings, `context_tag` fallback parameter,
default-architecture + `all`/`everything` scope handling) over leaving the
flat "No imported local evidence" message as the answer.

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
`relval_recurrence`, `latest_ib_stalled`, `unittest_failure_count`,
`build_failure_count`) evaluated on demand via `evaluate_rules()`
(`POST /api/alerts/evaluate`), notifying through a `Notifier` protocol
(currently only `LogNotifier`). There is no `addon_*` rule type yet: the
dashboard data has no per-test addOn results, and the addOn results now
imported from the raw logs (`TestLogRun {suite:'addon'}`) aren't wired into
any rule.

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
