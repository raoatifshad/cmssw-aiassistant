# CMSSDT Knowledge Graph Chatbot

A standalone, Neo4j-backed chatbot for CMSSW Integration Build data.
Completely independent from the SQL-based chatbot — shares no code, no
database, no imports. Same API shape (`POST /api/chat`, `GET /api/health`)
so the existing React chat widget works by changing one env var.

## Meeting Documentation

The plain-language implementation walkthrough, API/data-flow explanation, and
architecture diagrams are in [docs/meeting/cmssdt-chatbot-meeting-pack.md](docs/meeting/cmssdt-chatbot-meeting-pack.md).
The current Neo4j labels, relationships, identities, and local refresh flow are
documented in [docs/current-neo4j-database-structure.md](docs/current-neo4j-database-structure.md).

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in CERN_API_BASE, CERN_API_KEY
```

## Neo4j (Docker)

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/cmssdt2026 -v $HOME/neo4j-data:/data \
  docker.io/library/neo4j:5
```

## Build the graph

```bash
# get the raw data
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/cms-sw/cms-sw.github.io.git cmsrepo
cd cmsrepo && git sparse-checkout set data && cd ..

# one-time bootstrap after upgrading to architecture-safe graph identities
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/build_graph.py \
  --cmsrepo cmsrepo \
  --clang-root /home/rao/CHAT-BOT-AI/CLANG-LOGS \
  --clear
```

Every following run is incremental: it scans source paths but hashes each JSON
file and each `log.html`, skipping unchanged inputs. The Clang corpus is kept
outside `cmsrepo` and must have this layout:

```
/home/rao/CHAT-BOT-AI/CLANG-LOGS/<ib-tag>/<arch>/build-logs/**/log.html
```

### Shifter documentation (optional, additive)

A second, static evidence source the chat LLM can cite alongside live graph
facts: CMSSW/CMSSDT shifter documentation (Jenkins job runbooks,
release-cycle procedures, CI topology). Cloned locally once, like `cmsrepo`
— never fetched live during a chat request.

```bash
git clone --depth 1 https://github.com/cms-sw/cmssdt-wiki.git shifter-wiki

PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/ingest_shifter_docs.py \
  --wiki-root shifter-wiki \
  --cmsrepo cmsrepo \
  --reconcile
```

`--cmsrepo` here additionally ingests `cmsrepo`'s own unused top-level `*.md`
pages (`troubleshooting.md`, `faq.md`, etc.) — `build_graph.py` only ever
reads `cmsrepo/data/...`, so these sit on disk unused otherwise. Re-running
is incremental (same `ImportState` content-hashing as every other source);
pass `--force` to reimport unchanged files. Skipping this step entirely is
safe: `search_shifter_docs` degrades to "no documentation evidence" if the
`DocChunk` fulltext index has nothing in it yet.

### RelVal exception logs

The CMSSDT logreader (`cmssdt.cern.ch/SDT/cgi-bin/logreader/...`) requires a
CERN SSO login, so exceptions are sourced from Elasticsearch instead: run
cms-bot's `es-cmssw-exceptions.py` and redirect its output to a file, kept at
the fixed path `/home/rao/CHAT-BOT-AI/exceptions-json-file/exceptions.json` —

```bash
es-cmssw-exceptions.py > /home/rao/CHAT-BOT-AI/exceptions-json-file/exceptions.json
```

— then pass that file with `--es-exceptions-json`. Ingestion parses each
document's `exception` field (category, message, context — a single document
can concatenate several occurrences of the same category back to back) and
links them to the matching `RelVal` via `HAS_EXCEPTION`. There's no exact
source-line citation this way (an ES document is a structured field, not a
byte offset into a real log file), but coverage is far broader than manually
staging individual logs, since one export can span many releases and
architectures at once. The whole file is tracked as a single hashed source,
so overwriting it with a fresh export and re-running ingestion picks up the
update automatically — no code changes needed:

```bash
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/build_graph.py \
  --cmsrepo cmsrepo \
  --es-exceptions-json /home/rao/CHAT-BOT-AI/exceptions-json-file/exceptions.json \
  --reconcile
```

An exception only attaches if its `release`/`architecture`/`workflow` already
matches an ingested `RelVal` record — one without the other is silently
skipped, not stored.

Run the normal update after refreshing `cmsrepo` or adding Clang logs:

```bash
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/build_graph.py \
  --cmsrepo cmsrepo \
  --clang-root /home/rao/CHAT-BOT-AI/CLANG-LOGS \
  --reconcile
```

`--reconcile` removes graph records whose previously imported source file has
been deleted. Use `--force` only when intentionally reprocessing all sources.
IBs are identified by both their timestamped tag and architecture, so results
for `el9_amd64_gcc13` and `el9_amd64_gcc14` remain distinct.

### IB test logs (unit tests, addOn tests, RelVals) from cmssdt9-02

The dashboard only says whether a test passed. A test can pass even though
its log contains a fatal exception, a crash, a sanitizer report or an
infrastructure error, and nobody sees it. This pipeline reads the raw logs
so those problems are stored and can be shown to the shifter.

**Source:** `cmssdt9-02:/data/sdt/buildlogs/<arch>/www/<day>/<queue>/<IB tag>/`
(plus `gpu/<type>/` and `other/<name>/` sub-folders) holds
`unitTestLogs.zip`, `addOnTests.zip` (+ `addOnTests.log`) and
`pyRelValMatrixLogs.zip` for every IB. **The server deletes them after about
one week**, so the routine below has to run at least every few days.

**Flow:** copy → scan → keep the extracted facts → delete the copy. The raw
logs are never kept; `results/` (a few MB per day) is.

1. Copy new zips into `/home/rao/CHAT-BOT-AI/IB-TEST-LOGS/server-copy/`
   with rsync. `results/rsync-exclude.txt` (rewritten by every scan run)
   keeps already-finished zips from being downloaded again:

   ```bash
   mkdir -p /home/rao/CHAT-BOT-AI/IB-TEST-LOGS/results && touch /home/rao/CHAT-BOT-AI/IB-TEST-LOGS/results/rsync-exclude.txt   # first time only
   rsync -a --prune-empty-dirs --info=progress2 \
     --exclude-from=/home/rao/CHAT-BOT-AI/IB-TEST-LOGS/results/rsync-exclude.txt \
     --exclude='/filtered/' \
     --include='/*/' --include='/*/www/' --include='/*/www/*/' --include='/*/www/*/*/' \
     --include='/*/www/*/*/CMSSW_*/' \
     --include='/*/www/*/*/CMSSW_*/gpu/' --include='/*/www/*/*/CMSSW_*/gpu/*/' \
     --include='/*/www/*/*/CMSSW_*/other/' --include='/*/www/*/*/CMSSW_*/other/*/' \
     --include='unitTestLogs.zip' --include='unitTestLogs.zip.checked' \
     --include='addOnTests.zip' --include='addOnTests.zip.checked' --include='addOnTests.log' \
     --include='pyRelValMatrixLogs.zip' \
     --exclude='*' \
     root@cmssdt9-02.cern.ch:/data/sdt/buildlogs/ /home/rao/CHAT-BOT-AI/IB-TEST-LOGS/server-copy/
   ```

   A full week is ~250 GB, ~100 GB of it the three DBG RelVal zips (~33 GB
   each). When disk is short, add `--max-size=5G` first (skips only the DBG
   RelVal zips), run step 2, then rsync again without it.

2. Scan and delete the copies (`--dry-run` first to see what it would do):

   ```bash
   cd cmsrepo && git fetch --prune origin master && git reset --hard origin/master \
     && git sparse-checkout reapply && cd ..
   nohup env PYTHONPATH=ingestion venv/bin/python ingestion/fetch_ib_test_logs.py \
     --no-server --delete-local > /home/rao/CHAT-BOT-AI/IB-TEST-LOGS/fetch.log 2>&1 &
   ```

   It can run while rsync is still copying (it only takes zips rsync has
   finished); run it once more after rsync ends. Each zip's facts go to
   `results/<suite>/<arch>/<tag>/<variant>/*.json.gz` and
   `results/ledger.jsonl`; only then is the zip deleted. A zip that fails to
   scan keeps its local copy and is retried next run. The end of `fetch.log`
   lists every zip the dashboard data (cmsrepo) expects that is neither
   scanned nor in `server-copy/` (`MISSING`, also in `results/missing.txt`):
   copy those before the server deletes them.

   Without `--no-server` the script lists and copies from the server itself
   over ssh (open one ssh `ControlMaster` connection first, see the script's
   docstring); `--delete-local` is then only needed for zips you copied by hand.

3. Import into Neo4j (incremental; unchanged results files are skipped):

   ```bash
   PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python \
     ingestion/ingest_test_logs.py --results /home/rao/CHAT-BOT-AI/IB-TEST-LOGS/results
   ```

**What is extracted** (`ingestion/ib_test_log_scanner.py`, logs are streamed,
never loaded whole — one CLANG log has a single 2 GB line):

- every unit test, addOn test step and RelVal workflow, with its status from
  the log itself — passing ones included;
- fatal exceptions, sanitizer/UBSan reports, crashes, assertions, uncaught
  exceptions, Python tracebacks and known infrastructure errors, with up to
  60 evidence lines, the test they belong to and whether it passed;
- MessageLogger warnings/errors grouped by category, and plain lines that
  say error/failed, both grouped (millions of repeated lines become a few
  hundred records); `%MSG-i`/`%MSG-d` debug output is dropped.

Each problem gets a normalized signature (dates, paths, IB names, numbers,
host names, GUIDs and hex values stripped), so the same problem is
recognised across IBs and architectures. That gives "first seen"/"last seen",
and lets a pasted exception (whole or a fragment) be matched later through the
`logProblemText` fulltext index.

**Not stored:** the raw logs, `%MSG-i`/`%MSG-d` messages, minified JavaScript
(a test prints a downloaded web page), stack frames beyond 20, evidence
beyond 60 lines per problem, and plain error lines beyond 1,000 distinct ones
per test (those are still counted). The server path of each zip is kept,
even though it stops resolving after about a week.

**In the chat:** a pasted error -- a CMSSW exception, sanitizer report,
traceback, assertion or crash, whole or a fragment -- is looked up in these
logs first (`search_test_log_problems`, also available to the LLM path). The
paste is read with the same parser as the logs, so the same problem from
another run is an exact match; otherwise the answer says the exact error is
not in the logs and shows, each under its own heading, problems that contain
the pasted text, problems sharing its specific names (plugin, class, module,
library -- never just the exception category), and same-category problems in
an IB the paste names. All of these run at once: an exact signature lookup,
the pasted text, every specific name (matched whole in the `logProblemNames`
index, which holds the modules, producers and names of *every* occurrence --
not just the stored example) and the producer ("same producer, different
message"). Weak matches (only ordinary identifiers in common) are shown only
when nothing stronger is found, and the answer ends with every key it
searched by and how many problems each found. When the paste shows it was raised by a build-time
tool such as `edmWriteConfigs`, the answer says it belongs in build logs, not
test logs. The answer lists the tests/workflows that hit it and their status ("reported as
PASSED in all N runs"), how many IBs and architectures, first/last IB seen,
whether newer builds no longer show it, any IB named in the question, and one
evidence excerpt with its log location. PRs/Issues the text names by number are
looked up, PRs/Issues mentioning the paste's specific names are listed (most
names first), and the category-only GitHub matches follow, shortened and
labelled as broader context.

The graph never gets new `IB` nodes from this import: results attach to IBs
already imported from the dashboard data, and ones imported earlier are linked
on a later run once their IB exists.

### Primary and sub-IB runs

`CMSSW_<major>_<minor>_X_<timestamp>` is a primary Integration Build run.
Tags with an inserted variant, such as `CMSSW_20_1_NONLTO_X_2026-08-10-2300`,
are sub-IBs of `CMSSW_20_1_X_2026-08-10-2300`. Ingestion stores each sub-IB's
`parent_tag` and `variant`; latest-IB answers select primary runs and list any
recorded sub-IB variants beneath them.

### GitHub exports

The normal graph build imports PR metadata from `cmsrepo/data/prs`, the supplied
PR-to-commit cache at `cmsrepo/cmssw-prs-cache.json`, and workflow command hashes
from `cmsrepo/data/commands`. A command hash is linked to a PR only when it exactly
matches the prefix of a commit hash in the supplied cache. The importer never calls
GitHub or infers a PR from a textual similarity.

To keep normal refreshes bounded, the importer projects the latest command snapshot
for each architecture/release/flavor. Use `--command-history` only when a historical
command-provenance investigation needs every snapshot.

If a separately supplied raw CMSSW GitHub Issues export is available, pass its
directory or `.tar`/`.tar.gz` archive with `--github-issues` to import PR/Issue
bodies, labels, and comments. The importer hashes each raw `issue.json` and
`comments.json`, so normal refreshes only update changed records:

```bash
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/build_graph.py \
  --cmsrepo cmsrepo \
  --github-issues /path/to/cmssw-github-issues.tar.gz \
  --clang-root /home/rao/CHAT-BOT-AI/CLANG-LOGS \
  --reconcile
```

The graph retains explicit `#number` and CMSSW GitHub URL references found in
PR/Issue bodies and comments. These links are textual facts, not evidence that
one PR caused, fixed, or depends on another item.

Unlike `cmsrepo` and the exceptions feed, **this source has no automatic
refresh** and will silently go stale. The live export lives on
`cmssdt9-02.cern.ch:/data/sdt/SDT/public/github/cms-sw/cmssw/issues`; see
"Raw CMSSW GitHub export" in `docs/cmssdt-pipeline-guide.md` for the full
sync procedure (freshness check via `issues/latest.txt`, why the sync must be
pushed *from* `cmssdt9-02` rather than pulled, and the exact `rsync`/ingest
commands).

## Run the API server

```bash
uvicorn server.api_server:app --reload --port 8001
curl http://localhost:8001/api/health
```

## Conversation history

`POST /api/chat` keeps the existing request and response shape. When the
client sends the same `session_id`, the API stores user and assistant turns as
separate `ChatSession` and `ChatTurn` metadata in Neo4j. Imported CMSSW,
Clang, and GitHub facts are never changed. On an LLM fallback, the API restores
the most recent 12 turns so deterministic graph answers can inform a natural
follow-up question after a server restart.

Retrieve a session or export it as readable Markdown:

```bash
curl http://localhost:8001/api/chat/<session-id>/history
curl http://localhost:8001/api/chat/<session-id>/export.md
```

`DELETE /api/chat/<session-id>` clears both the old in-memory context and the
persisted conversation metadata for that session.

## Answer Feedback

Submit a correct/incorrect verdict to `POST /api/feedback`. For latest-IB
questions, the API verifies the submitted verdict against the latest imported
IB tag. Other feedback is checked by a dedicated LLM verifier that may use only
the read-only graph tools and must return an evidence-based decision. The
response states `verification_method` as `deterministic`, `llm_graph_tools`, or
`unverified`; inconclusive evidence remains `pending`. Feedback never
overwrites imported IB or warning facts.

```json
{
  "question": "What is the latest IB data?",
  "answer": "CMSSW_9_4_X_2026-08-09-0000",
  "verdict": "incorrect",
  "corrected_answer": "CMSSW_20_1_X_2026-08-10-1100 is the latest overall IB.",
  "ib_tag": "CMSSW_20_1_X_2026-08-10-1100"
}
```

An `accepted` correction (`verdict: incorrect` with a `corrected_answer`)
becomes a durable `Concept` fact, retrievable in later chats — but never
because a user asserted it, only because the same evidence gate above
confirmed it against the graph. That trust isn't permanent: run
`ingestion/revalidate_feedback.py` on a schedule (same standalone,
cron-invoked pattern as `ingestion/run_alerts.py`) to re-check `pending`
feedback against the graph as it changes, and to re-verify active Concepts
before they expire — a Concept that no longer holds is revoked
automatically, and one that comes back undecidable gets a single grace
period before revocation. No human review step is involved at any point;
the graph is always the one deciding.

```bash
# crontab, once a day
0 3 * * * cd /path/to/cmssdt-kg-chatbot && \
    NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/revalidate_feedback.py
```

`CONCEPT_TTL_DAYS` (default 14) and `CONCEPT_PENDING_GRACE_DAYS` (default 3)
control the expiry and grace windows; `--limit` bounds how many rows a single
run re-checks, since each re-check may cost an LLM call.

## Router shadow mode

The deterministic router is regex/keyword-based, so it only ever recognizes
the phrasings someone has already tested — a phrasing it doesn't recognize
either gets no answer or falls to the constrained LLM investigation path.
Rather than hand-patching a new regex for every new phrasing forever
(unbounded, no stopping point), `ENABLE_ROUTER_SHADOW=true` logs an
independent LLM classification of every question next to the deterministic
router's choice (`RouterShadowLog` nodes) — purely for comparison, **never**
used to answer. This is the evidence to look at before ever considering
promoting an LLM router to *select* a capability more robustly than regex
matching — the answer itself must always stay Cypher-only, that doesn't
change either way.

Summarize accumulated agreement/disagreement:

```bash
NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/summarize_router_shadow.py --days 7
```

`ENABLE_ANSWER_SYNTHESIS=true` is a separate, independent flag: it lets an
LLM rephrase an already-deterministic answer into more natural prose,
verified to preserve every number in it before being shown, falling back to
the original deterministic text otherwise. Both flags are off by default in
`.env.example` since each costs a real per-message LLM call.

### Capability dead-end fallback

The router correctly picking a capability doesn't guarantee that
capability's own regex-driven `_*_answer()` helpers recognize the exact
phrasing — when they don't, `chat()` used to give up immediately with a flat
"No imported local evidence is available for the selected `X` request"
message, even though the same evidence-gated LLM investigation path
described above was one step away and already reachable from every other
routing gap in the same function. `ENABLE_CAPABILITY_FALLBACK_TO_LLM=true`
closes that one remaining dead end: instead of the flat message, it falls
through to that same investigation path (still Cypher-tool-only, still
verified against evidence before being shown). Off by default for a safe
rollout — flip it, re-ask a phrasing that used to dead-end, and compare
against the flat-message baseline before leaving it on. Unlike
`ENABLE_ROUTER_SHADOW`/`ENABLE_ANSWER_SYNTHESIS`, this only adds an LLM call
on the minority of messages that would otherwise dead-end, not on every
message, so it's a safer default-on candidate once validated.

### Deterministic answer verification

Deterministic routing only ever tests keyword/substring presence, never
phrase-level meaning — so "workflows that are NOT failing" and "workflows
that are failing" both trigger the same "failing" keyword in the relval
handler and can return the identical, wrong-for-one-of-them answer. Found
live 2026-09-22: both phrasings returned "No failing RelVals were found",
even though the negated question actually wanted the passing workflows.

`ENABLE_DETERMINISTIC_ANSWER_VERIFICATION=true` closes this gap: every
deterministic capability answer is checked against the full question with a
small, constrained LLM call (`verify_answer_matches_question`,
`server/agent.py`) — the same non-agentic pattern as
`propose_capability_classification`, judging yes/no only, never writing the
answer itself. A mismatch reroutes through the same evidence-gated LLM
investigation path used elsewhere (`_run_llm_investigation`) instead of
showing the wrong answer. The verifier fails open at the call site — any
error (network, timeout, malformed response) defaults to trusting the
deterministic answer, so a verifier-layer failure can never be worse than
not having verification at all.

Unlike `ENABLE_CAPABILITY_FALLBACK_TO_LLM`, this one *does* cost an LLM call
on every deterministic answer, not just misses. Off by default — not for
LLM-call cost, but because the verifier itself isn't infallible and needs
validating against real traffic before it's trusted to override a
deterministic answer by default.

## Log RAG platform

The API also provides a durable Log RAG pipeline. It normalizes JSON/JSONL and
Clang `log.html` inputs into deterministic event IDs, redacts common credential
fields, archives the original payload plus a checksum manifest, projects event
context to Neo4j, and returns cited evidence with chat responses.

For the single-host platform, start the persistent services with Compose:

```bash
# Set strong NEO4J_PASSWORD and CERN credentials in .env first.
./start.sh
```

Compose starts Neo4j on `7688`, OpenSearch on `9200`, Logstash on `8080`, and
the API on `8002` (configure `LOG_RAG_API_PORT`). OpenSearch is the raw-event retrieval path at scale; the
filesystem evidence scan is retained only as an inspectable fallback when
`OPENSEARCH_URL` is unset or unavailable. Before starting OpenSearch on Linux,
set `vm.max_map_count=262144` on the host.

`start.sh` detects Podman or Docker, builds the API image, and starts only this
project's Compose services. To stop those services while preserving their data
volumes, run:

```bash
./stop.sh
```

## PR-to-warning evidence links

The graph can enrich imported CMSSW PRs with their GitHub changed-file lists and
link them to warnings in those files. This is file-change evidence, not proof
that a PR introduced a warning. For the current corpus, run:

```bash
NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/link_pr_warnings.py \
  --release CMSSW_20_1 \
  --ib-tag CMSSW_20_1_X_2026-08-03-2300
```

Set `GITHUB_TOKEN` to use GitHub's paginated files API. When that API is rate
limited, the importer falls back to the public PR diff endpoint.

```bash
# JSON, JSONL, or text/json payload
curl -X POST 'http://localhost:8001/api/ingest?source=build-42.jsonl&build=CMSSW_20_1_X' \
  -H 'Content-Type: application/json' --data-binary @build-42.jsonl

# Inspect grounded evidence without invoking the LLM
curl -X POST http://localhost:8001/api/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which service emitted timeout errors?"}'
```

For large JSONL files, use the chunked importer rather than sending the entire
file to the API in one request. It preserves the original source path in every
event and can optionally project to Neo4j and bulk-index OpenSearch.

```bash
venv/bin/python ingestion/ingest_json_logs.py /logs/build-42.jsonl \
  --build CMSSW_20_1_X \
  --opensearch-url http://localhost:9200 \
  --project-neo4j
```

Archive files default to `data/log-rag/` (configure `LOG_RAG_ARCHIVE_DIR`).
This directory contains content-addressed raw source files, checksum manifests,
normalized event JSONL, and retrievable evidence JSONL; preserve it for replay
and audit. Chat responses now include a `citations` field containing the source
event IDs and excerpts used as retrieved context.

## Switch the widget

In your cmssdt-ib checkout's `.env.development`:
```
VITE_CHAT_API_BASE=http://localhost:8001
```
Restart the Vite dev server — same chat widget, now backed by Neo4j.

## Project layout

```
ingestion/
  clang_parser.py    — standalone clang warning extractor (same proven regex)
  build_graph.py     — reads raw sources, builds the Neo4j graph
  ingest_json_logs.py — chunked JSONL importer for large backfills
  ib_test_log_scanner.py — streams unit/addOn/RelVal log zips, extracts tests + problems
  fetch_ib_test_logs.py  — copy → scan → results/ → delete, for the cmssdt9-02 test logs
  ingest_test_logs.py    — imports results/ into Neo4j (TestLogRun, LogProblem, ...)
graph/
  neo4j_conn.py      — connection helper, schema constraints
  tools.py           — Cypher query tools the LLM calls
server/
  llm_client.py      — CERN LiteLLM client
  agent.py           — agent loop (tool-calling + <tool_call> fallback)
  api_server.py      — FastAPI (same API shape as the SQL chatbot)
log_rag/
  models.py          — canonical event IDs, timestamps, and redaction
  archive.py         — append-only evidence archive and manifests
  pipeline.py        — JSON/JSONL/Clang adapters and Neo4j projection
  retriever.py       — hybrid local evidence + graph-context retrieval
  opensearch.py      — optional raw-event bulk indexing and search adapter
logstash/
  pipeline/logs.conf — HTTP JSON ingestion to the raw OpenSearch index
```
