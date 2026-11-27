# CMSSDT Knowledge Graph Chatbot

The backend of the CMSSW IB chatbot. It imports CMSSW Integration Build (IB) data
into Neo4j -- RelVals, unit tests, build status, Clang warnings, RelVal exceptions,
problems found in the raw IB test logs, CMSSW GitHub PRs/Issues and the shifter
documentation -- and answers questions about it at `POST /api/chat`.

Answers come from the graph only: deterministic lookups first, and an LLM only to
pick lookups, investigate with read-only graph tools, or check an answer. It never
answers from its own knowledge and never calls GitHub or the web during a chat.

The chat widget and the Shift Console pages are a separate repository
([cms-sw/cmssdt-ib](https://github.com/cms-sw/cmssdt-ib)); they talk to this
backend over HTTP only (`VITE_CHAT_API_BASE`).

## Where it runs

| Where | What | How it is set up |
|---|---|---|
| `cmssdt-chatbot.cern.ch` (production VM, port 8001) | Neo4j + API + worker as containers | [deploy/vm/README.md](deploy/vm/README.md) |
| `cmschatbot-01.cern.ch` (new VM, being set up) | the same stack, everything set up by Puppet | hostgroup `vocmssdt/sdt/cms_chatbot` in `ai/it-puppet-hostgroup-vocmssdt` |
| cmssdt (Jenkins) | scans the IB test logs and sends the results to the VM | [deploy/cmssdt/README.md](deploy/cmssdt/README.md) |
| a developer PC | local stack, daily refresh, nightly tester | this README |

## Documentation

- [docs/cmssdt-pipeline-guide.md](docs/cmssdt-pipeline-guide.md) -- plain-language overview of the pipeline
- [docs/current-neo4j-database-structure.md](docs/current-neo4j-database-structure.md) -- labels, relationships, identities
- [docs/llm-knowledge-contract.md](docs/llm-knowledge-contract.md) -- what the LLM may and may not do
- [docs/shifter-sso-cutover.md](docs/shifter-sso-cutover.md) -- CERN SSO for the Shift Console
- [docs/DEVELOPER.md](docs/DEVELOPER.md) -- architecture, request routing, invariants and the full command list, for developers

## Quick start (local)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in CERN_API_BASE, CERN_API_KEY; set NEO4J_URI=bolt://localhost:7688

# Neo4j on bolt://localhost:7688 (the port every command below uses)
docker run -d --name neo4j -p 7475:7474 -p 7688:7687 \
  -e NEO4J_AUTH=neo4j/<NEO4J_PASSWORD from .env> -v $HOME/neo4j-data:/data docker.io/library/neo4j:5
#   or the same containers as production (Neo4j, API, worker): see deploy/vm/README.md

uvicorn server.api_server:app --reload --port 8001
curl http://localhost:8001/api/health
```

To use it from the chat widget, set `VITE_CHAT_API_BASE=http://localhost:8001` in the
cmssdt-ib checkout's `.env.development` and restart the Vite dev server.

## Data sources

| Source | Importer | Production VM (worker) |
|---|---|---|
| Dashboard data: RelVals, unit tests, build status, merged PRs ([cms-sw.github.io](https://github.com/cms-sw/cms-sw.github.io) `data/`) | `ingestion/build_graph.py --cmsrepo` | every hour at :30 |
| RelVal exceptions (Elasticsearch, cms-bot `es-cmssw-exceptions.py`) | `build_graph.py --es-exceptions-json` | 00:30, 12:30 |
| Problems in the raw IB test logs (unit, addOn, RelVal) | `merge_test_log_batches.py` + `ingest_test_logs.py` | every 5 min, from what cmssdt sends (once cms-bot#2908 is merged) |
| Shifter documentation ([cmssdt-wiki](https://github.com/cms-sw/cmssdt-wiki)) | `ingest_shifter_docs.py` | 00:30, 12:30 |
| CMSSW GitHub PRs/Issues + PR patches (backup on cmssdt9-02) | `build_graph.py --github-issues`, `ingest_github_patches.py` | not yet (needs a Kerberos service account) |
| Clang warnings (`log.html` build logs) | `build_graph.py --clang-root` | no (staged by hand) |

All importers are incremental: each source file is hashed (`ImportState` nodes) and
skipped when unchanged; `--force` reimports. An IB is always the pair
`(tag, architecture)`: results for `el9_amd64_gcc13` and `el9_amd64_gcc14` stay separate.

### Dashboard data (cmsrepo)

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/cms-sw/cms-sw.github.io.git cmsrepo
cd cmsrepo && git sparse-checkout set data && cd ..

PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/build_graph.py \
  --cmsrepo cmsrepo
```

**Do not add `--reconcile` for cmsrepo.** cms-sw.github.io drops IBs after about two
weeks; `--reconcile` deletes graph records whose source file is gone, so it would delete
those IBs' RelVals, unit tests and PRs. Old data is removed by the retention script
instead (below). Only `relvals/<arch>/<date>/*.json` is read for RelVals: the GPU and
RNTuple results in `<date>/gpu/` and `<date>/other/rntuple/` are not imported.

`CMSSW_<major>_<minor>_X_<timestamp>` is a primary IB; a tag with an inserted variant,
such as `CMSSW_20_1_NONLTO_X_2026-08-10-2300`, is a sub-IB of
`CMSSW_20_1_X_2026-08-10-2300` (stored as `parent_tag` and `variant`).
"The latest IB" ranks the release series first, then the build time.

### RelVal exceptions

The CMSSDT logreader needs a CERN SSO login, so exceptions come from Elasticsearch:

```bash
es-cmssw-exceptions.py > exceptions-<date>.json       # cms-bot; covers the last 7 days
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/build_graph.py \
  --cmsrepo cmsrepo --es-exceptions-json exceptions-<date>.json
```

Use a new file name per export: importing a file first deletes what was imported earlier
from the same path. An exception is only stored when its release, architecture and
workflow match an imported `RelVal`.

### IB test logs

The dashboard only says whether a test passed. A test can pass although its log contains
a fatal exception, a crash, a sanitizer report or an infrastructure error. The test logs
are scanned so those problems are stored and can be shown to the shifter.

**Production path** (being switched on: the cms-bot hook is cms-sw/cms-bot#2908, still open;
the sender #2909 is merged; the Jenkins job is disabled until #2908 is in): cms-bot scans the logs where it already unpacks them
(`ib_test_log_problems.py`, a copy of `ingestion/ib_test_log_problems.py` and
`ib_test_log_scanner.py`; RelVals step by step). The Jenkins job `send-test-log-scans`
(`deploy/cmssdt/send_test_log_scans.py`) sends the scan files to
`POST /api/test-logs/batch` on the VM (token: `TEST_LOGS_INGEST_TOKEN`), which saves them
in `received/`. The worker merges them per IB into `results/`
(`ingestion/merge_test_log_batches.py`) and imports what changed
(`ingestion/ingest_test_logs.py`). Details: [deploy/cmssdt/README.md](deploy/cmssdt/README.md).

The raw zips are on `cmssdt9-02:/data/sdt/buildlogs/<arch>/www/<day>/<queue>/<IB tag>/`
(`unitTestLogs.zip`, `addOnTests.zip` + `addOnTests.log`, `pyRelValMatrixLogs.zip`), deleted by
the server after about a week. `results/` (a few MB per day) is the durable copy. To import
results by hand:

```bash
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python \
  ingestion/ingest_test_logs.py --results <results dir>
```

One zip by hand, read-only: `PYTHONPATH=ingestion venv/bin/python ingestion/ib_test_log_scanner.py <zip>`.

**What is extracted** (`ingestion/ib_test_log_scanner.py`; logs are streamed, never
loaded whole -- one CLANG log has a single 2 GB line):

- every unit test, addOn test step and RelVal workflow, with its status from the log itself;
- fatal exceptions, sanitizer/UBSan reports, crashes, assertions, uncaught exceptions,
  Python tracebacks and known infrastructure errors, with up to 60 evidence lines, the
  test they belong to and whether it passed;
- MessageLogger warnings/errors by category, and plain error/failed lines, both grouped;
  `%MSG-i`/`%MSG-d` debug output is dropped.

Each problem gets a normalized signature (dates, paths, IB names, numbers, host names,
GUIDs and hex values stripped), so the same problem is recognised across IBs and
architectures ("first seen", "last seen"). Not stored: the raw logs, debug messages,
stack frames beyond 20, evidence beyond 60 lines per problem, plain error lines beyond
1,000 distinct ones per test (those are still counted). The import never creates `IB`
nodes: results attach to IBs imported from the dashboard data.

**In the chat:** a pasted error (CMSSW exception, sanitizer report, traceback, assertion or
crash, whole or a fragment) is read with the same parser and searched at once by exact
signature, by the pasted text, by every specific name in it (plugin, class, module,
library -- never just the exception category) and by producer. The answer says whether the
exact error is in the logs, which tests/workflows hit it and with what status, how many IBs
and architectures, first and last seen, an evidence excerpt, related PRs/Issues, and which
keys it searched by. For the Shift Console, `GET /api/test-log-problems?from=&to=[&arch=]`
returns problems in PASSED tests as `new` / `returned` / `gone` / `still_present` per track.

### Shifter documentation

```bash
git clone --depth 1 https://github.com/cms-sw/cmssdt-wiki.git shifter-wiki
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/ingest_shifter_docs.py \
  --wiki-root shifter-wiki --cmsrepo cmsrepo --reconcile
```

`--cmsrepo` also imports cmsrepo's own top-level `*.md` pages. The chat cites these
documents through `search_shifter_docs` (fulltext index `shifterDocText`).

### GitHub PRs and Issues

The dashboard data brings PR metadata (`cmsrepo/data/prs`, `cmsrepo/cmssw-prs-cache.json`)
and workflow command hashes (`cmsrepo/data/commands`); a command is linked to a PR only on
an exact commit-hash prefix match. The full CMSSW GitHub backup on cmssdt9-02
(`/data/sdt/SDT/public/github/cms-sw/cmssw/issues`, freshness in `issues/latest.txt`) adds
bodies, labels, comments and explicit `#number`/URL references:

```bash
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/build_graph.py \
  --cmsrepo cmsrepo --github-issues <dir or .tar.gz>
PYTHONPATH=ingestion NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/ingest_github_patches.py \
  --github-issues <dir>          # which files each PR touched, from its patch.txt
```

References, changed files and comment mentions are relevance evidence only, never proof
that a PR caused or fixed something; every answer that shows them says so.
Nothing here calls the GitHub API: GitHub data comes only from the backup.

### Clang warnings

`--clang-root` reads `<root>/<ib-tag>/<arch>/build-logs/**/log.html`
(`ingestion/clang_parser.py`). This corpus is staged by hand outside cmsrepo.

### Retention

The graph keeps 90 days of per-build data. PRs, Issues, IBs, `ProblemFirstSeen` and
`Concept` nodes are kept. Dry run unless `--apply`; `--chat` also drops idle chat
sessions and router shadow logs:

```bash
NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/prune_old_data.py [--days 90] [--apply]
```

## API

| Endpoint | What |
|---|---|
| `POST /api/chat`, `GET /api/health` | the chat (request/response shape shared with the widget) |
| `GET /api/chat/<id>/history`, `GET /api/chat/<id>/export.md`, `DELETE /api/chat/<id>` | a stored conversation |
| `POST /api/feedback` | correct/incorrect verdict on an answer |
| `GET /api/sample-questions` | sample questions and "ask next" suggestions |
| `GET /api/shift-summary`, `GET /api/shift-summary/windows` | the shift digest (markdown; its table shape is shared with the frontend parser) |
| `GET /api/test-log-problems` | problems in PASSED tests between two builds (JSON) |
| `GET /api/release-explorer/failing` | failing items per build, for the Release Explorer |
| `GET /shift`, `GET /api/shift-whoami` | the Shift Console page, behind CERN SSO |
| `POST /api/test-logs/batch`, `POST /api/test-logs/ingest` | receive test-log scan files (token) |
| `GET/POST /api/alerts`, `POST /api/alerts/evaluate`, `DELETE /api/alerts/<id>`, `GET /api/alerts/status` | alert rules |
| `POST /api/ingest`, `POST /api/search` | Log RAG ingestion and evidence search |

### How a question is answered

1. **Deterministic first.** `server/request_planner.py` extracts tags, architectures,
   PR/workflow numbers and keywords by regex and picks a capability; the `_*_answer()`
   helpers in `server/api_server.py` answer it with Cypher.
2. **LLM investigation** when no lookup fits: the model proposes a plan from registered
   capabilities and read-only graph tools only (`graph/tools.py`, `graph/analytics.py`),
   the plan is validated and run, and the answer is written from that evidence only.
3. **AI supervisor** (`ENABLE_AI_SUPERVISOR`): every answer is read by the LLM against the
   question; one that doesn't fit is sent back through the lookups or to an investigation.
   Quick checks ask the main model (`CERN_MODEL`) and the review model (`CERN_REVIEW_MODEL`)
   at once.

### LLM settings

| Variable | `.env.example` | Production VM | What |
|---|---|---|---|
| `CERN_MODEL` | `hf-qwen3-32b-awq` | `glm-5.3-flash-int4` | main model (CERN LLM gateway, `CERN_API_BASE`/`CERN_API_KEY`) |
| `CERN_REVIEW_MODEL` | `gpt-oss-20b` (code default) | `gpt-oss-20b` | second opinion for the supervisor's quick checks |
| `ENABLE_AI_SUPERVISOR` | off | on | read every answer against the question (above) |
| `ENABLE_ROUTER_SHADOW` | off | on | log an LLM classification next to the regex router (`RouterShadowLog`), never used to answer |
| `ENABLE_CAPABILITY_FALLBACK_TO_LLM` | off | on | when the router picked a capability but no helper understood the phrasing, investigate instead of answering "no imported local evidence" |
| `ENABLE_DETERMINISTIC_ANSWER_VERIFICATION` | off | on | check each deterministic answer against the question (e.g. "NOT failing" vs "failing"); fails open |
| `ENABLE_ANSWER_SYNTHESIS` | off | off | rephrase a deterministic answer, only if every number is kept |

## Conversation history and feedback

Turns with the same `session_id` are stored as `ChatSession`/`ChatTurn` nodes; imported
facts are never changed. `POST /api/feedback` verdicts are checked against the graph
(deterministically for latest-IB questions, otherwise by an LLM with read-only graph tools)
and come back as `accepted`, `rejected` or `pending`. An accepted correction becomes a
`Concept` fact only because the graph confirmed it, never because a user said so.
Concepts expire (`CONCEPT_TTL_DAYS`, default 14); `ingestion/revalidate_feedback.py`
re-checks pending feedback and Concepts near expiry against the current graph.

```json
{"question": "What is the latest IB?", "answer": "...", "verdict": "incorrect",
 "corrected_answer": "CMSSW_20_1_X_2026-08-10-1100 is the latest IB.",
 "ib_tag": "CMSSW_20_1_X_2026-08-10-1100"}
```

## Alerts

`AlertRule` nodes (`graph/alerts.py`) define threshold checks: `relval_failure_count`,
`relval_recurrence`, `latest_ib_stalled`, `unittest_failure_count`, `build_failure_count`
and the regression rules. They are evaluated by `POST /api/alerts/evaluate` or
`ingestion/run_alerts.py` and notify through a log notifier.

## Log RAG

A separate pipeline for arbitrary JSON/JSONL and Clang `log.html` inputs (`log_rag/`):
events get deterministic IDs and redaction, the original payload is archived with a
checksum manifest under `LOG_RAG_ARCHIVE_DIR` (default `data/log-rag/`), event context is
projected into Neo4j, and chat responses carry the retrieved evidence as `citations`.

```bash
curl -X POST 'http://localhost:8001/api/ingest?source=build-42.jsonl&build=CMSSW_20_1_X' \
  -H 'Content-Type: application/json' --data-binary @build-42.jsonl
curl -X POST http://localhost:8001/api/search -H 'Content-Type: application/json' \
  -d '{"question":"Which service emitted timeout errors?"}'
```

OpenSearch (`OPENSEARCH_URL`) is optional and not used on the production VM; without it,
retrieval reads the archive files.

## Tests and validation

```bash
# unit tests (unittest, not pytest); mocked, no database needed -- the same as GitHub Actions
PYTHONPATH=ingestion python -m unittest discover -s tests

# live checks against a running server and real data
venv/bin/python scripts/validate_bot.py [--base-url http://localhost:8002]   # multi-turn scenarios
venv/bin/python scripts/question_bank.py --base-url http://localhost:8002    # ~320 real questions
venv/bin/python scripts/question_bank.py --check-facts                       # which expected facts went stale
venv/bin/python scripts/validate_shift_summary.py                            # /api/shift-summary structure
venv/bin/python scripts/validate_test_log_data.py --base-url http://localhost:8002 --username <user>
venv/bin/python scripts/stress_test_bot.py                                   # load test
```

Nightly testing on the test machine (a copy of the production data, the live checks
above, reports) is kept on the separate `tester` branch, not on `master`.

## Project layout

```
server/
  api_server.py          FastAPI app: chat routing, the _*_answer() helpers, all endpoints
  request_planner.py     regex entity extraction + capability scoring (no LLM)
  agent.py               LLM tool-calling loop, plan/verify/supervise calls
  llm_client.py          CERN LLM gateway client (CERN_MODEL, CERN_REVIEW_MODEL)
  knowledge_contract.py  fixed operational knowledge given to the model
  github_ask.py          "is there a PR/Issue about X?" by rules
  pr_cause.py            "which PR may have caused this?" from the graph
  paste_diagnosis.py     reads a pasted CMSSW fatal exception like a shifter would
  cmssw_source.py        read-only lookups in a local CMSSW checkout (CMSSW_SOURCE_DIR)
  question_catalog.py    known questions the lookups answer, mapped from what people type
  sample_questions.py    sample questions and "ask next" suggestions
  shifter_auth.py        CERN SSO (OIDC) for /shift
  test_log_ingest.py     receiving end of the test-log scan files
  static/                index.html, shift.html
graph/
  neo4j_conn.py          connection helper, schema constraints and indexes
  tools.py               GRAPH_SCHEMA + the read-only Cypher tools (query_graph is MATCH/RETURN only)
  analytics.py           rankings, comparisons, date windows for the LLM's tool calls
  alerts.py              alert rules
ingestion/
  build_graph.py         cmsrepo, exceptions, Clang, GitHub export -> Neo4j
  clang_parser.py        Clang warnings from log.html
  relval_log_parser.py   RelVal exception text from the Elasticsearch export
  ingest_shifter_docs.py cmssdt-wiki + cmsrepo pages -> DocChunk
  ingest_github_patches.py  files each PR touched, from the backup's patch.txt
  ib_test_log_scanner.py the test-log parser (also copied into cms-bot)
  ib_test_log_problems.py   the cms-bot entry point (also copied into cms-bot)
  merge_test_log_batches.py batches from cmssdt -> results/, per IB
  ingest_test_logs.py    results/ -> TestLogRun, LogProblem, ProblemSignature, ProblemFirstSeen
  prune_old_data.py      90-day retention
  generate_shift_summaries.py, log_summary.py  shift digest data
  revalidate_feedback.py, run_alerts.py  scheduled checks
log_rag/                 Log RAG: models, archive, pipeline, retriever, opensearch
deploy/vm/               production VM: compose.yaml, worker, backup/restore, setup, runbook
deploy/cmssdt/           the Jenkins sender for the test-log scan files
scripts/                 live validation scripts, question bank, load test
tests/                   unit tests (mocked Neo4j and LLM)
```
