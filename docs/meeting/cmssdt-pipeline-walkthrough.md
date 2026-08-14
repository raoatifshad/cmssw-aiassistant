# CMSSDT Pipeline Walkthrough

This guide explains the current system diagram in everyday language. It is
intended to help someone understand how a CMSSDT question becomes an answer,
what Neo4j contributes, and why the application does not call an LLM first.

The editable diagram is [cmssdt-current-pipeline.drawio](../cmssdt-current-pipeline.drawio).
For a presentation-ready overview, see
[cmssdt-latest-pipeline.svg](cmssdt-latest-pipeline.svg).

## The short version

The system has two jobs:

1. Keep a connected, inspectable record of CMSSDT data up to date.
2. Answer a user question from those records.

The left three lanes of the diagram prepare trusted data. The right lane
answers a question. The API handles known factual questions directly; it uses
an LLM only when the question needs broader interpretation.

## How to read the picture

The diagram is read from left to right.

| Lane | Meaning |
| --- | --- |
| `01 Source data` | Where the raw CMSSDT facts originate. |
| `02 Incremental ingestion` | Code that parses and refreshes only changed data. |
| `03 Neo4j knowledge graph` | The connected store of facts and relationships. |
| `04 Chat query path` | What happens after someone asks a question. |

The arrows mean a record or query is moving from one responsibility to the
next. They do not mean every user question visits every box.

## 1. Source data

The system starts with information that already exists elsewhere. It does not
invent build status or scrape it while a user waits.

### `cmsrepo/data`

This source provides structured CMSSW build information, including queue
summaries, RelVal results, unit tests, and aggregate PR data.

Example: a RelVal record can say that workflow `136.88811` ran for a particular
IB and architecture, ended as `FAILED`, and has diagnostic counters.

### Raw CMSSW GitHub export

This contains Issues, pull requests, comments, labels, authors, and the text
that mentions other GitHub items.

Example: if a PR description says `Fixes #12345`, the importer can represent
that explicit reference as a graph relationship. It does **not** prove that
the PR caused a specific build warning.

### External Clang logs

Clang logs contain individual diagnostic occurrences. They supply the detail
behind a warning: its message, package, source file, and checker when known.

Example: a question such as "Which packages have this Clang warning?" needs
these individual log-derived facts, not only a total warning count from a build
summary.

### Log RAG API payloads

These are archived event payloads with checksums and provenance. Neo4j keeps
their inspectable facts; OpenSearch can optionally support evidence retrieval.

Their role is to preserve what was received and where it came from, rather
than make an LLM answer from an untracked text fragment.

## 2. Incremental ingestion

`ingestion/build_graph.py` is the main importer. It turns source-specific
formats into a consistent graph model.

### Why it is incremental

Importing every file again for every update is slow and risks needless churn.
Each source unit gets a content hash stored in an `ImportState` node.

1. The importer calculates a hash for an input file or payload.
2. It compares that value with the previous hash.
3. If the input is unchanged, it skips it.
4. If it changed, it replaces only graph records owned by that input.

This keeps updates quick while preserving unrelated imported facts.

### What the parsing boxes do

- **Build status** imports status and summary warning counts per IB and
  architecture.
- **GitHub import** creates PR, Issue, comment, label, author, and explicit
  reference facts.
- **Clang parser** extracts detailed diagnostic occurrences and connects them
  to packages, source files, and checkers.
- **Log RAG archive and projection** records event data, checksum, and
  provenance in a queryable form.

## 3. Neo4j knowledge graph

Neo4j is the system's factual store. Its value is not only that it stores data,
but that it stores relationships between data.

### IB results and diagnostics

An `IB` is identified by both its tag and architecture. This matters because
the same tag can have different results on different platforms.

The graph links an IB to RelVals, unit tests, and warnings. The importer also
derives `parent_tag` and `variant` from an IB tag.

Example:

```text
CMSSW_20_1_X_2026-08-10-2300          primary IB
CMSSW_20_1_NONLTO_X_2026-08-10-2300   sub-IB variant of that primary IB
```

This prevents the NONLTO run from being reported as an unrelated latest IB.

### GitHub discussion facts and references

The graph connects GitHub items to comments, labels, and authors. It can also
connect a body or comment to a PR/Issue mentioned by number.

This makes useful questions possible, such as "Show Issues that mention this
package" or "Which PRs explicitly reference Issue #12345?"

The red evidence note in the diagram is intentional: textual relevance is not
causality. A PR that changes the same file as a warning is worth showing, but
the graph does not claim it introduced or fixed the warning unless a source
explicitly records that fact.

### Conversation and feedback records

The graph also stores application metadata separately from imported CMSSDT
facts:

- A `ChatSession` owns ordered `ChatTurn` records.
- An answer can retain its route and citations.
- Feedback records retain a user verdict and verification result.

Feedback never changes a RelVal, warning, PR, or other imported source fact.
It is an observation about the answer, not a rewrite of evidence.

## 4. The chat query path

When the browser sends `POST /api/chat`, FastAPI receives the question and an
optional session ID. The next decision is the key design choice in this
system: can this question be answered as a known factual query?

### Path A: deterministic answer first

For recognized question shapes, Python chooses a specific Neo4j query and
formats its result. No LLM request is made.

Typical examples:

- "What is the latest IB?"
- "How many RelVal errors are in `CMSSW_20_1_EVTGEN3_X_2026-08-09-2300`?"
- "What are the build-status warnings for this IB?"
- "Compare these two IBs."
- "Which PRs or Issues have direct graph evidence for this term?"

This is the right path for counts, statuses, identifiers, and known reports
because the application already knows exactly what they mean.

#### Example: latest IB

The handler selects primary IB tags, not variant tags. It then finds their
recorded variants and returns them beneath the primary run. This is more
reliable than asking a model to infer tag semantics from a list of strings.

#### Example: RelVal failures versus counters

For an error-count question, the system counts failed RelVal workflows and can
separately report aggregate step-level error counters. Those are different
facts. For the imported EVTGEN3 example, `13` is the failed-workflow count;
`100,777` is an aggregate diagnostic counter across step records.

### Path B: LLM graph fallback

If no deterministic route matches, the system creates or restores a
`KGChatSession`. The LLM receives the question and recent conversation context,
then uses named graph tools or guarded read-only Cypher to retrieve evidence.

Example:

```text
Which open discussions might be relevant to warnings in this package, and why?
```

This requires interpreting words such as "relevant" and combining multiple
types of evidence. The LLM is useful for that explanation, but it is limited
to read-only retrieval. It cannot write Cypher, change imported data, or turn a
text match into a causal claim.

## 5. Why this pipeline was chosen

This is a **deterministic-first, knowledge-graph-grounded assistant**.

| Need | Pipeline choice | Benefit |
| --- | --- | --- |
| Exact operational facts | Deterministic handler and focused Neo4j query | Correct meaning for counts and statuses; low latency. |
| Cross-system context | Neo4j relationships | Connects builds, diagnostics, and GitHub records without manual joins. |
| Flexible natural-language questions | LLM fallback with read-only tools | Provides interpretation without making the model the database. |
| Evidence and auditability | Stored sources, hashes, chat turns, and citations | A result can be inspected and explained later. |
| Safe updates | Incremental source ownership | Changed imports do not overwrite unrelated graph facts. |

A pure LLM-first chatbot would be easier to route, but it would make the model
decide what every factual question means. That creates avoidable ambiguity for
CMSSDT questions such as which IB is "latest" or whether an `errors` field
means failed workflows or summed diagnostic events.

Conversely, a deterministic-only system would be precise but would require a
new endpoint or parser rule for every wording and exploratory question. The
two paths combine the strengths of both approaches.

## 6. End-to-end examples

### Exact factual question

Question:

```text
What is the latest IB?
```

Flow:

1. Browser sends the question to `/api/chat`.
2. The deterministic router recognizes a latest-IB question.
3. It queries primary IB records in Neo4j and discovers variants through
   `parent_tag`.
4. The API formats the primary result plus its sub-IBs.
5. The answer and question are saved as chat turns.

No LLM call is needed.

### Open-ended evidence question

Question:

```text
What GitHub work looks relevant to this Clang warning?
```

Flow:

1. The router sees that the request is exploratory rather than an exact
   report.
2. The LLM fallback retrieves warning, file, package, PR, and Issue evidence
   through read-only tools.
3. It explains the evidence strength: same changed file is stronger than a
   title-text match, and neither alone proves cause or fix.
4. The API returns and saves the grounded explanation.

The answer should say what was found and what cannot be concluded from the
available evidence.

## 7. A simple way to explain it in a meeting

> We import CMSSDT and GitHub data into Neo4j as connected, inspectable facts.
> For known operational questions, the API queries those facts directly. For
> broader questions, an LLM can explain retrieved evidence but cannot change
> the data or bypass the graph. This gives us both reliable numbers and useful
> natural-language investigation.