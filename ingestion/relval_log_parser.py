"""RelVal exception-text extractor for Elasticsearch-sourced data.

Parses the `exception` field cmssdt-ib-matrix-* documents store in
Elasticsearch (e.g. via cms-bot's es-cmssw-exceptions.py, redirected to a
JSON file and imported with --es-exceptions-json), e.g.:

    An exception of category 'WrongTrackerSubDet' occurred while
    [0] Processing  Event run: 1 lumi: 7 event: 601 stream: 3
    [1] Running path 'validation_step13'
    [2] Calling method for module Phase2TrackerValidateDigi/'pixDigiValid'
    Exception Message:
    Invalid DetID: no GeomDetUnit associated with raw ID 347616260 of subdet ID 2

A single field commonly concatenates several occurrences of the same
category back to back, with no delimiter between them other than the next
"An exception of category" line.

Self-contained, same spirit as clang_parser.py: no shared state, one function
that takes raw text and returns structured records.
"""
import re

CATEGORY_RE = re.compile(r"^An exception of category '([^']+)' occurred")
CONTEXT_RE = re.compile(r"^\s*\[\d+\]\s+(.*)$")
MESSAGE_HEADER_RE = re.compile(r"^Exception Message:\s*$")


def extract_es_exception_blocks(exception_text: str) -> list[dict]:
    """Return one record per exception block, in text order: category,
    message, and context (the numbered [0]/[1]/... frames, newline-joined).

    No line_start/line_end: this is a structured field, not a byte offset
    into a real log file, so there's nothing to cite a line range in.
    """
    lines = exception_text.splitlines()
    results = []
    i = 0
    n = len(lines)
    while i < n:
        start_match = CATEGORY_RE.match(lines[i].strip())
        if not start_match:
            i += 1
            continue
        category = start_match.group(1)
        context_lines: list[str] = []
        message_lines: list[str] = []
        in_message = False
        j = i + 1
        while j < n and not CATEGORY_RE.match(lines[j].strip()):
            stripped = lines[j].strip()
            if MESSAGE_HEADER_RE.match(stripped):
                in_message = True
            elif in_message:
                if stripped:
                    message_lines.append(stripped)
            else:
                context_match = CONTEXT_RE.match(lines[j])
                if context_match:
                    context_lines.append(context_match.group(1).strip())
            j += 1
        message = " ".join(" ".join(message_lines).split())
        results.append({
            "category": category,
            "message": message,
            "context": "\n".join(context_lines),
        })
        i = j
    return results
