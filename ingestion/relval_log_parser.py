"""RelVal step-log Fatal Exception extractor.

Parses the ``----- Begin/End Fatal Exception -----`` blocks CMSSW's EDM
framework writes to pyRelValMatrixLogs step logs, e.g.:

    ----- Begin Fatal Exception 17-Aug-2026 18:18:37 CEST-----------------------
    An exception of category 'WrongTrackerSubDet' occurred while
       [0] Processing  Event run: 1 lumi: 7 event: 601 stream: 3
       [1] Running path 'validation_step13'
       [2] Calling method for module Phase2TrackerValidateDigi/'pixDigiValid'
    Exception Message:
    Invalid DetID: no GeomDetUnit associated with raw ID 347616260 of subdet ID 2
    ----- End Fatal Exception -------------------------------------------------

Self-contained, same spirit as clang_parser.py: no shared state, one function
that takes raw text and returns structured records.
"""
import re

BEGIN_RE = re.compile(r"^-{3,}\s*Begin Fatal Exception\b")
END_RE = re.compile(r"^-{3,}\s*End Fatal Exception\b")
CATEGORY_RE = re.compile(r"^An exception of category '([^']+)' occurred")
CONTEXT_RE = re.compile(r"^\s*\[\d+\]\s+(.*)$")
MESSAGE_HEADER_RE = re.compile(r"^Exception Message:\s*$")


def extract_fatal_exceptions(log_text: str) -> list[dict]:
    """Return one record per Begin/End Fatal Exception block, in file order.

    Each record: category, message, context (the numbered [0]/[1]/... frames,
    newline-joined), line_start/line_end (1-indexed, spanning the whole block —
    used to build a `#/start-end` logreader citation anchor).
    """
    lines = log_text.splitlines()
    results = []
    i = 0
    n = len(lines)
    while i < n:
        if not BEGIN_RE.match(lines[i].strip()):
            i += 1
            continue
        line_start = i + 1
        category = None
        context_lines: list[str] = []
        message_lines: list[str] = []
        in_message = False
        j = i + 1
        while j < n and not END_RE.match(lines[j].strip()):
            stripped = lines[j].strip()
            category_match = CATEGORY_RE.match(stripped)
            if category_match:
                category = category_match.group(1)
            elif MESSAGE_HEADER_RE.match(stripped):
                in_message = True
            elif in_message:
                if stripped:
                    message_lines.append(stripped)
            else:
                context_match = CONTEXT_RE.match(lines[j])
                if context_match:
                    context_lines.append(context_match.group(1).strip())
            j += 1
        line_end = (j + 1) if j < n else j
        message = " ".join(" ".join(message_lines).split())
        if category or message:
            results.append({
                "category": category or "Unknown",
                "message": message,
                "context": "\n".join(context_lines),
                "line_start": line_start,
                "line_end": line_end,
            })
        i = j + 1
    return results


def extract_es_exception_blocks(exception_text: str) -> list[dict]:
    """Parse the `exception` field cmssdt-ib-matrix-* documents already store
    in Elasticsearch (e.g. via es-cmssw-exceptions.py). Same block shape as
    extract_fatal_exceptions but pre-stripped of the "----- Begin/End Fatal
    Exception -----" delimiter lines, so the block boundary is the next
    category line (or end of text) instead. A single field commonly
    concatenates several occurrences of the same category back to back.

    No line_start/line_end here: this is a structured field, not a byte
    offset into a real log file, so there's nothing to cite a line range in.
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
