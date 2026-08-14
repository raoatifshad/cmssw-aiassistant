"""Clang warning extractor — standalone copy of the proven regex logic.

Handles both CMS flag formats:
  - Inline:    file:line:col: warning: message [flag]
  - Next-line: file:line:col: warning: message
               [cms.FunctionChecker]

Validated against real CMSSDT data: 7/7 on the MillePedeAlignmentAlgorithm
sample, 16140/16142 at full IB scale. This is a self-contained copy, not
an import from the other project — deliberately independent.
"""
import re
import html as html_module

WARNING_RE = re.compile(
    r"^(?P<file>[\w./-]+):(?P<line>\d+):(?P<col>\d+):\s*warning:\s*(?P<message>.+?)"
    r"(?:\s*\[(?P<flag>[\w.-]+)\])?\s*$"
)
FLAG_ONLY_LINE_RE = re.compile(r"^\[(?P<flag>[\w.-]+)\]\s*$")


def html_to_text(raw_html: str) -> str:
    text = re.sub(r"<[^>]+>", "", raw_html)
    return html_module.unescape(text)


def extract_warnings(log_text: str) -> list[dict]:
    results = []
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        m = WARNING_RE.match(line.strip())
        if not m:
            continue
        flag = m.group("flag")
        context_end = i + 3
        if not flag and i + 1 < len(lines):
            fm = FLAG_ONLY_LINE_RE.match(lines[i + 1].strip())
            if fm:
                flag = fm.group("flag")
                context_end = i + 4
        context = "\n".join(lines[max(0, i - 1): context_end])
        results.append({
            "file": m.group("file"),
            "line": int(m.group("line")),
            "message": m.group("message"),
            "flag": flag or "unknown",
            "snippet": context,
        })
    return results


def aggregate_occurrences(warnings: list[dict]) -> list[dict]:
    """Group by (file, line, message), count occurrences."""
    counts: dict[tuple, dict] = {}
    for w in warnings:
        key = (w["file"], w["line"], w["message"])
        if key not in counts:
            counts[key] = {**w, "count": 0}
        counts[key]["count"] += 1
    return list(counts.values())
