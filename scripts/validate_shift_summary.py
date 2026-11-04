"""Live correctness checks for GET /api/shift-summary against a running server
+ real graph, mirroring how the Shift Console frontend (cmssdt-18V) actually
parses this markdown (see its src/Components/ShiftConsole/shiftMarkdown.js).

Unlike tests/test_shift_summary.py (mocked graph, fast, run in CI), this hits
a real server so it exercises real data shapes -- this is the layer that
caught the original bug this script is built around: a workflow (140.051)
appearing to contradictorily resolve on one sub-IB variant while remaining
unlabeled-identical to five other variant rows, which only a real multi-variant
IB set actually reproduces (see the "Correct RelVal resolved/failing
classification" and "Verify the LLM-fallback answer is grounded" fixes on
this branch).

Structural checks, per architecture and per newly-failing/resolved category
(RelVal, Unit Tests, Build, AddOn):

1. Every RelVal table (once it has any data rows) includes a "Variant"
   column, non-empty on every row -- the column that makes multiple
   independent sub-IB resolutions of the same workflow_id legible instead of
   looking like duplicated/contradictory data.
2. No identity (workflow_id+variant for RelVal, name for everything else)
   appears in BOTH the "Newly failing" and "Resolved" table for the same
   architecture+category in the same digest -- a real self-contradiction,
   not a display quirk.
3. Every data row has the same cell count as its table's header.
4. "Newly failing Unit Tests" has exactly the header
   | Name | Errors | Tests | Failed tests | Exception | Recurrence |
   and on every row: Tests is "<N> failed / <M>" (or empty when the dashboard has no
   ok count), Failed tests is either test names or exactly one of the two fixed strings
   (then Exception is empty), Recurrence is non-empty in the RelVal wording.
   ("Resolved Unit Tests" stays | Name | Errors |.) Cells may carry an escaped "\|".

Usage:
    venv/bin/python scripts/validate_shift_summary.py [--base-url http://localhost:8002]
        [--username mrao] [--windows 3]

Requires the API server running with DEV_SHIFTER_LOGIN=true (this script
logs in via the same /shift/dev-login bypass documented in
docs/shifter-sso-cutover.md -- it never talks to real CERN SSO) and against
real imported data.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

import httpx

HEADING_RE = re.compile(r"^(#{2,4})\s+(.*)$")
SEPARATOR_CELL_RE = re.compile(r"^:?-{2,}:?$")


def split_table_row(row: str) -> list[str]:
    """Same as the frontend's splitTableRow: split on unescaped "|" only, then unescape."""
    row = row.strip()
    row = re.sub(r"^\|", "", row)
    row = re.sub(r"(?<!\\)\|$", "", row)
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", row)]


UNITTEST_NEWLY_HEADER = ["Name", "Errors", "Tests", "Failed tests", "Exception", "Recurrence"]
UNITTEST_FIXED_FAILED = ("no logs imported for this build yet", "not found in the imported log")
TESTS_CELL_RE = re.compile(r"^\d+ failed / \d+$")
RECURRENCE_CELL_RE = re.compile(
    r"^(?:Failed in \d+ prior build\(s\) this series \(most recent CMSSW_\S+\)|First failure in this series); "
    r"\d+/\d+ overall\.$")


def check_unittest_table(section, label: str, arch_name: str) -> list[str]:
    """Rule 4: the Newly failing Unit Tests columns the frontend's unit-test cards read."""
    violations = []
    header = section.table.header
    if header != UNITTEST_NEWLY_HEADER:
        return [f"[{label}] {arch_name}/{section.heading}: header {header} != {UNITTEST_NEWLY_HEADER}"]
    for row in section.table.rows:
        if len(row) != len(header):
            continue  # reported by rule 3
        cells = dict(zip(header, row))
        where = f"[{label}] {arch_name}/{section.heading} {cells['Name']!r}"
        if cells["Tests"] and not TESTS_CELL_RE.match(cells["Tests"]):
            violations.append(f"{where}: Tests {cells['Tests']!r} is not '<N> failed / <M>'")
        failed = cells["Failed tests"]
        if not failed:
            violations.append(f"{where}: empty Failed tests (must be names or a fixed string)")
        elif failed in UNITTEST_FIXED_FAILED:
            if cells["Exception"]:
                violations.append(f"{where}: Exception {cells['Exception'][:60]!r} without failed tests")
        elif any(fixed in failed for fixed in UNITTEST_FIXED_FAILED):
            violations.append(f"{where}: a fixed string mixed with test names: {failed!r}")
        if not RECURRENCE_CELL_RE.match(cells["Recurrence"]):
            violations.append(f"{where}: Recurrence {cells['Recurrence']!r} not in the RelVal wording")
    return violations


def is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(SEPARATOR_CELL_RE.match(cell) for cell in cells)


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]


@dataclass
class Section:
    heading: str
    table: Table | None = None
    is_none_text: bool = False


@dataclass
class Arch:
    name: str
    sections: dict[str, Section] = field(default_factory=dict)


def parse_digest(markdown: str) -> list[Arch]:
    """Structural walk matching buildDigestDocument() in shiftMarkdown.js:
    ## title -> ### architecture -> #### section, with |-tables or a bare
    "None." paragraph as each section's body."""
    lines = markdown.replace("\r\n", "\n").split("\n")
    archs: list[Arch] = []
    current_arch: Arch | None = None
    current_heading: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        trimmed = line.strip()
        if trimmed == "":
            i += 1
            continue
        heading = HEADING_RE.match(trimmed)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).strip()
            if level == 3:
                current_arch = Arch(name=text)
                archs.append(current_arch)
                current_heading = None
            elif level == 4:
                current_heading = text
                if current_arch is not None:
                    current_arch.sections[current_heading] = Section(heading=current_heading)
            i += 1
            continue
        if trimmed.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_table_row(lines[i]))
                i += 1
            header = rows[0] if rows else []
            data_rows = rows[2:] if len(rows) > 1 and is_separator_row(rows[1]) else rows[1:]
            if current_arch is not None and current_heading is not None:
                current_arch.sections[current_heading].table = Table(header=header, rows=data_rows)
            continue
        if trimmed.lower().rstrip(".") == "none" or trimmed == "None.":
            if current_arch is not None and current_heading is not None:
                current_arch.sections[current_heading].is_none_text = True
            i += 1
            continue
        i += 1
    return archs


# (newly-failing heading, resolved heading, requires-variant-column) triples --
# requires-variant-column is True only for the two RelVal sections, the only
# category whose identity (workflow_id) is legitimately reused across
# independent sub-IB variant tracks.
CATEGORY_PAIRS = [
    ("Newly failing RelVal workflows", "Resolved RelVal workflows", True),
    ("Newly failing Unit Tests", "Resolved Unit Tests", False),
    ("Newly failing Build", "Resolved Build", False),
    ("Newly failing AddOn", "Resolved AddOn", False),
]


def row_identity(header: list[str], row: list[str], with_variant: bool) -> tuple:
    lower = [h.lower() for h in header]
    name_idx = lower.index("workflow") if "workflow" in lower else lower.index("name")
    key = (row[name_idx],)
    if with_variant and "variant" in lower:
        key = key + (row[lower.index("variant")],)
    return key


def check_digest(markdown: str, label: str) -> list[str]:
    violations: list[str] = []
    archs = parse_digest(markdown)
    if not archs:
        return [f"[{label}] no architecture sections parsed at all -- markdown: {markdown[:200]!r}"]

    for arch in archs:
        for newly_heading, resolved_heading, needs_variant in CATEGORY_PAIRS:
            newly = arch.sections.get(newly_heading)
            resolved = arch.sections.get(resolved_heading)

            for section, side in ((newly, "newly-failing"), (resolved, "resolved")):
                if section is None or section.table is None:
                    continue
                header = section.table.header
                if needs_variant and "Variant" not in header:
                    violations.append(
                        f"[{label}] {arch.name}/{section.heading}: table has rows but no "
                        f"'Variant' column (header={header})"
                    )
                for row in section.table.rows:
                    if len(row) != len(header):
                        violations.append(
                            f"[{label}] {arch.name}/{section.heading}: row {row!r} has "
                            f"{len(row)} cells, header has {len(header)}"
                        )
                    elif needs_variant and "Variant" in header:
                        variant = row[header.index("Variant")]
                        if not variant.strip():
                            violations.append(
                                f"[{label}] {arch.name}/{section.heading}: row {row!r} has an "
                                f"empty Variant cell"
                            )

            if newly_heading.lower().startswith("newly failing unit tests") and newly and newly.table:
                violations.extend(check_unittest_table(newly, label, arch.name))
            if newly and newly.table and resolved and resolved.table:
                newly_ids = {row_identity(newly.table.header, row, needs_variant) for row in newly.table.rows}
                resolved_ids = {row_identity(resolved.table.header, row, needs_variant) for row in resolved.table.rows}
                overlap = newly_ids & resolved_ids
                if overlap:
                    violations.append(
                        f"[{label}] {arch.name}: identity(ies) {sorted(overlap)} appear in BOTH "
                        f"'{newly_heading}' and '{resolved_heading}' -- self-contradictory digest"
                    )
    return violations


def dev_login(client: httpx.Client, username: str) -> None:
    resp = client.get("/shift/dev-login", params={"username": username})
    who = client.get("/api/shift-whoami")
    who.raise_for_status()
    if who.json().get("username") != username:
        raise RuntimeError(
            f"dev-login did not establish a session for {username!r} (got {who.json()!r}). "
            "Is DEV_SHIFTER_LOGIN=true and is that username in SHIFTER_CERN_USERNAMES?"
        )


def run(base_url: str, username: str, num_windows: int, num_cycles: int = 3) -> int:
    # Windows list is every historical window, not just adjacent ones --
    # found live 2026-10-01: a window spanning two distant release series
    # (e.g. CMSSW_16_1_X -> CMSSW_17_0_X) legitimately takes ~40s (confirmed
    # server-side: 0 errors logged, a real 200 eventually), since a bigger
    # generational diff means more newly-failing workflows, each needing its
    # own PR/Issue evidence lookup. 30s was tripping on that real latency,
    # not a hang -- this is a correctness checker, not a performance test.
    client = httpx.Client(base_url=base_url, timeout=90.0)

    health = client.get("/api/health")
    health.raise_for_status()
    print(f"Server reachable: {health.json()}")

    dev_login(client, username)
    print(f"Signed in as {username!r} via dev-login bypass.\n")

    windows_resp = client.get("/api/shift-summary/windows")
    windows_resp.raise_for_status()
    tags = [w["release_name"] for w in windows_resp.json().get("windows", [])]
    print(f"{len(tags)} imported window tag(s) available.")

    # Only windows a shifter can pick: the Shift Console limits its pickers to one release cycle
    # (ShiftConsolePage.js releaseCycle). The windows list is across all cycles and builds of one
    # timestamp come in random order per server process, so pairing neighbours gave a random
    # cross-series window (found 2026-11-04: 20_1 -> 13_0 failed on one server, passed on another).
    by_cycle: dict[str, list[str]] = {}
    for tag in tags:
        by_cycle.setdefault(tag.rsplit("_", 1)[0], []).append(tag)
    def series_key(cycle: str) -> tuple[int, int]:
        major, minor = cycle.split("_")[1:3]
        return int(major), int(minor)
    checks: list[tuple[str, str]] = [("default (latest vs previous)", None)]
    for cycle in sorted(by_cycle, key=series_key, reverse=True)[:num_cycles]:
        cycle_tags = by_cycle[cycle]
        for i in range(min(num_windows, max(0, len(cycle_tags) - 1))):
            to_tag, from_tag = cycle_tags[i], cycle_tags[i + 1]
            checks.append((f"{from_tag} -> {to_tag}", (from_tag, to_tag)))

    all_violations: list[str] = []
    for label, window in checks:
        params = {}
        if window:
            params = {"from": window[0], "to": window[1]}
        resp = client.get("/api/shift-summary", params=params)
        if resp.status_code != 200:
            all_violations.append(f"[{label}] HTTP {resp.status_code}: {resp.text[:200]}")
            print(f"=== {label} === FAIL (HTTP {resp.status_code})")
            continue
        markdown = resp.json().get("markdown", "")
        violations = check_digest(markdown, label)
        status = "ok" if not violations else f"{len(violations)} violation(s)"
        print(f"=== {label} === {status}")
        all_violations.extend(violations)

    print("\n" + "=" * 70)
    if all_violations:
        print(f"{len(all_violations)} VIOLATION(S):")
        for v in all_violations:
            print(f"- {v}")
        return 1
    print("No violations across all checked windows.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--username", default="mrao", help="Must be in SHIFTER_CERN_USERNAMES.")
    parser.add_argument("--windows", type=int, default=3, help="Past windows to check per release cycle, beyond the default.")
    parser.add_argument("--cycles", type=int, default=3, help="Newest release cycles to check (highest series first).")
    args = parser.parse_args()
    sys.exit(run(args.base_url, args.username, args.windows, args.cycles))
