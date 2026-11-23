"""Merge the batches received from cmssdt9-02 (received/*.tar) into results/.

A batch holds the files cms-bot wrote: one per unit/addOn zip (already a whole result) and
one per RelVal step. results/ holds ONE file per build -- what ingest_test_logs.py imports:

    <results>/relval/<arch>/<IB tag>/<variant or _>/pyRelValMatrixLogs.json.gz

A RelVal build arrives in pieces (a batch of its jobs at a time), so its steps are merged into
that file: a step replaces the same step, all others stay. `steps_scanned` in the file says
which steps it holds. Rules:

  * newer wins: a later log (log_mtime), then a later scan of the same log. A failed scan
    (scan_error) never replaces a good scan of the same log;
  * a workflow that ran again: steps of the earlier run that did not run this time are dropped;
  * a step that ran but has no good scan (yet) is listed in source.not_scanned: unknown, not clean;
  * which jobs reported back (progress.json of the sender) goes into source.jobs/complete/closed.

Each file written is read back and compared with what went in. Only then the batch is moved
to received/merged/ (the worker deletes it there once it is imported). A batch that fails stays
and is tried again: merging twice changes nothing. Exit 1 = something failed.

    PYTHONPATH=ingestion python ingestion/merge_test_log_batches.py --received /data/received --results /data/results
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import zlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from server import test_log_ingest as tli  # noqa: E402

TARS_AT_ONCE = 40            # ~20,000 steps in memory
PROGRESS_STATE = "progress.json"
PROGRESS_KEEP_DAYS = 14
NOT_RUN = ("NOTRUN", "NORUN")


class MergeError(Exception):
    pass


def read_batch(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, mode="r:") as tar:
        return {info.name: tar.extractfile(info).read() for info in tar if info.isfile()}


def load_result(path: Path) -> dict:
    """The build's result so far; an unreadable file is an error, never "nothing yet"."""
    if not path.exists():
        return dict(kind="relval", workflows=[], problems=[], source={}, steps_scanned={})
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("steps_scanned", {})
    return data


def write_result(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            gz.write(json.dumps(data, sort_keys=True).encode())
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(tmp, path)


def run_state(doc: dict) -> str:
    """Same workflow.log = same run of the workflow."""
    text = json.dumps([doc["workflow_steps"], doc["workflow_exit_codes"]], sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def workflow_status(steps: dict) -> str | None:
    if not steps:
        return None
    return "passed" if all(s == "PASSED" for s in steps.values()) else "failed"


def is_newer(doc: dict, known: dict) -> bool:
    if doc.get("scan_error") and not known.get("scan_error") and doc["log_mtime"] <= known["log_mtime"]:
        return False
    if known.get("scan_error") and not doc.get("scan_error") and doc["log_mtime"] >= known["log_mtime"]:
        return True
    return (doc["log_mtime"], doc["scanned_at"]) >= (known["log_mtime"], known["scanned_at"])


def apply_steps(data: dict, docs: list[dict]) -> set[tuple[str, str]]:
    """Puts the step files into the build's result; returns the (workflow, step) that went in."""
    scanned, taken, gone = data["steps_scanned"], {}, set()
    workflows = {w["workflow"]: w for w in data["workflows"]}
    whole_scan = set(workflows) - set(scanned)      # from a whole-folder scan, not step by step
    for doc in docs:
        wf, step = doc["workflow"], str(doc["step"])
        known = scanned.setdefault(wf, {}).get(step)
        entry = dict(log_mtime=doc["log_mtime"], scanned_at=doc["scanned_at"], run=run_state(doc),
                     problems=len(doc["problems"]))
        if doc.get("scan_error"):
            entry["scan_error"] = str(doc["scan_error"])[:300]
        if known and (known == entry or not is_newer(doc, known)):
            continue                                # the same file again, or an older one
        scanned[wf][step] = entry
        taken[(wf, step)] = doc
        info = workflows.get(wf)
        if doc.get("scan_error") and info:
            continue                                # a failed scan did not read workflow.log
        if wf in whole_scan or not info or not info["steps"] or (info.get("log_mtime") or 0, info.get("scanned_at") or "") <= \
                (doc["log_mtime"], doc["scanned_at"]):
            workflows[wf] = dict(workflow=wf, steps=doc["workflow_steps"], exit_codes=doc["workflow_exit_codes"],
                                 status=workflow_status(doc["workflow_steps"]), run=run_state(doc),
                                 log_mtime=doc["log_mtime"], scanned_at=doc["scanned_at"])
    for wf in {w for w, _ in taken}:                # left over from an earlier run of the workflow
        info = workflows[wf]
        for step, entry in list(scanned[wf].items()):
            if entry["run"] != info.get("run") and info["steps"].get(step, "NOTRUN") in NOT_RUN:
                del scanned[wf][step]
                taken.pop((wf, step), None)
                gone.add((wf, step))

    replaced = set(taken) | gone
    touched = {w for w, _ in taken}

    def stays(p: dict) -> bool:
        wf = p.get("workflow")
        if "scan_step" in p:
            return (wf, p["scan_step"]) not in replaced
        return not (wf in touched and (p.get("step") is None or (wf, str(p["step"])) in replaced))

    data["problems"] = [p for p in data["problems"] if stays(p)] + [
        dict(p, scan_step=step) for (_, step), doc in sorted(taken.items()) for p in doc["problems"]]
    data["workflows"] = [workflows[wf] for wf in sorted(workflows)]
    return set(taken)


def not_scanned(data: dict) -> list[str]:
    """workflow:stepN that ran (or whose scan failed) but has no good scan."""
    out = []
    for w in data["workflows"]:
        steps = data["steps_scanned"].get(w["workflow"])
        if steps is None:
            continue                                # a whole-folder scan read everything
        ran = {s for s, status in w["steps"].items() if status not in NOT_RUN}
        bad = {s for s in ran if s not in steps} | {s for s, e in steps.items() if e.get("scan_error")}
        out += [f"{w['workflow']}:step{s}" for s in sorted(bad, key=int)]
    return out


def set_source(data: dict, docs: list[dict], progress: dict | None) -> None:
    source = data["source"]
    if docs:
        newest = max(docs, key=lambda d: d["scanned_at"])
        source.update({k: newest[k] for k in ("arch", "tag", "variant", "queue", "path")}, kind="relval",
                      scanned_by="merge_test_log_batches")
        source["scanned_at"] = max(newest["scanned_at"], source.get("scanned_at") or "")
        source["scanner_version"] = max(d["scanner_version"] for d in docs)
    if progress:
        jobs = sorted(set(progress["jobs"]) | set(source.get("jobs") or []))
        total = progress["total"] if progress["total"] is not None else source.get("jobs_total")
        source.update(jobs=jobs, jobs_done=len(jobs), jobs_total=total, closed=progress["closed"],
                      complete=total is not None and len(jobs) >= total)
    source["not_scanned"] = not_scanned(data)


def check_written(path: Path, taken: set[tuple[str, str]], docs: list[dict]) -> None:
    """Reads the file back: every step that went in is there, with all its problems."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for p in data["problems"]:
        if "scan_step" in p:
            counts[(p["workflow"], p["scan_step"])] += 1
    known = {(w, s): e for w, steps in data["steps_scanned"].items() for s, e in steps.items()}
    for key in taken:
        if key not in known or known[key]["problems"] != counts[key]:
            raise MergeError(f"{path}: {key[0]} step{key[1]} is not complete after writing")
    listed = {w["workflow"] for w in data["workflows"]}
    missing = {d["workflow"] for d in docs} - listed
    if missing:
        raise MergeError(f"{path}: workflows missing after writing: {sorted(missing)[:5]}")


def merge_build(path: Path, docs: list[dict], progress: dict | None) -> bool:
    """Returns whether the file changed."""
    data = load_result(path)
    before = json.dumps(data, sort_keys=True)
    taken = apply_steps(data, docs)
    set_source(data, docs, progress)
    if json.dumps(data, sort_keys=True) == before:
        return False
    write_result(path, data)
    check_written(path, taken, docs)
    return True


def load_progress(received: Path) -> dict:
    try:
        return json.loads((received / PROGRESS_STATE).read_text())
    except (OSError, ValueError):
        return {}


def save_progress(received: Path, state: dict) -> None:
    oldest = (dt.datetime.now() - dt.timedelta(days=PROGRESS_KEEP_DAYS)).isoformat()
    state = {k: v for k, v in state.items() if v.get("at", "") >= oldest}
    tmp = received / f".{PROGRESS_STATE}.part"
    tmp.write_text(json.dumps(state, sort_keys=True))
    os.replace(tmp, received / PROGRESS_STATE)


def merge_progress(state: dict, new: dict) -> set[str]:
    """Adds one progress.json to the state; returns the builds that changed."""
    changed = set()
    for key, entry in new.items():
        old = state.get(key) or dict(jobs=[], total=None, closed=False)
        now = dict(jobs=sorted(set(old["jobs"]) | set(entry["jobs"])),
                   total=entry["total"] if entry["total"] is not None else old["total"],
                   closed=entry["closed"] or old["closed"])
        if now != {k: old[k] for k in now}:
            changed.add(key)
        state[key] = dict(now, at=dt.datetime.now().isoformat(timespec="seconds"))
    return changed


def merge_tars(tars: list[Path], received: Path, results: Path, log=print) -> tuple[dict, list[Path], list[str]]:
    """Merges some batches. Returns (counts, the batches that are completely in, errors)."""
    counts, errors, failed_tars = defaultdict(int), [], set()
    steps: dict[Path, list[tuple[Path, dict]]] = defaultdict(list)
    state, changed = load_progress(received), set()
    good = []
    for tar in tars:
        try:
            files = read_batch(tar)
            for name, payload in files.items():
                tli.check_member(name, payload)
        except (tarfile.TarError, OSError, EOFError, tli.Rejected) as exc:
            bad = received / "bad"
            bad.mkdir(exist_ok=True)
            os.replace(tar, bad / tar.name)
            errors.append(f"{tar.name}: unreadable, moved to bad/ ({exc})")
            continue
        good.append(tar)
        for name, payload in files.items():
            if name == tli.PROGRESS_MEMBER:
                changed |= merge_progress(state, json.loads(payload))
            elif name.startswith("relval/"):
                doc = json.load(gzip.open(io.BytesIO(payload)))
                steps[results / tli.check_step(doc)].append((tar, doc))
                counts["steps"] += 1
            else:
                try:
                    counts[tli.receive(payload, results).status] += 1
                except OSError as exc:
                    failed_tars.add(tar)
                    errors.append(f"{name}: {exc}")
    if changed or not (received / PROGRESS_STATE).exists():
        save_progress(received, state)
    by_path = {results / tli.progress_target(key): entry for key, entry in state.items()}
    also = {results / tli.progress_target(key) for key in changed}
    for path in sorted(set(steps) | {p for p in also if p.exists()}):
        docs = [doc for _, doc in steps.get(path, [])]
        try:
            counts["builds" if merge_build(path, docs, by_path.get(path)) else "builds_unchanged"] += 1
        except (MergeError, OSError, EOFError, zlib.error, ValueError, KeyError, TypeError, AttributeError) as exc:
            failed_tars |= {tar for tar, _ in steps.get(path, [])}
            errors.append(f"{path.relative_to(results)}: {type(exc).__name__}: {exc}")
    return dict(counts), [t for t in good if t not in failed_tars], errors


def merge_pending(received: Path, results: Path, log=print) -> int:
    tars = sorted(received.glob("*.tar"))
    failures = 0
    for i in range(0, len(tars), TARS_AT_ONCE):
        counts, done, errors = merge_tars(tars[i:i + TARS_AT_ONCE], received, results, log)
        merged = received / "merged"
        merged.mkdir(exist_ok=True)
        for tar in done:
            os.utime(tar)                           # its age from now on = when it was merged
            os.replace(tar, merged / tar.name)
        for error in errors:
            log(f"  ERROR {error}")
        failures += len(errors)
        log(f"  {len(done)} batches merged: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if not tars:
        log("  no batches waiting")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--received", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.received.is_dir():
        print(f"  {args.received} does not exist")
        return 1
    return merge_pending(args.received, args.results)


if __name__ == "__main__":
    sys.exit(main())
