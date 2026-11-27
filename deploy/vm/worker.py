"""The VM stack's worker: the scheduled imports that cron + scripts/daily_refresh.sh
do on the developer PC, as one long-running container.

  every 5 min       testlogs   merge the batches in /data/received into /data/results
                               (merge_test_log_batches.py), import what changed
                               (ingest_test_logs.py), delete batches that are imported
  every hour at :30 cmsrepo    git update of cms-sw.github.io (data/ only) -> build_graph.py
                               (Jenkins summary-of-merged-prs pushes there about hourly;
                               ~25 changed files and 20-70 s per run, measured 2026-11-19)
  00:30 and 12:30   exceptions cms-bot's es-cmssw-exceptions.py (through the cmssdt proxy,
                               no password) -> build_graph.py --es-exceptions-json
                    docs       git update of cmssdt-wiki -> ingest_shifter_docs.py --reconcile

Same rules as daily_refresh.sh: cmsrepo is imported WITHOUT --reconcile (keep >= 90 days;
cms-sw.github.io drops IBs after ~2 weeks), a failed step is recorded and the next one still
runs, an empty or broken exceptions export never reaches the graph. One job at a time,
never two. A refresh slot missed while the container was down (reboot, update) runs as
soon as it starts again.

Not here (they need a Kerberos login to cmssdt9-02; waiting for a service-account
decision): copying test-log zips and the GitHub issues backup. Test logs arrive in
/data/received instead, sent from cmssdt9-02 to POST /api/test-logs/batch
(deploy/cmssdt/send_test_log_scans.py).

Everything it keeps is under /data (DATA_DIR on the host):
  /data/sources/{cmsrepo,shifter-wiki,cms-bot,exceptions}   checkouts + newest export
  /data/worker/state.json     last run and result of every job (read by people/status page)
  /data/worker/logs/<date>.log
Environment: WORKER_STEPS ("cmsrepo exceptions docs"), WORKER_REFRESH_AT ("00:30,12:30"),
WORKER_CMSREPO_AT (cmsrepo only; default every hour at :30), WORKER_TESTLOGS_EVERY_MIN (5),
RECEIVED_KEEP_DAYS (7: imported batches are kept that long; 0 = deleted at once),
TZ (Europe/Zurich). `worker.py --once <job>` runs one job now.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

APP = Path(os.environ.get("WORKER_APP_DIR", "/app"))
DATA = Path(os.environ.get("WORKER_DATA_DIR", "/data"))
SOURCES = DATA / "sources"
RESULTS = DATA / "results"
RECEIVED = DATA / "received"
CLANG = DATA / "clang-logs"
STATE_DIR = DATA / "worker"
STATE = STATE_DIR / "state.json"

CMSREPO_URL = "https://github.com/cms-sw/cms-sw.github.io.git"
WIKI_URL = "https://github.com/cms-sw/cmssdt-wiki.git"
CMSBOT_URL = "https://github.com/cms-sw/cms-bot.git"

REFRESH_STEPS = os.environ.get("WORKER_STEPS", "cmsrepo exceptions docs").split()


def _times(text: str) -> list[tuple[int, int]]:
    return [tuple(map(int, t.split(":"))) for t in text.split(",") if t]


REFRESH_AT = _times(os.environ.get("WORKER_REFRESH_AT") or "00:30,12:30")
CMSREPO_AT = _times(os.environ.get("WORKER_CMSREPO_AT") or ",".join(f"{h:02d}:30" for h in range(24)))
TESTLOGS_EVERY = dt.timedelta(minutes=int(os.environ.get("WORKER_TESTLOGS_EVERY_MIN", "5")))
REJECTED_KEEP_DAYS = 30
RECEIVED_KEEP_DAYS = float(os.environ.get("RECEIVED_KEEP_DAYS") or 7)
PY = sys.executable


class StepFailed(Exception):
    pass


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    try:
        (STATE_DIR / "logs").mkdir(parents=True, exist_ok=True)
        with open(STATE_DIR / "logs" / f"{dt.date.today()}.log", "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    tmp.replace(STATE)


def run(cmd: list[str], timeout_s: int, cwd: Path = APP, stdout=None) -> None:
    """Runs one command; its output goes to the worker log. Raises StepFailed."""
    log("  $ " + " ".join(map(str, cmd)))
    env = dict(os.environ, PYTHONPATH="ingestion")
    try:
        proc = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, timeout=timeout_s, text=True,
                              stdout=stdout or subprocess.PIPE, stderr=subprocess.STDOUT if stdout is None else subprocess.PIPE)
    except subprocess.TimeoutExpired:
        raise StepFailed(f"timed out after {timeout_s // 60} min: {cmd[0]} {cmd[1] if len(cmd) > 1 else ''}")
    out = proc.stdout if stdout is None else proc.stderr
    for line in (out or "").splitlines()[-15:]:
        log("    " + line)
    if proc.returncode:
        raise StepFailed(f"exit {proc.returncode}: {' '.join(map(str, cmd[:3]))}")


def git_update(path: Path, url: str, sparse: str | None = None, shallow: bool = False) -> str:
    """Clone once, then fetch + hard reset to origin/master. Returns the commit date."""
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--branch", "master"]
        cmd += ["--depth", "1"] if shallow else []
        cmd += ["--filter=blob:none", "--sparse"] if sparse else []
        run(cmd + [url, path], 3600, cwd=path.parent)
        if sparse:
            run(["git", "-C", path, "sparse-checkout", "set", sparse], 1800)
    else:
        run(["git", "-C", path, "fetch", "--prune"] + (["--depth", "1"] if shallow else []) + ["origin", "master"], 1800)
        run(["git", "-C", path, "reset", "--hard", "origin/master"], 600)
        if sparse:
            run(["git", "-C", path, "sparse-checkout", "reapply"], 600)
    return subprocess.run(["git", "-C", str(path), "log", "-1", "--format=%cd", "--date=format:%F %H:%M"],
                          text=True, capture_output=True).stdout.strip()


# ---------------------------------------------------------------- jobs

def job_cmsrepo() -> str:
    when = git_update(SOURCES / "cmsrepo", CMSREPO_URL, sparse="data")
    cmd = [PY, "ingestion/build_graph.py", "--cmsrepo", SOURCES / "cmsrepo"]
    if CLANG.is_dir() and any(CLANG.iterdir()):
        cmd += ["--clang-root", CLANG]
    run(cmd, 3 * 3600)          # no --reconcile: see the module docstring
    return f"cmsrepo at {when}"


def export_is_usable(path: Path) -> bool:
    """Only a real export is imported: an empty or broken file must never reach the graph."""
    try:
        return bool(json.loads(path.read_text()).get("hits", {}).get("hits"))
    except (OSError, ValueError, AttributeError):
        return False


def job_exceptions() -> str:
    cmsbot = SOURCES / "cms-bot"
    git_update(cmsbot, CMSBOT_URL, shallow=True)
    out_dir = SOURCES / "exceptions"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"exceptions-{dt.date.today()}.json"
    part = final.with_suffix(".json.part")
    with open(part, "w") as fh:
        run([PY, "es-cmssw-exceptions.py"], 2 * 3600, cwd=cmsbot, stdout=fh)
    if not export_is_usable(part):
        raise StepFailed(f"export empty or not valid JSON, kept as {part}; nothing imported")
    part.replace(final)
    if not (SOURCES / "cmsrepo").is_dir():
        raise StepFailed("no cmsrepo checkout yet (exceptions attach to its RelVals)")
    run([PY, "ingestion/build_graph.py", "--cmsrepo", SOURCES / "cmsrepo", "--es-exceptions-json", final], 2 * 3600)
    for old in out_dir.glob("exceptions-*.json"):
        if old != final:
            old.unlink()
    return f"{final.name}, {final.stat().st_size // 1024} KB"


def job_docs() -> str:
    when = git_update(SOURCES / "shifter-wiki", WIKI_URL)
    cmd = [PY, "ingestion/ingest_shifter_docs.py", "--wiki-root", SOURCES / "shifter-wiki", "--reconcile"]
    if (SOURCES / "cmsrepo").is_dir():
        cmd += ["--cmsrepo", SOURCES / "cmsrepo"]
    run(cmd, 3600)
    return f"wiki at {when}"


def newest_result_mtime() -> float:
    """Newest results file. An unreadable folder raises (os.walk would skip it silently
    and the job would say "no results files yet" -- found with a wrong SELinux label)."""
    def unreadable(err: OSError) -> None:
        raise StepFailed(f"cannot read {err.filename}: {err.strerror}")

    if not RESULTS.exists():
        return 0.0                      # a new VM: nothing received yet
    if not os.access(RESULTS, os.R_OK | os.X_OK):
        raise StepFailed(f"cannot read {RESULTS} (permissions / SELinux label?)")
    newest = 0.0
    for kind in ("unit", "addon", "relval"):
        if not (RESULTS / kind).exists():
            continue
        for dirpath, _, files in os.walk(RESULTS / kind, onerror=unreadable):
            for name in files:
                if name.endswith(".json.gz"):
                    newest = max(newest, os.stat(os.path.join(dirpath, name)).st_mtime)
    return newest


def merge_received() -> StepFailed | None:
    """received/*.tar -> results/. A failure is reported at the end: what did merge is imported."""
    if not RECEIVED.exists():
        return None
    if not os.access(RECEIVED, os.R_OK | os.W_OK | os.X_OK):
        raise StepFailed(f"cannot use {RECEIVED} (permissions / SELinux label?)")
    if not any(RECEIVED.glob("*.tar")):
        return None
    try:
        run([PY, "ingestion/merge_test_log_batches.py", "--received", RECEIVED, "--results", RESULTS], 3600)
    except StepFailed as exc:
        return exc
    return None


def refused_files() -> int:
    """Files the endpoint refused (received/rejected/): old ones are deleted, the rest counted."""
    left = 0
    for path in (RECEIVED / "rejected").glob("*"):
        if path.stat().st_mtime < time.time() - REJECTED_KEEP_DAYS * 86400:
            path.unlink()
        elif not path.name.endswith(".reason.txt"):
            left += 1
    return left


def delete_imported_batches(merged_before: float) -> int:
    """Batches merged before the import that just succeeded started, older than RECEIVED_KEEP_DAYS."""
    limit = min(merged_before, time.time() - RECEIVED_KEEP_DAYS * 86400)
    old = [p for p in (RECEIVED / "merged").glob("*.tar") if p.stat().st_mtime < limit]
    for path in old:
        path.unlink()
    return len(old)


def job_testlogs(state: dict) -> str:
    failed = merge_received()
    state.setdefault("testlogs", {})["refused_files"] = refused_files()
    merged_before = time.time()
    newest = newest_result_mtime()
    if not newest:
        result = "no results files yet"
    elif newest <= state.get("testlogs", {}).get("imported_up_to", 0):
        result = "nothing new"
    else:
        run([PY, "ingestion/ingest_test_logs.py", "--results", RESULTS], 3 * 3600)
        state.setdefault("testlogs", {})["imported_up_to"] = newest
        result = f"imported (newest file {dt.datetime.fromtimestamp(newest):%F %H:%M})"
    if newest and not failed:
        deleted = delete_imported_batches(merged_before)
        result += f", {deleted} imported batches deleted" if deleted else ""
    if state["testlogs"]["refused_files"]:
        result += f"; {state['testlogs']['refused_files']} refused files in received/rejected/"
    if failed:
        raise StepFailed(f"merge: {failed}; {result}")
    return result


REFRESH_JOBS = {"cmsrepo": job_cmsrepo, "exceptions": job_exceptions, "docs": job_docs}


# ---------------------------------------------------------------- schedule

def last_refresh_slot(now: dt.datetime, at: list[tuple[int, int]] | None = None) -> dt.datetime:
    """The most recent of the `at` times (default REFRESH_AT) at or before `now`."""
    slots = [now.replace(hour=h, minute=m, second=0, microsecond=0) - dt.timedelta(days=d)
             for d in (0, 1) for h, m in (at or REFRESH_AT)]
    return max(s for s in slots if s <= now)


def due_jobs(state: dict, now: dt.datetime) -> list[str]:
    due = []
    for name in REFRESH_STEPS:
        slot = last_refresh_slot(now, CMSREPO_AT if name == "cmsrepo" else REFRESH_AT).isoformat()
        if state.get(name, {}).get("tried_at", "") < slot:   # missed or not yet run for this slot
            due.append(name)
    tried = state.get("testlogs", {}).get("tried_at")
    if not tried or now - dt.datetime.fromisoformat(tried) >= TESTLOGS_EVERY:
        due.append("testlogs")
    return due


def run_job(name: str, state: dict) -> bool:
    entry = state.setdefault(name, {})
    entry["tried_at"] = dt.datetime.now().isoformat(timespec="seconds")
    started = time.time()
    log(f"-- {name}")
    try:
        result = job_testlogs(state) if name == "testlogs" else REFRESH_JOBS[name]()
        ok = True
    except StepFailed as exc:
        result, ok = f"FAILED ({exc})", False
    except Exception as exc:   # a bug in one job must not stop the worker
        result, ok = f"FAILED ({type(exc).__name__}: {exc})", False
    entry.update(result=result, ok=ok, seconds=round(time.time() - started))
    if ok:
        entry["ok_at"] = entry["tried_at"]
    log(f"== {name}: {result} ({entry['seconds']} s)")
    save_state(state)
    return ok


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--once":
        return 0 if run_job(argv[2], load_state()) else 1
    log(f"worker started: cmsrepo {len(CMSREPO_AT)} times a day, refresh {REFRESH_STEPS} at {REFRESH_AT}, "
        f"test logs every {TESTLOGS_EVERY}")
    while True:
        state = load_state()
        for name in due_jobs(state, dt.datetime.now()):
            run_job(name, state)
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
