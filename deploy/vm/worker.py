"""The VM stack's worker: the scheduled imports that cron + scripts/daily_refresh.sh
do on the developer PC, as one long-running container.

  every 30 min      testlogs   import new scan files from /data/results
                               (ingest_test_logs.py; skipped when no file changed)
  00:30 and 12:30   cmsrepo    git update of cms-sw.github.io (data/ only) -> build_graph.py
                    exceptions cms-bot's es-cmssw-exceptions.py (through the cmssdt proxy,
                               no password) -> build_graph.py --es-exceptions-json
                    docs       git update of cmssdt-wiki -> ingest_shifter_docs.py --reconcile

Same rules as daily_refresh.sh: cmsrepo is imported WITHOUT --reconcile (keep >= 90 days;
cms-sw.github.io drops IBs after ~2 weeks), a failed step is recorded and the next one still
runs, an empty or broken exceptions export never reaches the graph. One job at a time,
never two. A refresh slot missed while the container was down (reboot, update) runs as
soon as it starts again.

Not here (they need a Kerberos login to cmssdt9-02; waiting for a service-account
decision): copying test-log zips and the GitHub issues backup. Test logs arrive in
/data/results instead (later: POST /api/test-logs/ingest).

Everything it keeps is under /data (DATA_DIR on the host):
  /data/sources/{cmsrepo,shifter-wiki,cms-bot,exceptions}   checkouts + newest export
  /data/worker/state.json     last run and result of every job (read by people/status page)
  /data/worker/logs/<date>.log
Environment: WORKER_STEPS ("cmsrepo exceptions docs"), WORKER_REFRESH_AT ("00:30,12:30"),
WORKER_TESTLOGS_EVERY_MIN (30), TZ (Europe/Zurich). `worker.py --once <job>` runs one job now.
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
CLANG = DATA / "clang-logs"
STATE_DIR = DATA / "worker"
STATE = STATE_DIR / "state.json"

CMSREPO_URL = "https://github.com/cms-sw/cms-sw.github.io.git"
WIKI_URL = "https://github.com/cms-sw/cmssdt-wiki.git"
CMSBOT_URL = "https://github.com/cms-sw/cms-bot.git"

REFRESH_STEPS = os.environ.get("WORKER_STEPS", "cmsrepo exceptions docs").split()
REFRESH_AT = [tuple(map(int, t.split(":"))) for t in os.environ.get("WORKER_REFRESH_AT", "00:30,12:30").split(",") if t]
TESTLOGS_EVERY = dt.timedelta(minutes=int(os.environ.get("WORKER_TESTLOGS_EVERY_MIN", "30")))
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
    newest = 0.0
    for kind in ("unit", "addon", "relval"):
        for dirpath, _, files in os.walk(RESULTS / kind):
            for name in files:
                if name.endswith(".json.gz"):
                    newest = max(newest, os.stat(os.path.join(dirpath, name)).st_mtime)
    return newest


def job_testlogs(state: dict) -> str:
    newest = newest_result_mtime()
    if not newest:
        return "no results files yet"
    if newest <= state.get("testlogs", {}).get("imported_up_to", 0):
        return "nothing new"
    run([PY, "ingestion/ingest_test_logs.py", "--results", RESULTS], 3 * 3600)
    state.setdefault("testlogs", {})["imported_up_to"] = newest
    return f"imported (newest file {dt.datetime.fromtimestamp(newest):%F %H:%M})"


REFRESH_JOBS = {"cmsrepo": job_cmsrepo, "exceptions": job_exceptions, "docs": job_docs}


# ---------------------------------------------------------------- schedule

def last_refresh_slot(now: dt.datetime) -> dt.datetime:
    """The most recent REFRESH_AT time at or before `now`."""
    slots = [now.replace(hour=h, minute=m, second=0, microsecond=0) - dt.timedelta(days=d)
             for d in (0, 1) for h, m in REFRESH_AT]
    return max(s for s in slots if s <= now)


def due_jobs(state: dict, now: dt.datetime) -> list[str]:
    due = []
    slot = last_refresh_slot(now).isoformat()
    for name in REFRESH_STEPS:
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
    log(f"worker started: refresh {REFRESH_STEPS} at {REFRESH_AT}, test logs every {TESTLOGS_EVERY}")
    while True:
        state = load_state()
        for name in due_jobs(state, dt.datetime.now()):
            run_job(name, state)
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
