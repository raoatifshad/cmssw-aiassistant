"""Copy -> scan -> keep results -> delete, for every IB test-log zip on cmssdt9-02.

cmssdt9-02 keeps unitTestLogs.zip / addOnTests.zip / pyRelValMatrixLogs.zip
for about a week, then deletes them. This script makes sure every one of
them is scanned (ib_test_log_scanner.py) before that happens, without
piling the raw logs up locally:

  1. list every test-log zip on the server (one ssh call: path, size, mtime)
  2. skip the ones the ledger already has as done with the same size+mtime
  3. copy the rest, oldest first, one at a time (rsync), while up to
     --workers scans run in parallel on the copies already here
  4. write each zip's results to results/<kind>/<arch>/<tag>/<variant>/*.json.gz
     -- every test/workflow with its status, and every problem found
  5. record it as done in results/ledger.jsonl, then delete the local copy

Nothing is skipped silently:
  * a zip that fails to copy or scan is recorded as failed, its local copy is
    kept, and the run ends with a list of them; the next run retries
  * a zip that is still being written (no .checked marker yet for unit/addOn,
    or modified less than --min-age-minutes ago) is reported as "not ready"
  * the end-of-run report compares against the local cms-sw.github.io data
    (cmsrepo), which knows every IB the dashboard shows, and lists every
    IB/architecture whose logs are no longer on the server and were never
    scanned -- those are lost, and the report says so

results/ is the only copy of what was extracted until it is imported into
Neo4j -- don't delete it. Files this script downloaded into its own work/
folder are deleted once done; nothing on the server is ever touched.
server-copy/ is only cleaned with --delete-local.

Copying with your own rsync instead (into server-copy/): run with
--no-server --delete-local. Each zip (and its .checked marker, and
addOnTests.log) is deleted from server-copy/ once its results are saved, and
results/rsync-exclude.txt lists every finished zip -- pass it to rsync with
--exclude-from so the same zips are not downloaded again.

Run it on a machine with ssh access to cmssdt9-02. Open one shared ssh
connection first, so the script never waits for a password:

    ssh -o ControlMaster=yes -o "ControlPath=~/.ssh/cm-ibtestlogs-%r@%h:%p" \
        -o ControlPersist=yes -fN root@cmssdt9-02.cern.ch
    venv/bin/python ingestion/fetch_ib_test_logs.py --dry-run      # list only
    venv/bin/python ingestion/fetch_ib_test_logs.py --limit 2      # try 2 zips
    venv/bin/python ingestion/fetch_ib_test_logs.py                # everything

The few huge DBG zips (>5 GB) go last, one at a time. Zips that exist only in
server-copy/ (the server already deleted them) are processed from disk.
"""
import argparse
import concurrent.futures
import fcntl
import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures.process import BrokenProcessPool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ib_test_log_scanner as scanner  # noqa: E402

REMOTE = "root@cmssdt9-02.cern.ch"
REMOTE_ROOT = "/data/sdt/buildlogs"
DEFAULT_BASE = "/home/rao/CHAT-BOT-AI/IB-TEST-LOGS"
ZIP_KINDS = {"pyRelValMatrixLogs.zip": "relval", "unitTestLogs.zip": "unit", "addOnTests.zip": "addon"}
SSH_OPTS = ["-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/cm-ibtestlogs-%r@%h:%p",
            "-o", "ControlPersist=10m", "-o", "ServerAliveInterval=30"]
EVIDENCE_LINES_QUIET = 8  # evidence lines kept for message_logger / generic records
MARKER_WAIT_HOURS = 24  # unit/addOn zip without a .checked marker: scanned anyway after this
BIG_ZIP_BYTES = 5e9  # DBG RelVal zips are ~33 GB; at most one of these is copied/scanned at a time

# Same scope as the agreed rsync: every IB folder (slc* included), its gpu/<type>/
# and other/<name>/ sub-folders, but not filtered/ and not the new*/ build-log trees.
LISTING_CMD = (
    f"cd {REMOTE_ROOT} && find $(ls -d */www/*/*/CMSSW_*/ | grep -v '^filtered/') -maxdepth 5 "
    r"\( -name new -o -name 'new_*' \) -prune -o "
    r"\( -name pyRelValMatrixLogs.zip -o -name unitTestLogs.zip -o -name addOnTests.zip "
    r"-o -name '*.zip.checked' -o -name addOnTests.log \) -printf '%s %T@ %p\n'"
)
PATH_RE = re.compile(r"^(?P<arch>[^/]+)/www/(?P<day>[^/]+)/(?P<queue>[^/]+)/(?P<tag>CMSSW_[^/]+)/+(?P<rest>.+)$")


# ---- inventory -------------------------------------------------------------

def parse_listing(text: str) -> tuple[list[dict], set[str], dict[str, dict]]:
    """`size mtime path` lines -> (zips, paths of .checked markers, addOnTests.log entries)."""
    zips, checked, addon_logs = [], set(), {}
    for line in text.splitlines():
        parts = line.split(" ", 2)
        if len(parts) != 3:
            continue
        size, mtime, path = int(parts[0]), float(parts[1]), parts[2].replace("//", "/")
        name = path.rsplit("/", 1)[-1]
        if name.endswith(".zip.checked"):
            checked.add(path[: -len(".checked")])
        elif name == "addOnTests.log":
            addon_logs[path] = dict(size=size, mtime=mtime)
        elif name in ZIP_KINDS:
            info = describe_path(path)
            if info:
                zips.append(dict(info, path=path, size=size, mtime=mtime))
    return zips, checked, addon_logs


def local_only_zips(copy_root: str, on_server: set[str]) -> tuple[list[dict], set[str]]:
    """Zips in server-copy/ that the server has already deleted: they are the
    only copy left, so they are processed from disk like any other zip.
    Returns (zips, paths of their .checked markers)."""
    zips, checked = [], set()
    for dirpath, _, files in os.walk(copy_root):
        for name in files:
            path = os.path.relpath(os.path.join(dirpath, name), copy_root)
            if name.endswith(".zip.checked"):
                checked.add(path[: -len(".checked")])
            elif name in ZIP_KINDS and path not in on_server:
                info = describe_path(path)
                if info:
                    st = os.stat(os.path.join(dirpath, name))
                    zips.append(dict(info, path=path, size=st.st_size, mtime=st.st_mtime, local_only=True))
    return zips, checked


def describe_path(path: str) -> dict | None:
    m = PATH_RE.match(path)
    if not m:
        return None
    rest = m.group("rest")
    name = rest.rsplit("/", 1)[-1]
    variant = "/".join(rest.split("/")[:2]) if rest.startswith(("gpu/", "other/")) else ""
    return dict(kind=ZIP_KINDS.get(name), arch=m.group("arch"), tag=m.group("tag"), variant=variant,
                queue=m.group("queue"))


def ledger_key(item: dict) -> str:
    return f"{item['path']}|{item['size']}|{int(item['mtime'])}"


def is_ready(item: dict, checked: set[str], now: float, min_age_minutes: int) -> tuple[bool, str]:
    """unit/addOn zips are rewritten once by cms-bot, which then drops a
    .checked marker; RelVal zips have no marker, so they only need to have
    stopped changing."""
    if now - item["mtime"] < min_age_minutes * 60:
        return False, f"modified less than {min_age_minutes} min ago"
    if item["kind"] in ("unit", "addon") and item["path"] not in checked:
        # cms-bot sometimes never marks a zip (e.g. an empty 176-byte GPU unit zip);
        # waiting forever would lose it silently, so after a day it is scanned anyway
        if now - item["mtime"] < MARKER_WAIT_HOURS * 3600:
            return False, "no .checked marker yet (cms-bot still processing)"
    return True, ""


def is_big(item: dict) -> bool:
    return item["size"] >= BIG_ZIP_BYTES


def order_todo(items: list[dict]) -> list[dict]:
    """Oldest first (the server deletes after ~1 week), except the few huge
    DBG zips, which go after all the normal ones and are done one at a time."""
    return sorted(items, key=lambda z: (is_big(z), z["mtime"]))


def result_path(base: str, item: dict) -> str:
    variant = item["variant"].replace("/", "_") or "_"
    name = item["path"].rsplit("/", 1)[-1].replace(".zip", ".json.gz")
    return os.path.join(base, "results", item["kind"], item["arch"], item["tag"], variant, name)


# ---- ledger ----------------------------------------------------------------

def read_ledger(path: str) -> dict[str, dict]:
    """Last entry per key wins (a failed zip that later succeeds is done)."""
    entries = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    entry = json.loads(line)
                    entries[entry["key"]] = entry
    return entries


def append_ledger(path: str, entry: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ---- expected IBs (from the dashboard data) --------------------------------

def expected_zips(cmsrepo_data_dir: str) -> set[str]:
    """Every test-log zip path the dashboard data says should exist. Each
    test result there carries a `file` path next to its zip -- under
    relvals/utests/addons for the main IB, under gpu_data/... for GPU runs --
    so the whole record is searched for such paths rather than named
    sections, and a section added later can't be missed."""
    expected = set()
    prefix = REMOTE_ROOT + "/"
    suffixes = ((re.compile(r"/pyRelValMatrixLogs/run/[^/]+$"), "pyRelValMatrixLogs.zip"),
                (re.compile(r"/unitTests-summary\.log$"), "unitTestLogs.zip"),
                (re.compile(r"/addOnTests\.log$"), "addOnTests.zip"))

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "file" and isinstance(value, str) and value.startswith(prefix):
                    rel = value[len(prefix):]
                    for pattern, zip_name in suffixes:
                        if pattern.search(rel):
                            path = pattern.sub("", rel) + "/" + zip_name
                            if describe_path(path):
                                expected.add(path)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for json_path in glob.glob(os.path.join(cmsrepo_data_dir, "CMSSW_*.json")):
        try:
            with open(json_path) as fh:
                walk(json.load(fh).get("comparisons", []))
        except (OSError, ValueError):
            continue
    return expected


# ---- copy + scan -----------------------------------------------------------

def rsync_from_server(remote_path: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    ssh = "ssh " + " ".join(SSH_OPTS)
    subprocess.run(["rsync", "-a", "--partial", "-e", ssh, f"{REMOTE}:{REMOTE_ROOT}/{remote_path}", local_path],
                   check=True)


def _trim(problem: dict) -> dict:
    if problem["kind"] not in scanner.SERIOUS_KINDS:
        problem = dict(problem, block=problem["block"][:EVIDENCE_LINES_QUIET])
    return problem


def scan_one(item: dict, local_zip: str, addon_log: str | None, out_path: str) -> dict:
    """Runs in a worker process. Writes the result file atomically and returns a summary."""
    started = time.time()
    if item["kind"] == "relval":
        result = scanner.scan_relval_zip(local_zip)
    elif item["kind"] == "unit":
        result = scanner.scan_unit_test_zip(local_zip)
    else:
        with open(addon_log, encoding="utf-8", errors="replace") as fh:
            result = scanner.scan_addon_zip(local_zip, fh.read())
    result["problems"] = [_trim(p) for p in result["problems"]]
    result["source"] = dict(item, remote=f"{REMOTE}:{REMOTE_ROOT}/{item['path']}",
                            scanned_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            scanner_version=scanner.SCANNER_VERSION)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(result, fh, default=str)
    os.replace(tmp, out_path)
    units = result.get("workflows") or result.get("tests") or []
    return dict(units=len(units), problems=len(result["problems"]), seconds=round(time.time() - started))


# ---- main ------------------------------------------------------------------

def human(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default=DEFAULT_BASE, help="local folder (results/, work/, server-copy/)")
    parser.add_argument("--cmsrepo-data", default=os.path.join(os.path.dirname(__file__), "..", "cmsrepo", "data"))
    parser.add_argument("--kinds", default="relval,unit,addon")
    parser.add_argument("--workers", type=int, default=4, help="parallel scans")
    parser.add_argument("--disk-budget-gb", type=float, default=80,
                        help="max size of copied-but-not-yet-scanned zips on local disk")
    parser.add_argument("--min-age-minutes", type=int, default=60)
    parser.add_argument("--limit", type=int, help="process at most this many zips (for a trial run)")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be done")
    parser.add_argument("--no-server", action="store_true",
                        help="don't ssh to the server; process only what is in server-copy/ (copied by your own rsync)")
    parser.add_argument("--delete-local", action="store_true",
                        help="delete a zip from server-copy/ once its results are saved")
    args = parser.parse_args()
    kinds = set(args.kinds.split(","))

    base = os.path.abspath(args.base)
    os.makedirs(os.path.join(base, "results"), exist_ok=True)
    lock = open(os.path.join(base, "results", ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another run is already in progress -- stopping")
        return 1
    ledger_path = os.path.join(base, "results", "ledger.jsonl")
    ledger = read_ledger(ledger_path)
    done_keys = {k for k, e in ledger.items() if e["status"] == "done"}

    write_rsync_exclude(base, ledger)
    if args.no_server:
        zips, checked = [], set()
    else:
        print(f"listing {REMOTE}:{REMOTE_ROOT} ...", flush=True)
        listing = subprocess.run(["ssh", *SSH_OPTS, REMOTE, LISTING_CMD], check=True, capture_output=True,
                                 text=True).stdout
        zips, checked, _ = parse_listing(listing)
    local_zips, local_checked = local_only_zips(os.path.join(base, "server-copy"), {z["path"] for z in zips})
    zips += local_zips
    checked |= local_checked
    zips = [z for z in zips if z["kind"] in kinds]
    if local_zips:
        print(f"{len(local_zips)} zips are processed from server-copy/"
              + ("" if args.no_server else " (already deleted on the server)"))
    now = time.time()
    todo, not_ready, already = [], [], 0
    for item in sorted(zips, key=lambda z: z["mtime"]):
        if ledger_key(item) in done_keys:
            already += 1
            if args.delete_local and item.get("local_only") and not args.dry_run:
                # copied again by an rsync whose exclude list predates this zip being done
                remove_local_copy(os.path.join(base, "server-copy", item["path"]), item)
            continue
        ready, why = is_ready(item, checked, now, args.min_age_minutes)
        (todo if ready else not_ready).append(item if ready else dict(item, why=why))
    todo = order_todo(todo)
    if args.limit:
        todo = todo[: args.limit]

    print(f"found: {len(zips)} zips ({human(sum(z['size'] for z in zips))}); already done: {already}; "
          f"to do now: {len(todo)} ({human(sum(z['size'] for z in todo))}); not ready yet: {len(not_ready)}")
    for kind in sorted(kinds):
        part = [z for z in todo if z["kind"] == kind]
        print(f"  {kind:7s} to do: {len(part):4d}  {human(sum(z['size'] for z in part))}")
    if args.dry_run:
        for item in todo[:20]:
            print(f"  would do: {human(item['size']):>8s}  {item['path']}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        report_missing(args, base, zips, read_ledger(ledger_path), not_ready)
        return 0

    work = os.path.join(base, "work")
    failures = []
    in_flight: dict = {}

    def settle(future):
        item, local_zip, downloaded, out_path = in_flight.pop(future)
        entry = dict(key=ledger_key(item), path=item["path"], kind=item["kind"], size=item["size"],
                     mtime=item["mtime"], finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        try:
            summary = future.result()
        except Exception as exc:  # noqa: BLE001 -- recorded and reported, never swallowed
            entry.update(status="failed", error="".join(traceback.format_exception_only(type(exc), exc))[-1000:],
                         local_copy=local_zip)
            append_ledger(ledger_path, entry)
            failures.append(entry)
            print(f"FAILED  {item['path']}: {entry['error'].strip()} (local copy kept: {local_zip})", flush=True)
            return
        entry.update(status="done", result=out_path, **summary)
        append_ledger(ledger_path, entry)
        if downloaded:
            os.remove(local_zip)
        elif args.delete_local:
            remove_local_copy(local_zip, item)
        print(f"done    {item['path']}  ({summary['units']} tests/workflows, {summary['problems']} problem "
              f"records, {summary['seconds']}s)", flush=True)

    def in_flight_bytes():
        return sum(v[0]["size"] for v in in_flight.values() if v[2])

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
    try:
        for n, item in enumerate(todo, 1):
            while in_flight and (len(in_flight) >= args.workers * 2
                                 or in_flight_bytes() + item["size"] > args.disk_budget_gb * 1e9
                                 or (is_big(item) and any(is_big(v[0]) for v in in_flight.values()))):
                finished, _ = concurrent.futures.wait(list(in_flight), return_when=concurrent.futures.FIRST_COMPLETED)
                for future in finished:
                    settle(future)
            local_copy = os.path.join(base, "server-copy", item["path"])
            downloaded = not (os.path.exists(local_copy) and os.path.getsize(local_copy) == item["size"])
            local_zip = os.path.join(work, item["path"]) if downloaded else local_copy
            addon_log = None
            try:
                if downloaded:
                    print(f"copy    [{n}/{len(todo)}] {human(item['size']):>8s}  {item['path']}", flush=True)
                    rsync_from_server(item["path"], local_zip)
                    if os.path.getsize(local_zip) != item["size"]:
                        raise RuntimeError("size changed while copying (still being written?)")
                if item["kind"] == "addon":
                    log_rel = item["path"].rsplit("/", 1)[0] + "/addOnTests.log"
                    addon_log = os.path.join(base, "server-copy", log_rel)
                    if not os.path.exists(addon_log):
                        addon_log = os.path.join(work, log_rel)
                        rsync_from_server(log_rel, addon_log)
            except Exception as exc:  # noqa: BLE001
                entry = dict(key=ledger_key(item), path=item["path"], kind=item["kind"], size=item["size"],
                             mtime=item["mtime"], status="failed", error=f"copy failed: {exc}"[-1000:],
                             finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                append_ledger(ledger_path, entry)
                failures.append(entry)
                print(f"FAILED  {item['path']}: {entry['error']}", flush=True)
                continue
            out_path = result_path(base, item)
            future = pool.submit(scan_one, item, local_zip, addon_log, out_path)
            in_flight[future] = (item, local_zip, downloaded, out_path)
        for future in concurrent.futures.as_completed(list(in_flight)):
            settle(future)
    except BrokenProcessPool:
        print("a scan worker died (out of memory?) -- the zips still in progress are marked failed, "
              "their local copies kept; rerun to retry", flush=True)
        for future in list(in_flight):
            item, local_zip, _, _ = in_flight.pop(future)
            entry = dict(key=ledger_key(item), path=item["path"], kind=item["kind"], size=item["size"],
                         mtime=item["mtime"], status="failed", error="scan worker died", local_copy=local_zip,
                         finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            append_ledger(ledger_path, entry)
            failures.append(entry)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    write_rsync_exclude(base, read_ledger(ledger_path))
    report_missing(args, base, zips, read_ledger(ledger_path), not_ready)
    if failures:
        print(f"\n{len(failures)} FAILED (will be retried next run):")
        for entry in failures:
            print(f"  {entry['path']}: {entry['error'].strip()[:200]}")
        return 2
    return 0


def finished_files(item_path: str, kind: str) -> list[str]:
    """The zip, its .checked marker, and (addOn) addOnTests.log -- what is
    deleted locally and kept out of the next rsync once a zip is done."""
    files = [item_path, item_path + ".checked"]
    if kind == "addon":
        files.append(item_path.rsplit("/", 1)[0] + "/addOnTests.log")
    return files


def remove_local_copy(local_zip: str, item: dict) -> None:
    copy_root = local_zip[: -len(item["path"])]
    for rel in finished_files(item["path"], item["kind"]):
        try:
            os.remove(os.path.join(copy_root, rel))
        except FileNotFoundError:
            pass


def write_rsync_exclude(base: str, ledger: dict) -> None:
    """results/rsync-exclude.txt: every finished zip, for rsync --exclude-from."""
    lines = sorted({"/" + rel for e in ledger.values() if e["status"] == "done"
                    for rel in finished_files(e["path"], e["kind"])})
    tmp = os.path.join(base, "results", "rsync-exclude.txt.tmp")
    with open(tmp, "w") as fh:
        fh.write("".join(line + "\n" for line in lines))
    os.replace(tmp, os.path.join(base, "results", "rsync-exclude.txt"))


def report_missing(args, base: str, zips: list[dict], ledger: dict, not_ready: list[dict]) -> None:
    on_server = {z["path"] for z in zips}
    done_paths = {e["path"] for e in ledger.values() if e["status"] == "done"}
    kinds = set(args.kinds.split(","))
    expected = {p for p in expected_zips(args.cmsrepo_data) if describe_path(p)["kind"] in kinds}
    lost = sorted(p for p in expected if p not in on_server and p not in done_paths)
    pending = sorted(p for p in expected if p in on_server and p not in done_paths)
    label = "MISSING" if args.no_server else "LOST"
    why = ("not in server-copy/ and never scanned -- copy them before the server deletes them (~1 week)"
           if args.no_server else "gone from the server, no local copy, never scanned")
    print(f"\ndashboard data knows {len(expected)} test-log zips: {len(expected & done_paths)} scanned, "
          f"{len(pending)} available (on the server or in server-copy/) but not scanned yet, {len(lost)} {label} "
          f"({why})")
    for path in lost[:50]:
        print(f"  {label}  {path}")
    if len(lost) > 50:
        print(f"  ... and {len(lost) - 50} more")
    if lost:
        with open(os.path.join(base, "results", "missing.txt" if args.no_server else "lost.txt"), "w") as fh:
            fh.write("\n".join(lost) + "\n")
        print(f"  full list: {os.path.join(base, 'results', 'missing.txt' if args.no_server else 'lost.txt')}")
    for item in not_ready:
        print(f"  not ready: {item['path']} -- {item['why']}")


if __name__ == "__main__":
    sys.exit(main())
