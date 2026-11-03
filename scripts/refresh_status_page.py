#!/usr/bin/env python3
"""Local status page for the daily refresh: http://localhost:8090

Read-only. Every page load re-reads
  - the newest refresh log (/home/rao/CHAT-BOT-AI/daily-refresh-logs/<date>.log): steps of the
    current/last run, and the test-log fetch plan ("found: ... to do now: ...")
  - the test-log ledger (IB-TEST-LOGS/results/ledger.jsonl): exact size + time of every zip
    copied and scanned, so "copied / left" is in real bytes
  - Neo4j (MATCH/RETURN only): result files imported vs. scanned, TestLogRun per IB date
  - the tester history (daily-test-reports/history.txt)
and auto-reloads every 30 s.

  venv/bin/python scripts/refresh_status_page.py [--port 8090]
"""
import argparse
import datetime as dt
import glob
import html
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = "/home/rao/CHAT-BOT-AI/daily-refresh-logs"
RESULTS = "/home/rao/CHAT-BOT-AI/IB-TEST-LOGS/results"
REPORTS = "/home/rao/CHAT-BOT-AI/daily-test-reports"

STEP_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (--|==) ([\w-]+): ?(.*)$")
START_RE = re.compile(r"^\[(.{19})\] ===== daily refresh")
DONE_RE = re.compile(r"^\[(.{19})\] ===== done, (\d+) problem")
FOUND_RE = re.compile(r"^found: (\d+) zips .*?to do now: (\d+) \(([\d.]+) GB\)")
KIND_RE = re.compile(r"^  (addon|relval|unit)\s+to do:\s+(\d+)\s+([\d.]+) GB")
COPY_RE = re.compile(r"^copy\s+\[(\d+)/(\d+)\]\s+([\d.]+) GB\s+(\S+)")
IMPORT_RE = re.compile(r"^  (\d+)/(\d+) files \((\d+) imported, (\d+) unchanged\)")
TAG_DATE_RE = re.compile(r"_X_(\d{4}-\d\d-\d\d)-(\d{4})")


def env(name, default=""):
    if os.environ.get(name):
        return os.environ[name]
    try:
        for line in open(os.path.join(REPO, ".env")):
            k, _, v = line.strip().partition("=")
            if k == name:
                return v.strip().strip('"').strip("'")
    except OSError:
        pass
    return default


def parse_time(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def last_run():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "20*.log")))
    if not logs:
        return None
    lines = open(logs[-1], errors="replace").read().splitlines()
    start = max((i for i, l in enumerate(lines) if START_RE.match(l)), default=0)
    run = dict(log=logs[-1], started=None, finished=None, problems=None, steps=[], found=None,
               kinds={}, copy=None, imp=None, failed=[])
    m = START_RE.match(lines[start]) if lines else None
    if m:
        run["started"] = parse_time(m.group(1))
    steps = {}
    for l in lines[start:]:
        if m := STEP_RE.match(l):
            t, kind, name, rest = m.groups()
            if kind == "--":   # a new sub-step means the previous open one finished
                for s in steps.values():
                    if s["end"] is None:
                        s["end"], s["result"] = parse_time(t), "done"
                steps.setdefault(name, dict(name=name, start=parse_time(t), end=None, result=None))
            else:   # "== step: result" closes the sub-steps of that step
                for s in steps.values():
                    if s["name"] == name or (s["name"].startswith(name + "-") and s["end"] in (None, parse_time(t)) or
                                             s["name"].startswith(name + "-") and s["result"] == "done"):
                        s["end"] = s["end"] or parse_time(t)
                        s["result"] = rest
                if not any(x["name"].startswith(name + "-") for x in steps.values()):
                    steps.setdefault(name, dict(name=name, start=parse_time(t), end=parse_time(t), result=rest))
        elif m := FOUND_RE.match(l):
            run["found"] = dict(total=int(m.group(1)), todo=int(m.group(2)), gb=float(m.group(3)))
        elif m := KIND_RE.match(l):
            run["kinds"][m.group(1)] = dict(n=int(m.group(2)), gb=float(m.group(3)))
        elif m := COPY_RE.match(l):
            run["copy"] = dict(n=int(m.group(1)), of=int(m.group(2)), path=m.group(4))
        elif m := IMPORT_RE.match(l):
            run["imp"] = dict(n=int(m.group(1)), of=int(m.group(2)), imported=int(m.group(3)),
                              unchanged=int(m.group(4)))
        elif l.startswith("FAILED  "):
            run["failed"].append(l[8:])
        elif m := DONE_RE.match(l):
            run["finished"], run["problems"] = parse_time(m.group(1)), int(m.group(2))
    run["steps"] = list(steps.values())
    return run


def ledger_since(since):
    out = {}
    try:
        for line in open(os.path.join(RESULTS, "ledger.jsonl")):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            fin = e.get("finished_at")
            if since and fin and dt.datetime.fromisoformat(fin) >= since:
                out[e["path"]] = e   # last entry per path wins
    except OSError:
        pass
    return list(out.values())


def db_stats():
    try:
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver(env("NEO4J_URI", "bolt://localhost:7688"),
                                   auth=(env("NEO4J_USER", "neo4j"), env("NEO4J_PASSWORD")),
                                   connection_timeout=5, notifications_min_severity="OFF")
        with drv.session() as s:
            imported = s.run("MATCH (st:ImportState) WHERE st.source_id STARTS WITH 'testlogs:' "
                             "AND st.pruned_at IS NULL RETURN count(st) AS n").single()["n"]
            runs = s.run("MATCH (r:TestLogRun) RETURN r.ib_tag AS tag, r.suite AS suite, count(*) AS n").data()
            problems = s.run("MATCH (p:LogProblem) RETURN count(p) AS n").single()["n"]
        drv.close()
        by_date = {}
        for r in runs:
            m = TAG_DATE_RE.search(r["tag"] or "")
            d = m.group(1) if m else "?"
            by_date.setdefault(d, {}).setdefault(r["suite"], 0)
            by_date[d][r["suite"]] += r["n"]
        return dict(imported=imported, problems=problems, by_date=by_date, error=None)
    except Exception as exc:   # noqa: BLE001 -- the page must still render
        return dict(error=f"{type(exc).__name__}: {exc}")


def bar(done, total, label=""):
    pct = 0 if not total else min(100, 100 * done / total)
    return (f'<div class="bar"><div class="fill" style="width:{pct:.1f}%"></div>'
            f'<span>{html.escape(label)} {pct:.0f}%</span></div>')


def fmt_dur(sec):
    sec = int(max(sec, 0))
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}s"


def tester_section():
    hist = os.path.join(REPORTS, "history.txt")
    lines = open(hist).read().splitlines()[-8:] if os.path.exists(hist) else []
    running = bool(lines) and "started" in lines[-1]
    state = ("running now" if running else "idle") if lines else "has not run yet"
    reports = sorted(glob.glob(os.path.join(REPORTS, "20*.md")))
    newest = f'newest report: <a href="/report">{html.escape(os.path.basename(reports[-1]))}</a>' if reports else ""
    rows = "".join(f"<li>{html.escape(l)}</li>" for l in reversed(lines)) or "<li>no runs yet</li>"
    return f"<h2>Bot tester — {state}</h2><p>{newest}</p><ul class=small>{rows}</ul>"


def page():
    now = dt.datetime.now()
    run = last_run()
    out = [f"<h1>CMSSDT data refresh</h1><p class=small>{now:%Y-%m-%d %H:%M:%S} · reloads every 30 s · "
           f"log {html.escape(run['log'] if run else '-')}</p>"]
    if not run:
        out.append("<p>No refresh log yet.</p>")
    else:
        if run["finished"]:
            state = (f"finished {run['finished']:%H:%M}, {run['problems']} problem(s), took "
                     f"{fmt_dur((run['finished'] - run['started']).total_seconds())}")
        else:
            state = f"RUNNING for {fmt_dur((now - run['started']).total_seconds())}" if run["started"] else "?"
        out.append(f"<h2>Refresh started {run['started']:%Y-%m-%d %H:%M} — {state}</h2><table>"
                   "<tr><th>step</th><th>started</th><th>took</th><th>result</th></tr>")
        for s in run["steps"]:
            took = fmt_dur(((s["end"] or now) - s["start"]).total_seconds())
            res = s["result"] or "<b>running…</b>"
            cls = "bad" if s["result"] and ("FAILED" in s["result"] or "problem" in s["result"]) else ""
            out.append(f"<tr class='{cls}'><td>{s['name']}</td><td>{s['start']:%H:%M}</td><td>{took}</td>"
                       f"<td>{res if not s['result'] else html.escape(res)}</td></tr>")
        out.append("</table>")

        # test logs: copy + scan (ledger = exact bytes), then import into Neo4j
        if run["found"]:
            done = ledger_since(run["started"])
            ok = [e for e in done if e.get("status") == "done"]
            bad = [e for e in done if e.get("status") != "done"]
            total_gb = run["found"]["gb"]
            done_gb = sum(e.get("size", 0) for e in ok) / 1e9
            todo = run["found"]["todo"]
            out.append(f"<h2>IB test logs from cmssdt9-02</h2><p>{run['found']['total']} zips on the server, "
                       f"{todo} to do this run ({total_gb:.1f} GB)</p>")
            out.append(bar(len(ok), todo, f"copied + scanned: {len(ok)} / {todo} zips"))
            out.append(bar(done_gb, total_gb, f"{done_gb:.1f} / {total_gb:.1f} GB, {total_gb - done_gb:.1f} GB left"))
            fetch = next((s for s in run["steps"] if s["name"] == "testlogs-fetch"), None)
            if fetch and not fetch["end"] and done_gb > 0:
                rate = done_gb / max((now - fetch["start"]).total_seconds(), 1)
                out.append(f"<p>rate {rate * 3600:.1f} GB/h → about {fmt_dur((total_gb - done_gb) / rate)} left "
                           "(relval zips are the big ones, so it varies)</p>")
            if run["copy"] and fetch and not fetch["end"]:
                out.append(f"<p class=small>now copying [{run['copy']['n']}/{run['copy']['of']}] "
                           f"{html.escape(run['copy']['path'])}</p>")
            out.append("<table><tr><th>kind</th><th>zips done / to do</th><th>GB done / to do</th></tr>")
            for kind, plan in run["kinds"].items():
                k_ok = [e for e in ok if e.get("kind") == kind]
                out.append(f"<tr><td>{kind}</td><td>{len(k_ok)} / {plan['n']}</td>"
                           f"<td>{sum(e.get('size', 0) for e in k_ok) / 1e9:.1f} / {plan['gb']:.1f}</td></tr>")
            out.append("</table>")
            if bad or run["failed"]:
                items = "".join(f"<li>{html.escape(e['path'])}: {html.escape(str(e.get('error', ''))[:200])}</li>"
                                for e in bad) or "".join(f"<li>{html.escape(f[:250])}</li>" for f in run["failed"])
                out.append(f"<p class=bad>{len(bad) or len(run['failed'])} failed (retried next run):</p><ul class=small>{items}</ul>")

    db = db_stats()
    out.append("<h2>In the database (Neo4j)</h2>")
    if db.get("error"):
        out.append(f"<p class=bad>Neo4j not reachable: {html.escape(db['error'])}</p>")
    else:
        n_results = len(glob.glob(os.path.join(RESULTS, "**", "*.json.gz"), recursive=True))
        if run and run["imp"]:
            i = run["imp"]
            out.append(bar(i["n"], i["of"], f"import running: {i['n']} / {i['of']} result files"))
        out.append(bar(db["imported"], n_results, f"scanned result files imported: {db['imported']} / {n_results}"))
        out.append(f"<p>{n_results - db['imported']} scanned but not in the DB yet (imported after the copy "
                   f"finishes) · {db['problems']:,} LogProblem nodes</p>")
        dates = sorted((d for d in db["by_date"] if d != "?"), reverse=True)[:25]
        if dates:
            first = dt.date.fromisoformat(dates[-1])
            all_days = [(first + dt.timedelta(n)).isoformat()
                        for n in range((dt.date.fromisoformat(dates[0]) - first).days + 1)][::-1]
            out.append("<table><tr><th>IB date</th><th>unit</th><th>addon</th><th>relval</th><th></th></tr>")
            peak = max(sum(v.values()) for v in db["by_date"].values()) or 1
            for d in all_days:
                v = db["by_date"].get(d, {})
                tot = sum(v.values())
                cell = (f'<div class="mini" style="width:{200 * tot / peak:.0f}px"></div>' if tot
                        else '<span class=bad>missing</span>')
                out.append(f"<tr><td>{d}</td><td>{v.get('unit', '')}</td><td>{v.get('addon', '')}</td>"
                           f"<td>{v.get('relval', '')}</td><td>{cell}</td></tr>")
            out.append("</table><p class=small>TestLogRun count per IB build date (from the tag). "
                       "“missing” = no test logs for that day.</p>")

    out.append(tester_section())
    hist = os.path.join(LOG_DIR, "history.txt")
    if os.path.exists(hist):
        rows = "".join(f"<li>{html.escape(l)}</li>" for l in reversed(open(hist).read().splitlines()[-10:]))
        out.append(f"<h2>Refresh history</h2><ul class=small>{rows}</ul>")
    return ("<!doctype html><html><head><meta charset=utf-8><meta http-equiv=refresh content=30>"
            "<title>CMSSDT refresh status</title><style>"
            "body{font:14px system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}"
            "h2{margin-top:28px;border-bottom:1px solid #ddd}table{border-collapse:collapse;margin:8px 0}"
            "td,th{padding:3px 10px;border-bottom:1px solid #eee;text-align:left}"
            ".bar{position:relative;background:#eee;border-radius:4px;height:22px;margin:6px 0;max-width:700px}"
            ".fill{background:#4a8;height:100%;border-radius:4px}.bar span{position:absolute;left:8px;top:2px}"
            ".mini{background:#4a8;height:10px}.bad{color:#b33}.small{font-size:12px;color:#555}"
            "</style></head><body>" + "".join(out) + "</body></html>")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/report":
            reports = sorted(glob.glob(os.path.join(REPORTS, "20*.md")))
            body, ctype = (open(reports[-1]).read() if reports else "no report yet"), "text/plain; charset=utf-8"
        elif self.path == "/":
            body, ctype = page(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
