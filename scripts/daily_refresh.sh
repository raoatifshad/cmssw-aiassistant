#!/usr/bin/env bash
# Daily unattended data refresh -- the manual routine, one step after another:
#
#   1. cmsrepo (dashboard data: RelVals, unit tests, build status, merged PRs) -> build_graph.py
#   2. RelVal exceptions: es-cmssw-exceptions.py -> a dated JSON file -> build_graph.py --es-exceptions-json
#   3. IB test logs from cmssdt9-02: fetch_ib_test_logs.py (copies one zip at a time, scans,
#      deletes it) -> ingest_test_logs.py
#   4. shifter documentation (cmssdt-wiki) -> ingest_shifter_docs.py
#   5. health checks: API up, validate_bot.py, validate_test_log_data.py
#
# A failing step is logged and the next one still runs. Every run writes
#   $LOG_DIR/<date>.log    the full output (kept 45 days)
#   $LOG_DIR/status.txt    one line per step for the last run (OK / FAILED / SKIPPED)
#   $LOG_DIR/history.txt   one summary line per run
#
# Exceptions go to a new file NAME per day on purpose: importing a file first deletes every
# exception imported earlier from that same path, and the export only covers the last 7 days --
# one fixed name would wipe all older exceptions on the first daily run. The old files are not
# needed after import (the graph keeps the exceptions), so only the newest one is kept.
#
# Login to cmssdt9-02 uses your Kerberos ticket, nothing stored: run `kinit` before leaving and
# every run renews it (`kinit -R`). CERN tickets renew for at most 5 days; after that the
# test-log step is skipped (status.txt says so). The server keeps zips about a week, so one
# manual run when you are back recovers the last 7 days. Only ssh to cmssdt9-02 is the fixed
# listing and rsync downloads of ingestion/fetch_ib_test_logs.py.
# If $KEYTAB exists (chmod 600, on this PC only), each run gets a fresh ticket from it instead,
# so the step works for as long as you are away. Delete it when you are back.
#
# Install (runs at 07:30 every day):   scripts/daily_refresh.sh --install-cron
# Try it now in the foreground:        scripts/daily_refresh.sh
# Only some steps:                     STEPS="cmsrepo exceptions" scripts/daily_refresh.sh
set -u

REPO=/home/rao/CHAT-BOT-AI/NEO4J/cmssdt-kg-chatbot
CMSBOT=/home/rao/CHAT-BOT-AI/NEO4J/cms-bot
CLANG_ROOT=/home/rao/CHAT-BOT-AI/CLANG-LOGS
EXC_DIR=/home/rao/CHAT-BOT-AI/exceptions-json-file
TESTLOGS=/home/rao/CHAT-BOT-AI/IB-TEST-LOGS
LOG_DIR=/home/rao/CHAT-BOT-AI/daily-refresh-logs
KEYTAB=/home/rao/.mrao.keytab
PRINCIPAL=mrao@CERN.CH
API=${API:-http://localhost:8002}
MIN_FREE_GB=${MIN_FREE_GB:-60}          # never start copying test logs below this much free disk
DISK_BUDGET_GB=${DISK_BUDGET_GB:-45}     # max unscanned zips on disk at once (a DBG RelVal zip is ~33 GB)
STEPS=${STEPS:-"cmsrepo exceptions testlogs docs checks"}
export NEO4J_URI=bolt://localhost:7688
export PATH=/usr/sue/bin:/usr/local/bin:/usr/bin:/bin:$PATH

if [ "${1:-}" = "--install-cron" ]; then
    line="30 7 * * * $REPO/scripts/daily_refresh.sh >/dev/null 2>&1"
    ( crontab -l 2>/dev/null | grep -v 'scripts/daily_refresh.sh'; echo "$line" ) | crontab -
    echo "installed: $line"; crontab -l | grep daily_refresh
    exit 0
fi
if [ "${1:-}" = "--uninstall-cron" ]; then
    crontab -l 2>/dev/null | grep -v 'scripts/daily_refresh.sh' | crontab -
    echo "removed"; exit 0
fi

mkdir -p "$LOG_DIR" "$EXC_DIR"
TODAY=$(date +%F)
LOG="$LOG_DIR/$TODAY.log"
STATUS="$LOG_DIR/status.txt.new"
exec 9>"$LOG_DIR/.lock"
if ! flock -n 9; then
    echo "$(date -Is) another refresh is still running, not starting a second one" >> "$LOG_DIR/history.txt"
    exit 0
fi
exec >>"$LOG" 2>&1
: > "$STATUS"
FAILED=0

log() { echo "[$(date '+%F %T')] $*"; }
status() { echo "$(printf '%-11s' "$1") $2" >> "$STATUS"; log "== $1: $2"; }
want() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
# run <step> <timeout> <command...>: a failure is recorded and the script carries on
run() {
    local step=$1 limit=$2; shift 2
    log "-- $step: $*"
    if timeout --kill-after=5m "$limit" "$@"; then return 0; fi
    local rc=$?
    [ $rc -eq 124 ] && status "$step" "FAILED (timed out after $limit)" || status "$step" "FAILED (exit $rc)"
    FAILED=$((FAILED + 1)); return 1
}

log "===== daily refresh, steps: $STEPS"
cd "$REPO" || exit 1

# Neo4j and the API run in containers with restart=always; after a reboot give them time.
for i in $(seq 1 30); do
    venv/bin/python - <<'EOF' >/dev/null 2>&1 && break
import os, sys; sys.path.insert(0, "graph")
from neo4j_conn import get_driver
d = get_driver(); d.verify_connectivity(); d.close()
EOF
    [ "$i" -eq 30 ] && { status neo4j "FAILED (not reachable on 7688 after 10 min) -- nothing imported"; mv "$STATUS" "$LOG_DIR/status.txt"; \
        echo "$(date -Is) FAILED: neo4j down" >> "$LOG_DIR/history.txt"; exit 1; }
    sleep 20
done
NEWEST_BEFORE=$(venv/bin/python - <<'EOF' 2>/dev/null
import sys; sys.path.insert(0, "graph"); sys.path.insert(0, "ingestion")
from neo4j_conn import get_driver
d = get_driver()
with d.session() as s:
    print(s.run("MATCH (ib:IB)-[:RAN_RELVAL]->() WITH DISTINCT ib.tag AS t RETURN max(right(t, 15)) AS w").single()["w"])
d.close()
EOF
)

# ---------------------------------------------------------------- 1. dashboard data
if want cmsrepo; then
    if run cmsrepo-git 20m bash -c "cd cmsrepo && git fetch --prune origin master && git reset --hard origin/master && git sparse-checkout reapply" \
       && run cmsrepo-import 3h env PYTHONPATH=ingestion venv/bin/python ingestion/build_graph.py \
              --cmsrepo "$REPO/cmsrepo" --clang-root "$CLANG_ROOT" --reconcile; then
        status cmsrepo "OK (cmsrepo at $(git -C cmsrepo log -1 --format=%cd --date=format:'%F %H:%M'))"
    fi
else status cmsrepo SKIPPED; fi

# ---------------------------------------------------------------- 2. RelVal exceptions
if want exceptions; then
    EXC_FILE="$EXC_DIR/exceptions-$TODAY.json"
    if run exceptions-export 2h bash -c "cd '$CMSBOT' && ./es-cmssw-exceptions.py > '$EXC_FILE.part'"; then
        # only a real export is imported: an empty or broken file must never reach the graph
        if venv/bin/python -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('hits',{}).get('hits') else 1)" "$EXC_FILE.part"; then
            mv "$EXC_FILE.part" "$EXC_FILE"
            if run exceptions-import 2h env PYTHONPATH=ingestion venv/bin/python ingestion/build_graph.py \
                   --cmsrepo cmsrepo --es-exceptions-json "$EXC_FILE" --reconcile; then
                status exceptions "OK ($(basename "$EXC_FILE"), $(du -h "$EXC_FILE" | cut -f1))"
                # once imported, the graph has the exceptions (--reconcile never removes exception
                # imports); keep only the newest export, for a re-run or a look
                find "$EXC_DIR" -name 'exceptions-*.json' ! -name "$(basename "$EXC_FILE")" -delete
            fi
        else
            status exceptions "FAILED (export empty or not valid JSON, kept as $EXC_FILE.part; nothing imported)"
            FAILED=$((FAILED + 1))
        fi
    fi
    find "$EXC_DIR" -name 'exceptions-*.json.part' -mtime +7 -delete  # failed exports, kept a week to look at
else status exceptions SKIPPED; fi

# ---------------------------------------------------------------- 3. IB test logs
if want testlogs; then
    if [ -f "$KEYTAB" ]; then
        kinit -kt "$KEYTAB" "$PRINCIPAL" 2>/dev/null || kinit -R 2>/dev/null
    else
        kinit -R 2>/dev/null                # extend the ticket you left (max 5 days from kinit)
    fi
    renew_until=$(klist 2>/dev/null | awk '/renew until/ {print $3, $4; exit}')
    FREE_GB=$(df -BG --output=avail "$TESTLOGS" | tail -1 | tr -dc 0-9)
    if ! klist -s 2>/dev/null || ! timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=20 root@cmssdt9-02.cern.ch true 2>/dev/null; then
        status testlogs "SKIPPED (Kerberos ticket expired -- no login to cmssdt9-02; run kinit, then this script, to catch up the last ~7 days)"
        FAILED=$((FAILED + 1))
    elif [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
        status testlogs "SKIPPED (only ${FREE_GB} GB free on /home, need ${MIN_FREE_GB})"
        FAILED=$((FAILED + 1))
    else
        budget=$DISK_BUDGET_GB
        [ $((FREE_GB - 20)) -lt "$budget" ] && budget=$((FREE_GB - 20))
        # copies one zip at a time, oldest first, scans it and deletes it; --delete-local also
        # scans and removes what an earlier manual rsync left in server-copy/
        if run testlogs-fetch 14h env PYTHONPATH=ingestion venv/bin/python ingestion/fetch_ib_test_logs.py \
               --delete-local --disk-budget-gb "$budget"; then
            if run testlogs-import 3h env PYTHONPATH=ingestion venv/bin/python ingestion/ingest_test_logs.py \
                   --results "$TESTLOGS/results"; then
                lost=$( [ -s "$TESTLOGS/results/lost.txt" ] && wc -l < "$TESTLOGS/results/lost.txt" || echo 0 )
                status testlogs "OK (ticket renewable until ${renew_until:-?}; free disk now $(df -BG --output=avail "$TESTLOGS" | tail -1 | tr -d ' '), lost zips listed: $lost)"
            fi
        fi
    fi
else status testlogs SKIPPED; fi

# ---------------------------------------------------------------- 4. shifter documentation
if want docs; then
    if [ -d shifter-wiki/.git ]; then
        if run docs-git 10m bash -c "cd shifter-wiki && git fetch --prune origin master && git reset --hard origin/master" \
           && run docs-import 1h env PYTHONPATH=ingestion venv/bin/python ingestion/ingest_shifter_docs.py \
                  --wiki-root "$REPO/shifter-wiki" --cmsrepo "$REPO/cmsrepo" --reconcile; then
            status docs OK
        fi
    else status docs "SKIPPED (no shifter-wiki clone)"; fi
else status docs SKIPPED; fi

# ---------------------------------------------------------------- 5. health checks
NEWEST_AFTER=$(venv/bin/python - <<'EOF' 2>/dev/null
import sys; sys.path.insert(0, "graph")
from neo4j_conn import get_driver
d = get_driver()
with d.session() as s:
    print(s.run("MATCH (ib:IB)-[:RAN_RELVAL]->() WITH DISTINCT ib.tag AS t RETURN max(right(t, 15)) AS w").single()["w"])
d.close()
EOF
)
if want checks; then
    if curl -sf -m 30 "$API/api/health" >/dev/null; then
        bot=$(timeout 30m venv/bin/python scripts/validate_bot.py --base-url "$API" 2>&1 | grep -E "turns passed" | tail -1)
        logs=$(timeout 30m venv/bin/python scripts/validate_test_log_data.py --base-url "$API" --username testshifter 2>&1 \
               | grep -E "[0-9]+ failures" | tail -1)
        status checks "API up; validate_bot: ${bot:-no result}; test logs: ${logs:-no result}"
        case "$bot" in *"turns passed"*) n=${bot%%/*}; t=${bot#*/}; t=${t%% *}; [ "$n" != "$t" ] && FAILED=$((FAILED + 1));; esac
    else
        status checks "FAILED (API not answering at $API)"; FAILED=$((FAILED + 1))
    fi
else status checks SKIPPED; fi

status data "newest IB with RelVal results: before ${NEWEST_BEFORE:-?}, after ${NEWEST_AFTER:-?}"
[ "${NEWEST_AFTER:-}" = "${NEWEST_BEFORE:-}" ] && log "NOTE: no newer RelVal results than yesterday"
mv "$STATUS" "$LOG_DIR/status.txt"
echo "$(date -Is) $([ $FAILED -eq 0 ] && echo OK || echo "$FAILED problem(s)") -- newest RelVals ${NEWEST_AFTER:-?} -- log $LOG" >> "$LOG_DIR/history.txt"
find "$LOG_DIR" -name '20*.log' -mtime +45 -delete
log "===== done, $FAILED problem(s)"
exit 0
