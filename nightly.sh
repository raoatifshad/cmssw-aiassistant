#!/bin/bash
# The test machine's night: test the dev branch on a copy of yesterday's production data.
#   1. update the dev checkout (fast-forward only)
#   2. load the newest VM dump into the test stack's Neo4j (deploy/vm/restore.sh), if it is new
#   3. rebuild and restart the test API from dev (Neo4j and its data are not recreated)
#   4. live checks against the test API
#   5. start tester.sh in the background (TESTER=1)
# Results: $REPORTS_DIR/<date-HHMM>/ (status.txt = one line per step). Exit 1 = a step that the
# later ones need failed. Production is never touched: the VM is only read through its backups.
#   ./nightly.sh            # from cron, after fetch_backup.sh copied the night's dump
#   SKIP_RESTORE=1 ./nightly.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
[ "$PROJECT" != kg ] || { echo "PROJECT=kg is production's name, refusing"; exit 2; }

STAMP=$(date +%F-%H%M)
OUT="$REPORTS_DIR/$STAMP"
mkdir -p "$OUT"
exec 9>"$REPORTS_DIR/.nightly.lock"
flock -n 9 || { echo "$(date -Is) nightly.sh already running" >> "$REPORTS_DIR/history.txt"; exit 0; }
log() { echo "$(date '+%F %T') $*" >> "$OUT/nightly.log"; }
status() { echo "$(date '+%F %T') $*" >> "$OUT/status.txt"; log "== $*"; }
VM_DIR="$CHATBOT_DIR/deploy/vm"
compose() { (cd "$VM_DIR" && podman compose -p "$PROJECT" "$@"); }

# 1. code
cur=$(git -C "$CHATBOT_DIR" branch --show-current)
if [ "$cur" != dev ]; then
    status "code: FAILED ($CHATBOT_DIR is on '$cur', not dev)"; exit 1
fi
if git -C "$CHATBOT_DIR" fetch -q origin dev >> "$OUT/nightly.log" 2>&1 \
   && git -C "$CHATBOT_DIR" merge -q --ff-only origin/dev >> "$OUT/nightly.log" 2>&1; then
    status "code: OK (dev at $(git -C "$CHATBOT_DIR" log -1 --format='%h %s'))"
else
    status "code: FAILED (dev could not be fast-forwarded; local changes?)"; exit 1
fi

# 2. data
dump=$(ls -1t "$BACKUP_DIR"/neo4j-[0-9]*.dump 2>/dev/null | head -1)
if [ "${SKIP_RESTORE:-0}" = 1 ]; then
    status "data: SKIPPED (SKIP_RESTORE=1; $(cat "$REPORTS_DIR/.restored" 2>/dev/null || echo 'nothing restored yet'))"
elif [ -z "$dump" ]; then
    status "data: FAILED (no neo4j-*.dump in $BACKUP_DIR)"; exit 1
elif [ "$(cat "$REPORTS_DIR/.restored" 2>/dev/null)" = "$(basename "$dump")" ]; then
    status "data: OK ($(basename "$dump") already loaded)"
elif (cd "$VM_DIR" && KG_PROJECT=$PROJECT ./restore.sh "$dump") >> "$OUT/nightly.log" 2>&1; then
    basename "$dump" > "$REPORTS_DIR/.restored"
    status "data: OK (loaded $(basename "$dump"))"
else
    status "data: FAILED (restore of $(basename "$dump"), see nightly.log)"; exit 1
fi

# 3. test API from dev
if [ "$KEEP_WORKER" != 1 ]; then compose stop worker >> "$OUT/nightly.log" 2>&1; fi
if compose build api >> "$OUT/nightly.log" 2>&1 \
   && podman rm -f "${PROJECT}_api_1" >> "$OUT/nightly.log" 2>&1 \
   && compose up -d --no-deps api >> "$OUT/nightly.log" 2>&1; then
    for _ in $(seq 60); do curl -fs "$API_URL/api/health" > /dev/null && break; sleep 5; done
    if curl -fs "$API_URL/api/health" > /dev/null; then
        status "api: OK ($API_URL)"
    else
        status "api: FAILED (no answer on $API_URL/api/health after 5 min)"; exit 1
    fi
else
    status "api: FAILED (build or start, see nightly.log)"; exit 1
fi

# 4. live checks (each one's output in its own file; a failing check does not stop the others)
NEO4J_PASSWORD=$(grep -E '^NEO4J_PASSWORD=' "$VM_DIR/.env" | cut -d= -f2-)
export NEO4J_URI NEO4J_PASSWORD
check() {   # check <name> <timeout> <command...>
    local name=$1 limit=$2; shift 2
    local started; started=$(date +%s)
    (cd "$CHATBOT_DIR" && PYTHONPATH=ingestion timeout "$limit" "$@") > "$OUT/$name.txt" 2>&1
    local rc=$?
    status "check $name: $([ $rc = 0 ] && echo OK || echo "FAILED (exit $rc)") ($(( $(date +%s) - started )) s)"
}
check facts          10m "$PYTHON" scripts/question_bank.py --check-facts
check question_bank  90m "$PYTHON" scripts/question_bank.py --base-url "$API_URL" --json "$OUT/question_bank.json"
check validate_bot   60m "$PYTHON" scripts/validate_bot.py --base-url "$API_URL"
check test_log_data  60m "$PYTHON" scripts/validate_test_log_data.py --base-url "$API_URL" --username "$SHIFTER_USER"
check shift_summary  30m "$PYTHON" scripts/validate_shift_summary.py --base-url "$API_URL" --username "$SHIFTER_USER"

# 5. tester
if [ "${TESTER:-0}" = 1 ]; then
    status "tester: started in the background (see $REPORTS_DIR/history.txt)"
    setsid nohup "$HERE/tester.sh" "$OUT" > /dev/null 2>&1 < /dev/null &
fi
echo "$(date -Is) nightly $STAMP done: $(grep -c FAILED "$OUT/status.txt") failed step(s)" >> "$REPORTS_DIR/history.txt"
find "$REPORTS_DIR" -mindepth 1 -maxdepth 1 -type d -name '20*' -mtime +45 -exec rm -rf {} +
exit 0
