#!/usr/bin/env bash
# Unattended bot tester: runs Claude Code headless with scripts/tester_prompt.md in the separate
# worktree $WORKTREE (branch auto/vacation-fixes). Started by daily_refresh.sh after each refresh
# (TESTER=0 to skip); can also be run by hand.
#
#   report:  $REPORTS/<YYYY-MM-DD-HHMM>.md   (written by the tester)
#   log:     $REPORTS/<YYYY-MM-DD-HHMM>.claude.log
#
# Never two at once (flock); if the previous tester is still running, this one is skipped.
# The hard rules live in the prompt; the most important ones are also enforced here with
# --disallowedTools, and auto permission mode refuses anything else risky.
set -u

REPO=/home/rao/CHAT-BOT-AI/NEO4J/cmssdt-kg-chatbot
WORKTREE=/home/rao/CHAT-BOT-AI/cmssdt-kg-vacation
BRANCH=auto/vacation-fixes
REPORTS=/home/rao/CHAT-BOT-AI/daily-test-reports
CLAUDE=${CLAUDE:-/home/rao/.local/bin/claude}
MAX_HOURS=${MAX_HOURS:-6}
export PATH=/home/rao/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH

mkdir -p "$REPORTS"
STAMP=$(date +%F-%H%M)
LOG="$REPORTS/$STAMP.claude.log"

exec 8>"$REPORTS/.lock"
if ! flock -n 8; then
    echo "$(date -Is) previous tester still running, skipped" >> "$REPORTS/history.txt"
    exit 0
fi

cd "$WORKTREE" || { echo "$(date -Is) FAILED: no worktree $WORKTREE" >> "$REPORTS/history.txt"; exit 1; }
if [ "$(git branch --show-current)" != "$BRANCH" ]; then
    echo "$(date -Is) FAILED: $WORKTREE is not on $BRANCH, not running" >> "$REPORTS/history.txt"
    exit 1
fi

prompt=$(sed "s/__DATE__/$STAMP/g" "$REPO/scripts/tester_prompt.md")

echo "$(date -Is) started -- log $LOG" >> "$REPORTS/history.txt"
timeout --kill-after=5m "${MAX_HOURS}h" "$CLAUDE" -p "$prompt" \
    --permission-mode auto \
    --add-dir "$REPORTS" \
    --disallowedTools "Bash(git push:*)" "Bash(git checkout:*)" "Bash(git switch:*)" \
        "Bash(podman:*)" "Bash(docker:*)" "Bash(ssh:*)" "Bash(scp:*)" "Bash(rsync:*)" \
        "Bash(kinit:*)" "Bash(kdestroy:*)" "Bash(crontab:*)" "Bash(sudo:*)" \
        "WebFetch" "WebSearch" \
    > "$LOG" 2>&1 < /dev/null
rc=$?
pkill -f "uvicorn server.api_server:app --port 8003" 2>/dev/null   # in case the tester left it running
echo "$(date -Is) finished rc=$rc -- report $REPORTS/$STAMP.md $([ -e "$REPORTS/$STAMP.md" ] || echo '(MISSING)')" \
    >> "$REPORTS/history.txt"
find "$REPORTS" -name '20*.claude.log' -mtime +45 -delete
exit 0
