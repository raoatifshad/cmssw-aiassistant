#!/usr/bin/env bash
# Automated tester: runs Claude Code headless with tester_prompt.md in FIX_WORKTREE (a worktree
# of the chatbot repo on FIX_BRANCH). It asks the test API new questions, reads the answers,
# and commits fixes to FIX_BRANCH; a person reviews them and merges into dev. It never pushes.
#   ./tester.sh [<nightly result folder>]      # started by nightly.sh; FORCE=1 runs it by day
# Night only (a run takes hours and loads the machine); never two at once (flock).
# The hard rules are in the prompt; the most important ones are also enforced here with
# --disallowedTools.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
OUT=${1:-$REPORTS_DIR/$(date +%F-%H%M)}
mkdir -p "$OUT"
HIST="$REPORTS_DIR/history.txt"

hour=$((10#$(date +%H)))
if [ "${FORCE:-0}" != 1 ] && [ "$hour" -lt "$NIGHT_START" ] && [ "$hour" -ge "$NIGHT_END" ]; then
    echo "$(date -Is) tester: daytime, not running (${NIGHT_START}:00-${NIGHT_END}:00; FORCE=1 to override)" >> "$HIST"
    exit 0
fi
exec 8>"$REPORTS_DIR/.tester.lock"
if ! flock -n 8; then echo "$(date -Is) tester: previous run still going, skipped" >> "$HIST"; exit 0; fi

cd "$FIX_WORKTREE" || { echo "$(date -Is) tester: FAILED, no worktree $FIX_WORKTREE" >> "$HIST"; exit 1; }
if [ "$(git branch --show-current)" != "$FIX_BRANCH" ]; then
    echo "$(date -Is) tester: FAILED, $FIX_WORKTREE is not on $FIX_BRANCH" >> "$HIST"; exit 1
fi
LOG="$OUT/tester.log"
# Start from the code under test: merge dev into FIX_BRANCH (a fast-forward when nothing is
# waiting for review). On a conflict stop: testing older code would report fixed bugs again.
git fetch -q origin dev >> "$LOG" 2>&1
if ! git merge --no-edit origin/dev >> "$LOG" 2>&1; then
    git merge --abort 2>/dev/null
    echo "$(date -Is) tester: FAILED, origin/dev does not merge into $FIX_BRANCH (conflict), see $LOG" >> "$HIST"
    exit 1
fi

prompt=$(sed -e "s|__DATE__|$(basename "$OUT")|g" -e "s|__OUT__|$OUT|g" -e "s|__API_URL__|$API_URL|g" \
             -e "s|__FIX_BRANCH__|$FIX_BRANCH|g" -e "s|__FIX_PORT__|$FIX_API_PORT|g" \
             -e "s|__NEO4J_URI__|$NEO4J_URI|g" -e "s|__USER__|$SHIFTER_USER|g" -e "s|__PYTHON__|$PYTHON|g" \
             "$HERE/tester_prompt.md")

echo "$(date -Is) tester: started, log $LOG" >> "$HIST"
timeout --kill-after=5m "${MAX_HOURS}h" "$CLAUDE_BIN" -p "$prompt" \
    --permission-mode auto \
    --add-dir "$OUT" \
    --disallowedTools "Bash(git push:*)" "Bash(git checkout:*)" "Bash(git switch:*)" \
        "Bash(podman:*)" "Bash(docker:*)" "Bash(ssh:*)" "Bash(scp:*)" "Bash(rsync:*)" \
        "Bash(kinit:*)" "Bash(kdestroy:*)" "Bash(crontab:*)" "Bash(sudo:*)" \
        "WebFetch" "WebSearch" \
    >> "$LOG" 2>&1 < /dev/null
rc=$?
pkill -f "uvicorn server.api_server:app --port $FIX_API_PORT" 2>/dev/null   # in case it was left running
echo "$(date -Is) tester: finished rc=$rc, report $OUT/report.md $([ -e "$OUT/report.md" ] || echo '(MISSING)')" >> "$HIST"
exit 0
