# Settings of the test machine. Copy to config.sh (not committed) and adjust.
# Every script here reads config.sh; nothing in this branch has a path of its own.

# --- the chatbot code under test
CHATBOT_DIR=/path/to/chatbot-dev          # a checkout of the chatbot repo on branch dev
PYTHON=$CHATBOT_DIR/venv/bin/python       # venv with requirements.txt installed

# --- the test stack: deploy/vm/compose.yaml of CHATBOT_DIR with its own deploy/vm/.env
#     (DATA_DIR, ports, DEV_SHIFTER_LOGIN=true, SHIFTER_CERN_USERNAMES incl. SHIFTER_USER)
PROJECT=kg-test                           # podman compose project name (never "kg": that is production)
API_URL=http://localhost:8006             # = API_PORT in that .env
NEO4J_URI=bolt://localhost:7690           # = NEO4J_BOLT_PORT in that .env (read-only checks)
SHIFTER_USER=testshifter
KEEP_WORKER=0                             # 0 = stop the test stack's worker: the data comes from the backup

# --- production data: the VM's nightly dumps, copied here by deploy/vm/fetch_backup.sh
BACKUP_DIR=/path/to/vm-backups

# --- results
REPORTS_DIR=/path/to/test-reports         # one folder per night: status.txt, logs, check outputs

# --- the automated tester (optional; TESTER=0 runs only the checks)
TESTER=1
CLAUDE_BIN=claude
FIX_WORKTREE=/path/to/chatbot-tester-fixes   # a second worktree of the chatbot repo, on FIX_BRANCH
FIX_BRANCH=tester-fixes                      # fixes land here; a person reviews them and merges into dev
FIX_API_PORT=8003                            # the tester's own server for checking a fix
NIGHT_START=20                               # tester.sh runs only between these hours (FORCE=1 overrides)
NIGHT_END=6
MAX_HOURS=6
