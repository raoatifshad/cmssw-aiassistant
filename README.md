# Test machine tools

This branch (`tester`) has its own history and no chatbot code. It holds the tools of the
test machine, which tests the `dev` branch at night on a copy of yesterday's production data.

```
production VM ── nightly backup (deploy/vm/backup.sh) ──> fetch_backup.sh copies it here
                                                                  │
test machine:  nightly.sh ── restore dump into the test Neo4j ────┘
                          ── rebuild the test API from dev
                          ── live checks (question bank, validate_*) ──> REPORTS_DIR/<date>/
                          ── tester.sh (optional): new questions, fixes committed to FIX_BRANCH
                                                                  │
             a person reviews FIX_BRANCH ── merges into dev ── PR dev -> master ── tag ── deploy
```

Production is never touched: the VM is only read through its backup files.

| File | What it does |
|---|---|
| `config.example.sh` | every setting; copy to `config.sh` (not committed) |
| `nightly.sh` | update dev, restore the newest dump, rebuild the test API, run the checks, start the tester |
| `tester.sh` + `tester_prompt.md` | automated tester (Claude Code headless): asks new questions, reads the answers, commits fixes to `FIX_BRANCH`, never pushes |
| `crontab.example` | the cron lines of the test machine |
| `tools/fetch_ib_test_logs.py` | copy + scan the IB test-log zips from cmssdt9-02 by ssh (the old way, before cms-bot sends the scans to the VM) |
| `tools/summarize_router_shadow.py` | how often the LLM router agrees with the regex router (`RouterShadowLog`) |

## Setup

1. A checkout of the chatbot repo on `dev` (`CHATBOT_DIR`) with a venv
   (`python3 -m venv venv && venv/bin/pip install -r requirements.txt`).
2. A test stack from that checkout: `deploy/vm/.env` from `deploy/vm/.env.example` with its own
   `DATA_DIR` and ports, `DEV_SHIFTER_LOGIN=true`, and `SHIFTER_CERN_USERNAMES` including
   `SHIFTER_USER`. Start it once: `cd deploy/vm && podman compose -p kg-test up -d`
   (see `deploy/vm/README.md`). Size Neo4j's memory to the machine (`NEO4J_HEAP`, `NEO4J_PAGECACHE`).
3. The VM's backups copied here every night: `deploy/vm/fetch_backup.sh` with `DEST=BACKUP_DIR`.
4. For the tester only: a second worktree on the fix branch
   (`git -C $CHATBOT_DIR worktree add -b tester-fixes /path/to/chatbot-tester-fixes origin/dev`)
   and Claude Code installed and logged in.
5. `cp config.example.sh config.sh`, fill it in, try `SKIP_RESTORE=1 ./nightly.sh`, then
   install the cron lines from `crontab.example`.

## Results

`REPORTS_DIR/<date-HHMM>/`: `status.txt` (one line per step: code, data, api, each check,
tester), `nightly.log`, one `.txt` per check, `question_bank.json`, and the tester's
`report.md` + `tester.log`. `REPORTS_DIR/history.txt` has one line per run. Folders older than
45 days are deleted.

## Rules

- `PROJECT` is never `kg` (production's name); `nightly.sh` refuses it.
- The test stack's worker is stopped (`KEEP_WORKER=0`): its data comes from the backup, not
  from imports.
- The tester never pushes, never deploys, never writes to Neo4j; its fixes wait on
  `FIX_BRANCH` for review.
