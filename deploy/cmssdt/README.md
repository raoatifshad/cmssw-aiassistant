# Sender job on cmssdt9-02

`send_test_log_scans.py` sends the IB test-log scan files that cms-bot's jobs write on
cmssdt9-02 to the chatbot VM (`POST /api/test-logs/ingest`). Standard library only,
Python 3.8+, reads nothing but `*-scan.json.gz`.

## Where the files come from (checked in the Jenkins configs, 2026-11-17)

| Jenkins job | runs | writes (our hook) |
|---|---|---|
| `ib-build-logs` (every 30 min) | `es_ibs_log.py` | `unitTestLogs-scan.json.gz`, `addOnTests-scan.json.gz` |
| `process-relval-logs` (every 20 min) | `jenkins-jobs/process-relval-logs` | `pyRelValMatrixLogs-scan.json.gz` |

Both run on node `cmssdt` = `cmsbuild@cmssdt9-02.cern.ch`. Every scan file sits next to its zip:

    /data/sdt/buildlogs/<arch>/www/<day>/<queue>/<IB tag>/[gpu/<card>/|other/<name>/]<name>-scan.json.gz

## The Jenkins job (to be created by the cms-bot maintainers)

* node: `cmssdt` (same machine and user as the two jobs above)
* schedule: `H/15 * * * *`
* a secret file credential holding the token (= `TEST_LOGS_INGEST_TOKEN` in the VM's `.env`)
* build step:

      python3 $CMS_BOT_DIR/send_test_log_scans.py --url https://<vm>:<port> --token-file "$TOKEN_FILE"

Exit codes: 0 = done, 1 = some files failed or were rejected (shown in the log; failed ones
are retried next run), 2 = the VM refused the token or ingest is off (nothing sent).

## What it leaves behind

Next to each scan file a marker `<scan file>.sent`: `<sha256> <answer> <time>`. A file is
sent again only when its content changed (a RelVal scan is rewritten when a late job is
merged). The markers go away with the build folder when cms-bot reuses the weekday folder.

## By hand

    send_test_log_scans.py --dry-run                     # what would be sent
    send_test_log_scans.py --url ... --token-file ... --resend-days 7
        # after the VM was down or rebuilt: the VM answers "unchanged" for what it has

## Tested (2026-11-17)

`tests/test_test_log_ingest.py` (sender against a real local HTTP server running the
endpoint's checks: send once, skip, rewritten file sent again, resend days, short outage
retried, bad file reported once, wrong token stops at once, VM down gives up). Live on the
test stack (kg-vm-test, :8006): the 8 real scan files from the R1-R3 tests sent, a repeat
skipped all 8, `--resend-days 7` got 8 "unchanged", a wrong token stopped with exit 2; the
worker imported them (6,802 tests/workflows, 2,139 problems), one run per build. Python 3.8
dry run on the same tree: all 8 found (incl. gpu/ and other/).
