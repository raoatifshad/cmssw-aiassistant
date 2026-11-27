# Chatbot backend on the CERN VM

Containers hold only programs. Every byte of data is under `DATA_DIR` on the host
(put it on the separate attached disk), so a container can be deleted, updated or
rebuilt at any time without losing anything.

| Container | What | Data (on the host) |
|---|---|---|
| `neo4j` | Neo4j 5.26.29 Community | `DATA_DIR/neo4j`, dumps in `DATA_DIR/backups` |
| `api` | chat API (`/api/chat`, `/api/health`, ...); receives test-log scan files from cmssdt9-02 (`POST /api/test-logs/batch`, saved to `DATA_DIR/received`) | `DATA_DIR/received`, `DATA_DIR/results`, `DATA_DIR/log-rag`, `DATA_DIR/clang-logs` (read-only) |
| `worker` | test-log batches merged and imported every 5 min; cmsrepo every hour at :30 (`WORKER_CMSREPO_AT`); exceptions, docs at 00:30/12:30 (`worker.py`) | `DATA_DIR/received`, `DATA_DIR/results`, `DATA_DIR/sources` (checkouts), `DATA_DIR/worker` (state.json, logs) |

Not included on purpose: OpenSearch and Logstash (only ever held one smoke-test
event; no chat answer used them). The nightly tester stays on the developer PC.

## Rebuild from scratch (the VM is gone)

The production VM (created 2026-11-17) is `cmssdt-chatbot.cern.ch`, OpenStack project
"CMS SDT Build": image `ALMA9 - x86_64`, flavor `m4.xlarge` (8 vCPU, 29 GB RAM, 160 GB disk),
metadata `cern-activedirectory`, `cern-checkdns`, `cern-services`, `cern-waitdns` all `true`,
an SSH key pair. Not Puppet-managed (yet). To build a new one:

1. OpenStack: Launch Instance with the settings above; optionally a data volume (~200 GB),
   attached to it (shows up as e.g. `/dev/vdb`).
2. Code: `git clone <this repo> /opt/cmssdt-kg-chatbot` (not under /root), check out the branch
   in use. Without network access to the repo: `git bundle create kg.bundle <branch>` on a machine
   that has it, `scp`, `git clone -b <branch> kg.bundle /opt/cmssdt-kg-chatbot`.
3. As root: `/opt/cmssdt-kg-chatbot/deploy/vm/setup-vm.sh [--data-device /dev/vdb]`
   (packages, data disk at `/srv/kg-data`, user `kgbot`, `.env` with generated secrets and
   Neo4j memory from the RAM, systemd units for start at boot and the 03:00 backup, firewall).
   Safe to run again.
4. Fill in `CERN_API_BASE` / `CERN_API_KEY` in `deploy/vm/.env`. `setup-vm.sh` made a new
   `TEST_LOGS_INGEST_TOKEN`: copy it into Jenkins credential `chatbot-test-logs-token`
   ("The ingest token" below), or the sender job gets 401.
5. Data, as `kgbot` (`runuser -l kgbot`, `cd /opt/cmssdt-kg-chatbot/deploy/vm`):
   `podman compose -p kg up -d neo4j`, `./restore.sh <newest dump>` (from EOS or another copy);
   a dump of the database that already ran here (or on another VM with the same paths) loads as is;
   then, as root, `systemctl --user -M kgbot@ enable --now kg-stack` (`systemctl --user` does not
   work inside `runuser`: no session bus) and `curl -s localhost:8001/api/health`.
   Changed `.env` later? `podman compose -p kg up -d --force-recreate --no-deps api worker`:
   a plain `up -d` keeps the old containers with the old settings.
   With no dump: just start; the worker imports cmsrepo, exceptions and docs (history older
   than what cms-sw.github.io still has is lost).

Update the code later (as `kgbot`): `git fetch origin && git checkout <branch> && git pull`,
then `podman compose -p kg build api worker`, `podman rm -f kg_worker_1 kg_api_1`,
`podman compose -p kg up -d --no-deps api worker`. Without `--no-deps` podman-compose also
recreates Neo4j (found 2026-11-23).

## Cheat sheet: working on cmssdt-chatbot.cern.ch

```bash
# --- log in (from the PC that has the key; never copy the key anywhere else) ---
ssh -i ~/.ssh/cmssdt-chatbot.pem root@cmssdt-chatbot.cern.ch

# --- as root: the stack as a whole (systemd user units of kgbot) ---
systemctl --user -M kgbot@ status kg-stack            # is it up?
systemctl --user -M kgbot@ restart kg-stack           # stop all 3 cleanly (Neo4j gets 120 s) and start again
systemctl --user -M kgbot@ stop kg-stack              # stop all 3; `start` to start again
systemctl --user -M kgbot@ disable kg-stack           # not at boot any more (`enable` to undo)
systemctl --user -M kgbot@ list-timers                # next/last 03:00 backup
systemctl --user -M kgbot@ status kg-backup           # how the last backup ended
journalctl _UID=$(id -u kgbot) --since today | grep backup.sh   # the backup's own log
curl -s localhost:8001/api/health                     # {"ok":true,"model":...}
cat /srv/kg-data/worker/state.json                    # last run + result of each worker job
df -h /srv/kg-data                                    # disk

# --- as kgbot: containers, logs, code (podman is rootless: as root you see no containers) ---
runuser -l kgbot
cd /opt/cmssdt-kg-chatbot/deploy/vm
podman ps                                             # kg_neo4j_1, kg_api_1, kg_worker_1
podman logs --tail 50 -f kg_api_1                     # API log, follow (Ctrl-C ends)
podman logs --since 10m kg_api_1 | grep -E "AI supervisor|AI routing|AI check"   # which model decided, how long
podman logs --tail 50 kg_worker_1
podman compose -p kg restart api                      # restart one container (same code, same .env)
podman compose -p kg up -d --force-recreate --no-deps api worker   # after changing .env
podman compose -p kg exec worker python worker.py --once cmsrepo   # run one worker job now
./backup.sh                                           # a dump now (Neo4j down ~30 s)

# --- new code (as kgbot): from GitHub once the branch is pushed ---
cd /opt/cmssdt-kg-chatbot && git fetch origin && git checkout <branch> && git pull --ff-only
cd deploy/vm && podman compose -p kg build api worker && podman compose -p kg up -d --force-recreate --no-deps api worker
#   without GitHub: on the PC `git bundle create kg.bundle <last commit on the VM>..<branch>`, scp it to /tmp,
#   then here `git pull --ff-only /tmp/kg.bundle <branch>`
```

`exit` leaves `runuser`. Never `podman compose down -v`, never delete `/srv/kg-data`.

## After a reboot

Nothing needs to be started by hand: `kgbot` has lingering (its units run without a login),
`kg-stack` and `kg-backup.timer` are enabled, firewalld is enabled and every container has
`restart=always` (checked 2026-11-18). `reboot` as root stops the stack cleanly (Neo4j gets up
to 120 s). After 2-3 minutes, check that it came back:

```bash
curl -s http://cmssdt-chatbot.cern.ch:8001/api/health; echo         # from any CERN machine, no login

# on the VM, as root
systemctl --user -M kgbot@ is-active kg-stack kg-backup.timer        # active, active
runuser -l kgbot -c 'podman ps --format "{{.Names}}  {{.Status}}"'   # 3 containers Up, neo4j (healthy)
ss -ltn | grep -E ':(8001|7474|7687) '                              # *:8001, 127.0.0.1:7474, 127.0.0.1:7687
curl -s localhost:8001/api/health; echo                             # {"ok":true,...}
curl -s -X POST localhost:8001/api/chat -H 'Content-Type: application/json' \
     -d '{"question":"what is the latest IB?","session_id":"vm-check"}' | head -c 300; echo
python3 -m json.tool /srv/kg-data/worker/state.json | head -40      # last run + result of each worker job
ls -lt /srv/kg-data/backups | head -3                               # newest dump (03:00)
uptime -s                                                           # when the VM last started
```

If something is not up:

* `systemctl --user -M kgbot@ restart kg-stack` fixes most cases, including Neo4j "Up" but
  without its `7474`/`7687` ports (the API then cannot reach the database).
* kg-stack itself fails: as kgbot, `podman compose -p kg up -d` in this folder, then
  `podman logs --tail 50 kg_neo4j_1` / `kg_api_1` / `kg_worker_1`; as root,
  `journalctl _UID=$(id -u kgbot) --since today --no-pager | tail -50`.

## First start (by hand, e.g. a test stack on a PC)

```bash
sudo dnf install -y podman podman-compose git     # or the VM's equivalent
git clone <this repo> && cd <repo>/deploy/vm
cp .env.example .env && chmod 600 .env            # fill in; never commit .env
mkdir -p $DATA_DIR/{neo4j,backups,log-rag,clang-logs,results,received,sources,worker}   # DATA_DIR from .env
podman compose -p kg up -d --build
curl -s http://localhost:$API_PORT/api/health     # {"ok":true,...}
```

On the VM set `DEV_SHIFTER_LOGIN=false` and give Neo4j real memory
(e.g. `NEO4J_HEAP=4g`, `NEO4J_PAGECACHE=6g` on a 16 GB machine).

## Test-log scan files from cmssdt9-02

A sender job on cmssdt9-02 (`deploy/cmssdt/`, see its README) posts tars of small files (one
per unit/addOn zip, one per RelVal step) to `POST /api/test-logs/batch` and deletes every file
the answer confirms, so from then on this VM has the only copy.

1. **API**: checks every file, saves the good ones as one tar in `DATA_DIR/received/` (fsync),
   then answers `{"saved": {name: sha256}, "rejected": {name: reason}, "kept": {name: sha256}}`. A
   refused file is kept in `received/rejected/` with its reason (30 days, at most 200 MB), because
   the sender deletes it too; after a fix, tar them and post them again. 401 wrong token; 503 when
   `TEST_LOGS_INGEST_TOKEN` is empty or less than 5 GB is free (the sender keeps its files).
2. **Worker**, every `WORKER_TESTLOGS_EVERY_MIN` (5) minutes:
   `merge_test_log_batches.py` merges the steps into one file per build,
   `DATA_DIR/results/<kind>/<arch>/<tag>/<variant or _>/<name>.json.gz`, reads it back to check,
   and moves the batch to `received/merged/`; `ingest_test_logs.py` imports the changed files;
   then imported batches older than `RECEIVED_KEEP_DAYS` (7; 0 = at once) are deleted.

A batch that cannot be merged stays in `received/` and is tried again; an unreadable one goes to
`received/bad/`. Both make the testlogs job FAILED in `worker/state.json` and its log. Never delete
`results/`: it is what the graph can be rebuilt from.

    ls DATA_DIR/received/ DATA_DIR/received/{bad,rejected} 2>/dev/null   # waiting / broken / refused
    cat DATA_DIR/received/rejected/*.reason.txt                          # why (count: worker/state.json)

`POST /api/test-logs/ingest` (one whole result of a build, gzip body, saved straight into
`results/`) is still there for sending one by hand:

    curl -s -H "X-Test-Logs-Token: $TOKEN" -H 'Content-Type: application/gzip' \
         --data-binary @unit__<arch>__<tag>__main.json.gz http://localhost:$API_PORT/api/test-logs/ingest

### The ingest token

`TEST_LOGS_INGEST_TOKEN` in `.env` (random, made by `setup-vm.sh`) is the only thing that lets
an upload in: port 8001 is open to the whole CERN network and not behind SSO, and the API
can't filter by address (rootless podman shows every client as 10.89.0.x). The Jenkins job
`send-test-log-scans` on cmssdt9-02 holds a copy as the Secret text credential
`chatbot-test-logs-token`, bound to the variable `TEST_LOGS_INGEST_TOKEN` (details and the
copy command: `deploy/cmssdt/README.md`). Never print it, never commit `.env`.

New token (leaked, or someone left), as root:

    cd /opt/cmssdt-kg-chatbot/deploy/vm
    new=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
    sed -i "s/^TEST_LOGS_INGEST_TOKEN=.*/TEST_LOGS_INGEST_TOKEN=$new/" .env; unset new
    runuser -l kgbot -c 'cd /opt/cmssdt-kg-chatbot/deploy/vm && podman compose -p kg up -d --force-recreate --no-deps api'

then update the Jenkins credential (the sender job is red with exit 2 until then; scan files
are not lost: the next good run sends everything not yet sent).

## Moving the live database here (one-off, done 2026-11-16)

The database remembers the full path of every file it imported (`source_id`). When it moved
from the developer PC (`/home/rao/...`) to the VM (`/data/sources/...`), `migrate_paths.py`
rewrote those paths once. It was removed afterwards (2026-11-27); it is in the git history
(commit f5adcfb) if a database ever moves from a machine with other paths again.

## Everyday

```bash
podman compose -p kg ps                           # what is running
podman compose -p kg logs --tail 50 api           # API log
cat $DATA_DIR/worker/state.json                   # last run + result of every worker job
podman compose -p kg exec worker python worker.py --once cmsrepo   # run one job now
podman compose -p kg up -d --build api            # deploy new code: rebuilds only the API, Neo4j keeps running
podman compose -p kg down                         # stop and DELETE the containers -- data stays in DATA_DIR
podman compose -p kg up -d                        # start again on the same data
```

Never use `down -v` or delete `DATA_DIR` unless you mean to lose the database.

## Files in DATA_DIR/neo4j belong to Neo4j's user

With rootless podman they show up as uid 10xxxx on the host. To copy or remove
them by hand, go through podman's user namespace:

```bash
podman unshare ls -l $DATA_DIR/neo4j
podman unshare rm -rf $DATA_DIR/neo4j/*           # only with Neo4j stopped, and only if you mean it
```

## Backups

```bash
./backup.sh                     # received/ + results/ -> DATA_DIR/backups/testlogs-<date>.tar, then
                                # stop neo4j -> dump -> start -> neo4j-<date>.dump (both with .sha256)
./restore.sh <dump file>        # check .sha256 -> stop -> load (replaces the database) -> start -> count nodes
```

* Community edition can only dump a **stopped** database, so the API gets errors
  for the few minutes the dump takes. Run it at night, e.g. a systemd timer at 03:00.
* Never copy the files in `DATA_DIR/neo4j` while Neo4j runs: such a copy may not load.
* The newest `BACKUP_KEEP` (3) nightly dumps and tars are kept; files named otherwise
  (`neo4j-live-*.dump`) are never deleted.
* Test-log files: `testlogs-<date>.tar` holds `received/` (batches not merged yet) and `results/`.
  Into a new VM, with the worker stopped:
  `podman unshare tar -C $DATA_DIR -xpf testlogs-<date>.tar`, then start the worker.
  It only adds or replaces files; stop if `results/` there is newer than the tar.
* Copy outside the VM: `fetch_backup.sh` on the developer PC (cron 03:30 and 15 min after a boot)
  copies the newest dump and tar to `/home/rao/CHAT-BOT-AI/vm-backups/`, checks the sha256, keeps 3;
  `status.txt` there says OK or FAILED. Restored on the PC on 2026-11-23: same node counts.
* EOS copy is built in but **off** (`BACKUP_EOS_ENABLED=0`). To switch it on, mount
  EOS (or make the folder reachable), set `BACKUP_EOS_ENABLED=1` and
  `BACKUP_EOS_DIR=/eos/...`; each copy is checked against its .sha256.
* `stop_grace_period: 2m`: podman's default 10 s ends in SIGKILL for Neo4j, which
  a database should never get.

## Tested (on the developer PC, 2026-11-16)

Stack as `kg-vm-test` on ports 8006/7690/7477 beside the live bot: health OK,
empty database, a record written, `down` (containers deleted), `up` -- record
still there. Backup (dump 3 s), database folder deleted, `restore.sh` --
record back, checksum OK.

Real data: the live database (12.8 GB store) dumped in 8 s -- live Neo4j down
16 s -- to a 1.3 GB dump; `restore.sh` loaded it in 30 s. Node counts for all
32 labels, 8,206,830 relationships and 70 indexes identical to live; the same
chat questions gave the same answers on both.
