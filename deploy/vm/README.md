# Chatbot backend on the CERN VM

Containers hold only programs. Every byte of data is under `DATA_DIR` on the host
(put it on the separate attached disk), so a container can be deleted, updated or
rebuilt at any time without losing anything.

| Container | What | Data (on the host) |
|---|---|---|
| `neo4j` | Neo4j 5.26.29 Community | `DATA_DIR/neo4j`, dumps in `DATA_DIR/backups` |
| `api` | chat API (`/api/chat`, `/api/health`, ...) | `DATA_DIR/results`, `DATA_DIR/log-rag`, `DATA_DIR/clang-logs` (read-only) |
| `worker` | test-log import every 30 min; cmsrepo, exceptions, docs at 00:30/12:30 (`worker.py`) | `DATA_DIR/sources` (checkouts), `DATA_DIR/worker` (state.json, logs) |

Not included on purpose: OpenSearch and Logstash (only ever held one smoke-test
event; no chat answer used them). The nightly tester stays on the developer PC.

## First start

```bash
sudo dnf install -y podman podman-compose git     # or the VM's equivalent
git clone <this repo> && cd <repo>/deploy/vm
cp .env.example .env && chmod 600 .env            # fill in; never commit .env
mkdir -p $DATA_DIR/{neo4j,backups,log-rag,clang-logs,results,sources,worker}   # DATA_DIR from .env
podman compose -p kg up -d --build
curl -s http://localhost:$API_PORT/api/health     # {"ok":true,...}
```

On the VM set `DEV_SHIFTER_LOGIN=false` and give Neo4j real memory
(e.g. `NEO4J_HEAP=4g`, `NEO4J_PAGECACHE=6g` on a 16 GB machine).

## Moving the live database here (one-off)

The database remembers the full path of every file it imported (`source_id`). On
the developer PC those are `/home/rao/...`, here `/data/sources/...`; without
rewriting them the worker re-imports everything once and the wiki docs end up in
the graph twice.

```bash
# on the PC: a dump of the live database (stops it ~20 s), copy it here
./restore.sh /path/to/neo4j-live-<date>.dump      # loads it (worker not started yet!)
podman compose -p kg stop worker                 # if it is running
P=/home/rao/CHAT-BOT-AI/NEO4J/cmssdt-kg-chatbot
podman compose -p kg run --rm --no-deps worker python migrate_paths.py \
    --map $P/cmsrepo=/data/sources/cmsrepo --map $P/shifter-wiki=/data/sources/shifter-wiki \
    --map /home/rao/CHAT-BOT-AI/CLANG-LOGS=/data/clang-logs            # dry run: counts only
#   ... the same + --apply --check   (must end with "old paths anywhere in the graph: none")
rm -f $DATA_DIR/worker/state.json && podman compose -p kg up -d worker
```

Rehearsed on the PC (2026-11-16, live copy of 13:25): 2.97 million paths rewritten in
~110 s (5,000 per transaction: the 512 MB test heap ran out with bigger ones); the
worker's first round then took 1 min instead of 19 (only IBs newer than the copy),
0 docs re-imported, no duplicates, same answers.

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
./backup.sh                     # stop neo4j -> dump -> start -> DATA_DIR/backups/neo4j-<date>.dump + .sha256
./restore.sh <dump file>        # check .sha256 -> stop -> load (replaces the database) -> start -> count nodes
```

* Community edition can only dump a **stopped** database, so the API gets errors
  for the few minutes the dump takes. Run it at night, e.g. a systemd timer at 03:00.
* Never copy the files in `DATA_DIR/neo4j` while Neo4j runs: such a copy may not load.
* The newest `BACKUP_KEEP` (14) dumps are kept.
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
