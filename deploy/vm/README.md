# Chatbot backend on the CERN VM

Containers hold only programs. Every byte of data is under `DATA_DIR` on the host
(put it on the separate attached disk), so a container can be deleted, updated or
rebuilt at any time without losing anything.

| Container | What | Data (on the host) |
|---|---|---|
| `neo4j` | Neo4j 5.26.29 Community | `DATA_DIR/neo4j`, dumps in `DATA_DIR/backups` |
| `api` | chat API (`/api/chat`, `/api/health`, ...) | `DATA_DIR/results`, `DATA_DIR/log-rag`, `DATA_DIR/clang-logs` (read-only) |
| `worker` | imports + daily refresh | *(next step, not here yet)* |

Not included on purpose: OpenSearch and Logstash (only ever held one smoke-test
event; no chat answer used them). The nightly tester stays on the developer PC.

## First start

```bash
sudo dnf install -y podman podman-compose git     # or the VM's equivalent
git clone <this repo> && cd <repo>/deploy/vm
cp .env.example .env && chmod 600 .env            # fill in; never commit .env
mkdir -p $DATA_DIR/{neo4j,backups,log-rag,clang-logs,results}   # DATA_DIR from .env
podman compose -p kg up -d --build
curl -s http://localhost:$API_PORT/api/health     # {"ok":true,...}
```

On the VM set `DEV_SHIFTER_LOGIN=false` and give Neo4j real memory
(e.g. `NEO4J_HEAP=4g`, `NEO4J_PAGECACHE=6g` on a 16 GB machine).

## Everyday

```bash
podman compose -p kg ps                           # what is running
podman compose -p kg logs --tail 50 api           # API log
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

*(next step: `backup.sh` -- nightly stop, dump, start, copy to EOS; `restore.sh`)*

## Tested (on the developer PC, 2026-11-16)

Stack as `kg-vm-test` on ports 8006/7690/7477 beside the live bot: health OK,
empty database, a record written, `down` (containers deleted), `up` -- record
still there.
