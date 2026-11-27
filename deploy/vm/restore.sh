#!/bin/bash
# Load a dump into the VM stack's Neo4j, replacing what is there.
#   ./restore.sh /path/to/neo4j-YYYYmmdd-HHMMSS.dump      (or any neo4j dump file)
#   KG_PROJECT=kg-vm-test ./restore.sh <dump>
# Steps: check the .sha256 if there is one -> stop neo4j -> load (overwrite) -> start -> count nodes.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ./.env; set +a

DUMP=$(realpath "${1:?usage: restore.sh <dump file>}")
PROJECT=${KG_PROJECT:-kg}
IMAGE=docker.io/library/neo4j:5.26.29
log() { echo "$(date '+%F %T') restore: $*"; }

if [ -e "$DUMP.sha256" ]; then
    (cd "$(dirname "$DUMP")" && podman unshare sha256sum -c "$(basename "$DUMP").sha256") || { log "ERROR: checksum mismatch"; exit 1; }
fi

STAGE="$DATA_DIR/backups/.restore-$$"
mkdir -p "$STAGE"
podman unshare cp "$DUMP" "$STAGE/neo4j.dump"     # neo4j-admin load wants <db>.dump in a folder
podman unshare chown -R 7474:7474 "$STAGE"

log "stopping neo4j ($PROJECT)"
podman compose -p "$PROJECT" stop -t 120 neo4j >/dev/null
trap 'log "starting neo4j"; podman compose -p "$PROJECT" start neo4j >/dev/null; podman unshare rm -rf "$STAGE"' EXIT

started=$(date +%s)
podman run --rm -v "$DATA_DIR/neo4j:/data:Z" -v "$STAGE:/dump:Z" "$IMAGE" \
    neo4j-admin database load neo4j --from-path=/dump --overwrite-destination=true
log "load took $(( $(date +%s) - started )) s"
trap - EXIT
log "starting neo4j"; podman compose -p "$PROJECT" start neo4j >/dev/null
podman unshare rm -rf "$STAGE"

for _ in $(seq 60); do
    n=$(podman compose -p "$PROJECT" exec -T neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
        --format plain "MATCH (n) RETURN count(n)" 2>/dev/null | tail -1) && break
    sleep 5
done
log "neo4j is back with ${n:-?} nodes"
