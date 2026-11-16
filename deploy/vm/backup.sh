#!/bin/bash
# Nightly Neo4j backup for the VM stack.
#   1. stop the neo4j container   (Community edition can only dump a stopped database)
#   2. neo4j-admin database dump  -> DATA_DIR/backups/neo4j-<date>.dump (+ .sha256)
#   3. start neo4j again          (always, even if the dump failed)
#   4. keep the newest BACKUP_KEEP dumps, delete older ones
#   5. copy the dump to EOS       (only if BACKUP_EOS_ENABLED=1; off for now)
# Run from deploy/vm (reads .env). Exit code != 0 means the backup did NOT happen.
#   ./backup.sh                 # project "kg"
#   KG_PROJECT=kg-vm-test ./backup.sh
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ./.env; set +a

PROJECT=${KG_PROJECT:-kg}
IMAGE=docker.io/library/neo4j:5.26.29
KEEP=${BACKUP_KEEP:-14}
BACKUPS="$DATA_DIR/backups"
STAMP=$(date +%Y%m%d-%H%M%S)
TMP="$BACKUPS/.tmp-$STAMP"
OUT="$BACKUPS/neo4j-$STAMP.dump"
log() { echo "$(date '+%F %T') backup: $*"; }

mkdir -p "$TMP"
podman unshare chown 7474:7474 "$TMP"   # the dump runs as neo4j's user inside the image
log "stopping neo4j ($PROJECT)"
podman compose -p "$PROJECT" stop -t 120 neo4j >/dev/null
restart() { log "starting neo4j"; podman compose -p "$PROJECT" start neo4j >/dev/null || log "ERROR: neo4j did not start"; }
trap restart EXIT

started=$(date +%s)
podman run --rm -v "$DATA_DIR/neo4j:/data:Z" -v "$TMP:/dump:Z" "$IMAGE" \
    neo4j-admin database dump neo4j --to-path=/dump
podman unshare mv "$TMP/neo4j.dump" "$OUT"
podman unshare rmdir "$TMP"
trap - EXIT
restart
log "dump took $(( $(date +%s) - started )) s, neo4j was down for that long"

(cd "$BACKUPS" && podman unshare sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256")
size=$(podman unshare stat -c %s "$OUT")
[ "$size" -gt 1000 ] || { log "ERROR: dump is only $size bytes"; exit 1; }
log "wrote $OUT ($(numfmt --to=iec "$size"))"

# keep the newest $KEEP dumps
ls -1t "$BACKUPS"/neo4j-*.dump 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r old; do
    log "removing old $old"; podman unshare rm -f "$old" "$old.sha256"
done

if [ "${BACKUP_EOS_ENABLED:-0}" = "1" ]; then
    : "${BACKUP_EOS_DIR:?set BACKUP_EOS_DIR when BACKUP_EOS_ENABLED=1}"
    log "copying to EOS $BACKUP_EOS_DIR"
    podman unshare cat "$OUT" > "$BACKUP_EOS_DIR/$(basename "$OUT").part"
    mv "$BACKUP_EOS_DIR/$(basename "$OUT").part" "$BACKUP_EOS_DIR/$(basename "$OUT")"
    cp "$OUT.sha256" "$BACKUP_EOS_DIR/"
    (cd "$BACKUP_EOS_DIR" && sha256sum -c "$(basename "$OUT").sha256") || { log "ERROR: EOS copy differs"; exit 1; }
    ls -1t "$BACKUP_EOS_DIR"/neo4j-*.dump | tail -n +$((KEEP + 1)) | while read -r old; do rm -f "$old" "$old.sha256"; done
else
    log "EOS copy disabled (BACKUP_EOS_ENABLED=0)"
fi
log "done"
