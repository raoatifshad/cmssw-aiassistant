#!/bin/bash
# Nightly backup for the VM stack: the test-log files and Neo4j.
#   1. received/ + results/ -> DATA_DIR/backups/testlogs-<date>.tar (+ .sha256), nothing stopped.
#      cmssdt9-02 deletes what it sent, so these files exist only here.
#   2. stop the neo4j container   (Community edition can only dump a stopped database)
#   3. neo4j-admin database dump  -> DATA_DIR/backups/neo4j-<date>.dump (+ .sha256)
#   4. start neo4j again          (always, even if the dump failed)
#   5. keep the newest BACKUP_KEEP (3) of each, delete older ones -- only after the new one is
#      there, and never files named otherwise (e.g. a hand-made neo4j-live-*.dump)
#   6. copy both to EOS           (only if BACKUP_EOS_ENABLED=1; off for now)
# Run from deploy/vm (reads .env). Exit code != 0 means the backup did NOT happen.
#   ./backup.sh                 # project "kg"
#   KG_PROJECT=kg-vm-test ./backup.sh
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ./.env; set +a

PROJECT=${KG_PROJECT:-kg}
IMAGE=docker.io/library/neo4j:5.26.29
KEEP=${BACKUP_KEEP:-3}
BACKUPS="$DATA_DIR/backups"
STAMP=$(date +%Y%m%d-%H%M%S)
TMP="$BACKUPS/.tmp-$STAMP"
OUT="$BACKUPS/neo4j-$STAMP.dump"
log() { echo "$(date '+%F %T') backup: $*"; }

mkdir -p "$TMP"
prune() {   # prune <folder> <prefix> <suffix>: newest $KEEP of <prefix><date...><suffix> stay
    { ls -1t "$1"/"$2"[0-9]*"$3" 2>/dev/null || true; } | tail -n +$((KEEP + 1)) | while read -r old; do
        log "removing old $old"; podman unshare rm -f "$old" "$old.sha256"
    done
}
checksum() { (cd "$BACKUPS" && podman unshare sha256sum "$1" > "$1.sha256"); }

# received/ first: a batch merged meanwhile is then in results/, read after it. merged/ is in
# results/ already. tar exit 1 = a file changed or went while reading (they are only ever
# replaced whole), 2 = real failure. A failure here must not stop the database backup.
FILES="$BACKUPS/testlogs-$STAMP.tar"
files_ok=1; parts=()
for d in received results; do if [ -d "$DATA_DIR/$d" ]; then parts+=("$d"); fi; done
if [ ${#parts[@]} -gt 0 ]; then
    rc=0
    podman unshare tar -C "$DATA_DIR" --exclude='received/merged' --exclude='.*.part' \
        -cf "$FILES.part" "${parts[@]}" || rc=$?
    if [ $rc -le 1 ] && podman unshare tar -tf "$FILES.part" >/dev/null; then
        podman unshare mv "$FILES.part" "$FILES"
        checksum "$(basename "$FILES")"
        log "wrote $FILES ($(numfmt --to=iec "$(podman unshare stat -c %s "$FILES")"))"
        prune "$BACKUPS" testlogs- .tar
    else
        files_ok=0; podman unshare rm -f "$FILES.part"
        log "ERROR: the test-log files were not saved (tar exit $rc)"
    fi
fi
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

checksum "$(basename "$OUT")"
size=$(podman unshare stat -c %s "$OUT")
[ "$size" -gt 1000 ] || { log "ERROR: dump is only $size bytes"; exit 1; }
log "wrote $OUT ($(numfmt --to=iec "$size"))"

prune "$BACKUPS" neo4j- .dump

if [ "${BACKUP_EOS_ENABLED:-0}" = "1" ]; then
    : "${BACKUP_EOS_DIR:?set BACKUP_EOS_DIR when BACKUP_EOS_ENABLED=1}"
    for f in "$FILES" "$OUT"; do
        [ -e "$f" ] || continue
        log "copying $(basename "$f") to EOS $BACKUP_EOS_DIR"
        podman unshare cat "$f" > "$BACKUP_EOS_DIR/$(basename "$f").part"
        mv "$BACKUP_EOS_DIR/$(basename "$f").part" "$BACKUP_EOS_DIR/$(basename "$f")"
        cp "$f.sha256" "$BACKUP_EOS_DIR/"
        (cd "$BACKUP_EOS_DIR" && sha256sum -c "$(basename "$f").sha256") || { log "ERROR: EOS copy differs"; exit 1; }
    done
    prune "$BACKUP_EOS_DIR" neo4j- .dump
    prune "$BACKUP_EOS_DIR" testlogs- .tar
else
    log "EOS copy disabled (BACKUP_EOS_ENABLED=0)"
fi
[ $files_ok = 1 ] || { log "database saved, test-log files NOT saved"; exit 1; }
log "done"
