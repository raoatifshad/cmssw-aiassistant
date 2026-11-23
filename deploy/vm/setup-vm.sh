#!/bin/bash
# setup-vm.sh -- prepare a fresh AlmaLinux 9 VM for the chatbot stack (Neo4j + API + worker).
#
# Run as root from the repo checkout (it uses the checkout it lives in):
#     deploy/vm/setup-vm.sh [--data-device /dev/vdb] [--data-dir /srv/kg-data] [--start]
#
# What it does (safe to run again: every step checks what is already there):
#   1. installs podman, podman-compose, git, rsync
#   2. data disk: with --data-device, makes an XFS file system on it ONLY if the device has
#      none, and mounts it at --data-dir via /etc/fstab (by UUID, nofail)
#   3. user "kgbot" (rootless podman, like the tested PC setup; backup.sh/restore.sh need
#      `podman unshare`, which does not work as root) with lingering, so its units run at boot
#   4. DATA_DIR folders, owned by kgbot
#   5. deploy/vm/.env from .env.example if missing (mode 600): DATA_DIR, Neo4j memory from the
#      VM's RAM, Neo4j ports bound to 127.0.0.1 only, random NEO4J_PASSWORD, SESSION_SECRET_KEY
#      and TEST_LOGS_INGEST_TOKEN. Never overwrites a value already set. CERN_API_KEY etc.
#      must be filled in by hand.
#   6. systemd user units for kgbot: kg-stack.service (compose up, clean stop; at boot see 8) and
#      kg-backup.timer (backup.sh every night at 03:00)
#   7. firewalld: opens the API port (the CERN perimeter firewall still blocks outside access)
#   8. with --start: builds the images, starts the stack and enables it at boot. Without it
#      nothing is started, not even after a reboot, so a database can be restored first (an
#      empty one would make the worker import everything from scratch).
set -euo pipefail

DATA_DIR=/srv/kg-data
DATA_DEVICE=""
RUN_USER=kgbot
START=0
while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir) DATA_DIR=$2; shift 2 ;;
        --data-device) DATA_DEVICE=$2; shift 2 ;;
        --user) RUN_USER=$2; shift 2 ;;
        --start) START=1; shift ;;
        -h|--help) sed -n 2,26p "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

log() { echo "== $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run as root"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VMDIR="$REPO/deploy/vm"
ENV="$VMDIR/.env"
[ -f "$VMDIR/compose.yaml" ] || die "compose.yaml not found next to this script"
case "$REPO" in /root/*) die "the checkout is under /root, which $RUN_USER cannot read: put it in e.g. /opt/cmssdt-kg-chatbot" ;; esac

log "1. packages"
dnf -y -q install podman podman-compose git rsync python3 >/dev/null

log "2. data disk"
mkdir -p "$DATA_DIR"
if [ -n "$DATA_DEVICE" ]; then
    [ -b "$DATA_DEVICE" ] || die "$DATA_DEVICE is not a block device"
    if [ -z "$(blkid -o value -s TYPE "$DATA_DEVICE" || true)" ]; then
        log "   $DATA_DEVICE has no file system: making XFS"
        mkfs.xfs -q -L kg-data "$DATA_DEVICE"
    fi
    uuid=$(blkid -o value -s UUID "$DATA_DEVICE")
    if ! grep -q "UUID=$uuid" /etc/fstab; then
        echo "UUID=$uuid $DATA_DIR xfs defaults,nofail 0 2" >> /etc/fstab
        systemctl daemon-reload
    fi
    mountpoint -q "$DATA_DIR" || mount "$DATA_DIR"
    log "   $DATA_DEVICE mounted at $DATA_DIR"
else
    log "   no --data-device: $DATA_DIR is on the root disk"
fi

log "3. user $RUN_USER"
id "$RUN_USER" >/dev/null 2>&1 || useradd -m "$RUN_USER"
grep -q "^$RUN_USER:" /etc/subuid || usermod --add-subuids 100000-165535 --add-subgids 100000-165535 "$RUN_USER"
loginctl enable-linger "$RUN_USER"
uid=$(id -u "$RUN_USER")
for _ in $(seq 30); do [ -S "/run/user/$uid/bus" ] && break; sleep 1; done
[ -S "/run/user/$uid/bus" ] || die "the user manager of $RUN_USER did not start"
chown -R "$RUN_USER:$RUN_USER" "$REPO"

log "4. folders in $DATA_DIR"
for d in neo4j backups log-rag clang-logs results received sources worker; do
    [ -d "$DATA_DIR/$d" ] || install -d -o "$RUN_USER" -g "$RUN_USER" "$DATA_DIR/$d"
done
chown "$RUN_USER:$RUN_USER" "$DATA_DIR"

log "5. $ENV"
if [ ! -f "$ENV" ]; then
    install -m 600 -o "$RUN_USER" -g "$RUN_USER" "$VMDIR/.env.example" "$ENV"
    log "   created from .env.example"
fi
chmod 600 "$ENV"; chown "$RUN_USER:$RUN_USER" "$ENV"
get() { sed -n "s/^$1=//p" "$ENV" | tail -1; }
put() {   # put KEY VALUE [force]: only when empty/placeholder unless force
    local cur; cur=$(get "$1")
    if [ -n "$cur" ] && [ "$cur" != "change-me" ] && [ "${3:-}" != force ]; then return; fi
    if grep -q "^$1=" "$ENV"; then
        python3 - "$ENV" "$1" "$2" <<'PY'
import sys
path, key, value = sys.argv[1:]
lines = open(path).read().splitlines()
out = [f"{key}={value}" if l.startswith(key + "=") else l for l in lines]
open(path, "w").write("\n".join(out) + "\n")
PY
    else
        echo "$1=$2" >> "$ENV"
    fi
}
rand() { python3 -c 'import secrets; print(secrets.token_urlsafe(32))'; }
[ "$(get DATA_DIR)" = /srv/kg-data ] && put DATA_DIR "$DATA_DIR" force
put DATA_DIR "$DATA_DIR"
# an unquoted value with spaces breaks `source .env` in backup.sh/restore.sh
sed -i -E 's/^WORKER_STEPS=([^"].* .*)$/WORKER_STEPS="\1"/' "$ENV"
ram_gb=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
if [ "$ram_gb" -ge 24 ]; then heap=8g; pc=12g; elif [ "$ram_gb" -ge 12 ]; then heap=4g; pc=6g; else heap=1g; pc=2g; fi
[ "$(get NEO4J_HEAP)" = 512m ] && put NEO4J_HEAP "$heap" force
[ "$(get NEO4J_PAGECACHE)" = 512m ] && put NEO4J_PAGECACHE "$pc" force
[ "$(get NEO4J_HTTP_PORT)" = 7474 ] && put NEO4J_HTTP_PORT 127.0.0.1:7474 force
[ "$(get NEO4J_BOLT_PORT)" = 7687 ] && put NEO4J_BOLT_PORT 127.0.0.1:7687 force
put NEO4J_PASSWORD "$(rand)"
put SESSION_SECRET_KEY "$(rand)"
put TEST_LOGS_INGEST_TOKEN "$(rand)"
API_PORT=$(get API_PORT); API_PORT=${API_PORT:-8001}
missing=""
for k in CERN_API_BASE CERN_API_KEY; do [ -n "$(get $k)" ] || missing="$missing $k"; done

log "6. systemd units for $RUN_USER"
UNITS="/home/$RUN_USER/.config/systemd/user"
install -d -o "$RUN_USER" -g "$RUN_USER" "/home/$RUN_USER/.config" "/home/$RUN_USER/.config/systemd" "$UNITS"
cat > "$UNITS/kg-stack.service" <<EOF
[Unit]
Description=cmssdt chatbot stack (podman compose project kg: neo4j, api, worker)

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$VMDIR
ExecStart=/usr/bin/podman compose -p kg up -d
ExecStop=/usr/bin/podman compose -p kg stop -t 120
TimeoutStartSec=1800
TimeoutStopSec=300

[Install]
WantedBy=default.target
EOF
cat > "$UNITS/kg-backup.service" <<EOF
[Unit]
Description=cmssdt chatbot: nightly Neo4j dump (backup.sh)

[Service]
Type=oneshot
WorkingDirectory=$VMDIR
ExecStart=$VMDIR/backup.sh
# backup.sh starts Neo4j again, so its conmon and rootlessport land in this unit; the default KillMode killed them when the
# backup ended (2026-11-18 03:02: Neo4j kept running, without its 127.0.0.1 ports and without podman watching it)
KillMode=process
EOF
cat > "$UNITS/kg-backup.timer" <<EOF
[Unit]
Description=cmssdt chatbot: nightly Neo4j dump at 03:00

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF
chown "$RUN_USER:$RUN_USER" "$UNITS"/kg-*
ucl() { systemctl --user -M "$RUN_USER@" "$@"; }
ucl daemon-reload
ucl enable --now kg-backup.timer >/dev/null 2>&1
# start at boot only once the stack is meant to run: enabled here before a restore, a reboot
# would start the worker on the empty database (happened 2026-11-17)
[ "$START" = 1 ] && ucl enable kg-stack.service >/dev/null 2>&1

log "7. firewall: API port $API_PORT"
if systemctl is-active -q firewalld; then
    firewall-cmd -q --permanent --add-port="$API_PORT/tcp" && firewall-cmd -q --reload
fi

if [ "$START" = 1 ]; then
    [ -z "$missing" ] || die "fill in$missing in $ENV before --start"
    log "8. building images and starting the stack (first build takes a few minutes)"
    runuser -l "$RUN_USER" -c "cd '$VMDIR' && podman compose -p kg build"
    ucl restart kg-stack.service
    for _ in $(seq 60); do curl -fs "http://localhost:$API_PORT/api/health" >/dev/null && break; sleep 5; done
    curl -fs "http://localhost:$API_PORT/api/health" && echo || die "API not healthy: runuser -l $RUN_USER -c 'cd $VMDIR && podman compose -p kg logs --tail 50 api'"
else
    log "8. not started (no --start)"
fi

echo
echo "Done. Code: $REPO   Data: $DATA_DIR   Config: $ENV (user $RUN_USER)"
[ -z "$missing" ] || echo "Fill in by hand in $ENV:$missing"
echo "Work as the stack's user:  runuser -l $RUN_USER    (then: cd $VMDIR)"
[ "$START" = 1 ] || echo "Not started, not started at boot. As $RUN_USER: podman compose -p kg up -d neo4j; ./restore.sh <dump>;
                           then as root: systemctl --user -M $RUN_USER@ enable --now kg-stack   (api + worker, and at every boot)"
