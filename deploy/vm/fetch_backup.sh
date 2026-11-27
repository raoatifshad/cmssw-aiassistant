#!/bin/bash
# Runs on the PC (not on the VM): copies the VM's newest nightly backup here, so losing the VM
# costs at most a day. Read-only on the VM. Copies the newest neo4j-<date>.dump and
# testlogs-<date>.tar with their .sha256, checks them, keeps the newest KEEP of each.
#   fetch_backup.sh            # from cron, 03:30 (the VM backs up at 03:00) and after a boot
# Result in $DEST/status.txt, log in $DEST/fetch.log. Exit 1 = no fresh, verified copy.
set -uo pipefail
VM=${VM:-root@cmssdt-chatbot.cern.ch}
KEY=${KEY:-$HOME/.ssh/cmssdt-chatbot.pem}
SRC=${SRC:-/srv/kg-data/backups}
DEST=${DEST:-/home/rao/CHAT-BOT-AI/vm-backups}
KEEP=${KEEP:-3}
MAX_AGE_H=${MAX_AGE_H:-30}         # an older newest backup means the VM did not make one
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -i $KEY"

mkdir -p "$DEST"; chmod 700 "$DEST"
exec 9>"$DEST/.lock"; flock -n 9 || exit 0      # already running
log() { echo "$(date '+%F %T') $*" | tee -a "$DEST/fetch.log"; }
finish() { echo "$(date '+%F %T') $1: $2" > "$DEST/status.txt"; log "$1: $2"; [ "$1" = OK ]; exit $?; }

names=$($SSH "$VM" "cd $SRC && ls -1t neo4j-[0-9]*.dump | head -1; ls -1t testlogs-[0-9]*.tar | head -1") \
    || finish FAILED "cannot list the backups on $VM"
[ "$(echo "$names" | wc -w)" = 2 ] || finish FAILED "expected one dump and one tar on the VM, found: $names"
got=""
for name in $names; do
    if [ ! -e "$DEST/$name" ]; then
        rsync -t --partial -e "$SSH" "$VM:$SRC/$name" "$DEST/.$name.part" \
            && rsync -t -e "$SSH" "$VM:$SRC/$name.sha256" "$DEST/$name.sha256" \
            || finish FAILED "copy of $name"
        want=$(cut -d' ' -f1 "$DEST/$name.sha256"); have=$(sha256sum "$DEST/.$name.part" | cut -d' ' -f1)
        if [ -z "$want" ] || [ "$want" != "$have" ]; then
            rm -f "$DEST/.$name.part" "$DEST/$name.sha256"
            finish FAILED "checksum of $name"
        fi
        mv "$DEST/.$name.part" "$DEST/$name"
        got="$got $name ($(du -h "$DEST/$name" | cut -f1))"
    fi
    age=$(( ($(date +%s) - $(stat -c %Y "$DEST/$name")) / 3600 ))
    [ "$age" -le "$MAX_AGE_H" ] || finish FAILED "newest on the VM is $name, $age hours old: the VM makes no backups?"
done
for kind in 'neo4j-[0-9]*.dump' 'testlogs-[0-9]*.tar'; do
    ( cd "$DEST" && { ls -1t $kind 2>/dev/null || true; } | tail -n +$((KEEP + 1)) | while read -r old; do rm -f "$old" "$old.sha256"; done )
done
find "$DEST" -maxdepth 1 -name '.*.part' -mtime +2 -delete
finish OK "${got:-nothing new}; newest: $(echo $names)"
