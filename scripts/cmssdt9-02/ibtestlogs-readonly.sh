#!/bin/bash
# Forced command for the cmssdt-kg-chatbot test-log key in root's authorized_keys on
# cmssdt9-02. Whatever command the client asks for, this script decides what runs, and it
# allows only three things, all read-only:
#
#   ping                                   -> prints "ok" (the daily job's login check)
#   the fixed zip listing                  -> exactly LISTING_CMD of ingestion/fetch_ib_test_logs.py
#   rsync --server --sender <flags> . <f>  -> download ONE file under /data/sdt/buildlogs that is a
#                                             test-log zip, its .checked marker or addOnTests.log
#
# Anything else is refused and logged to syslog (journalctl -t ibtestlogs-ro). Nothing here can
# write, delete or run another program. Install as /root/bin/ibtestlogs-readonly.sh, mode 0755,
# owned by root (see scripts/cmssdt9-02/README.md).
set -f
cmd=${SSH_ORIGINAL_COMMAND-}
ROOT=/data/sdt/buildlogs
LISTING="cd /data/sdt/buildlogs && find \$(ls -d */www/*/*/CMSSW_*/ | grep -v '^filtered/') -maxdepth 5 \\( -name new -o -name 'new_*' \\) -prune -o \\( -name pyRelValMatrixLogs.zip -o -name unitTestLogs.zip -o -name addOnTests.zip -o -name '*.zip.checked' -o -name addOnTests.log \\) -printf '%s %T@ %p\\n'"

refuse() {
    logger -t ibtestlogs-ro "refused from ${SSH_CLIENT%% *}: ${cmd:0:300}"
    echo "ibtestlogs-ro: command not allowed" >&2
    exit 1
}

if [ "$cmd" = "ping" ]; then
    echo ok
    exit 0
fi

if [ "$cmd" = "$LISTING" ]; then
    exec /bin/bash -c "$LISTING"
fi

re='^rsync --server --sender -[A-Za-z.]+ \. /data/sdt/buildlogs/[A-Za-z0-9_.+/-]+$'
if [[ $cmd =~ $re ]] && [[ $cmd != *..* ]]; then
    read -r -a parts <<< "$cmd"
    path=${parts[5]}   # rsync --server --sender <flags> . <path>
    case "$path" in
        */pyRelValMatrixLogs.zip|*/unitTestLogs.zip|*/addOnTests.zip|*.zip.checked|*/addOnTests.log) ;;
        *) refuse ;;
    esac
    # no symlink may lead out of the build-log tree
    real=$(realpath -e -- "$path" 2>/dev/null) || refuse
    case "$real" in "$ROOT"/*) ;; *) refuse ;; esac
    [ -f "$real" ] || refuse
    exec /usr/bin/rsync --server --sender "${parts[3]}" . "$real"
fi

refuse
