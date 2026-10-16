# Read-only key for the daily test-log copy (cmssdt9-02)

`scripts/daily_refresh.sh` copies IB test-log zips from `root@cmssdt9-02.cern.ch` every day
with no one at the keyboard. It never uses your Kerberos ticket, a password or a keytab for
that: it logs in with one dedicated key, and the server ties that key to a forced command,
`ibtestlogs-readonly.sh`. Whatever command is sent with the key, only these run:

| Request | What runs |
| --- | --- |
| `ping` | prints `ok` (the daily login check) |
| the fixed zip listing of `ingestion/fetch_ib_test_logs.py` (exact string match) | that `find` under `/data/sdt/buildlogs` |
| `rsync --server --sender <flags> . <file>` | a download of that one file, only if it is a test-log zip, its `.checked` marker or `addOnTests.log`, really inside `/data/sdt/buildlogs` (symlinks resolved) |

Everything else (a shell, `id`, chained commands with `;`, an upload, `--remove-source-files`,
a directory, `..`) is refused and logged: `journalctl -t ibtestlogs-ro`. The key entry also
has `restrict` (no terminal, no port/agent/X11 forwarding) and `from=` (only this PC's
addresses). If the key ever leaked, it could only read test-log zips from this PC.

## Install (once, as root on cmssdt9-02)

From this PC, with your normal (Kerberos) login:

```bash
cd /home/rao/CHAT-BOT-AI/NEO4J/cmssdt-kg-chatbot
ssh root@cmssdt9-02.cern.ch 'mkdir -p /root/bin'
scp scripts/cmssdt9-02/ibtestlogs-readonly.sh root@cmssdt9-02.cern.ch:/root/bin/ibtestlogs-readonly.sh
ssh root@cmssdt9-02.cern.ch 'chown root:root /root/bin/ibtestlogs-readonly.sh && chmod 0755 /root/bin/ibtestlogs-readonly.sh'
KEY=$(cat ~/.ssh/ibtestlogs_ro.pub)
ssh root@cmssdt9-02.cern.ch "umask 077; mkdir -p /root/.ssh; echo 'from=\"194.12.165.1,2001:1458:204:1::102:1f9e\",command=\"/root/bin/ibtestlogs-readonly.sh\",restrict $KEY' >> /root/.ssh/authorized_keys"
```

`from=` holds this PC's current IPv4 and IPv6 addresses (`ip -o addr show scope global`). The
PC is on dyndns, so if its address changes the server refuses the key: the daily test-log step
is then skipped with a clear line in `daily-refresh-logs/status.txt`; update the addresses in
`/root/.ssh/authorized_keys` to fix it.

## Check it

```bash
K="ssh -i ~/.ssh/ibtestlogs_ro -o IdentitiesOnly=yes -o GSSAPIAuthentication=no -o BatchMode=yes root@cmssdt9-02.cern.ch"
$K ping            # ok
$K id              # ibtestlogs-ro: command not allowed
IBTESTLOGS_SSH_KEY=~/.ssh/ibtestlogs_ro PYTHONPATH=ingestion venv/bin/python ingestion/fetch_ib_test_logs.py --dry-run | tail -3
```

## Remove

Delete the `ibtestlogs-readonly.sh` line from `/root/.ssh/authorized_keys` on cmssdt9-02 (and
`/root/bin/ibtestlogs-readonly.sh`), and `~/.ssh/ibtestlogs_ro*` here.

After changing `LISTING_CMD` in `ingestion/fetch_ib_test_logs.py`, copy the same string into
the wrapper's `LISTING=` line and re-install it, or the listing is refused.
