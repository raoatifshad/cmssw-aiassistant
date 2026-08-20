"""Generate and store a ShiftSummary node for every release series that has
at least two imported primary IBs, skipping series whose (previous_tag,
latest_tag) pair already has a stored ShiftSummary. This is what lets the
chat "shift_summary" capability answer instantly instead of running the
full RelVal/warning diff live on every question.

Invoke from cron / a systemd timer / manually -- this process never
schedules itself, matching how run_alerts.py, revalidate_feedback.py, and
build_graph.py are invoked rather than daemonized.

Example crontab entry (every 30 minutes, offset from the alert-evaluation job):
    5,35 * * * * cd /path/to/cmssdt-kg-chatbot && \\
        NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/generate_shift_summaries.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from server.api_server import _all_release_series_latest_and_previous, generate_shift_summary  # noqa: E402
from graph.tools import get_shift_summary, store_shift_summary  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate even if a ShiftSummary already exists")
    args = parser.parse_args()

    generated, skipped = 0, 0
    for latest_tag, previous_tag in _all_release_series_latest_and_previous():
        if not args.force and get_shift_summary(previous_tag, latest_tag):
            skipped += 1
            continue
        markdown = generate_shift_summary(latest_tag, previous_tag)
        store_shift_summary(previous_tag, latest_tag, markdown)
        generated += 1
        print(f"  generated ShiftSummary for {previous_tag} -> {latest_tag}")
    print(f"Shift summaries: {generated} generated, {skipped} already current.")


if __name__ == "__main__":
    main()
