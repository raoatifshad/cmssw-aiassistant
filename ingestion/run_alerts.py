"""Evaluate every active AlertRule once and exit. Invoke from cron / a
systemd timer / manually -- this process never schedules itself, matching
how build_graph.py is invoked rather than daemonized.

Example crontab entry (every 15 minutes):
    */15 * * * * cd /path/to/cmssdt-kg-chatbot && \\
        NEO4J_URI=bolt://localhost:7688 venv/bin/python ingestion/run_alerts.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))
from alerts import LogNotifier, evaluate_rules  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", default="alerts.log.jsonl", help="Where fired alerts are appended")
    args = parser.parse_args()

    fired = evaluate_rules(LogNotifier(args.log_path))
    print(f"Evaluated rules: {len(fired)} fired.")


if __name__ == "__main__":
    main()
