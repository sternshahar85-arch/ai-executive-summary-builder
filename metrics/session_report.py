"""
SessionStart hook entrypoint: syncs new metrics records from GCS, then prints
the trailing-30-day rollup table so it's automatically part of every Claude
Code session's context in this repo -- regardless of how long it's been since
the last session. Always exits 0; a reporting failure must never block
session startup.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import TRAILING_WINDOW_DAYS

LOCAL_NDJSON = os.path.join(os.path.dirname(__file__), "meetings.ndjson")


def main():
    try:
        from sync_from_gcs import sync
        _, sync_message = sync()
    except Exception as e:
        sync_message = f"Sync step failed unexpectedly ({e}); showing local data only."

    try:
        from rollup import load_records, trailing_window, render_table
        records = load_records(LOCAL_NDJSON)
        window = trailing_window(records, days=TRAILING_WINDOW_DAYS)
        table = render_table(window)
    except Exception as e:
        print(f"[Tamlelan Metrics] Could not compute rollup: {e}")
        return

    print(f"[Tamlelan Metrics] Trailing {TRAILING_WINDOW_DAYS} days ({len(window)} meeting(s)) -- {sync_message}")
    print(table)
    print(f"[Tamlelan Metrics] {len(records)} total record(s) in local history.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Tamlelan Metrics] Report unavailable this session: {e}")
    sys.exit(0)
