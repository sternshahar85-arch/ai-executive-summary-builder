"""
Pulls new per-meeting metrics records from gs://tamlelan-inbox-stgliding/metrics/
into the local, git-tracked metrics/meetings.ndjson -- the durable system of
record the user actually trusts, since Cloud Run can't write to this machine
directly and Cloud Logging here only retains 30 days.

Never raises: any failure (offline, credentials not configured, transient GCS
error) prints one clear line and leaves whatever's already on disk untouched.
Idempotent -- safe to run every session via the SessionStart hook.

Credentials: uses Application Default Credentials via the user's own
already-authenticated `gcloud` identity, NOT the write-only tamlelan-recorder
service account (which deliberately can't list/read). Requires a one-time:
    gcloud auth application-default login
"""
import json
import os
import sys

BUCKET_NAME = "tamlelan-inbox-stgliding"
METRICS_PREFIX = "metrics/"
LOCAL_NDJSON = os.path.join(os.path.dirname(__file__), "meetings.ndjson")


def _existing_event_ids(path):
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("event_id"):
                    ids.add(str(rec["event_id"]))
            except json.JSONDecodeError:
                continue
    return ids


def _load_all_records(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def sync():
    """Returns (new_count, message). Never raises."""
    try:
        from google.cloud import storage
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError as e:
        return 0, f"google-cloud-storage not available ({e}) -- run via .venv-cloud/Scripts/python.exe"

    existing_ids = _existing_event_ids(LOCAL_NDJSON)

    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        new_records = []
        for blob in client.list_blobs(bucket, prefix=METRICS_PREFIX):
            if not blob.name.endswith(".json"):
                continue
            event_id = blob.name[len(METRICS_PREFIX):-len(".json")]
            if event_id in existing_ids:
                continue
            try:
                content = blob.download_as_bytes()
                record = json.loads(content.decode("utf-8"))
                new_records.append(record)
            except Exception as e:
                print(f"[WARNING] Skipping unreadable metrics object {blob.name}: {e}")
    except DefaultCredentialsError:
        return 0, ("Metrics sync unavailable -- run 'gcloud auth application-default login' once "
                    "to enable it. Showing last locally synced data.")
    except Exception as e:
        return 0, f"Metrics sync failed ({e}) -- showing last locally synced data."

    if not new_records:
        return 0, "No new meetings since last sync."

    all_records = _load_all_records(LOCAL_NDJSON) + new_records
    all_records.sort(key=lambda r: r.get("finished_at") or "")

    tmp_path = LOCAL_NDJSON + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, LOCAL_NDJSON)  # atomic on Windows within the same volume

    return len(new_records), f"Pulled {len(new_records)} new meeting record(s)."


if __name__ == "__main__":
    count, message = sync()
    print(message)
    sys.exit(0)
