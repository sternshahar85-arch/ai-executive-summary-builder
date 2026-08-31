"""
Verifies metrics/sync_from_gcs.py: dedup against existing local records,
atomic write behavior, and graceful no-op when GCS is unreachable or
credentials aren't configured. Mocks google.cloud.storage the same way the
cloud-side tests mock it -- no real network calls.

Run with: .venv-cloud/Scripts/python.exe tests/test_metrics_sync.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "metrics"))


class FakeBlob:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def download_as_bytes(self):
        return self._content


class TestSyncFromGcs(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ndjson_path = os.path.join(self.tmpdir, "meetings.ndjson")
        # Reload sync_from_gcs pointed at our temp path each test.
        import sync_from_gcs
        import importlib
        importlib.reload(sync_from_gcs)
        sync_from_gcs.LOCAL_NDJSON = self.ndjson_path
        self.mod = sync_from_gcs

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pulls_new_records_into_local_file(self):
        remote = [
            FakeBlob("metrics/evt-a.json", json.dumps({"event_id": "evt-a", "finished_at": "2026-08-25T10:00:00Z"}).encode()),
            FakeBlob("metrics/evt-b.json", json.dumps({"event_id": "evt-b", "finished_at": "2026-08-26T10:00:00Z"}).encode()),
        ]
        fake_client = MagicMock()
        fake_client.list_blobs.return_value = remote

        with patch("google.cloud.storage.Client", return_value=fake_client):
            count, message = self.mod.sync()

        self.assertEqual(count, 2)
        self.assertTrue(os.path.exists(self.ndjson_path))
        with open(self.ndjson_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual({r["event_id"] for r in lines}, {"evt-a", "evt-b"})

    def test_dedups_against_existing_local_records(self):
        with open(self.ndjson_path, "w") as f:
            f.write(json.dumps({"event_id": "evt-a", "finished_at": "2026-08-25T10:00:00Z"}) + "\n")

        remote = [
            FakeBlob("metrics/evt-a.json", json.dumps({"event_id": "evt-a", "finished_at": "2026-08-25T10:00:00Z"}).encode()),
            FakeBlob("metrics/evt-b.json", json.dumps({"event_id": "evt-b", "finished_at": "2026-08-26T10:00:00Z"}).encode()),
        ]
        fake_client = MagicMock()
        fake_client.list_blobs.return_value = remote

        with patch("google.cloud.storage.Client", return_value=fake_client):
            count, message = self.mod.sync()

        self.assertEqual(count, 1)  # only evt-b is new
        with open(self.ndjson_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 2)

    def test_no_new_records_is_a_clean_noop(self):
        fake_client = MagicMock()
        fake_client.list_blobs.return_value = []
        with patch("google.cloud.storage.Client", return_value=fake_client):
            count, message = self.mod.sync()
        self.assertEqual(count, 0)
        self.assertFalse(os.path.exists(self.ndjson_path))  # nothing written, no file created

    def test_missing_credentials_fails_soft(self):
        from google.auth.exceptions import DefaultCredentialsError
        with patch("google.cloud.storage.Client", side_effect=DefaultCredentialsError("no ADC")):
            count, message = self.mod.sync()
        self.assertEqual(count, 0)
        self.assertIn("application-default login", message)

    def test_transient_gcs_error_fails_soft(self):
        with patch("google.cloud.storage.Client", side_effect=Exception("network unreachable")):
            count, message = self.mod.sync()
        self.assertEqual(count, 0)
        self.assertIn("network unreachable", message)

    def test_unreadable_remote_object_is_skipped_not_fatal(self):
        remote = [
            FakeBlob("metrics/evt-bad.json", b"NOT VALID JSON {{{"),
            FakeBlob("metrics/evt-good.json", json.dumps({"event_id": "evt-good", "finished_at": "2026-08-26T10:00:00Z"}).encode()),
        ]
        fake_client = MagicMock()
        fake_client.list_blobs.return_value = remote
        with patch("google.cloud.storage.Client", return_value=fake_client):
            count, message = self.mod.sync()
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
