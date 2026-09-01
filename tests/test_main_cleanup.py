"""
Verifies the Phase 1 fix to cloud/main.py's cleanup logic: the source audio blob
must be deleted on success, but preserved (moved to failed/<name>) on failure --
not deleted unconditionally as it was before the fix (which caused a real,
confirmed loss of a meeting recording on 2026-08-12, see
RECENT/Tamlelan_Implementation_Plan_26-8-26.md, Phase 1).

Run with: .venv-cloud/Scripts/python.exe tests/test_main_cleanup.py
"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cloud"))


class FakeCloudEvent:
    def __init__(self, event_id, bucket, name):
        self._id = event_id
        self.data = {"bucket": bucket, "name": name}

    def __getitem__(self, key):
        if key == "id":
            return self._id
        raise KeyError(key)


class FakeBlob:
    """Tracks delete()/exists()/upload_from_string() calls per blob path."""
    def __init__(self, bucket, path):
        self.bucket = bucket
        self.path = path
        self.deleted = False
        self.uploaded_content = None
        self.crc32c = None
        self.md5_hash = None

    def reload(self):
        pass

    def exists(self):
        return not self.deleted

    def delete(self):
        self.deleted = True
        self.bucket.deleted_paths.append(self.path)

    def download_to_filename(self, local_path):
        with open(local_path, "wb") as f:
            f.write(b"FAKE_AUDIO_BYTES")

    def download_as_bytes(self):
        return b"FAKE_AUDIO_BYTES"

    def upload_from_string(self, content, if_generation_match=None):
        self.uploaded_content = content
        self.bucket.uploaded_paths.append(self.path)


class FakeBucket:
    def __init__(self):
        self.blobs = {}
        self.deleted_paths = []
        self.uploaded_paths = []

    def blob(self, path):
        if path not in self.blobs:
            self.blobs[path] = FakeBlob(self, path)
        return self.blobs[path]


class FakeGeminiFile:
    def __init__(self, name="files/fake123"):
        self.name = name
        self.state = MagicMock(name="ACTIVE")
        self.state.name = "ACTIVE"


def make_fake_genai_client(summary_ok=True):
    """summary_ok=False simulates a failure during Pass 1 (invalid JSON)."""
    client = MagicMock()
    client.files.upload.return_value = FakeGeminiFile()
    client.files.get.return_value = FakeGeminiFile()
    client.files.delete.return_value = None
    client.caches.create.return_value.name = "cachedContents/fake123"

    summary_json = json.dumps({
        "executive_summary": "Test summary",
        "key_topics": ["Topic A"],
        "decisions_log": ["Decision A"],
        "action_items": ["Do the thing"],
        "diagram_needed": False,
    })
    summary_response = MagicMock()
    summary_response.text = summary_json if summary_ok else "NOT VALID JSON {{{"

    transcript_response = MagicMock()
    transcript_response.text = "Full transcript text."

    client.models.generate_content.side_effect = [summary_response, transcript_response]
    return client


class TestCleanupOnSuccess(unittest.TestCase):
    def test_source_blob_deleted_not_moved_to_failed(self):
        bucket = FakeBucket()

        with patch("main.storage.Client") as mock_storage_client, \
             patch("main.genai.Client", return_value=make_fake_genai_client(summary_ok=True)), \
             patch("main.urllib.request.urlopen") as mock_urlopen:

            mock_storage_client.return_value.bucket.return_value = bucket

            fake_response = MagicMock()
            fake_response.read.return_value = json.dumps({"status": "ok"}).encode()
            mock_urlopen.return_value.__enter__.return_value = fake_response

            os.environ["APPS_SCRIPT_URL"] = "https://example.invalid/webhook"
            os.environ["DRIVE_FOLDER_ID"] = "fake_folder"
            os.environ["WEBHOOK_SECRET"] = "fake_secret"
            os.environ["GEMINI_API_KEY"] = "fake_key"

            import main
            event = FakeCloudEvent("evt-success-1", "fake-bucket", "meeting.wav")
            result = main.tamlelan_handler(event)

        self.assertEqual(result, ("Success", 200))
        self.assertIn("meeting.wav", bucket.deleted_paths)
        failed_uploads = [p for p in bucket.uploaded_paths if p.startswith("failed/")]
        self.assertEqual(failed_uploads, [],
                          "No blob should be uploaded to failed/ on the success path")
        self.assertNotIn("failed/meeting.wav", bucket.blobs)


class TestCleanupOnFailure(unittest.TestCase):
    def test_source_blob_preserved_under_failed_prefix_not_deleted_only(self):
        bucket = FakeBucket()

        with patch("main.storage.Client") as mock_storage_client, \
             patch("main.genai.Client", return_value=make_fake_genai_client(summary_ok=False)), \
             patch("main.urllib.request.urlopen") as mock_urlopen:

            mock_storage_client.return_value.bucket.return_value = bucket

            fake_response = MagicMock()
            fake_response.read.return_value = json.dumps({"status": "ok"}).encode()
            mock_urlopen.return_value.__enter__.return_value = fake_response

            os.environ["APPS_SCRIPT_URL"] = "https://example.invalid/webhook"
            os.environ["DRIVE_FOLDER_ID"] = "fake_folder"
            os.environ["WEBHOOK_SECRET"] = "fake_secret"
            os.environ["GEMINI_API_KEY"] = "fake_key"

            import main
            event = FakeCloudEvent("evt-failure-1", "fake-bucket", "meeting.wav")

            with self.assertRaises(json.JSONDecodeError):
                main.tamlelan_handler(event)

        # The core assertion: the source is deleted from its original path
        # (deleted_paths includes it) BUT a copy was preserved under failed/
        # first -- this is the fix. Before the fix, only the unconditional
        # delete happened with no failed/ copy at all.
        self.assertIn("meeting.wav", bucket.deleted_paths)
        self.assertIn("failed/meeting.wav", bucket.uploaded_paths)
        self.assertEqual(
            bucket.blobs["failed/meeting.wav"].uploaded_content,
            b"FAKE_AUDIO_BYTES",
            "The failed/ copy must contain the actual audio bytes, not be empty",
        )


class TestCacheCreationFallback(unittest.TestCase):
    """If client.caches.create() fails (e.g. audio too short to meet the cache's
    minimum token count, or a transient API error), the pipeline must still
    succeed by falling back to passing the audio file directly into each pass --
    never fail because of the caching optimization itself."""

    def test_pipeline_succeeds_when_cache_creation_raises(self):
        bucket = FakeBucket()
        client = make_fake_genai_client(summary_ok=True)
        client.caches.create.side_effect = Exception("cache too small")

        with patch("main.storage.Client") as mock_storage_client, \
             patch("main.genai.Client", return_value=client), \
             patch("main.urllib.request.urlopen") as mock_urlopen:

            mock_storage_client.return_value.bucket.return_value = bucket

            fake_response = MagicMock()
            fake_response.read.return_value = json.dumps({"status": "ok"}).encode()
            mock_urlopen.return_value.__enter__.return_value = fake_response

            os.environ["APPS_SCRIPT_URL"] = "https://example.invalid/webhook"
            os.environ["DRIVE_FOLDER_ID"] = "fake_folder"
            os.environ["WEBHOOK_SECRET"] = "fake_secret"
            os.environ["GEMINI_API_KEY"] = "fake_key"

            import main
            event = FakeCloudEvent("evt-cache-fallback-1", "fake-bucket", "meeting.wav")
            result = main.tamlelan_handler(event)

        self.assertEqual(result, ("Success", 200))
        self.assertIn("meeting.wav", bucket.deleted_paths)

        # Both passes must have fallen back to passing the Gemini file object
        # directly, with no cached_content reference, since the cache was never
        # created.
        for call in client.models.generate_content.call_args_list:
            self.assertIsNone(call.kwargs["config"].cached_content)
            self.assertEqual(len(call.kwargs["contents"]), 2,
                              "Fallback path must include the gemini_file alongside the prompt")

        # The cache-delete cleanup step must not blow up when there was never a
        # cache to delete.
        client.caches.delete.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
