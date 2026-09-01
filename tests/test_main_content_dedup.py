"""
Verifies the content-hash dedup guard added to cloud/main.py: the pre-existing
locks/{event_id}.lock mechanism only deduplicates retries of the SAME Eventarc
event -- it does nothing when the same audio content arrives as two genuinely
distinct GCS objects/events (e.g. a manual re-upload after a transient Gemini
error). This is exactly the mechanism behind two real duplicate metrics records
observed in production on 2026-08-31 (identical duration_sec/speaker_count/
cache_write_tokens under two different event_ids).

Covers: first-time processing writes a content_hashes/{key}.json marker on real
success; a second event with identical content is detected and skipped (no
Gemini call made) once that marker exists; a FAILED first attempt must not
write a marker, so a legitimate retry with the same content still runs for
real.

Run with: .venv-cloud/Scripts/python.exe tests/test_main_content_dedup.py
"""
import json
import os
import struct
import sys
import unittest
import wave
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cloud"))


def make_wav_bytes(duration_sec=2.0, frame_rate=16000):
    import io
    n_frames = int(duration_sec * frame_rate)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(frame_rate)
        wf.writeframes(struct.pack("<%dh" % n_frames, *([0] * n_frames)))
    return buf.getvalue()


WAV_BYTES = make_wav_bytes()


class FakeCloudEvent:
    def __init__(self, event_id, bucket, name):
        self._id = event_id
        self.data = {"bucket": bucket, "name": name}

    def __getitem__(self, key):
        if key == "id":
            return self._id
        raise KeyError(key)


class FakeBlob:
    def __init__(self, bucket, path):
        self.bucket = bucket
        self.path = path
        self.deleted = False
        self._exists = False
        self._content = None
        self.crc32c = None
        self.md5_hash = None

    def reload(self):
        pass

    def exists(self):
        return self._exists and not self.deleted

    def delete(self):
        self.deleted = True
        self.bucket.deleted_paths.append(self.path)

    def download_to_filename(self, local_path):
        with open(local_path, "wb") as f:
            f.write(WAV_BYTES)

    def download_as_bytes(self):
        return self._content if self._content is not None else b"FAKE_AUDIO_BYTES"

    def upload_from_string(self, content, if_generation_match=None, content_type=None):
        self._exists = True
        self._content = content.encode() if isinstance(content, str) else content
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

    def set_crc32c(self, wav_name, crc32c_value):
        self.blob(wav_name).crc32c = crc32c_value

    def seed_dedup_marker(self, crc32c_value, original_event_id):
        b = self.blob(f"content_hashes/{crc32c_value}.json")
        b._exists = True
        b._content = json.dumps({"event_id": original_event_id, "file_stem": "prior"}).encode()

    def metrics_record(self, event_id):
        blob = self.blobs.get(f"metrics/{event_id}.json")
        if blob is None or blob._content is None:
            return None
        return json.loads(blob._content.decode())


class FakeGeminiFile:
    def __init__(self, name="files/fake123"):
        self.name = name
        self.state = MagicMock()
        self.state.name = "ACTIVE"


def make_usage(prompt, cached, output, total):
    u = MagicMock()
    u.prompt_token_count = prompt
    u.cached_content_token_count = cached
    u.candidates_token_count = output
    u.total_token_count = total
    return u


def make_fake_genai_client(summary_ok=True):
    client = MagicMock()
    client.files.upload.return_value = FakeGeminiFile()
    client.files.get.return_value = FakeGeminiFile()
    client.files.delete.return_value = None
    client.caches.create.return_value.name = "cachedContents/fake123"
    client.caches.create.return_value.usage_metadata = make_usage(None, None, None, 9001)

    summary_response = MagicMock()
    summary_response.text = json.dumps({
        "executive_summary": "Test summary", "attendees": [], "people_mentioned": [],
        "key_topics": [], "decisions_log": [], "action_items": [], "diagram_needed": False,
    }) if summary_ok else "NOT VALID JSON {{{"
    summary_response.usage_metadata = make_usage(1000, 500, 200, 1200)

    transcript_response = MagicMock()
    transcript_response.text = "Full transcript text."
    transcript_response.usage_metadata = make_usage(1000, 500, 800, 1800)

    client.models.generate_content.side_effect = [summary_response, transcript_response]
    return client


def run_handler(bucket, client, event_id, file_name="meeting.wav"):
    def fake_send_urlopen(req, *args, **kwargs):
        fake_response = MagicMock()
        fake_response.read.return_value = json.dumps({"status": "ok"}).encode()
        cm = MagicMock()
        cm.__enter__.return_value = fake_response
        return cm

    with patch("main.storage.Client") as mock_storage_client, \
         patch("main.genai.Client", return_value=client), \
         patch("main.urllib.request.urlopen", side_effect=fake_send_urlopen):

        mock_storage_client.return_value.bucket.return_value = bucket
        os.environ["APPS_SCRIPT_URL"] = "https://example.invalid/webhook"
        os.environ["DRIVE_FOLDER_ID"] = "fake_folder"
        os.environ["WEBHOOK_SECRET"] = "fake_secret"
        os.environ["GEMINI_API_KEY"] = "fake_key"

        import main
        import importlib
        importlib.reload(main)
        event = FakeCloudEvent(event_id, "fake-bucket", file_name)
        try:
            return main.tamlelan_handler(event)
        except Exception:
            return None


class TestFirstTimeProcessingWritesMarker(unittest.TestCase):
    def test_marker_written_on_real_success(self):
        bucket = FakeBucket()
        bucket.set_crc32c("meeting.wav", "AAAA1111")
        client = make_fake_genai_client()

        result = run_handler(bucket, client, "evt-first-time")

        self.assertEqual(result, ("Success", 200))
        marker = bucket.blobs.get("content_hashes/AAAA1111.json")
        self.assertIsNotNone(marker, "Expected a content_hashes/AAAA1111.json marker to be written")
        self.assertTrue(marker._exists)
        marker_data = json.loads(marker._content.decode())
        self.assertEqual(marker_data["event_id"], "evt-first-time")

        record = bucket.metrics_record("evt-first-time")
        self.assertIsNone(record["duplicate_of_event_id"])


class TestDuplicateContentSkipped(unittest.TestCase):
    def test_duplicate_skips_pipeline_no_gemini_call(self):
        bucket = FakeBucket()
        bucket.set_crc32c("meeting2.wav", "AAAA1111")
        bucket.seed_dedup_marker("AAAA1111", original_event_id="evt-original")
        client = make_fake_genai_client()

        result = run_handler(bucket, client, "evt-duplicate", file_name="meeting2.wav")

        self.assertEqual(result, ("Duplicate Content - Already Processed", 200))
        self.assertFalse(client.models.generate_content.called,
                          "A detected duplicate must never reach the real Gemini pipeline")

        record = bucket.metrics_record("evt-duplicate")
        self.assertIsNotNone(record)
        self.assertTrue(record["success"])
        self.assertEqual(record["duplicate_of_event_id"], "evt-original")

    def test_duplicate_source_blob_still_cleaned_up(self):
        bucket = FakeBucket()
        bucket.set_crc32c("meeting3.wav", "BBBB2222")
        bucket.seed_dedup_marker("BBBB2222", original_event_id="evt-original-2")
        # Mark the duplicate .wav as actually present in the bucket (uploaded).
        bucket.blob("meeting3.wav")._exists = True
        client = make_fake_genai_client()

        run_handler(bucket, client, "evt-duplicate-2", file_name="meeting3.wav")

        self.assertIn("meeting3.wav", bucket.deleted_paths,
                       "The duplicate's source audio should still be cleaned up, not left orphaned")


class TestFailedAttemptDoesNotBlockRetry(unittest.TestCase):
    def test_failed_first_attempt_leaves_no_marker_real_retry_still_runs(self):
        bucket = FakeBucket()
        bucket.set_crc32c("meeting.wav", "CCCC3333")
        failing_client = make_fake_genai_client(summary_ok=False)

        first_result = run_handler(bucket, failing_client, "evt-fail-1")
        self.assertIsNone(first_result)  # tamlelan_handler re-raises on Pass 1 failure

        marker = bucket.blobs.get("content_hashes/CCCC3333.json")
        self.assertTrue(marker is None or not marker._exists,
                         "A failed run must never write a dedup marker")

        # A legitimate retry with the SAME content (crc32c) but a fresh event_id
        # must still run the real pipeline, not be treated as a duplicate.
        retry_client = make_fake_genai_client(summary_ok=True)
        second_result = run_handler(bucket, retry_client, "evt-retry-1")

        self.assertEqual(second_result, ("Success", 200))
        self.assertTrue(retry_client.models.generate_content.called,
                         "A retry after a failed attempt must reach the real Gemini pipeline")


if __name__ == "__main__":
    unittest.main(verbosity=2)
