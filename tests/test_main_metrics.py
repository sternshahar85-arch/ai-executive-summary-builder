"""
Verifies the per-meeting structured metrics record written to
gs://<bucket>/metrics/<event_id>.json in cloud/main.py's cleanup finally block
(the meeting-metrics tracking feature). Covers the happy path, the
cache-creation-failed fallback, the Pass-1-failure path, and confirms a
metrics-write failure can never turn a successful pipeline run into a failure.

Run with: .venv-cloud/Scripts/python.exe tests/test_main_metrics.py
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
    """A real, minimal, valid mono 16-bit PCM WAV so wave.open() succeeds."""
    import io
    n_frames = int(duration_sec * frame_rate)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(frame_rate)
        wf.writeframes(struct.pack("<%dh" % n_frames, *([0] * n_frames)))
    return buf.getvalue()


WAV_BYTES = make_wav_bytes(duration_sec=2.0, frame_rate=16000)


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

    def copy_blob(self, source_blob, destination_bucket, new_name):
        """Server-side copy, as used by main.py's failed/ preservation path."""
        dest = destination_bucket.blob(new_name)
        dest.uploaded_content = source_blob.download_as_bytes()
        dest.deleted = False
        if hasattr(dest, "_exists"):
            dest._exists = True
        destination_bucket.uploaded_paths.append(new_name)
        return dest

    def seed_companion(self, wav_name, diar_dict):
        companion_path = wav_name[:-4] + ".diarization.json"
        blob = self.blob(companion_path)
        blob._exists = True
        blob._content = json.dumps(diar_dict).encode()
        return blob

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


def make_fake_genai_client(summary_ok=True, cache_fails=False):
    client = MagicMock()
    client.files.upload.return_value = FakeGeminiFile()
    client.files.get.return_value = FakeGeminiFile()
    client.files.delete.return_value = None

    if cache_fails:
        client.caches.create.side_effect = Exception("cache too small")
    else:
        client.caches.create.return_value.name = "cachedContents/fake123"
        client.caches.create.return_value.usage_metadata = make_usage(None, None, None, 9001)

    summary_json = json.dumps({
        "executive_summary": "Test summary", "attendees": [], "people_mentioned": [],
        "key_topics": [], "decisions_log": [], "action_items": [],
        "diagram_needed": False,
    })
    summary_response = MagicMock()
    summary_response.text = summary_json if summary_ok else "NOT VALID JSON {{{"
    summary_response.usage_metadata = make_usage(1000, 500, 200, 1200)

    transcript_response = MagicMock()
    transcript_response.text = "Full transcript text."
    transcript_response.usage_metadata = make_usage(1000, 500, 800, 1800)

    client.models.generate_content.side_effect = [summary_response, transcript_response]
    return client


STEREO_DIAR = {
    "schema_version": 1, "channel_mode": "stereo_operator_left", "sample_rate": 16000,
    "speaker_count": 3,
    "speakers": [{"label": "OPERATOR", "channel": "left"}],
    "segments": [[0.0, 1.0, "OPERATOR"]],
}


def run_handler(bucket, client, event_id="evt-metrics-1", file_name="meeting.wav"):
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


class TestMetricsHappyPath(unittest.TestCase):
    def test_record_written_with_expected_fields(self):
        bucket = FakeBucket()
        bucket.seed_companion("meeting.wav", STEREO_DIAR)
        client = make_fake_genai_client()
        result = run_handler(bucket, client, "evt-metrics-happy")

        self.assertEqual(result, ("Success", 200))
        record = bucket.metrics_record("evt-metrics-happy")
        self.assertIsNotNone(record, "Expected a metrics/evt-metrics-happy.json object to exist")

        self.assertEqual(record["event_id"], "evt-metrics-happy")
        self.assertEqual(record["file_stem"], "meeting")
        self.assertTrue(record["success"])
        self.assertIsNone(record["error"])
        self.assertAlmostEqual(record["duration_sec"], 2.0, places=1)
        self.assertEqual(record["speaker_count"], 3)
        self.assertEqual(record["channel_mode"], "stereo_operator_left")
        # Explicit context caching was removed on 2026-09-03 when Pass 2 became
        # chunked: a single-use cache costs more than not caching at all. The field
        # is kept so historical records stay comparable.
        self.assertFalse(record["cache_used"])
        self.assertEqual(record["schema_version"], 2)
        self.assertIn("transcript_quality", record)
        self.assertIsNone(record["cache_write_tokens"])  # no cache is created any more
        self.assertFalse(record["diagram_generated"])
        self.assertEqual(record["usage"]["pass1_summary"], {
            "prompt_tokens": 1000, "cached_tokens": 500, "output_tokens": 200, "total_tokens": 1200,
        })
        self.assertEqual(record["usage"]["pass2_transcript"], {
            "prompt_tokens": 1000, "cached_tokens": 500, "output_tokens": 800, "total_tokens": 1800,
        })


class TestMetricsDiagramGeneration(unittest.TestCase):
    def test_diagram_usage_captured_when_diagram_generated(self):
        bucket = FakeBucket()
        bucket.seed_companion("meeting.wav", STEREO_DIAR)
        client = make_fake_genai_client()

        # diagram_needed=True this time, plus a third generate_content response
        # for the flash-lite diagram call (client.models.generate_content is
        # called Pass 1, Pass 2, then the diagram pass, in that order). Build
        # fresh response mocks rather than mutating make_fake_genai_client()'s
        # output -- once assigned, Mock.side_effect from a list becomes a
        # non-subscriptable iterator, so the originals can't be read back.
        summary_response = MagicMock()
        summary_response.text = json.dumps({
            "executive_summary": "Test summary", "attendees": [], "people_mentioned": [],
            "key_topics": [], "decisions_log": [], "action_items": [],
            "diagram_needed": True,
        })
        summary_response.usage_metadata = make_usage(1000, 500, 200, 1200)

        transcript_response = MagicMock()
        transcript_response.text = "Full transcript text."
        transcript_response.usage_metadata = make_usage(1000, 500, 800, 1800)

        diagram_response = MagicMock()
        diagram_response.text = "<html><body>diagram</body></html>"
        diagram_response.usage_metadata = make_usage(300, None, 150, 450)

        client.models.generate_content.side_effect = [summary_response, transcript_response, diagram_response]

        result = run_handler(bucket, client, "evt-metrics-diagram")

        self.assertEqual(result, ("Success", 200))
        record = bucket.metrics_record("evt-metrics-diagram")
        self.assertIsNotNone(record)
        self.assertTrue(record["diagram_generated"])
        self.assertEqual(record["usage"]["diagram_generation"], {
            "prompt_tokens": 300, "cached_tokens": None, "output_tokens": 150, "total_tokens": 450,
        })

    def test_diagram_usage_absent_when_no_diagram_generated(self):
        bucket = FakeBucket()
        bucket.seed_companion("meeting.wav", STEREO_DIAR)
        client = make_fake_genai_client()  # diagram_needed=False by default
        result = run_handler(bucket, client, "evt-metrics-nodiagram")

        self.assertEqual(result, ("Success", 200))
        record = bucket.metrics_record("evt-metrics-nodiagram")
        self.assertFalse(record["diagram_generated"])
        self.assertIsNone(record["usage"]["diagram_generation"])


class TestMetricsCacheFallback(unittest.TestCase):
    def test_cache_not_used_still_records_usage(self):
        bucket = FakeBucket()
        client = make_fake_genai_client(cache_fails=True)
        result = run_handler(bucket, client, "evt-metrics-nocache")

        self.assertEqual(result, ("Success", 200))
        record = bucket.metrics_record("evt-metrics-nocache")
        self.assertIsNotNone(record)
        self.assertFalse(record["cache_used"])
        self.assertIsNone(record["cache_write_tokens"])
        self.assertEqual(record["usage"]["pass1_summary"]["prompt_tokens"], 1000)


class TestMetricsFailurePath(unittest.TestCase):
    def test_pass1_failure_still_records_duration_and_error(self):
        bucket = FakeBucket()
        client = make_fake_genai_client(summary_ok=False)
        result = run_handler(bucket, client, "evt-metrics-fail")

        self.assertIsNone(result)  # tamlelan_handler re-raises on Pass 1 failure
        record = bucket.metrics_record("evt-metrics-fail")
        self.assertIsNotNone(record, "A metrics record should still be written on failure")
        self.assertFalse(record["success"])
        self.assertIsNotNone(record["error"])
        # Duration is captured before any Gemini call, so it survives a Pass 1 failure.
        self.assertAlmostEqual(record["duration_sec"], 2.0, places=1)
        self.assertIsNone(record["usage"]["pass2_transcript"])  # Pass 2 never ran


class TestMetricsWriteNeverBreaksPipeline(unittest.TestCase):
    def test_metrics_write_failure_does_not_affect_pipeline_result(self):
        bucket = FakeBucket()
        client = make_fake_genai_client()

        original_blob = bucket.blob
        def blob_that_fails_for_metrics(path):
            b = original_blob(path)
            if path.startswith("metrics/"):
                b.upload_from_string = MagicMock(side_effect=Exception("simulated GCS failure"))
            return b
        bucket.blob = blob_that_fails_for_metrics

        result = run_handler(bucket, client, "evt-metrics-writefail")
        self.assertEqual(result, ("Success", 200))


if __name__ == "__main__":
    unittest.main(verbosity=2)
