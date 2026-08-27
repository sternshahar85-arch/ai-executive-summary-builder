"""
Verifies Phase 9's cloud-side diarization consumption: load_diarization,
format_diarization_for_prompt, prompt wiring, the attendee cross-check, and
companion-file cleanup. All of this must degrade gracefully to exact
pre-Phase-9 behavior when no companion file exists.

Run with: .venv-cloud/Scripts/python.exe tests/test_main_diarization.py
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
    def __init__(self, bucket, path):
        self.bucket = bucket
        self.path = path
        self.deleted = False
        self._exists = False
        self._content = None

    def exists(self):
        return self._exists and not self.deleted

    def delete(self):
        self.deleted = True
        self.bucket.deleted_paths.append(self.path)

    def download_to_filename(self, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(b"FAKE_AUDIO_BYTES")

    def download_as_bytes(self):
        if self._content is not None:
            return self._content
        return b"FAKE_AUDIO_BYTES"

    def upload_from_string(self, content, if_generation_match=None):
        self.bucket.uploaded_paths.append(self.path)
        self._exists = True
        self._content = content.encode() if isinstance(content, str) else content


class FakeBucket:
    def __init__(self):
        self.blobs = {}
        self.deleted_paths = []
        self.uploaded_paths = []

    def blob(self, path):
        if path not in self.blobs:
            self.blobs[path] = FakeBlob(self, path)
        return self.blobs[path]

    def seed_companion(self, wav_name, diar_dict):
        """Pre-populate a companion diarization file as if the client uploaded it."""
        companion_path = wav_name[:-4] + ".diarization.json"
        blob = self.blob(companion_path)
        blob._exists = True
        blob._content = json.dumps(diar_dict).encode()
        return blob


class FakeGeminiFile:
    def __init__(self, name="files/fake123"):
        self.name = name
        self.state = MagicMock()
        self.state.name = "ACTIVE"


STEREO_DIAR = {
    "schema_version": 1,
    "channel_mode": "stereo_operator_left",
    "sample_rate": 16000,
    "speaker_count": 3,
    "speakers": [
        {"label": "OPERATOR", "channel": "left"},
        {"label": "REMOTE_00", "channel": "right"},
        {"label": "REMOTE_01", "channel": "right"},
    ],
    "segments": [[0.0, 4.2, "OPERATOR"], [4.5, 9.1, "REMOTE_00"], [9.2, 12.0, "REMOTE_01"]],
}

MONO_DIAR = {
    "schema_version": 1,
    "channel_mode": "mono_single_track",
    "sample_rate": 16000,
    "speaker_count": 2,
    "speakers": [{"label": "SPEAKER_00", "channel": "mono"}, {"label": "SPEAKER_01", "channel": "mono"}],
    "segments": [[0.0, 3.0, "SPEAKER_00"], [3.1, 6.0, "SPEAKER_01"]],
}


def base_summary_fixture(attendee_count=2):
    return {
        "executive_summary": "Test summary",
        "attendees": [{"name": f"Person{i}", "role": None, "organization": None} for i in range(attendee_count)],
        "people_mentioned": [],
        "key_topics": [],
        "decisions_log": [],
        "action_items": [],
        "diagram_needed": False,
    }


def make_fake_genai_client(summary_fixture):
    client = MagicMock()
    client.files.upload.return_value = FakeGeminiFile()
    client.files.get.return_value = FakeGeminiFile()
    client.files.delete.return_value = None

    summary_response = MagicMock()
    summary_response.text = json.dumps(summary_fixture)
    transcript_response = MagicMock()
    transcript_response.text = "Full transcript text."

    client.models.generate_content.side_effect = [summary_response, transcript_response]
    return client


def run_handler(bucket, summary_fixture, event_id="evt-1", file_name="meeting.wav"):
    sent_docs = {}

    def fake_send_urlopen(req, *args, **kwargs):
        payload = json.loads(req.data.decode())
        sent_docs[payload["filename"]] = payload["content"]
        fake_response = MagicMock()
        fake_response.read.return_value = json.dumps({"status": "ok"}).encode()
        cm = MagicMock()
        cm.__enter__.return_value = fake_response
        return cm

    fake_client = make_fake_genai_client(summary_fixture)

    with patch("main.storage.Client") as mock_storage_client, \
         patch("main.genai.Client", return_value=fake_client), \
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
        result = main.tamlelan_handler(event)

    pass1_prompt = fake_client.models.generate_content.call_args_list[0].kwargs["contents"][0]
    pass2_prompt = fake_client.models.generate_content.call_args_list[1].kwargs["contents"][0]
    summary_md = next((v for k, v in sent_docs.items() if k.startswith("Summary_")), None)
    return result, pass1_prompt, pass2_prompt, summary_md, bucket


class TestFormatDiarizationForPrompt(unittest.TestCase):
    def test_none_returns_empty_string(self):
        import main
        self.assertEqual(main.format_diarization_for_prompt(None), "")

    def test_stereo_wording(self):
        import main
        block = main.format_diarization_for_prompt(STEREO_DIAR)
        self.assertIn("OPERATOR is the left channel", block)
        self.assertIn("3 distinct speaking voices", block)
        self.assertIn("0:04-0:09 REMOTE_00", block)

    def test_mono_wording(self):
        import main
        block = main.format_diarization_for_prompt(MONO_DIAR)
        self.assertIn("single in-room microphone", block)
        self.assertNotIn("left channel", block)

    def test_over_cap_omits_segment_list_keeps_roster(self):
        import main
        big_diar = dict(STEREO_DIAR)
        big_diar["segments"] = [[float(i), float(i) + 1, "OPERATOR"] for i in range(main.MAX_PROMPT_SEGMENTS + 1)]
        block = main.format_diarization_for_prompt(big_diar)
        self.assertIn("3 distinct speaking voices", block)
        self.assertIn("omitted", block)
        self.assertNotIn("0:00-0:01 OPERATOR", block)


class TestLoadDiarizationNeverRaises(unittest.TestCase):
    def test_missing_companion_returns_none(self):
        import main
        bucket = FakeBucket()
        self.assertIsNone(main.load_diarization(bucket, "meeting.wav"))

    def test_invalid_json_returns_none(self):
        import main
        bucket = FakeBucket()
        blob = bucket.blob("meeting.diarization.json")
        blob._exists = True
        blob._content = b"NOT VALID JSON {{{"
        self.assertIsNone(main.load_diarization(bucket, "meeting.wav"))

    def test_wrong_schema_version_returns_none(self):
        import main
        bucket = FakeBucket()
        bad = dict(STEREO_DIAR)
        bad["schema_version"] = 99
        bucket.seed_companion("meeting.wav", bad)
        self.assertIsNone(main.load_diarization(bucket, "meeting.wav"))

    def test_segments_not_a_list_returns_none(self):
        import main
        bucket = FakeBucket()
        bad = dict(STEREO_DIAR)
        bad["segments"] = "not a list"
        bucket.seed_companion("meeting.wav", bad)
        self.assertIsNone(main.load_diarization(bucket, "meeting.wav"))

    def test_non_wav_filename_returns_none(self):
        import main
        bucket = FakeBucket()
        self.assertIsNone(main.load_diarization(bucket, "locks/x.lock"))


class TestEndToEndGracefulDegradation(unittest.TestCase):
    def test_with_companion_both_prompts_get_speaker_data(self):
        bucket = FakeBucket()
        bucket.seed_companion("meeting.wav", STEREO_DIAR)
        result, p1, p2, md, _ = run_handler(bucket, base_summary_fixture(2), "evt-diar-1")
        self.assertEqual(result, ("Success", 200))
        self.assertIn("SPEAKER TURN DATA", p1)
        self.assertIn("SPEAKER TURN DATA", p2)

    def test_without_companion_neither_prompt_mentions_speaker_data(self):
        bucket = FakeBucket()  # no companion seeded
        result, p1, p2, md, _ = run_handler(bucket, base_summary_fixture(2), "evt-diar-2")
        self.assertEqual(result, ("Success", 200))
        self.assertNotIn("SPEAKER TURN DATA", p1)
        self.assertNotIn("SPEAKER TURN DATA", p2)


class TestAttendeeCrossCheck(unittest.TestCase):
    def test_mismatch_flagged(self):
        bucket = FakeBucket()
        bucket.seed_companion("meeting.wav", STEREO_DIAR)  # speaker_count=3
        _, _, _, md, _ = run_handler(bucket, base_summary_fixture(16), "evt-mismatch-1")
        header_line = next(line for line in md.split("\n") if "משתתפים" in line and line.startswith("##"))
        self.assertIn("⚠", header_line)

    def test_match_not_flagged(self):
        bucket = FakeBucket()
        bucket.seed_companion("meeting.wav", STEREO_DIAR)  # speaker_count=3
        _, _, _, md, _ = run_handler(bucket, base_summary_fixture(2), "evt-mismatch-2")
        header_line = next(line for line in md.split("\n") if "משתתפים" in line and line.startswith("##"))
        self.assertNotIn("⚠", header_line)

    def test_no_companion_never_flags(self):
        bucket = FakeBucket()
        _, _, _, md, _ = run_handler(bucket, base_summary_fixture(16), "evt-mismatch-3")
        header_line = next(line for line in md.split("\n") if "משתתפים" in line and line.startswith("##"))
        self.assertNotIn("⚠", header_line)


class TestCompanionCleanup(unittest.TestCase):
    def test_companion_deleted_on_success(self):
        bucket = FakeBucket()
        bucket.seed_companion("meeting.wav", STEREO_DIAR)
        run_handler(bucket, base_summary_fixture(2), "evt-cleanup-1")
        self.assertIn("meeting.diarization.json", bucket.deleted_paths)
        self.assertNotIn("failed/meeting.diarization.json", bucket.uploaded_paths)

    def test_companion_preserved_under_failed_on_failure(self):
        bucket = FakeBucket()
        bucket.seed_companion("meeting.wav", STEREO_DIAR)
        bad_fixture = "NOT VALID JSON {{{"  # forces json.JSONDecodeError -> failure path

        fake_client = MagicMock()
        fake_client.files.upload.return_value = FakeGeminiFile()
        fake_client.files.get.return_value = FakeGeminiFile()
        fake_client.files.delete.return_value = None
        bad_response = MagicMock()
        bad_response.text = bad_fixture
        fake_client.models.generate_content.side_effect = [bad_response]

        with patch("main.storage.Client") as mock_storage_client, \
             patch("main.genai.Client", return_value=fake_client):
            mock_storage_client.return_value.bucket.return_value = bucket
            os.environ["GEMINI_API_KEY"] = "fake_key"
            import main
            import importlib
            importlib.reload(main)
            event = FakeCloudEvent("evt-cleanup-2", "fake-bucket", "meeting.wav")
            with self.assertRaises(json.JSONDecodeError):
                main.tamlelan_handler(event)

        self.assertIn("failed/meeting.diarization.json", bucket.uploaded_paths)
        self.assertIn("meeting.diarization.json", bucket.deleted_paths)


class TestCompanionFileIgnoredByTrigger(unittest.TestCase):
    def test_diarization_json_short_circuits_zero_cost(self):
        fake_client = MagicMock()
        bucket = FakeBucket()
        with patch("main.storage.Client") as mock_storage_client, \
             patch("main.genai.Client", return_value=fake_client):
            mock_storage_client.return_value.bucket.return_value = bucket
            os.environ["GEMINI_API_KEY"] = "fake_key"
            import main
            import importlib
            importlib.reload(main)
            event = FakeCloudEvent("evt-companion-1", "fake-bucket", "meeting.diarization.json")
            result = main.tamlelan_handler(event)
        self.assertEqual(result, ("Ignored", 200))
        fake_client.models.generate_content.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
