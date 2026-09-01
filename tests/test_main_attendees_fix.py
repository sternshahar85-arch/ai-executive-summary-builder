"""
Verifies Phase 7 (Item 1): the attendees-over-inclusion prompt fix, the new
people_mentioned section, and the failed/ recursion guard.

Run with: .venv-cloud/Scripts/python.exe tests/test_main_attendees_fix.py
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
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(b"FAKE_AUDIO_BYTES")

    def download_as_bytes(self):
        return b"FAKE_AUDIO_BYTES"

    def upload_from_string(self, content, if_generation_match=None):
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
        self.state = MagicMock()
        self.state.name = "ACTIVE"


SUMMARY_FIXTURE = {
    "executive_summary": "Test summary",
    "attendees": [
        {"name": "Shachar", "role": None, "organization": None},
        {"name": "Sivan", "role": None, "organization": None},
    ],
    "people_mentioned": [
        {"name": "Elad", "context": "student receiving a certificate"},
        {"name": "Pinchas", "context": "to receive the updated syllabus file"},
    ],
    "key_topics": [],
    "decisions_log": [],
    "action_items": [],
    "diagram_needed": False,
}


def make_fake_genai_client():
    client = MagicMock()
    client.files.upload.return_value = FakeGeminiFile()
    client.files.get.return_value = FakeGeminiFile()
    client.files.delete.return_value = None
    client.caches.create.return_value.name = "cachedContents/fake123"

    summary_response = MagicMock()
    summary_response.text = json.dumps(SUMMARY_FIXTURE)

    transcript_response = MagicMock()
    transcript_response.text = "Full transcript text."

    client.models.generate_content.side_effect = [summary_response, transcript_response]
    return client


def run_handler(event_id="evt-1", file_name="meeting.wav"):
    """Shared harness: runs tamlelan_handler with everything mocked, returns
    (result, prompt_sent_to_pass1, rendered_summary_markdown)."""
    bucket = FakeBucket()
    sent_docs = {}

    def fake_send_urlopen(req, *args, **kwargs):
        payload = json.loads(req.data.decode())
        sent_docs[payload["filename"]] = payload["content"]
        fake_response = MagicMock()
        fake_response.read.return_value = json.dumps({"status": "ok"}).encode()
        cm = MagicMock()
        cm.__enter__.return_value = fake_response
        return cm

    fake_client = make_fake_genai_client()

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

    pass1_prompt = fake_client.models.generate_content.call_args_list[0].kwargs["contents"][0] \
        if fake_client.models.generate_content.call_args_list else None
    summary_md = next((v for k, v in sent_docs.items() if k.startswith("Summary_")), None)
    return result, pass1_prompt, summary_md, bucket


class TestAttendeesPromptFix(unittest.TestCase):
    def test_old_broad_instruction_gone_new_criteria_present(self):
        result, prompt, md, bucket = run_handler("evt-attendees-1")
        self.assertEqual(result, ("Success", 200))
        self.assertNotIn("list every person mentioned by name", prompt)
        self.assertIn("whose own voice you can hear speaking", prompt)
        self.assertIn("are NOT attendees", prompt)
        self.assertIn("do not guess", prompt)  # Phase 2 fix survives
        self.assertIn("People mentioned:", prompt)

    def test_pass1_still_first_call_pass2_still_second(self):
        # Guards against accidental call reordering, which would silently break
        # the side_effect list ordering this and other test files rely on.
        _, prompt, _, _ = run_handler("evt-attendees-2")
        self.assertIn("Attendees:", prompt)
        self.assertIn("Analyze this meeting audio", prompt)


class TestPeopleMentionedRendering(unittest.TestCase):
    def test_attendees_and_people_mentioned_render_in_separate_sections(self):
        result, _, md, _ = run_handler("evt-people-1")
        self.assertEqual(result, ("Success", 200))
        self.assertIn("## משתתפים", md)
        self.assertIn("## אנשים שהוזכרו", md)

        attendees_section = md.split("## משתתפים")[1].split("## אנשים שהוזכרו")[0]
        self.assertIn("Shachar", attendees_section)
        self.assertIn("Sivan", attendees_section)
        self.assertNotIn("Elad", attendees_section)
        self.assertNotIn("Pinchas", attendees_section)

        mentioned_section = md.split("## אנשים שהוזכרו")[1]
        self.assertIn("Elad", mentioned_section)
        self.assertIn("student receiving a certificate", mentioned_section)
        self.assertIn("Pinchas", mentioned_section)


class TestFailedRecursionGuard(unittest.TestCase):
    def test_failed_prefix_ignored_zero_gemini_cost(self):
        fake_client = make_fake_genai_client()
        bucket = FakeBucket()
        with patch("main.storage.Client") as mock_storage_client, \
             patch("main.genai.Client", return_value=fake_client):
            mock_storage_client.return_value.bucket.return_value = bucket
            os.environ["GEMINI_API_KEY"] = "fake_key"
            import main
            import importlib
            importlib.reload(main)
            event = FakeCloudEvent("evt-failed-1", "fake-bucket", "failed/meeting.wav")
            result = main.tamlelan_handler(event)

        self.assertEqual(result, ("Ignored", 200))
        fake_client.models.generate_content.assert_not_called()

    def test_locks_prefix_still_ignored_regression(self):
        fake_client = make_fake_genai_client()
        bucket = FakeBucket()
        with patch("main.storage.Client") as mock_storage_client, \
             patch("main.genai.Client", return_value=fake_client):
            mock_storage_client.return_value.bucket.return_value = bucket
            os.environ["GEMINI_API_KEY"] = "fake_key"
            import main
            import importlib
            importlib.reload(main)
            event = FakeCloudEvent("evt-locks-1", "fake-bucket", "locks/somelock.lock")
            result = main.tamlelan_handler(event)

        self.assertEqual(result, ("Ignored", 200))
        fake_client.models.generate_content.assert_not_called()

    def test_legitimate_wav_still_processed_regression(self):
        result, _, _, _ = run_handler("evt-legit-1", "meeting.wav")
        self.assertEqual(result, ("Success", 200))


if __name__ == "__main__":
    unittest.main(verbosity=2)
