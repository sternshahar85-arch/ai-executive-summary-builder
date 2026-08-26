"""
Verifies the Phase 2 schema/prompt changes to cloud/main.py: attendees (Defect 5),
decision status + hedge_note (Defects 3 & 4), and nullable owner/deadline on action
items (Defect 6). Captures the rendered markdown via the real send_to_drive() call
(mocked) and asserts on its content.

Run with: .venv-cloud/Scripts/python.exe tests/test_main_render.py
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
        {"name": "Rachel", "role": "Course Lead", "organization": "Tech-Career"},
        {"name": "David", "role": None, "organization": None},
    ],
    "key_topics": ["Topic A"],
    "decisions_log": [
        {"statement": "We will use vendor X", "status": "decided", "hedge_note": None},
        {"statement": "Course length might expand later", "status": "proposed",
         "hedge_note": "speaker said 'I'd like it to eventually be 400 hours'"},
        {"statement": "Ministry of Defense partnership", "status": "open", "hedge_note": None},
    ],
    "action_items": [
        {"task": "Rachel to schedule a meeting", "owner": "Rachel", "deadline": None},
        {"task": "Send the proposal", "owner": None, "deadline": None},
    ],
    "diagram_needed": False,
}


def make_fake_genai_client():
    client = MagicMock()
    client.files.upload.return_value = FakeGeminiFile()
    client.files.get.return_value = FakeGeminiFile()
    client.files.delete.return_value = None

    summary_response = MagicMock()
    summary_response.text = json.dumps(SUMMARY_FIXTURE)

    transcript_response = MagicMock()
    transcript_response.text = "Full transcript text."

    client.models.generate_content.side_effect = [summary_response, transcript_response]
    return client


class TestPhase2Rendering(unittest.TestCase):
    def test_attendees_decisions_and_nullable_action_items_render_correctly(self):
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

        with patch("main.storage.Client") as mock_storage_client, \
             patch("main.genai.Client", return_value=make_fake_genai_client()), \
             patch("main.urllib.request.urlopen", side_effect=fake_send_urlopen):

            mock_storage_client.return_value.bucket.return_value = bucket

            os.environ["APPS_SCRIPT_URL"] = "https://example.invalid/webhook"
            os.environ["DRIVE_FOLDER_ID"] = "fake_folder"
            os.environ["WEBHOOK_SECRET"] = "fake_secret"
            os.environ["GEMINI_API_KEY"] = "fake_key"

            import main
            import importlib
            importlib.reload(main)  # pick up SUMMARY_FIXTURE-shaped schema/prompt changes cleanly
            event = FakeCloudEvent("evt-render-1", "fake-bucket", "meeting.wav")
            result = main.tamlelan_handler(event)

        self.assertEqual(result, ("Success", 200))
        summary_key = next(k for k in sent_docs if k.startswith("Summary_"))
        md = sent_docs[summary_key]

        # Never render the literal string "None" for a null field
        self.assertNotIn("None", md)

        # Defect 5: attendees with and without role/org
        self.assertIn("Rachel", md)
        self.assertIn("Course Lead", md)
        self.assertIn("Tech-Career", md)
        self.assertIn("David", md)

        # Defects 3 & 4: all three statuses render with their Hebrew label, hedge shown
        self.assertIn("הוחלט", md)   # decided
        self.assertIn("הוצע", md)    # proposed
        self.assertIn("פתוח", md)    # open
        self.assertIn("400 hours", md)  # hedge_note preserved

        # Defect 6: null owner/deadline render as the placeholder, not "None" or blank
        self.assertIn("Rachel to schedule a meeting", md)
        self.assertIn("Send the proposal", md)
        # count of '-' placeholder cells for the two null fields (owner of item 2, deadline of both)
        self.assertGreaterEqual(md.count("| - |"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
