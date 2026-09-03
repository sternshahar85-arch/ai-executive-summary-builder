"""
Verifies Phase 8 (Item 2): speaker labels added to the Pass 2 transcript prompt,
and confirms the grounding pass (Phase 3) still works correctly against a
labeled-format transcript -- the real watch item called out when this change
was planned (labels inserted into the fuzzy-match haystack).

Run with: .venv-cloud/Scripts/python.exe tests/test_main_transcript_labels.py
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

    def upload_from_string(self, content, if_generation_match=None, content_type=None):
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


class FakeGeminiFile:
    def __init__(self, name="files/fake123"):
        self.name = name
        self.state = MagicMock()
        self.state.name = "ACTIVE"


# Modeled on the real transcript: a labeled-format Pass 2 response, with the
# quote genuinely present but embedded inside a speaker-labeled line.
LABELED_TRANSCRIPT = (
    "0:11 [שחר]: היי.\n"
    "0:14 [סיוון]: בוקר טוב.\n"
    "0:46 [סיוון]: היי, אני סיוון.\n"
    "0:48 [שחר]: יופי, אז מי שהציגה את עצמה עכשיו זאת סיוון ואני שחר.\n"
    "6:51 [סיוון]: אני צריכה סילבוס מעודכן של הקורס.\n"
)

SUMMARY_FIXTURE = {
    "executive_summary": "Test summary",
    "attendees": [{"name": "שחר", "role": None, "organization": None},
                  {"name": "סיוון", "role": None, "organization": None}],
    "people_mentioned": [],
    "key_topics": [],
    "decisions_log": [
        {
            "statement": "Sivan needs an updated syllabus",
            "status": "decided",
            "hedge_note": None,
            "source_quote": "אני צריכה סילבוס מעודכן של הקורס",
            "related_topic_id": None,
        },
    ],
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
    transcript_response.text = LABELED_TRANSCRIPT

    client.models.generate_content.side_effect = [summary_response, transcript_response]
    return client


def run_handler(event_id="evt-1"):
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
        event = FakeCloudEvent(event_id, "fake-bucket", "meeting.wav")
        result = main.tamlelan_handler(event)

    pass2_prompt = fake_client.models.generate_content.call_args_list[1].kwargs["contents"][0]
    summary_md = next((v for k, v in sent_docs.items() if k.startswith("Summary_")), None)
    return result, pass2_prompt, summary_md


class TestPass2PromptHasSpeakerLabeling(unittest.TestCase):
    def test_channel_instruction_and_format_spec_present(self):
        _, prompt, _ = run_handler("evt-labels-1")
        self.assertIn("LEFT channel is the meeting operator", prompt)
        self.assertIn("RIGHT channel is remote/system audio", prompt)
        self.assertIn("M:SS [SPEAKER]:", prompt)
        self.assertIn("NEVER invent or guess a name", prompt)
        self.assertIn("Do not summarize", prompt)  # original intent preserved

    def test_retroactive_naming_rule_present(self):
        _, prompt, _ = run_handler("evt-labels-2")
        self.assertIn("use that real name on EVERY", prompt)
        self.assertIn("including lines earlier in the meeting", prompt)


class TestGroundingSurvivesLabeledTranscript(unittest.TestCase):
    def test_genuine_quote_in_labeled_line_still_grounds(self):
        # This is the actual regression check: a source_quote that is genuinely
        # present, but now embedded inside a "M:SS [NAME]: " prefixed line,
        # must still pass threshold-80 fuzzy grounding -- confirming the label
        # insertion doesn't break Phase 3's fuzzy-match haystack.
        result, _, md = run_handler("evt-labels-3")
        self.assertEqual(result, ("Success", 200))
        decision_line = next(line for line in md.split("\n")
                              if "syllabus" in line or "סילבוס" in line)
        self.assertNotIn("⚠", decision_line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
