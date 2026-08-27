"""
Verifies Phase 3 grounding: source_quote fuzzy-matched against the Pass 2
transcript (no extra Gemini call), flagging likely-hallucinated claims with a
warning marker instead of dropping them.

Run with: .venv-cloud/Scripts/python.exe tests/test_main_grounding.py
"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cloud"))


class TestIsGroundedUnit(unittest.TestCase):
    """Direct tests of the fuzzy-match function -- no API, no mocking."""

    def test_verbatim_quote_is_grounded(self):
        import main
        transcript = "We discussed the budget and Rachel agreed to send the proposal by Friday."
        self.assertTrue(main.is_grounded("Rachel agreed to send the proposal", transcript))

    def test_fabricated_quote_is_not_grounded(self):
        import main
        transcript = "We discussed the budget and Rachel agreed to send the proposal by Friday."
        self.assertFalse(main.is_grounded(
            "The company will merge with a competitor next year", transcript))

    def test_minor_rewording_still_grounded(self):
        import main
        transcript = "We discussed the budget and Rachel agreed to send the proposal by Friday."
        # Small paraphrase/typo-level difference should still pass the fuzzy threshold
        self.assertTrue(main.is_grounded("Rachel agreed to send proposal by Friday", transcript))

    def test_missing_quote_is_treated_as_grounded_nothing_to_check(self):
        import main
        transcript = "Anything at all."
        self.assertTrue(main.is_grounded(None, transcript))
        self.assertTrue(main.is_grounded("", transcript))


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


TRANSCRIPT_TEXT = (
    "Rachel: I think we should go with vendor X for the COBOL course. "
    "David: agreed, let's confirm that. "
    "Rachel: I'll schedule a meeting with Shoshi and Israel about it next week."
)

SUMMARY_FIXTURE = {
    "executive_summary": "Test summary",
    "attendees": [],
    "key_topics": [],
    "decisions_log": [
        {
            "statement": "Go with vendor X for the COBOL course",
            "status": "decided",
            "hedge_note": None,
            "source_quote": "we should go with vendor X for the COBOL course",
        },
        {
            "statement": "The course will be run jointly with the Ministry of Defense",
            "status": "decided",
            "hedge_note": None,
            "source_quote": "the course will be run jointly with the Ministry of Defense",
        },
    ],
    "action_items": [
        {
            "task": "Rachel to schedule a meeting with Shoshi and Israel",
            "owner": "Rachel",
            "deadline": None,
            "source_quote": "I'll schedule a meeting with Shoshi and Israel",
        },
    ],
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
    transcript_response.text = TRANSCRIPT_TEXT

    client.models.generate_content.side_effect = [summary_response, transcript_response]
    return client


class TestGroundingEndToEnd(unittest.TestCase):
    def test_ungrounded_decision_is_flagged_grounded_ones_are_not(self):
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
            importlib.reload(main)
            event = FakeCloudEvent("evt-ground-1", "fake-bucket", "meeting.wav")
            result = main.tamlelan_handler(event)

        self.assertEqual(result, ("Success", 200))
        summary_key = next(k for k in sent_docs if k.startswith("Summary_"))
        md = sent_docs[summary_key]

        # Decision 1 (vendor X) is genuinely in the transcript -- must NOT be flagged
        vendor_line = next(line for line in md.split("\n") if "vendor X" in line)
        self.assertNotIn("⚠", vendor_line)

        # Decision 2 (Ministry of Defense) is NOT in the transcript at all -- this is
        # exactly the Defect-1-style fabricated cross-topic claim; must be flagged
        mod_line = next(line for line in md.split("\n") if "Ministry of Defense" in line)
        self.assertIn("⚠", mod_line)

        # Action item's quote IS in the transcript -- must NOT be flagged
        action_line = next(line for line in md.split("\n") if "Shoshi and Israel" in line)
        self.assertNotIn("⚠", action_line)

        # Nothing was dropped -- the ungrounded decision still appears in the output
        self.assertIn("Ministry of Defense", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
