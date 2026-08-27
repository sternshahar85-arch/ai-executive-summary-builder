"""
Verifies Phase 5 Option B: self-reported topic linkage (Defect 1 mitigation).
key_topics is now an array of {topic_id, title, source_quote} objects; decisions
and action_items carry an optional related_topic_id. Checks: a valid link renders
with the linked topic's title, a fabricated/nonexistent topic_id reference is
flagged (referential integrity, no new Gemini call), and topic-level source_quote
grounding works the same way decisions/action_items already do.

Run with: .venv-cloud/Scripts/python.exe tests/test_main_topic_linkage.py
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


TRANSCRIPT_TEXT = (
    "Rachel: let's talk about the women's COBOL course recruitment with the banks. "
    "David: separately, the Ministry of Defense contact approves candidates for course 82. "
    "Rachel: those are two different things, not connected."
)

# Models the real logged Defect 1 case: two genuinely unrelated topics. A well-behaved
# model leaves related_topic_id null on both decisions. We ALSO include one decision
# with a fabricated related_topic_id (referencing a topic_id that doesn't exist) to
# simulate the failure mode the referential-integrity check must catch.
SUMMARY_FIXTURE = {
    "executive_summary": "Test summary",
    "attendees": [],
    "key_topics": [
        {"topic_id": "t1", "title": "COBOL course recruitment", "source_quote": "women's COBOL course recruitment with the banks"},
        {"topic_id": "t2", "title": "Ministry of Defense candidate approval", "source_quote": "Ministry of Defense contact approves candidates for course 82"},
    ],
    "decisions_log": [
        {
            "statement": "Proceed with COBOL course recruitment via the banks",
            "status": "decided",
            "hedge_note": None,
            "source_quote": "let's talk about the women's COBOL course recruitment with the banks",
            "related_topic_id": "t1",  # valid, real link
        },
        {
            "statement": "COBOL course is run in partnership with the Ministry of Defense",
            "status": "decided",
            "hedge_note": None,
            "source_quote": "those are two different things, not connected",
            "related_topic_id": "t99",  # fabricated topic_id -- does not exist in key_topics
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
    transcript_response.text = TRANSCRIPT_TEXT

    client.models.generate_content.side_effect = [summary_response, transcript_response]
    return client


class TestTopicLinkage(unittest.TestCase):
    def test_valid_link_renders_topic_title_fabricated_link_is_flagged(self):
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
            event = FakeCloudEvent("evt-topic-1", "fake-bucket", "meeting.wav")
            result = main.tamlelan_handler(event)

        self.assertEqual(result, ("Success", 200))
        summary_key = next(k for k in sent_docs if k.startswith("Summary_"))
        md = sent_docs[summary_key]

        # Both topics are genuinely grounded (their quotes appear in the transcript)
        self.assertNotIn("⚠ COBOL course recruitment", md)
        self.assertNotIn("⚠ Ministry of Defense candidate approval", md)

        # Decision 1: valid link to t1 -> shows the real topic title, not flagged
        d1_line = next(line for line in md.split("\n") if "Proceed with COBOL course recruitment" in line)
        self.assertIn("קשור לנושא: COBOL course recruitment", d1_line)
        self.assertNotIn("⚠", d1_line)

        # Decision 2: references t99, which does not exist among key_topics.
        # This is exactly the Defect 1 failure shape (a claimed cross-topic link
        # that isn't real) -- must be flagged, and must NOT resolve to a fake title.
        d2_line = next(line for line in md.split("\n")
                        if "COBOL course is run in partnership with the Ministry of Defense" in line)
        self.assertIn("⚠", d2_line)
        self.assertIn("הפניה לנושא לא תקין", d2_line)

        # Nothing dropped -- both decisions still present in the output
        self.assertIn("Proceed with COBOL course recruitment", md)
        self.assertIn("COBOL course is run in partnership with the Ministry of Defense", md)

    def test_topic_with_fabricated_quote_is_flagged(self):
        import main
        transcript = "We only discussed pricing today."
        fake_topic_quote = "we decided to merge with a competitor"
        self.assertFalse(main.is_grounded(fake_topic_quote, transcript))


if __name__ == "__main__":
    unittest.main(verbosity=2)
