"""
Verifies Phase 9's client-side diarization: merge_speaker_segments,
build_diarization_payload (channel-selection logic), diarize_channel's dtype
handling, and upload_to_gcp's companion-file ordering/cleanup.

Must pass WITHOUT sherpa-onnx installed -- per the existing test_scribe_audio.py
model, only _make_diarizer/diarize_channel touch that import, and this file
patches those out rather than exercising the real library.

Run with: python tests/test_scribe_diarization.py
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import scribe


class TestMergeSpeakerSegments(unittest.TestCase):
    def test_adjacent_same_label_within_gap_merges(self):
        segs = [(0.0, 2.0, "A"), (2.3, 4.0, "A")]  # 0.3s gap < default 0.8
        merged = scribe.merge_speaker_segments(segs)
        self.assertEqual(merged, [(0.0, 4.0, "A")])

    def test_same_label_beyond_gap_does_not_merge(self):
        segs = [(0.0, 2.0, "A"), (5.0, 7.0, "A")]  # 3.0s gap > 0.8
        merged = scribe.merge_speaker_segments(segs)
        self.assertEqual(merged, [(0.0, 2.0, "A"), (5.0, 7.0, "A")])

    def test_different_labels_never_merge(self):
        segs = [(0.0, 2.0, "A"), (2.1, 4.0, "B")]
        merged = scribe.merge_speaker_segments(segs)
        self.assertEqual(len(merged), 2)

    def test_empty_input_returns_empty(self):
        self.assertEqual(scribe.merge_speaker_segments([]), [])

    def test_unsorted_input_handled(self):
        segs = [(5.0, 7.0, "B"), (0.0, 2.0, "A")]
        merged = scribe.merge_speaker_segments(segs)
        self.assertEqual(merged, [(0.0, 2.0, "A"), (5.0, 7.0, "B")])


class TestBuildDiarizationPayload(unittest.TestCase):
    def test_stereo_calls_diarizer_twice_num_clusters_1_and_minus1(self):
        calls = []

        def fake_diarize_channel(samples, label_prefix, num_clusters=-1):
            calls.append((label_prefix, num_clusters))
            if label_prefix == "OPERATOR":
                return [(0.0, 5.0, "OPERATOR00")]
            return [(0.0, 3.0, "REMOTE_00"), (3.5, 6.0, "REMOTE_01")]

        with patch("scribe.diarize_channel", side_effect=fake_diarize_channel):
            payload = scribe.build_diarization_payload(
                np.zeros(100), np.zeros(100), has_loopback=True, duration_sec=6.0)

        self.assertEqual(len(calls), 2)
        self.assertIn(("OPERATOR", 1), calls)
        self.assertIn(("REMOTE_", -1), calls)
        self.assertEqual(payload["channel_mode"], "stereo_operator_left")
        self.assertIsNotNone(payload)

        # Regression check: diarize_channel's raw f"{prefix}{speaker:02d}"
        # formatting would produce "OPERATOR00", not "OPERATOR" -- the operator
        # channel must be relabeled to the plain "OPERATOR" string (num_clusters=1
        # guarantees exactly one label on that channel), or channel tagging below
        # silently breaks since it depends on an exact "OPERATOR" string match.
        operator_entry = next(s for s in payload["speakers"] if "OPERATOR" in s["label"])
        self.assertEqual(operator_entry["label"], "OPERATOR")
        self.assertEqual(operator_entry["channel"], "left")
        remote_entries = [s for s in payload["speakers"] if s["label"] != "OPERATOR"]
        self.assertTrue(all(s["channel"] == "right" for s in remote_entries))
        operator_segments_in_output = [seg for seg in payload["segments"] if seg[2] == "OPERATOR"]
        self.assertTrue(len(operator_segments_in_output) > 0)

    def test_mono_calls_diarizer_once(self):
        calls = []

        def fake_diarize_channel(samples, label_prefix, num_clusters=-1):
            calls.append((label_prefix, num_clusters))
            return [(0.0, 3.0, "SPEAKER_00"), (3.5, 6.0, "SPEAKER_01")]

        with patch("scribe.diarize_channel", side_effect=fake_diarize_channel):
            payload = scribe.build_diarization_payload(
                np.zeros(100), np.zeros(100), has_loopback=False, duration_sec=6.0)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "SPEAKER_")
        self.assertEqual(payload["channel_mode"], "mono_single_track")
        labels = {s["label"] for s in payload["speakers"]}
        self.assertTrue(all(lbl.startswith("SPEAKER_") for lbl in labels))

    def test_raising_diarizer_returns_none_no_exception_escapes(self):
        with patch("scribe.diarize_channel", side_effect=RuntimeError("model load failed")):
            payload = scribe.build_diarization_payload(
                np.zeros(100), np.zeros(100), has_loopback=True, duration_sec=6.0)
        self.assertIsNone(payload)  # this is the test that protects the recording

    def test_disabled_flag_returns_none_without_calling_diarizer(self):
        with patch("scribe.diarize_channel") as mock_diarize:
            with patch.object(scribe, "DIARIZATION_ENABLED", False):
                payload = scribe.build_diarization_payload(
                    np.zeros(100), np.zeros(100), has_loopback=True, duration_sec=6.0)
        self.assertIsNone(payload)
        mock_diarize.assert_not_called()


class TestDiarizeChannelDtypeConversion(unittest.TestCase):
    def test_int16_input_reaches_diarizer_as_float32_in_range(self):
        captured = {}

        class FakeResult:
            def __init__(self, start, end, speaker):
                self.start, self.end, self.speaker = start, end, speaker

        class FakeDiarizer:
            def process(self, arr):
                captured["arr"] = arr
                return self

            def sort_by_start_time(self):
                return [FakeResult(0.0, 1.0, 0)]

        with patch("scribe._make_diarizer", return_value=FakeDiarizer()):
            int16_input = np.array([16384, -16384, 32767, -32768], dtype=np.int16)
            scribe.diarize_channel(int16_input, "TEST_")

        arr = captured["arr"]
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.all(arr >= -1.0) and np.all(arr <= 1.0))

    def test_float64_input_reaches_diarizer_as_float32_in_range(self):
        captured = {}

        class FakeResult:
            def __init__(self, start, end, speaker):
                self.start, self.end, self.speaker = start, end, speaker

        class FakeDiarizer:
            def process(self, arr):
                captured["arr"] = arr
                return self

            def sort_by_start_time(self):
                return [FakeResult(0.0, 1.0, 0)]

        with patch("scribe._make_diarizer", return_value=FakeDiarizer()):
            float64_input = np.array([16384.0, -16384.0], dtype=np.float64)
            scribe.diarize_channel(float64_input, "TEST_")

        arr = captured["arr"]
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.all(np.abs(arr) <= 1.0))


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name
        self.chunk_size = None
        self.deleted = False

    def upload_from_string(self, content, content_type=None):
        self.bucket.upload_calls.append(("string", self.name))

    def upload_from_filename(self, path, timeout=None):
        self.bucket.upload_calls.append(("filename", self.name))
        if self.bucket.fail_wav_upload:
            raise RuntimeError("simulated upload failure")

    def delete(self):
        self.deleted = True
        self.bucket.delete_calls.append(self.name)


class FakeBucket:
    def __init__(self, fail_wav_upload=False):
        self.upload_calls = []
        self.delete_calls = []
        self.fail_wav_upload = fail_wav_upload
        self._blobs = {}

    def blob(self, name):
        if name not in self._blobs:
            self._blobs[name] = FakeBlob(self, name)
        return self._blobs[name]


class TestUploadToGcpOrdering(unittest.TestCase):
    def _run(self, diar_payload, fail_wav=False):
        bucket = FakeBucket(fail_wav_upload=fail_wav)
        fake_client = MagicMock()
        fake_client.bucket.return_value = bucket
        fake_creds = MagicMock()
        fake_creds.project_id = "fake-project"

        with patch("scribe.service_account.Credentials.from_service_account_file", return_value=fake_creds), \
             patch("scribe.storage.Client", return_value=fake_client), \
             patch("scribe.os.path.exists", return_value=True):
            if fail_wav:
                with self.assertRaises(RuntimeError):
                    scribe.upload_to_gcp("fake_local_path.wav", diar_payload)
            else:
                scribe.upload_to_gcp("fake_local_path.wav", diar_payload)
        return bucket

    def test_companion_uploaded_before_wav_shared_stem(self):
        bucket = self._run({"schema_version": 1, "segments": []})
        self.assertEqual(len(bucket.upload_calls), 2)
        self.assertEqual(bucket.upload_calls[0][0], "string")     # companion first
        self.assertEqual(bucket.upload_calls[1][0], "filename")   # .wav second
        companion_name = bucket.upload_calls[0][1]
        wav_name = bucket.upload_calls[1][1]
        stem = companion_name.replace(".diarization.json", "")
        self.assertEqual(wav_name, f"{stem}.wav")

    def test_no_diar_payload_uploads_only_wav(self):
        bucket = self._run(None)
        self.assertEqual(len(bucket.upload_calls), 1)
        self.assertEqual(bucket.upload_calls[0][0], "filename")

    def test_wav_failure_deletes_orphaned_companion(self):
        bucket = self._run({"schema_version": 1, "segments": []}, fail_wav=True)
        self.assertEqual(len(bucket.delete_calls), 1)
        self.assertTrue(bucket.delete_calls[0].endswith(".diarization.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
