"""
Tests for cloud/chunking.py.

Calibration reference (real 2026-09-01 meeting, 2681.9s, 723 diarization
segments): the planner produces 5 windows of ~536s, every cut landing on a
diarization turn boundary, gapless, ending exactly at the recording duration.

Run:  .venv-cloud/Scripts/python.exe -m unittest tests.test_chunking -v
"""
import os
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cloud"))

import chunking as ck


def diar_with_turns(duration, turn=4.0):
    """A companion whose turns end every `turn` seconds, so boundaries are dense."""
    segs, t = [], 0.0
    i = 0
    while t + turn <= duration:
        segs.append([t, t + turn - 0.2, f"ROOM_{i % 3:02d}"])
        t += turn
        i += 1
    return {"schema_version": 1, "channel_mode": "stereo_operator_left",
            "speaker_count": 3, "duration_sec": duration, "segments": segs}


class TestPlanChunks(unittest.TestCase):
    def test_short_recording_is_one_chunk(self):
        d = diar_with_turns(300)
        self.assertEqual(ck.plan_chunks(d, 300), [(0.0, 300.0)])

    def test_zero_or_missing_duration_returns_empty(self):
        self.assertEqual(ck.plan_chunks(diar_with_turns(100), 0), [])
        self.assertEqual(ck.plan_chunks(diar_with_turns(100), None), [])

    def test_plan_is_gapless_and_covers_whole_recording(self):
        d = diar_with_turns(2682)
        plan = ck.plan_chunks(d, 2682)
        self.assertEqual(plan[0][0], 0.0)
        self.assertAlmostEqual(plan[-1][1], 2682)
        for i in range(len(plan) - 1):
            self.assertAlmostEqual(plan[i][1], plan[i + 1][0])

    def test_no_chunk_exceeds_maximum(self):
        d = diar_with_turns(2682)
        for a, b in ck.plan_chunks(d, 2682):
            self.assertLessEqual(b - a, ck.MAX_CHUNK_SEC + 1e-6)

    def test_cuts_land_on_turn_boundaries(self):
        """An utterance must never be split -- this is what removes the need for
        overlap and overlap-deduplication."""
        d = diar_with_turns(2682)
        ends = {round(float(s[1]), 3) for s in d["segments"]}
        plan = ck.plan_chunks(d, 2682)
        for a, b in plan[:-1]:
            self.assertIn(round(b, 3), ends)

    def test_windows_are_evenly_distributed(self):
        """Guards the regression this replaced: fixed 600s steps left a 892s
        remainder, and the largest window carries the highest loop risk."""
        plan = ck.plan_chunks(diar_with_turns(2682), 2682)
        widths = [b - a for a, b in plan]
        self.assertLess(max(widths) - min(widths), 60)

    def test_missing_companion_falls_back_to_fixed_windows(self):
        plan = ck.plan_chunks(None, 2682)
        self.assertGreater(len(plan), 1)
        self.assertAlmostEqual(plan[-1][1], 2682)

    def test_malformed_segments_do_not_raise(self):
        bad = {"segments": [["x", "y", "z"], [1], None, [10, 20, "A"]]}
        plan = ck.plan_chunks(bad, 2682)
        self.assertAlmostEqual(plan[-1][1], 2682)


class TestShiftTimestamps(unittest.TestCase):
    def test_offset_zero_is_identity(self):
        t = "0:05 [דנה]: שלום"
        self.assertEqual(ck.shift_timestamps(t, 0), t)

    def test_speaker_line_is_shifted(self):
        self.assertEqual(ck.shift_timestamps("0:05 [דנה]: שלום", 600),
                         "10:05 [דנה]: שלום")

    def test_minute_rollover(self):
        self.assertEqual(ck.shift_timestamps("1:30 [דנה]: א", 90),
                         "3:00 [דנה]: א")

    def test_silence_range_shifts_both_ends(self):
        out = ck.shift_timestamps("0:00 - 0:26: [שקט]", 600)
        self.assertIn("10:00", out)
        self.assertIn("10:26", out)

    def test_non_timestamp_lines_untouched(self):
        self.assertEqual(ck.shift_timestamps("no timestamp here", 600),
                         "no timestamp here")


class TestEstablishedNames(unittest.TestCase):
    def test_real_names_extracted_in_order(self):
        t = "0:01 [ורד]: א\n0:05 [שחר]: ב\n0:09 [ורד]: ג"
        self.assertEqual(ck.established_names(t), ["ורד", "שחר"])

    def test_generic_labels_excluded(self):
        """דובר N / ROOM_xx are placeholders for 'could not name', so carrying
        them forward would teach the next chunk a name that isn't one."""
        t = "0:01 [דובר 1]: א\n0:05 [ROOM_00]: ב\n0:09 [REMOTE_01]: ג\n0:12 [חיים]: ד"
        self.assertEqual(ck.established_names(t), ["חיים"])

    def test_empty_transcript_gives_no_names(self):
        self.assertEqual(ck.established_names(""), [])

    def test_names_hint_empty_when_no_names(self):
        self.assertEqual(ck.names_hint([]), "")

    def test_names_hint_lists_names(self):
        h = ck.names_hint(["ורד", "שחר"])
        self.assertIn("ורד", h)
        self.assertIn("שחר", h)


class TestStitch(unittest.TestCase):
    def test_parts_are_joined_in_absolute_time(self):
        out = ck.stitch([(0, "0:05 [דנה]: א"), (600, "0:10 [דנה]: ב")])
        self.assertIn("0:05 [דנה]: א", out)
        self.assertIn("10:10 [דנה]: ב", out)

    def test_empty_chunks_are_skipped(self):
        out = ck.stitch([(0, "0:05 [דנה]: א"), (600, "   "), (1200, "")])
        self.assertEqual(out.strip(), "0:05 [דנה]: א")

    def test_none_chunk_does_not_raise(self):
        self.assertEqual(ck.stitch([(0, None)]), "")


class TestSliceWav(unittest.TestCase):
    def _make(self, path, seconds=30, rate=16000, channels=2, width=2):
        with wave.open(path, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(width)
            w.setframerate(rate)
            w.writeframes(b"\x00\x01" * channels * rate * seconds)

    def test_slice_preserves_format_and_duration(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "s.wav")
            dst = os.path.join(d, "d.wav")
            self._make(src, seconds=30)
            ck.slice_wav(src, 10, 20, dst)
            with wave.open(dst) as w:
                self.assertEqual(w.getnchannels(), 2)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertEqual(w.getframerate(), 16000)
                self.assertAlmostEqual(w.getnframes() / w.getframerate(), 10.0, places=3)

    def test_slice_past_end_is_clamped(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "s.wav"), os.path.join(d, "d.wav")
            self._make(src, seconds=10)
            ck.slice_wav(src, 5, 999, dst)
            with wave.open(dst) as w:
                self.assertAlmostEqual(w.getnframes() / w.getframerate(), 5.0, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
