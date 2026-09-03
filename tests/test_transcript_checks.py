"""
Tests for cloud/transcript_checks.py.

All fixtures are synthetic. Real meeting transcripts are deliberately NOT
committed -- the two real defective transcripts live outside the repo, and the
thresholds here were calibrated against them (see the 2026-09-03 audit):

  armA_full     timestamp/label echo 1.00, block duplication 0.414
  armB_header   intra-line run 29,654, coverage 0.405, finish_reason MAX_TOKENS
  clean control 0 warnings

Run:  .venv-cloud/Scripts/python.exe -m unittest tests.test_transcript_checks -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cloud"))

import transcript_checks as tc


def line(sec, speaker, text):
    return f"{sec // 60}:{sec % 60:02d} [{speaker}]: {text}"


def clean_transcript(n=60, start=0, step=10):
    """A well-formed transcript: increasing timestamps, two speakers, varied text."""
    return "\n".join(
        line(start + i * step, "דנה" if i % 2 else "יוסי", f"משפט מספר {i} על נושא הפגישה.")
        for i in range(n)
    )


class TestLineParsingMatchesRealModelOutput(unittest.TestCase):
    """The prompt asks for `M:SS [SPEAKER]:` but the model often emits
    `M:SS SPEAKER:` with no brackets. A bracket-only pattern parsed almost
    nothing in the 2026-09-03 production run, silently blinding every check in
    this module and producing a spurious coverage failure."""

    def test_unbracketed_speaker_is_parsed(self):
        p = tc.parse_lines("0:00 דובר 1: למס הכנסה\n0:15 דובר 2: כן.")
        self.assertEqual(len(p), 2)
        self.assertEqual(p[0][2], "דובר 1")
        self.assertEqual(p[1][1], 15)

    def test_bracketed_speaker_still_parsed(self):
        p = tc.parse_lines("0:00 [ורד]: שלום")
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0][2], "ורד")

    def test_mixed_formats_in_one_transcript(self):
        p = tc.parse_lines("0:00 [ורד]: א\n0:05 שחר: ב\n0:10 - 0:20: [שקט]")
        self.assertEqual(len(p), 3)

    def test_coverage_correct_on_unbracketed_transcript(self):
        """The exact production symptom: a full-length transcript reported as
        short because none of its lines parsed."""
        t = "\n".join(f"{i}:00 דובר 1: שורה {i}" for i in range(0, 45))
        r = tc.check_coverage(t, duration_sec=2681.9)
        self.assertFalse(r["detected"])
        self.assertGreater(r["coverage"], 0.95)


class TestBlockRepetition(unittest.TestCase):
    def test_clean_transcript_is_not_flagged(self):
        """False-positive guard -- written first deliberately."""
        r = tc.detect_block_repetition(clean_transcript())
        self.assertFalse(r["detected"])
        self.assertEqual(r["duplicated_fraction"], 0.0)

    def test_repeated_block_is_detected(self):
        lines = clean_transcript(60).splitlines()
        doctored = "\n".join(lines + lines[10:20] * 3)
        r = tc.detect_block_repetition(doctored)
        self.assertTrue(r["detected"])
        self.assertGreater(r["duplicated_fraction"], 0.15)

    def test_repeat_with_different_timestamps_still_detected(self):
        """The real defect replayed identical content under NEW timestamps, so a
        whole-line comparison would have missed it."""
        body = [f"משפט חוזר מספר {i}." for i in range(6)]
        first = [line(100 + i * 5, "דנה", b) for i, b in enumerate(body)]
        later = [line(900 + i * 5, "דנה", b) for i, b in enumerate(body)]
        filler = [line(200 + i * 5, "יוסי", f"מילוי {i}") for i in range(40)]
        r = tc.detect_block_repetition("\n".join(first + filler + later))
        self.assertTrue(r["detected"])

    def test_repeat_with_degraded_labels_still_detected(self):
        """Labels degraded real-name -> generic across repeats in the real defect."""
        body = [f"תוכן זהה {i}." for i in range(6)]
        first = [line(100 + i * 5, "שרה", b) for i, b in enumerate(body)]
        later = [line(900 + i * 5, "דובר 2", b) for i, b in enumerate(body)]
        filler = [line(200 + i * 5, "יוסי", f"מילוי {i}") for i in range(40)]
        r = tc.detect_block_repetition("\n".join(first + filler + later))
        self.assertTrue(r["detected"])

    def test_short_common_utterances_are_not_flagged(self):
        """"כן"/"אוקיי" recur constantly in real meetings and must not trip it."""
        out = []
        for i in range(80):
            out.append(line(i * 10, "דנה", "כן."))
            out.append(line(i * 10 + 5, "יוסי", f"נקודה ייחודית מספר {i}."))
        self.assertFalse(tc.detect_block_repetition("\n".join(out))["detected"])

    def test_reports_period_for_diagnosis(self):
        lines = clean_transcript(60).splitlines()
        r = tc.detect_block_repetition("\n".join(lines + lines[10:20]))
        self.assertTrue(r["detected"])
        self.assertIn("period", r["first_events"][0])


class TestIntraLineDegeneration(unittest.TestCase):
    def test_natural_repetition_is_not_flagged(self):
        t = line(60, "דנה", "לא, לא, לא, זה לא מה שאמרתי.")
        r = tc.detect_intra_line_degeneration(t)
        self.assertFalse(r["detected"])
        self.assertLess(r["max_run"], tc.INTRA_LINE_RUN_LIMIT)

    def test_degenerate_loop_is_detected(self):
        """Models the real 29,654-token run inside a single line."""
        t = line(1085, "שחר", ", ".join(["לא"] * 500))
        r = tc.detect_intra_line_degeneration(t)
        self.assertTrue(r["detected"])
        self.assertGreaterEqual(r["max_run"], 500)
        self.assertEqual(r["token"], "לא")

    def test_block_check_alone_would_miss_it(self):
        """Why both checks exist: intra-line degeneration is invisible to a
        line-window duplication check."""
        t = clean_transcript(40) + "\n" + line(1085, "שחר", ", ".join(["לא"] * 500))
        self.assertFalse(tc.detect_block_repetition(t)["detected"])
        self.assertTrue(tc.detect_intra_line_degeneration(t)["detected"])

    def test_clean_transcript_not_flagged(self):
        self.assertFalse(tc.detect_intra_line_degeneration(clean_transcript())["detected"])


class TestDiarizationEcho(unittest.TestCase):
    def _diar(self, n):
        return {"schema_version": 1, "channel_mode": "stereo_operator_left",
                "speaker_count": 2, "duration_sec": n * 10.0,
                "segments": [[i * 10, i * 10 + 8, f"ROOM_{i % 2:02d}"] for i in range(n)]}

    def test_template_echo_is_detected(self):
        """One output line per segment, copying its timestamp and label -- the
        exact shape measured at 1.00 on both real defective transcripts."""
        d = self._diar(40)
        t = "\n".join(line(int(s[0]), s[2], f"טקסט {i}") for i, s in enumerate(d["segments"]))
        r = tc.detect_diarization_echo(t, d)
        self.assertTrue(r["detected"])
        self.assertEqual(r["timestamp_echo"], 1.0)
        self.assertEqual(r["label_echo"], 1.0)

    def test_genuine_transcript_is_not_flagged(self):
        d = self._diar(40)
        t = "\n".join(line(i * 7 + 3, "דנה" if i % 2 else "יוסי", f"טקסט {i}") for i in range(40))
        r = tc.detect_diarization_echo(t, d)
        self.assertFalse(r["detected"])
        self.assertLess(r["timestamp_echo"], 0.5)

    def test_no_companion_degrades_gracefully(self):
        """Mirrors the project-wide graceful-degradation contract."""
        r = tc.detect_diarization_echo(clean_transcript(), None)
        self.assertFalse(r["detected"])
        self.assertIn("reason", r)


class TestTimestampsAndCoverage(unittest.TestCase):
    def test_monotonic_clean(self):
        self.assertFalse(tc.check_timestamps_monotonic(clean_transcript())["detected"])

    def test_backwards_timestamp_detected(self):
        t = "\n".join([line(600, "דנה", "א"), line(300, "יוסי", "ב")])
        r = tc.check_timestamps_monotonic(t)
        self.assertTrue(r["detected"])
        self.assertEqual(r["count"], 1)

    def test_coverage_short_transcript_detected(self):
        r = tc.check_coverage(clean_transcript(10, step=10), duration_sec=2000.0)
        self.assertTrue(r["detected"])
        self.assertLess(r["coverage"], 0.9)

    def test_coverage_full_transcript_ok(self):
        r = tc.check_coverage(clean_transcript(60, step=10), duration_sec=600.0)
        self.assertFalse(r["detected"])

    def test_coverage_unknown_duration_is_not_flagged(self):
        self.assertFalse(tc.check_coverage(clean_transcript(), None)["detected"])


class TestVerifyTranscript(unittest.TestCase):
    def test_clean_transcript_produces_no_warnings(self):
        w, _ = tc.verify_transcript(clean_transcript(60, step=10),
                                    diar=None, duration_sec=600.0, finish_reason="STOP")
        self.assertEqual(w, [])
        self.assertEqual(tc.warning_banner(w), "")

    def test_max_tokens_is_reported(self):
        w, r = tc.verify_transcript(clean_transcript(60, step=10), diar=None,
                                    duration_sec=600.0, finish_reason="FinishReason.MAX_TOKENS")
        self.assertTrue(w)
        self.assertIn("finish_reason", r)

    def test_banner_lists_every_warning_and_preserves_nothing_else(self):
        w = ["בעיה א", "בעיה ב"]
        b = tc.warning_banner(w)
        self.assertIn("בעיה א", b)
        self.assertIn("בעיה ב", b)
        self.assertTrue(b.startswith(">"))

    def test_never_raises_on_garbage_input(self):
        for bad in ("", "   ", "no timestamps here at all", "::::", "1:2:3:4 [x]:"):
            w, r = tc.verify_transcript(bad, diar={"segments": "not-a-list"},
                                        duration_sec=None, finish_reason=None)
            self.assertIsInstance(w, list)
            self.assertIsInstance(r, dict)

    def test_report_is_deterministic(self):
        t = clean_transcript(60)
        a = tc.verify_transcript(t, None, 600.0, "STOP")
        b = tc.verify_transcript(t, None, 600.0, "STOP")
        self.assertEqual(a, b)

    def test_completes_quickly_on_a_large_transcript(self):
        """Guards against an O(n^2) implementation reaching the request path."""
        import time
        t = clean_transcript(2000, step=2)
        t0 = time.time()
        tc.verify_transcript(t, None, 4000.0, "STOP")
        self.assertLess(time.time() - t0, 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
