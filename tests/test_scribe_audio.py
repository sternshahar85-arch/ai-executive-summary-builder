"""
Verifies Phase 4's audio-processing changes to scribe.py: process_audio (lifted to
module level, unchanged logic) and build_output_audio (new -- replaces the additive
mono mix with stereo channel separation, or mono when no loopback device exists).

Run with: python tests/test_scribe_audio.py
(scribe.py's own deps -- numpy, scipy -- not a separate venv, per requirements-client.txt)
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# scribe.py opens a Tk GUI at import time only inside __main__ guard, and touches
# pyaudiowpatch only inside functions -- safe to import for its pure functions.
import scribe


def synth_frames(num_samples, channels=1, amplitude=5000, freq_hz=440, rate=48000):
    """Build a list of raw int16 frame byte-chunks like PyAudio would produce,
    optionally interleaved across channels, containing a simple sine wave."""
    t = np.arange(num_samples) / rate
    tone = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)
    if channels > 1:
        interleaved = np.repeat(tone, channels)  # same tone on every channel
    else:
        interleaved = tone
    raw_bytes = interleaved.tobytes()
    # split into a few chunks, like real streaming reads would arrive
    chunk_size = len(raw_bytes) // 4 or len(raw_bytes)
    return [raw_bytes[i:i + chunk_size] for i in range(0, len(raw_bytes), chunk_size)]


class TestProcessAudio(unittest.TestCase):
    def test_empty_frames_returns_empty_int16_array(self):
        result = scribe.process_audio([], channels=1, original_rate=48000)
        self.assertEqual(len(result), 0)
        self.assertEqual(result.dtype, np.int16)

    def test_multichannel_downmixes_to_mono_same_sample_count(self):
        n = 4800  # 0.1s at 48kHz
        frames = synth_frames(n, channels=2, rate=48000)
        result = scribe.process_audio(frames, channels=2, original_rate=scribe.TARGET_SAMPLE_RATE)
        # same rate as target -> no resampling, so length matches sample count exactly
        self.assertEqual(len(result), n)

    def test_resampling_changes_length_by_correct_ratio(self):
        n = 48000  # 1 second at 48kHz
        frames = synth_frames(n, channels=1, rate=48000)
        result = scribe.process_audio(frames, channels=1, original_rate=48000)
        expected = n * scribe.TARGET_SAMPLE_RATE / 48000  # -> 16000
        self.assertAlmostEqual(len(result), expected, delta=2)


class TestBuildOutputAudio(unittest.TestCase):
    def test_stereo_when_loopback_present(self):
        mic = np.array([100.0, -200.0, 300.0], dtype=np.float64)
        sys_ = np.array([50.0, -60.0, 70.0], dtype=np.float64)
        out = scribe.build_output_audio(mic, sys_, has_loopback=True)
        self.assertEqual(out.shape, (3, 2))
        self.assertEqual(out.dtype, np.int16)
        np.testing.assert_array_equal(out[:, 0], [100, -200, 300])  # left = mic
        np.testing.assert_array_equal(out[:, 1], [50, -60, 70])     # right = sys

    def test_mono_when_no_loopback(self):
        mic = np.array([100.0, -200.0, 300.0], dtype=np.float64)
        sys_ = np.array([9999.0, 9999.0, 9999.0], dtype=np.float64)  # must be ignored
        out = scribe.build_output_audio(mic, sys_, has_loopback=False)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.dtype, np.int16)
        np.testing.assert_array_equal(out, [100, -200, 300])

    def test_clips_rather_than_wraps_on_overflow(self):
        mic = np.array([40000.0, -40000.0], dtype=np.float64)  # beyond int16 range
        sys_ = np.array([0.0, 0.0], dtype=np.float64)
        out = scribe.build_output_audio(mic, sys_, has_loopback=True)
        # must clip to int16 bounds, not wrap around to a negative/positive garbage value
        self.assertEqual(out[0, 0], 32767)
        self.assertEqual(out[1, 0], -32768)

    def test_rounds_before_casting_no_truncation_bias(self):
        mic = np.array([100.6, -100.6], dtype=np.float64)
        sys_ = np.array([0.0, 0.0], dtype=np.float64)
        out = scribe.build_output_audio(mic, sys_, has_loopback=True)
        self.assertEqual(out[0, 0], 101)   # round(100.6) = 101, not truncated to 100
        self.assertEqual(out[1, 0], -101)  # round(-100.6) = -101, not -100


if __name__ == "__main__":
    unittest.main(verbosity=2)
