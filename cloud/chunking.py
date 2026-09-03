"""
Splitting a long recording into per-chunk transcription windows.

Why this exists (measured 2026-09-03 on the real 44-minute Sept-1 meeting):

  * Sending the diarization turn-by-turn list made the model emit one line per
    segment, copying its timestamp and label -- timestamp/label echo 1.00, 41%
    duplicated content, finish_reason=STOP. Complete-looking and fabricated.
  * Removing that list made the model transcribe for real and correctly name all
    four speakers -- but it entered a token-level degenerate loop at 18:05,
    repeating one word 29,654 times until the 65,536 output cap was exhausted
    (finish_reason=MAX_TOKENS), covering only 40% of the meeting.

So the fix is both: drop the turn list, AND chunk. Chunking gives every window
its own output budget and CONTAINS a degenerate loop to one window, where it is
cheap to detect (transcript_checks) and retry.

Boundaries are placed in the silence between diarization turns, never inside an
utterance -- which is why no overlap is needed and no overlap-deduplication is
required. Speaker identity stays consistent across chunks because the local
sherpa-onnx clustering runs GLOBALLY over the whole file before chunking, so a
label means the same person in every window.
"""
import os
import re
import wave

TARGET_CHUNK_SEC = 600      # 10 min. Arm B transcribed 18 min inside one output
MAX_CHUNK_SEC = 900         # budget, so 10 min carries ~45% headroom.
MIN_CHUNK_SEC = 120         # never emit a trailing sliver as its own API call

# Labels the model produces when it could NOT establish a real name. These must
# not be carried forward as "already established" -- they are placeholders.
_GENERIC_LABEL = re.compile(r"^(דובר|דוברת)\s*\d+$|^(ROOM|REMOTE|OPERATOR|SPEAKER)_?\d*$", re.I)
_LINE = re.compile(r"^(\s*)(\d+):(\d{2})(\s*(?:-\s*\d+:\d{2})?\s*)(.*)$")


def plan_chunks(diar, duration_sec, target=TARGET_CHUNK_SEC, maximum=MAX_CHUNK_SEC,
                minimum=MIN_CHUNK_SEC):
    """[(start_sec, end_sec)] covering [0, duration_sec] with no gaps or overlap.

    Cuts land on diarization turn boundaries near each target, so an utterance is
    never split. Falls back to fixed-width windows when no usable companion
    exists -- the same graceful-degradation contract used everywhere else."""
    if not duration_sec or duration_sec <= 0:
        return []
    if duration_sec <= maximum:
        return [(0.0, float(duration_sec))]

    # Segment contents are not trusted: the companion is an uploaded object, and a
    # malformed entry must degrade to "no boundaries", never raise. Coerce first,
    # then sort -- sorting raw entries crashes on mixed types.
    boundaries = []
    for s in (diar or {}).get("segments") or []:
        try:
            if len(s) >= 2:
                boundaries.append(float(s[1]))   # end of a turn == a silence point
        except (TypeError, ValueError):
            continue
    boundaries = sorted(set(b for b in boundaries if 0 < b < duration_sec))

    # Spread evenly rather than taking fixed `target` steps and leaving a long
    # remainder: a 2682s meeting becomes 5 x ~536s, not 4 x 600s + one 892s
    # window. The largest window carries the highest risk of a degenerate loop,
    # so the goal is to minimise the maximum, not to hit `target` exactly.
    import math
    n = max(1, math.ceil(duration_sec / target))
    even = duration_sec / n

    cuts, pos = [], 0.0
    for k in range(1, n):
        ideal = even * k
        window = [b for b in boundaries if pos + minimum <= b <= pos + maximum]
        cut = min(window, key=lambda b: abs(b - ideal)) if window else min(ideal, pos + maximum)
        if cut <= pos or cut >= duration_sec:   # defensive: never fail to advance
            cut = min(pos + even, duration_sec)
        cuts.append(cut)
        pos = cut
    bounds = [0.0] + cuts + [float(duration_sec)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def slice_wav(src_path, start_sec, end_sec, dest_path):
    """Write [start_sec, end_sec) of a PCM wav to dest_path, preserving format."""
    with wave.open(src_path, "rb") as src:
        rate = src.getframerate()
        start_f = max(0, int(start_sec * rate))
        end_f = min(src.getnframes(), int(end_sec * rate))
        src.setpos(start_f)
        frames = src.readframes(max(0, end_f - start_f))
        with wave.open(dest_path, "wb") as dst:
            dst.setnchannels(src.getnchannels())
            dst.setsampwidth(src.getsampwidth())
            dst.setframerate(rate)
            dst.writeframes(frames)
    return dest_path


def shift_timestamps(text, offset_sec):
    """Rewrite leading `M:SS` (and `M:SS - M:SS`) to absolute meeting time.

    Each chunk is transcribed in isolation, so the model numbers it from 0:00."""
    if not offset_sec:
        return text
    off = int(offset_sec)

    def fix(m):
        lead, mm, ss, mid, rest = m.groups()
        total = int(mm) * 60 + int(ss) + off
        mid2 = re.sub(
            r"(\d+):(\d{2})",
            lambda n: f"{(int(n.group(1))*60+int(n.group(2))+off)//60}:"
                      f"{(int(n.group(1))*60+int(n.group(2))+off)%60:02d}",
            mid or "",
        )
        return f"{lead}{total//60}:{total%60:02d}{mid2}{rest}"

    return "\n".join(_LINE.sub(fix, ln) if _LINE.match(ln) else ln
                     for ln in text.splitlines())


def established_names(text):
    """Real names the model resolved in a chunk, in first-appearance order.

    Fed into later chunks so the same voice keeps the same name across windows --
    the one thing chunking does not get for free from global diarization."""
    names, seen = [], set()
    for ln in text.splitlines():
        m = re.match(r"^\s*\d+:\d{2}\s*\[([^\]]+)\]\s*:", ln)
        if not m:
            continue
        label = m.group(1).strip()
        if label and not _GENERIC_LABEL.match(label) and label not in seen:
            seen.add(label)
            names.append(label)
    return names


def names_hint(names):
    """Prompt fragment carrying names already established earlier in the meeting."""
    if not names:
        return ""
    return (
        "\n\nNames already established earlier in this same meeting: "
        + ", ".join(names)
        + ".\nIf you hear one of these voices again, reuse that exact name. "
        "Do not invent a new label for a voice that already has a name, and do "
        "not assume a name applies to a different voice.\n"
    )


def stitch(parts):
    """Join [(offset_sec, chunk_text)] into one transcript in absolute time."""
    out = []
    for offset, text in parts:
        t = shift_timestamps((text or "").strip(), offset)
        if t:
            out.append(t)
    return "\n".join(out)


def chunk_paths(base_dir, stem, index):
    return os.path.join(base_dir, f"{stem}.chunk{index:02d}.wav")
