"""
Deterministic, zero-cost quality checks on the Pass-2 transcript.

Motivation (measured, not theoretical -- see the 2026-09-03 audit):
the pipeline validated the Pass-2 transcript for exactly one property, that it
was non-empty, and then shipped it. Two distinct real failure modes got through:

  1. TEMPLATE ECHO / BLOCK DUPLICATION. When the diarization turn-by-turn list
     was injected into the prompt, the model emitted one line per diarization
     segment and copied that segment's timestamp and label verbatim, instead of
     transcribing the audio. Measured on the real 2026-09-01 meeting: 723 output
     lines for 723 segments, timestamp echo 1.00, label echo 1.00, 41% of the
     transcript duplicated, finish_reason=STOP. It looked complete and was not.

  2. TOKEN-LEVEL DEGENERATION. With the turn list removed the model transcribes
     for real, but on the same audio it entered a degenerate loop at 18:05 and
     emitted the same short token 29,654 times inside ONE line, consuming the
     whole output budget (finish_reason=MAX_TOKENS). A line-window duplication
     check cannot see this -- it is intra-line -- which is why both checks exist.

Design rules, matching the existing grounding/attendee checks in main.py:
  - FLAG, never mutate or drop. A false positive must not cost real content.
  - Never raise. A bug in a checker must not fail a pipeline run.
  - Deterministic and linear-time; this runs inside the request path.
"""
import re

# The prompt specifies `M:SS [SPEAKER]: text`, but the model frequently emits
# `M:SS SPEAKER: text` with no brackets -- confirmed in the 2026-09-03 production
# run, where a bracket-only pattern parsed almost nothing, silently blinding every
# check here and reporting a spurious coverage failure. Parse what the model
# actually produces, not only what it was asked to produce.
SPEAKER_LINE = re.compile(
    r"^\s*(\d+):(\d{2})\s*(?:\[(?P<b>[^\]]{1,60})\]|(?P<p>[^:\[\]\n]{1,60}?))\s*:\s*(?P<text>.*)$"
)
# `M:SS - M:SS: [שקט]`, the non-speech form -- carries a timestamp but no speaker.
SILENCE_LINE = re.compile(r"^\s*(\d+):(\d{2})\s*-\s*\d+:\d{2}\s*:")

# A real speaker may legitimately repeat a short word ("לא, לא, לא"). 30 consecutive
# identical tokens is far outside natural speech and inside observed degeneration
# (29,654), so the gap between signal and noise is ~3 orders of magnitude.
INTRA_LINE_RUN_LIMIT = 30
# 5 consecutive identical lines recurring >=20 lines later is a repeated block, not
# a coincidence; both figures come from the real defective transcripts.
BLOCK_WINDOW = 5
BLOCK_MIN_GAP = 20
# Above this share of lines echoing the diarization template, the transcript is a
# copy of the prompt rather than of the audio. Real runs measured 1.00 vs 0.00.
ECHO_LIMIT = 0.5
# A transcript whose last timestamp falls short of this share of the real audio
# duration is truncated, whatever finish_reason says.
COVERAGE_MIN = 0.9


def parse_lines(transcript):
    """[(index, seconds, speaker|None, text)] for every parseable line."""
    out = []
    for i, raw in enumerate(transcript.splitlines()):
        if not raw.strip():
            continue
        m = SPEAKER_LINE.match(raw)
        if m:
            speaker = (m.group("b") or m.group("p") or "").strip()
            out.append((i, int(m.group(1)) * 60 + int(m.group(2)), speaker, m.group("text")))
            continue
        s = SILENCE_LINE.match(raw)
        if s:
            out.append((i, int(s.group(1)) * 60 + int(s.group(2)), None, ""))
    return out


def _max_consecutive_run(text):
    """Longest run of one token repeated back-to-back. Degeneration is consecutive,
    so this is tighter than a frequency count and will not fire on a filler word
    that merely recurs often across a long utterance."""
    tokens = [t for t in re.split(r"[\s,.!?;:־–-]+", text) if t]
    best = run = 0
    prev = None
    for t in tokens:
        run = run + 1 if t == prev else 1
        prev = t
        if run > best:
            best = run
    return best


def detect_intra_line_degeneration(transcript, limit=INTRA_LINE_RUN_LIMIT):
    worst_run, worst_line, worst_token = 0, None, None
    for idx, _sec, _spk, text in parse_lines(transcript):
        run = _max_consecutive_run(text)
        if run > worst_run:
            worst_run, worst_line = run, idx
            toks = [t for t in re.split(r"[\s,.!?;:־–-]+", text) if t]
            worst_token = max(set(toks), key=toks.count) if toks else None
    return {"detected": worst_run >= limit, "max_run": worst_run,
            "line_index": worst_line, "token": worst_token}


def detect_block_repetition(transcript, window=BLOCK_WINDOW, min_gap=BLOCK_MIN_GAP):
    """Repeated blocks of dialogue, ignoring timestamps and speaker labels -- the
    Sept-1 defect repeated the same content with *different* timestamps and
    progressively degraded labels, so comparing whole lines would have missed it."""
    parsed = parse_lines(transcript)
    bodies = [t.strip() for (_i, _s, _sp, t) in parsed]
    seen, dup, events = {}, set(), []
    for i in range(len(bodies) - window + 1):
        key = "\n".join(bodies[i:i + window])
        if not key.strip():
            continue
        first = seen.get(key)
        if first is not None and i - first >= min_gap:
            dup.update(range(first, first + window))
            dup.update(range(i, i + window))
            events.append({"first_line": first, "repeat_line": i, "period": i - first})
        elif first is None:
            seen[key] = i
    frac = round(len(dup) / len(bodies), 3) if bodies else 0.0
    return {"detected": bool(events), "duplicated_fraction": frac,
            "events": len(events), "first_events": events[:5]}


def detect_diarization_echo(transcript, diar, limit=ECHO_LIMIT):
    """Is the transcript a copy of the diarization template rather than the audio?

    Compares output line i against diarization segment i. This is the single
    highest-signal check available: it scored 1.00 on every known-bad transcript
    and 0.00 on the genuine one."""
    if not diar:
        return {"detected": False, "reason": "no diarization companion"}
    segs = sorted((diar.get("segments") or []), key=lambda s: s[0] if s else 0)
    parsed = parse_lines(transcript)
    if not segs or not parsed:
        return {"detected": False, "reason": "nothing to compare"}
    ts_hits = lb_hits = compared = 0
    for i, (_idx, sec, spk, _t) in enumerate(parsed):
        if i >= len(segs) or len(segs[i]) < 3:
            break
        compared += 1
        if sec == max(0, int(segs[i][0])):
            ts_hits += 1
        if spk is not None and str(spk).strip() == str(segs[i][2]).strip():
            lb_hits += 1
    if not compared:
        return {"detected": False, "reason": "nothing to compare"}
    ts_rate = round(ts_hits / compared, 3)
    lb_rate = round(lb_hits / compared, 3)
    return {"detected": max(ts_rate, lb_rate) >= limit,
            "timestamp_echo": ts_rate, "label_echo": lb_rate,
            "line_count": len(parsed), "segment_count": len(segs)}


def check_timestamps_monotonic(transcript):
    """Timestamps must not run backwards. A repeated block replays an earlier
    time, so this catches duplication independently of content matching."""
    parsed = parse_lines(transcript)
    regressions = []
    prev = None
    for idx, sec, _spk, _t in parsed:
        if prev is not None and sec < prev - 2:  # 2s tolerance for rounding
            regressions.append({"line_index": idx, "seconds": sec, "previous": prev})
        prev = max(prev, sec) if prev is not None else sec
    return {"detected": bool(regressions), "count": len(regressions),
            "first": regressions[:5]}


def check_coverage(transcript, duration_sec, minimum=COVERAGE_MIN):
    if not duration_sec:
        return {"detected": False, "reason": "duration unknown"}
    parsed = parse_lines(transcript)
    if not parsed:
        return {"detected": True, "reason": "no parseable lines", "coverage": 0.0}
    last = max(sec for (_i, sec, _s, _t) in parsed)
    cov = round(last / duration_sec, 3)
    return {"detected": cov < minimum, "coverage": cov,
            "last_timestamp_sec": last, "duration_sec": round(duration_sec, 1)}


def verify_transcript(transcript, diar=None, duration_sec=None, finish_reason=None):
    """Runs every check. Returns (warnings, report). Never raises -- a checker bug
    must not fail a run that produced a real transcript."""
    warnings, report = [], {}
    try:
        if finish_reason is not None and "MAX_TOKENS" in str(finish_reason).upper():
            report["finish_reason"] = str(finish_reason)
            warnings.append("התמלול נקטע באמצע (חריגה ממכסת הפלט) -- ייתכן שחסר חלק מהפגישה.")

        checks = (
            ("diarization_echo", detect_diarization_echo(transcript, diar),
             "התמלול משכפל את מבנה זיהוי-הדוברים במקום לתמלל את האודיו -- אינו אמין."),
            ("intra_line_degeneration", detect_intra_line_degeneration(transcript),
             "זוהתה חזרה מילולית חריגה בתוך שורה -- ייתכן שהמודל נתקע בלולאה."),
            ("block_repetition", detect_block_repetition(transcript),
             "זוהו קטעי דיאלוג החוזרים על עצמם -- ייתכן שחלק מהתמלול משוכפל."),
            ("timestamps_monotonic", check_timestamps_monotonic(transcript),
             "חותמות הזמן אינן עולות באופן רציף -- ייתכן שחלק מהתמלול משוכפל."),
            ("coverage", check_coverage(transcript, duration_sec),
             "התמלול אינו מכסה את כל משך ההקלטה."),
        )
        for name, result, message in checks:
            report[name] = result
            if result.get("detected"):
                warnings.append(message)
    except Exception as err:  # never fail the pipeline on a checker bug
        report["checker_error"] = str(err)[:200]
    return warnings, report


def warning_banner(warnings):
    """Markdown banner prepended to the delivered transcript. Empty when clean."""
    if not warnings:
        return ""
    lines = ["> ⚠️ **בדיקות איכות אוטומטיות זיהו בעיות בתמלול הזה:**", ">"]
    lines += [f"> - {w}" for w in warnings]
    lines += [">", "> התמלול המלא מובא במלואו למטה ללא שינוי.", ""]
    return "\n".join(lines) + "\n"
