import os
import re
import json
import time
import wave
import urllib.request
import urllib.error
import functions_framework
import httpx
from google import genai
from google.genai import types, errors
from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed
from rapidfuzz import fuzz

import chunking
import transcript_checks

GROUNDING_THRESHOLD = 80

# Promoted from a handler local so the chunked Pass 2 can share it.
FILE_PROCESSING_MAX_WAIT_SEC = 300

# google-genai leaves HttpOptions.timeout unset, which means a stalled call blocks
# forever. Reproduced 2026-09-03 as a 64-minute hang with 1.4s of CPU inside
# caches.create(), and twice during the August investigation. Cloud Run kills the
# request at its own timeout with no finally block and no alert, so an explicit
# per-call bound is what turns a silent hang into a retryable error.
GEMINI_HTTP_TIMEOUT_MS = 300_000

RETRYABLE_ERRORS = (errors.ServerError, httpx.ReadError, httpx.ConnectError,
                    httpx.RemoteProtocolError, httpx.ReadTimeout,
                    # The Drive webhook is called through urllib, not httpx, so its
                    # transient failures raise a different family entirely.
                    urllib.error.URLError, TimeoutError)
RETRY_MAX_ATTEMPTS = 3
RETRY_WAIT_SEC = 8

# urlopen defaults to no timeout, so a hung Apps Script blocked until Cloud Run
# killed the whole request -- taking the finally block, the failed/ preservation
# and the alert down with it.
DRIVE_WEBHOOK_TIMEOUT_SEC = 60

# Diarization labels are interpolated into both prompts; a label is an identifier
# like "ROOM_00" or a short name, never prose. Anything longer is not a label.
MAX_LABEL_CHARS = 40


def with_retries(label, fn):
    """
    Retries fn() up to RETRY_MAX_ATTEMPTS times on transient Gemini server
    errors (5xx, including the real 503 "high demand" error seen in
    production) or transient network errors, with a fixed backoff between
    attempts. Re-raises on the final attempt so real failures still surface.
    """
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return fn()
        except RETRYABLE_ERRORS as e:
            print(f"[WARNING] {label}: transient error on attempt {attempt}/{RETRY_MAX_ATTEMPTS}: {e}")
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
            time.sleep(RETRY_WAIT_SEC)


def is_grounded(source_quote, transcript, threshold=GROUNDING_THRESHOLD):
    """
    Returns True if source_quote is missing (nothing to check) or fuzzy-matches
    somewhere in transcript above threshold. Returns False only when a quote was
    given but doesn't actually appear in the transcript -- i.e. likely hallucinated.
    Deterministic, local, no extra Gemini call.
    """
    if not source_quote:
        return True
    return fuzz.partial_ratio(source_quote, transcript) >= threshold


MAX_PROMPT_SEGMENTS = 2000
DIARIZATION_SCHEMA_VERSION = 1


def companion_name_for(wav_name):
    if not wav_name.endswith(".wav"):
        return None
    return wav_name[:-4] + ".diarization.json"


def load_diarization(bucket, file_name):
    """
    Downloads and parses the companion diarization file for file_name, if one
    exists. Returns a validated dict, or None on ANY failure (missing file, bad
    JSON, wrong schema version, malformed segments) -- this function must never
    raise. That is the entire graceful-degradation contract for clients that
    don't produce a companion file (feature disabled, older client, or the
    client-side diarization step itself failed).
    """
    try:
        companion_name = companion_name_for(file_name)
        if not companion_name:
            return None
        companion_blob = bucket.blob(companion_name)
        if not companion_blob.exists():
            return None
        raw = companion_blob.download_as_bytes()
        diar = json.loads(raw)
        if not isinstance(diar, dict):
            return None
        if diar.get("schema_version") != DIARIZATION_SCHEMA_VERSION:
            return None
        if not isinstance(diar.get("segments"), list):
            return None
        if diar.get("channel_mode") not in ("stereo_operator_left", "mono_single_track"):
            return None

        # Validate segment CONTENTS, not just that segments is a list. This object
        # is uploaded by whoever holds the recorder credential, and its labels are
        # interpolated straight into both Gemini prompts -- so an unvalidated label
        # is a prompt-injection channel that turns a write-only, correctly-scoped
        # credential into control over what lands in the owner's Drive. Non-numeric
        # times were also a crash: _format_mmss/int() raised outside this function's
        # try, failing the whole pipeline.
        clean = []
        for seg in diar["segments"]:
            if not isinstance(seg, (list, tuple)) or len(seg) < 3:
                continue
            try:
                start, end = float(seg[0]), float(seg[1])
            except (TypeError, ValueError):
                continue
            if not isinstance(seg[2], str):
                continue
            label = re.sub(r"[^\w֐-׿ .\-]", "", seg[2])[:MAX_LABEL_CHARS].strip()
            if not label:
                continue
            clean.append([start, end, label])
        diar["segments"] = clean
        return diar
    except Exception as e:
        print(f"[WARNING] Could not load diarization companion for {file_name}: {e}")
        return None


def _format_mmss(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def format_diarization_for_prompt(diar, include_turns=True):
    """
    Returns "" when diar is None (graceful degradation -- both prompts are then
    byte-identical to their pre-diarization form). Otherwise returns a prompt
    block describing the speaker roster and turn boundaries as local, structural
    ground truth for Gemini to resolve real names against.
    """
    if not diar:
        return ""

    segments = sorted(diar.get("segments") or [], key=lambda s: s[0] if len(s) > 0 else 0)
    speaker_count = diar.get("speaker_count") or len({s[2] for s in segments if len(s) > 2})
    channel_mode = diar.get("channel_mode")

    if channel_mode == "stereo_operator_left":
        has_multi_room_speakers = any(
            len(s) > 2 and str(s[2]).startswith("ROOM_") for s in segments
        )
        if has_multi_room_speakers:
            # room_participants >= 2 was supplied at recording time -- the left/
            # room channel got real clustering instead of being forced to a
            # single "OPERATOR" label, so multiple physical people may share it.
            channel_note = (
                "The audio is stereo. The left channel is the recording device's own "
                "room microphone and captures more than one physical person -- labels "
                "starting with ROOM_ are distinct voices on that channel, none "
                "privileged as \"the operator\"; resolve each to a real name from "
                "context the same way as any other label. Labels starting with "
                "REMOTE_ are distinct voices separated out of the right channel "
                "(remote participants)."
            )
        else:
            channel_note = (
                "The audio is stereo. OPERATOR is the left channel (the person running the "
                "recording). Other labels are distinct voices separated out of the right "
                "channel (remote participants)."
            )
    else:
        channel_note = (
            "The audio is mono (a single in-room microphone). The labels below are "
            "distinct voices in the room; one of them is the operator, but which one is "
            "not known from the audio channel alone -- determine it from context."
        )

    lines = [
        "SPEAKER TURN DATA (ground truth, computed locally from the audio waveform --",
        "this is structural evidence, not a guess):",
        "",
        f"This recording contains {speaker_count} distinct speaking voices.",
        channel_note,
        "",
    ]

    turns_included = include_turns and len(segments) <= MAX_PROMPT_SEGMENTS
    if turns_included:
        lines.append("Speaking turns (start-end, label):")
        for seg in segments:
            if len(seg) < 3:
                continue
            start, end, label = seg[0], seg[1], seg[2]
            lines.append(f"{_format_mmss(start)}-{_format_mmss(end)} {label}")
    elif not include_turns:
        lines.append(
            "(Turn-by-turn list deliberately omitted -- see the note in Pass 2.)"
        )
    else:
        lines.append(
            f"(Turn-by-turn list omitted -- {len(segments)} segments exceeds the prompt "
            f"limit. Use the speaker count and channel information above.)"
        )

    lines.append("")
    if turns_included:
        lines.append(
            "Use these turn boundaries as authoritative for WHEN the speaker changes and for "
            "HOW MANY distinct people speak. The labels are anonymous -- resolve each one to a "
            "real name yourself from the audio (e.g. when someone introduces themselves) and "
            "use the real name in your output. If you cannot resolve a label to a real name, "
            "keep the anonymous label."
        )
    else:
        # Without the turn list, an instruction to "use these turn boundaries" has
        # no referent -- and pointing the model at a turn list is exactly what made
        # it emit one line per segment instead of transcribing (echo 1.00).
        lines.append(
            "Use the speaker count above as authoritative for HOW MANY distinct people "
            "speak. Determine WHEN each speaker changes from the audio itself. Resolve "
            "each voice to a real name from the audio (e.g. when someone introduces "
            "themselves); if you cannot, use a stable generic label."
        )

    return "\n\n" + "\n".join(lines)


def log_token_usage(label, response):
    """Prints prompt/cached/total token counts for a Gemini response so cache
    hits (cached_content_token_count > 0) are verifiable from Cloud Logging,
    not just assumed from the cache having been created."""
    try:
        usage = response.usage_metadata
        cached = getattr(usage, "cached_content_token_count", None)
        prompt = getattr(usage, "prompt_token_count", None)
        total = getattr(usage, "total_token_count", None)
        print(f"[Usage] {label}: prompt_tokens={prompt} cached_tokens={cached} total_tokens={total}")
    except Exception as usage_err:
        print(f"[Usage] {label}: could not read usage_metadata ({usage_err})")


_MERMAID_DANGEROUS = re.compile(r"(?i)(<\s*/?\s*\w|script|javascript\s*:|on\w+\s*=|&#|data\s*:)")


def sanitize_mermaid(text):
    """Reduce model output to a safe Mermaid graph body, or "" if it isn't one.

    The model is prompted for graph source only, but a prompt is a request, not a
    guarantee -- and this content derives from untrusted meeting speech. Anything
    that could become markup is removed rather than trusted."""
    if not text:
        return ""
    body = text.replace("```mermaid", "").replace("```html", "").replace("```", "").strip()
    kept = []
    for line in body.splitlines():
        if _MERMAID_DANGEROUS.search(line):
            continue
        kept.append(line.replace("<", "").replace("&", "&amp;"))
    body = "\n".join(kept).strip()
    if not body:
        return ""
    # Must actually look like a Mermaid graph, or we are rendering arbitrary text.
    first = body.splitlines()[0].strip().lower()
    if not (first.startswith("flowchart") or first.startswith("graph")):
        return ""
    return body[:20000]


def render_diagram_html(mermaid_body):
    """Build the diagram page locally from a fixed template.

    The Content-Security-Policy is defence in depth: even if something slipped
    past sanitize_mermaid, inline script cannot execute and nothing can be
    exfiltrated to an arbitrary host."""
    return f"""<!DOCTYPE html>
<html dir="rtl">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; script-src https://cdn.jsdelivr.net 'unsafe-eval'; style-src 'unsafe-inline'; font-src data:;">
<style>body {{ background-color: #121212; color: white; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; font-family: Arial, sans-serif; }}</style>
</head>
<body>
<div class="mermaid">
{mermaid_body}
</div>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
</script>
</body>
</html>"""


def _sum_chunk_usage(chunk_reports):
    """Aggregate per-chunk token usage into one record-shaped dict.

    Pass 2 is no longer a single call, so the metrics record sums its windows."""
    if not chunk_reports:
        return None
    total = {"prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    seen = False
    for rep in chunk_reports:
        u = rep.get("usage") or {}
        for k in total:
            v = u.get(k)
            if isinstance(v, (int, float)):
                total[k] += v
                seen = True
    return total if seen else None


def _upload_and_wait(client, path, label):
    """Upload one file and block until Gemini reports it ACTIVE, with a bound."""
    gfile = with_retries(f"{label} upload", lambda: client.files.upload(file=path))
    poll_start = time.time()
    while gfile.state.name == "PROCESSING":
        if time.time() - poll_start > FILE_PROCESSING_MAX_WAIT_SEC:
            raise TimeoutError(
                f"Gemini file {gfile.name} ({label}) stuck in PROCESSING for over "
                f"{FILE_PROCESSING_MAX_WAIT_SEC}s -- aborting rather than hanging."
            )
        time.sleep(2)
        gfile = client.files.get(name=gfile.name)
    return gfile


def _transcribe_chunk(client, gfile, prompt, diar_header, prior_names, label):
    """One chunk, with a single retry when the output is provably degenerate.

    Retrying at a higher temperature is deliberate: near-greedy decoding is the
    classic condition for a repetition loop, and the observed failure was exactly
    that -- one token repeated 29,654 times until the output budget was gone."""
    attempts = [
        {"temperature": 0.1, "note": ""},
        {"temperature": 0.4, "note": "\n\nIMPORTANT: never repeat the same phrase or "
                                     "sentence consecutively. Each moment of the audio "
                                     "appears exactly once."},
    ]
    last_text, last_report = "", {}
    for i, cfg in enumerate(attempts):
        full_prompt = prompt + diar_header + chunking.names_hint(prior_names) + cfg["note"]
        resp = with_retries(f"{label} pass2", lambda: client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=[full_prompt, gfile],
            config=types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=cfg["temperature"],
                # Kept: removing thinking_config reproduces MAX_TOKENS truncation
                # (verified 2026-08-31, reconfirmed 2026-09-02).
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            ),
        ))
        log_token_usage(f"{label} pass2", resp)
        text = (resp.text or "").strip()
        finish = str(getattr(resp.candidates[0], "finish_reason", "")) if resp.candidates else ""

        degenerate = transcript_checks.detect_intra_line_degeneration(text)
        truncated = "MAX_TOKENS" in finish.upper()
        u = getattr(resp, "usage_metadata", None)
        last_text, last_report = text, {
            "attempt": i + 1, "finish_reason": finish,
            "max_intra_line_run": degenerate.get("max_run"),
            "retried": i > 0,
            "usage": {
                "prompt_tokens": getattr(u, "prompt_token_count", None),
                "cached_tokens": getattr(u, "cached_content_token_count", None),
                "output_tokens": getattr(u, "candidates_token_count", None),
                "total_tokens": getattr(u, "total_token_count", None),
            } if u is not None else None,
        }
        if not degenerate.get("detected") and not truncated:
            return text, last_report
        print(f"[WARNING] {label}: degenerate={degenerate.get('detected')} "
              f"(run={degenerate.get('max_run')}), truncated={truncated}"
              + (" -- retrying once" if i == 0 else " -- keeping best effort"))
    return last_text, last_report


def run_chunked_transcript(client, local_audio_path, stem, diar, duration_sec,
                           prompt, diar_header, fallback_file=None):
    """Transcribe a recording as independent windows and stitch them.

    Chunking is what CONTAINS the degenerate-loop failure: a loop can only ruin
    one window instead of the whole transcript, and one window is cheap to detect
    and retry. Boundaries fall in the silence between diarization turns, so no
    utterance is split and no overlap-deduplication is needed."""
    try:
        plan = chunking.plan_chunks(diar, duration_sec)
    except Exception as plan_err:
        print(f"[WARNING] Chunk planning failed ({plan_err}) -- single-pass transcript.")
        plan = []

    # Probe before committing to the chunked path. A recording we cannot slice
    # (unknown duration, non-PCM container, corrupt header) must still produce a
    # transcript from the already-uploaded whole file, exactly as before chunking
    # existed -- the same graceful-degradation contract used for diarization.
    sliceable = False
    if len(plan) > 1:
        try:
            with wave.open(local_audio_path, "rb"):
                sliceable = True
        except Exception as probe_err:
            print(f"[WARNING] Audio is not sliceable ({probe_err}) -- single-pass transcript.")

    if not sliceable:
        if fallback_file is None:
            raise RuntimeError("cannot chunk audio and no whole-file upload available")
        text, rep = _transcribe_chunk(client, fallback_file, prompt, diar_header, [], "single pass")
        rep.update({"index": 0, "chunked": False})
        return text, [rep]

    print(f"[Step 3b] Transcribing in {len(plan)} chunk(s).")

    parts, reports, names = [], [], []
    for idx, (start, end) in enumerate(plan):
        label = f"chunk {idx + 1}/{len(plan)}"
        cpath = chunking.chunk_paths("/tmp", stem, idx)
        gfile = None
        try:
            chunking.slice_wav(local_audio_path, start, end, cpath)
            gfile = _upload_and_wait(client, cpath, label)
            text, rep = _transcribe_chunk(client, gfile, prompt, diar_header, names, label)
            parts.append((start, text))
            rep.update({"index": idx, "start_sec": round(start, 1), "end_sec": round(end, 1)})
            reports.append(rep)
            for n in chunking.established_names(text):
                if n not in names:
                    names.append(n)
        finally:
            # Per-chunk cleanup, so a long meeting cannot fill /tmp (which is
            # instance memory on Cloud Run) or leak uploaded files on failure.
            try:
                if gfile:
                    client.files.delete(name=gfile.name)
            except Exception:
                pass
            try:
                if os.path.exists(cpath):
                    os.remove(cpath)
            except Exception:
                pass
    return chunking.stitch(parts), reports


@functions_framework.cloud_event
def tamlelan_handler(cloud_event):
    event_id = cloud_event["id"]
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]
    
    if file_name.startswith("locks/") or file_name.startswith("failed/") or not file_name.endswith(".wav"):
        return "Ignored", 200

    print(f"[Step 1] Waking up for Event ID {event_id} | File: {file_name}...")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    lock_blob = bucket.blob(f"locks/{event_id}.lock")
    try:
        lock_blob.upload_from_string("locked", if_generation_match=0)
    except PreconditionFailed:
        print(f"DUPLICATE EVENT DETECTED: Lock {event_id}.lock already exists. Aborting.")
        return "Duplicate Event Aborted", 200

    local_audio_path = f"/tmp/{os.path.basename(file_name)}"
    blob = bucket.blob(file_name)
    client = None
    gemini_file = None
    cache = None
    succeeded = False
    duration_sec = None
    diar = None
    summary_response = None
    transcript_response = None
    chunk_reports = []
    transcript_warnings = []
    transcript_report = {}
    transcript_usage = None
    flash_res = None
    diagram_needed = False
    pipeline_error = None
    duplicate_of_event_id = None
    started_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    # Content-hash dedup guard: the locks/{event_id}.lock mechanism above only
    # deduplicates retries of the SAME Eventarc event -- it does nothing for the
    # same audio content arriving as two genuinely distinct GCS objects/events
    # (e.g. a manual re-upload after a transient Gemini error). GCS already
    # computes crc32c on upload, so this check costs nothing extra and needs no
    # download. Uses the same atomic if_generation_match=0 pattern as the lock
    # above, so it's race-safe under concurrent invocations too.
    content_key = None

    try:
        # Inside the try deliberately: this is a live GCS call and it used to sit
        # OUTSIDE, so a transient failure here escaped with no finally block --
        # no failed/ preservation, no metrics record, and critically no
        # "[CRITICAL ERROR] Pipeline failed" line, which is the only thing the
        # Cloud Monitoring alert matches on. The meeting vanished silently.
        blob.reload()
        content_key = blob.crc32c or blob.md5_hash

        if content_key:
            dup_marker_blob = bucket.blob(f"content_hashes/{content_key}.json")
            if dup_marker_blob.exists():
                prior = json.loads(dup_marker_blob.download_as_bytes())
                duplicate_of_event_id = prior.get("event_id")
                print(f"[Step 1c] Duplicate content detected (content_key={content_key}), "
                      f"already successfully processed as event {duplicate_of_event_id}. Skipping pipeline.")
                succeeded = True
                return "Duplicate Content - Already Processed", 200

        blob.download_to_filename(local_audio_path)

        try:
            with wave.open(local_audio_path, 'rb') as wf:
                duration_sec = wf.getnframes() / float(wf.getframerate())
        except Exception as dur_err:
            print(f"[WARNING] Could not read audio duration from WAV header: {dur_err}")

        diar = load_diarization(bucket, file_name)
        diar_block = format_diarization_for_prompt(diar)
        if diar:
            print(f"[Step 1b] Diarization companion loaded: {diar.get('speaker_count')} speakers, {diar.get('channel_mode')}.")

        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=GEMINI_HTTP_TIMEOUT_MS),
        )
        
        gemini_file = with_retries("file upload", lambda: client.files.upload(file=local_audio_path))
        print("[Step 2] Uploaded to Gemini. Waiting for processing...")

        FILE_PROCESSING_MAX_WAIT_SEC = 300
        poll_start = time.time()
        while gemini_file.state.name == "PROCESSING":
            if time.time() - poll_start > FILE_PROCESSING_MAX_WAIT_SEC:
                raise TimeoutError(
                    f"Gemini file {gemini_file.name} stuck in PROCESSING for over "
                    f"{FILE_PROCESSING_MAX_WAIT_SEC}s -- aborting rather than hanging "
                    f"indefinitely (Cloud Run has a hard request timeout regardless)."
                )
            time.sleep(2)
            gemini_file = client.files.get(name=gemini_file.name)

        # Explicit context cache: the audio AND the diarization block are sent to
        # Gemini twice (Pass 1 + Pass 2) below, so caching them once here means
        # Pass 2 reads them back at ~10% of the input-token price instead of paying
        # full price again. For a large/complex meeting the diarization turn-by-turn
        # block can itself be tens of thousands of tokens, so it's cached alongside
        # the audio, not left as fresh per-pass text. Falls back to the uncached path
        # on any failure (e.g. audio too short to meet the cache's minimum token
        # count) -- this must never be able to break the pipeline.
        # SUPERSEDED BY CHUNKING (2026-09-03). The cache existed because the same
        # full audio was sent twice. Pass 2 is now chunked and reads its own
        # per-chunk files, so the cache would serve exactly ONE read -- and a
        # single-use cache costs more than not caching at all: you pay the full
        # input rate to WRITE it ($2.00/M), plus hourly storage ($4.50/M/hr), and
        # still pay to read it. Measured per meeting: $0.315 cached-once vs $0.214
        # uncached. Pass 1 therefore uses the inline path, which already existed as
        # the cache-failure fallback and is covered by
        # tests/test_main_cleanup.py::TestCacheCreationFallback.
        cache = None

        # ==========================================
        # PASS 1: THE STRUCTURED SUMMARY (JSON)
        # ==========================================
        print("[Step 3a] Running AI Analysis (Pass 1: Structured Summary)...")
        summary_prompt = """
        Analyze this meeting audio. All output text MUST be in fluent Hebrew.

        If the audio is stereo: the LEFT channel is the meeting operator (the person
        running the recording), and the RIGHT channel is remote/system audio (other
        participants on the call). Use this to help distinguish who is speaking --
        the right channel may itself contain multiple remote speakers mixed together,
        so treat it as "not the operator" rather than a single identified person unless
        a name is stated. If the audio is mono, no such channel distinction exists.

        1. Provide an executive summary, key topics, decisions, and action items.
        2. Evaluate if technical architectures or system designs were discussed (diagram_needed).

        Attendees: list ONLY the people who actually PARTICIPATED in this meeting -- that
        is, people whose own voice you can hear speaking in the recording. A person counts
        as an attendee only if they speak.

        Do NOT list a person merely because their name was said out loud. People who are
        talked ABOUT -- students, customers, suppliers, colleagues from other teams,
        absent managers, third parties -- are NOT attendees, no matter how often their
        name comes up. If you are unsure whether a given name belongs to a voice you can
        actually hear, leave that person OUT of attendees (see people_mentioned below
        instead).

        Most meetings have between 2 and 6 attendees. If your attendees list is longer
        than the number of distinct voices you can hear, it is wrong -- remove the names
        you cannot match to a voice.

        For each attendee, if their role or organization is stated (e.g. in an
        introduction), include it. If not stated, leave it null -- do not guess.

        People mentioned: separately, list every OTHER named person who came up in the
        conversation but did not speak -- students, customers, suppliers, third parties,
        anyone talked about rather than heard. For each, give a short context string
        describing what they were mentioned in relation to (e.g. "student receiving a
        certificate", "supplier for the closing ceremony catering").

        Decisions: for each decision-like statement, classify its status:
        - "decided": the group reached a final, settled conclusion.
        - "proposed": one option was suggested but not finally confirmed, or multiple
          options were still being weighed.
        - "open": the topic was raised but no resolution was reached at all.
        Never upgrade a hedge ("we're leaning toward X", "I'd like this to eventually be
        Y", "maybe we do Z") into "decided" -- if the speaker expressed a hope, intention,
        or one option among several, mark it "proposed" or "open" and record the hedge
        language itself in hedge_note.

        Action items: extract the owner and deadline ONLY if explicitly stated in the
        audio. If either is not stated, return null for that field -- never guess or
        infer a plausible owner or deadline.

        Do not assert a relationship or connection between two topics unless it was
        explicitly stated in the meeting. Two topics mentioned in the same meeting are
        not automatically related.

        Each key topic has a topic_id (assign short stable ids: "t1", "t2", ...). Each
        decision and action item has an optional related_topic_id field -- set it ONLY
        if that specific decision/action item was explicitly discussed as part of that
        topic. Leave it null by default. Being close together in time or sharing similar
        wording is NOT enough to link them -- the connection must have been actually
        stated. When in doubt, leave related_topic_id null.

        For every key topic, decision, and action item, also include source_quote: a
        short VERBATIM excerpt (a few words to one sentence) copied directly from the
        audio that supports it, in the original spoken language. Do not paraphrase the
        quote.
        """

        # Notice: full_transcript is REMOVED from the schema to save tokens
        schema = {
            "type": "OBJECT",
            "properties": {
                "executive_summary": {"type": "STRING"},
                "attendees": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "name": {"type": "STRING"},
                            "role": {"type": "STRING", "nullable": True},
                            "organization": {"type": "STRING", "nullable": True}
                        },
                        "required": ["name"]
                    }
                },
                "people_mentioned": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "name": {"type": "STRING"},
                            "context": {"type": "STRING", "nullable": True}
                        },
                        "required": ["name"]
                    }
                },
                "key_topics": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "topic_id": {"type": "STRING"},
                            "title": {"type": "STRING"},
                            "source_quote": {"type": "STRING", "nullable": True}
                        },
                        "required": ["topic_id", "title"]
                    }
                },
                "decisions_log": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "statement": {"type": "STRING"},
                            "status": {"type": "STRING", "enum": ["decided", "proposed", "open"]},
                            "hedge_note": {"type": "STRING", "nullable": True},
                            "source_quote": {"type": "STRING", "nullable": True},
                            "related_topic_id": {"type": "STRING", "nullable": True}
                        },
                        "required": ["statement", "status"]
                    }
                },
                "action_items": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "task": {"type": "STRING"},
                            "owner": {"type": "STRING", "nullable": True},
                            "deadline": {"type": "STRING", "nullable": True},
                            "source_quote": {"type": "STRING", "nullable": True},
                            "related_topic_id": {"type": "STRING", "nullable": True}
                        },
                        "required": ["task"]
                    }
                },
                "diagram_needed": {"type": "BOOLEAN"}
            },
            "required": ["executive_summary", "attendees", "people_mentioned", "key_topics", "decisions_log", "action_items", "diagram_needed"]
        }
        
        summary_response = with_retries("Pass 1 (summary)", lambda: client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=[summary_prompt] if cache else [summary_prompt + diar_block, gemini_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.2,
                cached_content=cache.name if cache else None
            )
        ))
        log_token_usage("Pass 1 (summary)", summary_response)

        raw_text = summary_response.text.replace("```json", "").replace("```", "").strip()
        
        try:
            res_data = json.loads(raw_text)
            diagram_needed = bool(res_data.get('diagram_needed'))
            print(f"[Step 4a] Summary Analysis complete and JSON parsed successfully.")
        except json.JSONDecodeError as e:
            print(f"[CRITICAL ERROR] JSON Parsing failed: {e}")
            raise e

        # ==========================================
        # PASS 2: THE FULL VERBATIM TRANSCRIPT (PLAIN TEXT)
        # ==========================================
        print("[Step 3b] Running AI Analysis (Pass 2: Full Transcript)...")
        transcript_prompt = """
        Please provide a highly accurate, full verbatim transcript of this entire meeting
        audio. Ensure all text is in fluent Hebrew. Do not summarize.

        If the audio is stereo: the LEFT channel is the meeting operator (the person
        running the recording), and the RIGHT channel is remote/system audio (other
        participants on the call). Use this to distinguish who is speaking -- the right
        channel may itself contain multiple remote speakers mixed together, so treat it
        as "not the operator" rather than a single identified person unless a name is
        stated. If the audio is mono, no such channel distinction exists.

        Label every line with the speaker. Use this exact line format:

          M:SS [SPEAKER]: <what was said>

        Rules for SPEAKER:
        - If a person's real name is established anywhere in the audio (they introduce
          themselves, or someone addresses them by name), use that real name on EVERY
          line they speak -- including lines earlier in the meeting, before the name was
          said.
        - If you cannot establish a real name, use a stable generic label: "דובר 1",
          "דובר 2", ... Reuse the same generic label for the same voice throughout.
          Never renumber mid-meeting.
        - NEVER invent or guess a name. A name mentioned in conversation does not mean
          that person is speaking.
        - For non-speech stretches use "M:SS - M:SS: [שקט]" with no speaker label.

        Start a new line whenever the speaker changes. Output ONLY the raw transcript
        text -- no preamble, no headings, no commentary.
        """
        
        # Pass 2 is chunked (2026-09-03). Two measured failures drove this:
        #  * With the diarization turn list in the prompt, the model emitted one
        #    line per segment and copied its timestamp and label -- echo 1.00,
        #    41% duplicated content, finish_reason=STOP. Complete-looking, fake.
        #  * Without it, the model transcribes for real and names every speaker
        #    correctly, but entered a degenerate loop at 18:05 repeating one token
        #    29,654 times until the 65,536 output cap was gone (MAX_TOKENS, 40%
        #    coverage).
        # So: send the roster header WITHOUT the turn list, and chunk so a loop can
        # only ruin one window -- which _transcribe_chunk detects and retries.
        diar_header = format_diarization_for_prompt(diar, include_turns=False)
        full_transcript_text, chunk_reports = run_chunked_transcript(
            client, local_audio_path, os.path.basename(file_name), diar,
            duration_sec, transcript_prompt, diar_header, fallback_file=gemini_file,
        )
        if not full_transcript_text:
            full_transcript_text = "לא זוהה מלל."
        print(f"[Step 4b] Transcript generation complete.")

        # Deterministic, zero-cost quality gate on the stitched transcript. Flags
        # only -- the full text is always delivered unchanged. Before this, the
        # transcript was checked for exactly one property (non-empty), which is why
        # a 41%-duplicated transcript shipped to Drive looking like a success.
        transcript_warnings, transcript_report = transcript_checks.verify_transcript(
            full_transcript_text, diar=diar, duration_sec=duration_sec,
        )
        transcript_report["chunks"] = chunk_reports
        transcript_usage = _sum_chunk_usage(chunk_reports)
        if transcript_warnings:
            # Log the English check names, never the Hebrew banner text: stdout is
            # not guaranteed to be UTF-8 (a cp1252 console raised UnicodeEncodeError
            # here and killed the whole run), and Cloud Logging is easier to filter
            # and alert on with stable ASCII keys.
            fired = [name for name, res in transcript_report.items()
                     if isinstance(res, dict) and res.get("detected")]
            print(f"[WARNING] Transcript quality checks flagged "
                  f"{len(transcript_warnings)} issue(s) -- delivering with a banner. "
                  f"Checks fired: {', '.join(fired) or 'finish_reason'}")

        # ==========================================
        # PASS 3 (LOCAL, NO API CALL): GROUNDING VERIFICATION
        # ==========================================
        # Fuzzy-match each source_quote against the transcript Pass 2 already
        # produced. Flags likely-hallucinated claims rather than dropping them --
        # a false-negative fuzzy match must not silently delete a true claim.
        # Also checks related_topic_id referential integrity: a decision/action item
        # claiming a link to a topic_id that doesn't actually exist among the
        # extracted key_topics is flagged too (Defect 1 mitigation, Option B).
        topics = res_data.get('key_topics') or []
        known_topic_ids = {t.get('topic_id') for t in topics if isinstance(t, dict) and t.get('topic_id')}

        for t in topics:
            if isinstance(t, dict):
                t['_grounded'] = is_grounded(t.get('source_quote'), full_transcript_text)

        for d in (res_data.get('decisions_log') or []):
            if isinstance(d, dict):
                quote_ok = is_grounded(d.get('source_quote'), full_transcript_text)
                topic_ref = d.get('related_topic_id')
                topic_ref_ok = (topic_ref is None) or (topic_ref in known_topic_ids)
                d['_grounded'] = quote_ok and topic_ref_ok

        for item in (res_data.get('action_items') or []):
            if isinstance(item, dict):
                quote_ok = is_grounded(item.get('source_quote'), full_transcript_text)
                topic_ref = item.get('related_topic_id')
                topic_ref_ok = (topic_ref is None) or (topic_ref in known_topic_ids)
                item['_grounded'] = quote_ok and topic_ref_ok

        # Attendee cross-check against the local diarization companion, if one was
        # loaded: a deterministic backstop for the attendees-over-inclusion defect --
        # if diarization detected N distinct voices and Gemini lists more than N+1
        # attendees, something is wrong. Flag, don't drop -- same philosophy as the
        # grounding checks above. The +1 tolerance absorbs diarization merging two
        # similar-sounding voices into one cluster.
        attendee_count_ok = True
        if diar and isinstance(diar.get('speaker_count'), int) and diar['speaker_count'] > 0:
            attendee_count_ok = len(res_data.get('attendees') or []) <= diar['speaker_count'] + 1

        print("[Step 4c] Grounding verification complete.")

        # ==========================================
        # BUILD MARKDOWN FILES
        # ==========================================
        executive_summary = res_data.get('executive_summary') or "לא זוהה מידע בולט בהקלטה."
        attendees = res_data.get('attendees') or []
        people_mentioned = res_data.get('people_mentioned') or []
        key_topics = res_data.get('key_topics') or []
        decisions = res_data.get('decisions_log') or []
        action_items = res_data.get('action_items') or []

        STATUS_LABELS = {"decided": "הוחלט", "proposed": "הוצע", "open": "פתוח"}
        topic_titles = {t.get('topic_id'): t.get('title') for t in key_topics
                         if isinstance(t, dict) and t.get('topic_id')}

        def topic_link_suffix(related_topic_id):
            if not related_topic_id:
                return ""
            title = topic_titles.get(related_topic_id)
            return f" _(קשור לנושא: {title})_" if title else " ⚠ _(הפניה לנושא לא תקין)_"

        md_summary = f"<div dir='rtl'>\n# סיכום פגישה\n\n"
        md_summary += f"## תקציר מנהלים\n{executive_summary}\n\n"

        attendee_warn = " ⚠ _(מספר המשתתפים אינו תואם למספר הקולות שזוהו בהקלטה)_" if not attendee_count_ok else ""
        md_summary += f"## משתתפים{attendee_warn}\n"
        if not attendees: md_summary += "* לא זוהו משתתפים\n"
        else:
            for a in attendees:
                if isinstance(a, dict):
                    name = a.get('name') or '-'
                    role = a.get('role')
                    org = a.get('organization')
                    detail = " (" + ", ".join(x for x in (role, org) if x) + ")" if (role or org) else ""
                    md_summary += f"* {name}{detail}\n"
                else:
                    md_summary += f"* {a}\n"

        md_summary += "\n## אנשים שהוזכרו\n"
        if not people_mentioned: md_summary += "* לא הוזכרו אנשים נוספים\n"
        else:
            for p in people_mentioned:
                if isinstance(p, dict):
                    name = p.get('name') or '-'
                    context = p.get('context')
                    detail = f" ({context})" if context else ""
                    md_summary += f"* {name}{detail}\n"
                else:
                    md_summary += f"* {p}\n"

        md_summary += "\n## נושאים מרכזיים\n"
        if not key_topics: md_summary += "* לא זוהו נושאים מרכזיים\n"
        else:
            for t in key_topics:
                if isinstance(t, dict):
                    title = t.get('title') or '-'
                    warn_prefix = "⚠ " if not t.get('_grounded', True) else ""
                    md_summary += f"* {warn_prefix}{title}\n"
                else:
                    md_summary += f"* {t}\n"

        md_summary += "\n## החלטות שהתקבלו\n"
        if not decisions: md_summary += "* לא התקבלו החלטות\n"
        else:
            for d in decisions:
                if isinstance(d, dict):
                    statement = d.get('statement') or '-'
                    status = STATUS_LABELS.get(d.get('status'), d.get('status') or '-')
                    hedge = d.get('hedge_note')
                    hedge_suffix = f" _({hedge})_" if hedge else ""
                    link_suffix = topic_link_suffix(d.get('related_topic_id'))
                    warn_prefix = "⚠ " if not d.get('_grounded', True) else ""
                    md_summary += f"* {warn_prefix}**[{status}]** {statement}{hedge_suffix}{link_suffix}\n"
                else:
                    md_summary += f"* {d}\n"

        md_summary += "\n## משימות לביצוע\n| משימה | אחראי | יעד |\n|---|---|---|\n"
        if not action_items:
            md_summary += "| לא זוהו משימות | - | - |\n"
        else:
            for item in action_items:
                if isinstance(item, dict):
                    warn_prefix = "⚠ " if not item.get('_grounded', True) else ""
                    link_suffix = topic_link_suffix(item.get('related_topic_id'))
                    md_summary += f"| {warn_prefix}{item.get('task') or '-'}{link_suffix} | {item.get('owner') or '-'} | {item.get('deadline') or '-'} |\n"
                else:
                    md_summary += f"| {item} | - | - |\n"
        md_summary += "\n</div>"

        # Build Transcript Markdown using the result from Pass 2
        # The banner is empty when every check passed, so a clean meeting is
        # delivered exactly as before. The transcript itself is never modified.
        transcript_banner = transcript_checks.warning_banner(transcript_warnings)
        md_transcript = (f"<div dir='rtl'>\n# תמלול מלא\n\n"
                         f"{transcript_banner}{full_transcript_text}\n</div>")

        # ==========================================
        # SEND TO GOOGLE DRIVE
        # ==========================================
        webhook_url = os.environ.get("APPS_SCRIPT_URL")
        folder_id = os.environ.get("DRIVE_FOLDER_ID")
        webhook_secret = os.environ.get("WEBHOOK_SECRET")

        def send_to_drive(filename, content):
            """Deliver one file to Drive via the Apps Script webhook.

            The webhook answers HTTP 200 with {"status":"error"} for a bad secret
            or a disallowed folder -- it does NOT use an error status code. The
            previous version parsed that body and threw it away, so a rejected
            write looked identical to a successful one: the pipeline reported
            success, no alert fired, and the `finally` block then DELETED the
            source recording. That is a silent, permanent data-loss path, and the
            status check below is what closes it.

            Also bounded and retried: urlopen had no timeout, so a hung Apps
            Script blocked until Cloud Run killed the request -- and this is the
            one component with a confirmed real production failure (HTTP 404 on
            2026-08-12), yet it was the only external call with no retry.
            """
            payload = json.dumps({"filename": filename, "content": content, "folder_id": folder_id, "secret": webhook_secret}).encode('utf-8')

            def _post():
                req = urllib.request.Request(webhook_url, data=payload, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=DRIVE_WEBHOOK_TIMEOUT_SEC) as res:
                    body = res.read().decode()
                # Fail on a KNOWN-BAD status, not on the absence of a known-good one.
                # The deployed Apps Script is not version-controlled with this repo,
                # so requiring an exact "success" string would risk failing every
                # delivery if its response shape ever differs. The documented failure
                # modes all return {"status":"error", "message": ...}, and that is
                # precisely the case that used to be swallowed.
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    print(f"[WARNING] Drive webhook returned non-JSON for {filename}: "
                          f"{body[:200]!r} -- treating as delivered.")
                    return {"status": "unknown", "raw": body[:200]}
                if str(parsed.get("status", "")).lower() == "error":
                    raise RuntimeError(
                        f"Drive webhook rejected {filename}: {parsed.get('message') or parsed}")
                return parsed

            return with_retries(f"Drive upload ({filename})", _post)

        base_name = time.strftime('%Y%m%d_%H%M%S')
        print("[Step 5a] Sending MD Summary to Google Drive...")
        send_to_drive(f"Summary_{base_name}.md", md_summary)
        
        print("[Step 5b] Sending Full Transcript to Google Drive...")
        send_to_drive(f"Transcript_{base_name}.md", md_transcript)

        # ==========================================
        # DIAGRAM GENERATION
        # ==========================================
        if res_data.get('diagram_needed'):
            # Isolated: the diagram is a nice-to-have generated AFTER both expensive
            # Gemini passes and both Drive deliveries have already succeeded. It used
            # to share the main try, so one flash-lite hiccup failed the entire run,
            # sent the audio to failed/, and wrote no content-hash marker -- meaning
            # a re-upload paid for everything again and duplicated the Drive files.
            try:
                print("[Step 6] Generating Diagram with Flash...")
                # The model is asked for the Mermaid GRAPH BODY ONLY, never for HTML.
                # It previously authored the whole page ("Output ONLY raw HTML") from a
                # summary derived from untrusted meeting speech, and the result was
                # written to Drive as .html after nothing but two ``` strips -- a
                # second-order injection path straight into a file the user opens in a
                # browser. The page is now built locally from a fixed template.
                flash_prompt = f"""
            Read the Hebrew meeting summary below and produce a Mermaid.js flowchart
            body describing the technical architecture discussed.

            Output ONLY Mermaid graph source, starting with "flowchart TD".
            Do NOT output HTML, <script>, <style>, markdown fences, or commentary.

            SUMMARY:
            {md_summary}
            """
                flash_res = client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=flash_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="text/plain",
                    )
                )
                log_token_usage("Diagram (flash-lite)", flash_res)
                mermaid_body = sanitize_mermaid(flash_res.text)
                if mermaid_body:
                    print("[Step 7] Sending HTML Diagram to Drive...")
                    send_to_drive(f"Diagram_{base_name}.html",
                                  render_diagram_html(mermaid_body))
                else:
                    print("[WARNING] Diagram output failed validation -- skipping diagram.")
            except Exception as diagram_err:
                print(f"[WARNING] Diagram generation failed, continuing: {diagram_err}")

        succeeded = True

    except Exception as e:
        print(f"[CRITICAL ERROR] Pipeline failed: {e}")
        pipeline_error = str(e)[:500]
        raise e

    finally:
        print("[Step 8] Cleaning up...")
        preserved_to_failed = False
        try:
            if blob.exists():
                if succeeded:
                    blob.delete()
                else:
                    print(f"[WARNING] Processing failed -- preserving source audio at failed/{file_name} instead of deleting it.")
                    # copy_blob is server-side: no download, no re-upload, and no
                    # second 169 MB allocation on the failure path, which is the
                    # most memory-constrained path in the whole pipeline.
                    bucket.copy_blob(blob, bucket, f"failed/{file_name}")
                    blob.delete()
                    preserved_to_failed = True
        except Exception as cleanup_err:
            print(f"[WARNING] Cleanup could not preserve/delete source blob: {cleanup_err}")

        # Release the Eventarc lock unless the audio was moved to failed/.
        #
        # The lock is written before processing and was never deleted anywhere, so
        # every meeting left one behind forever. Worse, Eventarc redelivery reuses
        # the same event id, so a crashed run's retry hit the lock, returned 200
        # "Duplicate Event Aborted" and ACKED the message -- the only automatic
        # retry the system has was defeated by the mechanism meant to make it safe.
        #
        # It is NOT released when the audio was preserved under failed/: the source
        # is gone from the inbox, so a redelivery could only fail again. In that
        # case the lock correctly stops a pointless retry loop, and failed/ plus the
        # [CRITICAL ERROR] alert are the recovery path.
        try:
            if not preserved_to_failed:
                lock_blob.delete()
        except Exception as lock_err:
            print(f"[WARNING] Could not release lock {event_id}: {lock_err}")
        try:
            companion_name = companion_name_for(file_name)
            if companion_name:
                companion_blob = bucket.blob(companion_name)
                if companion_blob.exists():
                    if succeeded:
                        companion_blob.delete()
                    else:
                        bucket.blob(f"failed/{companion_name}").upload_from_string(companion_blob.download_as_bytes())
                        companion_blob.delete()
        except Exception as cleanup_err:
            print(f"[WARNING] Cleanup could not preserve/delete diarization companion: {cleanup_err}")
        try:
            if gemini_file and client: client.files.delete(name=gemini_file.name)
        except Exception: pass
        try:
            if cache and client: client.caches.delete(name=cache.name)
        except Exception: pass
        try:
            if os.path.exists(local_audio_path): os.remove(local_audio_path)
        except Exception: pass
        if succeeded and content_key and not duplicate_of_event_id:
            try:
                bucket.blob(f"content_hashes/{content_key}.json").upload_from_string(
                    json.dumps({"event_id": event_id, "processed_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                "file_stem": os.path.basename(file_name)}),
                    if_generation_match=0)
            except PreconditionFailed:
                pass  # a concurrent invocation for the same content already won the race
            except Exception as marker_err:
                print(f"[WARNING] Could not write content-hash dedup marker: {marker_err}")
        try:
            def _safe_num(v):
                return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

            def _usage_dict(resp):
                u = getattr(resp, "usage_metadata", None) if resp is not None else None
                if u is None:
                    return None
                prompt = _safe_num(getattr(u, "prompt_token_count", None))
                output = _safe_num(getattr(u, "candidates_token_count", None))
                total = _safe_num(getattr(u, "total_token_count", None))
                # Recorded explicitly rather than left to be derived. Thinking bills
                # at the OUTPUT rate but appears in neither prompt_token_count nor
                # candidates_token_count, so it was invisible to every cost figure
                # this project produced until a Cloud Billing cross-check on
                # 2026-09-04 showed spend was 2x what the records implied.
                thinking = None
                if all(isinstance(x, (int, float)) for x in (prompt, output, total)):
                    thinking = max(0, int(total) - int(prompt) - int(output))
                reported = _safe_num(getattr(u, "thoughts_token_count", None))
                return {
                    "prompt_tokens": prompt,
                    "cached_tokens": _safe_num(getattr(u, "cached_content_token_count", None)),
                    "output_tokens": output,
                    "thinking_tokens": reported if reported is not None else thinking,
                    "total_tokens": total,
                }

            cache_write_tokens = None
            if cache is not None:
                cache_write_tokens = _safe_num(getattr(getattr(cache, "usage_metadata", None), "total_token_count", None))

            metrics_record = {
                # v2 (2026-09-03): explicit context caching removed (cache_used is
                # now always False), pass2_transcript usage is summed across chunks
                # rather than one call, and transcript_quality was added so the
                # duplication defect becomes a measurable rate instead of an anecdote.
                "schema_version": 2,
                "event_id": event_id,
                "file_stem": os.path.basename(file_name)[:-4] if file_name.endswith(".wav") else os.path.basename(file_name),
                "started_at": started_at,
                "finished_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "success": succeeded,
                "error": pipeline_error,
                "duplicate_of_event_id": duplicate_of_event_id,
                "duration_sec": _safe_num(duration_sec),
                "speaker_count": diar.get("speaker_count") if diar else None,
                "channel_mode": diar.get("channel_mode") if diar else None,
                "cache_used": cache is not None,
                "cache_write_tokens": cache_write_tokens,
                "diagram_generated": diagram_needed,
                "transcript_quality": {
                    "warning_count": len(transcript_warnings),
                    "checks_fired": [n for n, r in transcript_report.items()
                                     if isinstance(r, dict) and r.get("detected")],
                    "chunk_count": len(chunk_reports),
                    "chunks_retried": sum(1 for r in chunk_reports if r.get("retried")),
                    "max_intra_line_run": max(
                        [r.get("max_intra_line_run") or 0 for r in chunk_reports] or [0]),
                },
                "usage": {
                    "pass1_summary": _usage_dict(summary_response),
                    "pass2_transcript": transcript_usage,
                    "diagram_generation": _usage_dict(flash_res),
                },
            }
            safe_event_id = str(event_id).replace("/", "_")
            bucket.blob(f"metrics/{safe_event_id}.json").upload_from_string(
                json.dumps(metrics_record, ensure_ascii=False), content_type="application/json")
        except Exception as metrics_err:
            print(f"[WARNING] Could not write metrics record: {metrics_err}")
        print("--- PIPELINE FINISHED ---")

    return "Success", 200