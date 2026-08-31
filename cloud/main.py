import os
import json
import time
import wave
import urllib.request
import functions_framework
from google import genai
from google.genai import types
from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed
from rapidfuzz import fuzz

GROUNDING_THRESHOLD = 80


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
        return diar
    except Exception as e:
        print(f"[WARNING] Could not load diarization companion for {file_name}: {e}")
        return None


def _format_mmss(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def format_diarization_for_prompt(diar):
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

    if len(segments) > MAX_PROMPT_SEGMENTS:
        lines.append(
            f"(Turn-by-turn list omitted -- {len(segments)} segments exceeds the prompt "
            f"limit. Use the speaker count and channel information above.)"
        )
    else:
        lines.append("Speaking turns (start-end, label):")
        for seg in segments:
            if len(seg) < 3:
                continue
            start, end, label = seg[0], seg[1], seg[2]
            lines.append(f"{_format_mmss(start)}-{_format_mmss(end)} {label}")

    lines.append("")
    lines.append(
        "Use these turn boundaries as authoritative for WHEN the speaker changes and for "
        "HOW MANY distinct people speak. The labels are anonymous -- resolve each one to a "
        "real name yourself from the audio (e.g. when someone introduces themselves) and "
        "use the real name in your output. If you cannot resolve a label to a real name, "
        "keep the anonymous label."
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
    diagram_needed = False
    pipeline_error = None
    started_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    try:
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
        client = genai.Client(api_key=api_key)
        
        gemini_file = client.files.upload(file=local_audio_path)
        print("[Step 2] Uploaded to Gemini. Waiting for processing...")
        
        while gemini_file.state.name == "PROCESSING":
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
        cache_contents = [gemini_file]
        if diar_block:
            cache_contents.append(diar_block)

        cache = None
        try:
            cache = client.caches.create(
                model='gemini-3.1-pro-preview',
                config=types.CreateCachedContentConfig(
                    contents=cache_contents,
                    ttl="600s",
                )
            )
            print(f"[Step 2b] Audio cached ({cache.name}) for reuse across both passes.")
        except Exception as cache_err:
            print(f"[WARNING] Context cache creation failed, falling back to uncached calls: {cache_err}")
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
        
        summary_response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=[summary_prompt] if cache else [summary_prompt + diar_block, gemini_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.2,
                cached_content=cache.name if cache else None
            )
        )
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
        
        transcript_response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=[transcript_prompt] if cache else [transcript_prompt + diar_block, gemini_file],
            config=types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=0.1,
                cached_content=cache.name if cache else None
            )
        )
        log_token_usage("Pass 2 (transcript)", transcript_response)

        full_transcript_text = transcript_response.text.strip()
        if not full_transcript_text:
            full_transcript_text = "לא זוהה מלל."
        print(f"[Step 4b] Transcript generation complete.")

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
        md_transcript = f"<div dir='rtl'>\n# תמלול מלא\n\n{full_transcript_text}\n</div>"

        # ==========================================
        # SEND TO GOOGLE DRIVE
        # ==========================================
        webhook_url = os.environ.get("APPS_SCRIPT_URL")
        folder_id = os.environ.get("DRIVE_FOLDER_ID")
        webhook_secret = os.environ.get("WEBHOOK_SECRET")

        def send_to_drive(filename, content):
            payload = json.dumps({"filename": filename, "content": content, "folder_id": folder_id, "secret": webhook_secret}).encode('utf-8')
            req = urllib.request.Request(webhook_url, data=payload, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as res:
                return json.loads(res.read().decode())

        base_name = time.strftime('%Y%m%d_%H%M%S')
        print("[Step 5a] Sending MD Summary to Google Drive...")
        send_to_drive(f"Summary_{base_name}.md", md_summary)
        
        print("[Step 5b] Sending Full Transcript to Google Drive...")
        send_to_drive(f"Transcript_{base_name}.md", md_transcript)

        # ==========================================
        # DIAGRAM GENERATION
        # ==========================================
        if res_data.get('diagram_needed'):
            print("[Step 6] Generating Diagram with Flash...")
            flash_prompt = f"""
            Based on this Hebrew meeting summary, generate a valid, dark-themed HTML file containing a Mermaid.js flowchart (Flowchart TD) mapping the technical architecture discussed. Output ONLY raw HTML.
            
            Template:
            <!DOCTYPE html>
            <html dir="rtl">
            <head>
                <meta charset="UTF-8">
                <script type="module">
                    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                    mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
                </script>
                <style>body {{ background-color: #121212; color: white; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; font-family: Arial, sans-serif; }}</style>
            </head>
            <body>
                <div class="mermaid">
                %% MERMAID CODE HERE %%
                </div>
            </body>
            </html>
            
            SUMMARY: {md_summary}
            """
            
            flash_res = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=flash_prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            html_content = flash_res.text.replace("```html", "").replace("```", "").strip()
            
            print("[Step 7] Sending HTML Diagram to Drive...")
            send_to_drive(f"Diagram_{base_name}.html", html_content)

        succeeded = True

    except Exception as e:
        print(f"[CRITICAL ERROR] Pipeline failed: {e}")
        pipeline_error = str(e)[:500]
        raise e

    finally:
        print("[Step 8] Cleaning up...")
        try:
            if blob.exists():
                if succeeded:
                    blob.delete()
                else:
                    print(f"[WARNING] Processing failed -- preserving source audio at failed/{file_name} instead of deleting it.")
                    bucket.blob(f"failed/{file_name}").upload_from_string(blob.download_as_bytes())
                    blob.delete()
        except Exception as cleanup_err:
            print(f"[WARNING] Cleanup could not preserve/delete source blob: {cleanup_err}")
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
        try:
            def _safe_num(v):
                return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

            def _usage_dict(resp):
                u = getattr(resp, "usage_metadata", None) if resp is not None else None
                if u is None:
                    return None
                return {
                    "prompt_tokens": _safe_num(getattr(u, "prompt_token_count", None)),
                    "cached_tokens": _safe_num(getattr(u, "cached_content_token_count", None)),
                    "output_tokens": _safe_num(getattr(u, "candidates_token_count", None)),
                    "total_tokens": _safe_num(getattr(u, "total_token_count", None)),
                }

            cache_write_tokens = None
            if cache is not None:
                cache_write_tokens = _safe_num(getattr(getattr(cache, "usage_metadata", None), "total_token_count", None))

            metrics_record = {
                "schema_version": 1,
                "event_id": event_id,
                "file_stem": os.path.basename(file_name)[:-4] if file_name.endswith(".wav") else os.path.basename(file_name),
                "started_at": started_at,
                "finished_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "success": succeeded,
                "error": pipeline_error,
                "duration_sec": _safe_num(duration_sec),
                "speaker_count": diar.get("speaker_count") if diar else None,
                "channel_mode": diar.get("channel_mode") if diar else None,
                "cache_used": cache is not None,
                "cache_write_tokens": cache_write_tokens,
                "diagram_generated": diagram_needed,
                "usage": {
                    "pass1_summary": _usage_dict(summary_response),
                    "pass2_transcript": _usage_dict(transcript_response),
                },
            }
            safe_event_id = str(event_id).replace("/", "_")
            bucket.blob(f"metrics/{safe_event_id}.json").upload_from_string(
                json.dumps(metrics_record, ensure_ascii=False), content_type="application/json")
        except Exception as metrics_err:
            print(f"[WARNING] Could not write metrics record: {metrics_err}")
        print("--- PIPELINE FINISHED ---")

    return "Success", 200