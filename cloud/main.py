import os
import json
import time
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

@functions_framework.cloud_event
def tamlelan_handler(cloud_event):
    event_id = cloud_event["id"]
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]
    
    if file_name.startswith("locks/") or not file_name.endswith(".wav"):
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
    succeeded = False

    try:
        blob.download_to_filename(local_audio_path)

        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        gemini_file = client.files.upload(file=local_audio_path)
        print("[Step 2] Uploaded to Gemini. Waiting for processing...")
        
        while gemini_file.state.name == "PROCESSING":
            time.sleep(2)
            gemini_file = client.files.get(name=gemini_file.name)
        
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

        Attendees: list every person mentioned by name. If their role or organization is
        stated (e.g. in an introduction), include it. If not stated, leave it null -- do
        not guess.

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

        For every decision and every action item, also include source_quote: a short
        VERBATIM excerpt (a few words to one sentence) copied directly from the audio
        that supports it, in the original spoken language. Do not paraphrase the quote.
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
                "key_topics": {"type": "ARRAY", "items": {"type": "STRING"}},
                "decisions_log": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "statement": {"type": "STRING"},
                            "status": {"type": "STRING", "enum": ["decided", "proposed", "open"]},
                            "hedge_note": {"type": "STRING", "nullable": True},
                            "source_quote": {"type": "STRING", "nullable": True}
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
                            "source_quote": {"type": "STRING", "nullable": True}
                        },
                        "required": ["task"]
                    }
                },
                "diagram_needed": {"type": "BOOLEAN"}
            },
            "required": ["executive_summary", "attendees", "key_topics", "decisions_log", "action_items", "diagram_needed"]
        }
        
        summary_response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=[summary_prompt, gemini_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                response_schema=schema,
                temperature=0.2
            )
        )
        
        raw_text = summary_response.text.replace("```json", "").replace("```", "").strip()
        
        try:
            res_data = json.loads(raw_text)
            print(f"[Step 4a] Summary Analysis complete and JSON parsed successfully.")
        except json.JSONDecodeError as e:
            print(f"[CRITICAL ERROR] JSON Parsing failed: {e}")
            raise e

        # ==========================================
        # PASS 2: THE FULL VERBATIM TRANSCRIPT (PLAIN TEXT)
        # ==========================================
        print("[Step 3b] Running AI Analysis (Pass 2: Full Transcript)...")
        transcript_prompt = """
        Please provide a highly accurate, full verbatim transcript of this entire meeting audio. 
        Ensure all text is in fluent Hebrew. Do not summarize. Output ONLY the raw transcript text.
        """
        
        transcript_response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=[transcript_prompt, gemini_file],
            config=types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=0.1
            )
        )
        
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
        for d in (res_data.get('decisions_log') or []):
            if isinstance(d, dict):
                d['_grounded'] = is_grounded(d.get('source_quote'), full_transcript_text)
        for item in (res_data.get('action_items') or []):
            if isinstance(item, dict):
                item['_grounded'] = is_grounded(item.get('source_quote'), full_transcript_text)
        print("[Step 4c] Grounding verification complete.")

        # ==========================================
        # BUILD MARKDOWN FILES
        # ==========================================
        executive_summary = res_data.get('executive_summary') or "לא זוהה מידע בולט בהקלטה."
        attendees = res_data.get('attendees') or []
        key_topics = res_data.get('key_topics') or []
        decisions = res_data.get('decisions_log') or []
        action_items = res_data.get('action_items') or []

        STATUS_LABELS = {"decided": "הוחלט", "proposed": "הוצע", "open": "פתוח"}

        md_summary = f"<div dir='rtl'>\n# סיכום פגישה\n\n"
        md_summary += f"## תקציר מנהלים\n{executive_summary}\n\n"

        md_summary += "## משתתפים\n"
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

        md_summary += "\n## נושאים מרכזיים\n"
        if not key_topics: md_summary += "* לא זוהו נושאים מרכזיים\n"
        else:
            for t in key_topics: md_summary += f"* {t}\n"

        md_summary += "\n## החלטות שהתקבלו\n"
        if not decisions: md_summary += "* לא התקבלו החלטות\n"
        else:
            for d in decisions:
                if isinstance(d, dict):
                    statement = d.get('statement') or '-'
                    status = STATUS_LABELS.get(d.get('status'), d.get('status') or '-')
                    hedge = d.get('hedge_note')
                    hedge_suffix = f" _({hedge})_" if hedge else ""
                    warn_prefix = "⚠ " if not d.get('_grounded', True) else ""
                    md_summary += f"* {warn_prefix}**[{status}]** {statement}{hedge_suffix}\n"
                else:
                    md_summary += f"* {d}\n"

        md_summary += "\n## משימות לביצוע\n| משימה | אחראי | יעד |\n|---|---|---|\n"
        if not action_items:
            md_summary += "| לא זוהו משימות | - | - |\n"
        else:
            for item in action_items:
                if isinstance(item, dict):
                    warn_prefix = "⚠ " if not item.get('_grounded', True) else ""
                    md_summary += f"| {warn_prefix}{item.get('task') or '-'} | {item.get('owner') or '-'} | {item.get('deadline') or '-'} |\n"
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
            if gemini_file and client: client.files.delete(name=gemini_file.name)
        except Exception: pass
        try:
            if os.path.exists(local_audio_path): os.remove(local_audio_path)
        except Exception: pass
        print("--- PIPELINE FINISHED ---")

    return "Success", 200