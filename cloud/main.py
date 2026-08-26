import os
import json
import time
import urllib.request
import functions_framework
from google import genai
from google.genai import types
from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed

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
        1. Provide an executive summary, key topics, decisions, and action items.
        2. Evaluate if technical architectures or system designs were discussed.
        """
        
        # Notice: full_transcript is REMOVED from the schema to save tokens
        schema = {
            "type": "OBJECT",
            "properties": {
                "executive_summary": {"type": "STRING"},
                "key_topics": {"type": "ARRAY", "items": {"type": "STRING"}},
                "decisions_log": {"type": "ARRAY", "items": {"type": "STRING"}},
                "action_items": {"type": "ARRAY", "items": {"type": "STRING"}},
                "diagram_needed": {"type": "BOOLEAN"}
            },
            "required": ["executive_summary", "key_topics", "decisions_log", "action_items", "diagram_needed"]
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
        # BUILD MARKDOWN FILES
        # ==========================================
        executive_summary = res_data.get('executive_summary') or "לא זוהה מידע בולט בהקלטה."
        key_topics = res_data.get('key_topics') or []
        decisions = res_data.get('decisions_log') or []
        action_items = res_data.get('action_items') or []

        md_summary = f"<div dir='rtl'>\n# סיכום פגישה\n\n"
        md_summary += f"## תקציר מנהלים\n{executive_summary}\n\n"
        
        md_summary += "## נושאים מרכזיים\n"
        if not key_topics: md_summary += "* לא זוהו נושאים מרכזיים\n"
        else:
            for t in key_topics: md_summary += f"* {t}\n"
        
        md_summary += "\n## החלטות שהתקבלו\n"
        if not decisions: md_summary += "* לא התקבלו החלטות\n"
        else:
            for d in decisions: md_summary += f"* {d}\n"
        
        md_summary += "\n## משימות לביצוע\n| משימה | אחראי | יעד |\n|---|---|---|\n"
        if not action_items:
            md_summary += "| לא זוהו משימות | - | - |\n"
        else:
            for item in action_items:
                if isinstance(item, dict):
                    md_summary += f"| {item.get('task','-')} | {item.get('owner','-')} | {item.get('deadline','-')} |\n"
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

    except Exception as e:
        print(f"[CRITICAL ERROR] Pipeline failed: {e}")
        raise e

    finally:
        print("[Step 8] Cleaning up...")
        try:
            if blob.exists(): blob.delete()
        except Exception: pass
        try:
            if gemini_file and client: client.files.delete(name=gemini_file.name)
        except Exception: pass
        try:
            if os.path.exists(local_audio_path): os.remove(local_audio_path)
        except Exception: pass
        print("--- PIPELINE FINISHED ---")

    return "Success", 200