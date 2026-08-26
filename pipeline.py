import os
import io
import time
import json
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# --- CONFIGURATION ---
GEMINI_PRO_MODEL = 'gemini-3.1-pro-preview' 
GEMINI_FLASH_MODEL = 'gemini-3.1-flash-lite'
RAW_AUDIO_FOLDER = "Meetings_Raw_Audio"
SUMMARIES_FOLDER = "Meetings_Summaries"
SCOPES = ['https://www.googleapis.com/auth/drive']
POLLING_INTERVAL = 30 # Seconds between checks

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")
client = genai.Client(api_key=GEMINI_API_KEY)

class ActionItem(BaseModel):
    task: str
    owner: str
    deadline: str

class MeetingSummary(BaseModel):
    executive_summary: str
    key_topics: list[str]
    decisions_log: list[str]
    action_items: list[ActionItem]
    diagram_needed: bool

class TamlelanService:
    def __init__(self):
        self.drive_service = self._authenticate_drive()
        self.raw_folder_id = self._get_folder_id(RAW_AUDIO_FOLDER)
        self.summaries_folder_id = self._get_folder_id(SUMMARIES_FOLDER)

    def _authenticate_drive(self):
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            return build('drive', 'v3', credentials=creds)
        raise FileNotFoundError("token.json not found. Run scribe.py first.")

    def _get_folder_id(self, folder_name):
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        items = results.get('files', [])
        if not items: raise Exception(f"Folder '{folder_name}' not found.")
        return items[0]['id']

    def get_pending_files(self):
        query = f"'{self.raw_folder_id}' in parents and mimeType='audio/wav' and trashed=false"
        results = self.drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        return results.get('files', [])

    def process_file(self, drive_file):
        local_path = drive_file['name']
        base_name = os.path.splitext(local_path)[0]
        try:
            # Download
            request = self.drive_service.files().get_media(fileId=drive_file['id'])
            with io.FileIO(local_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done: _, done = downloader.next_chunk()

            # Gemini Analysis
            print(f"[Service] Processing {local_path}...")
            gemini_audio = client.files.upload(file=local_path)
            time.sleep(2)
            
            response = client.models.generate_content(
                model=GEMINI_PRO_MODEL,
                contents=["Analyze this meeting in Hebrew. Extract Summary, Topics, Decisions, and Action Items. Set diagram_needed to true if technical workflows were discussed.", gemini_audio],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=MeetingSummary, temperature=0.2)
            )
            
            data = json.loads(response.text)
            
            # Build Markdown
            md = f"<div dir='rtl'>\n## תקציר מנהלים\n{data['executive_summary']}\n\n## נושאים\n"
            md += "\n".join([f"* {t}" for t in data['key_topics']])
            md += "\n\n## החלטות\n" + "\n".join([f"* {d}" for d in data['decisions_log']])
            md += "\n\n## משימות\n| משימה | אחראי | יעד |\n|---|---|---|\n"
            md += "\n".join([f"| {i['task']} | {i['owner']} | {i['deadline']} |" for i in data['action_items']]) + "\n</div>"

            md_file = f"{base_name}.md"
            with open(md_file, "w", encoding="utf-8") as f: f.write(md)

            # Upload & Cleanup
            for f_name, mtype in [(md_file, 'text/markdown')]:
                meta = {'name': f_name, 'parents': [self.summaries_folder_id]}
                self.drive_service.files().create(body=meta, media_body=MediaFileUpload(f_name, mimetype=mtype)).execute()

            self.drive_service.files().delete(fileId=drive_file['id']).execute()
            client.files.delete(name=gemini_audio.name)
            os.remove(local_path)
            os.remove(md_file)
            print(f"[Service] Successfully processed {local_path}")
        except Exception as e:
            print(f"[Error] {e}")

    def start(self):
        print(f"Tamlelan Service Active. Polling every {POLLING_INTERVAL}s...")
        while True:
            try:
                files = self.get_pending_files()
                for f in files:
                    self.process_file(f)
            except Exception as e:
                print(f"Polling error: {e}")
            time.sleep(POLLING_INTERVAL)

if __name__ == "__main__":
    TamlelanService().start()