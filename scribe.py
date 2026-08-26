import os
import sys
import time
import math
import logging
import uuid
import threading
import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from google.cloud import storage
from google.oauth2 import service_account
import pyaudiowpatch as pyaudio

# ==========================================
# CONFIGURATION & SPECS
# ==========================================
TARGET_SAMPLE_RATE = 16000
BUCKET_NAME = "tamlelan-inbox-stgliding"
DEAD_MIC_THRESHOLD = 50

# Global state
is_recording = False
root_window = None
current_mic_rms = 0
current_sys_rms = 0
mic_frames = []
sys_frames = []

# ==========================================
# PATH RESOLUTION & LOGGING
# ==========================================
def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

def get_executable_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.dirname(__file__))

log_file_path = os.path.join(get_executable_dir(), 'tamlelan_client.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==========================================
# CORE LOGIC
# ==========================================
def clean_old_backups():
    """Deletes backups older than 7 days to save hard drive space."""
    backup_dir = os.path.join(get_executable_dir(), "Tamlelan_Backups")
    if not os.path.exists(backup_dir): 
        return
    
    now = time.time()
    for filename in os.listdir(backup_dir):
        file_path = os.path.join(backup_dir, filename)
        if os.path.isfile(file_path):
            # If file is older than 7 days (7 * 24 * 60 * 60 seconds)
            if os.stat(file_path).st_mtime < now - (7 * 86400):
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted old backup: {filename}")
                except Exception as e:
                    logging.error(f"Failed to delete old backup {filename}: {e}")

def upload_to_gcp(file_path):
    logging.info("Authenticating with GCP...")
    cred_path = os.path.join(get_executable_dir(), 'service_account.json')
    
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"Credentials not found at {cred_path}")

    credentials = service_account.Credentials.from_service_account_file(cred_path)
    client = storage.Client(credentials=credentials, project=credentials.project_id)
    
    bucket = client.bucket(BUCKET_NAME)
    blob_name = f"tamlelan_audio_{uuid.uuid4().hex}.wav"
    blob = bucket.blob(blob_name)
    
    # ARCHITECTURAL FIX: 5MB Chunks and Tuple Timeout for Sleep-Mode Resilience
    blob.chunk_size = 5 * 1024 * 1024 
    logging.info(f"Uploading to gs://{BUCKET_NAME}/{blob_name}...")
    blob.upload_from_filename(file_path, timeout=(10, 120))
    logging.info("Upload successful.")

def show_mic_warning():
    logging.warning("1-Minute Health Check Failed: No audio detected.")
    messagebox.showwarning(
        "Microphone Warning", 
        "1 minute has passed and no sound was detected.\n\nPlease check if your microphone is muted. Recording is still running."
    )

def update_meter_ui(vol_canvas, vol_bar):
    if is_recording:
        max_rms = max(current_mic_rms, current_sys_rms)
        bar_width = min(230, int((max_rms / 3000) * 230))
        vol_canvas.coords(vol_bar, 0, 0, bar_width, 20)
        root_window.after(50, update_meter_ui, vol_canvas, vol_bar)
    else:
        vol_canvas.coords(vol_bar, 0, 0, 0, 20)

def recording_thread_task(status_label, start_btn, end_btn):
    global is_recording, mic_frames, sys_frames, current_mic_rms, current_sys_rms
    
    mic_frames = []
    sys_frames = []
    current_mic_rms = 0
    current_sys_rms = 0
    
    p = pyaudio.PyAudio()
    
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_mic = p.get_device_info_by_index(wasapi_info["defaultInputDevice"])
        
        try:
            default_loopback = p.get_default_wasapi_loopback()
        except OSError:
            default_loopback = None

        mic_rate = int(default_mic["defaultSampleRate"])
        mic_channels = default_mic["maxInputChannels"]
        
        sys_rate = int(default_loopback["defaultSampleRate"]) if default_loopback else TARGET_SAMPLE_RATE
        sys_channels = default_loopback["maxInputChannels"] if default_loopback else 1

        mic_stream = p.open(format=pyaudio.paInt16, channels=mic_channels, rate=mic_rate,
                            input=True, input_device_index=default_mic["index"])
        mic_stream.start_stream()
        
        sys_stream = None
        if default_loopback:
            sys_stream = p.open(format=pyaudio.paInt16, channels=sys_channels, rate=sys_rate,
                                input=True, input_device_index=default_loopback["index"])
            sys_stream.start_stream()

        def mic_worker():
            global current_mic_rms
            empty_reads = 0
            while is_recording:
                try:
                    avail = mic_stream.get_read_available()
                    if avail > 0:
                        empty_reads = 0
                        chunk = min(avail, 4096)
                        data = mic_stream.read(chunk, exception_on_overflow=False)
                        mic_frames.append(data)
                        
                        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                        if len(audio_data) > 0:
                            rms = np.sqrt(np.mean(np.square(audio_data)))
                            current_mic_rms = int(rms) if not np.isnan(rms) else 0
                    else:
                        empty_reads += 1
                        if empty_reads > 10:
                            current_mic_rms = 0
                        time.sleep(0.01)
                except Exception as e:
                    logging.error(f"Mic worker error: {e}")
                    break

        def sys_worker():
            global current_sys_rms
            if not sys_stream: return
            empty_reads = 0
            while is_recording:
                try:
                    avail = sys_stream.get_read_available()
                    if avail > 0:
                        empty_reads = 0
                        chunk = min(avail, 4096)
                        data = sys_stream.read(chunk, exception_on_overflow=False)
                        sys_frames.append(data)
                        
                        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                        if len(audio_data) > 0:
                            rms = np.sqrt(np.mean(np.square(audio_data)))
                            current_sys_rms = int(rms) if not np.isnan(rms) else 0
                    else:
                        empty_reads += 1
                        if empty_reads > 10:
                            current_sys_rms = 0
                        time.sleep(0.01)
                except Exception as e:
                    logging.error(f"Sys worker error: {e}")
                    break

        t_mic = threading.Thread(target=mic_worker, daemon=True)
        t_sys = threading.Thread(target=sys_worker, daemon=True)
        
        t_mic.start()
        t_sys.start()

        elapsed_time = 0
        health_check_done = False
        max_rms_first_minute = 0

        while is_recording:
            time.sleep(0.1)
            elapsed_time += 0.1
            if not health_check_done:
                combined_rms = max(current_mic_rms, current_sys_rms)
                if combined_rms > max_rms_first_minute:
                    max_rms_first_minute = combined_rms
                if elapsed_time >= 60.0:
                    health_check_done = True
                    if max_rms_first_minute < DEAD_MIC_THRESHOLD:
                        root_window.after(0, show_mic_warning)

        t_mic.join(timeout=1.0)
        t_sys.join(timeout=1.0)

        mic_stream.stop_stream()
        mic_stream.close()
        if sys_stream:
            sys_stream.stop_stream()
            sys_stream.close()
        p.terminate()

        root_window.after(0, lambda: status_label.config(text="Status: Processing Dual Audio..."))
        
        def process_audio(frames, channels, original_rate):
            if not frames:
                return np.array([], dtype=np.int16)
            raw_data = b''.join(frames)
            arr = np.frombuffer(raw_data, dtype=np.int16)
            
            if channels > 1:
                arr = arr.reshape(-1, channels).mean(axis=1)
                
            if original_rate != TARGET_SAMPLE_RATE:
                gcd = math.gcd(TARGET_SAMPLE_RATE, original_rate)
                up = TARGET_SAMPLE_RATE // gcd
                down = original_rate // gcd
                arr = resample_poly(arr, up, down)
            return arr

        mic_array = process_audio(mic_frames, mic_channels, mic_rate)
        sys_array = process_audio(sys_frames, sys_channels, sys_rate)

        max_len = max(len(mic_array), len(sys_array))
        if max_len == 0:
            root_window.after(0, lambda: status_label.config(text="Status: Recording failed (No data)."))
            return

        mic_array = np.pad(mic_array, (0, max_len - len(mic_array)))
        sys_array = np.pad(sys_array, (0, max_len - len(sys_array)))

        # ARCHITECTURAL FIX: Rounding before casting prevents audio artifacts
        mixed = np.round(mic_array).astype(np.int32) + np.round(sys_array).astype(np.int32)
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

        # The Local Backup Vault
        backup_dir = os.path.join(get_executable_dir(), "Tamlelan_Backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        safe_filename = f"Meeting_Audio_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        safe_file_path = os.path.join(backup_dir, safe_filename)
        
        logging.info(f"Writing audio to safe backup vault: {safe_file_path}")
        wavfile.write(safe_file_path, TARGET_SAMPLE_RATE, mixed)
        
        # Attempt Upload
        upload_to_gcp(safe_file_path)
        
        # ARCHITECTURAL FIX: Do NOT delete the file! Keep it for 7 days.
        logging.info("Upload complete. Audio retained in Tamlelan_Backups for 7 days.")
        root_window.after(0, lambda: status_label.config(text="Status: Upload Complete! Ready."))

    except Exception as e:
        logging.exception("Error during recording/upload:")
        root_window.after(0, lambda: status_label.config(text="Status: Error occurred. Check logs."))
        error_msg = f"An error occurred:\n{str(e)}\n\nDon't worry, your audio was saved locally in the 'Tamlelan_Backups' folder."
        root_window.after(0, lambda: messagebox.showerror("Upload Error", error_msg))
    finally:
        is_recording = False
        root_window.after(0, lambda: start_btn.config(state=tk.NORMAL))
        root_window.after(0, lambda: end_btn.config(state=tk.DISABLED))

def start_recording(status_label, start_btn, end_btn, vol_canvas, vol_bar):
    global is_recording
    is_recording = True
    status_label.config(text="Status: Recording... (Mic + System)")
    start_btn.config(state=tk.DISABLED)
    end_btn.config(state=tk.NORMAL)
    
    update_meter_ui(vol_canvas, vol_bar)
    
    threading.Thread(
        target=recording_thread_task, 
        args=(status_label, start_btn, end_btn), 
        daemon=True
    ).start()

def stop_recording(status_label):
    global is_recording
    if is_recording:
        is_recording = False
        status_label.config(text="Status: Stopping & Mixing... Please wait.")

def create_gui():
    global root_window
    root_window = tk.Tk()
    root_window.title("TAMLELAN Client V1.1")
    root_window.geometry("350x250")
    root_window.resizable(False, False)

    title_label = tk.Label(root_window, text="TAMLELAN Meeting Agent", font=("Helvetica", 14, "bold"))
    title_label.pack(pady=10)

    status_label = tk.Label(root_window, text="Status: Ready", font=("Helvetica", 10))
    status_label.pack(pady=5)

    vol_frame = tk.Frame(root_window)
    vol_frame.pack(pady=5)
    
    vol_label = tk.Label(vol_frame, text="Mic/Sys Level:", font=("Helvetica", 9))
    vol_label.pack(side=tk.LEFT, padx=5)
    
    vol_canvas = tk.Canvas(vol_frame, width=230, height=20, bg='gray', highlightthickness=1, highlightbackground="black")
    vol_canvas.pack(side=tk.LEFT)
    vol_bar = vol_canvas.create_rectangle(0, 0, 0, 20, fill='limegreen')

    btn_frame = tk.Frame(root_window)
    btn_frame.pack(pady=15)

    start_btn = tk.Button(btn_frame, text="START", font=("Helvetica", 12), bg="green", fg="white", width=10)
    end_btn = tk.Button(btn_frame, text="END", font=("Helvetica", 12), bg="red", fg="white", width=10, state=tk.DISABLED)

    start_btn.config(command=lambda: start_recording(status_label, start_btn, end_btn, vol_canvas, vol_bar))
    end_btn.config(command=lambda: stop_recording(status_label))

    start_btn.grid(row=0, column=0, padx=10)
    end_btn.grid(row=0, column=1, padx=10)

    root_window.mainloop()

if __name__ == "__main__":
    logging.info("=== TAMLELAN GUI Client V1.1 Started ===")
    clean_old_backups()
    create_gui()