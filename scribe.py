import os
import sys
import time
import math
import json
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

# Local speaker diarization (sherpa-onnx, CPU-only, no GCP cost)
DIARIZATION_ENABLED = True
DIAR_CLUSTER_THRESHOLD = 0.5
DIAR_MIN_DURATION_ON = 0.3
DIAR_MIN_DURATION_OFF = 0.5
DIAR_MERGE_GAP_SEC = 0.8
DIARIZATION_SCHEMA_VERSION = 1
SEGMENTATION_MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEGMENTATION_MODEL_SIZE = 6958444
EMBEDDING_MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
EMBEDDING_MODEL_SIZE = 28281164

# Global state
is_recording = False
root_window = None
current_mic_rms = 0
current_sys_rms = 0
mic_frames = []
sys_frames = []

# ==========================================
# AUDIO PROCESSING (module-level: unit-testable without recording hardware)
# ==========================================
def process_audio(frames, channels, original_rate):
    """Joins raw int16 frame bytes into one channel, downmixing multi-channel
    source devices to mono and resampling to TARGET_SAMPLE_RATE."""
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


def build_output_audio(mic_array, sys_array, has_loopback):
    """Rounds/clips/casts each channel to int16, then combines them into a
    stereo track (left=mic, right=system) when a loopback device was present,
    or mono (mic only) otherwise -- never a fake, silent right channel."""
    mic_int16 = np.clip(np.round(mic_array), -32768, 32767).astype(np.int16)
    sys_int16 = np.clip(np.round(sys_array), -32768, 32767).astype(np.int16)
    if has_loopback:
        return np.column_stack((mic_int16, sys_int16))
    return mic_int16


def _download_file(url, dest_path, expected_size=None):
    """Plain stdlib download (urllib) for a single file."""
    import urllib.request
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    if expected_size and os.path.getsize(dest_path) != expected_size:
        logging.warning(
            f"Downloaded file {dest_path} size {os.path.getsize(dest_path)} "
            f"does not match expected {expected_size} -- may be corrupt or the "
            f"upstream release changed.")


def _download_and_extract_tar(url, extract_dir):
    """Downloads a .tar.bz2 and extracts it into extract_dir. The archive
    already contains its own top-level folder (e.g. sherpa-onnx-pyannote-
    segmentation-3-0/model.onnx) -- extract_dir must be the PARENT of that
    folder, not the folder itself, or paths double up."""
    import urllib.request
    import tarfile
    os.makedirs(extract_dir, exist_ok=True)
    tmp_path = os.path.join(extract_dir, "_download.tar.bz2")
    urllib.request.urlretrieve(url, tmp_path)
    with tarfile.open(tmp_path, "r:bz2") as tar:
        tar.extractall(extract_dir)
    os.remove(tmp_path)


def get_model_paths():
    """
    Resolves the local paths to the two ONNX models diarization needs, next to
    the exe (not _MEIPASS -- must survive rebuilds, matching service_account.json's
    placement). Triggers a one-time download on first run if absent; a user can
    also drop the model files into this folder manually for an offline install.
    """
    models_dir = os.path.join(get_executable_dir(), "models")
    segmentation_path = os.path.join(
        models_dir, "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
    embedding_path = os.path.join(
        models_dir, "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")

    if not os.path.exists(segmentation_path):
        logging.info("Diarization segmentation model not found locally -- downloading (one-time, ~7MB)...")
        _download_and_extract_tar(SEGMENTATION_MODEL_URL, models_dir)
    if not os.path.exists(embedding_path):
        logging.info("Diarization embedding model not found locally -- downloading (one-time, ~28MB)...")
        _download_file(EMBEDDING_MODEL_URL, embedding_path, EMBEDDING_MODEL_SIZE)

    return segmentation_path, embedding_path


def _make_diarizer(num_clusters=-1):
    """Lazy import of sherpa_onnx -- keeps scribe.exe startup fast, and lets this
    module still be imported for testing on a machine without sherpa-onnx
    installed (only this function and diarize_channel touch the import)."""
    import sherpa_onnx

    segmentation_path, embedding_path = get_model_paths()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation_path,
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding_path),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters, threshold=DIAR_CLUSTER_THRESHOLD),
        min_duration_on=DIAR_MIN_DURATION_ON,
        min_duration_off=DIAR_MIN_DURATION_OFF,
    )
    config.validate()
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def diarize_channel(samples, label_prefix, num_clusters=-1):
    """Runs local diarization on one channel's samples (int16 or float64 --
    process_audio returns int16 when no resampling ran, float64 when
    resample_poly did). Returns a list of (start, end, label) tuples."""
    arr = np.asarray(samples)
    if arr.dtype != np.float32:
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / 32768.0
        else:
            arr = (arr / 32768.0).astype(np.float32)

    diarizer = _make_diarizer(num_clusters=num_clusters)
    result = diarizer.process(arr).sort_by_start_time()
    return [(r.start, r.end, f"{label_prefix}{r.speaker:02d}") for r in result]


def merge_speaker_segments(segments, gap=DIAR_MERGE_GAP_SEC):
    """Pure, no sherpa dependency. Merges temporally adjacent same-label
    segments separated by less than `gap` seconds, bounding payload/prompt size."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s[0])
    merged = [list(ordered[0])]
    for start, end, label in ordered[1:]:
        last = merged[-1]
        if label == last[2] and start - last[1] < gap:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end, label])
    return [tuple(m) for m in merged]


def build_diarization_payload(mic_array, sys_array, has_loopback, duration_sec, expected_participants=None):
    """
    Channel-selection logic:
    - Stereo (has_loopback=True, e.g. a 3+-person Zoom call): diarize the RIGHT
      (system/remote) channel to separate multiple remote voices -- Zoom already
      mixed them down before this ever saw the audio, and nothing else
      distinguishes them. Diarize the LEFT (mic) channel with num_clusters=1 for
      a free operator speech-activity timeline via the same code path.
    - Mono (has_loopback=False, in-room case): diarize the full mono track --
      no privileged "operator" identity, since there's no channel signal at all.

    expected_participants (optional, total headcount including the operator):
    real-world testing showed automatic clustering (num_clusters=-1) badly
    over-segments real Zoom audio -- a 2-remote-person meeting produced 16+
    distinct spurious labels, mostly from short utterances/background noise.
    Supplying the true count (sherpa-onnx's own docs recommend this) fixes the
    cluster count directly instead of leaving it to a similarity threshold that
    real, noisy audio doesn't cleanly meet. When not supplied, falls back to
    automatic detection exactly as before.

    Returns None on ANY failure. Diarization must never cost the user a recording.
    """
    if not DIARIZATION_ENABLED:
        return None
    try:
        if has_loopback:
            remote_num_clusters = -1
            if expected_participants and expected_participants >= 2:
                remote_num_clusters = expected_participants - 1

            # num_clusters=1 guarantees exactly one distinct label on this channel
            # by construction -- relabel uniformly to plain "OPERATOR" rather than
            # trusting diarize_channel's f"{prefix}{speaker:02d}" formatting (which
            # would produce "OPERATOR00"), since the channel/label equality check
            # below depends on the exact string "OPERATOR".
            operator_segments = [(s, e, "OPERATOR") for s, e, _ in
                                  diarize_channel(mic_array, "OPERATOR", num_clusters=1)]
            remote_segments = diarize_channel(sys_array, "REMOTE_", num_clusters=remote_num_clusters)
            all_segments = merge_speaker_segments(operator_segments + remote_segments)
            labels = {seg[2] for seg in all_segments}
            speaker_count = len(labels)
            speakers = [
                {"label": lbl, "channel": "left" if lbl == "OPERATOR" else "right"}
                for lbl in sorted(labels)
            ]
            channel_mode = "stereo_operator_left"
        else:
            mono_num_clusters = expected_participants if (expected_participants and expected_participants >= 1) else -1
            mono_segments = merge_speaker_segments(diarize_channel(mic_array, "SPEAKER_", num_clusters=mono_num_clusters))
            all_segments = mono_segments
            labels = {seg[2] for seg in all_segments}
            speaker_count = len(labels)
            speakers = [{"label": lbl, "channel": "mono"} for lbl in sorted(labels)]
            channel_mode = "mono_single_track"

        return {
            "schema_version": DIARIZATION_SCHEMA_VERSION,
            "channel_mode": channel_mode,
            "sample_rate": TARGET_SAMPLE_RATE,
            "duration_sec": duration_sec,
            "expected_participants": expected_participants,
            "speaker_count": speaker_count,
            "speakers": speakers,
            "segments": [[round(s, 2), round(e, 2), lbl] for s, e, lbl in sorted(all_segments)],
        }
    except Exception:
        logging.exception("Local diarization failed -- continuing without it (recording is unaffected):")
        return None

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
        # Diarization companion backups are small and deliberately kept longer
        # than the 7-day audio retention window -- not swept here.
        if filename.endswith(".diarization.json"):
            continue
        file_path = os.path.join(backup_dir, filename)
        if os.path.isfile(file_path):
            # If file is older than 7 days (7 * 24 * 60 * 60 seconds)
            if os.stat(file_path).st_mtime < now - (7 * 86400):
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted old backup: {filename}")
                except Exception as e:
                    logging.error(f"Failed to delete old backup {filename}: {e}")

def upload_to_gcp(file_path, diar_payload=None):
    logging.info("Authenticating with GCP...")
    cred_path = os.path.join(get_executable_dir(), 'service_account.json')

    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"Credentials not found at {cred_path}")

    credentials = service_account.Credentials.from_service_account_file(cred_path)
    client = storage.Client(credentials=credentials, project=credentials.project_id)

    bucket = client.bucket(BUCKET_NAME)
    stem = f"tamlelan_audio_{uuid.uuid4().hex}"  # one shared stem for both objects
    companion_blob = None

    # ORDER IS LOAD-BEARING: the companion must be fully uploaded BEFORE the
    # .wav, because the .wav's own finalize event is what triggers the cloud
    # function -- GCS's read-after-write consistency then guarantees the
    # companion is already present by the time the handler runs.
    if diar_payload:
        companion_blob = bucket.blob(f"{stem}.diarization.json")
        logging.info(f"Uploading diarization companion to gs://{BUCKET_NAME}/{stem}.diarization.json...")
        companion_blob.upload_from_string(
            json.dumps(diar_payload, ensure_ascii=False),
            content_type="application/json")

    blob_name = f"{stem}.wav"
    blob = bucket.blob(blob_name)

    # ARCHITECTURAL FIX: 5MB Chunks and Tuple Timeout for Sleep-Mode Resilience
    blob.chunk_size = 5 * 1024 * 1024
    logging.info(f"Uploading to gs://{BUCKET_NAME}/{blob_name}...")
    try:
        blob.upload_from_filename(file_path, timeout=(10, 120))
    except Exception:
        if companion_blob is not None:
            try:
                companion_blob.delete()  # best-effort orphan cleanup
            except Exception:
                pass
        raise
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

def recording_thread_task(status_label, start_btn, end_btn, expected_participants=None):
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

        mic_array = process_audio(mic_frames, mic_channels, mic_rate)
        sys_array = process_audio(sys_frames, sys_channels, sys_rate)

        max_len = max(len(mic_array), len(sys_array))
        if max_len == 0:
            root_window.after(0, lambda: status_label.config(text="Status: Recording failed (No data)."))
            return

        mic_array = np.pad(mic_array, (0, max_len - len(mic_array)))
        sys_array = np.pad(sys_array, (0, max_len - len(sys_array)))

        # ARCHITECTURAL FIX (Phase 4): keep channels separate instead of summing them
        # into one mono track. Left = operator's microphone, right = remote/system
        # audio. This lets the analysis prompt tell the two apart, which a mono mix
        # destroyed entirely. If no loopback device exists (nothing to distinguish
        # from), write mono rather than a fake, silent right channel.
        output_audio = build_output_audio(mic_array, sys_array, bool(default_loopback))

        # The Local Backup Vault
        backup_dir = os.path.join(get_executable_dir(), "Tamlelan_Backups")
        os.makedirs(backup_dir, exist_ok=True)

        safe_filename = f"Meeting_Audio_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        safe_file_path = os.path.join(backup_dir, safe_filename)

        logging.info(f"Writing audio to safe backup vault: {safe_file_path}")
        wavfile.write(safe_file_path, TARGET_SAMPLE_RATE, output_audio)

        # Local speaker diarization (Phase 9) -- runs on the already-in-memory,
        # already-padded channels, so segment timestamps align sample-for-sample
        # with the .wav just written. CPU-only, no GCP cost; ~30s for a 45-minute
        # meeting is the expected order of magnitude.
        diar_payload = None
        if DIARIZATION_ENABLED:
            root_window.after(0, lambda: status_label.config(text="Status: Analyzing speakers..."))
            diar_payload = build_diarization_payload(
                mic_array, sys_array, bool(default_loopback),
                duration_sec=len(mic_array) / TARGET_SAMPLE_RATE,
                expected_participants=expected_participants)
            if diar_payload:
                # Permanent local copy, alongside the audio backup -- unlike the
                # audio itself, NOT subject to the 7-day clean_old_backups() sweep;
                # these files are tiny and the user wants them kept longer.
                diar_backup_path = safe_file_path.replace(".wav", ".diarization.json")
                try:
                    with open(diar_backup_path, "w", encoding="utf-8") as f:
                        json.dump(diar_payload, f, ensure_ascii=False)
                except Exception:
                    logging.exception("Could not write local diarization backup copy (non-fatal):")

        # Attempt Upload
        root_window.after(0, lambda: status_label.config(text="Status: Uploading..."))
        upload_to_gcp(safe_file_path, diar_payload)
        
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

def start_recording(status_label, start_btn, end_btn, vol_canvas, vol_bar, participants_var=None):
    global is_recording
    is_recording = True
    status_label.config(text="Status: Recording... (Mic + System)")
    start_btn.config(state=tk.DISABLED)
    end_btn.config(state=tk.NORMAL)

    expected_participants = None
    if participants_var is not None:
        try:
            value = int(participants_var.get())
            if value >= 1:
                expected_participants = value
        except (ValueError, tk.TclError):
            pass  # left blank or invalid -- falls back to automatic detection

    update_meter_ui(vol_canvas, vol_bar)

    threading.Thread(
        target=recording_thread_task,
        args=(status_label, start_btn, end_btn, expected_participants),
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
    root_window.geometry("350x290")
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

    participants_frame = tk.Frame(root_window)
    participants_frame.pack(pady=5)

    participants_label = tk.Label(participants_frame, text="Participants (incl. you):", font=("Helvetica", 9))
    participants_label.pack(side=tk.LEFT, padx=5)

    participants_var = tk.StringVar(value="2")
    participants_spinbox = tk.Spinbox(participants_frame, from_=1, to=20, width=4, textvariable=participants_var)
    participants_spinbox.pack(side=tk.LEFT)

    btn_frame = tk.Frame(root_window)
    btn_frame.pack(pady=15)

    start_btn = tk.Button(btn_frame, text="START", font=("Helvetica", 12), bg="green", fg="white", width=10)
    end_btn = tk.Button(btn_frame, text="END", font=("Helvetica", 12), bg="red", fg="white", width=10, state=tk.DISABLED)

    start_btn.config(command=lambda: start_recording(status_label, start_btn, end_btn, vol_canvas, vol_bar, participants_var))
    end_btn.config(command=lambda: stop_recording(status_label))

    start_btn.grid(row=0, column=0, padx=10)
    end_btn.grid(row=0, column=1, padx=10)

    root_window.mainloop()

if __name__ == "__main__":
    logging.info("=== TAMLELAN GUI Client V1.1 Started ===")
    clean_old_backups()
    create_gui()