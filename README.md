# AI Executive Summary Builder

**Document Type:** Architecture & Current State
**As of:** commit `48f4a0d`, 2026-08-27
**System Objective:** An autonomous, serverless, zero-subscription AI meeting agent. It records
local audio, uploads it to a cloud bucket, triggers an LLM pipeline for structured data extraction
and transcription, and writes a Hebrew summary and transcript to a personal Google Drive.

This document describes the system **as it actually runs today**. For the reasoning behind
individual changes, see `docs/IMPLEMENTATION_LOG.md` (Phases 0&ndash;5) and the git log (`git log
--oneline`, one commit per phase, full justification in each commit body from Phase 6 onward — the
implementation log was not kept current past Phase 5).

---

## 1. Current Architecture & Deployed Components

### 1.1 Local Client Layer (Windows)

- **Component:** `scribe.py`, compiled to `scribe.exe` via PyInstaller (`build.bat`).
- **Frameworks:** Tkinter (GUI), `sounddevice`/`PyAudioWPatch` (capture), `scipy.io.wavfile`
  (encoding), `google-cloud-storage` (upload), `sherpa-onnx` (local diarization, lazy-imported).
- **Capture:** 16 kHz, 16-bit PCM. The meeting operator's microphone and remote/system audio are
  captured as **separate channels** and written to a stereo `.wav` (left = operator, right =
  remote) rather than mixed to mono, so the two can be analyzed independently downstream. A
  single-channel device (no loopback available) falls back to mono with no channel distinction.
- **Local diarization (see §2):** runs entirely on-device before upload, at zero API cost.
- **Backup retention:** local `.wav` copies are kept for 7 days in `Tamlelan_Backups/` regardless
  of outcome; the diarization companion JSON is kept **permanently** alongside it, explicitly
  excluded from the 7-day sweep.
- **Upload order (load-bearing):** the diarization companion is uploaded to GCS *before* the
  `.wav`. The `.wav`'s own finalize event is what triggers the cloud pipeline, so GCS's
  read-after-write consistency guarantees the companion already exists by the time the handler
  runs — this is not a timing assumption.
- **Auth:** local `service_account.json`. Path resolution handles both `python scribe.py` and the
  compiled `.exe` (`sys.frozen` check), writing tracebacks to `tamlelan_client.log` since the
  compiled build runs with `--noconsole`.

### 1.2 Ingestion Layer (Google Cloud Storage)

- **Bucket:** `tamlelan-inbox-stgliding`, region `us-west1`.
- **Security:** uniform bucket-level access enabled, public access prevention enforced (verified
  live).
- **Eventing:** Eventarc trigger on `google.cloud.storage.object.v1.finalized`, no path filter of
  its own — all filtering (ignoring `.diarization.json`, `locks/`, `failed/`) happens in
  application code.
- **`locks/` prefix:** one lock object per Eventarc event ID, written before processing starts, so
  Eventarc's at-least-once delivery can't cause duplicate processing — a duplicate delivery is
  detected and aborted before any Gemini call.
- **`failed/` prefix:** dead-letter location for recordings that failed processing. Guarded against
  self-recursion — a `failed/` object cannot re-trigger the pipeline and nest into
  `failed/failed/...`, which was a real, confirmed bug before the guard was added.

### 1.3 Compute & Intelligence Layer (Cloud Run)

- **Component:** `tamlelan-processor`, 2nd-gen Cloud Run function (source: `cloud/main.py`),
  triggered by the Eventarc trigger above.
- **Runtime (live, verified):** Python 3.12, 1 vCPU, 1 GiB memory, 540s timeout, 0 minimum
  instances (scale-to-zero), max 20 instances.
- **Libraries:** `functions-framework`, `google-genai`, `google-cloud-storage`, `rapidfuzz`.
- **Analysis — two Gemini passes per meeting, both on `gemini-3.1-pro-preview`:**
  - **Pass 1 (structured JSON):** executive summary, attendees (voice-audibility required, not
    merely "mentioned"), a separate `people_mentioned` list for named-but-not-heard people,
    key topics, a decisions log with a `decided` / `proposed` / `open` status enum and hedge-note
    capture, action items (owner/deadline only if explicitly stated, never inferred), and a
    `diagram_needed` flag.
  - **Pass 2 (verbatim transcript):** every line labeled `M:SS [SPEAKER]: text`, real names applied
    retroactively to lines spoken before that name was actually said, stable generic labels
    (`דובר 1`, `דובר 2`, ...) when no real name is ever established, never renumbered mid-meeting.
- **Diarization companion (see §2):** downloaded and validated (`load_diarization`); on any
  failure — missing, malformed, wrong schema — returns `None` and both prompts revert to their
  exact pre-diarization form. Its content is injected into both prompts as structural ground truth
  (speaker roster + turn boundaries) for Gemini to *resolve*, never as asserted identity.
- **Attendee cross-check:** flags the attendees list (no extra API call) when it names more people
  than diarization detected distinct voices for (+1 tolerance) — a deterministic second line of
  defense against attendee over-inclusion.
- **Grounding verification:** every `source_quote` is fuzzy-matched (`rapidfuzz`, threshold 80)
  against the Pass 2 transcript; an unmatched quote is flagged (⚠), never silently dropped.
- **Topic-referential-integrity check:** flags any decision/action item whose `related_topic_id`
  doesn't correspond to a real `key_topics` entry.
- **Context caching (see §3):** the audio is cached once via `client.caches.create()` and both
  passes reference it, instead of each paying full price for the same audio independently.
- **Secondary model:** `gemini-3.1-flash-lite`, used only when `diagram_needed` is true, to
  generate a Mermaid.js HTML architecture diagram from the already-extracted summary.
- **Cleanup:** strict `try`/`finally` — on success, the source `.wav`, diarization companion,
  uploaded Gemini file, and context cache are all deleted; on failure, the `.wav` and companion are
  preserved under `failed/` first (short-circuited if the source was already under `failed/`, per
  §1.2).

### 1.4 Storage Bridge Layer (Google Apps Script)

- **Architectural context:** workaround for GCP service accounts having a 0-byte default Drive
  quota — cannot write binary files to a personal `@gmail.com` Drive folder directly.
- **Component:** Apps Script `doPost(e)` webhook, deployed as a Web App, running as the personal
  account to inherit its Drive quota.
- **Security:** validates an `X-Tamlelan-Secret` header against the `WEBHOOK_SECRET` environment
  variable before writing anything.
- **Scope:** entirely outside this repository and this document — its source lives in Apps Script,
  and by standing project decision it is treated as an external, already-working black box, not
  investigated or modified as part of this work.

---

## 2. Multi-Speaker Diarization

Real-world testing surfaced two accuracy problems the original stereo split couldn't solve on its
own: (a) a stereo split only ever separates "operator" from "everyone else" — it can't tell two
remote participants apart from each other — and (b) a same-room, single-microphone meeting has no
second channel to split at all.

**What runs, client-side, before every upload:**
- **Models:** `pyannote-segmentation-3.0` (~6 MB) + 3D-Speaker CAM++ speaker embeddings (~28 MB),
  run locally via `sherpa-onnx`, CPU-only. Auto-downloaded on first run into `<exe_dir>/models/`,
  with a manual drop-in override for offline installs.
- **Stereo recordings:** the right (remote) channel is diarized to separate the distinct voices
  Zoom already mixed down into one signal; the left (operator) channel is diarized forced to
  exactly one cluster — a free, guaranteed operator timeline via the same code path.
- **Mono recordings** (no loopback device): the full track is diarized directly, with no privileged
  "operator" identity, since there is no channel signal to grant one from.
- **Manual headcount override:** an optional "Participants (incl. you)" field in the recording GUI
  (default 2, range 1&ndash;20) fixes the expected cluster count directly instead of leaving it to
  a similarity threshold. Automatic clustering (`num_clusters=-1`) was found to badly over-segment
  real conversation — a real 2-remote-participant call produced 16+ spurious labels. Fixing the
  count directly is sherpa-onnx's own documented recommendation for this situation.
- **Transport:** a companion GCS object (`<stem>.diarization.json`), not object metadata — GCS's
  8 KiB metadata cap is already tight for one real meeting's segment list and exceeded outright by
  a long one.
- **Zero added cost:** diarization runs entirely on the client. It is never a Gemini call and never
  runs inside Cloud Run.

---

## 3. Cost Optimization: Gemini Context Caching

Both Gemini passes (§1.3) analyze the same meeting audio. Gemini bills audio by duration regardless
of content, so without caching, every meeting's audio was paid for in full twice.

- **Mechanism:** the uploaded audio file is cached once via `client.caches.create()` (10-minute
  TTL) immediately after upload; both passes reference it via `cached_content=` instead of
  resending the audio. The second read is billed at roughly 10% of the direct-input rate.
- **Why explicit caching, not Gemini's free automatic caching:** automatic caching only fires when
  two requests share an identical opening. Both passes here put their (different) instructions
  first and the (identical) audio second, so the two requests never share a beginning — automatic
  caching could never have applied regardless of timing, given how the calls were structured.
- **Safety:** cache creation is wrapped in its own fallback. If it fails for any reason (a clip too
  short to meet the cache's minimum token threshold, a transient API error, a preview-model quirk),
  both passes silently revert to sending the audio directly — the exact pre-caching behavior. This
  cannot introduce a new pipeline failure mode; a dedicated test
  (`tests/test_main_cleanup.py::TestCacheCreationFallback`) asserts it.
- **Verification logging:** each pass logs its Gemini `usage_metadata` (prompt/cached/total token
  counts) specifically so a cache hit is a verifiable number from Cloud Logging, not an assumption.
- **Expected impact:** ≈79% lower audio-token cost per meeting on a worked 40-minute example
  ($0.31 → $0.06), with no prompt, schema, or output change. See the published report
  ["Who Spoke, What It Cost"](https://claude.ai/code/artifact/a889e204-54d3-4650-a69f-a96df0f03961)
  for the full reasoning and the alternatives that were ruled out first.

---

## 4. Known Open Items

Not yet resolved as of this document. Don't assume any of these are closed without checking.

1. **Diarization's Defect 1 gate (cross-topic contamination) hasn't been re-tested** against real
   content since the mitigations (prompt instruction + topic-referential-integrity checking)
   landed. If it's ever observed again, a real pre-segmentation rewrite is the documented, un-built
   fallback.
2. **Same-room, single-microphone diarization has only synthetic test coverage so far** — no real
   in-person recording has been run through the mono diarization path yet.
3. **Context caching's actual cache-hit rate is unconfirmed in production.** The mechanism ran
   cleanly on its first live test; the token-usage logging that proves a genuine cache hit (versus
   silent fallback to full price) shipped immediately after, and the first measurement is still
   pending the next real recording.
4. **Merging both Gemini passes into one call** would remove the duplicate-audio cost entirely
   (more than caching alone), but risks compressing two different jobs into one response and
   degrading either. Deliberately not pursued as part of the caching work.
5. **The Apps Script Drive-upload webhook remains an unexamined black box**, per standing project
   decision, despite one confirmed real HTTP 404 failure from it in production (2026-08-12).
6. **A latent hang risk** in the Gemini file-processing poll loop (no maximum retry count or
   timeout) has been identified but never observed to manifest, and is not fixed.
7. **A decoy GCP project** (`gen-lang-client-0736747503`, literally named "Tamlelan") exists,
   empty, alongside the real infrastructure project (`gen-lang-client-0839027862`, "Default Gemini
   Project") — a standing source of possible future confusion, flagged but not actioned.

---

## 5. Testing

66 tests across 9 files, spanning two isolated environments:

| Environment | Files | Tests |
|---|---|---|
| `.venv-cloud` (isolated venv, `cloud/requirements.txt`) | `test_main_attendees_fix.py`, `test_main_cleanup.py`, `test_main_diarization.py`, `test_main_grounding.py`, `test_main_render.py`, `test_main_topic_linkage.py`, `test_main_transcript_labels.py` | 37 |
| System Python (matches `requirements-client.txt`) | `test_scribe_audio.py`, `test_scribe_diarization.py` | 29 |

`test_scribe_diarization.py` must still import cleanly on a machine without `sherpa-onnx`
installed, since `scribe.py` only imports it lazily inside the two functions that actually need it
(§2) — this keeps `scribe.exe` startup fast and keeps the test file itself dependency-light.

All external services (GCS, Gemini, the Drive webhook) are mocked on the cloud side — no real
network calls or GCP credentials are needed to run the suite. Re-run the full suite before trusting
any future change hasn't regressed something:
```
.venv-cloud/Scripts/python.exe tests/test_main_<name>.py   # per cloud-side file
python tests/test_scribe_<name>.py                          # per client-side file
```

---

## 6. Deployment Procedure

`tamlelan-processor` has one known, previously-costly deploy trap: a missing `--function` flag
produces a container that reports `Ready: True` but crashes on every real request, with nothing in
the deploy output revealing it. The sequence below is mandatory, no exceptions:

```bash
gcloud run deploy tamlelan-processor --source=./cloud --function=tamlelan_handler \
  --region=us-west1 --project=gen-lang-client-0839027862 --no-traffic --quiet

gcloud run revisions describe <NEW_REVISION> --region=us-west1 --project=gen-lang-client-0839027862 \
  --format="value(metadata.annotations['run.googleapis.com/build-function-target'])"
# MUST print a JSON map with every value "tamlelan_handler", e.g.
# {"tamlelan-processor-1":"tamlelan_handler"} -- confirmed live 2026-09-02, current
# gcloud returns this keyed by container name, not a bare string. If empty or any
# value differs, do not migrate traffic, redeploy with the flag.

# then verify env vars survived (DRIVE_FOLDER_ID, APPS_SCRIPT_URL, GEMINI_API_KEY, WEBHOOK_SECRET)

gcloud run services update-traffic tamlelan-processor --to-latest --region=us-west1 \
  --project=gen-lang-client-0839027862
```

Old revisions are kept live at 0% traffic as the instant-rollback path — check the live traffic
split (`gcloud run services describe tamlelan-processor ... status.traffic`) rather than trusting a
remembered revision ID, since it changes on every deploy.

Rebuilding `dist/scribe.exe` (`build.bat`) unconditionally deletes the entire `dist/` directory
first — back up `dist/service_account.json`, `dist/tamlelan_client.log`, and any not-yet-uploaded
contents of `dist/Tamlelan_Backups/` before rebuilding.

---

## 7. Environment & Secrets

- **GCP project:** `gen-lang-client-0839027862` ("Default Gemini Project"), region `us-west1`.
  (A same-named decoy project exists — see §4.7.)
- **Cloud Run environment variables** (names only; values live in GCP Secret Manager / the Cloud
  Run service config, never in this repo): `DRIVE_FOLDER_ID`, `APPS_SCRIPT_URL`, `GEMINI_API_KEY`,
  `WEBHOOK_SECRET`.
- **Local credential files** (`Keys.txt`, `credentials.json`, `service_account.json`, `token.json`)
  are `.gitignore`d and, by standing project decision, never opened or read as part of any
  documentation or remediation effort.
