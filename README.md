# AI Executive Summary Builder

**Document type:** architecture and current state
**As of:** 2026-09-04, production revision `tamlelan-processor-00064-zuw`
**Objective:** an autonomous, serverless, zero-subscription AI meeting agent. It records
local audio, uploads it to a cloud bucket, triggers an LLM pipeline for structured data
extraction and transcription, and writes a Hebrew summary and transcript to a personal
Google Drive.

This document describes the system **as it actually runs today**. Every claim below was
checked against the live deployment or the code on 2026-09-04. Where something is
unverified, it says so.

---

## How This Was Built

This system was built with Claude Code as an implementation partner. My role across the
project: defining requirements, researching and deciding between architectural alternatives
— including whether to keep an audio-native LLM pipeline or move to a dedicated
speech-to-text service, which was evaluated against real recordings and rejected on the
measurements rather than on preference — approving plans after weighing their tradeoffs,
directing investigations into production defects, and continuously verifying real system
output against real recordings rather than trusting a green test suite alone.

One concrete example, carried through to its end. A production meeting surfaced a
transcript-duplication defect. I did not accept the first plausible-sounding hypothesis; I
directed a live A/B against the actual recording instead. What broke it open was testing the
output's **structure** rather than its content: comparing each output row against the
diarization segment at the same index showed **217 of 217 timestamps and labels matching at
identical indices**. That is not transcription with errors in it — that is a template being
copied, and no amount of reading the transcript would have revealed it.

The fix was not the obvious one either. Pass 1 was then tested the same way and behaved in
the **opposite** direction: the same speaker list that Pass 2 was copying is what keeps Pass
1 factually anchored to the audio. The two passes needed opposite treatment — Pass 1 keeps
the turn list, Pass 2 must never receive it (§3.2). A fix applied uniformly to both would
have traded one defect for another.

The deploy-safety workflow (`.github/workflows/deploy.yml`) exists for the same reason: a
past deploy broke silently in production, and the fix was to replace a manual checklist with
an automated gate that fails loudly instead of trusting that the checklist gets followed
correctly every time (§5, §6).

---

## 1. Current architecture

### 1.1 Local client (Windows)

- **Component:** `scribe.py`, compiled to `scribe.exe` via PyInstaller (`build.bat`).
- **Frameworks:** Tkinter (GUI), `sounddevice`/`PyAudioWPatch` (capture), `scipy.io.wavfile`
  (encoding), `google-cloud-storage` (upload), `sherpa-onnx` (local diarization, lazily
  imported).
- **Capture:** 16 kHz, 16-bit PCM. The operator's microphone and remote/system audio are
  captured as **separate channels** and written to a stereo `.wav` (left = operator,
  right = remote) rather than mixed to mono, so the two can be analysed independently. A
  single-channel device falls back to mono with no channel distinction.
- **Backup retention:** local `.wav` copies are kept for 7 days in `Tamlelan_Backups/`
  regardless of outcome; the diarization companion JSON is kept **permanently** alongside
  it. Note the sweep only runs at application launch, so files can outlive 7 days if the
  client is not started.
- **Upload order (load-bearing):** the diarization companion is uploaded to GCS *before*
  the `.wav`. The `.wav`'s finalize event is what triggers the pipeline, and GCS provides
  strong read-after-write consistency, so the companion is guaranteed present when the
  handler runs. If it were ever absent, the pipeline degrades cleanly rather than failing.
- **Auth:** a local `service_account.json` scoped to `roles/storage.objectCreator` only.

### 1.2 Ingestion (Google Cloud Storage)

- **Bucket:** `tamlelan-inbox-stgliding`, region `us-west1`, uniform bucket-level access,
  public access prevention enforced.
- **Eventing:** Eventarc trigger on `google.cloud.storage.object.v1.finalized` with no path
  filter; all filtering happens in application code.
- **Prefixes in use:**

| Prefix | Purpose | Retention |
|---|---|---|
| *(root)* | incoming `.wav` + `.diarization.json` | deleted on success |
| `locks/` | one lock per Eventarc event id, released in `finally` | 1 day (lifecycle) |
| `content_hashes/` | crc32c marker per successfully processed recording | 365 days (lifecycle) |
| `failed/` | dead-letter copy of recordings that failed processing | 30 days (lifecycle) |
| `metrics/` | one structured record per run | kept |

- **Duplicate protection is two independent mechanisms.** `locks/{event_id}` stops two
  concurrent deliveries of the *same* Eventarc event doing duplicate work, and is deleted
  when the run finishes. `content_hashes/{crc32c}` stops the *same audio content* being
  processed twice as two distinct uploads, and is written only on success. To deliberately
  reprocess a recording, delete its `content_hashes/` marker first.

### 1.3 Compute and intelligence (Cloud Run)

- **Component:** `tamlelan-processor`, 2nd-gen Cloud Run function (source: `cloud/`,
  entry point `tamlelan_handler`), triggered by the Eventarc trigger above.
- **Runtime (verified live):** Python 3.12, 1 vCPU, **2 GiB** memory, **3600 s** timeout,
  0 minimum instances, max 20, **ingress `internal`**, running as
  `tamlelan-processor-sa@`.
- **Libraries:** `functions-framework`, `google-genai`, `google-cloud-storage`,
  `rapidfuzz`, `httpx`.

**Pass 1 — structured JSON** (`gemini-3.1-pro-preview`, temperature 0.2, enforced via
`response_schema`): executive summary, attendees (voice-audibility required, not merely
"mentioned"), a separate `people_mentioned` list, key topics, a decisions log with a
`decided`/`proposed`/`open` enum and hedge capture, action items (owner/deadline only if
explicitly stated), and a `diagram_needed` flag. Pass 1 **receives the full diarization
turn list** — see §3.

**Pass 2 — verbatim Hebrew transcript** (`gemini-3.1-pro-preview`, temperature 0.1,
`thinking_level="LOW"`): every line labelled `M:SS [SPEAKER]: text`. Pass 2 is **chunked**
and receives the speaker roster **without** the turn list — see §3.

**Secondary model:** `gemini-3.1-flash-lite` generates a Mermaid diagram when
`diagram_needed` is true. The model returns only the graph body; the HTML page is built
locally from a fixed template with a Content-Security-Policy, and the model output is
sanitised first. The diagram step is isolated, so a failure there cannot discard a run
whose expensive work already succeeded.

**Verification performed locally, at no API cost:**

- **Grounding:** every `source_quote` is fuzzy-matched (`rapidfuzz`, threshold 80) against
  the Pass 2 transcript; an unmatched quote is flagged (⚠), never silently dropped.
- **Attendee cross-check:** flags the attendees list when it names more people than
  diarization detected voices (+1 tolerance).
- **Topic referential integrity:** flags any decision or action item whose
  `related_topic_id` has no matching `key_topics` entry.
- **Transcript integrity** (`cloud/transcript_checks.py`) — see §3.

**Cleanup:** strict `try`/`finally`. On success the source `.wav`, the companion, and the
uploaded Gemini files are deleted and the lock released. On failure the `.wav` and
companion are copied to `failed/` server-side (`copy_blob`) and the lock is deliberately
**retained**, since the source is no longer in the inbox and a redelivery could only fail
again.

### 1.4 Storage bridge (Google Apps Script)

Workaround for GCP service accounts having a 0-byte default Drive quota: they cannot write
files to a personal `@gmail.com` Drive folder directly.

The script is a `doPost(e)` web app deployed as a Web App, running as the personal account.
Its source was verified against the live deployment on 2026-09-03. It:

1. reads `WEBHOOK_SECRET` from **Script Properties** and compares it to the `secret` field
   **in the JSON request body** (there is no `X-Tamlelan-Secret` header — Apps Script
   `doPost` cannot read request headers at all);
2. checks `folder_id` against an allowlist containing only `SUMMARIES_FOLDER_ID`;
3. calls `DriveApp.getFolderById(...)` then `folder.createFile(...)`.

**It never reads, deletes or overwrites.** The OAuth grant is broader than that, but the
deployed code exercises only file creation in one folder, which bounds the practical blast
radius of a leaked secret.

`send_to_drive` in `cloud/main.py` **checks the response**. The script answers HTTP 200
with `{"status":"error"}` for a rejected write, so an unchecked response meant a failed
delivery looked identical to a success — and the source recording was then deleted. The
call is now bounded by a timeout and retried.

---

## 2. Multi-speaker diarization

A stereo split alone only separates "operator" from "everyone else", and a same-room
single-microphone meeting has no second channel to split at all.

**What runs client-side before every upload:**

- **Models:** `pyannote-segmentation-3.0` (~6 MB) + 3D-Speaker CAM++ embeddings (~28 MB),
  via `sherpa-onnx`, CPU-only, auto-downloaded on first run.
- **Stereo:** the right channel is diarized to separate the voices Zoom mixed together; the
  left channel is diarized forced to exactly one cluster.
- **Mono:** the full track is diarized directly, with no privileged operator identity.
- **Manual headcount override:** a "Participants (incl. you)" field fixes the expected
  cluster count. Automatic clustering (`num_clusters=-1`) badly over-segments real
  conversation — a real 2-participant call produced 16+ spurious labels.
- **Transport:** a companion GCS object, not object metadata — GCS's 8 KiB metadata cap is
  exceeded by a real meeting's segment list.
- **Zero added cost:** diarization never involves an API call.

Companion contents are validated on load: segment times are coerced to numbers, labels must
be strings and are length-capped and character-filtered. Labels are interpolated into both
prompts, so an unvalidated label would be a prompt-injection channel.

---

## 3. Transcript integrity — the central design constraint

This section replaces the former "context caching" section. Explicit context caching was
**removed** on 2026-09-03: it existed because the same audio was sent twice, and once Pass 2
became chunked the cache would serve a single read. A single-use cache costs more than none
($0.315 vs $0.214 per meeting), because writing it bills at the full input rate.

### 3.1 The defect, and its measured cause

A real 44-minute meeting produced a transcript whose last third was the same block repeated
three times, with speaker labels degrading, and a summary asserting a family relationship
nobody stated. A controlled A/B on that recording found the cause:

| | Pass 2 with the turn list | Pass 2 with roster only |
|---|---|---|
| Output lines | 723 — exactly the segment count | genuine transcription |
| Timestamp echo | **1.00** | **0.00** |
| Label echo | **1.00** | **0.00** |
| Duplicated content | **41%** | **0%** |

Pass 2 was not transcribing. It was **slot-filling the diarization turn list as a
template**, copying each row's timestamp and label. Removing the list eliminated it.

### 3.2 Why the two passes are treated differently

Pass 1 was then tested the same way, and behaves the **opposite** way: without the turn
list it invented a family relationship ("Clara and her family members") unsupported by the
audio. So **Pass 1 keeps the turn list; Pass 2 must not have it.** Pass 1 produces a small
structured object where the roster acts as a factual constraint; Pass 2 produces a long
parallel list where the same roster becomes a template to copy.

### 3.3 Why Pass 2 is chunked

Removing the turn list exposed a second, independent failure: the model entered a
degenerate loop and repeated one token 29,654 times inside a single line until the output
budget was exhausted. Chunking **contains** that: a loop can only ruin one window, which is
cheap to detect and retry.

- Windows of roughly 9 minutes, cut on diarization turn boundaries so no utterance is split
  and no overlap deduplication is needed.
- Speaker identity is stable across windows because clustering runs globally before
  chunking; names resolved in earlier chunks are carried into later ones.
- Each chunk is detected for degeneration and retried once at a higher temperature.
- If the audio cannot be sliced, the pipeline falls back to a single whole-file pass.

### 3.4 Deterministic transcript checks

`cloud/transcript_checks.py` runs locally at no cost and **flags, never drops**:

| Check | Catches |
|---|---|
| Diarization echo | output copying the template instead of the audio |
| Intra-line degeneration | a token repeated many times inside one line |
| Block repetition | duplicated dialogue, ignoring timestamps and labels |
| Timestamp monotonicity | a transcript replaying an earlier point |
| Coverage | a transcript that stops short of the recording |
| Speaker-label stability | more distinct labels than there are voices |

Results are surfaced as a banner on the delivered transcript and recorded in the metrics
record, so defect rates become measurable rather than anecdotal.

### 3.5 Validated in production

Both archived recordings were reprocessed through the live pipeline on 2026-09-03:

| | 44.7 min, 5 speakers | 54.2 min, 8 speakers |
|---|---|---|
| Chunks | 5 | 6 |
| Timestamp echo | 0.003 | 0.005 |
| Duplicated fraction | 0.0 | 0.0 | 
| Coverage | 0.999 | 0.999 |
| Transient error recovered | 504 | 503 |
| Cost (incl. thinking tokens) | $0.671 | $0.842 |

---

## 4. Known open items

1. **Speaker labels proliferate on long, many-voice meetings.** The 8-voice recording
   produced 19 distinct labels (10 names + 9 generic); the 5-voice one named only 3 of its
   5 people. The label-stability check now flags this, but the underlying attribution is
   not fixed.
2. **Same-room, single-microphone diarization has only synthetic test coverage.** No real
   in-person recording has been through the mono path.
3. **The Apps Script web app is deployed with "Anyone" access.** The secret was rotated on
   2026-09-03 and the deployed code only creates files in one allowlisted folder, which
   bounds the impact. Restricting access would require the pipeline to mint and send an
   OIDC identity token.
4. **The Apps Script `catch` block returns `error.toString()`** to the caller, a minor
   information leak.
5. **Running cost rose about 19% with chunking**, not fallen. Measured on the same
   recording processed by both designs: $0.706 -> $0.842. Input volume is unchanged;
   the increase is entirely thinking tokens, because six chunked calls each pay their
   own reasoning overhead. Real cost is roughly $0.67-0.95 per meeting.
6. **`gemini-3.1-pro-preview` is a preview model.** Its predecessor was shut down with
   weeks of notice, and preview behaviour can change silently.

---

## 5. Testing

**175 tests across 16 files**, in two isolated environments.

| Suite | Environment | Tests |
|---|---|---|
| `test_main_*.py` | `.venv-cloud` (`cloud/requirements.txt`) | 62 |
| `test_transcript_*.py` | `.venv-cloud` | 32 |
| `test_chunking.py` | `.venv-cloud` | 25 |
| `test_metrics_*.py` | `.venv-cloud` | 25 |
| `test_scribe_*.py` | system Python (`requirements-client.txt`) | 31 |

The two environments are genuinely separate: the client suite needs `numpy`/`scipy`, which
must not be present when the cloud suite runs, or a missing dependency could pass CI and
crash on Cloud Run. `test_scribe_diarization.py` must import cleanly without `sherpa-onnx`
installed. All external services are mocked; no credentials are needed.

```
.venv-cloud/Scripts/python.exe -m unittest discover -s tests -p "test_main_*.py"
.venv-cloud/Scripts/python.exe -m unittest discover -s tests -p "test_transcript_*.py"
.venv-cloud/Scripts/python.exe -m unittest discover -s tests -p "test_chunking.py"
.venv-cloud/Scripts/python.exe -m unittest discover -s tests -p "test_metrics_*.py"
python -m unittest discover -s tests -p "test_scribe_*.py"
```

---

## 6. Deployment

Deployment is a **GitHub Actions workflow** (`.github/workflows/deploy.yml`), manual by
design (`workflow_dispatch`). Tests additionally run on every push and pull request.

The test suites are a **structural gate**: the `deploy` job declares `needs: [test-cloud,
test-client]`, so a red suite makes deployment unreachable rather than merely out of order.

The deploy job then, in order:

1. deploys with `--no-traffic --tag`, pinning `--timeout=3600` and `--memory=2Gi`;
2. verifies the `run.googleapis.com/build-function-target` annotation is `tamlelan_handler`
   for every container — a missing `--function` flag once produced containers that reported
   `Ready: True` and crashed on every real request, silently, four deploys in a row;
3. verifies all four env vars are present **and non-empty**;
4. migrates traffic with `--to-revisions=<verified>=100` (not `--to-latest`, which routes by
   creation time and could select a revision the checks never saw);
5. confirms the migration actually landed, then removes the temporary tag.

Old revisions remain at 0% traffic as the rollback path:

```
gcloud run services update-traffic tamlelan-processor \
  --to-revisions=<PREVIOUS_REVISION>=100 --region=us-west1 --project=gen-lang-client-0839027862
```

There is no synthetic smoke test. The service is `ingress=internal`, evaluated *before*
IAM, so a GitHub runner is rejected at the network edge regardless of payload or
credentials. A `--no-traffic` tagged revision also cannot be reached by real Eventarc, so
any pre-migration smoke test is inherently synthetic. The test gate covers considerably
more.

Rebuilding `dist/scribe.exe` via `build.bat` **deletes the entire `dist/` directory first**,
including `dist/Tamlelan_Backups/`. Move any recordings you care about out first.

---

## 7. Environment and secrets

- **GCP project:** `gen-lang-client-0839027862`, region `us-west1`, project number
  `138198608878`.
- **Cloud Run env vars:** `GEMINI_API_KEY` and `WEBHOOK_SECRET` are **Secret Manager
  references** resolved at instance start; `DRIVE_FOLDER_ID` and `APPS_SCRIPT_URL` are
  literal values.
- **Deployment identity:** GitHub Actions authenticates via Workload Identity Federation.
  The provider condition is scoped to the numeric repository id, the numeric owner id, and
  `refs/heads/master` — numeric ids because repository *names* can be reclaimed after
  deletion.
- **Deployer permissions:** `run.admin`, `cloudbuild.builds.editor`,
  `artifactregistry.writer`, a custom role granting only `storage.buckets.list`, and
  `storage.admin` **scoped to the Cloud Build staging bucket alone**. The deployer has no
  access to the meeting-audio bucket.
- **Local credential files** (`Keys.txt`, `credentials.json`, `service_account.json`,
  `token.json`) are gitignored and have never been committed — verified across all refs.
- **`RECENT/`** is gitignored. It is scratch space and may contain real meeting content.

> **A deleted project can revoke your API key.** On 2026-09-03 an apparently empty
> "decoy" project was deleted; it had issued the Gemini API key, and production broke
> silently until the next recording would have failed. Check which project owns a key
> before deleting anything.
