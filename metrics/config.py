"""
Single source of truth for the two judgment calls the metrics rollup depends
on. Durable records (metrics/meetings.ndjson) store raw facts only -- never
dollar cost or a bucket label -- so changing either constant here is a
one-line edit with zero data migration, applied retroactively to every past
record on the next rollup run.
"""

GEMINI_PRICING = {
    # gemini-3.1-pro-preview, confirmed 2026-08-31. Re-verify against
    # ai.google.dev/pricing periodically -- this is a preview model and
    # pricing can change without notice.
    "input_standard_per_million": 2.00,
    "input_cached_per_million": 0.20,
    "cache_write_per_million": 0.375,
    "output_per_million": 12.00,
}

# Maps a minimum speaker_count to a bucket name. A record's bucket is the
# highest threshold its speaker_count meets or exceeds. speaker_count of 0/1,
# or missing (no diarization companion), falls through to "unknown" rather
# than being silently forced into a bucket.
SHAPE_BUCKET_EDGES = [
    (7, "large_complex"),
    (3, "small_group"),
    (2, "one_on_one"),
]

BUCKET_ORDER = ["one_on_one", "small_group", "large_complex", "unknown"]

BUCKET_LABELS = {
    "one_on_one": "1-on-1",
    "small_group": "Small group (3-6)",
    "large_complex": "Large/complex (7+)",
    "unknown": "Unknown (no speaker count)",
}

TRAILING_WINDOW_DAYS = 30
