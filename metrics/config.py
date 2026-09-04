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
    # Corrected 2026-09-04 against real billing SKUs. Was 0.375. There is no
    # separate "cache write" SKU: writes appear inside the ordinary audio/text
    # input counts and bill at the standard input rate. Storage is billed
    # separately as "cached ... storage token hours" at $4.50/M/hour, which
    # cost_for() does NOT model -- it was ~$0.04 per meeting at the 600s TTL,
    # and explicit caching was removed from the pipeline when Pass 2 became
    # chunked, so this constant now only affects historical records.
    "cache_write_per_million": 2.00,
    # Thinking tokens bill at this same rate -- see rollup.thinking_tokens().
    "output_per_million": 12.00,
}

GEMINI_FLASH_LITE_PRICING = {
    # gemini-3.1-flash-lite, used only for the diagram-generation pass
    # (cloud/main.py, no caching involved there -- no cache fields needed
    # here). Pricing for this model was volatile at time of writing (2026-09):
    # third-party trackers cited figures ranging $0.125-0.25/M input and
    # $0.75-1.50/M output depending on source and date, not yet cross-checked
    # against ai.google.dev/pricing directly. Re-verify before trusting this
    # component's dollar output for anything beyond a rough estimate.
    "input_standard_per_million": 0.25,
    "output_per_million": 1.50,
}

# KNOWN LIMITATION, verified 2026-08-31: the input-token half of cost_for()'s
# math (prompt_tokens/cached_tokens from Gemini's own usage_metadata) does not
# appear to be reliable for gemini-3.1-pro-preview under explicit caching --
# a real call reporting cached_content_token_count=107,326 reported
# prompt_token_count=85,363, smaller than the cache, which the documented
# semantics (prompt_token_count includes cached tokens as a subset) says
# shouldn't be possible. Directly measuring the real prompt text with
# count_tokens() showed the true new content was ~876 tokens; real GCP
# billing for the entire "text input" SKU across four months was ₪0.25 --
# nowhere near what a single call's prompt_token_count implied. This matches
# a known class of problem elsewhere too (see comet-ml/opik issue #6976,
# googleapis/python-genai issue #2064) -- not something specific to this
# codebase, and not something to silently trust. The OUTPUT-token numbers
# are unaffected by this (independently verified, and matched by real
# billing) -- only the input-side dollar estimate below is in question.
# Treat cost_for()'s output as directional, not precise, and cross-check
# periodically against Cloud Billing -> Reports (Group by: SKU).
#
# CORRECTION, 2026-09-04: the note above claims "the OUTPUT-token numbers are
# unaffected by this ... and matched by real billing". That was wrong, and it
# was the more expensive error of the two. cost_for() counted only
# candidates_token_count and ignored THINKING tokens, which bill at the same
# output rate. The first real Cloud Billing cross-check (the one this comment
# had been recommending since August, never actually performed) showed the
# "text output token count" SKU at 551,355 tokens against ~144,000 counted --
# the missing ~407,000 were thinking. Per-meeting cost was understated by
# 43-106%, and total spend for 2026-09-03 by 2.02x.
#
# Thinking tokens are now derived in rollup.thinking_tokens() from
# total_token_count, which the durable records had been storing all along.
# The input-side caveat above still stands, but it is worth under a cent per
# meeting; the output side was the one that mattered.

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
