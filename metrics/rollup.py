"""
Pure, deterministic arithmetic over metrics/meetings.ndjson: filters to a
trailing window, buckets by meeting size, computes $ cost from known Gemini
pricing, and renders a Markdown table. No network calls, no AI calls -- this
is exactly the kind of task that should never go through an LLM.
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    GEMINI_PRICING, GEMINI_FLASH_LITE_PRICING, SHAPE_BUCKET_EDGES, BUCKET_ORDER,
    BUCKET_LABELS, TRAILING_WINDOW_DAYS,
)


def load_records(path):
    """Parses meetings.ndjson, skipping malformed lines with a warning rather
    than failing the whole load."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping malformed record at line {line_no}: {e}")
    return records


def _parse_ts(ts):
    if not ts:
        return None
    try:
        return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    except (ValueError, TypeError):
        return None


def trailing_window(records, days=TRAILING_WINDOW_DAYS, now=None):
    """Genuinely trailing N days from `now` (defaults to current UTC time),
    not a calendar month -- records with no parseable finished_at are excluded."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=days)
    out = []
    for r in records:
        ts = _parse_ts(r.get("finished_at"))
        if ts is not None and ts >= cutoff:
            out.append(r)
    return out


def bucket_for(speaker_count):
    if not isinstance(speaker_count, (int, float)) or isinstance(speaker_count, bool):
        return "unknown"
    for threshold, name in SHAPE_BUCKET_EDGES:
        if speaker_count >= threshold:
            return name
    return "unknown"


def cost_for(record):
    """Returns a float $ cost, or None if usage data is missing/incomplete --
    callers must exclude None from totals rather than treating it as $0."""
    usage = record.get("usage") or {}
    passes = [usage.get("pass1_summary"), usage.get("pass2_transcript")]
    if any(p is None for p in passes):
        return None

    total = 0.0
    for p in passes:
        prompt = p.get("prompt_tokens")
        cached = p.get("cached_tokens") or 0
        output = p.get("output_tokens")
        if prompt is None or output is None:
            return None
        uncached = max(prompt - cached, 0)
        total += uncached * GEMINI_PRICING["input_standard_per_million"] / 1_000_000
        total += cached * GEMINI_PRICING["input_cached_per_million"] / 1_000_000
        total += output * GEMINI_PRICING["output_per_million"] / 1_000_000

    if record.get("cache_used") and isinstance(record.get("cache_write_tokens"), (int, float)):
        total += record["cache_write_tokens"] * GEMINI_PRICING["cache_write_per_million"] / 1_000_000

    diagram = usage.get("diagram_generation")
    if record.get("diagram_generated") and isinstance(diagram, dict):
        d_prompt = diagram.get("prompt_tokens")
        d_output = diagram.get("output_tokens")
        if isinstance(d_prompt, (int, float)) and isinstance(d_output, (int, float)):
            total += d_prompt * GEMINI_FLASH_LITE_PRICING["input_standard_per_million"] / 1_000_000
            total += d_output * GEMINI_FLASH_LITE_PRICING["output_per_million"] / 1_000_000

    return total


def summarize(records):
    """Groups records into buckets and computes per-bucket count/hours/$."""
    buckets = {name: {"count": 0, "hours": 0.0, "cost": 0.0, "cost_count": 0} for name in BUCKET_ORDER}
    excluded = 0

    for r in records:
        b = bucket_for(r.get("speaker_count"))
        buckets[b]["count"] += 1
        dur = r.get("duration_sec")
        if isinstance(dur, (int, float)):
            buckets[b]["hours"] += dur / 3600.0
        cost = cost_for(r)
        if cost is not None:
            buckets[b]["cost"] += cost
            buckets[b]["cost_count"] += 1
        else:
            excluded += 1

    return buckets, excluded


def render_table(records):
    buckets, excluded = summarize(records)
    lines = []
    lines.append("| Bucket | Meetings | Hours | Total $ | Avg $/meeting |")
    lines.append("|---|---|---|---|---|")

    total_count = total_hours = total_cost = total_cost_count = 0
    for name in BUCKET_ORDER:
        b = buckets[name]
        if b["count"] == 0:
            continue
        avg = b["cost"] / b["cost_count"] if b["cost_count"] else None
        lines.append(
            f"| {BUCKET_LABELS[name]} | {b['count']} | {b['hours']:.1f} | "
            f"${b['cost']:.2f} | {f'${avg:.2f}' if avg is not None else '—'} |"
        )
        total_count += b["count"]
        total_hours += b["hours"]
        total_cost += b["cost"]
        total_cost_count += b["cost_count"]

    if total_count == 0:
        return "No meetings recorded in this window yet."

    avg_total = total_cost / total_cost_count if total_cost_count else None
    lines.append(
        f"| **Total** | **{total_count}** | **{total_hours:.1f}** | "
        f"**${total_cost:.2f}** | {f'${avg_total:.2f}' if avg_total is not None else '—'} |"
    )

    if excluded:
        lines.append("")
        lines.append(f"*{excluded} meeting(s) excluded from $ totals (missing/failed usage data).*")

    lines.append("")
    lines.append(
        "*$ figures are estimated from Gemini's self-reported token counts, which this "
        "project found to be unreliable on the input side for this model under explicit "
        "caching (see metrics/config.py) -- treat as directional, and periodically "
        "cross-check against Cloud Billing -> Reports (Group by: SKU).*"
    )

    return "\n".join(lines)
