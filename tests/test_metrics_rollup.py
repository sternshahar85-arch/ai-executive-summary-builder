"""
Verifies metrics/rollup.py's pure arithmetic: loading, trailing-window
filtering, bucketing, cost computation, and table rendering. No mocking --
this is deterministic code with no external dependencies.

Run with: .venv-cloud/Scripts/python.exe tests/test_metrics_rollup.py
"""
import datetime
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "metrics"))

import rollup

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_meetings.ndjson")
REFERENCE_NOW = datetime.datetime(2026, 8, 31, 12, 0, 0, tzinfo=datetime.timezone.utc)


class TestLoadRecords(unittest.TestCase):
    def test_loads_all_fixture_records(self):
        records = rollup.load_records(FIXTURE_PATH)
        self.assertEqual(len(records), 5)

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(rollup.load_records("/no/such/file.ndjson"), [])

    def test_skips_malformed_lines(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson", delete=False) as f:
            f.write('{"event_id": "ok-1", "finished_at": "2026-08-25T10:00:00Z"}\n')
            f.write("NOT VALID JSON {{{\n")
            f.write('{"event_id": "ok-2", "finished_at": "2026-08-26T10:00:00Z"}\n')
            path = f.name
        try:
            records = rollup.load_records(path)
            self.assertEqual(len(records), 2)
        finally:
            os.remove(path)


class TestBucketFor(unittest.TestCase):
    def test_edges(self):
        self.assertEqual(rollup.bucket_for(2), "one_on_one")
        self.assertEqual(rollup.bucket_for(3), "small_group")
        self.assertEqual(rollup.bucket_for(6), "small_group")
        self.assertEqual(rollup.bucket_for(7), "large_complex")
        self.assertEqual(rollup.bucket_for(15), "large_complex")

    def test_unknown_for_missing_or_degenerate(self):
        self.assertEqual(rollup.bucket_for(None), "unknown")
        self.assertEqual(rollup.bucket_for(0), "unknown")
        self.assertEqual(rollup.bucket_for(1), "unknown")


class TestTrailingWindow(unittest.TestCase):
    def test_excludes_old_record(self):
        records = rollup.load_records(FIXTURE_PATH)
        window = rollup.trailing_window(records, days=30, now=REFERENCE_NOW)
        ids = {r["event_id"] for r in window}
        self.assertNotIn("evt-5-old", ids)
        self.assertEqual(len(window), 4)


class TestCostFor(unittest.TestCase):
    def test_one_on_one_no_cache(self):
        records = {r["event_id"]: r for r in rollup.load_records(FIXTURE_PATH)}
        cost = rollup.cost_for(records["evt-1"])
        # pass1: 10000*2/1e6 + 0 + 1000*12/1e6 = 0.032; pass2 same shape = 0.044; total 0.076
        self.assertAlmostEqual(cost, 0.076, places=6)

    def test_small_group_with_cache(self):
        records = {r["event_id"]: r for r in rollup.load_records(FIXTURE_PATH)}
        cost = rollup.cost_for(records["evt-2"])
        self.assertAlmostEqual(cost, 0.085625, places=6)

    def test_missing_pass2_returns_none(self):
        records = {r["event_id"]: r for r in rollup.load_records(FIXTURE_PATH)}
        self.assertIsNone(rollup.cost_for(records["evt-4"]))


class TestSummarizeAndRender(unittest.TestCase):
    def test_summarize_buckets_and_excludes_failed(self):
        records = rollup.load_records(FIXTURE_PATH)
        window = rollup.trailing_window(records, days=30, now=REFERENCE_NOW)
        buckets, excluded = rollup.summarize(window)
        self.assertEqual(buckets["one_on_one"]["count"], 1)
        self.assertEqual(buckets["small_group"]["count"], 1)
        self.assertEqual(buckets["large_complex"]["count"], 1)
        self.assertEqual(buckets["unknown"]["count"], 1)
        self.assertEqual(excluded, 1)  # evt-4's missing pass2 usage

    def test_render_table_produces_markdown(self):
        records = rollup.load_records(FIXTURE_PATH)
        window = rollup.trailing_window(records, days=30, now=REFERENCE_NOW)
        table = rollup.render_table(window)
        self.assertIn("| Bucket |", table)
        self.assertIn("**Total**", table)
        self.assertIn("excluded from $ totals", table)

    def test_empty_window_message(self):
        self.assertEqual(rollup.render_table([]), "No meetings recorded in this window yet.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
