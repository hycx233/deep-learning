import csv
import json
import unittest
from pathlib import Path

from src.window_splits import build_window_manifest, overlaps, validate_manifest, window_segments


def structure(identifier, center, kind="CHIN"):
    return {
        "ID": identifier, "type": kind, "chrom": "NC_000913.3",
        "start": center - 1, "end": center + 1, "center": center,
    }


class WindowSplitTests(unittest.TestCase):
    def test_circular_window_and_half_open_edges(self):
        start, end, segments = window_segments(5, 100, 20)
        self.assertEqual((start, end, segments), (95, 115, [[95, 100], [0, 15]]))
        self.assertTrue(overlaps(segments, [[98, 100]]))
        self.assertTrue(overlaps(segments, [[2, 4]]))
        self.assertFalse(overlaps(segments, [[15, 95]]))

    def test_transitive_and_origin_overlap_stay_together(self):
        source = [structure("a", 50), structure("b", 950), structure("c", 800),
                  structure("d", 500), structure("same_position", 50, "OPCID")]
        positions, summary = build_window_manifest(
            source, genome_length=1000, window_bp=200, background_count=0
        )
        assignments = {(row["group_id"], row["split"]) for row in positions
                       if row["structure_id"] != "d"}
        self.assertEqual(len(assignments), 1)
        self.assertEqual(summary["component_count"], 2)

    def test_validator_catches_overlap_across_origin(self):
        positions = [
            {"position_id": "a", "group_id": "a", "split": "train",
             "segments": '[{"start":95,"end":100},{"start":0,"end":15}]'},
            {"position_id": "b", "group_id": "b", "split": "test",
             "segments": '[{"start":10,"end":30}]'},
        ]
        with self.assertRaisesRegex(ValueError, "跨集合窗口重叠"):
            validate_manifest(positions)

    def test_nonoverlapping_windows_sharing_edge_bin_stay_together(self):
        source = [structure("a", 1010), structure("b", 1260)]
        positions, summary = build_window_manifest(
            source, genome_length=5000, window_bp=200, background_count=0
        )
        self.assertEqual(summary["component_count"], 1)
        self.assertEqual(positions[0]["group_id"], positions[1]["group_id"])

    def test_real_annotations_and_background(self):
        path = Path(__file__).resolve().parents[1] / "data" / "structures.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            source = list(csv.DictReader(handle))
        positions, summary = build_window_manifest(source)
        repeated, repeated_summary = build_window_manifest(source)
        self.assertEqual((positions, summary), (repeated, repeated_summary))
        known_only, _ = build_window_manifest(source, background_count=0)
        self.assertEqual(positions[:344], known_only)
        self.assertEqual(summary["component_count"], 59)
        self.assertEqual(summary["largest_component"], 20)
        self.assertEqual(summary["background_generated"], 24)
        self.assertEqual(summary["known_excluded"], [])
        for split, counts in summary["known_counts"].items():
            self.assertEqual(set(counts), {"OPCID", "CHIN", "CHID"}, split)
            self.assertTrue(all(count > 0 for count in counts.values()))
        known_intervals = [[int(row["start"]), int(row["end"])] for row in source]
        for position in positions:
            segments = [[part["start"], part["end"]]
                        for part in json.loads(position["segments"])]
            self.assertEqual(sum(end - start for start, end in segments), 24_000)
            if position["type"] == "background":
                self.assertFalse(overlaps(segments, known_intervals))
        # 独立逐碱基区间对照，不只依赖生产函数的 validate_manifest。
        for index, left in enumerate(positions):
            for right in positions[:index]:
                if left["split"] == right["split"]:
                    continue
                for segment in json.loads(left["segments"]):
                    start = segment["start"] // 100 * 100
                    end = min((segment["end"] + 99) // 100 * 100, 4_641_652)
                    for other in json.loads(right["segments"]):
                        other_start = other["start"] // 100 * 100
                        other_end = min((other["end"] + 99) // 100 * 100, 4_641_652)
                        self.assertTrue(end <= other_start or other_end <= start)


if __name__ == "__main__":
    unittest.main()
