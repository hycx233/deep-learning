import unittest

import numpy as np

from scripts.prepare_discovery_scan import intensity_score, overlaps_segments


class DiscoveryScanTests(unittest.TestCase):
    def test_overlap_checks_each_circular_segment(self) -> None:
        segments = [{"start": 4_630_000, "end": 4_641_652}, {"start": 0, "end": 12_348}]

        self.assertTrue(overlaps_segments(segments, 500, 900))
        self.assertTrue(overlaps_segments(segments, 4_635_000, 4_636_000))
        self.assertFalse(overlaps_segments(segments, 20_000, 21_000))
        self.assertFalse(overlaps_segments(segments, 12_348, 13_000))

    def test_intensity_uses_valid_upper_triangle_without_diagonal(self) -> None:
        values = np.array([[100.0, 2.0, 4.0], [2.0, 100.0, 8.0], [4.0, 8.0, 100.0]])
        valid = np.ones((3, 3), dtype=bool)
        valid[0, 2] = valid[2, 0] = False

        score = intensity_score({"intensity": values, "intensity_mask": valid})

        self.assertEqual(score, 5.0)


if __name__ == "__main__":
    unittest.main()
