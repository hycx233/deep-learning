"""计数、环状距离和首尾窗口几何的针对性检查；仅生成小型临时 Cooler。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cooler
import numpy as np
import pandas as pd

from src.preprocessing import _valid_distance_pairs, extract_window, load_sample, prepare_sample


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="preprocessing-test-")
        cls.root = Path(cls.temporary.name)
        starts = np.arange(106) * 10
        bins = pd.DataFrame({
            "chrom": ["NC_000913.3"] * 106, "start": starts,
            "end": np.minimum(starts + 10, 1052),
        })
        # 包含落在同一粗 bin 中的细粒度非对角计数，以及跨起点的接触。
        left = np.array(list(range(106)) + [0, 0, 1, 5, 10, 20, 94, 103])
        right = np.array(list(range(106)) + [1, 91, 99, 100, 90, 93, 103, 104])
        counts = np.array(list(range(1, 107)) + [13, 17, 19, 23, 29, 31, 37, 41])
        pixels = pd.DataFrame({"bin1_id": left, "bin2_id": right, "count": counts})
        pixels = pixels.sort_values(["bin1_id", "bin2_id"])
        cooler.create_cooler(str(cls.root / "source.cool"), bins, pixels, dtypes={"count": np.int64})
        cls.sample = {
            "sample_id": "toy", "condition": "toy", "replicate": "1",
            "chrom": "NC_000913.3", "path": "source.cool",
        }
        cls.metadata = prepare_sample(cls.sample, cls.root, cls.root / "cache", chunksize=25)
        cls.state = load_sample(cls.root / "cache", "toy")
        cls.upper = np.zeros((11, 11), dtype=np.int64)
        np.add.at(cls.upper, (left // 10, right // 10), counts)
        cls.manual = cls.upper + cls.upper.T - np.diag(np.diag(cls.upper))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_unique_counts_marginal_and_cache_reuse(self):
        self.assertEqual(self.metadata["total_counts"], int(self.upper.sum()))
        np.testing.assert_array_equal(self.state["cooler"].matrix(balance=False)[:], self.manual)
        np.testing.assert_array_equal(self.state["marginal"], self.manual.sum(axis=1))
        self.assertEqual(self.state["bin_widths"][-1], 52)
        self.assertFalse(self.state["valid_bins"][-1])
        before = (self.root / "cache/toy.cool").stat().st_mtime_ns
        metadata = prepare_sample(self.sample, self.root, self.root / "cache", chunksize=25)
        self.assertEqual(metadata, self.metadata)
        self.assertEqual((self.root / "cache/toy.cool").stat().st_mtime_ns, before)

    def test_fft_pairs_and_expected_include_zero_contacts(self):
        for n_bins in range(2, 18):
            for offset in range(5):
                valid = (np.arange(n_bins) + offset) % 3 != 0
                pairs = np.zeros(n_bins // 2 + 1, dtype=np.int64)
                for left in range(n_bins):
                    for right in range(left, n_bins):
                        if valid[left] and valid[right]:
                            pairs[min(right - left, n_bins - right + left)] += 1
                np.testing.assert_array_equal(_valid_distance_pairs(valid), pairs)
        sums = np.zeros(6)
        pairs = np.zeros(6)
        for left in range(10):
            for right in range(left, 10):
                distance = min(right - left, 11 - right + left)
                pairs[distance] += 1
                sums[distance] += self.upper[left, right]
        np.testing.assert_array_equal(self.state["distance_pair_counts"], pairs)
        np.testing.assert_allclose(self.state["expected_raw"], sums / pairs)

    def test_circular_blocks_coordinates_and_masks(self):
        for center in (0, 250, 1051):
            window = extract_window(self.state, center, window_bp=500, image_size=16)
            ids = window["bin_ids"]
            np.testing.assert_array_equal(window["counts"], self.manual[np.ix_(ids, ids)])
            valid = self.state["valid_bins"][ids]
            pair_mask = valid[:, None] & valid[None, :]
            np.testing.assert_array_equal(window["intensity_mask"], pair_mask)
            expected = self.manual[np.ix_(ids, ids)] * self.metadata["depth_factor"]
            expected[~pair_mask] = 0
            np.testing.assert_allclose(window["intensity"], expected)
            self.assertFalse(np.diag(window["mask"]).any())
            self.assertEqual(window["end"] - window["start"], 500)
            self.assertEqual(sum(piece["end"] - piece["start"] for piece in window["segments"]), 500)
            self.assertEqual(window["window_edges"][0], 0)
            self.assertEqual(window["window_edges"][-1], 500)
            self.assertTrue(np.all(np.diff(window["window_edges"]) > 0))
        np.testing.assert_array_equal(
            extract_window(self.state, 0, window_bp=500)["window_edges"],
            [0, 98, 198, 250, 350, 450, 500],
        )

    def test_area_resize_weights_and_zero_expected(self):
        window = extract_window(self.state, 0, window_bp=500, image_size=16)
        edges = window["window_edges"]
        target_edges = np.linspace(0, 500, 17)
        expected_model = np.zeros((16, 16))
        expected_fraction = np.zeros((16, 16))
        morphology = np.log1p(np.minimum(window["oe"].astype(float), 10)) / np.log(11)
        for row in range(16):
            row_overlap = np.maximum(0, np.minimum(target_edges[row + 1], edges[1:])
                                     - np.maximum(target_edges[row], edges[:-1]))
            for column in range(16):
                col_overlap = np.maximum(0, np.minimum(target_edges[column + 1], edges[1:])
                                         - np.maximum(target_edges[column], edges[:-1]))
                area = row_overlap[:, None] * col_overlap[None, :] * window["mask"]
                if area.sum():
                    expected_model[row, column] = (area * morphology).sum() / area.sum()
                expected_fraction[row, column] = area.sum() / (500 / 16) ** 2
        np.testing.assert_allclose(window["model"], expected_model, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(window["model_valid_fraction"], expected_fraction)
        self.assertTrue(np.isfinite(window["model"]).all())
        self.assertGreaterEqual(window["model"].min(), 0)
        self.assertLessEqual(window["model"].max(), 1)
        no_background = dict(self.state, expected_raw=np.zeros_like(self.state["expected_raw"]))
        empty = extract_window(no_background, 0, window_bp=500)
        self.assertFalse(empty["mask"].any())
        self.assertFalse(empty["model_mask"].any())
        self.assertEqual(empty["model"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
