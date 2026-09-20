from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.discovery_data import (
    DiscoveryDataError,
    load_discovery_dataset,
    select_background_rows,
    split_background_rows,
)


def write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    meta = pd.DataFrame(
        {
            "window_id": ["bg_train", "bg_val", "known"],
            "chrom": ["NC_000913.3"] * 3,
            "start": [0, 30_000, 60_000],
            "end": [24_000, 54_000, 84_000],
            "sample_id": ["WT_rep1"] * 3,
            "condition": ["WT"] * 3,
            "replicate": [1] * 3,
            "known_overlap": [False, False, True],
            "split": ["train", "val", "test"],
        }
    )
    windows = tmp_path / "windows.csv"
    arrays = tmp_path / "windows.npz"
    meta.to_csv(windows, index=False)
    np.savez_compressed(
        arrays,
        X=np.zeros((3, 128, 128), dtype=np.float32),
        intensity_scores=np.arange(3, dtype=np.float64),
    )
    return windows, arrays


class DiscoveryDataTests(unittest.TestCase):
    def test_load_and_select_background(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            windows, arrays = write_dataset(Path(directory))
            matrices, meta, intensity = load_discovery_dataset(windows, arrays)
            rows = select_background_rows(meta)
            train_rows, val_rows = split_background_rows(meta, rows)

        self.assertEqual(matrices.shape, (3, 128, 128))
        self.assertEqual(intensity.tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(train_rows.tolist(), [0])
        self.assertEqual(val_rows.tolist(), [1])

    def test_formal_input_requires_comparable_intensity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            windows, arrays = write_dataset(Path(directory))
            with np.load(arrays) as data:
                matrices = data["X"].copy()
            np.savez_compressed(arrays, X=matrices)

            with self.assertRaisesRegex(DiscoveryDataError, "强度分数"):
                load_discovery_dataset(windows, arrays)

    def test_background_selection_requires_known_relation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            windows, _ = write_dataset(Path(directory))
            meta = pd.read_csv(windows).drop(columns="known_overlap")

            with self.assertRaisesRegex(DiscoveryDataError, "known_overlap"):
                select_background_rows(meta)


if __name__ == "__main__":
    unittest.main()
