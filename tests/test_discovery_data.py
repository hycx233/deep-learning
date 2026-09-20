from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_load_and_select_background(tmp_path: Path) -> None:
    windows, arrays = write_dataset(tmp_path)

    matrices, meta, intensity = load_discovery_dataset(windows, arrays)
    rows = select_background_rows(meta)
    train_rows, val_rows = split_background_rows(meta, rows)

    assert matrices.shape == (3, 128, 128)
    assert intensity.tolist() == [0.0, 1.0, 2.0]
    assert train_rows.tolist() == [0]
    assert val_rows.tolist() == [1]


def test_formal_input_requires_comparable_intensity(tmp_path: Path) -> None:
    windows, arrays = write_dataset(tmp_path)
    with np.load(arrays) as data:
        matrices = data["X"].copy()
    np.savez_compressed(arrays, X=matrices)

    with pytest.raises(DiscoveryDataError, match="强度分数"):
        load_discovery_dataset(windows, arrays)


def test_background_selection_requires_known_relation(tmp_path: Path) -> None:
    windows, arrays = write_dataset(tmp_path)
    meta = pd.read_csv(windows).drop(columns="known_overlap")

    with pytest.raises(DiscoveryDataError, match="known_overlap"):
        select_background_rows(meta)
