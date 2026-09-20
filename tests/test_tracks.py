from pathlib import Path

import numpy as np
import pandas as pd

from scripts.plot_tracks import load_tracks, plot_interval
from src.tracks import (
    add_symmetric_upper_counts,
    aggregate_signal,
    assign_lanes,
    genome_segments,
    normalize_per_sample,
)


def test_genome_segments_include_short_final_interval() -> None:
    segments = genome_segments(4_641_652, 10_000)

    assert len(segments) == 465
    assert segments[0] == (0, 10_000)
    assert segments[-1] == (4_640_000, 4_641_652)


def test_symmetric_upper_counts_do_not_double_diagonal() -> None:
    signal = np.zeros(3, dtype=np.float64)
    bin1 = np.array([0, 0, 1], dtype=np.int64)
    bin2 = np.array([0, 1, 1], dtype=np.int64)
    counts = np.array([2.0, 3.0, 5.0])

    add_symmetric_upper_counts(signal, bin1, bin2, counts)

    assert signal.tolist() == [5.0, 8.0, 0.0]


def test_aggregate_and_normalize_signal() -> None:
    normalized, total = normalize_per_sample(np.array([1.0, 2.0, 3.0, 4.0]), scale=10.0)
    starts, ends, aggregated = aggregate_signal(normalized, 10, 20, 40)

    assert total == 10.0
    assert starts.tolist() == [0, 20]
    assert ends.tolist() == [20, 40]
    assert np.allclose(aggregated, [3.0, 7.0])


def test_assign_lanes_separates_overlaps() -> None:
    intervals = pd.DataFrame({"start": [0, 5, 10], "end": [8, 12, 15]})
    lanes = assign_lanes(intervals)

    assert lanes.tolist() == [0, 1, 0]


def test_plot_interval_creates_four_panel_figure(tmp_path: Path) -> None:
    starts = np.arange(0, 10_000, 100)
    ends = starts + 100
    x = np.linspace(0.0, 1.0, len(starts))
    tracks = {
        "starts": starts,
        "ends": ends,
        "signals": np.stack([x + index for index in range(6)]),
        "sample_id": np.array(["WT1", "WT2", "D1", "D2", "H1", "H2"]),
        "condition": np.array(["WT", "WT", "DstpA", "DstpA", "DhnsDstpA", "DhnsDstpA"]),
        "replicate": np.array([1, 2, 1, 2, 1, 2]),
    }
    genes = pd.DataFrame(
        {"start": [500, 1200], "end": [1000, 1700], "name": ["a", "b"], "strand": ["+", "-"]}
    )
    structures = pd.DataFrame(
        {"ID": ["CHIN_1", "OPCID_1"], "type": ["CHIN", "OPCID"], "start": [2000, 4000], "end": [2800, 4300]}
    )
    output = tmp_path / "tracks.png"

    plot_interval(
        tracks,
        genes,
        structures,
        ["WT", "DstpA", "DhnsDstpA"],
        0,
        10_000,
        output,
        sqrt_display=False,
        dpi=80,
    )

    assert output.is_file()
    assert output.stat().st_size > 10_000


def test_track_archive_uses_pickle_free_strings(tmp_path: Path) -> None:
    path = tmp_path / "tracks.npz"
    np.savez_compressed(
        path,
        starts=np.array([0]),
        ends=np.array([100]),
        signals=np.ones((1, 1), dtype=np.float32),
        sample_id=np.asarray(["WT_rep1"], dtype=str),
        condition=np.asarray(["WT"], dtype=str),
        replicate=np.array([1], dtype=np.int64),
    )

    loaded = load_tracks(path)

    assert loaded["sample_id"].tolist() == ["WT_rep1"]
