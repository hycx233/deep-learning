from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plot_tracks import load_tracks, plot_interval
from src.tracks import (
    aggregate_signal,
    assign_lanes,
    cached_marginal_track,
    genome_segments,
)


class TrackTests(unittest.TestCase):
    def _plotted_signals(self, valid, *, sqrt_display):
        signals = np.array([[1., 4.], [9., 16.], [4., 9.], [16., 25.], [9., 16.], [25., 36.]])
        tracks = {
            "starts": np.array([0, 100]), "ends": np.array([100, 200]),
            "signals": signals, "valid": valid,
            "sample_id": np.array(["WT1", "WT2", "D1", "D2", "H1", "H2"]),
            "condition": np.array(["WT", "WT", "DstpA", "DstpA", "DhnsDstpA", "DhnsDstpA"]),
            "replicate": np.array([1, 2, 1, 2, 1, 2]),
        }
        genes = pd.DataFrame(columns=["start", "end", "name", "strand"])
        structures = pd.DataFrame(columns=["ID", "type", "start", "end"])
        with tempfile.TemporaryDirectory() as directory, patch("scripts.plot_tracks.plt.close"):
            plot_interval(
                tracks, genes, structures, ["WT", "DstpA", "DhnsDstpA"],
                0, 200, Path(directory) / "tracks.png", sqrt_display=sqrt_display, dpi=60,
            )
            figure = plt.gcf()
            replicates = [line.get_ydata().copy() for line in figure.axes[0].lines[:6]]
            means = [line.get_ydata().copy() for line in figure.axes[1].lines]
        plt.close(figure)
        return replicates, means

    def test_condition_mean_is_linear_before_sqrt_display(self) -> None:
        replicates, means = self._plotted_signals(np.ones((6, 2), dtype=bool), sqrt_display=True)

        np.testing.assert_allclose(replicates[0], [1., 2.])
        np.testing.assert_allclose(means[0], np.sqrt([5., 10.]))
        np.testing.assert_allclose(means[1], np.sqrt([10., 17.]))
        np.testing.assert_allclose(means[2], np.sqrt([17., 26.]))

    def test_missing_replicate_masks_all_condition_means(self) -> None:
        valid = np.ones((6, 2), dtype=bool)
        valid[1, 0] = False
        replicates, means = self._plotted_signals(valid, sqrt_display=False)

        self.assertEqual(replicates[0][0], 1.)
        self.assertTrue(np.isnan(replicates[1][0]))
        self.assertEqual(replicates[2][0], 4.)
        self.assertTrue(all(np.isnan(mean[0]) for mean in means))
        np.testing.assert_allclose([mean[1] for mean in means], [10., 17., 26.])

    def test_genome_segments_include_short_final_interval(self) -> None:
        segments = genome_segments(4_641_652, 10_000)

        self.assertEqual(len(segments), 465)
        self.assertEqual(segments[0], (0, 10_000))
        self.assertEqual(segments[-1], (4_640_000, 4_641_652))

    def test_cached_marginal_uses_shared_depth_factor(self) -> None:
        state = {
            "metadata": {"bin_size": 100, "genome_length": 400},
            "marginal": np.array([1.0, 2.0, 3.0, 4.0]),
            "valid_bins": np.array([True, True, False, True]),
            "depth_factor": np.array(2.0),
        }

        starts, ends, signal, valid = cached_marginal_track(state, target_bin_bp=200)

        self.assertEqual(starts.tolist(), [0, 200])
        self.assertEqual(ends.tolist(), [200, 400])
        self.assertEqual(signal.tolist(), [6.0, 8.0])
        self.assertEqual(valid.tolist(), [True, True])

    def test_aggregate_signal_rejects_nonmultiple_resolution(self) -> None:
        with self.assertRaisesRegex(ValueError, "整数倍"):
            aggregate_signal(np.ones(4), 100, 150, 400)

    def test_assign_lanes_separates_overlaps(self) -> None:
        intervals = pd.DataFrame({"start": [0, 5, 10], "end": [8, 12, 15]})
        lanes = assign_lanes(intervals)

        self.assertEqual(lanes.tolist(), [0, 1, 0])

    def test_plot_interval_creates_four_panel_figure(self) -> None:
        starts = np.arange(0, 10_000, 100)
        ends = starts + 100
        x = np.linspace(0.0, 1.0, len(starts))
        tracks = {
            "starts": starts,
            "ends": ends,
            "signals": np.stack([x + index for index in range(6)]),
            "valid": np.ones((6, len(starts)), dtype=bool),
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
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tracks.png"
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
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)

    def test_track_archive_uses_pickle_free_strings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.npz"
            np.savez_compressed(
                path,
                starts=np.array([0]),
                ends=np.array([100]),
                signals=np.ones((1, 1), dtype=np.float32),
                valid=np.ones((1, 1), dtype=bool),
                sample_id=np.asarray(["WT_rep1"], dtype=str),
                condition=np.asarray(["WT"], dtype=str),
                replicate=np.array([1], dtype=np.int64),
            )
            loaded = load_tracks(path)

        self.assertEqual(loaded["sample_id"].tolist(), ["WT_rep1"])


if __name__ == "__main__":
    unittest.main()
