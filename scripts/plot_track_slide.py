#!/usr/bin/env python3
"""从保存的线性轨道提取一个条件均值面板，供演示使用。"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_tracks import CONDITION_COLORS, DEFAULT_CONDITIONS, load_tracks
from src.tracks import condition_mean_signals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-npz", type=Path,
                        default=Path("outputs/tracks/contact_signal/contact_tracks.npz"))
    parser.add_argument("--start", type=int, default=1210000)
    parser.add_argument("--end", type=int, default=1220000)
    parser.add_argument("--output", type=Path,
                        default=Path("docs/results/tracks/tracks_1210000_1220000_slide.png"))
    args = parser.parse_args()
    tracks = load_tracks(args.tracks_npz)
    if not 0 <= args.start < args.end <= int(tracks["ends"][-1]):
        parser.error("区间须位于基因组范围内且 start < end")
    selected = np.isin(tracks["condition"], DEFAULT_CONDITIONS)
    common_valid = np.all(tracks["valid"][selected] &
                          np.isfinite(tracks["signals"][selected]), axis=0)
    comparable = np.where(common_valid, tracks["signals"], np.nan)
    means = condition_mean_signals(comparable, tracks["condition"], DEFAULT_CONDITIONS)
    keep = (tracks["starts"] < args.end) & (tracks["ends"] > args.start)
    x = (tracks["starts"][keep] + tracks["ends"][keep]) / 2000.0
    names = {"WT": "WT", "DstpA": r"$\Delta stpA$", "DhnsDstpA": r"$\Delta hns\Delta stpA$"}
    with plt.rc_context({"font.size": 13}):
        figure, axis = plt.subplots(figsize=(10, 3.5), layout="constrained")
        for condition in DEFAULT_CONDITIONS:
            axis.plot(x, np.sqrt(means[condition][keep]), color=CONDITION_COLORS[condition],
                      linewidth=2, label=names[condition])
        axis.set(xlabel="Genomic coordinate (kb)", ylabel="Contact signal (sqrt)",
                 xlim=(args.start / 1000, args.end / 1000))
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=200)
        plt.close(figure)
    print(f"已提取 [{args.start}, {args.end}) 条件均值面板：{args.output}")


if __name__ == "__main__":
    main()
