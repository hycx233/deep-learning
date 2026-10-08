#!/usr/bin/env python3
"""从共享缓存绘制演示用的双条件均值热图；完整重复与 O/E 对照仍见报告案例。"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from compare_conditions import ROOT, read_csv
from src.condition_differences import compare_structure
from src.preprocessing import extract_window, load_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/100bp")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/samples.csv")
    parser.add_argument("--structures", type=Path, default=ROOT / "data/structures.csv")
    parser.add_argument("--cases", type=Path, default=ROOT / "docs/results/differences/cases.csv")
    parser.add_argument("--structure-id", default="CHIN_126")
    parser.add_argument("--condition", choices=("DstpA", "DhnsDstpA"), default="DstpA")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs/results/differences/figures/CHIN_126_DstpA_slide.png")
    args = parser.parse_args()
    samples = read_csv(args.samples)
    structure = next((s for s in read_csv(args.structures) if s["ID"] == args.structure_id), None)
    row = next((r for r in read_csv(args.cases) if r["structure_id"] == args.structure_id
                and r["condition"] == args.condition), None)
    if structure is None or row is None:
        parser.error("结构或条件不在已保存的代表案例中，请检查 structure-id、condition 和输入表。")

    windows = {s["sample_id"]: extract_window(load_sample(args.cache_dir, s["sample_id"]),
                                             int(structure["center"])) for s in samples}
    # 核对输入仍对应保存的案例，避免拿不同版本缓存画图却沿用旧数值。
    result = next(r for r in compare_structure(structure, windows, samples)
                  if r["condition"] == args.condition)
    for key in ("wt_mean", "mutant_mean", "log2_fc", "valid_weight_fraction"):
        np.testing.assert_allclose(result[key], float(row[key]), rtol=1e-7, atol=1e-10,
                                   err_msg=f"缓存与已保存案例的 {key} 不一致")

    common = np.logical_and.reduce([w["intensity_mask"] for w in windows.values()])
    reference = windows[samples[0]["sample_id"]]
    values = []
    for condition in ("WT", args.condition):
        replicates = [windows[s["sample_id"]]["intensity"] for s in samples
                      if s["condition"] == condition]
        values.append(np.where(common, np.log1p(np.mean(replicates, axis=0)), np.nan))
    edges = reference["window_edges"] / 1000
    roi_start = (int(structure["start"]) - reference["start"]) % int(samples[0]["genome_length"]) / 1000
    roi_width = (int(structure["end"]) - int(structure["start"])) / 1000
    vmax = max(float(np.nanmax(v)) for v in values)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for axis, value, title in zip(axes, values, ("WT mean", f"{args.condition} mean")):
        plot = axis.pcolormesh(edges, edges, value, cmap="magma", vmin=0, vmax=vmax, shading="flat")
        axis.set(title=title, xlabel="Window offset (kb)", ylabel="Window offset (kb)", aspect="equal")
        axis.set_facecolor("#dddddd")
        axis.add_patch(Rectangle((roi_start, roi_start), roi_width, roi_width,
                                 fill=False, edgecolor="#56dfea", linewidth=1.6))
    figure.colorbar(plot, ax=axes, label="log1p(contacts per million)", shrink=0.88)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=160)
    plt.close(figure)
    print(f"已核对保存案例并绘图：{args.out}；log2FC={result['log2_fc']:.6f}")


if __name__ == "__main__":
    main()
