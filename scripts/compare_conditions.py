#!/usr/bin/env python3
"""用六样本共同有效区域比较已知结构强度，导出重复对照与代表热图。"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import load_sample, extract_window
from src.condition_differences import compare_structure

CONDITIONS = ("DstpA", "DhnsDstpA")
LABEL_TEXT = {
    "increase_consistent": "两个重复均高于本结构的描述性参考带。",
    "decrease_consistent": "两个重复均低于本结构的描述性参考带。",
    "within_reference": "两个重复均位于描述性参考带内。",
    "inconsistent_or_borderline": "两个重复的方向或变化幅度未同时满足一致变化规则。",
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def choose_cases(rows, count):
    eligible = [r for r in rows if r["status"] == "ok" and r["valid_weight_fraction"] >= 0.8]
    chosen, ids = [], set()
    # 每种条件先展示增强、减弱、参考带内和边界情形；没有某类时不强行补造。
    for condition in CONDITIONS:
        for label in LABEL_TEXT:
            candidates = [r for r in eligible if r["condition"] == condition
                          and r["effect_label"] == label and r["structure_id"] not in ids]
            candidates.sort(key=lambda r: (
                abs(r["log2_fc"]) if label == "within_reference" else -abs(r["log2_fc"]),
                r["structure_id"],
            ))
            if candidates and len(chosen) < count:
                chosen.append(candidates[0])
                ids.add(candidates[0]["structure_id"])
    remaining = sorted(eligible, key=lambda r: (-abs(r["log2_fc"]), r["structure_id"], r["condition"]))
    for row in remaining:
        if len(chosen) == count:
            break
        if row["structure_id"] not in ids:
            chosen.append(row)
            ids.add(row["structure_id"])
    return chosen


def draw_case(structure, row, windows, samples, path):
    sample_ids = {
        (s["condition"], int(s["replicate"])): s["sample_id"] for s in samples
    }
    condition = row["condition"]
    ids = [sample_ids[c, rep] for c in ("WT", condition) for rep in (1, 2)]
    common = np.logical_and.reduce([w["intensity_mask"] for w in windows.values()])
    morphology_mask = np.logical_and.reduce([w["mask"] for w in windows.values()])
    reference = windows[ids[0]]
    edges = reference["window_edges"] / 1000
    intensity = [windows[key]["intensity"] for key in ids]
    means = [(intensity[0] + intensity[1]) / 2, (intensity[2] + intensity[3]) / 2]
    raw_display = [np.where(common, np.log1p(values), np.nan) for values in intensity + means]
    vmax = max(float(np.nanmax(v)) for v in raw_display)
    oe_means = [(windows[ids[i]]["oe"] + windows[ids[i+1]]["oe"]) / 2 for i in (0, 2)]
    oe_display = [np.where(morphology_mask, np.log1p(np.minimum(v, 10)) / np.log(11), np.nan)
                  for v in oe_means]
    figure = plt.figure(figsize=(14, 8), layout="constrained")
    grid = figure.add_gridspec(3, 4, height_ratios=[1, 1, 0.08])
    axes = np.array([[figure.add_subplot(grid[i, j]) for j in range(4)] for i in range(2)])
    slots = [*axes[0], axes[1, 0], axes[1, 1]]
    titles = [*ids, "WT mean intensity", f"{condition} mean intensity"]
    chrom_length = int(samples[0]["genome_length"])
    roi_start = (int(structure["start"]) - reference["start"]) % chrom_length / 1000
    roi_width = (int(structure["end"]) - int(structure["start"])) / 1000
    for axis, values, title in zip(slots, raw_display, titles):
        plot = axis.pcolormesh(edges, edges, values, cmap="magma", vmin=0, vmax=vmax, shading="flat")
        axis.set_title(title, fontsize=10)
    for axis, values, title in zip(axes[1, 2:], oe_display, ("WT mean O/E", f"{condition} mean O/E")):
        oe_plot = axis.pcolormesh(edges, edges, values, cmap="magma", vmin=0, vmax=1, shading="flat")
        axis.set_title(title, fontsize=10)
    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_facecolor("#dddddd")
        axis.add_patch(Rectangle((roi_start, roi_start), roi_width, roi_width,
                                 fill=False, edgecolor="#56dfea", linewidth=1))
        axis.set_xlabel("Window offset (kb)", fontsize=8)
        axis.set_ylabel("Window offset (kb)", fontsize=8)
        axis.tick_params(labelsize=8)
    figure.colorbar(plot, cax=figure.add_subplot(grid[2, :2]), orientation="horizontal",
                   label="Intensity: log1p(contacts per million), shared scale")
    figure.colorbar(oe_plot, cax=figure.add_subplot(grid[2, 2:]), orientation="horizontal",
                   label="O/E: log1p(min(O/E, 10)) / log(11)")
    figure.suptitle(
        f"{structure['ID']} | {condition} vs WT | log2FC={row['log2_fc']:.3f}\n"
        f"{row['effect_label']} | cyan: quantified interval | grey: shared invalid pixels", fontsize=12)
    figure.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / "data/samples.csv")
    parser.add_argument("--structures", type=Path, default=ROOT / "data/structures.csv")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/100bp")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/differences/known_structures")
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--min-fold-change", type=float, default=1.25)
    args = parser.parse_args()
    if not 1 <= args.case_count <= 10 or args.min_fold_change <= 1:
        parser.error("case-count 需为1—10，min-fold-change需大于1")
    started = time.monotonic()
    samples, structures = read_csv(args.samples), read_csv(args.structures)
    states = {s["sample_id"]: load_sample(args.cache_dir, s["sample_id"]) for s in samples}
    args.outdir.mkdir(parents=True, exist_ok=True)
    figures = args.outdir / "figures"
    figures.mkdir(exist_ok=True)
    results = []
    for index, structure in enumerate(structures, 1):
        windows = {key: extract_window(state, int(structure["center"])) for key, state in states.items()}
        results.extend(compare_structure(structure, windows, samples, args.min_fold_change))
        if index % 50 == 0 or index == len(structures):
            print(f"已比较 {index}/{len(structures)} 个结构", flush=True)
    write_csv(args.outdir / "differences.csv", results)
    selected = choose_cases(results, args.case_count)
    lookup = {s["ID"]: s for s in structures}
    cases = []
    for row in selected:
        structure = lookup[row["structure_id"]]
        windows = {key: extract_window(state, int(structure["center"])) for key, state in states.items()}
        filename = f"{row['structure_id']}_{row['condition']}.png"
        draw_case(structure, row, windows, samples, figures / filename)
        cases.append({**row, "figure": f"figures/{filename}", "note": LABEL_TEXT[row["effect_label"]]})
    if cases:
        write_csv(args.outdir / "cases.csv", cases)
    summary = {
        "known_structures": len(structures), "comparisons": len(results),
        "status_counts": dict(Counter(r["status"] for r in results)),
        "conditions": {c: dict(Counter(r["effect_label"] for r in results if r["condition"] == c))
                       for c in CONDITIONS},
        "minimum_valid_weight_fraction": min(r["valid_weight_fraction"] for r in results),
        "structures_below_80pct_valid_weight": len({r["structure_id"] for r in results
                                                   if r["valid_weight_fraction"] < 0.8}),
        "comparisons_below_80pct_valid_weight": sum(r["valid_weight_fraction"] < 0.8 for r in results),
        "min_fold_change": args.min_fold_change, "representative_cases": len(cases),
        "candidate_analysis": "等待发现/聚类PR修订，仅报告已知结构",
        "interpretation": "描述性条件差异；标签并非显著性检验，不作单因子因果归因",
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    sources = [Path(__file__), ROOT / "src/condition_differences.py", ROOT / "src/preprocessing.py"]
    run = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(["python", *sys.argv]), "device": "CPU", "randomness": "none",
        "git_revision": revision.stdout.strip() or None, "git_dirty": bool(dirty.stdout.strip()),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.samples, args.structures)},
        "cache_total_counts": {key: state["metadata"]["total_counts"] for key, state in states.items()},
        "parameters": {"window_bp": 24000, "bin_size_bp": 100, "case_count": args.case_count,
                       "min_fold_change": args.min_fold_change, "pseudocount": 0},
        "runtime_seconds": round(time.monotonic() - started, 2), "outdir": str(args.outdir),
    }
    (args.outdir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"完成，耗时 {run['runtime_seconds']} s；结果位于 {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
