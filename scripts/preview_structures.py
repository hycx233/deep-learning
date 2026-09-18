#!/usr/bin/env python3
"""导出 WT 两个重复中的三类真实窗口，供接口交接和人工查看。"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import cooler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IDS = ("WT_rep1", "WT_rep2")
STRUCTURE_IDS = ("OPCID_1", "CHIN_2", "CHID_1")
WINDOW_BP = 24_000
BIN_BP = 100


def read_rows(path, key):
    with path.open(encoding="utf-8", newline="") as source:
        return {row[key]: row for row in csv.DictReader(source)}


def read_count_window(sample, structure, data_root):
    """仅取局部原始上三角计数，聚合至 100 bp 后再补对称。"""
    matrix_path = data_root / sample["path"]
    contact_map = cooler.Cooler(str(matrix_path))
    if contact_map.storage_mode != "symmetric-upper":
        raise ValueError(f"{matrix_path}: 本交接样例要求 symmetric-upper 存储")
    source_bin = contact_map.binsize
    if source_bin is None or BIN_BP % source_bin:
        raise ValueError(f"{matrix_path}: 分辨率 {source_bin} 无法整合为 {BIN_BP} bp")
    if sample["chrom"] != structure["chrom"]:
        raise ValueError(f"{sample['sample_id']}/{structure['ID']}: 标准染色体名称不一致")
    chrom = sample["source_chrom"]
    genome_length = int(contact_map.chromsizes[chrom])
    # 将左边界对齐到 100 bp，使每格都有明确的基因组区间。
    start = (int(structure["center"]) - WINDOW_BP // 2) // BIN_BP * BIN_BP
    end = start + WINDOW_BP
    if start < 0 or end > genome_length:
        raise ValueError(f"{structure['ID']}: 窗口 [{start}, {end}) 越界；环状切窗由正式预处理实现")
    if not start <= int(structure["start"]) < int(structure["end"]) <= end:
        raise ValueError(f"{structure['ID']}: 标注未完整落在本次 24 kb 窗口内")
    local = contact_map.matrix(balance=False, sparse=True).fetch((chrom, start, end)).tocoo()
    expected_bins = WINDOW_BP // source_bin
    if local.shape != (expected_bins, expected_bins):
        raise ValueError(f"{matrix_path}: 局部形状 {local.shape} 与窗口范围不一致")
    # fetch 已补对称；先取原分辨率上三角，避免粗粒度对角格的计数翻倍。
    keep = local.row <= local.col
    factor = BIN_BP // source_bin
    coarse_upper = np.zeros((WINDOW_BP // BIN_BP, WINDOW_BP // BIN_BP), dtype=np.int64)
    np.add.at(coarse_upper, (local.row[keep] // factor, local.col[keep] // factor), local.data[keep])
    coarse = coarse_upper + coarse_upper.T
    np.fill_diagonal(coarse, np.diag(coarse_upper))
    source_unique_count = int(local.data[keep].sum(dtype=np.int64))
    assert int(np.triu(coarse).sum()) == source_unique_count, "聚合前后上三角接触总数不一致"
    assert np.array_equal(coarse, coarse.T), "聚合结果不对称"
    return coarse, start, end, source_unique_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=REPO_ROOT / "data/samples.csv")
    parser.add_argument("--structures", type=Path, default=REPO_ROOT / "data/structures.csv")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT, help="样本清单中相对路径的基准目录")
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "outputs/data_preparation")
    args = parser.parse_args()
    started = time.monotonic()
    samples = read_rows(args.samples, "sample_id")
    structures = read_rows(args.structures, "ID")
    args.outdir.mkdir(parents=True, exist_ok=True)
    positions = []
    windows = []
    for sample_id in SAMPLE_IDS:
        sample = samples[sample_id]
        for structure_id in STRUCTURE_IDS:
            structure = structures[structure_id]
            counts, start, end, unique_count = read_count_window(sample, structure, args.data_root)
            filename = f"{sample_id}_{structure_id}_raw_100bp.npz"
            np.savez_compressed(
                args.outdir / filename, counts=counts, sample_id=sample_id,
                structure_id=structure_id, chrom=structure["chrom"], start=start, end=end,
                bin_size_bp=BIN_BP,
            )
            positions.append({
                "sample_id": sample_id, "condition": sample["condition"],
                "replicate": sample["replicate"], "structure_id": structure_id,
                "type": structure["type"], "chrom": structure["chrom"],
                "start": start, "end": end, "bin_size_bp": BIN_BP,
                "structure_start": structure["start"], "structure_end": structure["end"],
                "structure_center": structure["center"], "unique_contact_count": unique_count,
                "matrix_file": filename,
            })
            windows.append(counts)

    with (args.outdir / "positions.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(positions[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(positions)

    log_windows = [np.log1p(counts) for counts in windows]
    vmax = max(float(window.max()) for window in log_windows)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained")
    for axis, values, position in zip(axes.flat, log_windows, positions):
        extent = [position["start"] / 1000, position["end"] / 1000] * 2
        plot = axis.imshow(values, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
        mark_start = int(position["structure_start"]) / 1000
        mark_width = (int(position["structure_end"]) - int(position["structure_start"])) / 1000
        axis.add_patch(Rectangle((mark_start, mark_start), mark_width, mark_width,
                                 fill=False, edgecolor="#5ad9f5", linewidth=1.2))
        axis.set_title(f"{position['sample_id']} / {position['structure_id']}", fontsize=11)
        axis.set_xlabel("Genome position (kb)")
        axis.set_ylabel("Genome position (kb)")
    figure.colorbar(plot, ax=axes, shrink=0.85, label="log1p(raw contact counts), shared scale")
    figure.suptitle("Real WT windows at 100 bp - cyan box: annotated interval\n"
                    "Preview only: no depth normalization or observed/expected correction", fontsize=12)
    image_path = args.outdir / "structure_previews.png"
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    run = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(["python", *sys.argv]),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "sample_ids": list(SAMPLE_IDS), "structure_ids": list(STRUCTURE_IDS),
        "window_bp": WINDOW_BP, "bin_size_bp": BIN_BP,
        "coordinates": "0-based, half-open; window start aligned to 100 bp",
        "aggregation": "select original upper triangle, sum into 100 bp bins, mirror off-diagonal",
        "display": "log1p(raw counts), shared vmin=0 and vmax across six panels",
        "display_vmax": vmax,
        "purpose": "真实输入交接样例；未经测序深度归一化或 O/E，不用于正式训练或跨条件结论",
        "versions": {"python": sys.version.split()[0], "cooler": cooler.__version__,
                     "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "runtime_seconds": round(time.monotonic() - started, 3),
        "outputs": ["positions.csv", image_path.name, *[row["matrix_file"] for row in positions]],
    }
    (args.outdir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存 6 个 240×240 原始 count 窗口，耗时 {run['runtime_seconds']:.2f} s")
    print(f"热图: {image_path.resolve()}")


if __name__ == "__main__":
    main()
