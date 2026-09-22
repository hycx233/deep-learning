"""从真实 WT rep1 生成少量开发窗口，验证任务二代码路径。

该脚本只做 ``log1p(raw counts)`` 和一次全局截断缩放，没有正式的测序深度归一化或
O/E 背景，因此输出只能用于烟雾验证，不能当作候选结果或报告指标。#2 合入后，正式
训练与扫描应直接使用共享预处理产出的 ``windows.csv + arrays.npz``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

CHROM = "NC_000913.3"
WINDOW_BP = 24_000
WORK_BIN_BP = 100
TARGET_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成任务二真实数据烟雾验证窗口")
    parser.add_argument(
        "--cool",
        type=Path,
        default=Path("data/GSE272159_37C_rep1.mapq_30.10.cool"),
    )
    parser.add_argument("--structures-csv", type=Path, default=Path("data/structures.csv"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/discovery/smoke_input")
    )
    parser.add_argument("--background-windows", type=int, default=20)
    parser.add_argument("--known-per-type", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260918)
    return parser.parse_args()


def read_cool_metadata(path: Path) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        chroms = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["chroms/name"][:]
        ]
        if chroms != [CHROM]:
            raise ValueError(f"当前烟雾脚本只支持 {CHROM}，实际为 {chroms}")
        return int(handle.attrs["bin-size"]), int(handle["chroms/length"][0])


def extract_window(
    handle: h5py.File,
    start: int,
    end: int,
    native_bin_bp: int,
) -> tuple[np.ndarray, float]:
    """局部读取 symmetric-upper 像素并聚合；仅供烟雾验证。"""
    start_bin = start // native_bin_bp
    end_bin = (end + native_bin_bp - 1) // native_bin_bp
    offsets = handle["indexes/bin1_offset"][start_bin : end_bin + 1]
    pixel_start, pixel_end = int(offsets[0]), int(offsets[-1])
    bin1 = handle["pixels/bin1_id"][pixel_start:pixel_end]
    bin2 = handle["pixels/bin2_id"][pixel_start:pixel_end]
    counts = handle["pixels/count"][pixel_start:pixel_end].astype(np.float64)
    keep = (bin2 >= start_bin) & (bin2 < end_bin)
    bin1 = bin1[keep]
    bin2 = bin2[keep]
    counts = counts[keep]

    factor = WORK_BIN_BP // native_bin_bp
    work_bins = int(np.ceil((end_bin - start_bin) / factor))
    matrix = np.zeros((work_bins, work_bins), dtype=np.float32)
    row = ((bin1 - start_bin) // factor).astype(np.int64)
    col = ((bin2 - start_bin) // factor).astype(np.int64)
    np.add.at(matrix, (row, col), counts)
    off_diagonal = row != col
    np.add.at(matrix, (col[off_diagonal], row[off_diagonal]), counts[off_diagonal])
    intensity = float(counts.sum())

    tensor = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0)
    resized = functional.interpolate(
        tensor, size=(TARGET_SIZE, TARGET_SIZE), mode="area"
    )[0, 0].numpy()
    return resized, intensity


def overlaps_any(start: int, end: int, structures: pd.DataFrame) -> bool:
    return bool(((structures["start"] < end) & (structures["end"] > start)).any())


def choose_background_starts(
    structures: pd.DataFrame,
    genome_length: int,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    candidates = np.arange(50_000, genome_length - WINDOW_BP - 50_000, 50_000)
    candidates = np.array(
        [
            int(start)
            for start in candidates
            if not overlaps_any(int(start), int(start + WINDOW_BP), structures)
        ],
        dtype=np.int64,
    )
    if len(candidates) < count:
        raise ValueError(f"只能找到 {len(candidates)} 个非重叠背景窗口，少于要求的 {count}")
    rng.shuffle(candidates)
    return sorted(candidates[:count].tolist())


def build_rows(
    structures: pd.DataFrame,
    genome_length: int,
    background_count: int,
    known_per_type: int,
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []
    background_starts = choose_background_starts(
        structures, genome_length, background_count, rng
    )
    val_count = max(2, round(background_count * 0.2))
    val_starts = set(background_starts[-val_count:])
    for start in background_starts:
        rows.append(
            {
                "window_id": f"smoke_bg_{len(rows):03d}",
                "chrom": CHROM,
                "start": start,
                "end": start + WINDOW_BP,
                "sample_id": "WT_rep1",
                "condition": "WT",
                "replicate": 1,
                "known_overlap": False,
                "structure_id": "",
                "type": "background",
                "split": "val" if start in val_starts else "train",
            }
        )

    for structure_type in ("OPCID", "CHIN", "CHID"):
        selected = structures[structures["type"] == structure_type].head(known_per_type)
        if len(selected) < known_per_type:
            raise ValueError(
                f"{structure_type} 只有 {len(selected)} 条，少于要求的 {known_per_type}"
            )
        for _, structure in selected.iterrows():
            center = int(structure["center"])
            start = max(0, min(center - WINDOW_BP // 2, genome_length - WINDOW_BP))
            rows.append(
                {
                    "window_id": f"smoke_known_{structure['ID']}",
                    "chrom": CHROM,
                    "start": start,
                    "end": start + WINDOW_BP,
                    "sample_id": "WT_rep1",
                    "condition": "WT",
                    "replicate": 1,
                    "known_overlap": True,
                    "structure_id": structure["ID"],
                    "type": structure_type,
                    "split": "test",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if not args.cool.is_file():
        raise SystemExit(f"找不到 WT rep1：{args.cool}")
    if not args.structures_csv.is_file():
        raise SystemExit(
            f"找不到 {args.structures_csv}；等待 #1 合入，或通过 --structures-csv 指定其产物"
        )
    if args.background_windows < 4 or args.known_per_type <= 0:
        raise SystemExit("background-windows 至少为 4，known-per-type 必须为正整数")

    structures = pd.read_csv(args.structures_csv)
    required = {"ID", "type", "start", "end", "center"}
    missing = sorted(required - set(structures.columns))
    if missing:
        raise SystemExit(f"结构表缺少列 {missing}")
    native_bin_bp, genome_length = read_cool_metadata(args.cool)
    if WORK_BIN_BP % native_bin_bp:
        raise SystemExit(
            f"工作分辨率 {WORK_BIN_BP} 不是原始分辨率 {native_bin_bp} 的整数倍"
        )

    rng = np.random.default_rng(args.seed)
    rows = build_rows(
        structures,
        genome_length,
        args.background_windows,
        args.known_per_type,
        rng,
    )
    raw_matrices: list[np.ndarray] = []
    intensity_scores: list[float] = []
    with h5py.File(args.cool, "r") as handle:
        for index, row in enumerate(rows, start=1):
            matrix, intensity = extract_window(
                handle, int(row["start"]), int(row["end"]), native_bin_bp
            )
            raw_matrices.append(matrix)
            intensity_scores.append(intensity)
            print(f"读取真实窗口 {index}/{len(rows)}：{row['window_id']}", flush=True)

    log_matrices = np.log1p(np.stack(raw_matrices)).astype(np.float32)
    scale = float(np.quantile(log_matrices, 0.995))
    if scale <= 0:
        raise RuntimeError("真实窗口全部为零，无法完成烟雾验证")
    matrices = np.clip(log_matrices / scale, 0.0, 1.0).astype(np.float32)

    meta = pd.DataFrame(rows)
    meta["normalization"] = "SMOKE_ONLY_log1p_global_p995_not_OE"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(args.out_dir / "windows.csv", index=False, encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "windows.npz",
        X=matrices,
        intensity_scores=np.asarray(intensity_scores, dtype=np.float64),
    )
    (args.out_dir / "README.txt").write_text(
        "这里是 WT rep1 真实局部 counts 生成的开发烟雾窗口。\n"
        "归一化仅为 log1p + 全部窗口统一 p99.5 截断，不含正式深度归一化或 O/E。\n"
        "这些分数、候选和召回不得写入报告；#2 合入后必须改用共享预处理产物。\n",
        encoding="utf-8",
    )
    print(f"真实烟雾窗口 {len(meta)} 个，已写入 {args.out_dir}")


if __name__ == "__main__":
    main()
