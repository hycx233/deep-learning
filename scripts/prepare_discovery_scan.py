"""从共享 100 bp 缓存生成 WT rep1 的 1 kb 步长全基因组扫描窗口。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import extract_window, load_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成新结构发现的全基因组扫描输入")
    parser.add_argument("--sample-id", default="WT_rep1")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/100bp"))
    parser.add_argument("--structures-csv", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/discovery/scan_1kb"))
    parser.add_argument("--step-bp", type=int, default=1_000)
    parser.add_argument("--window-bp", type=int, default=24_000)
    parser.add_argument("--image-size", type=int, default=128)
    return parser.parse_args()


def code_version() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def overlaps_segments(segments: list[dict[str, int]], start: int, end: int) -> bool:
    """半开区间重叠判定；扫描窗口跨环状起点时 segments 含两段。"""
    return any(max(int(segment["start"]), start) < min(int(segment["end"]), end) for segment in segments)


def intensity_score(window: dict) -> float:
    """深度归一化 counts 的有效非对角上三角均值，不做逐窗口缩放。"""
    values = np.asarray(window["intensity"], dtype=np.float64)
    valid = np.asarray(window["intensity_mask"], dtype=bool)
    selected = np.triu(valid, k=1)
    if not selected.any():
        return 0.0
    return float(values[selected].mean())


def main() -> None:
    args = parse_args()
    if args.step_bp <= 0 or args.window_bp <= 0 or args.image_size <= 0:
        raise SystemExit("step-bp、window-bp 和 image-size 必须为正整数")
    if not args.structures_csv.is_file():
        raise SystemExit(f"找不到结构表：{args.structures_csv}")

    state = load_sample(args.cache_dir, args.sample_id)
    metadata = state["metadata"]
    genome_length = int(metadata["genome_length"])
    structures = pd.read_csv(args.structures_csv)
    required = {"ID", "type", "chrom", "start", "end"}
    missing = sorted(required - set(structures.columns))
    if missing:
        raise SystemExit(f"结构表缺少列 {missing}")
    structures = structures.sort_values(["start", "end", "ID"], kind="stable").reset_index(drop=True)

    centers = np.arange(0, genome_length, args.step_bp, dtype=np.int64)
    matrices = np.empty((len(centers), args.image_size, args.image_size), dtype=np.float32)
    masks = np.empty_like(matrices, dtype=bool)
    intensity_scores = np.empty(len(centers), dtype=np.float64)
    rows: list[dict] = []
    started = time.time()

    for index, center in enumerate(centers):
        window = extract_window(
            state, int(center), window_bp=args.window_bp, image_size=args.image_size
        )
        matrices[index] = window["model"]
        masks[index] = window["model_mask"]
        intensity_scores[index] = intensity_score(window)
        related = structures[
            structures.apply(
                lambda row: overlaps_segments(window["segments"], int(row["start"]), int(row["end"])),
                axis=1,
            )
        ]
        structure_ids = related["ID"].astype(str).tolist()
        known_types = sorted(set(related["type"].astype(str)))
        rows.append(
            {
                "window_id": f"{args.sample_id}__scan_{int(center):07d}",
                "chrom": metadata["chrom"],
                "start": int(window["start"]),
                "end": int(window["end"]),
                "center": int(center),
                "length_bp": args.window_bp,
                "segments": json.dumps(window["segments"], separators=(",", ":")),
                "sample_id": args.sample_id,
                "condition": metadata["condition"],
                "replicate": int(metadata["replicate"]),
                "known_overlap": bool(structure_ids),
                "known_structure_ids": ";".join(structure_ids),
                "known_types": ";".join(known_types),
                "valid_pixel_fraction": float(masks[index].mean()),
                "intensity_score": intensity_scores[index],
            }
        )
        if (index + 1) % 250 == 0 or index + 1 == len(centers):
            print(f"[{index + 1}/{len(centers)}] center={int(center):,}", flush=True)

    if not np.isfinite(matrices).all() or not np.isfinite(intensity_scores).all():
        raise RuntimeError("扫描输出含 NaN/Inf")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "windows.csv", index=False, encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "windows.npz",
        X=matrices,
        mask=masks,
        intensity_scores=intensity_scores,
    )
    run = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "code_version": code_version(),
        "sample_id": args.sample_id,
        "cache_dir": str(args.cache_dir),
        "structures_csv": str(args.structures_csv),
        "genome_length": genome_length,
        "step_bp": args.step_bp,
        "window_bp": args.window_bp,
        "image_size": args.image_size,
        "windows": int(len(table)),
        "known_overlap_windows": int(table["known_overlap"].sum()),
        "seconds": round(time.time() - started, 2),
        "intensity_rule": "mean upper-triangle off-diagonal depth-normalized counts over valid pixels",
    }
    (args.out_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成 {len(table)} 个扫描窗口；输出 {args.out_dir}")


if __name__ == "__main__":
    main()
