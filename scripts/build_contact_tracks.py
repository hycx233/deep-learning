"""从六个 Cooler 样本构建可比较的全基因组接触强度轨道。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tracks import (  # noqa: E402
    add_symmetric_upper_counts,
    aggregate_signal,
    normalize_per_sample,
)

DEFAULT_CONDITIONS = ("WT", "DstpA", "DhnsDstpA")


def code_version() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return f"{result.stdout.strip()}{' (dirty)' if dirty else ''}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建三条件全基因组接触强度轨道")
    parser.add_argument("--samples-csv", type=Path, default=Path("data/samples.csv"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/tracks/contact_signal")
    )
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--track-bin-bp", type=int, default=100)
    parser.add_argument("--scale", type=float, default=1_000_000.0)
    parser.add_argument("--pixel-chunk", type=int, default=5_000_000)
    return parser.parse_args()


def sample_track(
    path: Path,
    track_bin_bp: int,
    scale: float,
    pixel_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with h5py.File(path, "r") as handle:
        chrom_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["chroms/name"][:]
        ]
        if len(chrom_names) != 1:
            raise ValueError(f"{path} 应只有一条染色体，实际为 {chrom_names}")
        native_bin_bp = int(handle.attrs["bin-size"])
        genome_length = int(handle["chroms/length"][0])
        n_bins = int(handle["bins/start"].shape[0])
        n_pixels = int(handle["pixels/bin1_id"].shape[0])
        native_signal = np.zeros(n_bins, dtype=np.float64)
        for pixel_start in range(0, n_pixels, pixel_chunk):
            pixel_end = min(pixel_start + pixel_chunk, n_pixels)
            bin1 = handle["pixels/bin1_id"][pixel_start:pixel_end]
            bin2 = handle["pixels/bin2_id"][pixel_start:pixel_end]
            counts = handle["pixels/count"][pixel_start:pixel_end].astype(np.float64)
            add_symmetric_upper_counts(native_signal, bin1, bin2, counts)
            print(
                f"  pixels {pixel_end:,}/{n_pixels:,}", end="\r", flush=True
            )
        print()

    normalized_native, effective_total = normalize_per_sample(native_signal, scale)
    starts, ends, aggregated = aggregate_signal(
        normalized_native, native_bin_bp, track_bin_bp, genome_length
    )
    info = {
        "chrom": chrom_names[0],
        "genome_length": genome_length,
        "native_bin_bp": native_bin_bp,
        "track_bin_bp": track_bin_bp,
        "n_pixels": n_pixels,
        "effective_contact_total": effective_total,
        "normalized_sum": float(aggregated.sum()),
    }
    return starts, ends, aggregated, info


def main() -> None:
    args = parse_args()
    if args.track_bin_bp <= 0 or args.pixel_chunk <= 0 or args.scale <= 0:
        raise SystemExit("track-bin-bp、pixel-chunk 和 scale 必须为正数")
    if not args.samples_csv.is_file():
        raise SystemExit(f"找不到 {args.samples_csv}；等待 #1 合入或显式指定样本表")

    samples = pd.read_csv(args.samples_csv)
    required = {"sample_id", "condition", "replicate", "path"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise SystemExit(f"样本表缺少列 {missing}")
    samples = samples[samples["condition"].astype(str).isin(args.conditions)].copy()
    expected = {(condition, replicate) for condition in args.conditions for replicate in (1, 2)}
    actual = set(zip(samples["condition"].astype(str), samples["replicate"].astype(int)))
    if actual != expected:
        raise SystemExit(f"应有每个条件 rep1/rep2；缺少 {sorted(expected - actual)}，多出 {sorted(actual - expected)}")
    samples = samples.sort_values(["condition", "replicate"], kind="stable")

    tracks: list[np.ndarray] = []
    records: list[dict] = []
    reference_starts: np.ndarray | None = None
    reference_ends: np.ndarray | None = None
    started = time.time()
    for _, row in samples.iterrows():
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise SystemExit(f"找不到样本 {row['sample_id']}：{path}")
        print(f"处理 {row['sample_id']}：{path}")
        starts, ends, signal, info = sample_track(
            path, args.track_bin_bp, args.scale, args.pixel_chunk
        )
        if reference_starts is None:
            reference_starts, reference_ends = starts, ends
        elif not np.array_equal(starts, reference_starts) or not np.array_equal(ends, reference_ends):
            raise RuntimeError(f"样本 {row['sample_id']} 的 bin 边界与前面样本不一致")
        tracks.append(signal.astype(np.float32))
        records.append(
            {
                "sample_id": row["sample_id"],
                "condition": row["condition"],
                "replicate": int(row["replicate"]),
                "path": str(path),
                **info,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.out_dir / "samples_used.csv", index=False, encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "contact_tracks.npz",
        starts=reference_starts,
        ends=reference_ends,
        signals=np.stack(tracks),
        sample_id=np.asarray(samples["sample_id"].astype(str).tolist(), dtype=str),
        condition=np.asarray(samples["condition"].astype(str).tolist(), dtype=str),
        replicate=samples["replicate"].astype(np.int64).to_numpy(),
    )
    pd.DataFrame(records).to_csv(
        args.out_dir / "sample_summary.csv", index=False, encoding="utf-8"
    )
    run = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "code_version": code_version(),
        "samples_csv": str(args.samples_csv),
        "samples_manifest_snapshot": "samples_used.csv",
        "conditions": args.conditions,
        "track_bin_bp": args.track_bin_bp,
        "scale": args.scale,
        "pixel_chunk": args.pixel_chunk,
        "seconds": round(time.time() - started, 2),
        "normalization": (
            "symmetric-upper 非对角 counts 累加到两个端点、对角一次；"
            "每个重复除以自身有效接触总量后乘 scale，再按条件取均值"
        ),
        "outputs": {
            "tracks": "contact_tracks.npz",
            "sample_summary": "sample_summary.csv",
            "samples_manifest_snapshot": "samples_used.csv",
        },
    }
    (args.out_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"六样本轨道已写入 {args.out_dir}")


if __name__ == "__main__":
    main()
