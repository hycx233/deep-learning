"""复用 #2 的六样本缓存，构建可比较的全基因组接触强度轨道。"""

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

from src.preprocessing import load_sample  # noqa: E402
from src.tracks import cached_marginal_track  # noqa: E402

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
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/100bp"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/tracks/contact_signal")
    )
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--track-bin-bp", type=int, default=100)
    return parser.parse_args()


def sample_track(
    cache_dir: Path,
    sample_id: str,
    track_bin_bp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    state = load_sample(cache_dir, sample_id)
    metadata = state["metadata"]
    starts, ends, aggregated, valid = cached_marginal_track(state, track_bin_bp)
    info = {
        "chrom": metadata["chrom"],
        "genome_length": int(metadata["genome_length"]),
        "native_bin_bp": int(metadata["bin_size"]),
        "track_bin_bp": track_bin_bp,
        "n_pixels": int(metadata["n_pixels"]),
        "total_unique_counts": int(metadata["total_counts"]),
        "depth_factor": float(metadata["depth_factor"]),
        "valid_bin_count": int(metadata["valid_bin_count"]),
        "normalized_sum": float(aggregated.sum()),
    }
    return starts, ends, aggregated, valid, info


def main() -> None:
    args = parse_args()
    if args.track_bin_bp <= 0:
        raise SystemExit("track-bin-bp 必须为正数")
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
    validity: list[np.ndarray] = []
    records: list[dict] = []
    reference_starts: np.ndarray | None = None
    reference_ends: np.ndarray | None = None
    started = time.time()
    for _, row in samples.iterrows():
        sample_id = str(row["sample_id"])
        print(f"处理 {sample_id}：{args.cache_dir}")
        starts, ends, signal, valid, info = sample_track(
            args.cache_dir, sample_id, args.track_bin_bp
        )
        if reference_starts is None:
            reference_starts, reference_ends = starts, ends
        elif not np.array_equal(starts, reference_starts) or not np.array_equal(ends, reference_ends):
            raise RuntimeError(f"样本 {row['sample_id']} 的 bin 边界与前面样本不一致")
        tracks.append(signal.astype(np.float32))
        validity.append(valid)
        records.append(
            {
                "sample_id": row["sample_id"],
                "condition": row["condition"],
                "replicate": int(row["replicate"]),
                "cache_dir": str(args.cache_dir),
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
        valid=np.stack(validity),
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
        "cache_dir": str(args.cache_dir),
        "samples_manifest_snapshot": "samples_used.csv",
        "conditions": args.conditions,
        "track_bin_bp": args.track_bin_bp,
        "seconds": round(time.time() - started, 2),
        "normalization": (
            "复用 #2 缓存 marginal（对角一次、非对角每端点一次）；"
            "每个重复乘以 1,000,000 / 上三角唯一 counts 总量，再按条件取均值"
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
