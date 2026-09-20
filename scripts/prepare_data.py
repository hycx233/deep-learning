#!/usr/bin/env python3
"""生成共享 100 bp 工作数据、无泄漏分组和真实 128×128 模型输入。"""

import argparse
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
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.preprocessing import prepare_sample, load_sample, extract_window
from src.window_splits import build_window_manifest

CLASS_NAMES = ("OPCID", "CHIN", "CHID")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_dataset(samples, positions, cache_dir, outdir):
    """保持 CSV 与 X/y 行序一致；背景标签 -1 不送入三分类入口。"""
    records, matrices, masks = [], [], []
    for sample in samples:
        state = load_sample(cache_dir, sample["sample_id"])
        for position in positions:
            window = extract_window(state, int(position["center"]))
            model = window["model"]
            if model.shape != (128, 128) or not np.isfinite(model).all():
                raise ValueError(f"{sample['sample_id']}/{position['position_id']}: 模型输入无效")
            if not np.allclose(model, model.T, atol=1e-6):
                raise ValueError("模型输入不对称")
            row = {
                "window_id": f"{sample['sample_id']}__{position['position_id']}",
                **position,
                "sample_id": sample["sample_id"], "condition": sample["condition"],
                "replicate": int(sample["replicate"]),
                "valid_pixel_fraction": float(window["mask"].mean()),
            }
            if isinstance(row["segments"], list):
                row["segments"] = json.dumps(row["segments"], separators=(",", ":"))
            records.append(row)
            matrices.append(model)
            masks.append(window["model_mask"])
        print(f"导出 {outdir.name}: {sample['sample_id']} × {len(positions)} 个位置", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(matrices, dtype=np.float32)
    y = np.array([CLASS_NAMES.index(r["type"]) if r["type"] in CLASS_NAMES else -1
                  for r in records], dtype=np.int64)
    np.savez_compressed(outdir / "windows.npz", X=x, y=y, mask=np.asarray(masks, dtype=bool))
    write_csv(outdir / "windows.csv", records)
    return records, x


def plot_examples(records, matrices, outpath):
    samples = list(dict.fromkeys(row["sample_id"] for row in records))
    positions = list(dict.fromkeys(row["position_id"] for row in records))
    figure, axes = plt.subplots(len(samples), len(positions), figsize=(11, 13),
                                squeeze=False, layout="constrained")
    for axis, row, values in zip(axes.flat, records, matrices):
        plot = axis.imshow(values, origin="lower", extent=(0, 24, 0, 24),
                           cmap="magma", vmin=0, vmax=1)
        label = row["structure_id"] or row["position_id"]
        axis.set_title(f"{row['sample_id']}\n{label}", fontsize=9)
        axis.set_xlabel("Offset within window (kb)", fontsize=8)
        axis.set_ylabel("Offset (kb)", fontsize=8)
        axis.tick_params(labelsize=8)
    figure.colorbar(plot, ax=axes, shrink=0.6,
                   label="clip(log1p(O/E), 0, log(11)) / log(11)")
    figure.suptitle("Shared normalization: six samples, three structures and circular-origin probe\n"
                   "Fixed scale; masked pixels filled with zero", fontsize=11)
    figure.savefig(outpath, dpi=120, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=REPO_ROOT / "data/samples.csv")
    parser.add_argument("--structures", type=Path, default=REPO_ROOT / "data/structures.csv")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/cache/100bp")
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "outputs/preprocessing/default")
    parser.add_argument("--stage", choices=("all", "cache", "windows"), default="all")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--background-count", type=int, default=24)
    args = parser.parse_args()
    started = time.monotonic()
    samples, structures = read_csv(args.samples), read_csv(args.structures)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    if args.stage in ("all", "cache"):
        for sample in samples:
            print(f"准备 100 bp 工作数据: {sample['sample_id']}", flush=True)
            summary[sample["sample_id"]] = prepare_sample(
                sample, args.data_root, args.cache_dir, chunksize=args.chunksize)
    if args.stage in ("all", "windows"):
        lengths = {int(sample["genome_length"]) for sample in samples}
        if len(lengths) != 1:
            raise ValueError("样本的基因组长度不一致")
        positions, split_summary = build_window_manifest(
            structures, genome_length=lengths.pop(), seed=args.seed,
            background_count=args.background_count)
        serializable = [{**p, "segments": json.dumps(p["segments"], separators=(",", ":"))
                         if isinstance(p["segments"], list) else p["segments"]} for p in positions]
        write_csv(args.outdir / "positions.csv", serializable)
        (args.outdir / "splits.json").write_text(
            json.dumps(split_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        known = [p for p in positions if p["type"] in CLASS_NAMES]
        background = [p for p in positions if p["type"] == "background"]
        wt = [s for s in samples if s["condition"] == "WT"]
        if not wt:
            raise ValueError("分类与背景交接需要 WT 样本")
        export_dataset(wt, known, args.cache_dir, args.outdir / "classification")
        if background:
            export_dataset(wt, background, args.cache_dir, args.outdir / "background")
        examples = [next(p for p in known if p["type"] == cls) for cls in CLASS_NAMES]
        length = int(samples[0]["genome_length"])
        # 首尾连接专用检查位置，不参与训练/验证/测试。
        examples.append({
            "position_id": "origin_probe", "structure_id": "", "type": "probe",
            "chrom": "NC_000913.3", "center": 0, "start": length - 12000,
            "end": length + 12000,
            "segments": [{"start": length - 12000, "end": length}, {"start": 0, "end": 12000}],
            "group_id": "probe", "split": "probe",
        })
        rows, images = export_dataset(samples, examples, args.cache_dir, args.outdir / "examples")
        plot_examples(rows, images, args.outdir / "normalization_examples.png")
        summary["splits"] = split_summary
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                           capture_output=True, text=True)
    code_files = [Path(__file__), REPO_ROOT / "src/preprocessing.py", REPO_ROOT / "src/window_splits.py"]
    run = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(["python", *sys.argv]), "stage": args.stage,
        "seed": args.seed, "git_revision": revision.stdout.strip() or None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "source_sha256": {p.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in code_files},
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (args.samples, args.structures)},
        "cache_dir": str(args.cache_dir), "outdir": str(args.outdir),
        "runtime_seconds": round(time.monotonic() - started, 2), "summary": summary,
    }
    (args.outdir / f"run_{args.stage}.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"完成 {args.stage}，耗时 {run['runtime_seconds']:.1f} s；输出 {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
