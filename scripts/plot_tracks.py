"""绘制三条件四面板接触强度、基因和结构轨道图。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tracks import assign_lanes, condition_mean_signals, genome_segments  # noqa: E402

DEFAULT_CONDITIONS = ("WT", "DstpA", "DhnsDstpA")
CONDITION_COLORS = {
    "WT": "#2b6cb0",
    "DstpA": "#d97706",
    "DhnsDstpA": "#b91c1c",
}


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
    parser = argparse.ArgumentParser(description="绘制三条件四面板全基因组轨道")
    parser.add_argument(
        "--tracks-npz",
        type=Path,
        default=Path("outputs/tracks/contact_signal/contact_tracks.npz"),
    )
    parser.add_argument("--genes-csv", type=Path, default=Path("data/genes.csv"))
    parser.add_argument("--structures-csv", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tracks/figures"))
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--segment-bp", type=int, default=10_000)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--sqrt-display", action="store_true")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def load_tracks(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到轨道数组：{path}")
    with np.load(path) as arrays:
        required = {"starts", "ends", "signals", "valid", "sample_id", "condition", "replicate"}
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"轨道数组缺少键 {missing}")
        result = {name: np.asarray(arrays[name]) for name in required}
    if result["signals"].ndim != 2:
        raise ValueError(f"signals 应为 samples×bins，实际为 {result['signals'].shape}")
    if result["signals"].shape[0] != len(result["sample_id"]):
        raise ValueError("signals 样本数与 sample_id 不一致")
    if result["signals"].shape[1] != len(result["starts"]):
        raise ValueError("signals bin 数与 starts 不一致")
    if result["valid"].shape != result["signals"].shape:
        raise ValueError("valid 形状必须与 signals 一致")
    return result


def visible_rows(table: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return table[(table["start"] < end) & (table["end"] > start)].copy()


def plot_interval(
    tracks: dict[str, np.ndarray],
    genes: pd.DataFrame,
    structures: pd.DataFrame,
    conditions: list[str],
    start: int,
    end: int,
    out_path: Path,
    *,
    sqrt_display: bool,
    dpi: int,
) -> None:
    starts = tracks["starts"].astype(np.int64)
    ends = tracks["ends"].astype(np.int64)
    signals = tracks["signals"].astype(np.float64)
    valid = tracks["valid"].astype(bool)
    sample_ids = tracks["sample_id"].astype(str)
    sample_conditions = tracks["condition"].astype(str)
    replicates = tracks["replicate"].astype(int)
    bin_mask = (starts < end) & (ends > start)
    if not bin_mask.any():
        raise ValueError(f"区间 [{start}, {end}) 没有轨道 bin")
    x = (starts[bin_mask] + ends[bin_mask]) / 2.0
    display_signals = np.sqrt(signals) if sqrt_display else signals.copy()
    display_signals = np.where(valid, display_signals, np.nan)
    means = condition_mean_signals(display_signals, sample_conditions, conditions)
    baseline = float(np.nanmedian(display_signals))

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.2, 1.2, 1.2], "hspace": 0.12},
    )
    replicate_axis, mean_axis, gene_axis, structure_axis = axes
    for index, (sample_id, condition, replicate) in enumerate(
        zip(sample_ids, sample_conditions, replicates)
    ):
        if condition not in conditions:
            continue
        replicate_axis.plot(
            x,
            display_signals[index, bin_mask],
            color=CONDITION_COLORS.get(condition, None),
            alpha=0.55,
            linewidth=0.9,
            label=f"{condition} rep{replicate}",
        )
    replicate_axis.axhline(
        baseline, color="#777777", linestyle="--", linewidth=1.0, label="genome median"
    )
    replicate_axis.set_ylabel("contact signal")
    replicate_axis.set_title(
        f"Replicates: {start:,}–{end:,} bp"
        + (" (sqrt display)" if sqrt_display else "")
    )
    replicate_axis.legend(ncol=4, fontsize=8, loc="upper right")

    for condition in conditions:
        mean_axis.plot(
            x,
            means[condition][bin_mask],
            color=CONDITION_COLORS.get(condition, None),
            linewidth=1.5,
            label=f"{condition} mean",
        )
    mean_axis.set_ylabel("condition mean")
    mean_axis.legend(ncol=len(conditions), fontsize=8, loc="upper right")

    visible_genes = visible_rows(genes, start, end).reset_index(drop=True)
    visible_genes["lane"] = assign_lanes(visible_genes)
    for _, gene in visible_genes.iterrows():
        left = max(start, int(gene["start"]))
        right = min(end, int(gene["end"]))
        lane = int(gene["lane"])
        gene_axis.add_patch(
            Rectangle((left, lane - 0.28), max(1, right - left), 0.56, color="#2563eb", alpha=0.75)
        )
        if right - left >= 120:
            direction = str(gene.get("strand", "+"))
            arrow_start, arrow_end = (left, right) if direction != "-" else (right, left)
            gene_axis.annotate(
                "",
                xy=(arrow_end, lane),
                xytext=(arrow_start, lane),
                arrowprops={"arrowstyle": "->", "color": "#1e3a8a", "lw": 0.8},
            )
        gene_axis.text(
            (left + right) / 2,
            lane + 0.34,
            str(gene.get("name", gene.get("gene_id", ""))),
            ha="center",
            va="bottom",
            fontsize=6,
            clip_on=True,
        )
    gene_axis.set_ylabel("genes")
    gene_axis.set_yticks([])
    gene_axis.set_ylim(-0.7, max(1.0, (visible_genes["lane"].max() + 1.2) if len(visible_genes) else 1.0))

    visible_structures = visible_rows(
        structures[structures["type"].isin(["CHIN", "OPCID"])], start, end
    ).reset_index(drop=True)
    visible_structures["lane"] = assign_lanes(visible_structures)
    for _, structure in visible_structures.iterrows():
        left = max(start, int(structure["start"]))
        right = min(end, int(structure["end"]))
        lane = int(structure["lane"])
        structure_axis.add_patch(
            Rectangle((left, lane - 0.28), max(1, right - left), 0.56, color="#f97316", alpha=0.8)
        )
        structure_axis.text(
            (left + right) / 2,
            lane + 0.34,
            f"{structure['type']}:{structure['ID']}",
            ha="center",
            va="bottom",
            fontsize=6,
            clip_on=True,
        )
    structure_axis.set_ylabel("CHIN / OPCID")
    structure_axis.set_yticks([])
    structure_axis.set_ylim(
        -0.7,
        max(1.0, (visible_structures["lane"].max() + 1.2) if len(visible_structures) else 1.0),
    )
    structure_axis.set_xlabel("genomic coordinate (bp)")

    for axis in axes:
        axis.set_xlim(start, end)
        axis.grid(axis="x", color="#dddddd", linewidth=0.5, alpha=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    tracks = load_tracks(args.tracks_npz)
    if not args.genes_csv.is_file() or not args.structures_csv.is_file():
        raise SystemExit(
            f"找不到基因/结构表：{args.genes_csv}, {args.structures_csv}；等待 #1 合入或显式指定"
        )
    genes = pd.read_csv(args.genes_csv)
    structures = pd.read_csv(args.structures_csv)
    for name, table in (("genes", genes), ("structures", structures)):
        missing = sorted({"start", "end"} - set(table.columns))
        if missing:
            raise SystemExit(f"{name} 表缺少列 {missing}")

    genome_length = int(tracks["ends"].max())
    if (args.start is None) != (args.end is None):
        raise SystemExit("--start 与 --end 必须同时提供")
    if args.start is not None:
        if not 0 <= args.start < args.end <= genome_length:
            raise SystemExit(f"区间必须满足 0 <= start < end <= {genome_length}")
        intervals = [(args.start, args.end)]
    else:
        intervals = genome_segments(genome_length, args.segment_bp)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    genes.to_csv(args.out_dir / "genes_used.csv", index=False, encoding="utf-8")
    structures.to_csv(args.out_dir / "structures_used.csv", index=False, encoding="utf-8")
    for index, (start, end) in enumerate(intervals, start=1):
        output = args.out_dir / f"tracks_{start:07d}_{end:07d}.png"
        plot_interval(
            tracks,
            genes,
            structures,
            args.conditions,
            start,
            end,
            output,
            sqrt_display=args.sqrt_display,
            dpi=args.dpi,
        )
        print(f"[{index}/{len(intervals)}] {output}", flush=True)
    run = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "code_version": code_version(),
        "tracks_npz": str(args.tracks_npz),
        "genes_snapshot": "genes_used.csv",
        "structures_snapshot": "structures_used.csv",
        "conditions": args.conditions,
        "segment_bp": args.segment_bp,
        "sqrt_display": args.sqrt_display,
        "figure_count": len(intervals),
        "first_interval": intervals[0],
        "last_interval": intervals[-1],
        "dpi": args.dpi,
    }
    (args.out_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成 {len(intervals)} 张轨道图；最后区间 {intervals[-1]}")


if __name__ == "__main__":
    main()
