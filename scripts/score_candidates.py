"""用自编码器重建误差给扫描窗口打分并导出候选。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.autoencoder import ConvAutoencoder, masked_mse, upper_triangle_mask  # noqa: E402
from src.discovery_data import load_discovery_dataset  # noqa: E402


def code_version() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自编码器候选窗口打分")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--arrays-npz", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/discovery/candidates")
    )
    parser.add_argument("--candidate-quantile", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--plot-per-group", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def score_all(
    model: ConvAutoencoder,
    matrices: np.ndarray,
    masks: np.ndarray,
    mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(matrices).unsqueeze(1),
            torch.from_numpy(masks),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    errors: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for features, sample_mask in loader:
            features = features.to(device)
            sample_mask = sample_mask.to(device)
            reconstruction = model(features)
            errors.append(
                masked_mse(
                    reconstruction,
                    features,
                    sample_mask & mask,
                    reduction="none",
                )
                .cpu()
                .numpy()
            )
            latents.append(model.encode(features).cpu().numpy())
    return np.concatenate(errors), np.concatenate(latents)


def known_recall_summary(scores: pd.DataFrame) -> dict:
    if "known_overlap" not in scores.columns:
        return {"available": False, "reason": "窗口表没有 known_overlap 列"}
    known = scores[scores["known_overlap"].astype(bool)].copy()
    if known.empty:
        return {"available": False, "reason": "输入中没有已知结构窗口"}

    result: dict[str, object] = {
        "available": True,
        "known_windows": int(len(known)),
        "candidate_windows": int(known["is_candidate"].sum()),
        "window_recall": float(known["is_candidate"].mean()),
    }
    if "known_structure_ids" in known.columns:
        exploded = known.assign(
            known_structure_ids=known["known_structure_ids"].fillna("").astype(str).str.split(";")
        ).explode("known_structure_ids")
        exploded = exploded[exploded["known_structure_ids"].ne("")]
        if not exploded.empty:
            by_structure = exploded.groupby("known_structure_ids")["is_candidate"].max()
            result["known_structures"] = int(len(by_structure))
            result["recalled_structures"] = int(by_structure.sum())
            result["structure_recall"] = float(by_structure.mean())
    elif "structure_id" in known.columns:
        valid = known[known["structure_id"].notna()].copy()
        if not valid.empty:
            by_structure = valid.groupby("structure_id")["is_candidate"].max()
            result["known_structures"] = int(len(by_structure))
            result["recalled_structures"] = int(by_structure.sum())
            result["structure_recall"] = float(by_structure.mean())
    type_column = next((name for name in ("known_types", "type", "known_type", "label") if name in known.columns), None)
    if type_column == "known_types":
        split_types = known[type_column].fillna("").str.split(";")
        result["per_type_window_recall"] = {
            name: float(known.loc[split_types.map(lambda values: name in values), "is_candidate"].mean())
            for name in ("OPCID", "CHIN", "CHID")
            if split_types.map(lambda values: name in values).any()
        }
    elif type_column:
        result["per_type_window_recall"] = {
            str(name): float(value)
            for name, value in known.groupby(type_column)["is_candidate"].mean().items()
        }
    return result


def score_intensity_summary(errors: np.ndarray, intensity: np.ndarray) -> dict:
    """报告误差与强度的关系，检查候选是否只是在挑高密度窗口。"""
    if len(errors) < 2 or np.std(errors) == 0 or np.std(intensity) == 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(errors, intensity)[0, 1])
    error_ranks = pd.Series(errors).rank(method="average").to_numpy()
    intensity_ranks = pd.Series(intensity).rank(method="average").to_numpy()
    if np.std(error_ranks) == 0 or np.std(intensity_ranks) == 0:
        spearman = None
    else:
        spearman = float(np.corrcoef(error_ranks, intensity_ranks)[0, 1])
    return {
        "pearson": pearson,
        "spearman": spearman,
        "interpretation": (
            "相关系数接近 1 时，高重建误差可能主要反映接触强度；"
            "正式结论还需查看统一色标热图与已知/背景对照。"
        ),
    }


def plot_examples(
    model: ConvAutoencoder,
    matrices: np.ndarray,
    masks: np.ndarray,
    scores: pd.DataFrame,
    count: int,
    device: torch.device,
    out_path: Path,
) -> None:
    if count <= 0:
        return
    known_mask = (
        scores["known_overlap"].astype(bool).to_numpy()
        if "known_overlap" in scores.columns
        else np.zeros(len(scores), dtype=bool)
    )
    background = np.flatnonzero(~known_mask)
    groups: list[tuple[str, np.ndarray]] = []
    candidate_background = background[
        np.argsort(scores.iloc[background]["reconstruction_error"].to_numpy())[::-1]
    ][:count]
    if len(candidate_background):
        groups.append(("High-score background", candidate_background))
    known = np.flatnonzero(known_mask)
    if len(known):
        selected_known = known[
            np.argsort(scores.iloc[known]["reconstruction_error"].to_numpy())[::-1]
        ][:count]
        groups.append(("Known structure", selected_known))
    if len(background):
        low_background = background[
            np.argsort(scores.iloc[background]["reconstruction_error"].to_numpy())
        ][:count]
        groups.append(("Low-score background", low_background))
    if not groups:
        return

    chosen = [
        (group, int(index), int(scores.iloc[index]["latent_row"]))
        for group, indices in groups
        for index in indices
    ]
    inputs = torch.from_numpy(
        np.stack([matrices[array_row] for _, _, array_row in chosen])
    ).unsqueeze(1)
    with torch.no_grad():
        reconstructions = model(inputs.to(device)).cpu().numpy()[:, 0]
    originals = inputs.numpy()[:, 0]
    valid_masks = np.stack([masks[array_row] for _, _, array_row in chosen])
    valid_originals = np.concatenate(
        [matrix[valid] for matrix, valid in zip(originals, valid_masks)]
    )
    valid_errors = np.concatenate(
        [
            np.abs(original - reconstruction)[valid]
            for original, reconstruction, valid in zip(
                originals, reconstructions, valid_masks
            )
        ]
    )
    vmax = float(np.quantile(valid_originals, 0.995)) or 1.0
    error_max = float(np.quantile(valid_errors, 0.995)) or 1.0
    signal_cmap = plt.colormaps["magma"].copy()
    error_cmap = plt.colormaps["viridis"].copy()
    signal_cmap.set_bad("#d9d9d9")
    error_cmap.set_bad("#d9d9d9")

    figure, axes = plt.subplots(len(chosen), 3, figsize=(9, 3 * len(chosen)), squeeze=False)
    for row, ((group, index, _), original, reconstruction, valid) in enumerate(
        zip(chosen, originals, reconstructions, valid_masks)
    ):
        axes[row, 0].imshow(
            np.ma.masked_where(~valid, original),
            cmap=signal_cmap,
            vmin=0,
            vmax=vmax,
            origin="lower",
        )
        axes[row, 1].imshow(
            np.ma.masked_where(~valid, reconstruction),
            cmap=signal_cmap,
            vmin=0,
            vmax=vmax,
            origin="lower",
        )
        axes[row, 2].imshow(
            np.ma.masked_where(~valid, np.abs(original - reconstruction)),
            cmap=error_cmap,
            vmin=0,
            vmax=error_max,
            origin="lower",
        )
        axes[row, 0].set_ylabel(
            f"{group}\n{scores.iloc[index]['window_id']}\n"
            f"err={scores.iloc[index]['reconstruction_error']:.4g}"
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    for axis, title in zip(axes[0], ("Input", "Reconstruction", "Absolute error")):
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.candidate_quantile < 1.0:
        raise SystemExit("candidate-quantile 必须在 0 和 1 之间")
    if not args.checkpoint.is_file():
        raise SystemExit(f"找不到权重文件：{args.checkpoint}")

    matrices, masks, meta, intensity = load_discovery_dataset(
        args.windows_csv, args.arrays_npz, require_intensity=True
    )
    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if int(checkpoint["window_size"]) != matrices.shape[-1]:
        raise SystemExit(
            f"权重窗口大小 {checkpoint['window_size']} 与输入 {matrices.shape[-1]} 不一致"
        )
    model = ConvAutoencoder(int(checkpoint["latent_dim"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    mask = upper_triangle_mask(
        matrices.shape[-1], int(checkpoint["diagonal_exclusion"]), device=device
    )
    errors, latents = score_all(
        model, matrices, masks, mask, args.batch_size, device
    )
    threshold = float(np.quantile(errors, args.candidate_quantile))

    scores = meta.copy()
    scores["reconstruction_error"] = errors
    scores["intensity_score"] = intensity
    scores["candidate_threshold"] = threshold
    scores["is_candidate"] = errors >= threshold
    scores["latent_row"] = np.arange(len(scores), dtype=np.int64)
    scores = scores.sort_values("reconstruction_error", ascending=False, kind="stable")
    scores["score_rank"] = np.arange(1, len(scores) + 1)
    candidates = scores[scores["is_candidate"]].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out_dir / "window_scores.csv", index=False, encoding="utf-8")
    candidates.to_csv(args.out_dir / "candidates.csv", index=False, encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "latents.npz",
        window_id=meta["window_id"].astype(str).to_numpy(),
        Z=latents.astype(np.float32),
    )
    plot_examples(
        model,
        matrices,
        masks,
        scores.reset_index(drop=True),
        args.plot_per_group,
        device,
        args.out_dir / "score_examples.png",
    )

    summary = {
        "code_version": code_version(),
        "checkpoint": str(args.checkpoint),
        "windows_csv": str(args.windows_csv),
        "arrays_npz": str(args.arrays_npz),
        "windows": int(len(scores)),
        "candidate_quantile": args.candidate_quantile,
        "candidate_threshold": threshold,
        "loss_mask": "逐窗口有效 mask 与固定上三角 mask 的交集；每个窗口单独按有效像素数求均值",
        "candidates": int(len(candidates)),
        "error_intensity_relation": score_intensity_summary(errors, intensity),
        "known_recall": known_recall_summary(scores),
        "note": (
            "candidate_quantile 是本次运行筛选规则；正式阈值需结合已知结构和随机背景对照固定。"
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"窗口 {len(scores)}，阈值 {threshold:.6g}，候选 {len(candidates)}；"
        f"结果写入 {args.out_dir}"
    )


if __name__ == "__main__":
    main()
