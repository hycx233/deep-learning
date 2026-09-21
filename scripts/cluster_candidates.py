"""在统一自编码器表征上聚类候选、已知参考与背景。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.candidate_clustering import (  # noqa: E402
    as_bool,
    assign_independent_loci,
    assign_spatial_groups,
    nearest_rows,
    select_pure_background_rows,
    summarize_clusters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA + DBSCAN 候选聚类")
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--latents-npz", type=Path, required=True)
    parser.add_argument("--windows-npz", type=Path, required=True)
    parser.add_argument("--structures-csv", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/discovery/clustering"))
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--eps", type=float, default=2.5)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--background-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def load_latents(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        ids = archive["window_id"].astype(str)
        latent = archive["Z"].astype(np.float64)
    if latent.ndim != 2 or len(ids) != len(latent):
        raise ValueError("latents.npz 的 window_id 与 Z 形状不匹配")
    return ids, latent


def map_known_references(
    scores: pd.DataFrame, structures: pd.DataFrame
) -> tuple[dict[int, dict[str, str]], pd.DataFrame]:
    centers = scores["center"].to_numpy(dtype=np.int64)
    records: list[dict[str, object]] = []
    for row in structures.itertuples(index=False):
        nearest = int(np.abs(centers - int(row.center)).argmin())
        records.append(
            {
                "score_index": int(scores.index[nearest]),
                "latent_row": int(scores.iloc[nearest]["latent_row"]),
                "window_id": str(scores.iloc[nearest]["window_id"]),
                "structure_id": str(row.ID),
                "structure_type": str(row.type),
                "center_distance_bp": int(abs(centers[nearest] - int(row.center))),
            }
        )
    mapping_table = pd.DataFrame(records)
    reference: dict[int, dict[str, str]] = {}
    for latent_row, group in mapping_table.groupby("latent_row"):
        reference[int(latent_row)] = {
            "known_reference_ids": ";".join(sorted(group["structure_id"].unique())),
            "known_reference_types": ";".join(sorted(group["structure_type"].unique())),
        }
    return reference, mapping_table


def classification_check(
    mapping_table: pd.DataFrame,
    scores: pd.DataFrame,
    latent: np.ndarray,
    seed: int,
) -> dict[str, object]:
    unambiguous = []
    for latent_row, group in mapping_table.groupby("latent_row"):
        labels = group["structure_type"].unique()
        if len(labels) == 1:
            unambiguous.append((int(latent_row), str(labels[0])))
    rows = np.array([row for row, _ in unambiguous], dtype=np.int64)
    labels = np.array([label for _, label in unambiguous])
    counts = pd.Series(labels).value_counts()
    score_by_row = scores.set_index("latent_row")
    reference_windows = score_by_row.loc[rows, ["chrom", "start", "end"]].reset_index()
    reference_windows["spatial_group"] = assign_spatial_groups(reference_windows)
    groups = reference_windows["spatial_group"].to_numpy()
    folds = int(min(5, len(set(groups))))
    if folds < 2:
        return {"available": False, "reason": "空间连通组不足 2 个"}

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    predicted = cross_val_predict(
        model,
        latent[rows],
        labels,
        cv=splitter,
        groups=groups,
    )
    return {
        "available": True,
        "features": int(latent.shape[1]),
        "independent_reference_positions": int(len(rows)),
        "class_counts": {str(name): int(value) for name, value in counts.items()},
        "spatial_groups": int(len(set(groups))),
        "cross_fold_spatial_overlap_groups": 0,
        "folds": folds,
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "majority_accuracy": float(counts.max() / counts.sum()),
        "majority_balanced_accuracy": float(1.0 / len(counts)),
        "note": (
            "完整窗口按空间重叠连通组进入同一折；该分组交叉验证只检查 32 维表征"
            "是否含已知类别信息，不是最终分类模型指标。"
        ),
    }


def add_reference_distances(members: pd.DataFrame, coordinates: np.ndarray) -> pd.DataFrame:
    result = members.copy()
    known_mask = as_bool(result["is_known_reference"]).to_numpy()
    known = result.loc[known_mask].reset_index()
    nearest, distances = nearest_rows(coordinates, coordinates[known_mask])
    result["nearest_known_window_id"] = known.iloc[nearest]["window_id"].to_numpy()
    result["nearest_known_ids"] = known.iloc[nearest]["known_reference_ids"].to_numpy()
    result["nearest_known_types"] = known.iloc[nearest]["known_reference_types"].to_numpy()
    result["nearest_known_pca_distance"] = distances
    return result


def cluster_centers(members: pd.DataFrame, pc_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    known = members[as_bool(members["is_known_reference"])].reset_index(drop=True)
    for cluster, group in members[members["cluster"] != -1].groupby("cluster"):
        center = group[pc_columns].mean().to_numpy(dtype=float)
        nearest, distance = nearest_rows(center[None, :], known[pc_columns].to_numpy(dtype=float))
        reference = known.iloc[int(nearest[0])]
        record: dict[str, object] = {
            "cluster": int(cluster),
            "nearest_known_window_id": reference["window_id"],
            "nearest_known_ids": reference["known_reference_ids"],
            "nearest_known_types": reference["known_reference_types"],
            "nearest_known_pca_distance": float(distance[0]),
        }
        record.update({name: float(value) for name, value in zip(pc_columns, center)})
        rows.append(record)
    return pd.DataFrame(rows)


def plot_pca(members: pd.DataFrame, out_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    styles = [
        ("background", members["is_background_reference"], "#9e9e9e", 18),
        ("known reference", members["is_known_reference"], "#1f77b4", 28),
        ("candidate", members["is_candidate"], "#d62728", 42),
    ]
    for label, mask, color, size in styles:
        subset = members[as_bool(mask)]
        axis.scatter(subset["PC1"], subset["PC2"], s=size, c=color, alpha=0.72, label=label)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title("Shared autoencoder representation: PCA")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_representatives(
    members: pd.DataFrame,
    matrices: np.ndarray,
    masks: np.ndarray,
    pc_columns: list[str],
    out_path: Path,
) -> list[str]:
    choices: list[tuple[str, pd.Series]] = []
    candidate_clusters = members[(members["cluster"] != -1) & as_bool(members["is_candidate"])]
    for cluster, group in candidate_clusters.groupby("cluster"):
        center = group[pc_columns].mean().to_numpy(dtype=float)
        distance = ((group[pc_columns].to_numpy(dtype=float) - center) ** 2).sum(axis=1)
        choices.append((f"candidate cluster {cluster}", group.iloc[int(distance.argmin())]))
    if not choices:
        top = members[as_bool(members["is_candidate"])].sort_values(
            "reconstruction_error", ascending=False
        ).iloc[0]
        choices.append(("top candidate (noise)", top))
    choices = choices[:6]
    known = members[as_bool(members["is_known_reference"])]
    for known_type in ("OPCID", "CHIN", "CHID"):
        subset = known[known["known_reference_types"].str.split(";").map(lambda x: known_type in x)]
        if not subset.empty:
            choices.append((f"known {known_type}", subset.iloc[0]))

    images = [matrices[int(row["latent_row"])] for _, row in choices]
    valid_masks = [masks[int(row["latent_row"])] for _, row in choices]
    valid_values = np.concatenate(
        [matrix[valid] for matrix, valid in zip(images, valid_masks)]
    )
    vmax = float(np.quantile(valid_values, 0.995)) or 1.0
    color_map = plt.colormaps["magma"].copy()
    color_map.set_bad("#d9d9d9")
    columns = min(3, len(images))
    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.4 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.set_visible(False)
    ids: list[str] = []
    for axis, (label, row), matrix, valid in zip(
        axes.ravel(), choices, images, valid_masks
    ):
        axis.set_visible(True)
        axis.imshow(
            np.ma.masked_where(~valid, matrix),
            cmap=color_map,
            vmin=0,
            vmax=vmax,
            origin="lower",
        )
        axis.set_title(f"{label}\n{row['window_id']}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        ids.append(str(row["window_id"]))
    figure.suptitle(f"Unified color scale (vmin=0, vmax={vmax:.3g})")
    figure.tight_layout()
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return ids


def main() -> None:
    args = parse_args()
    if args.pca_components < 2 or args.eps <= 0 or args.min_samples < 2:
        raise SystemExit("PCA 维数至少为 2，eps > 0，min-samples 至少为 2")
    scores = pd.read_csv(args.scores_csv)
    required = {"window_id", "chrom", "start", "end", "center", "latent_row", "is_candidate"}
    missing = required.difference(scores.columns)
    if missing:
        raise SystemExit(f"分数表缺少字段：{sorted(missing)}")
    scores = scores.sort_values("latent_row").reset_index(drop=True)
    ids, latent = load_latents(args.latents_npz)
    if len(scores) != len(latent) or not np.array_equal(scores["window_id"].astype(str), ids):
        raise SystemExit("分数表与隐向量的 window_id 顺序不一致")
    structures = pd.read_csv(args.structures_csv)
    known_reference, mapping_table = map_known_references(scores, structures)

    candidate_rows = set(scores.loc[as_bool(scores["is_candidate"]), "latent_row"].astype(int))
    known_rows = set(known_reference)
    pure_background = select_pure_background_rows(
        scores, candidate_rows, known_rows
    )
    rng = np.random.default_rng(args.seed)
    background_rows = set(
        rng.choice(pure_background, size=min(args.background_count, len(pure_background)), replace=False)
    )
    selected_rows = sorted(candidate_rows | known_rows | background_rows)
    members = scores.set_index("latent_row").loc[selected_rows].reset_index()
    members["is_candidate"] = members["latent_row"].isin(candidate_rows)
    members["is_known_reference"] = members["latent_row"].isin(known_rows)
    members["is_background_reference"] = members["latent_row"].isin(background_rows)
    members["known_overlap"] = as_bool(members["known_overlap"])
    members["known_reference_ids"] = members["latent_row"].map(
        lambda row: known_reference.get(int(row), {}).get("known_reference_ids", "")
    )
    members["known_reference_types"] = members["latent_row"].map(
        lambda row: known_reference.get(int(row), {}).get("known_reference_types", "")
    )

    scaled = StandardScaler().fit_transform(latent[selected_rows])
    components = min(args.pca_components, scaled.shape[1], len(scaled) - 1)
    pca = PCA(n_components=components, whiten=True, random_state=args.seed)
    coordinates = pca.fit_transform(scaled)
    labels = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(coordinates)
    pc_columns = [f"PC{number + 1}" for number in range(components)]
    members[pc_columns] = coordinates
    members["cluster"] = labels

    candidate_mask = as_bool(members["is_candidate"])
    members["independent_locus_id"] = ""
    members.loc[candidate_mask, "independent_locus_id"] = assign_independent_loci(
        members.loc[candidate_mask]
    )
    members = add_reference_distances(members, coordinates)
    summaries = summarize_clusters(members)
    centers = cluster_centers(members, pc_columns)
    if not summaries.empty:
        centers = summaries.merge(centers, on="cluster", how="left")

    candidates = members[candidate_mask].copy()
    candidates.insert(0, "candidate_id", [f"candidate_{i:03d}" for i in range(1, len(candidates) + 1)])
    candidates["rep2_window_id"] = ""
    candidates["replicate_correlation"] = np.nan
    candidates["reproducibility_status"] = "pending_P09"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    members.to_csv(args.out_dir / "cluster_members.csv", index=False, encoding="utf-8")
    candidates.to_csv(args.out_dir / "candidate_clusters.csv", index=False, encoding="utf-8")
    centers.to_csv(args.out_dir / "clusters.csv", index=False, encoding="utf-8")
    mapping_table.to_csv(args.out_dir / "known_reference_mapping.csv", index=False, encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "pca.npz",
        latent_rows=np.array(selected_rows, dtype=np.int64),
        coordinates=coordinates.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        cluster=labels.astype(np.int32),
    )
    plot_pca(members, args.out_dir / "pca_groups.png")
    with np.load(args.windows_npz) as archive:
        matrices = archive["X"]
        masks = archive["mask"] if "mask" in archive else np.ones_like(matrices, dtype=bool)
        representative_ids = plot_representatives(
            members,
            matrices,
            masks,
            pc_columns,
            args.out_dir / "cluster_representatives.png",
        )

    classification = classification_check(mapping_table, scores, latent, args.seed)
    (args.out_dir / "classification.json").write_text(
        json.dumps(classification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    suspicious = (
        summaries.loc[summaries["suspected_new_cluster"], "cluster"].astype(int).tolist()
        if not summaries.empty
        else []
    )
    summary = {
        "command": (
            "cluster_candidates.py "
            f"--scores-csv {args.scores_csv} --latents-npz {args.latents_npz} "
            f"--windows-npz {args.windows_npz} --structures-csv {args.structures_csv} "
            f"--pca-components {args.pca_components} --eps {args.eps} "
            f"--min-samples {args.min_samples} --background-count {args.background_count} "
            f"--seed {args.seed} --out-dir {args.out_dir}"
        ),
        "code_version": git_revision(),
        "latent_dimension": int(latent.shape[1]),
        "selected_members": int(len(members)),
        "candidate_windows": int(candidate_mask.sum()),
        "independent_candidate_loci": int(candidates["independent_locus_id"].nunique()),
        "known_reference_positions": int(members["is_known_reference"].sum()),
        "background_reference_positions": int(members["is_background_reference"].sum()),
        "eligible_pure_background_positions": int(len(pure_background)),
        "background_known_overlap_positions": int(
            members.loc[
                as_bool(members["is_background_reference"]), "known_overlap"
            ].sum()
        ),
        "pca_components": components,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_whiten": True,
        "dbscan": {"eps": args.eps, "min_samples": args.min_samples},
        "non_noise_clusters": int(len(set(labels) - {-1})),
        "noise_members": int((labels == -1).sum()),
        "suspected_new_clusters": suspicious,
        "new_cluster_rule": (
            "至少 5 个互不重叠的独立候选位置；簇内没有已知参考成员；"
            "所有候选区间均不与已知结构重叠；DBSCAN 噪声 -1 不成簇。"
        ),
        "representative_window_ids": representative_ids,
        "classification": classification,
        "conclusion": (
            "存在满足本组规则的疑似新簇，仍需 P09 跨重复验证。"
            if suspicious
            else "没有簇满足本组疑似新簇规则；不宣称发现新结构。"
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"候选 {len(candidates)} 个，独立位置 {summary['independent_candidate_loci']} 个；"
        f"非噪声簇 {summary['non_noise_clusters']} 个，疑似新簇 {len(suspicious)} 个。"
    )


if __name__ == "__main__":
    main()
