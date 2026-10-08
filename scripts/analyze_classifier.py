#!/usr/bin/env python3
"""汇总固定测试集评价、基线、混淆矩阵与 Grad-CAM 解释图。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.classifier_data import CLASS_NAMES, load_dataset  # noqa: E402
from src.cnn import SmallCNN  # noqa: E402
from src.metrics import accuracy, confusion_counts, macro_f1, per_class_f1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True,
                        help="scripts/predict_classifier.py --split test 的输出")
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--arrays-npz", type=Path, required=True)
    parser.add_argument("--splits-csv", type=Path, default=None,
                        help="默认读取权重同目录下的 splits.csv；缺失时直接报错")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/classifier/evaluation"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--examples-per-class", type=int, choices=(2, 3), default=3)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixed_splits(meta: pd.DataFrame, path: Path) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到权重配套的划分文件：{path}。正式测试不按 seed 重新划分。"
        )
    splits = pd.read_csv(path)
    missing = [column for column in ("window_id", "split") if column not in splits]
    if missing:
        raise ValueError(f"{path} 缺少列 {missing}")
    if splits["window_id"].duplicated().any():
        raise ValueError(f"{path} 中 window_id 不唯一，不能确定测试样本归属")
    if not splits["split"].isin(("train", "val", "test")).all():
        raise ValueError(f"{path} 的 split 列包含空值或未知集合")
    if meta["window_id"].duplicated().any():
        raise ValueError("windows.csv 中 window_id 不唯一")
    if set(splits["window_id"]) != set(meta["window_id"]):
        raise ValueError("splits.csv 与 windows.csv 的 window_id 集合不同；请确认三者来自同一运行")
    mapping = splits.set_index("window_id")["split"]
    return meta["window_id"].map(mapping)


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    matrix = confusion_counts(y_true, y_pred, len(CLASS_NAMES))
    f1 = per_class_f1(y_true, y_pred, len(CLASS_NAMES))
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        per_class[name] = {
            "support": support,
            "predicted": predicted,
            "precision": round(tp / predicted, 4) if predicted else 0.0,
            "recall": round(tp / support, 4) if support else 0.0,
            "f1": round(float(f1[index]), 4),
        }
    return {
        "samples": int(len(y_true)),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred, len(CLASS_NAMES)), 4),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def plot_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 4.7), layout="constrained")
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Test-set confusion matrix",
    )
    threshold = float(matrix.max()) / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black")
    figure.colorbar(image, ax=axis, label="Samples")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def select_gradcam_examples(predictions: pd.DataFrame, count: int) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for name in CLASS_NAMES:
        group = predictions[predictions["true_label"] == name].copy()
        if len(group) < count:
            raise ValueError(f"测试集 {name} 只有 {len(group)} 个样本，无法选 {count} 个解释案例")
        # Each structure contributes both WT replicates. Choose one window per
        # structure so the explanation panel shows distinct positions.
        group["confidence"] = group[[f"score_{label}" for label in CLASS_NAMES]].max(axis=1)
        group = group.sort_values(["structure_id", "window_id"])
        group = group.drop_duplicates("structure_id", keep="first")
        if len(group) < count:
            raise ValueError(
                f"测试集 {name} 只有 {len(group)} 个独立结构，无法选 {count} 个解释案例"
            )
        correct = group[group["correct"]].sort_values(
            [f"score_{name}", "window_id"], ascending=[False, True]
        )
        incorrect = group[~group["correct"]].sort_values(
            ["confidence", "window_id"], ascending=[False, True]
        )
        picks: list[pd.Series] = []
        if not correct.empty:
            picks.append(correct.iloc[0])
        if not incorrect.empty and len(picks) < count:
            picks.append(incorrect.iloc[0])
        used = {row["window_id"] for row in picks}
        remaining = group[~group["window_id"].isin(used)].sort_values(
            [f"score_{name}", "window_id"], ascending=[False, True]
        )
        picks.extend(row for _, row in remaining.iterrows() if row["window_id"] not in used)
        selected.extend(picks[:count])
    return pd.DataFrame(selected).reset_index(drop=True)


def gradcam(model: SmallCNN, matrix: np.ndarray, target: int, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0).to(device)
    features = model.features(tensor)
    logits = model.head(features)
    gradients = torch.autograd.grad(logits[0, target], features)[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    activation = torch.relu((weights * features).sum(dim=1, keepdim=True))
    activation = functional.interpolate(activation, size=matrix.shape, mode="bilinear",
                                        align_corners=False)[0, 0]
    activation = activation.detach().cpu().numpy()
    low, high = float(activation.min()), float(activation.max())
    if high <= low:
        return np.zeros_like(matrix, dtype=np.float32)
    return ((activation - low) / (high - low)).astype(np.float32)


def draw_gradcam(matrix: np.ndarray, cam: np.ndarray, row: pd.Series,
                 class_names: tuple[str, ...], path: Path) -> None:
    scale_values = matrix[np.isfinite(matrix) & (matrix > 0)]
    vmax = float(np.quantile(scale_values, 0.995)) if scale_values.size else 1.0
    vmax = max(vmax, np.finfo(np.float32).eps)
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 4), layout="constrained")
    original = axes[0].imshow(matrix, cmap="magma", origin="lower", vmin=0, vmax=vmax)
    axes[0].set_title("Original input matrix")
    axes[1].imshow(matrix, cmap="magma", origin="lower", vmin=0, vmax=vmax)
    axes[1].imshow(cam, cmap="inferno", origin="lower", alpha=0.48, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM overlay for predicted class")
    for axis in axes:
        axis.set_xlabel("Window bin")
        axis.set_ylabel("Window bin")
        axis.set_aspect("equal")
    figure.colorbar(original, ax=axes[0], shrink=0.8, label="Input intensity")
    figure.colorbar(plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1)),
                    ax=axes[1], shrink=0.8, label="Relative activation")
    target_name = class_names[int(row["pred_index"])]
    figure.suptitle(
        f"{row['window_id']} | true {row['true_label']} / predicted {row['pred_label']}"
        f" (CAM target {target_name})"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    started = time.monotonic()
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size 必须大于 0")
    for path in (args.checkpoint, args.predictions_csv, args.windows_csv, args.arrays_npz):
        if not path.is_file():
            raise FileNotFoundError(f"找不到输入文件：{path}")
    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    class_names = tuple(checkpoint["class_names"])
    if class_names != CLASS_NAMES:
        raise ValueError(f"权重类别顺序 {class_names} 与当前代码 {CLASS_NAMES} 不一致")

    x, y, meta = load_dataset(args.windows_csv, args.arrays_npz)
    splits_path = args.splits_csv or (args.checkpoint.parent / "splits.csv")
    meta["split"] = load_fixed_splits(meta, splits_path)
    test_mask = meta["split"].eq("test").to_numpy()
    train_mask = meta["split"].eq("train").to_numpy()
    if not test_mask.any() or not train_mask.any():
        raise ValueError("固定划分必须同时包含 train 与 test 样本")

    predictions = pd.read_csv(args.predictions_csv)
    required = {"window_id", "true_label", "pred_label", "checkpoint_epoch", *
                (f"score_{name}" for name in CLASS_NAMES)}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"测试预测表缺少列 {missing}")
    if predictions["window_id"].duplicated().any():
        raise ValueError("测试预测表的 window_id 有重复")
    expected_ids = set(meta.loc[test_mask, "window_id"])
    if set(predictions["window_id"]) != expected_ids:
        raise ValueError("预测 CSV 与权重配套 splits.csv 的 test 窗口不完全一致")
    if set(predictions["checkpoint_epoch"].astype(int)) != {int(checkpoint["epoch"])}:
        raise ValueError("预测 CSV 的 checkpoint_epoch 与指定权重不一致")

    row_for_id = meta.reset_index(drop=True).set_index("window_id")
    expected_true = predictions["window_id"].map(row_for_id["label_name"])
    if not np.array_equal(expected_true.to_numpy(), predictions["true_label"].to_numpy()):
        raise ValueError("预测 CSV 的真实类别与窗口表不一致")
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    y_test = predictions["true_label"].map(class_to_index).to_numpy(dtype=np.int64)
    cnn_pred = predictions["pred_label"].map(class_to_index)
    if cnn_pred.isna().any():
        raise ValueError("预测 CSV 中出现未知预测类别")
    cnn_pred = cnn_pred.to_numpy(dtype=np.int64)
    score_columns = [f"score_{name}" for name in CLASS_NAMES]
    cnn_scores = predictions[score_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(cnn_scores).all():
        raise ValueError("预测分数中出现 NaN 或 Inf")
    if not np.array_equal(cnn_scores.argmax(axis=1), cnn_pred):
        raise ValueError("预测类别与三类分数的最大值不一致")

    model = SmallCNN(n_classes=len(CLASS_NAMES), dropout=float(checkpoint.get("dropout", 0.3))).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    test_indices = meta.index[test_mask].to_numpy()
    repeated_logits = []
    with torch.no_grad():
        for start in range(0, len(test_indices), args.batch_size):
            batch = torch.from_numpy(x[test_indices[start:start + args.batch_size]])
            logits = model(batch.unsqueeze(1).to(device)).cpu().numpy()
            repeated_logits.append(logits)
    repeated_scores = torch.softmax(torch.from_numpy(np.concatenate(repeated_logits)), dim=1).numpy()
    if not np.allclose(repeated_scores, cnn_scores, rtol=1e-5, atol=1e-7):
        raise ValueError("预测 CSV 分数与指定 checkpoint 重跑结果不一致")

    train_labels = y[train_mask]
    train_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    majority_class = int(train_counts.argmax())
    majority_pred = np.full(len(y_test), majority_class, dtype=np.int64)
    linear = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs",
                                random_state=20260918)
    convergence_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        linear.fit(x[train_mask].reshape(int(train_mask.sum()), -1), train_labels)
    convergence_messages.extend(str(item.message) for item in caught
                                if issubclass(item.category, ConvergenceWarning))
    linear_pred = linear.predict(x[test_indices].reshape(len(test_indices), -1)).astype(np.int64)

    summaries = {
        "cnn": summarize(y_test, cnn_pred),
        "majority_baseline": summarize(y_test, majority_pred),
        "linear_logistic_baseline": summarize(y_test, linear_pred),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(np.asarray(summaries["cnn"]["confusion_matrix"]),
                          args.out_dir / "confusion_matrix_test.png")

    id_to_original_index = dict(zip(meta["window_id"], meta.index))
    enriched = predictions.copy()
    enriched["correct"] = enriched["true_label"].eq(enriched["pred_label"])
    enriched["majority_pred"] = CLASS_NAMES[majority_class]
    enriched["linear_pred"] = [CLASS_NAMES[index] for index in linear_pred]
    enriched.to_csv(args.out_dir / "test_predictions_with_baselines.csv", index=False,
                    encoding="utf-8")
    failures = enriched[enriched["true_label"] != enriched["pred_label"]].copy()
    failures.to_csv(args.out_dir / "failure_cases.csv", index=False, encoding="utf-8")

    gradcam_dir = args.out_dir / "gradcam"
    gradcam_dir.mkdir(exist_ok=True)
    examples = select_gradcam_examples(enriched, args.examples_per_class)
    examples["pred_index"] = examples["pred_label"].map(class_to_index)
    for _, row in examples.iterrows():
        index = int(id_to_original_index[row["window_id"]])
        cam = gradcam(model, x[index], int(row["pred_index"]), device)
        safe_id = "".join(character if character.isalnum() or character in "-_" else "_"
                           for character in str(row["window_id"]))
        draw_gradcam(x[index], cam, row, CLASS_NAMES, gradcam_dir / f"{safe_id}.png")
    examples.drop(columns=["confidence", "pred_index"], errors="ignore").to_csv(
        args.out_dir / "gradcam_examples.csv", index=False, encoding="utf-8"
    )

    source_files = (
        Path(__file__), repo_root / "scripts/predict_classifier.py",
        repo_root / "src/classifier_data.py", repo_root / "src/cnn.py",
        repo_root / "src/metrics.py",
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "training_seed": int(checkpoint.get("split_seed", 20260918)),
        "linear_baseline_seed": 20260918,
        "split": "test",
        "split_file": str(splits_path),
        "split_file_sha256": sha256(splits_path),
        "windows_csv_sha256": sha256(args.windows_csv),
        "arrays_npz_sha256": sha256(args.arrays_npz),
        "predictions_csv_sha256": sha256(args.predictions_csv),
        "class_order": list(CLASS_NAMES),
        "train_samples_by_class": {name: int(train_counts[index])
                                    for index, name in enumerate(CLASS_NAMES)},
        "test_metrics": summaries,
        "cnn_prediction_counts": {
            name: int((enriched["pred_label"] == name).sum()) for name in CLASS_NAMES
        },
        "cnn_true_counts": {
            name: int((enriched["true_label"] == name).sum()) for name in CLASS_NAMES
        },
        "chin_bias_check": {
            "chin_is_most_predicted_class": bool(
                int((enriched["pred_label"] == "CHIN").sum())
                == max(int((enriched["pred_label"] == name).sum()) for name in CLASS_NAMES)
            ),
            "predicted_chin_count": int((enriched["pred_label"] == "CHIN").sum()),
            "true_chin_count": int((enriched["true_label"] == "CHIN").sum()),
        },
        "majority_class_from_train": CLASS_NAMES[majority_class],
        "linear_baseline": {
            "model": "LogisticRegression(l2, class_weight=balanced)",
            "max_iter": 1000,
            "random_state": 20260918,
            "convergence_warnings": convergence_messages,
        },
        "gradcam_examples": len(examples),
        "command": shlex.join(["python", *sys.argv]),
        "device": str(device),
        "python": platform.python_version(),
        "source_sha256": {str(path.relative_to(repo_root)): sha256(path)
                          for path in source_files},
        "runtime_seconds": round(time.monotonic() - started, 2),
    }
    (args.out_dir / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"评价结果、混淆矩阵、失败案例和 Grad-CAM 图已写入 {args.out_dir}")


if __name__ == "__main__":
    main()
