"""任务一：用训练好的权重复跑推理，导出每个评估样本的判定结果。

输出 CSV 每行一个窗口：ID、坐标、样本信息、真实类别、预测类别与三类分数。
逐类指标数值也打印出来并写入同名 ``.summary.json``；混淆矩阵图、基线与
Grad-CAM 属于 #4，不在本入口里做。

示例（从仓库根目录执行）：

    python scripts/predict_classifier.py \
        --checkpoint outputs/classifier/skeleton/best.pt \
        --windows-csv outputs/sample_windows/windows.csv \
        --arrays-npz outputs/sample_windows/windows.npz \
        --split test \
        --out-csv outputs/classifier/skeleton/predictions_test.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classifier_data import (  # noqa: E402
    CLASS_NAMES,
    assign_splits,
    load_dataset,
)
from src.cnn import SmallCNN  # noqa: E402
from src.metrics import accuracy, confusion_counts, macro_f1, per_class_f1  # noqa: E402

META_COLUMNS = (
    "window_id",
    "chrom",
    "start",
    "end",
    "sample_id",
    "condition",
    "replicate",
    "structure_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="已知结构三分类推理")
    parser.add_argument("--checkpoint", type=Path, required=True, help="train_classifier.py 的 best.pt")
    parser.add_argument(
        "--windows-csv", type=Path, default=Path("outputs/sample_windows/windows.csv")
    )
    parser.add_argument(
        "--arrays-npz", type=Path, default=Path("outputs/sample_windows/windows.npz")
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    parser.add_argument(
        "--splits-csv",
        type=Path,
        default=None,
        help="训练时保存的 splits.csv；默认取权重同目录下的那份",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/classifier/predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
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


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"找不到权重文件：{args.checkpoint}；先运行 scripts/train_classifier.py")

    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    class_names = tuple(checkpoint["class_names"])
    if class_names != CLASS_NAMES:
        raise SystemExit(
            f"权重里的类别顺序 {list(class_names)} 与当前代码 {list(CLASS_NAMES)} 不一致，"
            "请确认权重与代码版本是否匹配"
        )

    x, y, meta = load_dataset(args.windows_csv, args.arrays_npz)
    # 评价必须用训练时那一份划分。按种子重新划分会得到不同的集合，
    # 那样"测试集结果"就不是同一批样本，结论不可比。
    splits_path = args.splits_csv or (args.checkpoint.parent / "splits.csv")
    if splits_path.is_file():
        splits = pd.read_csv(splits_path)
        missing = [c for c in ("window_id", "split") if c not in splits.columns]
        if missing:
            raise SystemExit(f"{splits_path} 缺少列 {missing}；应为训练入口保存的 splits.csv")
        mapping = dict(zip(splits["window_id"], splits["split"]))
        unknown = sorted(set(meta["window_id"]) - set(mapping))
        if unknown:
            raise SystemExit(
                f"有 {len(unknown)} 个窗口不在 {splits_path} 里（例如 {unknown[:3]}），"
                "权重、窗口表与划分列表不是同一次运行"
            )
        meta["split"] = meta["window_id"].map(mapping)
        print(f"沿用训练时保存的划分：{splits_path}")
    else:
        seed = int(checkpoint.get("split_seed", 20260918))
        print(
            f"未找到 {splits_path}，按种子 {seed} 重新划分。"
            "这次评价的集合可能与训练时不同，正式结果请改用训练时保存的 splits.csv"
        )
        meta, _ = assign_splits(meta, seed=seed)

    if args.split != "all" and args.split not in set(meta["split"]):
        raise SystemExit(f"split={args.split} 在该数据中没有样本；现有集合 {sorted(set(meta['split']))}")
    mask = np.ones(len(meta), dtype=bool) if args.split == "all" else (meta["split"] == args.split).to_numpy()
    if not mask.any():
        raise SystemExit(f"split={args.split} 没有样本，无法推理")

    model = SmallCNN(
        n_classes=len(class_names),
        dropout=float(checkpoint.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[mask]).unsqueeze(1)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    logits_parts: list[np.ndarray] = []
    with torch.no_grad():
        for (features,) in loader:
            logits_parts.append(model(features.to(device)).cpu().numpy())
    logits = np.concatenate(logits_parts)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predicted = logits.argmax(axis=1)

    selected = meta.loc[mask]
    output = pd.DataFrame({column: selected[column].to_numpy() for column in META_COLUMNS if column in selected})
    output["true_label"] = [class_names[i] for i in y[mask]]
    output["pred_label"] = [class_names[i] for i in predicted]
    output["correct"] = output["true_label"] == output["pred_label"]
    for index, name in enumerate(class_names):
        output[f"score_{name}"] = probabilities[:, index]
    output["checkpoint_epoch"] = int(checkpoint["epoch"])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out_csv, index=False, encoding="utf-8")

    true_labels = y[mask]
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "samples": int(mask.sum()),
        "accuracy": round(accuracy(true_labels, predicted), 4),
        "macro_f1": round(macro_f1(true_labels, predicted, len(class_names)), 4),
        "per_class_f1": {
            name: round(float(value), 4)
            for name, value in zip(class_names, per_class_f1(true_labels, predicted, len(class_names)))
        },
        "confusion_matrix": confusion_counts(true_labels, predicted, len(class_names)).tolist(),
        "predictions_csv": str(args.out_csv),
    }
    args.out_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"{args.split} 集 {summary['samples']} 个样本，设备 {device}")
    print(f"accuracy {summary['accuracy']:.4f}  macro_f1 {summary['macro_f1']:.4f}")
    print("混淆矩阵（行=真实，列=预测，顺序 " + "/".join(class_names) + "）")
    for name, row in zip(class_names, summary["confusion_matrix"]):
        print(f"  {name:>5} {row}")
    print(f"逐样本结果已写入 {args.out_csv}")


if __name__ == "__main__":
    main()
