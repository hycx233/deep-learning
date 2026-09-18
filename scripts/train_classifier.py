"""任务一：训练轻量 CNN 做已知结构三分类。

只用训练集训练、用验证集选择轮次与早停；测试集在本入口里完全不加载，留给最终评价。
运行命令、随机种子、数据划分与关键参数写入 ``<out-dir>/run.json``，权重存 ``best.pt``。

示例（从仓库根目录执行）：

    python scripts/train_classifier.py \
        --windows-csv outputs/sample_windows/windows.csv \
        --arrays-npz outputs/sample_windows/windows.npz \
        --out-dir outputs/classifier/skeleton
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classifier_data import (  # noqa: E402
    CLASS_NAMES,
    assign_splits,
    describe_splits,
    load_dataset,
)
from src.cnn import SmallCNN, compute_class_weights  # noqa: E402
from src.metrics import accuracy, macro_f1, per_class_f1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练已知结构三分类 CNN")
    parser.add_argument(
        "--windows-csv",
        type=Path,
        default=Path("outputs/sample_windows/windows.csv"),
        help="窗口表；正式实验换成 #2 的产物",
    )
    parser.add_argument(
        "--arrays-npz",
        type=Path,
        default=Path("outputs/sample_windows/windows.npz"),
        help="与窗口表行序一致的数组文件",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/classifier/skeleton"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=8, help="验证 Macro-F1 连续无提升的容忍轮数")
    parser.add_argument("--seed", type=int, default=20260918, help="主随机种子")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto / cpu / mps / cuda；正式训练交给组长在 RTX 2080 Ti 上跑",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="关闭类别加权，仅用于对照实验",
    )
    return parser.parse_args()


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def code_version() -> str:
    """记录所用代码版本；工作区有未提交改动时如实标出。"""
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    head = run("rev-parse", "HEAD")
    if not head:
        return "unknown"
    dirty = bool(run("status", "--porcelain"))
    return f"{head}{' (dirty)' if dirty else ''}"


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y))
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()) * len(labels))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())
    total = sum(len(t) for t in targets)
    return (
        sum(losses) / max(total, 1),
        np.concatenate(targets),
        np.concatenate(predictions),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y, meta = load_dataset(args.windows_csv, args.arrays_npz)
    meta, split_notes = assign_splits(meta, seed=args.seed)
    table = describe_splits(meta)
    leaks = meta.groupby("region_id")["split"].nunique()
    if (leaks > 1).any():
        raise SystemExit(
            f"有 {int((leaks > 1).sum())} 个结构的窗口跨了多个集合，划分有误；"
            "同一结构的重复与平移窗口必须留在同一集合"
        )
    if (table.loc[["val", "test"]] == 0).any().any():
        raise SystemExit(
            "验证集或测试集有类别为空，无法进行可靠评价：\n"
            f"{table}\n请检查数据划分或补足样本后再训练"
        )

    print(f"数据 {args.windows_csv}，共 {len(meta)} 个窗口")
    print(f"划分（行=集合，列=类别）\n{table}")
    for note in split_notes:
        print(f"注意：{note}")

    device = pick_device(args.device)
    masks = {name: (meta["split"] == name).to_numpy() for name in ("train", "val")}
    train_loader = make_loader(x[masks["train"]], y[masks["train"]], args.batch_size, True)
    val_loader = make_loader(x[masks["val"]], y[masks["val"]], args.batch_size, False)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(y[masks["train"]], len(CLASS_NAMES)).to(device)
        print("类别权重 " + ", ".join(f"{n}={w:.3f}" for n, w in zip(CLASS_NAMES, class_weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = SmallCNN(n_classes=len(CLASS_NAMES), dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"设备 {device}，模型参数量 {model.count_parameters()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # 保存本次划分，推理与最终评价必须用同一份，不能各自按种子重新划分。
    split_columns = ["window_id", "chrom", "start", "end", "split", "region_id", "label_name"]
    split_columns += [c for c in ("sample_id", "condition", "replicate", "structure_id") if c in meta.columns]
    meta[split_columns].to_csv(args.out_dir / "splits.csv", index=False, encoding="utf-8")
    log_path = args.out_dir / "train_log.txt"
    best_score = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict] = []
    started = time.time()

    with log_path.open("w", encoding="utf-8") as log:
        def emit(line: str) -> None:
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        emit(f"# 训练开始 {time.strftime('%Y-%m-%d %H:%M:%S')}")
        emit(f"# 数据 {args.windows_csv} / {args.arrays_npz}")
        emit(f"# 划分\n{table.to_string()}")
        for note in split_notes:
            emit(f"# 注意：{note}")

        for epoch in range(1, args.epochs + 1):
            train_loss, train_true, train_pred = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val_loss, val_true, val_pred = run_epoch(model, val_loader, criterion, device)
            val_macro_f1 = macro_f1(val_true, val_pred, len(CLASS_NAMES))
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(accuracy(train_true, train_pred), 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(accuracy(val_true, val_pred), 4),
                "val_macro_f1": round(val_macro_f1, 4),
            }
            history.append(record)
            emit(
                f"epoch {epoch:3d} train_loss {record['train_loss']:.4f} "
                f"train_acc {record['train_accuracy']:.3f} val_loss {record['val_loss']:.4f} "
                f"val_acc {record['val_accuracy']:.3f} val_macro_f1 {record['val_macro_f1']:.4f}"
            )

            if val_macro_f1 > best_score:
                best_score = val_macro_f1
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "class_names": list(CLASS_NAMES),
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
                        "dropout": args.dropout,
                        "split_seed": args.seed,
                    },
                    args.out_dir / "best.pt",
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    emit(f"# 验证 Macro-F1 连续 {args.patience} 轮无提升，提前停止")
                    break

        # 逐类 F1 只是训练记录的一部分；完整评价、基线与混淆矩阵图属于 #4。
        best_state = torch.load(args.out_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(best_state["state_dict"])
        _, val_true, val_pred = run_epoch(model, val_loader, criterion, device)
        val_f1_per_class = per_class_f1(val_true, val_pred, len(CLASS_NAMES))
        emit(f"# 最佳轮次 {best_epoch}，验证 Macro-F1 {best_score:.4f}")
        emit(
            "# 验证逐类 F1 "
            + ", ".join(f"{n}={v:.3f}" for n, v in zip(CLASS_NAMES, val_f1_per_class))
        )
        emit(f"# 用时 {time.time() - started:.1f}s")

    run_record = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "code_version": code_version(),
        "seed": args.seed,
        "device": str(device),
        "data": {
            "windows_csv": str(args.windows_csv),
            "arrays_npz": str(args.arrays_npz),
            "windows": int(len(meta)),
            "split_table": table.to_dict(),
            "split_notes": split_notes,
        },
        "model": {
            "name": "SmallCNN",
            "parameters": model.count_parameters(),
            "dropout": args.dropout,
            "class_weighted": not args.no_class_weights,
        },
        "training": {
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(best_score, 4),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seconds": round(time.time() - started, 1),
        },
        "history": history,
        "outputs": {"checkpoint": "best.pt", "log": "train_log.txt", "splits": "splits.csv"},
    }
    (args.out_dir / "run.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"权重与运行记录已写入 {args.out_dir}")
    print("测试集在本入口未加载，留到最终评价（#4）")


if __name__ == "__main__":
    main()
