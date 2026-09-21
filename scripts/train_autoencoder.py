"""训练任务二的小型卷积自编码器。"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.autoencoder import ConvAutoencoder, masked_mse, upper_triangle_mask  # noqa: E402
from src.discovery_data import (  # noqa: E402
    load_discovery_dataset,
    select_background_rows,
    split_background_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 WT rep1 背景窗口卷积自编码器")
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--arrays-npz", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/discovery/autoencoder")
    )
    parser.add_argument("--sample-id", default="WT_rep1")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--diagonal-exclusion", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
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
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    head = run("rev-parse", "HEAD")
    if not head:
        return "unknown"
    return f"{head}{' (dirty)' if run('status', '--porcelain') else ''}"


def make_loader(
    matrices: np.ndarray,
    masks: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensors = torch.from_numpy(matrices[rows]).unsqueeze(1)
    mask_tensors = torch.from_numpy(masks[rows])
    return DataLoader(
        TensorDataset(tensors, mask_tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def run_epoch(
    model: ConvAutoencoder,
    loader: DataLoader,
    mask: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    weighted_loss = 0.0
    samples = 0
    with torch.set_grad_enabled(training):
        for features, sample_mask in loader:
            features = features.to(device)
            sample_mask = sample_mask.to(device)
            reconstruction = model(features)
            loss = masked_mse(reconstruction, features, sample_mask & mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            weighted_loss += float(loss.item()) * len(features)
            samples += len(features)
    if samples == 0:
        raise RuntimeError("数据加载器为空")
    return weighted_loss / samples


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise SystemExit("epochs、batch-size 和 patience 必须为正整数")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    matrices, masks, meta, _ = load_discovery_dataset(
        args.windows_csv, args.arrays_npz, require_intensity=False
    )
    background_rows = select_background_rows(meta, args.sample_id)
    train_rows, val_rows = split_background_rows(meta, background_rows)
    device = pick_device(args.device)
    model = ConvAutoencoder(args.latent_dim).to(device)
    mask = upper_triangle_mask(
        matrices.shape[-1], args.diagonal_exclusion, device=device
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_loader = make_loader(matrices, masks, train_rows, args.batch_size, True)
    val_loader = make_loader(matrices, masks, val_rows, args.batch_size, False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = meta.iloc[np.concatenate([train_rows, val_rows])].copy()
    selected.to_csv(args.out_dir / "training_windows.csv", index=False, encoding="utf-8")

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.time()
    log_path = args.out_dir / "train_log.csv"

    print(
        f"背景窗口 train={len(train_rows)}, val={len(val_rows)}；"
        f"设备={device}，latent_dim={args.latent_dim}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, mask, device, optimizer)
        val_loss = run_epoch(model, val_loader, mask, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        print(
            f"epoch {epoch:3d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "latent_dim": args.latent_dim,
                    "window_size": int(matrices.shape[-1]),
                    "diagonal_exclusion": args.diagonal_exclusion,
                    "sample_id": args.sample_id,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "seed": args.seed,
                },
                args.out_dir / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"验证损失连续 {args.patience} 轮未改善，提前停止")
                break

    pd.DataFrame(history).to_csv(log_path, index=False, encoding="utf-8")
    record = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "code_version": code_version(),
        "seed": args.seed,
        "device": str(device),
        "data": {
            "windows_csv": str(args.windows_csv),
            "arrays_npz": str(args.arrays_npz),
            "sample_id": args.sample_id,
            "background_windows": int(len(background_rows)),
            "train_windows": int(len(train_rows)),
            "val_windows": int(len(val_rows)),
        },
        "model": {
            "name": "ConvAutoencoder",
            "latent_dim": args.latent_dim,
            "parameters": sum(p.numel() for p in model.parameters()),
            "diagonal_exclusion": args.diagonal_exclusion,
            "loss_mask": "逐窗口有效 mask 与固定上三角 mask 的交集；每个窗口单独按有效像素数求均值",
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seconds": round(time.time() - started, 2),
        },
        "outputs": {
            "checkpoint": "best.pt",
            "history": "train_log.csv",
            "training_windows": "training_windows.csv",
        },
    }
    (args.out_dir / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"最佳轮次 {best_epoch}，val_loss={best_loss:.6f}")
    print(f"权重与运行记录已写入 {args.out_dir}")


if __name__ == "__main__":
    main()
