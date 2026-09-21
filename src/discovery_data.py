"""任务二的窗口输入校验与背景样本选择。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SIZE = 128
REQUIRED_COLUMNS = (
    "window_id",
    "chrom",
    "start",
    "end",
    "sample_id",
    "condition",
    "replicate",
)


class DiscoveryDataError(ValueError):
    """发现模块输入不符合接口约定。"""


def _parse_bool_series(values: pd.Series, column: str) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise DiscoveryDataError(f"{column} 列含无法识别的布尔值：{unknown}")
    return normalized.map(mapping).astype(bool)


def load_discovery_dataset(
    windows_csv: str | Path,
    arrays_npz: str | Path,
    *,
    require_intensity: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray | None]:
    """读取 #2 产出的形态窗口、逐窗口有效掩码、元数据和强度分数。"""
    windows_path = Path(windows_csv)
    arrays_path = Path(arrays_npz)
    if not windows_path.is_file():
        raise FileNotFoundError(f"找不到窗口表：{windows_path}")
    if not arrays_path.is_file():
        raise FileNotFoundError(f"找不到窗口数组：{arrays_path}")

    meta = pd.read_csv(windows_path)
    missing = [name for name in REQUIRED_COLUMNS if name not in meta.columns]
    if missing:
        raise DiscoveryDataError(f"窗口表缺少列 {missing}；实际列为 {list(meta.columns)}")
    if meta["window_id"].duplicated().any():
        duplicates = meta.loc[meta["window_id"].duplicated(), "window_id"].head().tolist()
        raise DiscoveryDataError(f"window_id 必须唯一，重复示例：{duplicates}")

    with np.load(arrays_path) as arrays:
        if "X" not in arrays:
            raise DiscoveryDataError(f"数组文件缺少 X；实际键为 {list(arrays.files)}")
        matrices = np.asarray(arrays["X"], dtype=np.float32)
        masks = (
            np.asarray(arrays["mask"], dtype=bool)
            if "mask" in arrays
            else np.ones_like(matrices, dtype=bool)
        )
        intensity = (
            np.asarray(arrays["intensity_scores"], dtype=np.float64)
            if "intensity_scores" in arrays
            else None
        )

    if matrices.ndim != 3 or matrices.shape[1:] != (WINDOW_SIZE, WINDOW_SIZE):
        raise DiscoveryDataError(
            f"X 应为 N×{WINDOW_SIZE}×{WINDOW_SIZE}，实际为 {matrices.shape}"
        )
    if len(matrices) != len(meta):
        raise DiscoveryDataError(
            f"X 有 {len(matrices)} 个窗口，窗口表有 {len(meta)} 行，两者必须一致"
        )
    if masks.shape != matrices.shape:
        raise DiscoveryDataError(
            f"mask 应与 X 形状一致，实际为 {masks.shape} 和 {matrices.shape}"
        )
    if (~masks.reshape(len(masks), -1).any(axis=1)).any():
        bad_rows = np.flatnonzero(~masks.reshape(len(masks), -1).any(axis=1))[:5].tolist()
        raise DiscoveryDataError(f"mask 存在完全无效的窗口，行号示例：{bad_rows}")
    bad = int(np.count_nonzero(~np.isfinite(matrices)))
    if bad:
        raise DiscoveryDataError(f"X 含 {bad} 个 NaN/Inf，请回查 #2 的归一化与无效像素处理")

    if intensity is None and "intensity_score" in meta.columns:
        intensity = pd.to_numeric(meta["intensity_score"], errors="raise").to_numpy(
            dtype=np.float64
        )
    if intensity is not None:
        if intensity.shape != (len(meta),):
            raise DiscoveryDataError(
                f"intensity_scores 应为长度 {len(meta)} 的一维数组，实际为 {intensity.shape}"
            )
        if not np.isfinite(intensity).all():
            raise DiscoveryDataError("intensity_scores 含 NaN/Inf")
    elif require_intensity:
        raise DiscoveryDataError(
            "缺少可比较的强度分数：请在 NPZ 中提供 intensity_scores，或在窗口表中提供 "
            "intensity_score。该分数应来自 #2 的深度归一化 counts，不能用逐窗口缩放值代替。"
        )

    meta = meta.copy()
    if "known_overlap" in meta.columns:
        meta["known_overlap"] = _parse_bool_series(meta["known_overlap"], "known_overlap")
    return matrices, masks, meta, intensity


def select_background_rows(meta: pd.DataFrame, sample_id: str = "WT_rep1") -> np.ndarray:
    """选择 WT rep1 中不与已知结构重叠的背景训练窗口。"""
    sample_mask = meta["sample_id"].astype(str).eq(sample_id)
    if "known_overlap" in meta.columns:
        background_mask = ~meta["known_overlap"].astype(bool)
    elif "structure_id" in meta.columns:
        background_mask = meta["structure_id"].isna() | meta["structure_id"].astype(str).eq("")
    else:
        raise DiscoveryDataError(
            "窗口表需要 known_overlap 布尔列或可为空的 structure_id 列，"
            "否则无法确认自编码器只使用排除已知结构后的背景窗口"
        )
    rows = np.flatnonzero((sample_mask & background_mask).to_numpy())
    if len(rows) == 0:
        raise DiscoveryDataError(f"没有找到 sample_id={sample_id!r} 的背景窗口")
    return rows


def split_background_rows(
    meta: pd.DataFrame,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """严格沿用 #2 的 train/val 标记，不在模型脚本里重新随机划分。"""
    if "split" not in meta.columns:
        raise DiscoveryDataError(
            "窗口表缺少 split 列；正式训练必须沿用 #2 的空间分组，"
            "不能在自编码器脚本里随机拆分相互重叠的滑窗"
        )
    splits = meta.iloc[rows]["split"].astype(str).to_numpy()
    unknown = sorted(set(splits) - {"train", "val", "test"})
    if unknown:
        raise DiscoveryDataError(f"split 列含未登记取值：{unknown}")
    train_rows = rows[splits == "train"]
    val_rows = rows[splits == "val"]
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise DiscoveryDataError(
            f"背景窗口需要同时有 train 和 val；实际 train={len(train_rows)}, val={len(val_rows)}"
        )
    return train_rows, val_rows
