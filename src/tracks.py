"""任务三的接触强度轨道计算与区间工具。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def genome_segments(genome_length: int, segment_bp: int = 10_000) -> list[tuple[int, int]]:
    if genome_length <= 0 or segment_bp <= 0:
        raise ValueError("genome_length 和 segment_bp 必须为正整数")
    return [
        (start, min(start + segment_bp, genome_length))
        for start in range(0, genome_length, segment_bp)
    ]


def add_symmetric_upper_counts(
    signal: np.ndarray,
    bin1: np.ndarray,
    bin2: np.ndarray,
    counts: np.ndarray,
) -> None:
    """把 symmetric-upper 像素转为每个 bin 的有效接触总量。

    非对角像素分别累加到两个端点；对角像素只累加一次。
    """
    if not (len(bin1) == len(bin2) == len(counts)):
        raise ValueError("bin1、bin2 与 counts 长度必须一致")
    n_bins = len(signal)
    if len(bin1) == 0:
        return
    if bin1.min() < 0 or bin2.min() < 0 or bin1.max() >= n_bins or bin2.max() >= n_bins:
        raise ValueError("像素 bin 编号超出 signal 范围")
    signal += np.bincount(bin1, weights=counts, minlength=n_bins)
    off_diagonal = bin1 != bin2
    signal += np.bincount(
        bin2[off_diagonal], weights=counts[off_diagonal], minlength=n_bins
    )


def aggregate_signal(
    native_signal: np.ndarray,
    native_bin_bp: int,
    target_bin_bp: int,
    genome_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_bin_bp % native_bin_bp:
        raise ValueError(
            f"target_bin_bp={target_bin_bp} 必须是 native_bin_bp={native_bin_bp} 的整数倍"
        )
    expected_native_bins = int(np.ceil(genome_length / native_bin_bp))
    if len(native_signal) != expected_native_bins:
        raise ValueError(
            f"native_signal 长度应为 {expected_native_bins}，实际为 {len(native_signal)}"
        )
    target_index = (np.arange(len(native_signal)) * native_bin_bp) // target_bin_bp
    n_target = int(target_index[-1]) + 1
    aggregated = np.bincount(
        target_index, weights=native_signal, minlength=n_target
    ).astype(np.float64)
    starts = np.arange(n_target, dtype=np.int64) * target_bin_bp
    ends = np.minimum(starts + target_bin_bp, genome_length)
    return starts, ends, aggregated


def normalize_per_sample(signal: np.ndarray, scale: float = 1_000_000.0) -> tuple[np.ndarray, float]:
    total = float(np.sum(signal, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"样本有效接触总量必须为正数，实际为 {total}")
    return signal.astype(np.float64) / total * scale, total


def condition_mean_signals(
    signals: np.ndarray,
    conditions: np.ndarray,
    condition_order: list[str],
) -> dict[str, np.ndarray]:
    if signals.ndim != 2 or len(signals) != len(conditions):
        raise ValueError("signals 应为 samples×bins，且样本数与 conditions 一致")
    result: dict[str, np.ndarray] = {}
    conditions = conditions.astype(str)
    for condition in condition_order:
        mask = conditions == condition
        if not mask.any():
            raise ValueError(f"条件 {condition!r} 没有样本")
        result[condition] = signals[mask].mean(axis=0)
    return result


def assign_lanes(intervals: pd.DataFrame) -> np.ndarray:
    """用贪心方式给相互重叠的基因/结构分配显示轨道。"""
    if intervals.empty:
        return np.empty(0, dtype=np.int64)
    if not {"start", "end"} <= set(intervals.columns):
        raise ValueError("区间表需要 start 和 end 列")
    order = np.argsort(intervals["start"].to_numpy(), kind="stable")
    lane_ends: list[int] = []
    lanes = np.empty(len(intervals), dtype=np.int64)
    for index in order:
        start = int(intervals.iloc[index]["start"])
        end = int(intervals.iloc[index]["end"])
        lane = next((i for i, lane_end in enumerate(lane_ends) if start >= lane_end), None)
        if lane is None:
            lane = len(lane_ends)
            lane_ends.append(end)
        else:
            lane_ends[lane] = end
        lanes[index] = lane
    return lanes
