"""候选聚类的独立位置统计与新簇判定。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    """兼容 CSV 中的布尔值与字符串。"""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def assign_independent_loci(candidates: pd.DataFrame) -> pd.Series:
    """把同一染色体上互相重叠的候选窗口合并为一个独立位置。"""
    required = {"chrom", "start", "end"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"候选表缺少字段：{sorted(missing)}")
    if (candidates["end"] <= candidates["start"]).any():
        raise ValueError("候选区间必须满足 end > start")

    result = pd.Series(index=candidates.index, dtype="object")
    for chrom, group in candidates.sort_values(["chrom", "start", "end"]).groupby(
        "chrom", sort=False
    ):
        locus_number = 0
        current_end: int | None = None
        for index, row in group.iterrows():
            start = int(row["start"])
            end = int(row["end"])
            if current_end is None or start >= current_end:
                locus_number += 1
                current_end = end
            else:
                current_end = max(current_end, end)
            result.loc[index] = f"{chrom}:locus_{locus_number:04d}"
    return result


def assign_spatial_groups(windows: pd.DataFrame) -> pd.Series:
    """按完整窗口的重叠连通分量分组，防止空间重叠跨验证折。"""
    required = {"chrom", "start", "end"}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"窗口表缺少字段：{sorted(missing)}")
    if (windows["end"] <= windows["start"]).any():
        raise ValueError("窗口区间必须满足 end > start")

    groups = pd.Series(index=windows.index, dtype="object")
    group_number = 0
    for chrom, chrom_windows in windows.sort_values(
        ["chrom", "start", "end"]
    ).groupby("chrom", sort=False):
        current_end: int | None = None
        for index, row in chrom_windows.iterrows():
            start = int(row["start"])
            end = int(row["end"])
            if current_end is None or start >= current_end:
                group_number += 1
                current_end = end
            else:
                current_end = max(current_end, end)
            groups.loc[index] = f"{chrom}:group_{group_number:04d}"
    return groups


def select_pure_background_rows(
    scores: pd.DataFrame,
    candidate_rows: set[int],
    known_reference_rows: set[int],
) -> list[int]:
    """返回不属于候选/已知参考且不与任何已知结构重叠的 latent_row。"""
    required = {"latent_row", "known_overlap"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"分数表缺少字段：{sorted(missing)}")
    eligible = set(
        scores.loc[~as_bool(scores["known_overlap"]), "latent_row"].astype(int)
    )
    return sorted(eligible - candidate_rows - known_reference_rows)


def summarize_clusters(members: pd.DataFrame) -> pd.DataFrame:
    """汇总 DBSCAN 非噪声簇并应用课程项目的新簇规则。"""
    required = {
        "cluster",
        "is_candidate",
        "is_known_reference",
        "known_overlap",
        "independent_locus_id",
    }
    missing = required.difference(members.columns)
    if missing:
        raise ValueError(f"成员表缺少字段：{sorted(missing)}")

    rows: list[dict[str, object]] = []
    for cluster, group in members[members["cluster"] != -1].groupby("cluster"):
        candidates = group[as_bool(group["is_candidate"])]
        independent_loci = int(candidates["independent_locus_id"].dropna().nunique())
        known_members = int(as_bool(group["is_known_reference"]).sum())
        overlapping_candidates = int(as_bool(candidates["known_overlap"]).sum())
        is_suspected_new = (
            independent_loci >= 5
            and known_members == 0
            and overlapping_candidates == 0
        )
        rows.append(
            {
                "cluster": int(cluster),
                "members": int(len(group)),
                "candidate_windows": int(len(candidates)),
                "independent_candidate_loci": independent_loci,
                "known_reference_members": known_members,
                "known_overlapping_candidates": overlapping_candidates,
                "suspected_new_cluster": bool(is_suspected_new),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "cluster",
                "members",
                "candidate_windows",
                "independent_candidate_loci",
                "known_reference_members",
                "known_overlapping_candidates",
                "suspected_new_cluster",
            ]
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def nearest_rows(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回每个 query 到最近 reference 的行号与欧氏距离。"""
    if len(reference) == 0:
        raise ValueError("reference 不能为空")
    squared = ((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
    indices = squared.argmin(axis=1)
    return indices, np.sqrt(squared[np.arange(len(query)), indices])
