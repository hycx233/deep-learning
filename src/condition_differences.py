"""在同一标注区间和共同有效像素上比较两个突变条件与 WT。"""

import numpy as np


CONDITIONS = ("WT", "DstpA", "DhnsDstpA")
BIN_SIZE = 100


def _effect_label(rep1_log2_fc, rep2_log2_fc, reference_band):
    if rep1_log2_fc > reference_band and rep2_log2_fc > reference_band:
        return "increase_consistent"
    if rep1_log2_fc < -reference_band and rep2_log2_fc < -reference_band:
        return "decrease_consistent"
    if abs(rep1_log2_fc) <= reference_band and abs(rep2_log2_fc) <= reference_band:
        return "within_reference"
    return "inconsistent_or_borderline"


def compare_structure(structure, windows, samples, min_fold_change=1.25):
    """返回 DstpA、DhnsDstpA 相对 WT 的两条描述性结果，不作显著性检验。

    ROI 沿用原始标注 [start,end)，各 bin 按落在 ROI 内的长度/100 加权。
    只统计上三角非对角接触；六样本共用 intensity_mask 的交集和权重。
    roi_weight_sum 是掩码前的 ROI 配对权重和，valid_weight_fraction 是
    共同有效权重除以该值。OE 辅助指标另用六样本共同形态 mask。
    """
    if not np.isfinite(min_fold_change) or min_fold_change < 1:
        raise ValueError("min_fold_change 必须为不小于 1 的有限数值")
    ids = {}
    for condition in CONDITIONS:
        matched = [sample for sample in samples if sample["condition"] == condition]
        by_replicate = {int(sample["replicate"]): sample["sample_id"] for sample in matched}
        if len(matched) != 2 or set(by_replicate) != {1, 2}:
            raise ValueError(f"{condition}: 需要且只需要 rep1、rep2 两个样本")
        ids[condition] = [by_replicate[1], by_replicate[2]]
    sample_ids = [sample_id for condition in CONDITIONS for sample_id in ids[condition]]
    reference = windows[sample_ids[0]]
    bin_ids = np.asarray(reference["bin_ids"])
    shape = (len(bin_ids), len(bin_ids))
    for sample_id in sample_ids:
        window = windows[sample_id]
        if not np.array_equal(window["bin_ids"], bin_ids):
            raise ValueError(f"{sample_id}: 六样本窗口的 bin_ids 或顺序不一致")
        for name in ("intensity", "intensity_mask", "oe", "mask"):
            if np.asarray(window[name]).shape != shape:
                raise ValueError(f"{sample_id}: {name} 形状与 bin_ids 不一致")
    start, end = int(structure["start"]), int(structure["end"])
    if end <= start:
        raise ValueError(f"{structure['ID']}: 标注区间必须满足 start < end")
    overlap_bp = np.maximum(0, np.minimum(bin_ids * BIN_SIZE + BIN_SIZE, end)
                            - np.maximum(bin_ids * BIN_SIZE, start))
    if int(overlap_bp.sum()) != end - start:
        raise ValueError(f"{structure['ID']}: 共享窗口没有完整覆盖标注区间")
    fractions = overlap_bp / BIN_SIZE
    roi_weights = np.triu(np.outer(fractions, fractions), k=1)
    roi_weight_sum = float(roi_weights.sum())
    common_intensity_mask = np.logical_and.reduce([
        np.asarray(windows[sample_id]["intensity_mask"], dtype=bool) for sample_id in sample_ids
    ])
    weights = roi_weights * common_intensity_mask
    selected = weights > 0
    valid_weight_sum = float(weights.sum())
    valid_fraction = valid_weight_sum / roi_weight_sum if roi_weight_sum > 0 else 0.0
    intensity_means = {
        sample_id: (float(np.dot(np.asarray(windows[sample_id]["intensity"])[selected], weights[selected])
                          / valid_weight_sum) if valid_weight_sum > 0 else float("nan"))
        for sample_id in sample_ids
    }
    common_oe_mask = common_intensity_mask & np.logical_and.reduce([
        np.asarray(windows[sample_id]["mask"], dtype=bool) for sample_id in sample_ids
    ])
    oe_weights = roi_weights * common_oe_mask
    oe_selected = oe_weights > 0
    oe_weight_sum = float(oe_weights.sum())
    wt_rep1, wt_rep2 = [intensity_means[sample_id] for sample_id in ids["WT"]]
    wt_mean = (wt_rep1 + wt_rep2) / 2
    rows = []
    for condition in CONDITIONS[1:]:
        mutant_rep1, mutant_rep2 = [intensity_means[sample_id] for sample_id in ids[condition]]
        mutant_mean = (mutant_rep1 + mutant_rep2) / 2
        means = np.array([wt_rep1, wt_rep2, mutant_rep1, mutant_rep2])
        status = "ok"
        if valid_weight_sum <= 0:
            status = "no_common_valid_pairs"
        elif not np.isfinite(means).all():
            status = "nonfinite_intensity"
        elif np.any(means <= 0):
            status = "nonpositive_intensity"
        log2_fc = rep1_log2_fc = rep2_log2_fc = wt_ratio = reference_band = float("nan")
        direction_agreement = False
        effect_label = "not_evaluable"
        if status == "ok":
            log2_fc = float(np.log2(mutant_mean / wt_mean))
            rep1_log2_fc = float(np.log2(mutant_rep1 / wt_mean))
            rep2_log2_fc = float(np.log2(mutant_rep2 / wt_mean))
            wt_ratio = float(np.log2(wt_rep2 / wt_rep1))
            reference_band = max(float(np.log2(min_fold_change)), abs(wt_ratio))
            direction_agreement = bool(np.sign(rep1_log2_fc) == np.sign(rep2_log2_fc))
            effect_label = _effect_label(rep1_log2_fc, rep2_log2_fc, reference_band)
        oe_difference = float("nan")
        if oe_weight_sum > 0:
            wt_oe = sum(np.asarray(windows[sample_id]["oe"], dtype=np.float64)[oe_selected]
                        for sample_id in ids["WT"]) / 2
            mutant_oe = sum(np.asarray(windows[sample_id]["oe"], dtype=np.float64)[oe_selected]
                            for sample_id in ids[condition]) / 2
            difference = np.abs(np.log1p(mutant_oe) - np.log1p(wt_oe))
            oe_difference = float(np.dot(difference, oe_weights[oe_selected]) / oe_weight_sum)
        rows.append({
            "structure_id": structure["ID"], "type": structure["type"], "chrom": structure["chrom"],
            "start": start, "end": end, "center": int(structure["center"]),
            "length_bp": int(structure["length_bp"]), "condition": condition,
            "wt_rep1": wt_rep1, "wt_rep2": wt_rep2, "wt_mean": wt_mean,
            "mutant_rep1": mutant_rep1, "mutant_rep2": mutant_rep2, "mutant_mean": mutant_mean,
            "log2_fc": log2_fc, "rep1_log2_fc": rep1_log2_fc, "rep2_log2_fc": rep2_log2_fc,
            "wt_rep_log2_ratio": wt_ratio, "reference_band": reference_band,
            "replicate_direction_agreement": direction_agreement, "effect_label": effect_label,
            "valid_pair_count": int(selected.sum()), "valid_weight_fraction": valid_fraction,
            "roi_weight_sum": roi_weight_sum, "oe_mean_abs_difference": oe_difference,
            "status": status,
        })
    return rows
