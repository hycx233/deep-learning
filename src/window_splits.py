"""以环状基因组上的窗口重叠分量划分集合，避免相邻结构和重复之间泄漏。"""

import json
import random
from collections import Counter, defaultdict


SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def window_segments(center, genome_length, window_bp):
    """返回展开的 start/end 和实际半开区间；跨原点窗口拆成两段。"""
    start = (center - window_bp // 2) % genome_length
    end = start + window_bp
    segments = [[start, min(end, genome_length)]]
    if end > genome_length:
        segments.append([0, end - genome_length])
    return start, end, segments


def overlaps(left, right):
    return any(max(a, c) < min(b, d) for a, b in left for c, d in right)


def _bin_segments(segments, genome_length):
    # 实际读取 100 bp bins 时，非对齐窗口会带入两侧边缘 bin。
    return [[start // 100 * 100, min((end + 99) // 100 * 100, genome_length)]
            for start, end in segments]


def _components(positions):
    parents = list(range(len(positions)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, position in enumerate(positions):
        for other_index in range(index):
            if overlaps(position["_read_segments"], positions[other_index]["_read_segments"]):
                parents[find(index)] = find(other_index)
    components = defaultdict(list)
    for index in range(len(positions)):
        components[find(index)].append(index)
    return list(components.values())


def _assign_components(positions, components, seed):
    """较大的分量先分配，平衡总样本量及各类注释的 70/15/15 比例。"""
    totals = Counter(position["type"] for position in positions)
    counts = {split: Counter() for split in SPLIT_RATIOS}
    order = list(range(len(components)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda index: len(components[index]), reverse=True)

    def score():
        return sum(
            (
                sum((counts[split][kind] / total - ratio) ** 2
                    for kind, total in totals.items())
                + (sum(counts[split].values()) / len(positions) - ratio) ** 2
            ) / ratio
            for split, ratio in SPLIT_RATIOS.items()
        )

    for component_index in order:
        component = components[component_index]
        additions = Counter(positions[index]["type"] for index in component)
        candidates = []
        for split in SPLIT_RATIOS:
            counts[split].update(additions)
            candidates.append((score(), split))
            counts[split].subtract(additions)
        split = min(candidates)[1]
        counts[split].update(additions)
        for index in component:
            positions[index]["group_id"] = f"group_{component_index + 1:04d}"
            positions[index]["split"] = split


def validate_manifest(positions, genome_length=None):
    """可独立用于检查导出表，触边不算重叠，首尾两段都参与判断。"""
    seen = {}
    segments = [[[segment["start"], segment["end"]]
                 for segment in json.loads(position["segments"])]
                for position in positions]
    if genome_length is not None:
        segments = [_bin_segments(segment, genome_length) for segment in segments]
    for index, position in enumerate(positions):
        assignment = (position["group_id"], position["split"])
        previous = seen.setdefault(position["position_id"], assignment)
        if previous != assignment:
            raise ValueError(f"同一位置有不同分组: {position['position_id']}")
        for other_index in range(index):
            other = positions[other_index]
            if position["split"] != other["split"] and overlaps(
                segments[index], segments[other_index]
            ):
                raise ValueError(
                    f"跨集合窗口重叠: {position['position_id']} / {other['position_id']}"
                )


def build_window_manifest(
    structures: list[dict], genome_length=4_641_652, window_bp=24_000,
    seed=20260918, background_count=24,
) -> tuple[list[dict], dict]:
    if not 0 < window_bp <= genome_length or background_count < 0:
        raise ValueError("窗口长度应在 (0, genome_length] 内，背景数量不能为负")
    if not structures:
        raise ValueError("需要真实结构标注来划分窗口")
    positions = []
    known_intervals = []
    for structure in structures:
        center = int(structure["center"])
        source_start, source_end = int(structure["start"]), int(structure["end"])
        if not 0 <= source_start <= center < source_end <= genome_length:
            raise ValueError(f"结构坐标超出范围: {structure['ID']}")
        start, end, segments = window_segments(center, genome_length, window_bp)
        known_intervals.append([source_start, source_end])
        positions.append({
            "position_id": f"pos_{start:07d}_{window_bp}",
            "structure_id": structure["ID"], "type": structure["type"],
            "chrom": structure["chrom"], "center": center,
            "start": start, "end": end, "_segments": segments,
            "_read_segments": _bin_segments(segments, genome_length),
        })
    if len({position["chrom"] for position in positions}) != 1:
        raise ValueError("当前课程数据只支持一条环状染色体")
    known_count = len(positions)
    components = _components(positions)
    _assign_components(positions, components, seed)

    # 背景不参与已有分量划分，也不允许它桥接两个已有组。
    rng = random.Random(seed + 1)
    background_targets = {
        "train": round(background_count * 0.70),
        "val": round(background_count * 0.15),
    }
    background_targets["test"] = background_count - sum(background_targets.values())
    background_counts = Counter()
    rejected = Counter()
    attempts = 0
    max_attempts = max(1000, background_count * 1000)
    while sum(background_counts.values()) < background_count and attempts < max_attempts:
        attempts += 1
        center = rng.randrange(genome_length)
        start, end, segments = window_segments(center, genome_length, window_bp)
        read_segments = _bin_segments(segments, genome_length)
        if overlaps(read_segments, known_intervals):
            rejected["overlaps_known_structure"] += 1
            continue
        neighbors = [position for position in positions
                     if overlaps(read_segments, position["_read_segments"])]
        groups = {position["group_id"] for position in neighbors}
        if len(groups) > 1:
            rejected["bridges_existing_groups"] += 1
            continue
        available = [split for split in SPLIT_RATIOS
                     if background_counts[split] < background_targets[split]]
        if neighbors:
            split = neighbors[0]["split"]
            if split not in available:
                rejected["split_background_quota_filled"] += 1
                continue
            group_id = neighbors[0]["group_id"]
        else:
            split = max(available, key=lambda item:
                        background_targets[item] - background_counts[item])
            group_id = f"background_{sum(background_counts.values()) + 1:04d}"
        positions.append({
            "position_id": f"pos_{start:07d}_{window_bp}",
            "structure_id": "", "type": "background", "chrom": positions[0]["chrom"],
            "center": center, "start": start, "end": end, "_segments": segments,
            "_read_segments": read_segments,
            "group_id": group_id, "split": split,
        })
        background_counts[split] += 1

    for position in positions:
        position["segments"] = json.dumps(
            [{"start": start, "end": end} for start, end in position.pop("_segments")],
            separators=(",", ":"),
        )
        position.pop("_read_segments")
    validate_manifest(positions, genome_length)
    known = positions[:known_count]
    summary = {
        "strategy": "overlap_connected_components", "seed": seed,
        "genome_length": genome_length, "window_bp": window_bp,
        "overlap_bin_size": 100,
        "target_ratios": SPLIT_RATIOS.copy(), "known_structures": known_count,
        "component_count": len(components),
        "largest_component": max(map(len, components)),
        "known_counts": {
            split: dict(Counter(position["type"] for position in known
                                if position["split"] == split))
            for split in SPLIT_RATIOS
        },
        "known_split_totals": dict(Counter(position["split"] for position in known)),
        "known_excluded": [],
        "background_requested": background_count,
        "background_generated": sum(background_counts.values()),
        "background_counts": {split: background_counts[split] for split in SPLIT_RATIOS},
        "background_attempts": attempts, "background_rejections": dict(rejected),
        "background_usage": "discovery_and_preview_only",
        "cross_split_overlap_pairs": 0,
    }
    return positions, summary
