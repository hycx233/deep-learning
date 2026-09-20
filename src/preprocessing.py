"""共享的 100 bp 工作数据、全基因组归一化与环状窗口读取。"""

import json
from pathlib import Path
import tempfile

import cooler
import numpy as np


BIN_SIZE = 100
MODEL_OE_CLIP = 10.0
CACHE_VERSION = 1


def _valid_distance_pairs(valid_bins):
    """每个环状 bin 距离的无序有效配对数，包含没有记录的零接触对。"""
    n_bins = len(valid_bins)
    spectrum = np.fft.rfft(valid_bins.astype(np.float64))
    correlations = np.fft.irfft(spectrum * spectrum.conj(), n=n_bins)
    pairs = np.rint(correlations[: n_bins // 2 + 1]).astype(np.int64)
    pairs = np.maximum(pairs, 0)
    if n_bins % 2 == 0:
        pairs[-1] //= 2
    return pairs


def prepare_sample(sample, data_root, cache_dir, chunksize=1_000_000):
    """流式聚合一个样本；缓存身份一致时复用，否则重新生成。"""
    if chunksize <= 0:
        raise ValueError("chunksize 必须大于零")
    data_root, cache_dir = Path(data_root), Path(cache_dir)
    source = (data_root / sample["path"]).resolve()
    source_stat = source.stat()
    sample_id = sample["sample_id"]
    paths = {kind: cache_dir / f"{sample_id}.{kind}" for kind in ("cool", "npz", "json")}
    identity = {
        "cache_version": CACHE_VERSION, "source_path": str(source),
        "source_size": source_stat.st_size, "source_mtime_ns": source_stat.st_mtime_ns,
        "bin_size": BIN_SIZE, "chunksize": chunksize,
    }
    if all(path.is_file() for path in paths.values()):
        metadata = json.loads(paths["json"].read_text(encoding="utf-8"))
        if metadata["cache_identity"] == identity:
            return metadata

    source_map = cooler.Cooler(str(source))
    if source_map.storage_mode != "symmetric-upper" or len(source_map.chromnames) != 1:
        raise ValueError(f"{source}: 需要单染色体 symmetric-upper Cooler")
    if source_map.binsize is None or BIN_SIZE % source_map.binsize:
        raise ValueError(f"{source}: 无法把 {source_map.binsize} bp 整数聚合到 {BIN_SIZE} bp")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 成功创建后再替换，避免把中断的聚合文件误当作完整缓存。
    with tempfile.TemporaryDirectory(prefix=f"{sample_id}-", dir=cache_dir) as temporary:
        coarse_path = Path(temporary) / "coarse.cool"
        cooler.coarsen_cooler(
            str(source), str(coarse_path), factor=BIN_SIZE // source_map.binsize,
            chunksize=chunksize, nproc=1, dtypes={"count": np.int64},
        )
        contact_map = cooler.Cooler(str(coarse_path))
        bins = contact_map.bins()[:]
        n_bins = len(bins)
        bin_widths = (bins["end"] - bins["start"]).to_numpy(dtype=np.int64)
        n_pixels = contact_map.info["nnz"]
        marginal = np.zeros(n_bins, dtype=np.int64)
        total_counts = 0
        pixels = contact_map.pixels()
        for offset in range(0, n_pixels, chunksize):
            chunk = pixels[offset : offset + chunksize]
            left = chunk["bin1_id"].to_numpy()
            right = chunk["bin2_id"].to_numpy()
            counts = chunk["count"].to_numpy(dtype=np.int64)
            total_counts += int(counts.sum())
            # 对称矩阵行和：对角一次，非对角分别累加到两个端点。
            marginal += np.bincount(left, weights=counts, minlength=n_bins).astype(np.int64)
            off_diagonal = left != right
            marginal += np.bincount(right[off_diagonal], weights=counts[off_diagonal],
                                    minlength=n_bins).astype(np.int64)
        if total_counts <= 0:
            raise ValueError(f"{source}: 总接触计数必须大于零")
        valid_bins = (marginal > 0) & (bin_widths == BIN_SIZE)
        pair_counts = _valid_distance_pairs(valid_bins)
        distance_sums = np.zeros(len(pair_counts), dtype=np.float64)
        for offset in range(0, n_pixels, chunksize):
            chunk = pixels[offset : offset + chunksize]
            left = chunk["bin1_id"].to_numpy()
            right = chunk["bin2_id"].to_numpy()
            keep = valid_bins[left] & valid_bins[right]
            separation = right[keep] - left[keep]
            distance = np.minimum(separation, n_bins - separation)
            distance_sums += np.bincount(distance, weights=chunk["count"].to_numpy()[keep],
                                         minlength=len(pair_counts))
        expected_raw = np.divide(distance_sums, pair_counts,
                                 out=np.zeros_like(distance_sums), where=pair_counts > 0)
        depth_factor = 1_000_000.0 / total_counts
        metadata = {
            "sample_id": sample_id, "condition": sample["condition"],
            "replicate": int(sample["replicate"]), "chrom": sample["chrom"],
            "source_chrom": contact_map.chromnames[0],
            "genome_length": int(contact_map.chromsizes.iloc[0]),
            "bin_size": BIN_SIZE, "n_bins": n_bins, "n_pixels": int(n_pixels),
            "total_counts": total_counts, "depth_factor": depth_factor,
            "valid_bin_count": int(valid_bins.sum()), "invalid_bin_count": int((~valid_bins).sum()),
            "partial_bin_count": int((bin_widths != BIN_SIZE).sum()),
            "zero_expected_distance_count": int((expected_raw <= 0).sum()),
            "distance_method": "min(abs(bin1-bin2), n_bins-abs(bin1-bin2))*100 bp; circular approximation",
            "valid_bin_rule": "positive marginal and full 100 bp width; final partial bin excluded",
            "marginal_rule": "symmetric row sum; diagonal once, off-diagonal once per endpoint",
            "expected_rule": "raw upper-triangle counts / all unordered valid bin pairs at distance, including zeros",
            "model_rule": "clip(log1p(O/E),0,log(11))/log(11); mask diagonal and zero expected",
            "cache_identity": identity,
            "versions": {"numpy": np.__version__, "cooler": cooler.__version__},
            "cool_path": str(paths["cool"].resolve()),
            "stats_path": str(paths["npz"].resolve()),
            "metadata_path": str(paths["json"].resolve()),
        }
        np.savez_compressed(
            Path(temporary) / "stats.npz", expected_raw=expected_raw, valid_bins=valid_bins,
            marginal=marginal, bin_widths=bin_widths, distance_pair_counts=pair_counts,
            distance_count_sums=distance_sums, depth_factor=depth_factor,
        )
        coarse_path.replace(paths["cool"])
        (Path(temporary) / "stats.npz").replace(paths["npz"])
        paths["json"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_sample(cache_dir, sample_id):
    """加载一份缓存；返回普通字典，供分类、发现、差异分析与绘图共用。"""
    cache_dir = Path(cache_dir)
    metadata = json.loads((cache_dir / f"{sample_id}.json").read_text(encoding="utf-8"))
    with np.load(cache_dir / f"{sample_id}.npz") as arrays:
        state = {name: arrays[name] for name in arrays.files}
    state["metadata"] = metadata
    state["cooler"] = cooler.Cooler(str(cache_dir / f"{sample_id}.cool"))
    return state


def extract_window(state, center, window_bp=24_000, image_size=128):
    """读取真实环状坐标窗口，保留原强度/OE及按有效面积缩放的模型输入。"""
    contact_map = state["cooler"]
    metadata = state["metadata"]
    genome_length = metadata["genome_length"]
    if not 0 <= center < genome_length or int(center) != center:
        raise ValueError(f"center 必须为 [0, {genome_length}) 内的整数坐标")
    if not 0 < window_bp < genome_length or int(window_bp) != window_bp or image_size <= 0:
        raise ValueError("window_bp 必须为小于基因组长度的正整数，image_size 必须大于零")
    center, window_bp = int(center), int(window_bp)
    start = (center - window_bp // 2) % genome_length
    stop = start + window_bp
    segments = [{"start": start, "end": min(stop, genome_length)}]
    if stop > genome_length:
        segments.append({"start": 0, "end": stop - genome_length})
    ranges = []
    bin_ids = []
    local_starts = []
    local_ends = []
    circular_offset = 0
    for segment in segments:
        first, last = contact_map.extent((metadata["source_chrom"], segment["start"], segment["end"]))
        indices = np.arange(first, last)
        ranges.append((first, last))
        bin_ids.append(indices)
        genomic_starts = indices * BIN_SIZE
        genomic_ends = genomic_starts + state["bin_widths"][indices]
        local_starts.append(np.maximum(genomic_starts, segment["start"]) - start + circular_offset)
        local_ends.append(np.minimum(genomic_ends, segment["end"]) - start + circular_offset)
        circular_offset += genome_length
    bin_ids = np.concatenate(bin_ids)
    local_starts = np.concatenate(local_starts)
    local_ends = np.concatenate(local_ends)
    # 两段时拼四个块，首尾之间的非对角接触不会丢失。
    selector = contact_map.matrix(balance=False, sparse=True)
    counts = np.block([
        [selector[left_start:left_end, right_start:right_end].toarray()
         for right_start, right_end in ranges]
        for left_start, left_end in ranges
    ]).astype(np.float64)
    valid = state["valid_bins"][bin_ids]
    intensity_mask = valid[:, None] & valid[None, :]
    intensity = np.where(intensity_mask, counts * float(state["depth_factor"]), 0.0)
    separation = np.abs(bin_ids[:, None] - bin_ids[None, :])
    distance = np.minimum(separation, metadata["n_bins"] - separation)
    expected = state["expected_raw"][distance]
    mask = intensity_mask & (expected > 0) & (bin_ids[:, None] != bin_ids[None, :])
    oe = np.divide(counts, expected, out=np.zeros_like(counts), where=mask)
    morphology = np.log1p(np.minimum(oe, MODEL_OE_CLIP)) / np.log1p(MODEL_OE_CLIP)
    target_edges = np.linspace(0, window_bp, image_size + 1)
    overlap = np.maximum(0.0, np.minimum(target_edges[1:, None], local_ends[None, :])
                         - np.maximum(target_edges[:-1, None], local_starts[None, :]))
    valid_area = overlap @ mask.astype(np.float64) @ overlap.T
    model = np.divide(overlap @ morphology @ overlap.T, valid_area,
                      out=np.zeros((image_size, image_size)), where=valid_area > 0)
    return {
        "counts": counts.astype(np.int64),
        "intensity": intensity.astype(np.float32), "intensity_mask": intensity_mask,
        "oe": oe.astype(np.float32), "mask": mask,
        "model": model.astype(np.float32), "model_mask": valid_area > 0,
        "model_valid_fraction": (valid_area / (window_bp / image_size) ** 2).astype(np.float32),
        "bin_ids": bin_ids, "window_edges": np.r_[local_starts, local_ends[-1]],
        "center": center, "start": start, "end": stop,
        "segments": segments, "chrom": metadata["chrom"], "sample_id": metadata["sample_id"],
    }
