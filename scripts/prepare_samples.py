#!/usr/bin/env python3
"""选择性解压四份突变样本，核对六份 Cooler 元数据并生成样本清单。"""

import argparse
import csv
import gzip
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

import h5py


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = "GSE272161_RAW.tar"
MUTANT_SAMPLES = (
    ("DstpA", 1, "GSM8950761_DstpA_rep1.MG1655.mapq_30.10.cool.gz"),
    ("DstpA", 2, "GSM8950762_DstpA_rep2.MG1655.mapq_30.10.cool.gz"),
    ("DhnsDstpA", 1, "GSM8950763_DhnsDstpA_rep1.MG1655.mapq_30.10.cool.gz"),
    ("DhnsDstpA", 2, "GSM8950764_DhnsDstpA_rep2.MG1655.mapq_30.10.cool.gz"),
)
CHROM_ALIASES = {"MG1655": "NC_000913.3", "NC_000913.3": "NC_000913.3"}


def inspect_cool(path):
    """只读取 HDF5 元数据及首尾 bin；不载入接触像素或稠密矩阵。"""
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format") != "HDF5::Cooler":
            raise ValueError(f"{path}: 不是 Cooler 文件")
        names = handle["chroms/name"].asstr()[:]
        if len(names) != 1 or names[0] not in CHROM_ALIASES:
            raise ValueError(f"{path}: 预期单条 MG1655/NC_000913.3 染色体，实际 {names}")
        source_chrom = names[0]
        genome_length = int(handle["chroms/length"][0])
        bin_size = int(handle.attrs["bin-size"])
        n_bins = len(handle["bins/start"])
        if bin_size <= 0 or n_bins != (genome_length + bin_size - 1) // bin_size:
            raise ValueError(f"{path}: bin 数量与染色体长度或分辨率不匹配")
        if int(handle["bins/start"][0]) != 0 or int(handle["bins/end"][-1]) != genome_length:
            raise ValueError(f"{path}: 首尾 bin 未覆盖完整染色体")
        n_pixels = len(handle["pixels/count"])
        if any(len(handle[f"pixels/{name}"]) != n_pixels for name in ("bin1_id", "bin2_id")):
            raise ValueError(f"{path}: pixels 三列长度不一致")
        return {
            "chrom": CHROM_ALIASES[source_chrom],
            "source_chrom": source_chrom,
            "bin_size": bin_size,
            "genome_length": genome_length,
            "n_bins": n_bins,
            "n_pixels": n_pixels,
            "count_dtype": str(handle["pixels/count"].dtype),
            "has_weight": str("weight" in handle["bins"]).lower(),
            "storage_mode": handle.attrs.get("storage-mode", "symmetric-upper"),
            "file_size_bytes": path.stat().st_size,
        }


def extract_mutants(archive_path, raw_dir):
    """只流式解压四个指定成员，已有文件交由后续元数据检查复用。"""
    missing = [member for _, _, member in MUTANT_SAMPLES if not (raw_dir / member[:-3]).exists()]
    if not missing:
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:") as archive:
        for member_name in missing:
            member = archive.getmember(member_name)
            if not member.isfile():
                raise ValueError(f"{archive_path}: {member_name} 不是普通文件")
            destination = raw_dir / member_name[:-3]
            started = time.monotonic()
            with tempfile.NamedTemporaryFile(dir=raw_dir, suffix=".partial", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                try:
                    with archive.extractfile(member) as compressed:
                        with gzip.GzipFile(fileobj=compressed, mode="rb") as source:
                            shutil.copyfileobj(source, temporary, length=4 * 1024 * 1024)
                    temporary.close()
                    inspect_cool(temporary_path)
                    temporary_path.replace(destination)
                finally:
                    temporary_path.unlink(missing_ok=True)
            print(f"解压 {destination.name}: {destination.stat().st_size:,} bytes, "
                  f"{time.monotonic() - started:.1f} s", flush=True)


def relative_path(path):
    """清单中的路径始终相对仓库根目录，而不是执行命令时的工作目录。"""
    return Path(os.path.relpath(path.resolve(), REPO_ROOT)).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="原始 WT 与 tar 所在目录")
    parser.add_argument("--raw-dir", type=Path, help="突变样本解压目录，默认 <data-dir>/raw")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/samples.csv", help="样本清单 CSV")
    args = parser.parse_args()
    raw_dir = args.raw_dir or args.data_dir / "raw"
    archive_path = args.data_dir / ARCHIVE_NAME
    samples = [
        ("WT", replicate, args.data_dir / f"GSE272159_37C_rep{replicate}.mapq_30.10.cool", "", "")
        for replicate in (1, 2)
    ]
    # WT 为独立原文件，先检查以免缺文件时仍进行大文件解压。
    for _, _, path, _, _ in samples:
        inspect_cool(path)
    extract_mutants(archive_path, raw_dir)
    samples.extend(
        (condition, replicate, raw_dir / member[:-3], relative_path(archive_path), member)
        for condition, replicate, member in MUTANT_SAMPLES
    )
    rows = []
    for condition, replicate, path, source_archive, source_member in samples:
        rows.append({
            "sample_id": f"{condition}_rep{replicate}",
            "condition": condition,
            "replicate": replicate,
            "path": relative_path(path),
            "source_archive": source_archive,
            "source_member": source_member,
            **inspect_cool(path),
        })
    signatures = {(row["chrom"], row["bin_size"], row["genome_length"]) for row in rows}
    if len(signatures) != 1:
        raise ValueError(f"六个样本的染色体、分辨率或基因组长度不一致: {signatures}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"已核对 {len(rows)} 个样本，清单写入 {args.output}")
    for row in rows:
        print(f"  {row['sample_id']}: {row['source_chrom']}, {row['bin_size']} bp, "
              f"{row['n_pixels']:,} pixels, {row['count_dtype']}, weight={row['has_weight']}")


if __name__ == "__main__":
    main()
