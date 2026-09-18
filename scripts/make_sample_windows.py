"""生成样例窗口，供 #2 交付前打通任务一的代码路径。

**这不是实验数据。** 这里的矩阵是按三类结构的形态特征人为合成的，只用来验证
训练与推理入口能否跑通、划分会不会把重叠窗口拆开。正式实验必须换成 #2 的共享
窗口读取函数产出的真实窗口（24 kb -> 128x128），届时删除本脚本生成的输出即可。

合成方式：三类按标注里的实际数量生成 OPCID 68、CHIN 250、CHID 26，每条结构生成
两个平移 6 kb 的 24 kb 窗口（相互重叠，用于检查同一结构的窗口不会跨集合）。
类别形态的差异模仿真实接触矩阵的粗略特征，而不是真实生物学信号。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classifier_data import CLASS_NAMES, WINDOW_SIZE  # noqa: E402

CHROM = "NC_000913.3"
CHROM_LENGTH = 4_641_652
WINDOW_BP = 24_000
SHIFT_BP = 6_000
STRUCTURE_COUNTS = {"OPCID": 68, "CHIN": 250, "CHID": 26}
# 三类结构在标注中的大致跨度，用于让合成窗口的比例接近真实情况。
STRUCTURE_LENGTHS = {"OPCID": (800, 3_000), "CHIN": (2_000, 8_000), "CHID": (10_000, 18_000)}


def _diagonal_band(size: int, width: int, amplitude: float) -> np.ndarray:
    """沿主对角线的一条带，模拟近距离接触。"""
    band = np.zeros((size, size), dtype=np.float32)
    offsets = np.arange(-width, width + 1)
    for offset in offsets:
        decay = amplitude * (1.0 - abs(offset) / (width + 1))
        if offset >= 0:
            idx = np.arange(0, size - offset)
            band[idx, idx + offset] += decay
        else:
            idx = np.arange(-offset, size)
            band[idx, idx + offset] += decay
    return band


def _blob(size: int, row: int, col: int, half: int, amplitude: float) -> np.ndarray:
    blob = np.zeros((size, size), dtype=np.float32)
    r0, r1 = max(0, row - half), min(size, row + half + 1)
    c0, c1 = max(0, col - half), min(size, col + half + 1)
    blob[r0:r1, c0:c1] = amplitude
    return blob


def _make_matrix(kind: str, rng: np.random.Generator, size: int) -> np.ndarray:
    matrix = _diagonal_band(size, width=2, amplitude=0.6)
    if kind == "OPCID":
        # 局部点状互作：偏离对角线的一个紧凑亮斑。
        offset = int(rng.integers(18, 45))
        row = int(rng.integers(0, size - offset - 8))
        matrix += _blob(size, row, row + offset, half=4, amplitude=float(rng.uniform(3.0, 5.0)))
    elif kind == "CHIN":
        # 近对角接触域：较宽的对角带。
        matrix += _diagonal_band(size, width=int(rng.integers(20, 32)), amplitude=1.8)
    elif kind == "CHID":
        # 两个相距较远的位点互作：两条对角带加一个远端块。
        matrix += _diagonal_band(size, width=int(rng.integers(10, 18)), amplitude=1.6)
        far = int(rng.integers(70, 100))
        start = int(rng.integers(0, size - far))
        matrix += _blob(size, start, start + far, half=9, amplitude=float(rng.uniform(2.0, 3.0)))
    else:
        raise ValueError(f"未知类别 {kind}")

    matrix += rng.normal(0.0, 0.15, size=(size, size)).astype(np.float32)
    matrix = np.clip(matrix, 0.0, None)
    matrix = 0.5 * (matrix + matrix.T)  # 接触矩阵对称
    # 真实数据会做深度与 O/E 归一化，这里只用 log1p 把量级压到与归一化结果相近。
    return np.log1p(matrix).astype(np.float32)


def build(out_dir: Path, seed: int, window_size: int) -> None:
    rng = np.random.default_rng(seed)
    total_structures = sum(STRUCTURE_COUNTS.values())
    gap = CHROM_LENGTH // (total_structures + 1)

    matrices: list[np.ndarray] = []
    rows: list[dict] = []
    index = 0
    for kind in CLASS_NAMES:
        for _ in range(STRUCTURE_COUNTS[kind]):
            index += 1
            low, high = STRUCTURE_LENGTHS[kind]
            length_bp = int(rng.integers(low, high))
            center = index * gap
            structure_id = f"SYN{index:04d}"
            for window_offset in (-SHIFT_BP, SHIFT_BP):
                # 24 kb 窗口居中于结构，两个窗口平移 6 kb，因此彼此重叠。
                start = center - WINDOW_BP // 2 + window_offset
                start = max(0, min(start, CHROM_LENGTH - WINDOW_BP))
                matrices.append(_make_matrix(kind, rng, window_size))
                rows.append(
                    {
                        "window_id": f"w{len(rows):06d}",
                        "chrom": CHROM,
                        "start": start,
                        "end": start + WINDOW_BP,
                        "sample_id": "WT_rep1",
                        "condition": "WT",
                        "replicate": 1,
                        "label": kind,
                        "label_index": CLASS_NAMES.index(kind),
                        "structure_id": structure_id,
                        "structure_center": center,
                        "structure_length_bp": length_bp,
                    }
                )

    x = np.stack(matrices).astype(np.float32)
    y = np.array([row["label_index"] for row in rows], dtype=np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "windows.npz", X=x, y=y)
    pd.DataFrame(rows).to_csv(out_dir / "windows.csv", index=False, encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "本目录是 scripts/make_sample_windows.py 生成的合成样例窗口，仅用于打通代码路径。\n"
        "不要用这里的数值写进报告或当作实验结果。正式实验请使用 #2 共享窗口函数产出的真实窗口。\n",
        encoding="utf-8",
    )

    print(f"样例窗口已写入 {out_dir}")
    print(f"窗口数 {x.shape[0]}，形状 {x.shape}，每类数量 {dict(zip(*np.unique(y, return_counts=True)))}")
    print("提醒：这是合成数据，正式实验需换成 #2 的真实窗口")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成任务一的样例窗口（非实验数据）")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sample_windows"))
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    args = parser.parse_args()
    build(args.out_dir, args.seed, args.window_size)


if __name__ == "__main__":
    main()
