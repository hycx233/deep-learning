"""任务一的数据接入层：只约定输入格式，不重复实现矩阵读取与归一化。

共享的窗口读取、归一化与分组由 #2 提供，本模块负责把它的产物变成分类器能吃的张量。
#2 的输出（``outputs/preprocessing/default/classification/windows.csv`` 与同目录
``windows.npz``）已按本模块的约定生成，``load_dataset`` 可原样读取，训练与推理入口不用动。

约定的两份输入
--------------
windows 表（CSV，UTF-8）
    至少包含 ``window_id, chrom, start, end, sample_id, condition, replicate`` 与类别列。
    类别列优先取 ``label``，其次取 ``type``（结构表的既有列名）。若已带 ``split`` 列且取值
    为 train/val/test，则直接沿用，不再自行划分；否则需要 ``structure_id`` 或 ``structure``
    列来判定同一结构的窗口，理由见 ``assign_region_ids``。

数组文件（``.npz``）
    键 ``X``：float32，形状 ``N x 128 x 128``，按 #2 约定的 24 kb 窗口缩放结果；
    键 ``y``：int64，形状 ``N``。行序必须与 windows 表一致。

坐标沿用 #2 的约定：0-based、左闭右开；chrom 已统一别名。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 类别顺序在此固定，避免各处 CSV 的字符串顺序不同造成标签错位。
CLASS_NAMES: tuple[str, ...] = ("OPCID", "CHIN", "CHID")
WINDOW_SIZE = 128

REQUIRED_COLUMNS: tuple[str, ...] = (
    "window_id",
    "chrom",
    "start",
    "end",
    "sample_id",
    "condition",
    "replicate",
)
LABEL_COLUMN_CANDIDATES: tuple[str, ...] = ("label", "type")
GROUP_COLUMN_CANDIDATES: tuple[str, ...] = ("structure_id", "structure")
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


class DataFormatError(ValueError):
    """输入不符合约定格式时抛出，附带可定位的信息。"""


def _pick_label_column(columns) -> str:
    for name in LABEL_COLUMN_CANDIDATES:
        if name in columns:
            return name
    raise DataFormatError(
        f"windows 表缺少类别列，需要 {' 或 '.join(LABEL_COLUMN_CANDIDATES)}；"
        f"实际列为 {list(columns)}"
    )


def encode_labels(labels: pd.Series) -> np.ndarray:
    """按 CLASS_NAMES 的固定顺序编码类别名，遇到未登记的类别直接报错。"""
    name_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    unknown = sorted(set(labels.astype(str)) - set(name_to_index))
    if unknown:
        raise DataFormatError(
            f"出现未登记的类别 {unknown}；本模块只认 {list(CLASS_NAMES)}，"
            "如需扩展请同步修改 CLASS_NAMES"
        )
    return labels.astype(str).map(name_to_index).to_numpy(dtype=np.int64)


def symmetrize(matrices: np.ndarray) -> np.ndarray:
    """接触矩阵是对称的；#2 若只给出上三角，这里补成对称后再送进模型。"""
    return 0.5 * (matrices + np.swapaxes(matrices, -1, -2))


def load_dataset(
    windows_csv: str | Path,
    arrays_npz: str | Path,
    symmetrize_input: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """读取窗口表与数组，返回 (X, y, meta)，三者行序一致。"""
    windows_csv = Path(windows_csv)
    arrays_npz = Path(arrays_npz)
    if not windows_csv.is_file():
        raise FileNotFoundError(f"找不到 windows 表：{windows_csv}")
    if not arrays_npz.is_file():
        raise FileNotFoundError(
            f"找不到数组文件：{arrays_npz}；先用 scripts/make_sample_windows.py 生成样例，"
            "或指定 #2 的分类输入 outputs/preprocessing/default/classification/windows.npz"
        )

    meta = pd.read_csv(windows_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in meta.columns]
    if missing:
        raise DataFormatError(f"windows 表缺少列 {missing}；实际列为 {list(meta.columns)}")
    label_column = _pick_label_column(meta.columns)

    with np.load(arrays_npz) as arrays:
        for key in ("X", "y"):
            if key not in arrays:
                raise DataFormatError(f"数组文件缺少键 {key}；实际键为 {list(arrays.files)}")
        x = np.asarray(arrays["X"], dtype=np.float32)
        y = np.asarray(arrays["y"], dtype=np.int64)

    if x.ndim != 3 or x.shape[1:] != (WINDOW_SIZE, WINDOW_SIZE):
        raise DataFormatError(
            f"X 形状应为 N x {WINDOW_SIZE} x {WINDOW_SIZE}，实际为 {x.shape}"
        )
    if len(x) != len(meta):
        raise DataFormatError(f"数组有 {len(x)} 行，windows 表有 {len(meta)} 行，两者需一致")
    if len(y) != len(x):
        raise DataFormatError(f"y 长度为 {len(y)}，X 长度为 {len(x)}，两者需一致")

    nonzero_y = int((y != 0).sum())
    if nonzero_y == 0:
        raise DataFormatError("y 全为 0，请检查数组文件是否写错；类别标签应由 windows 表决定")
    if not np.array_equal(y, encode_labels(meta[label_column])):
        raise DataFormatError(
            f"数组文件的 y 与 windows 表的 {label_column} 列不一致，行序或标签映射有误"
        )

    bad = int(np.count_nonzero(~np.isfinite(x)))
    if bad:
        raise DataFormatError(f"X 中有 {bad} 个 NaN/Inf；归一化阶段应已处理零背景与无效 bin")

    if symmetrize_input:
        x = symmetrize(x)

    meta = meta.copy()
    meta["label_index"] = y
    meta["label_name"] = [CLASS_NAMES[i] for i in y]
    return x, y, meta


def assign_region_ids(meta: pd.DataFrame) -> np.ndarray:
    """兜底用的窗口分组：同一结构的窗口归为一组。

    正式划分一律沿用 #2 给出的 ``split`` 列；只有在窗口表缺 ``split`` 时才走到这里，
    以免训练入口完全无法启动。这里按结构身份分组，是因为重叠连通分量的划分属于 #2 的
    共享实现（``src/window_splits.py``），本模块不复制一份。

    早期版本曾判断"按坐标重叠传递合并会把整条染色体并成一组，所以只能按结构身份分组"。
    该结论来自合成样例的均匀间距（13,454 bp），不成立：真实坐标下 344 条结构的 24 kb
    窗口只形成 59 个连通分量（最大 20 条），相邻间距有 16.9% 超过 24 kb，正是这些断点
    把基因组切开。详见 docs/分类模块说明.md。
    """
    column = next(
        (name for name in GROUP_COLUMN_CANDIDATES if name in meta.columns), None
    )
    if column is None:
        raise DataFormatError(
            f"windows 表需要 {' 或 '.join(GROUP_COLUMN_CANDIDATES)} 列来判定同一结构的窗口，"
            f"或者直接提供 split 列；实际列为 {list(meta.columns)}"
        )
    keys = meta["chrom"].astype(str) + "|" + meta[column].astype(str)
    codes, _ = pd.factorize(keys, sort=True)
    return codes.astype(np.int64)


def count_cross_region_overlaps(meta: pd.DataFrame) -> int:
    """统计坐标上与其它分组的窗口重叠的窗口数。

    这些重叠不是错误——它们说明"重叠窗口不跨集合"只能按结构身份落实，不能按坐标
    重叠传递合并，数量写入运行记录备查。
    """
    order = meta.sort_values(["chrom", "start", "end"], kind="stable")
    count = 0
    previous_chrom = None
    previous_end = -1
    previous_region = None
    for _, row in order.iterrows():
        chrom = row["chrom"]
        if chrom == previous_chrom and int(row["start"]) < previous_end:
            if row["region_id"] != previous_region:
                count += 1
        previous_chrom = chrom
        previous_end = int(row["end"])
        previous_region = row["region_id"]
    return count


def _split_regions_by_class(
    region_labels: pd.Series,
    ratios: tuple[float, float, float],
    rng: np.random.Generator,
    warnings: list[str],
) -> dict[int, str]:
    """按类别把 region 按比例分到 train/val/test，保证每类在各集合都有样本。"""
    assignment: dict[int, str] = {}
    for label_name, regions in region_labels.groupby(region_labels):
        region_list = list(regions.index)
        rng.shuffle(region_list)
        n = len(region_list)
        if n < len(SPLIT_NAMES):
            # 样本太少时无法同时兼顾三类与三个集合，先保证训练集有样本并如实记录。
            warnings.append(
                f"类别 {label_name} 只有 {n} 个 region，不足 3 个，"
                "该类的验证/测试集可能为空"
            )
        n_val = int(round(n * ratios[1]))
        n_test = int(round(n * ratios[2]))
        if n >= len(SPLIT_NAMES):
            n_val = max(1, min(n_val, n - 2))
            n_test = max(1, min(n_test, n - n_val - 1))
        else:
            n_val = min(n_val, max(0, n - 1))
            n_test = min(n_test, max(0, n - n_val - 1))
        for i, region in enumerate(region_list):
            if i < n_val:
                assignment[region] = "val"
            elif i < n_val + n_test:
                assignment[region] = "test"
            else:
                assignment[region] = "train"
    return assignment


def assign_splits(
    meta: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 20260918,
) -> tuple[pd.DataFrame, list[str]]:
    """返回带 ``split`` 与 ``region_id`` 列的 meta，以及需要记录在运行说明里的警告。

    若 windows 表已带 ``split`` 列（#2 给出的分组列表），直接沿用。否则按基因组
    region 划分：先按类别把 region 分成约 70/15/15，再落到窗口上，因此同一结构的
    重复与重叠窗口不会跨集合。
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios 之和应为 1，实际为 {sum(ratios)}")

    meta = meta.copy()
    if "label_name" not in meta.columns:
        # 允许直接传入未经过 load_dataset 的表：按同一套映射补出类别名。
        label_column = _pick_label_column(meta.columns)
        meta["label_name"] = [CLASS_NAMES[i] for i in encode_labels(meta[label_column])]
    meta["region_id"] = assign_region_ids(meta)
    warnings: list[str] = []
    overlapping = count_cross_region_overlaps(meta)
    if overlapping:
        warnings.append(
            f"有 {overlapping} 个窗口在坐标上与其它结构的窗口重叠；"
            "这类重叠在本项目里是正常的（24 kb 窗口 vs 344 条结构的间距），"
            "分组按结构身份而非坐标重叠处理"
        )

    if "split" in meta.columns:
        given = set(meta["split"].astype(str))
        if not given <= set(SPLIT_NAMES):
            raise DataFormatError(
                f"split 列出现未登记取值 {sorted(given - set(SPLIT_NAMES))}；"
                f"只接受 {list(SPLIT_NAMES)}"
            )
        meta["split"] = meta["split"].astype(str)
        warnings.append("沿用 windows 表已有的 split 列，未重新划分")
        return meta, warnings

    region_labels = (
        meta.groupby(["region_id", "label_name"]).size().reset_index(name="count")
    )
    mixed = region_labels.groupby("region_id").size()
    if (mixed > 1).any():
        warnings.append(
            f"有 {int((mixed > 1).sum())} 个 region 含多个类别（不同结构窗口重叠），"
            "按样本数最多的类别归组"
        )
    dominant = (
        region_labels.sort_values("count", ascending=False)
        .drop_duplicates("region_id")
        .set_index("region_id")["label_name"]
    )

    rng = np.random.default_rng(seed)
    assignment = _split_regions_by_class(dominant, ratios, rng, warnings)
    meta["split"] = meta["region_id"].map(assignment)
    if meta["split"].isna().any():
        raise RuntimeError("划分后仍有窗口未分配集合，请检查 region 划分逻辑")
    return meta, warnings


def describe_splits(meta: pd.DataFrame) -> pd.DataFrame:
    """按集合与类别统计样本量，写入运行记录。"""
    table = (
        meta.groupby(["split", "label_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(list(SPLIT_NAMES))
        .fillna(0)
        .astype(int)
    )
    for name in CLASS_NAMES:
        if name not in table.columns:
            table[name] = 0
    return table[list(CLASS_NAMES)]
