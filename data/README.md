# 课程配套数据

标注文件 `标注数据.xlsx` 随仓库提供。原始接触矩阵与压缩包约 5.5 GiB，不写入 Git 历史；从课程配套网盘或组长获取后放在本目录。

课程实践方案第 6 页提供的网盘：[课程数据下载](https://pan.quark.cn/s/bbe1466a855a?pwd=SsJM)，提取码 `SsJM`。这是课程方案中的原始链接，下载后的文件以以下名称放置。

| 文件 | 本地已核对大小（字节） | 用途 | 纳入 Git |
| --- | ---: | --- | --- |
| `GSE272159_37C_rep1.mapq_30.10.cool` | 482254398 | WT 生物学重复 1 | 否 |
| `GSE272159_37C_rep2.mapq_30.10.cool` | 468345755 | WT 生物学重复 2 | 否 |
| `GSE272161_RAW.tar` | 4980326400 | 多条件接触矩阵包 | 否 |
| `标注数据.xlsx` | 2698974 | 已知结构及基因注释 | 是 |

文件大小仅用于快速核对，不代替内容校验。不要直接修改原始矩阵和标注表，整理后的 CSV 单独生成。

## 本组需要的条件

WT 使用两个单独的 `.cool` 文件。`GSE272161_RAW.tar` 中优先使用以下四个成员，无需一次性解压所有条件：

```text
GSM8950761_DstpA_rep1.MG1655.mapq_30.10.cool.gz
GSM8950762_DstpA_rep2.MG1655.mapq_30.10.cool.gz
GSM8950763_DhnsDstpA_rep1.MG1655.mapq_30.10.cool.gz
GSM8950764_DhnsDstpA_rep2.MG1655.mapq_30.10.cool.gz
```

这四份分别为 ΔstpA、ΔhnsΔstpA 的两个重复。运行 `python scripts/prepare_samples.py` 会仅解压这四份到 `data/raw/`，核对六个实际文件，并生成 `data/samples.csv`；已准备的文件会复用。正式归一化在 #2 完成。

## 标注文件

| 工作表 | 本组使用的内容 |
| --- | --- |
| Supplementary Table 1 | 基因名称、位置和方向，可用于基因轨道 |
| Supplementary Table 4 | 68 条 OPCID 标注 |
| Supplementary Table 5 | 250 条 CHIN 标注，保留原有 Center 字段 |
| Supplementary Table 6 | 26 条 CHID 标注 |

三类结构共 344 条。运行 `python scripts/prepare_annotations.py` 可重建已纳入仓库的 `structures.csv` 和 `genes.csv`（4516 条基因）。统一染色体名为 `NC_000913.3`，保留 Excel 的原始字段与来源行。基因采用起点减一、终点不变的转换；结构保留作者接触图坐标数值，CHIN Center 沿用原字段。详细依据、表字段和验证记录见[数据准备说明](../docs/数据准备说明.md)。

已核对的六个矩阵均为 10 bp 分辨率，染色体长度 4,641,652 bp，实际染色体名均为 `NC_000913.3`，没有预先保存的 `weight` 列；预处理需要显式读取原始 counts 并归一化。使用稀疏或局部读取，避免展开全基因组稠密矩阵。
