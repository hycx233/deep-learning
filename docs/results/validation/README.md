# Issue #9：WT rep1/rep2 跨重复验证与已知结构召回

2026-10-02（北京时间）已用实际 WT 原始矩阵完成校验、共享缓存、代表窗口统计、配对图和逐位置审阅。
夸克直连失败的历史保留在 `download_attempt.json`；随后用户提供 WT 原始交接 ZIP，
来源、包哈希和两份原始文件实测哈希见 `handoff_received.json`。
原始矩阵和缓存仅在被 Git 忽略的 `data/`，没有处理突变样本。

## 固定规则与结果

原 WT_rep1 扫描 top 1% 规则及阈值 `0.04633597284555435` 不变。
47 个窗口按 C 已保存的重叠连通分量合为 10 个独立位置；每位置只比较原先固定的
rep1 最高重建误差 24 kb 代表窗口，并列按候选 ID。未根据 rep2 选择代表或改规则。
原发现/聚类结果与 A/C 源文件未修改。

- 10 个位置均可计算 Pearson：**0.6638–0.9158，中位数 0.8572**。
- 每位置共同有效上三角像素 **17,578–28,680**。无像素不足/常量数据失败，无整体无法计算位置。
- 逐图审阅保留 10 个位置作为可重复接触图形；**006、015、024、035 仅保留共同有效部分**，
  无法判定其完整窗口。035 有效上三角比例为 61.29%。缺口不算接触变化。
- 8 个位置与已知注释重叠；015、035 未重叠，但均不满足原新簇规则。
  全部位置排除作为“新结构发现”的证据，**可信新结构为 0**。
- 本次未见代表窗口的明显整体现象不一致；这不是预设阈值的“复现率 100%”。
  审阅结论是描述性图形一致性，不能证明结构类别、功能或因果。

| 类别 | 固定候选去重命中 | 已知总数 | 注释召回 | 审阅保留位置去重命中 |
| --- | ---: | ---: | ---: | ---: |
| OPCID | 10 | 68 | 14.71% | 10 |
| CHIN | 10 | 250 | 4.00% | 10 |
| CHID | 1 | 26 | 3.85% | 1 |

总计 21/344（6.10%）。坐标匹配为同染色体、0-based 左闭右开区间的正长度交集；
跨原点使用 `segments`，按唯一结构 ID 去重。保留位置未减少，因此最终保留位置的
注释命中与原统计一致；**不表示 21 条结构均被逐条跨重复验证**。

## O/E、相关与对照口径

复用 `src/preprocessing.py` 的 `load_sample` / `extract_window`，未复制归一化。
原生 100 bp O/E 为 `counts/expected_raw[环状距离]`，背景包含零接触对，无伪计数。
只在两个重复共同有效且有限的严格上三角（`k=1`）计算 Pearson，排主对角；
有效零 O/E 保留，不额外排相邻对角，不用模型的 128×128 图或显示截断值计算相关。
仓库未规定相关类型，所以 Pearson 在数据到位前已固定。

预先固定的各类首条注释与随机背景（seed=20261002）如下；背景完整 24 kb 窗口
排除已知注释、所有候选与其他背景，未根据结果替换。OPCID_1 与 CHIN_1 的窗口重叠，
仅作图形对照，不当作独立统计样本。

| 对照 | Pearson r | 有效上三角像素 |
| --- | ---: | ---: |
| OPCID_1 | 0.3601 | 28680 |
| CHIN_1 | 0.3453 | 28920 |
| CHID_1 | 0.3565 | 28920 |
| background_01 | 0.1953 | 28680 |
| background_02 | 0.3489 | 28680 |
| background_03 | 0.2420 | 28680 |

10 个候选的相关均高于这批预选对照，但对照规模小、密度不同、像素有空间相关，
不计算独立像素显著性，也不能由此推出新结构。
发现阶段重建误差与强度的 Pearson 已达 0.828，高相关可能仍受高密度/条带主导。

所有配对图使用相同坐标、每对共同 mask、固定 O/E 0–10 色标，灰色为共同无效像素；
截断只用于显示。`paired_oe/` 有 10 个候选和 6 个对照的 PDF/PNG。
`paired_examples.pdf`/`.png` 展示最高 rep1 得分候选 035、首个已知 OPCID_1 和首个背景 01，
没有按 rep2 相关挑“最好案例”。专用入口直接输出矢量 PDF，避免本机 Agg 文字绘制问题；
PNG 用本次 Codex PDFium 渲染，过程与图哈希在 `run_plot_render.json`，不改变矩阵或色标。

## 主表与审阅

- `candidate_loci_review.csv`：10 行最终位置主表，含代表 ID、实际坐标、全部成员、并集范围、
  Pearson 类型/值、有效与排除像素、复现状态、保留理由及完整窗口可判定标记。
- `candidate_windows_review.csv`：47 行最终衔接表；相关明确指向位置代表
  （`correlation_scope=locus_representative_not_this_window`），不是 47 次独立实测。
- `visual_review.csv`：实际逐图审阅描述，不是相关阈值；`--stage review` 将它接回表格。
- `recall_by_class.csv`：保留原候选口径列，同时附最终保留位置的去重注释命中列。
- `controls_manifest.csv` / `controls_review.csv`：固定对照及实测结果。
- `summary.json`、`run_*.json`、`checks.json`：最终摘要、命令/参数/代码与输入哈希、检查。
- `wt_cache_run.json`：共享缓存正式命令与记录的副本；`real_pair_smoke.json`：先执行的真实小样本。
- `history-before-raw/`：先前缺数据的校验/检查记录，保留历史，不作为当前状态。

## 实测文件校验

| 原始文件（data/） | 实测字节数 | 实测 SHA-256（与预期一致） |
| --- | ---: | --- |
| GSE272159_37C_rep1.mapq_30.10.cool | 482254398 | 0256e10da36abfe0238bf16952bc43bc78365f43c201e6e3ba0c49bd243549a8 |
| GSE272159_37C_rep2.mapq_30.10.cool | 468345755 | ede9b4f7c0c6b40c8f098e43cdd0abbc0a8987f05e8b37051a5e3cbf143ad106 |

两文件均为 Cooler symmetric-upper、10 bp、NC_000913.3（4,641,652 bp）、464166 bins、
int32 counts、无 weight。像素数分别为 257522134 / 249673544，总 counts 为
368695972 / 358452621，与原共享预处理记录一致；校验全部通过。
100 bp 缓存均为 46417 bins，有效 bins 分别 45739 / 45734，末尾不足 100 bp 的 bin 排除。
原始矩阵与 100 bp 缓存总 counts 一致。

## 实际运行与复跑

使用交接 ZIP 内仅含 WT_rep1/WT_rep2 的 `samples-wt.csv`，核对两条路径为上述 `data/*.cool`。
仓库 `docs/results/validation/samples-wt.csv` 保存其规范化副本；没有默认六样本处理。
实际环境 Python 3.14.6 / CPU、numpy 2.5.3、cooler 0.10.4、h5py 3.16.0、matplotlib 3.11.2、pandas 3.0.6。
从仓库根目录执行（本次实际 Python 为父目录 `.venv/bin/python`）：

```bash
python scripts/validate_wt_candidates.py --stage verify --samples dist/WT-raw-handoff-20261002/samples-wt.csv
OPENBLAS_NUM_THREADS=1 python scripts/prepare_data.py \
  --samples dist/WT-raw-handoff-20261002/samples-wt.csv --stage cache \
  --outdir outputs/preprocessing/wt_only
# 先以一个候选和一个已知结构执行同坐标/共同 mask 的真实小样本，记录在 real_pair_smoke.json。
OPENBLAS_NUM_THREADS=1 python scripts/validate_wt_candidates.py \
  --stage analyze --samples dist/WT-raw-handoff-20261002/samples-wt.csv
python scripts/validate_wt_candidates.py --stage review
```

缓存用时约 123.5 秒。为修复元数据 JSON 类型和出图文字问题做过必要修正，
数值口径不变，未训练/调参/试不同候选阈值。
重新运行 analyze 会更新数值表并置为待审阅，应再运行 review 接回已保存的逐图结论。
审计模式不会覆盖已有实测表；需要新审计时指定新的 `--outdir`。

HEAD 为 `72ba0c0c57c4cb04e607257cb6f1e1409e2ac34a`，分支保持 `feat/4-classifier-evaluation`。
不执行 add/commit/push，不创建/切换分支或 PR。代码和小型输入哈希分别记录；
运行时 HEAD 未包含当前未提交代码，所以不能只靠 HEAD 追溯。

## 限制与交付边界

只验证每个独立位置预选的 24 kb 代表，没有穷尽全部 47 个成员或位置并集。
4 个代表窗口有共同无效区域，不能判断缺口；相关不是生物学结构存在检验。
固定少量已知/背景对照不支持普适阈值，未补选阈值或声称新结构。
#9 的校验、统计、逐类召回、配对图与逐位置结论已提供；#12 打包/独立复跑/视频不在本次范围。
编译、视觉检查与针对性验证结果见 `checks.json`。
