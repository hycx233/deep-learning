# 基于深度学习的 Micro-C 接触矩阵处理

深度学习综合实践三人课程项目。完成已知结构识别、新结构候选发现、多轨道可视化，并选做不同实验条件下的结构差异定位。

**当前状态（2026-10-09）：实验、文稿与非视频交付内容已齐，后续主要为视频录制和最终提交。** #19/#23/#24 已合入 `main`；组长已收到 B 的正式 CNN 和 C 的发现/轨道包，完成哈希核验，并从独立解压目录复跑 CNN、AE 和聚类，结果与正式记录一致。代码 ZIP 保留两种正式权重；报告为 20 页、演示为 14 页。验收范围、输入指纹与未进行的检查见 [10/9 交接验收](docs/results/integration/交接验收-20261009.md)。47 个候选仍对应 10 个独立位置，4 个代表窗口只能解释共同有效区域，没有可信新结构。

| 成员 | 角色与主责 |
| --- | --- |
| [hycx233](https://github.com/hycx233)（A，组长） | 共享数据处理、差异定位、统筹与提交 |
| [onlysonder](https://github.com/onlysonder)（B） | 已知结构分类、跨重复验证、复跑与打包 |
| [lanshuheng9-oss](https://github.com/lanshuheng9-oss)（C） | 候选发现、聚类与多轨道可视化 |

已发布 [12 条任务](https://github.com/hycx233/deep-learning/issues)。[PR #23](https://github.com/hycx233/deep-learning/pull/23) 的跨重复验证和 [PR #19](https://github.com/hycx233/deep-learning/pull/19) 的整合内容已合入 `main`。A 已收到 B/C 的正式产物并完成整合验收；B/C 无需重复训练、推理或打包。A 负责最终内容、录制安排与提交核对。按组长 10/8 的安排，先推进到只剩视频，暂不要求成员分别录音；视频可能由组长统一录制，安排另定。

## 项目材料

- [Agent 协作规则](AGENTS.md)：统一实现、验证、实验记录和 issue/PR 习惯。
- [课程实践方案 PDF](基于深度学习的接触矩阵处理实践方案.pdf)
- [项目完成计划](docs/项目完成计划.md)：范围、分工、技术路线、进度与提交要求。
- [GitHub 任务与成员分工](docs/GitHub任务清单.md)：12 项任务、负责人、日期和验收条件。
- [运行与交付清单](docs/运行与交付清单.md)：从数据到四个任务的运行顺序、正式结果、权重交接和剩余工作。
- [数据说明](data/README.md)：原始矩阵的获取与放置方式。
- [数据准备与 CPU 交接样例](docs/数据准备说明.md)：运行方法、坐标依据、表字段与已验证结果。
- [共享预处理说明](docs/共享预处理说明.md)：六样本归一化、环状窗口、正式分组和下游数据接口。
- [条件差异分析](docs/条件差异分析.md)：已知结构的强度比较、重复参照、688 条结果及 8 个案例。
- [候选聚类说明](docs/候选聚类说明.md)：统一表征上的 PCA、DBSCAN、独立位置统计与候选表。
- [LaTeX 报告协作说明](docs/report/README.md)：章节负责人、编译方法与当前骨架状态。
- [Beamer 演示与讲解交接](docs/slides/README.md)：三人章节、PDF 录制、可选 PPTX 导出与讲稿。
- [课程标注 Excel](data/标注数据.xlsx)

## 项目范围

| 任务 | 计划实现 |
| --- | --- |
| 必做一：已知结构识别与解释 | 轻量 CNN 三分类、评价与显著性图 |
| 必做二：新结构发现 | 小型自编码器、候选打分、聚类与跨重复验证 |
| 必做三：接触频率与结构可视化 | WT、ΔstpA、ΔhnsΔstpA 三条件的全基因组多轨道图 |
| 选做四：结构变化分析 | 两种突变条件相对 WT 的差异定位 |

按 AI agent 辅助开发安排，每人预计约 13 小时人工投入，模型训练等待时间另算。当前轻量 CNN 和自编码器均已有 CPU 正式运行记录，现阶段的评价、文稿与整合无需等待 GPU；RTX 2080 Ti 仅按实际需要使用。

截止按组长于 **2026 年 9 月 27 日在对话中确认的 10 月 18 日**执行。课程 PDF 只写两周，没有具体截止日期，当前也没有另附教师通知。内部安排更新为：10/8 主要实验与 PR 已整合，10/9 核对文稿与产物交接，10/11 前完成非视频交付，视频录制安排另定，仍以 10/15 前检查并提交为目标，10/16—17 留作机动。原先从 9/18 假设两周的日期不再作为截止要求；保持原任务范围与人工预算。

## 加入后开始协作

克隆仓库：

```bash
git clone git@github.com:hycx233/deep-learning.git
cd deep-learning
```

随后阅读项目计划，按数据说明准备本地文件。开始让 AI agent 工作时，先要求它阅读根目录的 [AGENTS.md](AGENTS.md) 和当前 issue；不确定工具是否自动加载时，直接在提示中写明。

CPU 数据准备（已验证 Python 3.12）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_annotations.py
python scripts/prepare_samples.py
python scripts/preview_structures.py
OPENBLAS_NUM_THREADS=1 python scripts/prepare_data.py
```

这些入口不需要 PyTorch 或 CUDA。`preview_structures.py` 保留原始 counts 预览；`prepare_data.py` 则生成正式归一化后的 128×128 输入和共享 split。首轮全样本预处理需要数分钟，后续复用 100 bp 缓存。

分类模块接入 `outputs/preprocessing/default/classification/windows.csv` 与同目录 `windows.npz`，包含 WT 两个重复的 688 个窗口。共享划分为 482/102/104 个训练/验证/测试窗口，重叠窗口以及同一结构的重复均不跨集合。使用已有 `split`，不要自行重新按 ID 随机划分。完整命令与其他模块接口见共享预处理说明。

共享缓存准备完成后，可直接在 CPU 上运行已知结构的条件差异分析：

```bash
OPENBLAS_NUM_THREADS=1 python scripts/compare_conditions.py
```

输出位于 `outputs/differences/known_structures/`，已检查的结果与代表图见
[`docs/results/differences/`](docs/results/differences/)。此步骤不依赖模型训练；变化标签为描述性参照，不能作为统计显著性或单因子因果结论。

- 从 `main` 创建任务分支，如 `feat/12-classifier`，数字使用实际 issue 号。
- 每个普通任务提交一个 PR，写明实现内容、运行命令与代表结果；找一名其他成员审核后合并。
- PR 用 `Closes #编号` 关联普通任务。共同报告的分段 PR 用 `Refs #编号`，最后统一关闭任务。
- 原始矩阵、缓存、权重与批量输出保留在本地或共享存储；提交脚本、小配置、小型结果表、报告源文件和代表图。
- 每人负责自己模块的报告、演示文稿、讲稿与产物交接，按已分配的 issue 推进；暂不要求成员分别录音。

最终报告采用 **LaTeX**：`docs/report/main.tex` 作为主文件，各成员在 `sections/` 下分章节编辑，提交编译后的 PDF 并保留源文件。演示以 **Beamer + XeLaTeX（16:9）** 为唯一编辑源，模板为 `docs/slides/main.tex`；当前包含 A 5 页、B 5 页、C 4 页，共 14 页，已纳入 #9 结果。按组长 2026-10-02 的决定，当前默认使用 Beamer PDF 播放录屏；10/8 起优先完成非视频交付，录制人员与合成方式另定，若最终需 PPTX，可用备用导出入口，不维护另一套 PPT 模板。具体编译、检查与录制步骤见[演示说明](docs/slides/README.md)和共同的 [#11 文稿任务](https://github.com/hycx233/deep-learning/issues/11)。

仅在需要 PPTX 时运行以下备用命令，PPTX 不作为组内必交项或 PR 合并条件：

```bash
python scripts/export_beamer_pptx.py \
  --input-pdf docs/slides/build/main.pdf \
  --output-pptx dist/presentation.pptx
```

备用导出需要 Poppler 的 `pdftoppm` 和依赖文件中的 `python-pptx`；先按演示说明编译 PDF。PPTX 每页为整页图片，文字修改仍回到 Beamer 源文件。

任务二的自编码器、1 kb 全基因组扫描、候选打分与强度对照见
[发现模块说明](docs/发现模块说明.md)。正式输出使用 #2 的共享归一化和空间分组。

任务三的接触强度定义、轨道构建和 10 kb 分段出图命令见
[轨道图模块说明](docs/轨道图模块说明.md)。

仓库已有简短的 issue 与 PR 模板，不额外搭建服务或复杂 CI。

## 已实现的模块

任务一已完成 B 的独立 CPU 复跑和 #4 固定测试：104 个测试窗口上，CNN 的 Accuracy/Macro-F1 为 **0.5096/0.3454**，多数类基线为 0.7308/0.2815，逻辑回归为 0.8365/0.5644；三个方法的 CHID F1 均为 0。报告如实保留这一结果及失败分析，不据此扩大调参。正式评价见 [`docs/results/classifier/test_evaluation/`](docs/results/classifier/test_evaluation/)，使用 B 本次的 checkpoint，与组长首轮权重分开记录。

下面保留组长首轮训练与验证的运行示例；该次验证集 Macro-F1=0.5861 不属于 B 的正式测试。B 的当前权重路径、测试及解释命令见[分类模块说明](docs/分类模块说明.md)。自编码器发现、候选聚类、轨道图和 #9 跨重复验证均已有结果；验证范围与限制见[验证记录](docs/results/validation/README.md)。

```bash
pip install -r requirements.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python scripts/train_classifier.py \
    --windows-csv outputs/preprocessing/default/classification/windows.csv \
    --arrays-npz outputs/preprocessing/default/classification/windows.npz \
    --out-dir outputs/classifier/real_seed20260918 --device cpu --seed 20260918
python scripts/predict_classifier.py \
    --checkpoint outputs/classifier/real_seed20260918/best.pt \
    --windows-csv outputs/preprocessing/default/classification/windows.csv \
    --arrays-npz outputs/preprocessing/default/classification/windows.npz \
    --split val --device cpu \
    --out-csv outputs/classifier/real_seed20260918/predictions_val.csv
```

先运行上文的共享预处理生成真实窗口。模型权重保留本地，运行记录与验证结果见 `docs/results/classifier/`；这些验证指标用于选择轮次，不是最终测试性能。

## 目录

```text
.
├── README.md
├── AGENTS.md                   # 三人使用的统一 agent 规则
├── .gitignore
├── .github/                   # issue / PR 模板
├── requirements.txt           # 当前数据准备与预览所需依赖
├── scripts/
│   ├── prepare_annotations.py
│   ├── prepare_samples.py
│   ├── preview_structures.py
│   ├── prepare_data.py
│   └── compare_conditions.py
├── src/                       # 共用归一化、切窗与分组函数
├── tests/                     # 计数、环状坐标与防泄漏检查
├── 基于深度学习的接触矩阵处理实践方案.pdf
├── data/
│   ├── README.md
│   ├── 标注数据.xlsx
│   ├── structures.csv
│   ├── genes.csv
│   ├── samples.csv
│   ├── GSE272159_37C_rep1.mapq_30.10.cool  # 本地，不入 Git
│   ├── GSE272159_37C_rep2.mapq_30.10.cool  # 本地，不入 Git
│   └── GSE272161_RAW.tar                  # 本地，不入 Git
└── docs/
    ├── 项目完成计划.md
    ├── GitHub任务清单.md
    ├── 数据准备说明.md
    ├── 共享预处理说明.md
    ├── 条件差异分析.md
    ├── report/                # 分章节 LaTeX 报告骨架
    └── results/               # 已检查的代表图、分组与验证记录
```

当前公开源稿仅保留 GitHub 昵称；姓名、学号、组号和最终贡献比例在正式提交前本地填写，不入公开仓库，也不阻塞本轮非视频交付。贡献比例按实际工作确认。

最终按课程要求分别提交代码 ZIP、实验报告 PDF、讲解视频；组长提交三项，其他成员提交报告。
