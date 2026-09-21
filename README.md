# 基于深度学习的 Micro-C 接触矩阵处理

深度学习综合实践三人课程项目。完成已知结构识别、新结构候选发现、多轨道可视化，并选做不同实验条件下的结构差异定位。

**当前状态：三人按 GitHub issue 分工推进。** 已实现标注整理、六样本准备、共享归一化、环状切窗和空间分组，可向分类与发现模块提供真实模型输入。LaTeX 报告骨架已经建立，模型正式训练与后续分析仍按各自 issue 推进。

| 成员 | 角色与主责 |
| --- | --- |
| [hycx233](https://github.com/hycx233)（A，组长） | 共享数据处理、差异定位、统筹与提交 |
| [onlysonder](https://github.com/onlysonder)（B） | 已知结构分类、跨重复验证、复跑与打包 |
| [lanshuheng9-oss](https://github.com/lanshuheng9-oss)（C） | 候选发现、聚类与多轨道可视化 |

已发布 [12 条任务](https://github.com/hycx233/deep-learning/issues)。开始顺序：组长先做 [#1 标注与样本](https://github.com/hycx233/deep-learning/issues/1)和 [#2 共享预处理](https://github.com/hycx233/deep-learning/issues/2)；`onlysonder` 可先做 [#3 分类入口](https://github.com/hycx233/deep-learning/issues/3)；`lanshuheng9-oss` 可先做 [#5 自编码器](https://github.com/hycx233/deep-learning/issues/5)与 [#7 轨道样图](https://github.com/hycx233/deep-learning/issues/7)的骨架。B、C 先用小样本开发，依赖完成后接入共享实现。

## 项目材料

- [Agent 协作规则](AGENTS.md)：统一实现、验证、实验记录和 issue/PR 习惯。
- [课程实践方案 PDF](基于深度学习的接触矩阵处理实践方案.pdf)
- [项目完成计划](docs/项目完成计划.md)：范围、分工、技术路线、进度与提交要求。
- [GitHub 任务与成员分工](docs/GitHub任务清单.md)：12 项任务、负责人、日期和验收条件。
- [数据说明](data/README.md)：原始矩阵的获取与放置方式。
- [数据准备与 CPU 交接样例](docs/数据准备说明.md)：运行方法、坐标依据、表字段与已验证结果。
- [共享预处理说明](docs/共享预处理说明.md)：六样本归一化、环状窗口、正式分组和下游数据接口。
- [LaTeX 报告协作说明](docs/report/README.md)：章节负责人、编译方法与当前骨架状态。
- [课程标注 Excel](data/标注数据.xlsx)

## 项目范围

| 任务 | 计划实现 |
| --- | --- |
| 必做一：已知结构识别与解释 | 轻量 CNN 三分类、评价与显著性图 |
| 必做二：新结构发现 | 小型自编码器、候选打分、聚类与跨重复验证 |
| 必做三：接触频率与结构可视化 | WT、ΔstpA、ΔhnsΔstpA 三条件的全基因组多轨道图 |
| 选做四：结构变化分析 | 两种突变条件相对 WT 的差异定位 |

按 AI agent 辅助开发安排，每人预计约 13 小时人工投入，模型训练等待时间另算。正式训练主要使用组长的 RTX 2080 Ti，其他成员负责各自模块的开发、结果检查与分析。

日期暂按 2026 年 9 月 18 日开始：9 月 28 日完成可提交版本，9 月 29—30 日作为机动，10 月 1 日为内部提交日。正式截止以课程通知为准。

## 加入后开始协作

接受私有仓库邀请后克隆：

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

- 从 `main` 创建任务分支，如 `feat/12-classifier`，数字使用实际 issue 号。
- 每个普通任务提交一个 PR，写明实现内容、运行命令与代表结果；找一名其他成员审核后合并。
- PR 用 `Closes #编号` 关联普通任务。共同报告的分段 PR 用 `Refs #编号`，最后统一关闭任务。
- 原始矩阵、缓存、权重与批量输出保留在本地或共享存储；提交脚本、小配置、小型结果表、报告源文件和代表图。
- 每人负责自己模块的报告、演示文稿和讲解录音，按已分配的 issue 推进。

最终报告采用 **LaTeX**：`docs/report/main.tex` 作为主文件，各成员在 `sections/` 下分章节编辑，提交编译后的 PDF 并保留源文件。演示文稿可用 **LaTeX Beamer 或 PPT**，9/26 统一选一种格式和模板，不重复制作；讲解视频由各成员片段合成。具体交付见三人共同的 [#11 文稿任务](https://github.com/hycx233/deep-learning/issues/11)，章节骨架可提前建立，无需等待所有实验结束。

任务二的自编码器、1 kb 全基因组扫描、候选打分与强度对照见
[发现模块说明](docs/发现模块说明.md)。正式输出使用 #2 的共享归一化和空间分组。

任务三的接触强度定义、轨道构建和 10 kb 分段出图命令见
[轨道图模块说明](docs/轨道图模块说明.md)。

仓库已有简短的 issue 与 PR 模板，不额外搭建服务或复杂 CI。

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
│   └── prepare_data.py
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
    ├── report/                # 分章节 LaTeX 报告骨架
    └── results/               # 已检查的代表图、分组与验证记录
```

最终按课程要求分别提交代码 ZIP、实验报告 PDF、讲解视频；组长提交三项，其他成员提交报告。
