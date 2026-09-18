# 基于深度学习的 Micro-C 接触矩阵处理

深度学习综合实践三人课程项目。完成已知结构识别、新结构候选发现、多轨道可视化，并选做不同实验条件下的结构差异定位。

**当前状态：三人按 GitHub issue 分工推进。** 已实现标注整理、六样本准备与真实窗口预览，正式共享归一化和训练数据划分由 #2 继续完成。成员与职责见下表，具体任务见任务清单及其链接的 issue。

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
```

这三个入口不需要 PyTorch 或 CUDA；预览为原始 counts，不是已经完成归一化与划分的正式训练数据。

- 从 `main` 创建任务分支，如 `feat/12-classifier`，数字使用实际 issue 号。
- 每个普通任务提交一个 PR，写明实现内容、运行命令与代表结果；找一名其他成员审核后合并。
- PR 用 `Closes #编号` 关联普通任务。共同报告的分段 PR 用 `Refs #编号`，最后统一关闭任务。
- 原始矩阵、缓存、权重与批量输出保留在本地或共享存储；提交脚本、小配置、小型结果表、报告源文件和代表图。
- 每人负责自己模块的报告、演示文稿和讲解录音，按已分配的 issue 推进。

最终报告采用 **LaTeX**：`docs/report/main.tex` 作为主文件，各成员在 `sections/` 下分章节编辑，提交编译后的 PDF 并保留源文件。演示文稿可用 **LaTeX Beamer 或 PPT**，9/26 统一选一种格式和模板，不重复制作；讲解视频由各成员片段合成。具体交付见三人共同的 [#11 文稿任务](https://github.com/hycx233/deep-learning/issues/11)，章节骨架可提前建立，无需等待所有实验结束。

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
│   └── preview_structures.py
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
    └── results/data_preparation/ # 已检查的窗口预览图与位置表
```

最终按课程要求分别提交代码 ZIP、实验报告 PDF、讲解视频；组长提交三项，其他成员提交报告。
