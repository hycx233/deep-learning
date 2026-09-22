# LaTeX 报告

这是 [#11](https://github.com/hycx233/deep-learning/issues/11) 的阶段报告，**不是最终报告**。已填入 #1 的数据准备事实、#2 的归一化与共享分组，以及 #8 的 344 个已知结构条件差异分析和代表案例。分类、候选发现与验证仍需各负责人依据真实运行补充；文中的“待补”不要直接作为正式内容提交。

## 编译

需要 XeLaTeX、`ctex` 与常用 LaTeX 宏包。模板使用 TeX Live 自带的 Fandol 中文字体，不依赖某位成员的系统字体或绝对路径。从仓库根目录执行：

```bash
cd docs/report
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

执行两次用于解析交叉引用。预览文件为 `docs/report/build/main.pdf`，构建产物不提交 Git。章节修改后打开相关页面检查中文、引用、表格和图的位置。当前只产出骨架预览；最终定稿时再将正式 PDF 放入仓库根目录的 `dist/`，按课程要求命名，不把骨架当成最终交付。

2026-09-20 已使用 XeLaTeX（TeX Live 2026 / Arch Linux）连续编译两次，并以 Poppler 渲染检查骨架；补入 #2 后生成 7 页预览并再次检查受影响的页面。中文、公式、表格和参考文献显示正常，没有未解析引用或超出页边界的文本。

同日补入 #8 的方法、结果与代表图后，连续编译两次生成 10 页阶段报告，已渲染检查第 6—9 页的方法、表格、图片与章节衔接；无内容溢出或未解析引用。

2026-09-22 合入候选发现、聚类与多轨道图章节后，再次连续编译两次生成 12 页阶段报告。已用 Poppler 渲染并检查第 5—11 页，覆盖发现与聚类正文、代表热图、与差异分析章节的衔接、差异方法及结果表、两张差异案例图和后续章节。未见文本越界、乱码或未解析引用；编译仅有数学字体尺寸替代警告。第 9 页为短段续文，第 10 页集中排放两张案例图，阶段稿保留该布局，最终定稿时可再压缩留白。PDF 位于 `build/main.pdf`，本轮页面预览位于 `build/qa-20260922/`。

## 分章节协作

| 文件 | 负责人 | 内容 |
| --- | --- | --- |
| `main.tex` | A / hycx233 | 主文件、成员信息与章节顺序 |
| `sections/a_overview.tex` | A | 任务目标与独立的“设计思路”章节 |
| `sections/a_data.tex` | A | 数据来源、坐标与共享预处理 |
| `sections/a_difference.tex` | A | 条件差异定位 |
| `sections/a_summary.tex` | A 汇总，三人核对 | 讨论、结论与最终实际贡献 |
| `sections/b_classification.tex` | B / onlysonder | 分类、评价与解释 |
| `sections/b_validation.tex` | B | 候选跨重复验证、复跑与复现说明 |
| `sections/c_discovery_visualization.tex` | C / lanshuheng9-oss | 候选发现、聚类与多轨道图 |
| `references.tex` | 各自追加使用的来源 | 简单参考文献，无需额外 BibTeX/Biber 步骤 |

各成员修改自己的章节，通过 `\input` 汇入主文件；普通章节 PR 用 `Refs #11`，不用同时改写 `main.tex`。计划中的模型或参数应写成计划，经过正式实验确认后再替换成最终方法。图表、数值和结论要对应已保存的结果，并在各章节记录结果位置。

报告挑选图优先放在 `docs/results/`，在本目录编译时可用 `../results/...` 引用，不复制整批输出。成员贡献目前只写计划约三分之一，最终必须用实际百分比替换且总和为 100%。姓名、课程要求的封面信息与摘要在定稿时补齐。演示文稿仍按共同计划只选 Beamer 或 PPT 一种，不由这份报告替代讲解视频。
