# 演示文稿与讲解交接

本组采用 **Beamer + XeLaTeX，16:9**，复用报告环境，只维护这一种格式。`main.tex` 是共享模板；当前编译结果只含 A 的 5 页，供排版和讲解练习，**不是最终全组演示或视频**。C 已有的 [内容底稿](c_member_content.md) 继续使用。

## 编译与分文件

从仓库根目录执行：

```bash
cd docs/slides
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

PDF 输出到 `docs/slides/build/main.pdf`，构建目录不入 Git。不要分别改主题或建立第二套模板。

A 的案例页使用双条件均值热图以保持标签可读，完整重复与 O/E 对照仍在报告。该图由共享缓存和保存案例生成，并核对强度、log2FC、有效覆盖一致；需要重画时从仓库根目录执行 `OPENBLAS_NUM_THREADS=1 python scripts/plot_difference_slide.py`。此命令只处理一个真实案例，不训练模型或重跑全量差异分析。

| 成员 | 编辑位置 | 内容与预计时长 |
| --- | --- | --- |
| A | `sections/a_intro.tex`、`sections/a_difference.tex`，讲稿 `a_member_content.md` | 总体设计、数据、差异与收束，5 页约 3 分钟 |
| B | 新增 `sections/b_classification_validation.tex` 和自己的讲稿 | 分类测试、基线/解释、跨重复验证，约 3–4 分钟；负责最终合并 |
| C | 按既有底稿新增 `sections/c_discovery_visualization.tex` | 发现、聚类、轨道，4 页约 2 分 40 秒 |

各章节只写 `frame`，不写 `documentclass` 或 `begin{document}`。B 合并时将 `main.tex` 的顺序改为 A 开场 → B 分类与验证 → C 发现与轨道 → A 差异与结尾，并将页脚的“A 部分预览”改为组名。每个人的页面用相同模板，不重复加封面；图引用 `../results/` 的实际结果。

每页保留一个主句与少量数字；讲稿放 Markdown，投影片不贴长段文字。C 的高图不要整张缩得难以辨认，应只取相应对照或分开展示，保留色标与标签。引用正式结果时区分 CNN 测试、CNN 验证集和隐向量交叉验证三种指标。

## 录制与合成

各成员核对最终 PDF 后录制自己的声音与页面，或交付按页分段、顺序明确的录音；A 的开场和差异/结尾分别录制，方便插入 B、C 片段。用文件名标明讲解页次即可，无需额外平台。B 统一合成为 8–12 分钟左右的视频，实际抽看开头、中间、结尾并试听。

本轮已准备 A 的投影片及可照读稿，尚未生成任何人的真实录音。最终录制前，用 #4、#9 的正式结论更新末页阶段提示与相关讲稿，再由三人核对姓名、贡献和表述。目标 10/9 收齐各自章节与录音，10/11 形成可提交视频；课程截止 10/18。

2026-09-27 已连续 XeLaTeX 编译两次生成 5 页 PDF，逐页渲染检查，未见文字/图表溢出或缺失。案例页另从六样本缓存提取 CHIN_126 一次，核对保存的均值、log2FC 与有效覆盖后生成双图，保持两个条件使用同一色标；没有重跑全量实验。预览在 `build/main.pdf`，页面截图在 `build/qa-20260927/`。
