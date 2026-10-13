# 演示与单人录制

当前采用 **Beamer + XeLaTeX，16:9，17 页**，由组长代表小组完整讲解。录制使用 `dist/presentation-20261013.pdf`，逐页口播见 [单人讲稿](speaker_notes.md)，约 10 分钟。讲稿与 PDF 按相同页序编排，A/B/C 的模块文件保留内容来源索引。

## 内容与页序

演示围绕深度学习的输入、模型、训练、评价与结果展开。用三个简短代码片段说明窗口接口、分类训练和 masked MSE；生物学背景只介绍数据含义与标签。

| 页码 | 内容 | 编辑源 |
| --- | --- | --- |
| 1 | 开场、项目名称与汇报人 | `sections/opening.tex` |
| 2–3 | 学习任务、共享预处理与输入 | `sections/a_intro.tex` |
| 4–6 | CNN、训练循环、测试与基线 | `sections/b_classification_validation.tex` |
| 7–9 | 自编码器、有效像素损失与聚类 | `sections/c_discovery_visualization.tex` |
| 10–11 | WT 跨重复验证与配对图 | `sections/b_validation.tex` |
| 12 | 全基因组轨道 | `sections/c_tracks.tex` |
| 13–15 | 条件差异、案例与结果总结 | `sections/a_difference.tex` |
| 16–17 | 三人分工与结束页 | `sections/closing.tex` |

各章节只写 `frame`，共享主题在 `main.tex`。正文、列表和代码块已增加留白；后续修改保持一页一个主题。公开源稿保留 GitHub 昵称，实际贡献比例在正式提交前本地确认填写。

## 编译与预览

从仓库根目录执行：

```bash
cd docs/slides
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

PDF 位于 `docs/slides/build/main.pdf`。两遍编译完成后打开 PDF，检查新增或修改的页面，再导出到 `dist/`。

2026-10-13 版本已连续编译两次，并渲染检查全部 17 页，未见文字或图表裁切；检查记录见 [checks-20261013.json](../results/integration/checks-20261013.json)。此前 14 页版本和模型复跑记录保留于 [10/9 验收记录](../results/integration/交接验收-20261009.md)。

## 录制

1. 打开 10/13 版 PDF，进入全屏播放；另开 [单人讲稿](speaker_notes.md)。正文可以直接口播，页标题和时间提示不用念。
2. 先试录约半分钟，回放确认收音、画面比例和鼠标位置。正文按页翻阅，图表处留短暂停顿。
3. 完整讲述约 10 分钟，内部目标为 8–12 分钟。录完抽看开头、中间、结尾，并检查声音。

整段由组长讲解，统一使用“我们”，直接衔接各模块。讲到分工页时介绍三人的工作；B/C 负责模块核对，无需各自录一段。

报告、代码包与演示 PDF 已有可用版本；实际播放器检查、录制和最终视频提交在录制阶段完成。课程截止 10/18，仍以 10/15 前完成检查和提交为目标。

## 图表来源与备用导出

案例页使用两个条件的均值热图，完整重复与 O/E 对照保留在报告。轨道页放大同一区间的条件均值面板，完整四面板保留在报告与正式结果中。需要重画这两张图时，从仓库根目录执行：

```bash
OPENBLAS_NUM_THREADS=1 python scripts/plot_difference_slide.py
python scripts/plot_track_slide.py --tracks-npz outputs/tracks/contact_signal/contact_tracks.npz
```

图表和指标沿用已保存的正式结果。跨重复比较的单位是 10 个独立位置的固定代表窗口；CNN 测试与隐向量交叉验证分别说明。

当前直接播放 PDF。若最终需要 PPTX，可使用备用入口将每页转为整页图片，内容仍在 Beamer 编辑：

```bash
python scripts/export_beamer_pptx.py \
  --input-pdf docs/slides/build/main.pdf \
  --output-pptx dist/presentation.pptx
```

备用导出需要 Poppler 的 `pdftoppm` 和 `requirements.txt` 中的 `python-pptx==1.0.2`，默认每页宽 2560 像素。PPTX 为可选项。
