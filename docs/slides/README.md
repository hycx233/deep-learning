# 演示文稿与讲解交接

本组采用 **Beamer + XeLaTeX，16:9**，复用报告环境，只维护这一套源稿。`main.tex` 是共享模板；当前源文件按 A 开场 → B 分类与验证 → C 发现与轨道 → A 差异与结尾排列。B 已完成 #4 固定测试、基线、Grad-CAM 和 #9 跨重复验证；当前演示已纳入 #9 结论，仍是待成员核对、录音和最终播放检查的阶段稿。B、C 的讲稿分别见 [B 内容底稿](b_member_content.md) 和 [C 内容底稿](c_member_content.md)。

## 默认交付 PDF，PPTX 为备用

按组长 2026-10-02 的决定，当前默认使用编译后的 Beamer PDF 播放和录屏，不把 PPTX 导出作为组内必须交付或合并条件。课程方案第 14 页原文写的是 PPT；若最终需要 PPTX，已提供可选导出入口，不维护第二套内容。

备用脚本将 PDF 每页作为整页图片放入 16:9 PPTX；页面内容仍在 LaTeX 中编辑。需要 PATH 中的 Poppler `pdftoppm` 和 `requirements.txt` 中的 `python-pptx==1.0.2`；默认每页宽 2560 像素。macOS 可用 `brew install poppler`，Ubuntu/Debian 安装 `poppler-utils`，Windows 安装 Poppler 后加入 PATH。

```bash
python scripts/export_beamer_pptx.py \
  --input-pdf docs/slides/build/main.pdf \
  --output-pptx dist/presentation.pptx
```

## 演示重点

按深度学习课程来讲：**输入是什么 → 为什么选这个模型 → 如何训练 → 如何评价与检查结果**。生物学背景只交代数据含义与任务标签；不用讲基因调控机制，也不据此作因果推断。

主要用模型结构图、数据流程和代表结果解释思路。每人可挑一处几行核心代码作为辅助，不逐行讲源码，不把函数签名、文件目录或所有超参数铺满页面。比如 A 展示共享窗口接口，B 讲 CNN 和训练循环，C 讲编码/重建与有效像素损失；讲清这段代码解决什么问题即可。

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
| A | `sections/a_intro.tex`、`sections/a_difference.tex`，讲稿 `a_member_content.md` | 学习任务、共享输入、选做实现与结果核查，5 页约 3 分钟 |
| B | `sections/b_classification_validation.tex`，讲稿 `b_member_content.md` | CNN 选型与训练、测试/基线/解释、跨重复评价，约 3–4 分钟；负责最终合并 |
| C | `sections/c_discovery_visualization.tex`，讲稿 `c_member_content.md` | 自编码器重建、masked loss、候选筛选与轨道实现，4 页约 3 分钟 |

各章节只写 `frame`，不写 `documentclass` 或 `begin{document}`。最终定稿时将阶段预览页脚改为组名。每个人的页面用相同模板，不重复加封面；图引用 `../results/` 的实际结果。

每页保留一个主句与少量数字；讲稿放 Markdown，投影片不贴长段文字。C 的高图不要整张缩得难以辨认，应只取相应对照或分开展示，保留色标与标签。引用正式结果时区分 CNN 测试、CNN 验证集和隐向量交叉验证三种指标。

模板已提供 Python `lstlisting` 样式；含代码的页面用 `\begin{frame}[fragile]{标题}`，普通图表页照常。代码标明来自哪个实现，省略上下文或接口调用示意要说清楚；#9 只比较 10 个独立位置预先固定的代表窗口，不能表述为 47 个窗口逐一验证。

## 录制与合成

各成员核对最终演示 PDF 后播放录制，或交付按页分段、顺序明确的录音后由 B 配合演示页面合成；A 的开场和差异/结尾分别录制，方便插入 B、C 片段。用文件名标明讲解页次即可，无需额外平台。B 统一合成为 8–12 分钟左右的视频，实际抽看开头、中间、结尾并试听。

本轮已准备 A 的投影片及可照读稿，尚未生成任何人的真实录音。录制前由三人核对已纳入演示的 #9 结论、姓名、贡献和表述；#4 使用固定测试结果。目标 10/9 收齐各自章节与录音，10/11 形成可提交视频；课程截止 10/18。

2026-09-27 已连续 XeLaTeX 编译两次生成 5 页 PDF，逐页渲染检查，未见文字/图表溢出或缺失。案例页另从六样本缓存提取 CHIN_126 一次，核对保存的均值、log2FC 与有效覆盖后生成双图，保持两个条件使用同一色标；没有重跑全量实验。预览在 `build/main.pdf`，页面截图在 `build/qa-20260927/`。

同日按深度学习课程重点调整为“学习任务—共享输入—实现与评价”的讲解，仍为 5 页，只有一段 4 行接口调用示例。示例已用一个真实 WT 窗口执行，输出为 `(1, 1, 128, 128)` 的 float32 张量。重新连续编译两次并检查全部页面；新版截图在 `build/qa-code-focus/`。

2026-09-29 B 曾在本机合并预览中编译 13 页；当时 #22 单独检出缺少共享主文件与导出脚本，该记录不代表 #22 自身可独立编译。本次将 #21、#22 整合到 #19，并提交主文件、章节与导出入口，使三者在同一分支可复现。

C 的轨道页改为放大展示同一区间的条件均值面板，完整四面板保留在报告与正式结果中。复用保存的线性轨道、共同有效区域与“先平均后开方”的实现；重画单张演示图时执行：

```bash
python scripts/plot_track_slide.py --tracks-npz outputs/tracks/contact_signal/contact_tracks.npz
```

组长既有轨道缓存实际位于 `outputs/tracks/contact/contact_tracks.npz`，本次使用 `--tracks-npz` 指向它；路径差异不要求重新计算。#12 交付时检查实际使用的演示文件，抽看首、中、尾页的坐标和文字，实际播放后录制；若选用备用 PPTX，另确认其页数与 PDF 相同。演示源稿和采用的演示文件随代码包保留，课程仍单独提交代码 ZIP、报告 PDF、讲解视频三个文件。

2026-10-02 的早期整合预览为 13 页，编译和备用 PPTX 检查记录对应 #9 章节加入前的版本；该记录不代表当前源稿状态。

2026-10-02 接入 #9 的完整组稿后，当前 #23 分支工作区连续两次 XeLaTeX 编译为 14 页；用 PDFium 检查 B 的 #9 结果页与相邻页面、A 的结尾页（第 5–8、14 页），未见文字、图片或图例裁切。此次检查基于分支 HEAD `01fe38e` 及未提交工作区修改，尚无最终提交 SHA；视频录制与实际播放检查仍待 #12。
