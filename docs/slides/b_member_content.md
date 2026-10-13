# B 模块素材索引

录制使用统一的 [单人讲稿](speaker_notes.md)。B 的内容位于新版演示第 4–6、10–11 页，分别介绍分类与跨重复验证。

- CNN：[`SmallCNN`](../../src/cnn.py)，卷积通道 16/32/64，共 23,603 个参数。
- 训练代码：[`train_classifier.py`](../../scripts/train_classifier.py) 的 `run_epoch`，第 5 页为连续节选；省略批次读取、设备移动和指标汇总。
- 评价：[`analyze_classifier.py`](../../scripts/analyze_classifier.py)，测试结果与 Grad-CAM 见 [`test_evaluation/`](../results/classifier/test_evaluation/)。
- 验证：[`validate_wt_candidates.py`](../../scripts/validate_wt_candidates.py)，固定代表窗口与配对图见 [`validation/`](../results/validation/)。

正式 CNN 使用 B 的 checkpoint，第 12 轮最佳，测试 Accuracy / Macro-F1 为 0.5096 / 0.3454。测试集 104 个窗口来自 52 个独立位置，CHID 为 8 个窗口。Grad-CAM 对应模型预测类别。

跨重复检查以 10 个独立位置的固定代表窗口为单位，Pearson 中位数 0.8572；其中 4 个窗口包含无效区域。候选 035 共同有效比例 61.29%，相关 0.7402。去重注释命中 21/344，可信新结构数为 0。
