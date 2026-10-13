# C 模块素材索引

录制使用统一的 [单人讲稿](speaker_notes.md)。C 的内容位于新版演示第 7–9、12 页：自编码器、有效像素损失、聚类和全基因组轨道。

- 自编码器与损失：[`ConvAutoencoder` / `masked_mse`](../../src/autoencoder.py)。第 8 页为 `masked_mse` 节选，省略有效像素数检查；调用方将逐窗口有效 mask 与上三角 mask 取交集。
- 打分：[`score_candidates.py`](../../scripts/score_candidates.py)，32 维表示与每窗重建误差共用同一模型。
- 聚类：[`cluster_candidates.py`](../../scripts/cluster_candidates.py)，标准化后使用 10 维白化 PCA、DBSCAN eps=3.0 / min_samples=5。
- 轨道：[`src/tracks.py`](../../src/tracks.py) 和 [`plot_tracks.py`](../../scripts/plot_tracks.py)，重复在线性强度上平均，再开方显示。

正式背景划分为 WT_rep1 的 17 个 train、4 个 val、3 个 test 窗口；第 25 轮验证 masked MSE 为 0.022307。4,642 个扫描窗口得到 47 个候选，误差与强度 Pearson 相关为 0.828452。

47 个候选合为 10 个独立位置。唯一非噪声簇包含 297 个已知参考和 9 个候选窗口，候选来自 4 个独立位置；符合新簇规则的结果为 0。轨道图共 465 张。

结果依据见 [发现记录](../results/discovery/)、[聚类记录](../results/clustering/) 和 [轨道记录](../results/tracks/)。
