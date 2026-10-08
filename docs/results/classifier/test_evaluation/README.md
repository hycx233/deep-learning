# #4 固定测试集评价与 Grad-CAM（B 本地复跑）

本目录保存 #4 的正式测试分析产物。收到的 2026-09-29 输入包中的 `windows.csv` / `windows.npz` SHA-256 与仓库交接记录一致；CSV 的 482/102/104 划分逐 `window_id` 与仓库 `docs/results/classifier/splits.csv` 完全相同。另独立复核 344 个物理窗口的位置重叠分组，未发现跨集合重叠。

## checkpoint 来源

原交接包没有 A 的 `best.pt`，所以按 #4 交接约定，B 使用原 seed 和超参数在 CPU 上单独复跑。这里报告的是**新 checkpoint**（SHA-256 见 `verification.json`），与 A 原 `best.pt`（仓库记录的 SHA-256 `741a35035a29571929672d6366b58ee667716168667acb1d8d85ffe52ffa2928`）分开；不能把两次运行的指标混写。新模型最佳轮次 12，运行了 20 轮后早停，最佳验证 Macro-F1 0.5674。

测试集有 104 个 replicate-window 样本，来自 52 个独立位置、每个位置两个 WT 重复。测试集仅在固定训练完成后用于一次最终推理及基线/显著性分析，没有根据测试结果调参。宏平均指标按三类等权。

## 结果概览

| 模型 | Accuracy | Macro-F1 | OPCID F1 | CHIN F1 | CHID F1 |
|---|---:|---:|---:|---:|---:|
| SmallCNN | 0.5096 | 0.3454 | 0.4062 | 0.6299 | 0.0000 |
| 多数类（训练集 CHIN） | 0.7308 | 0.2815 | 0.0000 | 0.8444 | 0.0000 |
| 逻辑回归（展平输入，class-weight balanced） | 0.8365 | 0.5644 | 0.8000 | 0.8931 | 0.0000 |

逻辑回归基线只在 train 上拟合。它在此固定测试划分上的 Accuracy 与 Macro-F1 都高于 CNN。CNN 的 Macro-F1 高于多数类基线、Accuracy 低于多数类基线；CNN 最常预测 CHIN（51 次），但只有 51/104 次，低于测试集中 CHIN 的 76 个真值样本。主要问题是漏掉 CHID（8/8）以及把 27 个 CHIN 判为 OPCID。结论限于这一次固定划分，不据此声称 CNN 优于简单分类器。

## 运行命令

从仓库根目录执行，权重和中间数组仍保存在 Git 忽略的 `outputs/` 下：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/train_classifier.py \
  --windows-csv outputs/preprocessing/default/classification/windows.csv \
  --arrays-npz outputs/preprocessing/default/classification/windows.npz \
  --out-dir outputs/classifier/b_retrain_seed20260918 \
  --epochs 30 --patience 8 --batch-size 32 --lr 0.001 \
  --weight-decay 0.0001 --dropout 0.3 --seed 20260918 --device cpu

python scripts/predict_classifier.py \
  --checkpoint outputs/classifier/b_retrain_seed20260918/best.pt \
  --windows-csv outputs/preprocessing/default/classification/windows.csv \
  --arrays-npz outputs/preprocessing/default/classification/windows.npz \
  --splits-csv outputs/classifier/b_retrain_seed20260918/splits.csv \
  --split test --device cpu \
  --out-csv outputs/classifier/b_retrain_seed20260918/predictions_test.csv

python scripts/analyze_classifier.py \
  --checkpoint outputs/classifier/b_retrain_seed20260918/best.pt \
  --predictions-csv outputs/classifier/b_retrain_seed20260918/predictions_test.csv \
  --windows-csv outputs/preprocessing/default/classification/windows.csv \
  --arrays-npz outputs/preprocessing/default/classification/windows.npz \
  --splits-csv outputs/classifier/b_retrain_seed20260918/splits.csv \
  --out-dir outputs/classifier/b_retrain_seed20260918/evaluation --device cpu
```

验证过的执行环境为 Python 3.12.14、PyTorch 2.14.0、NumPy 2.5.3、pandas 3.0.6、scikit-learn 1.9.1、Matplotlib 3.11.2。逐轮训练、checkpoint 指纹、输入指纹、代码指纹与所有评价数字分别见本目录 `training_run.json`、`analysis_run.json`、`verification.json`。

## 文件内容

- `test_metrics.csv`、`confusion_matrix_test.csv`、`confusion_matrix_test.png`：指标与混淆矩阵。
- `test_predictions_with_baselines.csv`、`failure_cases.csv`：逐窗口输出及错误案例。
- `gradcam_examples.csv`、`gradcam/`：每类三个不同结构的位置；各图并排呈现原输入矩阵与**模型预测类别**的 Grad-CAM。错误例的 CAM target 因而是模型预测类。
- `training_run.json`、`training_log.txt`、`analysis_run.json`、`verification.json`：运行和来源记录。
- `best.pt` 不纳入 Git；模型可依上述命令重建。接手使用 A 原权重时，应重新生成对应预测和分析，不能复用本目录的分数。

## 组长整合复核（2026-10-02）

104 条测试预测的三组混淆矩阵、Accuracy 和 Macro-F1 已独立复算；共享输入和划分与交接文件一致。当前评价代码的逻辑回归只在 train 拟合，Grad-CAM 目标为预测类别。

来源记录尚有一项需随 #12 交接确认：`training_run.json` / `analysis_run.json` 中的 `scripts/train_classifier.py`、`src/classifier_data.py` 指纹与提交代码不匹配，LF/CRLF 和 BOM 差异不能解释。请 B 保留并提供运行时这两份源码或差异说明，连同本次正式 checkpoint 一并交接；不改写原 JSON，也不据此重训。这个来源缺口不改变已复算的评价数字。
