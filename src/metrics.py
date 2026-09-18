"""分类指标。任务一只用 Macro-F1 选模型；#4 的混淆矩阵图与基线比较在此之上扩展。"""

from __future__ import annotations

import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """行是真实类别、列是预测类别的计数矩阵。"""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """逐类 F1；某类在真实标签与预测结果中都没出现时记为 0。"""
    matrix = confusion_counts(y_true, y_pred, n_classes)
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """各类 F1 的算术平均，不按样本量加权，避免被 250 条 CHIN 主导。"""
    return float(per_class_f1(y_true, y_pred, n_classes).mean())


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())
