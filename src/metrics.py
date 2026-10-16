"""Chỉ số và biểu đồ đánh giá (thuần sklearn/matplotlib, không cần mô hình)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

import config


def single_label_report(y_true, y_pred, class_names, ordinal=False):
    """Chỉ số tổng, bảng theo lớp và ma trận nhầm lẫn. ordinal=True thêm chỉ số cho nhãn có thứ tự."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"class": class_names, "precision": p, "recall": r, "f1": f, "support": s})
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if ordinal:
        diff = np.abs(y_true - y_pred)
        summary.update({
            "mae_levels": float(diff.mean()),                       # sai lệch trung bình (số mức)
            "within_one_level": float((diff <= 1).mean()),          # đúng hoặc lệch 1 mức
            "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        })
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return summary, per_class, cm


def tune_thresholds(y_true, probs, min_support=20, grid=None):
    """Ngưỡng chung (tối đa F1 micro) + ngưỡng riêng cho lớp có đủ mẫu (tối đa F1 của lớp đó).

    Lớp quá ít mẫu trên dev (< min_support) dùng ngưỡng chung để tránh quá khớp.
    """
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    grid = np.round(np.arange(0.05, 0.61, 0.025), 3) if grid is None else grid
    best_global = max(grid, key=lambda t: f1_score(y_true, (probs >= t).astype(int),
                                                   average="micro", zero_division=0))
    per_class = {}
    for k, label in enumerate(config.EMOTION_LABELS):
        if y_true[:, k].sum() < min_support:
            continue
        per_class[label] = float(max(grid, key=lambda t: f1_score(
            y_true[:, k], (probs[:, k] >= t).astype(int), zero_division=0)))
    return {"global": float(best_global), "per_class": per_class, "min_support": min_support}


def thresholds_to_array(spec):
    glob = spec["global"]
    return np.array([spec["per_class"].get(l, glob) for l in config.EMOTION_LABELS])


def emotion_report(y_true, probs, thresholds):
    """thresholds: số (ngưỡng chung) hoặc mảng 28 phần tử. Trả (summary, bảng theo cảm xúc)."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    pred = (probs >= thresholds).astype(int)
    p, r, f, s = precision_recall_fscore_support(y_true, pred, zero_division=0)
    table = pd.DataFrame({"emotion": config.EMOTION_LABELS, "precision": p, "recall": r, "f1": f,
                          "support": y_true.sum(axis=0).astype(int)})
    summary = {"f1_micro": float(f1_score(y_true, pred, average="micro", zero_division=0)),
               "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0))}
    return summary, table


# ----------------------------------------------------------------- biểu đồ
def plot_confusion(cm, class_names, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Dự đoán", ylabel="Thực tế", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_f1_by_class(table, label_col, title, path, highlight=()):
    t = table.sort_values("f1")
    colors = ["#d9534f" if l in highlight else "#5b8def" for l in t[label_col]]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(t))))
    ax.barh(t[label_col], t["f1"], color=colors)
    ax.set(xlim=(0, 1), xlabel="F1-score", title=title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
