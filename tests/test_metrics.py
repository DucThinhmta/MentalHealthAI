import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import config
import metrics


def test_single_label_report_values():
    y_true = [0, 0, 0, 1, 1, 2, 3, 3]
    y_pred = [0, 0, 1, 1, 2, 2, 3, 0]
    summary, per_class, cm = metrics.single_label_report(
        y_true, y_pred, config.DEPRESSION_LABELS, ordinal=True)
    assert abs(summary["accuracy"] - 5 / 8) < 1e-9
    assert per_class.loc[0, "support"] == 3 and cm.shape == (4, 4) and cm.sum() == 8
    assert summary["within_one_level"] >= summary["accuracy"]
    assert -1 <= summary["quadratic_weighted_kappa"] <= 1
    assert abs(summary["mae_levels"] - (0 + 0 + 1 + 0 + 1 + 0 + 0 + 3) / 8) < 1e-9


def test_missing_class_does_not_crash():
    summary, per_class, _ = metrics.single_label_report([0, 0, 1], [0, 0, 0], ["a", "b", "c"])
    assert per_class.loc[2, "f1"] == 0 and summary["macro_f1"] < 1


def _synthetic_emotion(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 28)) < 0.08).astype(int)
    probs = np.clip(0.5 * y + rng.normal(0.15, 0.1, y.shape), 0, 1)
    return y, probs


def test_tune_thresholds_and_report():
    y, probs = _synthetic_emotion()
    spec = metrics.tune_thresholds(y, probs, min_support=20)
    assert 0.05 <= spec["global"] <= 0.6 and spec["per_class"]
    arr = metrics.thresholds_to_array(spec)
    assert arr.shape == (28,)
    s_glob, table = metrics.emotion_report(y, probs, spec["global"])
    s_cls, _ = metrics.emotion_report(y, probs, arr)
    assert len(table) == 28 and s_cls["f1_macro"] >= s_glob["f1_macro"] - 1e-9  # tối ưu trên chính tập này


def test_rare_class_falls_back_to_global():
    y, probs = _synthetic_emotion()
    y[:, 16] = 0; y[:5, 16] = 1                      # 'grief' chỉ 5 mẫu
    spec = metrics.tune_thresholds(y, probs, min_support=20)
    assert "grief" not in spec["per_class"]
    assert metrics.thresholds_to_array(spec)[16] == spec["global"]


def test_plots_written(tmp_path):
    summary, per_class, cm = metrics.single_label_report([0, 1, 1, 0], [0, 1, 0, 0], ["a", "b"])
    metrics.plot_confusion(cm, ["a", "b"], "t", tmp_path / "cm.png")
    metrics.plot_f1_by_class(per_class, "class", "t", tmp_path / "f1.png")
    assert (tmp_path / "cm.png").stat().st_size > 1000 and (tmp_path / "f1.png").stat().st_size > 1000
