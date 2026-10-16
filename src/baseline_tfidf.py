"""Baseline TF-IDF + Logistic Regression / LinearSVC cho Stress và Depression.

Không cần GPU/BERT; dùng làm mốc so sánh trong báo cáo. Kết quả ghi vào reports/baselines.csv.
    python src/baseline_tfidf.py
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import config


def load(task, data_dir):
    """Ưu tiên dữ liệu đã làm sạch (data/clean), nếu không có thì dùng tệp gốc."""
    clean = data_dir / "clean"
    if (clean / f"{task}_train.csv").exists():
        return pd.read_csv(clean / f"{task}_train.csv"), pd.read_csv(clean / f"{task}_test.csv")
    if task == "stress":
        tr = pd.read_csv(data_dir / "train_clean.csv").rename(columns={"text_clean": "text"})
        te = pd.read_csv(data_dir / "test_clean.csv").rename(columns={"text_clean": "text"})
        return tr, te
    return pd.read_csv(data_dir / "depression_train.csv"), pd.read_csv(data_dir / "depression_test.csv")


def run(data_dir):
    rows = []
    for task in ("stress", "depression"):
        train, test = load(task, data_dir)
        balanced = "balanced" if task == "depression" else None
        models = {
            "TF-IDF + LogReg": LogisticRegression(max_iter=1000, class_weight=balanced),
            "TF-IDF + LinearSVC": LinearSVC(class_weight=balanced),
        }
        for name, clf in models.items():
            pipe = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2), clf)
            pipe.fit(train["text"], train["label"])
            pred = pipe.predict(test["text"])
            row = {"task": task, "model": name, "accuracy": accuracy_score(test["label"], pred),
                   "macro_f1": f1_score(test["label"], pred, average="macro", zero_division=0)}
            if task == "depression":
                row["f1_per_class"] = [round(x, 3) for x in
                                       f1_score(test["label"], pred, average=None, zero_division=0)]
                row["qwk"] = cohen_kappa_score(test["label"], pred, weights="quadratic")
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    a = ap.parse_args()
    df = run(a.data_dir)
    config.REPORT_DIR.mkdir(exist_ok=True)
    df.to_csv(config.REPORT_DIR / "baselines.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
