"""Làm sạch dữ liệu: bỏ trùng, chống rò rỉ train/test, tách dev.

Giữ NGUYÊN tập test hiện có (để kết quả cũ vẫn so sánh được); chỉ làm sạch train và
tách thêm tập dev (10%) từ train cho Stress/Depression. Ghi vào data/clean/.

    python src/clean_data.py                 # dùng thư mục data/ của dự án
    python src/clean_data.py --data-dir D --out-dir O
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config


def _norm(series):
    return series.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def clean_train(train, other_texts, label="label"):
    """Bỏ mẫu rỗng, trùng, nhãn mâu thuẫn và mẫu xuất hiện ở tập khác. Trả (df, thống kê)."""
    stats = {"start": len(train)}
    train = train[train["text"].str.len() > 0]
    conflict = train.groupby("text")[label].transform("nunique") > 1
    stats["conflicting_label_rows"] = int(conflict.sum())
    train = train[~conflict]
    before = len(train)
    train = train.drop_duplicates("text")
    stats["duplicates"] = before - len(train)
    leak = train["text"].isin(other_texts)
    stats["overlap_with_eval"] = int(leak.sum())
    train = train[~leak]
    stats["end"] = len(train)
    return train, stats


def process_single_label(name, train, test, out_dir, dev_size=0.1, seed=42):
    train, test = train.copy(), test.copy()
    for df in (train, test):
        df["text"] = _norm(df["text"])
    train, stats = clean_train(train, set(test["text"]))
    train, dev = train_test_split(train, test_size=dev_size, random_state=seed, stratify=train["label"])
    for split, df in (("train", train), ("dev", dev), ("test", test)):
        df[["text", "label"]].to_csv(out_dir / f"{name}_{split}.csv", index=False, encoding="utf-8")
    print(f"\n[{name}] {stats}")
    print(f"  train={len(train)}  dev={len(dev)}  test={len(test)} (test giữ nguyên)")
    print("  phân bố nhãn train:", train["label"].value_counts().sort_index().to_dict())
    print("  trùng lặp BÊN TRONG test (không xóa):", int(test["text"].duplicated().sum()))


def process_emotion(data_dir, out_dir):
    tr = pd.read_csv(data_dir / "train_emotion.csv")
    dv = pd.read_csv(data_dir / "dev_emotion.csv")
    te = pd.read_csv(data_dir / "test_emotion.csv")
    for df in (tr, dv, te):
        df["text"] = _norm(df["text"])
    n0 = len(tr)
    tr = tr.drop_duplicates("text")
    dup = n0 - len(tr)
    leak = tr["text"].isin(set(dv["text"]) | set(te["text"]))
    tr = tr[~leak]
    tr[["text", "label_vector"]].to_csv(out_dir / "emotion_train.csv", index=False, encoding="utf-8")
    dv[["text", "label_vector"]].to_csv(out_dir / "emotion_dev.csv", index=False, encoding="utf-8")
    te[["text", "label_vector"]].to_csv(out_dir / "emotion_test.csv", index=False, encoding="utf-8")
    print(f"\n[emotion] start={n0} duplicates={dup} overlap_with_dev/test={int(leak.sum())} end={len(tr)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=config.CLEAN_DIR)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    s_tr = pd.read_csv(a.data_dir / "train_clean.csv").rename(columns={"text_clean": "text"})
    s_te = pd.read_csv(a.data_dir / "test_clean.csv").rename(columns={"text_clean": "text"})
    process_single_label("stress", s_tr, s_te, a.out_dir)

    d_tr = pd.read_csv(a.data_dir / "depression_train.csv")
    d_te = pd.read_csv(a.data_dir / "depression_test.csv")
    process_single_label("depression", d_tr, d_te, a.out_dir)

    process_emotion(a.data_dir, a.out_dir)
    print(f"\nĐã ghi vào {a.out_dir}")


if __name__ == "__main__":
    main()
