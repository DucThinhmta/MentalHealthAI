"""Đọc dữ liệu theo thứ tự ưu tiên: data/clean/ (sau clean_data.py) rồi tới tệp gốc."""
import ast

import pandas as pd

import config

_LEGACY = {
    ("stress", "train"): ("train_clean.csv", "text_clean"), ("stress", "test"): ("test_clean.csv", "text_clean"),
    ("depression", "train"): ("depression_train.csv", "text"), ("depression", "test"): ("depression_test.csv", "text"),
    ("emotion", "train"): ("train_emotion.csv", "text"), ("emotion", "dev"): ("dev_emotion.csv", "text"),
    ("emotion", "test"): ("test_emotion.csv", "text"),
}


def load_split(task, split, required=True):
    clean = config.CLEAN_DIR / f"{task}_{split}.csv"
    if clean.exists():
        df = pd.read_csv(clean)
    elif (task, split) in _LEGACY:
        name, text_col = _LEGACY[(task, split)]
        path = config.DATA_DIR / name
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            return None
        df = pd.read_csv(path).rename(columns={text_col: "text"})
    else:
        if required:
            raise FileNotFoundError(f"Không có {task}_{split}. Hãy chạy: python src/clean_data.py")
        return None
    df["text"] = df["text"].astype(str)
    if task == "emotion":
        df["labels"] = df["label_vector"].apply(lambda x: [float(v) for v in ast.literal_eval(x)])
    return df
