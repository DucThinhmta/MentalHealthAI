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


def extra_names(task="emotion"):
    """Tên các bộ dữ liệu bổ sung đã tải (fetch_datasets.py) cho một nhánh."""
    return sorted({p.name[len(task) + 1:].rsplit("_", 1)[0] for p in config.EXTRA_DIR.glob(f"{task}_*_train.csv")})


def resolve_extra(spec, task="emotion"):
    """'all' | 'a,b' | '' -> danh sách tên; báo lỗi rõ nếu chưa tải."""
    if not spec:
        return []
    have = extra_names(task)
    names = have if spec == "all" else [n.strip() for n in spec.split(",") if n.strip()]
    missing = [n for n in names if n not in have]
    if missing or not names:
        raise FileNotFoundError(f"Chưa có dữ liệu bổ sung {missing or spec}. Hãy chạy: python src/fetch_datasets.py")
    return names


def load_extra(task, split, names):
    """Ghép các bộ bổ sung (cột 'source' = tên bộ). Trả None nếu names rỗng."""
    if not names:
        return None
    frames = []
    for n in names:
        df = pd.read_csv(config.EXTRA_DIR / f"{task}_{n}_{split}.csv")
        df["text"] = df["text"].astype(str)
        df["source"] = n
        if task == "emotion":
            df["labels"] = df["label_vector"].apply(lambda x: [float(v) for v in ast.literal_eval(x)])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_translated(task, split, names=("core",), lang=None):
    """Bản dịch máy (translate_datasets.py) của các bộ `names` cho một nhánh; None nếu chưa có tệp nào.

    lang: chỉ lấy bản dịch SANG ngôn ngữ này ('vi' | 'en'); None = cả hai."""
    tdir = config.DATA_DIR / "translated"
    frames = []
    for n in names:
        for lg in ([lang] if lang else ["vi", "en"]):
            path = tdir / f"{task}_{n}_{split}.{lg}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["text"] = df["text"].astype(str)
                df["source"] = f"{n}:mt-{lg}"
                if task == "emotion":
                    df["labels"] = df["label_vector"].apply(lambda x: [float(v) for v in ast.literal_eval(x)])
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None

