"""Chẩn đoán nhánh Emotion khi kết quả bất thường (ví dụ mọi câu đều bị dự đoán là neutral).

    python src/diagnose_emotion.py            # kiểm tra mô hình đang được dùng + mọi checkpoint 28 nhãn tìm thấy
    python src/diagnose_emotion.py --n 1000   # dùng 1000 câu dev (mặc định 500)

In ra: đường dẫn mô hình mà app/evaluate_all đang chọn, thông tin config, và với MỖI thư mục mô hình 28 nhãn tìm
thấy trong models/: F1 micro/macro trên tập dev ở ngưỡng 0,25, độ lệch chuẩn của đầu ra (≈0 nghĩa là mô hình cho
cùng một kết quả với mọi câu), và số nhãn khác nhau mà mô hình từng dự đoán. Không ghi/ sửa gì.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import config
import metrics
import modeling
from datasets_io import load_split


def candidates():
    """Mọi thư mục có config.json với 28 nhãn dưới các thư mục models/."""
    seen, out = set(), []
    for base in config.MODEL_SEARCH_DIRS:
        if not base.exists():
            continue
        for cfg in base.rglob("config.json"):
            d = cfg.parent
            if "_archive" in d.parts or d in seen:
                continue
            try:
                c = json.loads(cfg.read_text(encoding="utf-8"))
            except Exception:
                continue
            n = c.get("num_labels") or len(c.get("id2label", {}))
            if n == len(config.EMOTION_LABELS):
                seen.add(d); out.append(d)
    return sorted(out)


def check(path, texts, y):
    tok, model = modeling.load(path)
    probs = modeling.predict_probs(tok, model, texts, config.MAX_LEN["emotion"], multilabel=True)
    s, _ = metrics.emotion_report(y, probs, config.DEFAULT_EMOTION_THRESHOLD)
    pred = probs >= config.DEFAULT_EMOTION_THRESHOLD
    used = int((pred.sum(axis=0) > 0).sum())
    std = float(probs.std(axis=0).mean())
    return {"f1_micro": round(s["f1_micro"], 4), "f1_macro": round(s["f1_macro"], 4),
            "output_std": round(std, 5), "labels_ever_predicted": used,
            "verdict": ("HỎNG/SỤP: cùng một đầu ra cho mọi câu" if std < 1e-3 or used <= 2 else "có vẻ bình thường")}


def deep(path, texts, y):
    """So sánh các cách nạp (tokenizer của checkpoint hay bert-base-uncased; kiểu số float32 hay mặc định)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = Path(path)
    print("Các tệp trong checkpoint:", sorted(p.name for p in path.iterdir()))
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    print("config: dtype =", cfg.get("dtype") or cfg.get("torch_dtype"), "| problem_type =", cfg.get("problem_type"),
          "| vocab_size =", cfg.get("vocab_size"))
    toks = {}
    try:
        toks["tokenizer_checkpoint"] = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    except Exception as e:
        print("Checkpoint không có tokenizer dùng được:", repr(e)[:120])
    toks["tokenizer_bert-base-uncased"] = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    sample = "I am so sad and worried about my exam"
    for name, t in toks.items():
        print(f"  {name}: {type(t).__name__}, vocab={len(t)}, tokens={t.tokenize(sample)[:12]}")
    base = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True).eval().to(modeling.get_device())
    print("kiểu số của mô hình khi nạp mặc định:", next(base.parameters()).dtype)
    def run(label):
        for tname, tok in toks.items():
            probs = modeling.predict_probs(tok, base, texts, config.MAX_LEN["emotion"], multilabel=True)
            sc, _ = metrics.emotion_report(y, probs, config.DEFAULT_EMOTION_THRESHOLD)
            print(f"  [{tname} | {label}] micro={sc['f1_micro']:.4f} macro={sc['f1_macro']:.4f} "
                  f"std={probs.std(axis=0).mean():.5f}")
    run("kiểu số mặc định")
    base.float()                                   # ép float32 rồi thử lại
    run("ép float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--deep", action="store_true", help="thử các cách nạp khác nhau trên mô hình đang dùng")
    a = ap.parse_args()
    try:
        chosen = config.find_model("emotion")
        print("Mô hình Emotion đang được chương trình chọn:\n  ", chosen)
    except FileNotFoundError as e:
        chosen = None
        print(e)
    dev = load_split("emotion", "dev").head(a.n)
    texts, y = dev["text"].tolist(), np.array(dev["labels"].tolist())
    if a.deep and chosen:
        print("\n=== CHẾ ĐỘ --deep ===")
        deep(chosen, texts[:200], y[:200])
        return
    print(f"\nĐánh giá trên {len(texts)} câu dev, ngưỡng {config.DEFAULT_EMOTION_THRESHOLD}:\n")
    for d in candidates():
        mark = "  <-- ĐANG DÙNG" if chosen and Path(d) == Path(chosen) else ""
        print(d, mark)
        try:
            print("   ", check(d, texts, y))
        except Exception as e:
            print("    lỗi khi nạp/chạy:", repr(e)[:200])
    print("\nNếu mô hình ĐANG DÙNG bị 'HỎNG/SỤP' còn checkpoint khác bình thường: đổi tên/ chuyển thư mục hỏng đi,"
          "\nhoặc sửa MODEL_CANDIDATES['emotion'] trong src/config.py trỏ tới checkpoint tốt.")


if __name__ == "__main__":
    main()