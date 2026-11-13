"""Chọn ngưỡng cho Emotion TRÊN TẬP DEV (không dùng test), lưu models/emotion_thresholds.json.

    python src/find_thresholds.py
Ngưỡng riêng chỉ áp cho cảm xúc có >= 20 mẫu dương trên dev; còn lại dùng ngưỡng chung.
analyze.py tự đọc tệp này.
"""
import argparse
import json

import numpy as np
import pandas as pd

import config
import metrics
import modeling
from datasets_io import load_extra, load_split, resolve_extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default="", help="chọn ngưỡng trên dev GỘP với dev của các bộ bổ sung")
    a = ap.parse_args()
    dev = load_split("emotion", "dev")
    extra = resolve_extra(a.extra)
    if extra:
        dev = pd.concat([dev, load_extra("emotion", "dev", extra)], ignore_index=True)
    y = np.array(dev["labels"].tolist())
    tok, model = modeling.load(config.find_model("emotion"))
    probs = modeling.predict_probs(tok, model, dev["text"].tolist(), config.MAX_LEN["emotion"], multilabel=True)
    spec = metrics.tune_thresholds(y, probs, min_support=20)
    for name, thr in (("0.25 (cũ)", 0.25), (f"chung {spec['global']}", spec["global"]),
                      ("riêng từng lớp", metrics.thresholds_to_array(spec))):
        s, _ = metrics.emotion_report(y, probs, thr)
        print(f"DEV | {name:>16}: micro={s['f1_micro']:.4f} macro={s['f1_macro']:.4f}")
    config.EMOTION_THRESHOLD_FILE.parent.mkdir(exist_ok=True)
    config.EMOTION_THRESHOLD_FILE.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print("Đã lưu", config.EMOTION_THRESHOLD_FILE)


if __name__ == "__main__":
    main()
