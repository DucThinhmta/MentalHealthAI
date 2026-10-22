"""Đánh giá thống nhất cả 3 nhánh trên tập TEST; ghi bảng/biểu đồ vào reports/.

    python src/evaluate_all.py --task all        # stress | emotion | depression | all
Chạy find_thresholds.py trước nếu muốn ngưỡng riêng cho từng cảm xúc.

Dùng cho ablation (chỉ với --task stress hoặc depression):
    --model-dir DIR   đánh giá mô hình ở thư mục khác (vd. bản huấn luyện không class weight)
    --tag NAME        thêm tiền tố NAME vào tên các tệp kết quả để không ghi đè kết quả chính
    --windowed        chấm từng mẫu bằng cửa sổ trượt (modeling.predict_text) thay vì cắt cụt ở max_length
"""
import argparse
import json

import numpy as np

import config
import metrics
import modeling
from datasets_io import load_split


TAG = ""


def _out(name):
    config.REPORT_DIR.mkdir(exist_ok=True)
    path = config.REPORT_DIR / (f"{TAG}_{name}" if TAG else name)
    config.archive_existing(path)                # giữ lại kết quả cũ trong reports/_archive/
    return path


def eval_single(task, model_dir=None, windowed=False):
    names = ["no_stress", "stress"] if task == "stress" else config.DEPRESSION_LABELS
    test = load_split(task, "test")
    tok, model = modeling.load(model_dir or config.find_model(task))
    if windowed:
        probs = np.array([modeling.predict_text(tok, model, t, config.MAX_LEN[task], config.WINDOW_WORDS[task])
                          for t in test["text"].tolist()])
    else:
        probs = modeling.predict_probs(tok, model, test["text"].tolist(), config.MAX_LEN[task])
    summary, per_class, cm = metrics.single_label_report(
        test["label"], probs.argmax(1), names, ordinal=(task == "depression"))
    per_class.to_csv(_out(f"{task}_per_class.csv"), index=False)
    metrics.plot_confusion(cm, names, f"Ma trận nhầm lẫn - {task}", _out(f"{task}_confusion.png"))
    metrics.plot_f1_by_class(per_class, "class", f"F1 theo lớp - {task}", _out(f"{task}_f1.png"))
    print(f"\n=== {task.upper()} (n={len(test)}) ===")
    print(json.dumps(summary, indent=2)); print(per_class.round(4).to_string(index=False)); print(cm)
    return summary


def eval_emotion():
    test = load_split("emotion", "test")
    y = np.array(test["labels"].tolist())
    tok, model = modeling.load(config.find_model("emotion"))
    probs = modeling.predict_probs(tok, model, test["text"].tolist(), config.MAX_LEN["emotion"], multilabel=True)

    results = {}
    s, _ = metrics.emotion_report(y, probs, 0.5)
    results["threshold_0.50"] = s
    s, _ = metrics.emotion_report(y, probs, config.DEFAULT_EMOTION_THRESHOLD)
    results["threshold_0.25"] = s
    spec_path = config.EMOTION_THRESHOLD_FILE
    thr = config.DEFAULT_EMOTION_THRESHOLD
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        s, _ = metrics.emotion_report(y, probs, spec["global"]); results[f"global_{spec['global']}"] = s
        thr = metrics.thresholds_to_array(spec)
        s, _ = metrics.emotion_report(y, probs, thr); results["per_class_thresholds"] = s
    _, table = metrics.emotion_report(y, probs, thr)
    table.to_csv(_out("emotion_per_class.csv"), index=False)
    metrics.plot_f1_by_class(table, "emotion", "F1 theo cảm xúc (đỏ: liên quan sức khỏe tâm lý)",
                             _out("emotion_f1.png"), highlight=config.MENTAL_HEALTH_EMOTIONS)
    print("\n=== EMOTION ===")
    print(json.dumps(results, indent=2))
    print(table.sort_values("f1").round(4).to_string(index=False))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["stress", "emotion", "depression", "all"], default="all")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--windowed", action="store_true")
    a = ap.parse_args()
    if (a.model_dir or a.windowed) and a.task not in ("stress", "depression"):
        raise SystemExit("--model-dir và --windowed chỉ dùng với --task stress hoặc depression")
    TAG = a.tag
    summary = {}
    if a.task in ("stress", "all"):
        summary["stress"] = eval_single("stress", a.model_dir, a.windowed)
    if a.task in ("emotion", "all"):
        summary["emotion"] = eval_emotion()
    if a.task in ("depression", "all"):
        summary["depression"] = eval_single("depression", a.model_dir, a.windowed)
    with open(_out("results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nĐã ghi vào", config.REPORT_DIR)
