"""So sánh các cách gộp nhiều tin nhắn trên bộ kịch bản eval/scenarios.json (cần mô hình Stress thật).

    python src/evaluate_scenarios.py

Chiến lược:
  concat         : nối mọi tin nhắn thành một đoạn rồi chấm một lần (cách cũ, gây lỗi "dính tiêu cực")
  latest_only    : chỉ chấm tin nhắn cuối
  per_msg_equal  : chấm từng tin, trung bình đều (decay = 1)
  per_msg_decay  : chấm từng tin, tin mới nặng hơn (decay = config.RECENCY_DECAY, cách hiện tại)
Nhãn kỳ vọng là trạng thái stress ở tin cuối do người soạn gán (bộ nhỏ, chỉ mang tính định tính).
Kết quả ghi vào reports/scenarios.csv.
"""
import json
from pathlib import Path

import pandas as pd

import config

SCENARIO_FILE = config.PROJECT_DIR / "eval" / "scenarios.json"


def load_scenarios(path=SCENARIO_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))["scenarios"]


def predict_all(messages, analyze):
    """Trả {chiến_lược: xác suất stress} cho một hội thoại."""
    w = len(messages)
    return {
        "concat": float(analyze.stress_probs(" ".join(messages))[1]),
        "latest_only": float(analyze.stress_probs(messages[-1])[1]),
        "per_msg_equal": float(analyze.analyze_conversation(messages, window=w, decay=1.0)["stress_probability"]),
        "per_msg_decay": float(analyze.analyze_conversation(messages, window=w,
                                                            decay=config.RECENCY_DECAY)["stress_probability"]),
    }


def run(scenarios, analyze):
    rows = []
    for s in scenarios:
        probs = predict_all(s["messages"], analyze)
        for name, p in probs.items():
            pred = "stress" if p >= 0.5 else "no_stress"
            rows.append({"id": s["id"], "category": s["category"], "expect": s["expect"],
                         "strategy": name, "prob_stress": round(p, 4), "pred": pred,
                         "correct": pred == s["expect"]})
    return pd.DataFrame(rows)


def summarize(df):
    overall = df.groupby("strategy")["correct"].mean().rename("accuracy")
    by_cat = df.pivot_table(index="category", columns="strategy", values="correct", aggfunc="mean")
    return overall, by_cat


def main():
    import analyze
    df = run(load_scenarios(), analyze)
    config.REPORT_DIR.mkdir(exist_ok=True)
    out = config.REPORT_DIR / "scenarios.csv"
    config.archive_existing(out)
    df.to_csv(out, index=False, encoding="utf-8")
    overall, by_cat = summarize(df)
    n = df["id"].nunique()
    print(f"\n=== {n} kịch bản ===\nĐộ chính xác tổng thể:")
    print(overall.map("{:.1%}".format).to_string())
    print("\nTheo loại kịch bản:")
    print(by_cat.map("{:.0%}".format).to_string())
    print("\nChi tiết các ca sai (failure cases):")
    bad = df[(df["strategy"] == "per_msg_decay") & (~df["correct"])]
    print(bad[["id", "category", "expect", "prob_stress"]].to_string(index=False) if len(bad) else "không có")
    print("\nĐã ghi reports/scenarios.csv")


if __name__ == "__main__":
    main()
