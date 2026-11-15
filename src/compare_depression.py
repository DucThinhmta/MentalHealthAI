"""So BERT (nhiều seed) với baseline TF-IDF cho nhánh Depression và ra kết luận có căn cứ.

    python src/compare_depression.py                       # dùng mọi reports/seeds_*.json của depression
    python src/compare_depression.py --tags focal base     # chỉ một số cấu hình

Vì sao cần: BERT hiện đạt macro-F1 0,365 trong khi TF-IDF + LogReg đạt 0,412 (một lần chạy mỗi bên). Không thể kết luận "mô hình nào tốt hơn"
từ một lần chạy; script này đặt các con số nhiều seed của BERT cạnh baseline và nói rõ khi KHÔNG phân biệt được.

Quy tắc kết luận (đặt trước khi nhìn số liệu):
  - "BERT vượt baseline":   cận dưới KTC 95% của macro-F1 BERT > macro-F1 baseline.
  - "baseline vượt BERT":    cận trên KTC 95% của macro-F1 BERT < macro-F1 baseline.
  - còn lại: "chưa phân biệt được" (cần thêm seed, hoặc dùng cả hai và nói rõ).
Baseline chỉ có một lần chạy (TF-IDF khá ổn định, nhưng vẫn là điểm yếu: nên nêu rõ trong báo cáo).
"""
import argparse
import json

import pandas as pd

import config

PRIMARY = "macro_f1"
SECONDARY = ["quadratic_weighted_kappa", "within_one_level", "accuracy"]


def load_baselines(path=None):
    path = path or config.REPORT_DIR / "baselines.csv"
    df = pd.read_csv(path)
    df = df[df["task"] == "depression"]
    return {r["model"]: {"macro_f1": float(r["macro_f1"]), "accuracy": float(r["accuracy"]),
                         "quadratic_weighted_kappa": None if pd.isna(r.get("qwk")) else float(r["qwk"])}
            for _, r in df.iterrows()}


def verdict(agg, baseline_f1):
    """So khoảng tin cậy 95% của macro-F1 (BERT, nhiều seed) với một con số baseline."""
    m = agg.get(PRIMARY)
    if not m or m["n"] < 2:
        return "chưa đủ seed (cần >= 2) để so sánh"
    lo, hi = m["ci95"]
    if lo > baseline_f1:
        return "BERT vượt baseline"
    if hi < baseline_f1:
        return "baseline vượt BERT"
    return "chưa phân biệt được"


def load_seed_files(tags=None):
    out = {}
    for path in sorted(config.REPORT_DIR.glob("seeds_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("task") == "depression" and (not tags or data["tag"] in tags):
            out[data["tag"]] = data["metrics"]
    return out


def build_report(seed_metrics, baselines):
    best_name, best_f1 = max(baselines.items(), key=lambda kv: kv[1]["macro_f1"]) if baselines else (None, None)
    lines = []
    if best_name is None:
        return "Thiếu reports/baselines.csv (chạy python src/baseline_tfidf.py).", []
    lines.append(f"Baseline tốt nhất: {best_name}  macro-F1 = {best_f1['macro_f1']:.3f}  (một lần chạy)")
    verdicts = []
    for tag, agg in seed_metrics.items():
        m = agg.get(PRIMARY)
        if not m:
            continue
        v = verdict(agg, best_f1["macro_f1"])
        verdicts.append((tag, v))
        lines.append(f"\nBERT cấu hình '{tag}' (n = {m['n']} seed): macro-F1 = {m['mean']:.3f} ± {m['std']:.3f}  "
                     f"KTC95% [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]  ->  {v}")
        for k in SECONDARY:
            if k in agg:
                lines.append(f"    {k:<26} {agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}")
    return "\n".join(lines), verdicts


def recommendation(verdicts):
    if not verdicts:
        return "Chưa có kết quả nhiều seed: chạy run_seeds.py trước."
    if any(v == "BERT vượt baseline" for _, v in verdicts):
        best = [t for t, v in verdicts if v == "BERT vượt baseline"]
        return f"Dùng cấu hình {best[0]} cho nhánh Depression và ghi KTC 95% vào model card."
    if all(v == "baseline vượt BERT" for _, v in verdicts):
        return ("TF-IDF + LogReg đang tốt hơn BERT trên macro-F1: nên dùng nó cho nhánh Depression (hoặc ghép hai mô hình), "
                "và ghi trung thực kết quả này trong báo cáo.")
    return ("Chưa phân biệt được BERT với baseline: tăng số seed (5-10), thử --base-model roberta-base, hoặc báo cáo cả hai "
            "và không khẳng định mô hình nào tốt hơn.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    a = ap.parse_args()
    text, verdicts = build_report(load_seed_files(a.tags), load_baselines())
    print(text)
    print("\nKết luận gợi ý:", recommendation(verdicts))


if __name__ == "__main__":
    main()
