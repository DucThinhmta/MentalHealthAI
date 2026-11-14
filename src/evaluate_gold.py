"""Đánh giá trên TẬP VÀNG tiếng Việt (data/gold/vi_gold.csv, xem annotation.py): số liệu thật cho tiếng Việt.

    python src/evaluate_gold.py                             # không cần mô hình: phát hiện khủng hoảng + khía cạnh sinh hoạt
    python src/evaluate_gold.py --translate                 # thêm stress qua dịch Việt->Anh trên máy + mô hình stress hiện có
    python src/evaluate_gold.py --stress-model-dir DIR      # thêm stress bằng mô hình đa ngữ huấn luyện trực tiếp trên tiếng Việt

Báo cáo có số mục (n) và khoảng tin cậy bootstrap cho các chỉ số chính; n nhỏ thì kết luận chỉ mang tính thăm dò.
"""
import argparse
import json

import numpy as np
import pandas as pd

import analyze_study as stats_
import annotation
import assessment
import config
import safety


def load_gold(path=None):
    df = pd.read_csv(path or annotation.GOLD_FILE, dtype=str, keep_default_na=False)
    return df


def _binary_with_ci(truth, pred, n_boot=1000):
    m = stats_.binary_metrics(truth, pred)
    t, p = np.asarray(truth).astype(int), np.asarray(pred).astype(int)
    if len(t) >= 20:
        rec = lambda a, b: float(((a == 1) & (b == 1)).sum() / max((a == 1).sum(), 1))
        _, m["recall_ci95"] = stats_.bootstrap(rec, [t, p], n_boot)
    return m


def eval_crisis(gold, predict=safety.detect_crisis):
    """Phát hiện nguy cơ tự hại trên nhãn 'crisis' của người gán (recall quan trọng nhất)."""
    truth = (gold["crisis"] == "1").astype(int)
    pred = [int(bool(predict(t))) for t in gold["text"]]
    return _binary_with_ci(truth, pred)


def eval_domains(gold, predict=assessment.detect_domains):
    """Với mỗi khía cạnh: bộ nhận diện luật có bắt được 'concern' mà người gán đánh dấu không (và ngược lại)."""
    detected = [predict(t) for t in gold["text"]]
    out = {}
    for d in assessment.DOMAIN_ORDER:
        col = f"dom_{d}"
        if col not in gold:
            continue
        truth = (gold[col] == "concern").astype(int)
        pred = [int(x.get(d) == "concern") for x in detected]
        out[d] = _binary_with_ci(truth, pred)
    return out


def eval_stress(gold, predict_prob):
    """Stress (co/khong; bỏ 'khong_ro'): AUC và F1 ở ngưỡng 0,5. predict_prob(văn bản) -> xác suất stress."""
    sub = gold[gold["stress"].isin(["co", "khong"])]
    if len(sub) < 10 or sub["stress"].nunique() < 2:
        return {"n": len(sub), "note": "quá ít mục có nhãn stress rõ ràng"}
    truth = (sub["stress"] == "co").astype(int).to_numpy()
    probs = np.array([predict_prob(t) for t in sub["text"]])
    auc, auc_ci = stats_.bootstrap(stats_._auc, [truth, probs], 1000)
    m = stats_.binary_metrics(truth, probs >= 0.5)
    f1 = 2 * m["ppv"] * m["sensitivity"] / (m["ppv"] + m["sensitivity"]) if m["ppv"] + m["sensitivity"] else float("nan")
    return {"n": len(sub), "auc": auc, "auc_ci95": auc_ci, "f1_at_0.5": f1, **{k: m[k] for k in ("sensitivity", "specificity", "ppv")}}


def translate_predictor():
    """Dịch Việt->Anh trên máy rồi dùng mô hình stress hiện có (cần models/translator và mô hình stress)."""
    import analyze
    import translator
    def predict(text):
        english = translator.runtime_vi2en(text)
        if not english:
            raise RuntimeError("Chưa tải mô hình dịch: python src/translator.py --download opus")
        return float(analyze.stress_probs(english)[1])
    return predict


def direct_predictor(model_dir):
    """Mô hình đa ngữ huấn luyện trực tiếp trên tiếng Việt (vd. xlm-roberta-base với --bilingual)."""
    import modeling
    tok, model = modeling.load(model_dir)
    return lambda text: float(modeling.predict_probs(tok, model, [text], config.MAX_LEN["stress"])[0][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=None)
    ap.add_argument("--translate", action="store_true")
    ap.add_argument("--stress-model-dir", default=None)
    a = ap.parse_args()
    gold = load_gold(a.gold)
    print(f"Tập vàng: {len(gold)} mục")
    res = {"n": len(gold), "crisis": eval_crisis(gold), "domains": eval_domains(gold)}
    c = res["crisis"]
    print(f"Nguy cơ tự hại: recall {c['sensitivity']:.3f}, precision {c['ppv']:.3f}, n dương = {c['tp'] + c['fn']}")
    for d, m in res["domains"].items():
        print(f"  {d:<10} recall {m['sensitivity']:.2f}  precision {m['ppv']:.2f}  n dương = {m['tp'] + m['fn']}")
    if a.translate:
        res["stress_translate"] = eval_stress(gold, translate_predictor())
        print("Stress (dịch rồi phân tích):", res["stress_translate"])
    if a.stress_model_dir:
        res["stress_direct"] = eval_stress(gold, direct_predictor(a.stress_model_dir))
        print("Stress (mô hình đa ngữ trực tiếp):", res["stress_direct"])
    config.REPORT_DIR.mkdir(exist_ok=True)
    (config.REPORT_DIR / "gold_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
