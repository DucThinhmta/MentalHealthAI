"""Đo chất lượng lớp phát hiện khủng hoảng trên eval/crisis_cases.json.

    python src/evaluate_safety.py                 # chỉ luật (không cần mạng)
    python src/evaluate_safety.py --llm           # luật + lớp LLM (cần ANTHROPIC_API_KEY)
    python src/evaluate_safety.py --show-errors

Chỉ số quan trọng nhất là RECALL (bỏ sót một người đang gặp nguy hiểm là lỗi nặng hơn nhiều so với báo động thừa),
nhưng cũng báo precision và tỉ lệ báo động thừa theo nhóm để chắc rằng bot không làm phiền người nói ẩn dụ
("buồn cười muốn chết"). Bộ test do nhóm phần mềm viết nên đây là kiểm thử hồi quy, KHÔNG phải thẩm định lâm sàng.
"""
import argparse
import json
from collections import defaultdict

import config
import safety

CASES_FILE = config.PROJECT_DIR / "eval" / "crisis_cases.json"


def load_cases(path=CASES_FILE):
    return [{"text": t, "expect": e, "cat": c} for t, e, c in json.loads(path.read_text(encoding="utf-8"))["cases"]]


def evaluate(predict, cases=None):
    """predict(text) -> bool. Trả dict số liệu tổng, theo nhóm và danh sách lỗi."""
    cases = load_cases() if cases is None else cases
    tp = fp = fn = tn = 0
    by_cat = defaultdict(lambda: [0, 0])                 # nhóm -> [đúng, tổng]
    misses, false_alarms, ambiguous = [], [], []
    for c in cases:
        pred = bool(predict(c["text"]))
        if c["expect"] is None:
            ambiguous.append({**c, "pred": pred})
            continue
        ok = pred == c["expect"]
        by_cat[c["cat"]][0] += ok
        by_cat[c["cat"]][1] += 1
        if c["expect"] and pred:
            tp += 1
        elif c["expect"] and not pred:
            fn += 1
            misses.append(c)
        elif not c["expect"] and pred:
            fp += 1
            false_alarms.append(c)
        else:
            tn += 1
    safe_div = lambda a, b: a / b if b else float("nan")
    return {
        "n": tp + fp + fn + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": safe_div(tp, tp + fn), "precision": safe_div(tp, tp + fp),
        "false_alarm_rate": safe_div(fp, fp + tn),
        "by_category": {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]} for k, v in sorted(by_cat.items())},
        "misses": misses, "false_alarms": false_alarms, "ambiguous": ambiguous,
    }


def rules_predict(text):
    return safety.detect_crisis(text)


def with_llm_predict(text):
    import llm_client
    if safety.detect_crisis(text):
        return True
    try:
        return llm_client.assess_risk(text, [])["risk"] == "high"
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="thêm lớp LLM (tốn tiền API: mỗi ca một lần gọi)")
    ap.add_argument("--show-errors", action="store_true")
    a = ap.parse_args()
    res = evaluate(with_llm_predict if a.llm else rules_predict)
    print(f"n={res['n']}  recall={res['recall']:.3f}  precision={res['precision']:.3f}  "
          f"báo động thừa={res['false_alarm_rate']:.3f}  (bỏ sót {res['fn']}, báo thừa {res['fp']})")
    for cat, v in res["by_category"].items():
        print(f"  {cat:<20} {v['correct']:>3}/{v['total']:<3} {v['accuracy']:.0%}")
    if a.show_errors:
        for title, key in (("BỎ SÓT", "misses"), ("BÁO ĐỘNG THỪA", "false_alarms")):
            print(f"\n{title}:")
            for c in res[key]:
                print(f"  [{c['cat']}] {c['text']}")
        print("\nKHÔNG RÕ (không tính điểm):")
        for c in res["ambiguous"]:
            print(f"  {'BÁO' if c['pred'] else 'bỏ qua'}: {c['text']}")
    config.REPORT_DIR.mkdir(exist_ok=True)
    out = {k: v for k, v in res.items() if k not in ("misses", "false_alarms", "ambiguous")}
    out["misses"] = [c["text"] for c in res["misses"]]
    out["false_alarms"] = [c["text"] for c in res["false_alarms"]]
    out["mode"] = "rules+llm" if a.llm else "rules"
    (config.REPORT_DIR / "safety_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
