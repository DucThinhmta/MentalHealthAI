"""Phân tích kiểm chứng: điểm của bot có tương ứng với thang đo chuẩn (PHQ-9, GAD-7) không?

    python src/analyze_study.py                       # đọc data/study/records.jsonl
    python src/analyze_study.py --min-evidence high   # chỉ tính người đã trò chuyện đủ nhiều
    python src/analyze_study.py --sample-size         # tính cỡ mẫu cần thiết trước khi thu dữ liệu
    python src/analyze_study.py --demo                # chạy thử quy trình trên dữ liệu GIẢ LẬP (không phải kết quả nghiên cứu)

Các phân tích (mọi khoảng tin cậy 95% bằng bootstrap):
1. Tương quan hạng Spearman giữa điểm 'mức quan tâm' của bot và tổng PHQ-9, GAD-7.
2. AUC của điểm bot khi phân biệt người có PHQ-9 >= 10 (hoặc GAD-7 >= 10), cùng độ nhạy/độ đặc hiệu tại ngưỡng mức >= 2.
3. Kappa có trọng số bậc hai giữa mức của bot (0-3) và mức độ nặng của thang (PHQ-9 gộp 5 mức thành 4).
4. Kiểm chứng ở mức triệu chứng: khía cạnh bot hỏi (giấc ngủ, năng lượng, ăn uống, hứng thú, tâm trạng, tập trung) so với câu
   PHQ-9 tương ứng (>= 'hơn một nửa số ngày'): độ nhạy, độ đặc hiệu, kappa.

Kết quả với n nhỏ chỉ mang tính thăm dò. Báo cáo luôn ghi n và nên nêu rõ các hạn chế (mẫu thuận tiện, tự báo cáo, bản dịch thang).
"""
import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score

import config
import study

EVIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


# --------------------------------------------------------------------------- dữ liệu
def to_frame(records):
    rows = []
    for r in records:
        bot = r.get("bot", {})
        row = {"code": r.get("code"), "bot_score": bot.get("score"), "bot_level": bot.get("level"),
               "evidence": bot.get("evidence", "low"), "n_turns": bot.get("n_turns"),
               "phq": r["phq9"]["total"], "gad": r["gad7"]["total"],
               "phq_band": min(r["phq9"]["band"], 3), "gad_band": r["gad7"]["band"],
               "item9": bool(r["phq9"].get("item9_positive"))}
        for d, item in study.DOMAIN_TO_PHQ9_ITEM.items():
            state = bot.get("domains", {}).get(d)
            row[f"bot_{d}"] = {"concern": 1, "ok": 0}.get(state, np.nan)
            row[f"phq_item_{d}"] = int(r["phq9"]["items"][item] >= study.ITEM_POSITIVE)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- thống kê
def _ci(values, level=0.95):
    values = np.asarray([v for v in values if not np.isnan(v)])
    if len(values) < 20:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2
    return (float(np.quantile(values, a)), float(np.quantile(values, 1 - a)))


def bootstrap(stat, arrays, n_boot=2000, seed=0):
    """(giá trị, (cận dưới, cận trên)) của stat(*arrays) bằng bootstrap theo cặp."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    est = stat(*arrays)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boots.append(stat(*[a[idx] for a in arrays]))
        except ValueError:
            boots.append(float("nan"))
    return float(est), _ci(boots)


def _spearman(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y)[0])


def _auc(y, s):
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _wkappa(a, b):
    if len(set(a)) < 2 and len(set(b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2, 3]))


def binary_metrics(truth, pred):
    truth, pred = np.asarray(truth).astype(int), np.asarray(pred).astype(int)
    tp = int(((truth == 1) & (pred == 1)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")
    kappa = float(cohen_kappa_score(truth, pred)) if len(set(truth)) > 1 and len(set(pred)) > 1 else float("nan")
    return {"n": len(truth), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "sensitivity": div(tp, tp + fn),
            "specificity": div(tn, tn + fp), "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn), "kappa": kappa,
            "prevalence": div(tp + fn, len(truth))}


def youden_threshold(y, s):
    """Ngưỡng điểm bot tối đa hóa độ nhạy + độ đặc hiệu - 1 (chỉ mang tính mô tả: chọn trên cùng dữ liệu là lạc quan)."""
    best, best_j = None, -1.0
    for t in sorted(set(np.round(s, 4))):
        m = binary_metrics(y, np.asarray(s) >= t)
        j = (m["sensitivity"] + m["specificity"] - 1) if not (math.isnan(m["sensitivity"]) or math.isnan(m["specificity"])) else -1
        if j > best_j:
            best, best_j = float(t), j
    return best, best_j


# --------------------------------------------------------------------------- cỡ mẫu
def n_for_correlation(rho, alpha=0.05, power=0.8):
    """Cỡ mẫu để phát hiện tương quan rho (biến đổi Fisher z, hai phía)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(math.ceil(((za + zb) / math.atanh(rho)) ** 2 + 3))


def auc_se(auc, n_pos, n_neg):
    """Sai số chuẩn của AUC (Hanley & McNeil, 1982)."""
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return math.sqrt(var)


def n_for_auc(auc=0.75, halfwidth=0.10, prevalence=0.3, alpha=0.05):
    """Tổng cỡ mẫu để khoảng tin cậy của AUC có nửa độ rộng <= halfwidth (với tỉ lệ dương tính dự kiến)."""
    z = stats.norm.ppf(1 - alpha / 2)
    for n in range(20, 5000):
        n_pos = max(2, round(n * prevalence))
        if z * auc_se(auc, n_pos, n - n_pos) <= halfwidth:
            return n
    return None


# --------------------------------------------------------------------------- phân tích chính
def analyze(records, min_evidence="medium", n_boot=2000):
    out = {"n_records": len(records), "min_evidence": min_evidence}
    if not records:
        out.update(n_analyzed=0, note="Chưa có hồ sơ nào (data/study/records.jsonl trống hoặc không tồn tại).")
        return out
    df = to_frame(records)
    df = df[df["bot_score"].notna() & (df["evidence"].map(EVIDENCE_RANK) >= EVIDENCE_RANK[min_evidence])].reset_index(drop=True)
    out["n_analyzed"] = len(df)
    if len(df) < 5:
        out["note"] = "Quá ít người (n < 5) để phân tích; cần thu thêm dữ liệu."
        return out
    out["descriptive"] = {c: {"mean": float(df[c].mean()), "sd": float(df[c].std(ddof=1)) if len(df) > 1 else float("nan")}
                          for c in ("bot_score", "phq", "gad")}
    out["item9_positive"] = int(df["item9"].sum())
    out["correlation"] = {}
    for name, col in (("PHQ-9", "phq"), ("GAD-7", "gad")):
        est, ci = bootstrap(_spearman, [df["bot_score"], df[col]], n_boot)
        out["correlation"][name] = {"spearman": est, "ci95": ci}
    out["auc"] = {}
    for name, col in (("PHQ-9>=10", "phq"), ("GAD-7>=10", "gad")):
        y = (df[col] >= 10).astype(int).to_numpy()
        entry = {"n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
        if len(set(y)) > 1:
            entry["auc"], entry["ci95"] = bootstrap(_auc, [y, df["bot_score"]], n_boot)
            entry["at_level_ge_2"] = binary_metrics(y, (df["bot_level"] >= 2).to_numpy())
            entry["youden_threshold"], entry["youden_j"] = youden_threshold(y, df["bot_score"].to_numpy())
        else:
            entry["note"] = "chỉ có một nhóm (tất cả cùng dương hoặc cùng âm): không tính được AUC"
        out["auc"][name] = entry
    lvl = df["bot_level"].astype(int).to_numpy()
    out["weighted_kappa"] = {}
    for name, col in (("PHQ-9", "phq_band"), ("GAD-7", "gad_band")):
        est, ci = bootstrap(_wkappa, [lvl, df[col].to_numpy()], n_boot)
        out["weighted_kappa"][name] = {"kappa": est, "ci95": ci}
    out["symptom_level"] = {}
    for d, item in study.DOMAIN_TO_PHQ9_ITEM.items():
        known = df[df[f"bot_{d}"].notna()]
        if len(known) >= 5:
            out["symptom_level"][d] = {"phq9_item": item + 1,
                                       **binary_metrics(known[f"phq_item_{d}"], known[f"bot_{d}"].astype(int))}
        else:
            out["symptom_level"][d] = {"phq9_item": item + 1, "n": len(known), "note": "ít hơn 5 người có dữ liệu"}
    out["sample_size_hint"] = {"n_for_rho_0.3": n_for_correlation(0.3), "n_for_auc_0.75_halfwidth_0.10": n_for_auc()}
    return out


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def format_report(res, synthetic=False):
    L = ["# Kết quả kiểm chứng" + (" (DỮ LIỆU GIẢ LẬP, KHÔNG PHẢI KẾT QUẢ NGHIÊN CỨU)" if synthetic else "")]
    L.append(f"Người tham gia: {res['n_records']}; được phân tích (độ tin cậy >= {res['min_evidence']}): {res['n_analyzed']}.")
    if "note" in res:
        return "\n".join(L + [res["note"]])
    for name, v in res["correlation"].items():
        L.append(f"- Tương quan Spearman điểm bot với {name}: ρ = {_f(v['spearman'])} (KTC 95%: {_f(v['ci95'][0])} đến {_f(v['ci95'][1])})")
    for name, v in res["auc"].items():
        if "auc" in v:
            m = v["at_level_ge_2"]
            L.append(f"- AUC {name}: {_f(v['auc'])} (KTC 95%: {_f(v['ci95'][0])} đến {_f(v['ci95'][1])}); n dương/âm = "
                     f"{v['n_pos']}/{v['n_neg']}; tại mức bot >= 2: độ nhạy {_f(m['sensitivity'])}, độ đặc hiệu {_f(m['specificity'])}")
        else:
            L.append(f"- AUC {name}: {v['note']} (n dương/âm = {v['n_pos']}/{v['n_neg']})")
    for name, v in res["weighted_kappa"].items():
        L.append(f"- Kappa có trọng số, mức bot so với mức {name}: κ = {_f(v['kappa'])} (KTC 95%: {_f(v['ci95'][0])} đến {_f(v['ci95'][1])})")
    L.append("- Kiểm chứng mức triệu chứng (khía cạnh bot so với câu PHQ-9 >= 'hơn một nửa số ngày'):")
    for d, v in res["symptom_level"].items():
        L.append(f"  - {d} (câu {v['phq9_item']}): " + (v["note"] if "note" in v else
                 f"n={v['n']}, độ nhạy {_f(v['sensitivity'])}, độ đặc hiệu {_f(v['specificity'])}, κ {_f(v['kappa'])}"))
    L.append(f"- Người chọn câu 9 PHQ-9 > 0: {res['item9_positive']}")
    L.append("Lưu ý: n nhỏ thì khoảng tin cậy rộng; mẫu thuận tiện và tự báo cáo; chọn ngưỡng Youden trên cùng dữ liệu là lạc quan.")
    return "\n".join(L)


# --------------------------------------------------------------------------- dữ liệu giả lập (chỉ để thử quy trình)
def synthetic_records(n=120, seed=0, signal=0.7):
    """Sinh hồ sơ GIẢ để chạy thử quy trình phân tích. signal: mức bot theo sát mức nặng thật (0 = ngẫu nhiên)."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        z = rng.normal()
        def items(k, shift):
            return [int(np.clip(round(1.2 + 0.9 * (z + rng.normal(0, 0.8)) + shift * rng.normal(0, 0.3)), 0, 3)) for _ in range(k)]
        phq, gad = items(9, 0), items(7, 0.5)
        zb = signal * z + (1 - signal) * rng.normal()
        score = float(1 / (1 + math.exp(-1.4 * zb)))
        domains = {d: ("concern" if (phq[it] >= 2) == (rng.random() < 0.8) else "ok")
                   for d, it in study.DOMAIN_TO_PHQ9_ITEM.items() if rng.random() < 0.7}
        recs.append(study.build_record(f"SYN-{i:03d}", {"score": score, "level": int(np.digitize(score, config.CONCERN_THRESHOLDS)),
                                                         "evidence": "high", "n_turns": 8, "n_words": 120, "domains": domains,
                                                         "self_moods": []}, phq, gad))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=None)
    ap.add_argument("--min-evidence", choices=list(EVIDENCE_RANK), default="medium")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--sample-size", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.sample_size:
        print(f"Tương quan rho=0,3 (hai phía, alpha 0,05, power 0,8): n = {n_for_correlation(0.3)}")
        print(f"Tương quan rho=0,4: n = {n_for_correlation(0.4)}")
        print(f"AUC kỳ vọng 0,75, KTC 95% rộng ±0,10, tỉ lệ dương 30%: n = {n_for_auc()}")
        print(f"AUC kỳ vọng 0,80, KTC 95% rộng ±0,07, tỉ lệ dương 30%: n = {n_for_auc(0.8, 0.07)}")
        return
    records = synthetic_records() if a.demo else study.load_records(a.records)
    res = analyze(records, a.min_evidence, a.boot)
    config.REPORT_DIR.mkdir(exist_ok=True)
    name = "study_analysis_DEMO_SYNTHETIC.json" if a.demo else "study_analysis.json"
    (config.REPORT_DIR / name).write_text(json.dumps({**res, "synthetic": a.demo}, ensure_ascii=False, indent=2, default=float),
                                          encoding="utf-8")
    print(format_report(res, synthetic=a.demo))


if __name__ == "__main__":
    main()
