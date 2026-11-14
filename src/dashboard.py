"""Bảng điều khiển TỔNG HỢP ẨN DANH cho nhóm nghiên cứu.   MHAI_ADMIN_CODE=... streamlit run src/dashboard.py

- Chỉ hiện số liệu gộp, KHÔNG có dòng theo từng người; ô đếm nhỏ hơn 5 bị che ("<5") để khó suy ngược ra cá nhân.
- Cần mã truy cập trong biến môi trường MHAI_ADMIN_CODE; chưa đặt thì từ chối hiển thị gì cả (an toàn mặc định).
- Nhật ký (analysis_log.json) chỉ được đọc trong bộ nhớ khi tải lên, không lưu lại.
- Theo dõi trôi dạt: điểm 'mức quan tâm' trung bình theo tuần; cờ cảnh báo nếu lệch nhiều giữa đầu và cuối (cần xem lại mô hình/dữ liệu).
"""
import hmac
import json
import os
from collections import Counter, defaultdict
from datetime import date

import config
import study

K_MIN = 5
DRIFT_THRESHOLD = 0.15


def suppress(n, k=K_MIN):
    """Che ô đếm nhỏ (0 < n < k) để bảo vệ quyền riêng tư."""
    return f"<{k}" if 0 < n < k else n


def _counts(values):
    return dict(Counter(values))


def aggregate_study(records):
    """Số liệu gộp từ hồ sơ nghiên cứu (không giữ mã người tham gia)."""
    out = {"n": len(records)}
    if not records:
        return out
    scores = [r["bot"]["score"] for r in records if r["bot"].get("score") is not None]
    out["level_counts"] = _counts(r["bot"].get("level") for r in records)
    out["evidence_counts"] = _counts(r["bot"].get("evidence") for r in records)
    out["phq9_band_counts"] = _counts(r["phq9"]["band_name"] for r in records)
    out["gad7_band_counts"] = _counts(r["gad7"]["band_name"] for r in records)
    out["item9_positive"] = sum(bool(r["phq9"].get("item9_positive")) for r in records)
    out["mean"] = {"bot_score": sum(scores) / len(scores) if scores else None,
                   "phq9": sum(r["phq9"]["total"] for r in records) / len(records),
                   "gad7": sum(r["gad7"]["total"] for r in records) / len(records)}
    known = defaultdict(lambda: [0, 0])
    for r in records:
        for d, state in r["bot"].get("domains", {}).items():
            known[d][1] += 1
            known[d][0] += state == "concern"
    out["domain_concern"] = {d: {"concern": c, "known": k, "rate": c / k} for d, (c, k) in known.items()}
    out["weekly_score"] = weekly_scores(records)
    out["drift"] = drift(out["weekly_score"])
    return out


def weekly_scores(records):
    """{'2026-W45': (điểm bot trung bình, số người)} theo tuần ISO của ngày thu thập."""
    weeks = defaultdict(list)
    for r in records:
        if r["bot"].get("score") is not None and r.get("date"):
            y, w, _ = date.fromisoformat(r["date"]).isocalendar()
            weeks[f"{y}-W{w:02d}"].append(r["bot"]["score"])
    return {k: (sum(v) / len(v), len(v)) for k, v in sorted(weeks.items())}


def drift(weekly, threshold=DRIFT_THRESHOLD, min_weeks=4):
    """Cờ trôi dạt: chênh lệch điểm trung bình giữa nửa đầu và nửa cuối các tuần >= threshold (cần >= min_weeks tuần)."""
    if len(weekly) < min_weeks:
        return {"flag": False, "note": f"cần ít nhất {min_weeks} tuần dữ liệu"}
    means = [m for m, _ in weekly.values()]
    half = len(means) // 2
    shift = sum(means[half:]) / len(means[half:]) - sum(means[:half]) / len(means[:half])
    return {"flag": abs(shift) >= threshold, "shift": shift}


def aggregate_log(entries):
    """Số liệu gộp từ nhật ký phân tích của ứng dụng (analysis_log.json)."""
    out = {"turns": len(entries)}
    if not entries:
        return out
    out["safety_counts"] = _counts(e.get("safety_level", "none") for e in entries)
    out["reply_source_counts"] = _counts(e.get("reply_source", "?") for e in entries)
    out["risk_llm_counts"] = _counts(e.get("risk_llm") or "n/a" for e in entries)
    out["analyzed_rate"] = sum(bool(e.get("analyzed")) for e in entries) / len(entries)
    scored = [e for e in entries if e.get("analyzed")]
    out["model_scored_rate"] = sum(bool(e.get("model_ok")) for e in scored) / len(scored) if scored else None
    out["crisis_rate"] = out["safety_counts"].get("crisis", 0) / len(entries)
    return out


def authorized(entered, expected=None):
    """So sánh mã truy cập theo thời gian không đổi; chưa đặt MHAI_ADMIN_CODE thì luôn từ chối."""
    expected = os.getenv("MHAI_ADMIN_CODE") if expected is None else expected
    return bool(expected) and hmac.compare_digest(entered.encode(), expected.encode())


def _show_counts(st, title, counts):
    if counts:
        st.markdown(f"**{title}**")
        st.table({"Nhóm": [str(k) for k in counts], "Số lượng": [str(suppress(v)) for v in counts.values()]})


def main():
    import pandas as pd
    import streamlit as st
    st.set_page_config(page_title="Bảng điều khiển nghiên cứu", page_icon="📊", layout="wide")
    st.title("📊 Bảng điều khiển nghiên cứu (tổng hợp ẩn danh)")
    if not os.getenv("MHAI_ADMIN_CODE"):
        st.error("Chưa đặt biến môi trường MHAI_ADMIN_CODE nên bảng điều khiển không hiển thị dữ liệu.")
        st.stop()
    if not authorized(st.text_input("Mã truy cập", type="password")):
        st.info("Nhập mã truy cập để xem.")
        st.stop()
    st.caption(f"Ô đếm nhỏ hơn {K_MIN} được che để bảo vệ quyền riêng tư. Không hiển thị dữ liệu từng người.")

    records = study.load_records()
    agg = aggregate_study(records)
    st.subheader(f"Hồ sơ nghiên cứu: {suppress(agg['n'])} người")
    if agg["n"] >= K_MIN:
        c1, c2, c3 = st.columns(3)
        c1.metric("Điểm bot trung bình", f"{agg['mean']['bot_score']:.2f}" if agg["mean"]["bot_score"] is not None else "n/a")
        c2.metric("PHQ-9 trung bình", f"{agg['mean']['phq9']:.1f}")
        c3.metric("GAD-7 trung bình", f"{agg['mean']['gad7']:.1f}")
        _show_counts(st, "Mức của bot (0-3)", agg["level_counts"])
        _show_counts(st, "Độ tin cậy của bot", agg["evidence_counts"])
        _show_counts(st, "Mức PHQ-9", agg["phq9_band_counts"])
        _show_counts(st, "Mức GAD-7", agg["gad7_band_counts"])
        st.write(f"Chọn câu 9 PHQ-9 > 0: {suppress(agg['item9_positive'])}")
        if agg["weekly_score"]:
            st.line_chart(pd.DataFrame({"Điểm bot trung bình": {k: v[0] for k, v in agg["weekly_score"].items()}}))
            d = agg["drift"]
            (st.warning if d["flag"] else st.caption)(
                f"Trôi dạt điểm bot theo tuần: {'CẢNH BÁO, lệch ' + format(d['shift'], '+.2f') if d['flag'] else d.get('note', 'chưa thấy bất thường')}")
    else:
        st.info(f"Chưa đủ {K_MIN} hồ sơ để hiển thị số liệu gộp.")

    st.subheader("Nhật ký phân tích (tải lên để xem, không lưu lại)")
    up = st.file_uploader("analysis_log.json", type="json")
    if up is not None:
        try:
            log = aggregate_log(json.loads(up.read().decode("utf-8")))
        except (ValueError, AttributeError):
            st.error("Tệp không đúng định dạng nhật ký.")
        else:
            st.write(f"Số lượt: {suppress(log['turns'])}")
            if log["turns"] >= K_MIN:
                _show_counts(st, "Mức an toàn", log["safety_counts"])
                _show_counts(st, "Nguồn phản hồi", log["reply_source_counts"])
                _show_counts(st, "Đánh giá nguy cơ của LLM", log["risk_llm_counts"])
                st.write(f"Tỉ lệ lượt được phân tích: {log['analyzed_rate']:.0%}; có điểm mô hình: "
                         f"{log['model_scored_rate'] if log['model_scored_rate'] is None else format(log['model_scored_rate'], '.0%')}")


if __name__ == "__main__":
    main()
