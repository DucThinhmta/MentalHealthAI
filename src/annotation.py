"""Gán nhãn tay tin nhắn tiếng Việt để xây TẬP VÀNG: lấy mẫu, tính độ đồng thuận giữa hai người gán (kappa), xử lý bất đồng.

Quy trình (chi tiết: docs/nghien_cuu/ANNOTATION_GUIDE.md):
    python src/sample_for_annotation.py --n 600        # tạo data/annotation/pool.csv (đã xoá thông tin nhận dạng)
    streamlit run src/annotate_app.py                  # MỖI người gán chạy riêng và gán ĐỘC LẬP, không bàn bạc
    python src/agreement.py an linh                    # kappa theo từng trường + danh sách bất đồng
    # hai người cùng xem file bất đồng, điền cột final_*; rồi:
    python src/agreement.py an linh --build-gold       # -> data/gold/vi_gold.csv
    python src/evaluate_gold.py                        # đo phát hiện khủng hoảng, khía cạnh, stress trên tập vàng

Nhãn: stress (co/khong/khong_ro), cảm xúc chính (28 nhãn GoEmotions hoặc khong_ro), mức nặng nề của nội dung (0-3, hoặc trống),
dấu hiệu nguy cơ tự hại (0/1), và với 8 khía cạnh sinh hoạt: trống (không nhắc) / ok / concern.
"""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import assessment
import config
import privacy
from textutils import strip_accents

STRESS_VALUES = ["co", "khong", "khong_ro"]
EMOTION_VALUES = config.EMOTION_LABELS + ["khong_ro"]
LEVEL_VALUES = [0, 1, 2, 3]
DOMAIN_VALUES = ["", "ok", "concern"]
DOMAIN_FIELDS = [f"dom_{d}" for d in assessment.DOMAIN_ORDER]
FIELDS = ["stress", "emotion", "level", "crisis"] + DOMAIN_FIELDS
ANNOTATION_DIR = config.DATA_DIR / "annotation"
GOLD_FILE = config.DATA_DIR / "gold" / "vi_gold.csv"


def annotation_path(name):
    if not name.strip():
        raise ValueError("Tên người gán trống")
    safe = re.sub(r"[^a-z0-9-]", "_", strip_accents(name.strip()))      # bỏ dấu để tên tệp an toàn trên mọi hệ điều hành
    return ANNOTATION_DIR / f"{safe}.csv"


# --------------------------------------------------------------------------- lấy mẫu
def sample_pool(sources, n=600, seed=0, min_words=3, max_words=120):
    """sources: {tên nguồn: [văn bản]}. Lấy mẫu CÂN BẰNG giữa các nguồn, bỏ trùng, xoá thông tin nhận dạng, trộn ngẫu nhiên."""
    rng = np.random.default_rng(seed)
    cleaned = {}
    for name, texts in sources.items():
        seen, kept = set(), []
        for t in texts:
            t = privacy.scrub(re.sub(r"\s+", " ", str(t)).strip())
            if min_words <= len(t.split()) <= max_words and t not in seen:
                seen.add(t)
                kept.append(t)
        cleaned[name] = kept
    per = max(1, n // max(len(cleaned), 1))
    rows = []
    for name, texts in cleaned.items():
        idx = rng.permutation(len(texts))[:per]
        rows += [(texts[i], name) for i in idx]
    rng.shuffle(rows)
    return pd.DataFrame([{"id": f"v{i:04d}", "text": t, "source": s} for i, (t, s) in enumerate(rows, 1)])


# --------------------------------------------------------------------------- đồng thuận
def landis_koch(k):
    """Cách đọc kappa theo Landis & Koch (1977)."""
    if k is None or np.isnan(k):
        return "không tính được"
    if k < 0:
        return "kém"
    for upper, name in ((0.20, "yếu"), (0.40, "tạm được"), (0.60, "vừa phải"), (0.80, "khá tốt")):
        if k <= upper:
            return name
    return "gần như hoàn hảo"


def _norm(series):
    return series.fillna("").astype(str).str.strip()


def field_kappa(a, b, field):
    """Kappa cho một trường (level dùng trọng số bậc hai). NaN nếu cả hai cùng một giá trị cho mọi mục."""
    a, b = _norm(a), _norm(b)
    if field == "level":
        a, b = a.str.replace(r"\.0$", "", regex=True), b.str.replace(r"\.0$", "", regex=True)
    if a.nunique() < 2 and b.nunique() < 2:
        return float("nan")
    weights = "quadratic" if field == "level" else None
    labels = ["0", "1", "2", "3"] if field == "level" else None
    return float(cohen_kappa_score(a, b, weights=weights, labels=labels))


def pair_agreement(df_a, df_b):
    """Bảng đồng thuận cho các mục cả hai đã gán: trường, n, % trùng, kappa, nhận xét."""
    m = df_a.merge(df_b, on="id", suffixes=("_a", "_b"))
    rows = []
    for f in FIELDS:
        if f + "_a" not in m or f + "_b" not in m:
            continue
        a, b = _norm(m[f + "_a"]), _norm(m[f + "_b"])
        k = field_kappa(a, b, f)
        rows.append({"field": f, "n": len(m), "percent_agreement": float((a == b).mean()) if len(m) else float("nan"),
                     "kappa": k, "reading": landis_koch(k)})
    return pd.DataFrame(rows)


def disagreements(df_a, df_b):
    """Các mục có ít nhất một trường khác nhau; kèm cột final_<trường> để hai người cùng thống nhất."""
    m = df_a.merge(df_b, on="id", suffixes=("_a", "_b"))
    out = []
    for _, r in m.iterrows():
        diff = [f for f in FIELDS if f + "_a" in m and str(r[f + "_a"] if pd.notna(r[f + "_a"]) else "") !=
                str(r[f + "_b"] if pd.notna(r[f + "_b"]) else "")]
        if diff:
            row = {"id": r["id"], "text": r["text_a"], "disagree_on": ";".join(diff)}
            for f in FIELDS:
                row[f + "_a"], row[f + "_b"], row["final_" + f] = r.get(f + "_a", ""), r.get(f + "_b", ""), ""
            out.append(row)
    return pd.DataFrame(out)


BLANK_TOKEN = "-"          # trong cột final_*: '-' nghĩa là đã thống nhất là ĐỂ TRỐNG (không nhắc / không rõ)


def build_gold(df_a, df_b, adjudicated=None):
    """Nhãn cuối: trường nào hai người trùng thì giữ; trường khác nhau lấy từ final_* nếu đã điền ('-' = để trống),
    chưa điền thì LOẠI mục đó khỏi tập vàng. Trả (gold, số mục bị loại vì chưa thống nhất)."""
    m = df_a.merge(df_b, on="id", suffixes=("_a", "_b"))
    adj = {} if adjudicated is None or adjudicated.empty else adjudicated.set_index("id").to_dict("index")
    rows, dropped = [], 0
    for _, r in m.iterrows():
        row, complete = {"id": r["id"], "text": r["text_a"], "source": r.get("source_a", "")}, True
        for f in FIELDS:
            va = "" if pd.isna(r.get(f + "_a")) else str(r.get(f + "_a"))
            vb = "" if pd.isna(r.get(f + "_b")) else str(r.get(f + "_b"))
            if va == vb:
                row[f] = va
                continue
            final = adj.get(r["id"], {}).get("final_" + f)
            final = "" if final is None or (isinstance(final, float) and np.isnan(final)) else str(final).strip()
            if not final:
                complete = False
                break
            row[f] = "" if final == BLANK_TOKEN else final
        if complete:
            rows.append(row)
        else:
            dropped += 1
    return pd.DataFrame(rows), dropped


def load_annotations(name):
    path = annotation_path(name)
    if not path.exists():
        return pd.DataFrame(columns=["id", "text", "source"] + FIELDS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_annotation(name, row):
    """Ghi/ghi đè nhãn của một mục theo id (an toàn khi người gán sửa lại)."""
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    df = load_annotations(name)
    df = df[df["id"] != row["id"]]
    df = pd.concat([df, pd.DataFrame([{k: str(v) for k, v in row.items()}])], ignore_index=True)
    df.to_csv(annotation_path(name), index=False, encoding="utf-8-sig")
    return len(df)
