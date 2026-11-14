"""Công cụ gán nhãn tiếng Việt: lấy mẫu, kappa, bất đồng, tập vàng, giao diện gán nhãn, đánh giá trên tập vàng."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import annotation as an
import config
import evaluate_gold as eg
import privacy
import sample_for_annotation as sfa

SRC = Path(__file__).resolve().parents[1] / "src"


def row(i, **kw):
    base = {"id": f"v{i:04d}", "text": f"tin nhắn số {i} dài vừa đủ", "source": "vsmec", "stress": "khong", "emotion": "neutral",
            "level": "0", "crisis": "0", **{f: "" for f in an.DOMAIN_FIELDS}}
    base.update(kw)
    return base


def frame(rows):
    return pd.DataFrame(rows)


# ---------- lấy mẫu ----------
def test_sample_pool_is_balanced_deduplicated_scrubbed_and_reproducible():
    sources = {"a": [f"bình luận a số {i} có đủ từ" for i in range(50)] + ["bình luận a số 1 có đủ từ"] * 5,
               "b": [f"tin nhắn b số {i} có đủ từ" for i in range(50)] + ["ngắn quá", " ".join(["từ"] * 200)]}
    sources["a"].append("gọi mình 0912 345 678 hoặc an@mail.com nhé bạn")
    pool = an.sample_pool(sources, n=20, seed=1)
    assert len(pool) == 20 and pool["source"].value_counts().to_dict() == {"a": 10, "b": 10}
    assert pool["text"].is_unique and not any(len(t.split()) < 3 or len(t.split()) > 120 for t in pool["text"])
    assert pool["id"].is_unique and pool["id"].iloc[0] == "v0001"
    assert an.sample_pool(sources, n=20, seed=1).equals(pool) and not an.sample_pool(sources, n=20, seed=2).equals(pool)
    big = an.sample_pool({"a": ["gọi mình 0912 345 678 hoặc an@mail.com nhé bạn"]}, n=5)
    assert "0912" not in big["text"].iloc[0] and "mail.com" not in big["text"].iloc[0] and "[SĐT]" in big["text"].iloc[0]


def test_sources_readers(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXTRA_DIR", tmp_path)
    assert sfa.vsmec_texts() == []                                         # chưa tải dữ liệu thì không có nguồn này
    log = tmp_path / "log.json"
    log.write_text(json.dumps([{"user": "mình mất ngủ cả tuần"}, {"user": privacy.NOT_STORED}, {"turn": 3}]), encoding="utf-8")
    assert sfa.log_texts([log]) == ["mình mất ngủ cả tuần"]


# ---------- kappa ----------
def test_kappa_perfect_none_and_constant():
    same = pd.Series(["co", "khong", "co", "khong"])
    assert an.field_kappa(same, same, "stress") == pytest.approx(1.0)
    assert an.field_kappa(same, pd.Series(["khong", "co", "khong", "co"]), "stress") == pytest.approx(-1.0)
    assert math.isnan(an.field_kappa(pd.Series(["co"] * 4), pd.Series(["co"] * 4), "stress"))


def test_level_kappa_is_weighted_so_near_misses_cost_less():
    truth = pd.Series(["0", "1", "2", "3", "0", "1", "2", "3"])
    near = pd.Series(["0", "2", "2", "3", "0", "1", "3", "3"])             # lệch 1 bậc
    far = pd.Series(["3", "0", "0", "0", "3", "0", "0", "0"])
    assert an.field_kappa(truth, near, "level") > an.field_kappa(truth, far, "level")
    assert an.field_kappa(truth, truth.astype(float).astype(str), "level") == pytest.approx(1.0)   # '1.0' coi như '1'


def test_landis_koch_readings():
    assert [an.landis_koch(k) for k in (-0.1, 0.1, 0.3, 0.5, 0.7, 0.9)] == \
        ["kém", "yếu", "tạm được", "vừa phải", "khá tốt", "gần như hoàn hảo"]
    assert an.landis_koch(float("nan")) == "không tính được"


def test_pair_agreement_table():
    a = frame([row(1, stress="co"), row(2, stress="khong"), row(3, stress="co"), row(4, stress="khong")])
    b = frame([row(1, stress="co"), row(2, stress="khong"), row(3, stress="khong"), row(4, stress="khong")])
    t = an.pair_agreement(a, b).set_index("field")
    assert t.loc["stress", "n"] == 4 and t.loc["stress", "percent_agreement"] == 0.75 and t.loc["emotion", "percent_agreement"] == 1.0


# ---------- bất đồng và tập vàng ----------
def test_disagreements_and_gold_building():
    a = frame([row(1), row(2, stress="co", crisis="1"), row(3, dom_sleep="concern"), row(4)])
    b = frame([row(1), row(2, stress="khong", crisis="1"), row(3, dom_sleep=""), row(4, emotion="sadness")])
    dis = an.disagreements(a, b)
    assert list(dis["id"]) == ["v0002", "v0003", "v0004"] and dis.loc[dis["id"] == "v0002", "disagree_on"].iloc[0] == "stress"
    # chưa thống nhất: các mục bất đồng bị loại, mục đồng thuận được giữ
    gold, dropped = an.build_gold(a, b)
    assert list(gold["id"]) == ["v0001"] and dropped == 3
    dis.loc[dis["id"] == "v0002", "final_stress"] = "co"
    dis.loc[dis["id"] == "v0003", "final_dom_sleep"] = "-"                    # '-' = thống nhất để trống
    dis.loc[dis["id"] == "v0004", "final_emotion"] = "sadness"
    gold, dropped = an.build_gold(a, b, dis)
    assert dropped == 0 and len(gold) == 4
    g = gold.set_index("id")
    assert g.loc["v0002", "stress"] == "co" and g.loc["v0002", "crisis"] == "1"          # trường đồng thuận được giữ nguyên
    assert g.loc["v0003", "dom_sleep"] == "" and g.loc["v0004", "emotion"] == "sadness"


def test_annotation_files_roundtrip_and_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(an, "ANNOTATION_DIR", tmp_path)
    assert an.load_annotations("An").empty and an.annotation_path("An Nguyễn!").name == "an_nguyen_.csv"
    assert an.save_annotation("an", row(1, stress="co")) == 1
    assert an.save_annotation("an", row(2)) == 2
    assert an.save_annotation("an", row(1, stress="khong")) == 2                   # sửa lại: ghi đè, không nhân đôi
    assert an.load_annotations("an").set_index("id").loc["v0001", "stress"] == "khong"
    with pytest.raises(ValueError):
        an.annotation_path("   ")


# ---------- giao diện gán nhãn ----------
@pytest.fixture
def annotate_env(tmp_path, monkeypatch):
    monkeypatch.setattr(an, "ANNOTATION_DIR", tmp_path)
    pd.DataFrame([{"id": "v0001", "text": "mình mất ngủ cả tuần nay rồi", "source": "vsmec"},
                  {"id": "v0002", "text": "hôm nay trời đẹp quá", "source": "vsmec"}]).to_csv(tmp_path / "pool.csv", index=False)
    return tmp_path


def _annotate_app():
    at = AppTest.from_file(str(SRC / "annotate_app.py"), default_timeout=30)
    at.run()
    return at


def test_annotate_app_requires_a_pool_and_a_name(tmp_path, monkeypatch):
    monkeypatch.setattr(an, "ANNOTATION_DIR", tmp_path)
    at = _annotate_app()
    assert any("pool.csv" in e.value for e in at.error)
    pd.DataFrame([{"id": "v1", "text": "xin chào các bạn", "source": "x"}]).to_csv(tmp_path / "pool.csv", index=False)
    at = _annotate_app()
    assert any("Nhập tên" in i.value for i in at.info)


def test_annotate_app_saves_labels_and_never_shows_bot_predictions(annotate_env):
    at = _annotate_app()
    at.sidebar.text_input[0].set_value("an").run()
    assert not at.exception and "mất ngủ cả tuần" in " ".join(i.value for i in at.info)
    page = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    for leaked in ("bot dự đoán", "mô hình dự đoán", "xác suất"):
        assert leaked not in page
    next(r for r in at.radio if r.label.startswith("Người viết")).set_value("Có dấu hiệu căng thẳng")
    next(s for s in at.selectbox if "Giấc ngủ" in s.label).set_value("Đáng chú ý")
    at.checkbox[0].check()
    next(b for b in at.button if b.label.startswith("Lưu")).click().run()
    assert not at.exception
    saved = an.load_annotations("an").set_index("id")
    assert saved.loc["v0001", "stress"] == "co" and saved.loc["v0001", "dom_sleep"] == "concern" and saved.loc["v0001", "crisis"] == "1"
    assert "trời đẹp" in " ".join(i.value for i in at.info)                      # chuyển sang mục kế tiếp
    next(b for b in at.button if b.label.startswith("Lưu")).click().run()
    assert any("gán xong" in s.value for s in at.success)


# ---------- đánh giá trên tập vàng ----------
def make_gold():
    rows = []
    for i in range(12):
        crisis = i < 4
        rows.append(row(i, text="mình muốn chết" if crisis else "hôm nay đi học", crisis="1" if crisis else "0",
                        stress="co" if i % 2 else "khong", dom_sleep="concern" if i in (1, 2) else ""))
    rows.append(row(99, text="t muốn đi gặp ông bà luôn", crisis="1"))    # cố ý: luật bỏ sót, cần ngữ cảnh/LLM
    return frame(rows)


def test_crisis_and_domain_evaluation_on_gold():
    gold = make_gold()
    c = eg.eval_crisis(gold)
    assert c["tp"] == 4 and c["fn"] == 1 and c["fp"] == 0 and c["sensitivity"] == pytest.approx(0.8)
    assert eg.eval_crisis(gold, lambda t: True)["sensitivity"] == 1.0
    d = eg.eval_domains(gold, predict=lambda t: {"sleep": "concern"} if "đi học" in t else {})
    assert d["sleep"]["n"] == len(gold) and "mood" in d and d["sleep"]["tp"] <= 2


def test_stress_evaluation_uses_only_clear_labels():
    gold = make_gold()
    gold.loc[0, "stress"] = "khong_ro"
    res = eg.eval_stress(gold, lambda t: 0.9 if "muốn chết" in t else 0.1)
    assert res["n"] == len(gold) - 1 and 0 <= res["auc"] <= 1
    tiny = eg.eval_stress(gold.head(5), lambda t: 0.5)
    assert "note" in tiny


def test_load_gold_reads_blank_fields_as_empty_strings(tmp_path):
    path = tmp_path / "g.csv"
    make_gold().to_csv(path, index=False)
    assert (eg.load_gold(path)["dom_mood"] == "").all()
