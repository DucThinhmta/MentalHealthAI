"""Chế độ nghiên cứu (study.py), phân tích kiểm chứng (analyze_study.py) và bảng hỏi trong ứng dụng."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import analyze_study as an
import config
import study

SRC = Path(__file__).resolve().parents[1] / "src"


# ---------- chấm điểm ----------
def test_instruments_are_complete():
    assert len(study.INSTRUMENTS["phq9"]["items"]) == 9 and len(study.INSTRUMENTS["gad7"]["items"]) == 7
    assert [o[0] for o in study.OPTIONS] == [0, 1, 2, 3]
    for inst in study.INSTRUMENTS.values():
        assert all(vi and en for vi, en in inst["items"])
        lows = [b[0] for b in inst["bands"]]
        assert lows[0] == 0 and inst["bands"][-1][1] == inst["max"]
        assert all(inst["bands"][i][1] + 1 == inst["bands"][i + 1][0] for i in range(len(inst["bands"]) - 1))  # không hở, không chồng


@pytest.mark.parametrize("total,band", [(0, 0), (4, 0), (5, 1), (9, 1), (10, 2), (14, 2), (15, 3), (19, 3), (20, 4), (27, 4)])
def test_phq9_bands_follow_the_published_cutoffs(total, band):
    answers = [0] * 9
    left = total
    for i in range(9):
        answers[i] = min(3, left)
        left -= answers[i]
    assert study.score("phq9", answers)[:2] == (total, band)


@pytest.mark.parametrize("total,band", [(0, 0), (4, 0), (5, 1), (9, 1), (10, 2), (14, 2), (15, 3), (21, 3)])
def test_gad7_bands_follow_the_published_cutoffs(total, band):
    answers = [0] * 7
    left = total
    for i in range(7):
        answers[i] = min(3, left)
        left -= answers[i]
    assert study.score("gad7", answers)[:2] == (total, band)


def test_score_validates_input():
    with pytest.raises(ValueError):
        study.score("phq9", [0] * 8)
    with pytest.raises(ValueError):
        study.score("phq9", [0] * 8 + [4])
    with pytest.raises(ValueError):
        study.score("khác", [0])


def test_item9_flag_and_feedback_are_cautious():
    safe = [0] * 9
    risky = [0] * 8 + [1]
    assert not study.item9_positive(safe) and study.item9_positive(risky)
    text = " ".join(study.feedback(risky, [0] * 7))
    assert "không phải chẩn đoán" in text and "Cảm ơn bạn đã nói ra" in text
    high = " ".join(study.feedback([2] * 9, [3] * 7))
    assert "bác sĩ hoặc chuyên gia" in high
    assert "bác sĩ hoặc chuyên gia" not in " ".join(study.feedback([0] * 9, [0] * 7))


def test_domain_mapping_points_at_the_right_phq9_items():
    items = study.INSTRUMENTS["phq9"]["items"]
    for domain, idx in study.DOMAIN_TO_PHQ9_ITEM.items():
        assert 0 <= idx < 9 and idx != study.PHQ9_SUICIDE_ITEM
    assert "sleep" in items[study.DOMAIN_TO_PHQ9_ITEM["sleep"]][1].lower()
    assert "tired" in items[study.DOMAIN_TO_PHQ9_ITEM["energy"]][1].lower()
    assert "appetite" in items[study.DOMAIN_TO_PHQ9_ITEM["appetite"]][1].lower()
    assert "concentrat" in items[study.DOMAIN_TO_PHQ9_ITEM["focus"]][1].lower()


def test_record_has_no_conversation_text_and_roundtrips(tmp_path):
    profile = {"score": 0.62, "level": 2, "evidence": "high", "n_turns": 7, "n_words": 90, "domains": {"sleep": "concern"},
               "self_moods": [2], "stress": {"overall": 0.7, "recent": 0.8, "share": 0.6, "series": [0.7]},
               "depression": {"score": 1.4, "label": "mild", "series": [1.4]}, "components": {"stress": 0.7}}
    rec = study.build_record("P-ABC123", profile, [1] * 9, [2] * 7)
    blob = json.dumps(rec, ensure_ascii=False)
    for forbidden in ("text", "english", "user", "message"):
        assert f'"{forbidden}"' not in blob
    assert rec["phq9"]["total"] == 9 and rec["gad7"]["total"] == 14 and rec["bot"]["stress"]["overall"] == 0.7
    path = tmp_path / "r.jsonl"
    study.save_record(rec, path)
    study.save_record(rec, path)
    path.write_text(path.read_text(encoding="utf-8") + "{hỏng\n", encoding="utf-8")
    assert len(study.load_records(path)) == 2 and study.load_records(tmp_path / "none.jsonl") == []


# ---------- thống kê ----------
def test_sample_size_matches_textbook_values():
    assert an.n_for_correlation(0.3) == 85 and an.n_for_correlation(0.5) == 30
    assert an.n_for_correlation(0.2) > an.n_for_correlation(0.4)
    assert an.n_for_auc(0.75, 0.10, 0.3) < an.n_for_auc(0.75, 0.05, 0.3)
    assert 0 < an.auc_se(0.8, 40, 60) < 0.1


def test_binary_metrics_and_kappa():
    m = an.binary_metrics([1, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1])
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 1, 2, 1) and m["sensitivity"] == pytest.approx(2 / 3)
    assert an.binary_metrics([1, 1, 1], [1, 1, 1])["kappa"] != an.binary_metrics([1, 1, 1], [1, 1, 1])["kappa"]   # NaN (một lớp)


def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    y = x + rng.normal(size=80)
    est, (lo, hi) = an.bootstrap(an._spearman, [x, y], n_boot=400, seed=3)
    assert lo < est < hi and an.bootstrap(an._spearman, [x, y], n_boot=400, seed=3) == (est, (lo, hi))


def test_analysis_detects_a_real_signal_and_a_null_signal():
    good = an.analyze(an.synthetic_records(150, seed=1, signal=0.8), n_boot=300)
    null = an.analyze(an.synthetic_records(150, seed=1, signal=0.0), n_boot=300)
    assert good["correlation"]["PHQ-9"]["spearman"] > 0.4 and good["auc"]["PHQ-9>=10"]["auc"] > 0.7
    assert abs(null["correlation"]["PHQ-9"]["spearman"]) < 0.25 and null["auc"]["PHQ-9>=10"]["auc"] < 0.65
    assert good["weighted_kappa"]["PHQ-9"]["kappa"] > null["weighted_kappa"]["PHQ-9"]["kappa"]
    assert good["symptom_level"]["sleep"]["sensitivity"] > 0.5 and good["n_analyzed"] == 150


def test_analysis_filters_by_evidence_and_handles_tiny_samples():
    recs = an.synthetic_records(30, seed=2)
    for r in recs[:20]:
        r["bot"]["evidence"] = "low"
    assert an.analyze(recs, "medium", 50)["n_analyzed"] == 10
    assert an.analyze(recs, "low", 50)["n_analyzed"] == 30
    tiny = an.analyze(recs[:3], "low", 50)
    assert "note" in tiny and "Quá ít" in an.format_report(tiny)
    assert an.analyze([], "low", 10)["n_analyzed"] == 0


def test_report_marks_synthetic_data_loudly():
    res = an.analyze(an.synthetic_records(60, seed=4), n_boot=100)
    assert "DỮ LIỆU GIẢ LẬP" in an.format_report(res, synthetic=True)
    assert "DỮ LIỆU GIẢ LẬP" not in an.format_report(res)
    assert "Lưu ý" in an.format_report(res)


def test_all_one_class_gives_a_note_not_a_crash():
    recs = an.synthetic_records(40, seed=5)
    for r in recs:
        r["phq9"]["total"], r["gad7"]["total"] = 2, 2
    res = an.analyze(recs, "low", 50)
    assert "note" in res["auc"]["PHQ-9>=10"] and math.isnan(an.binary_metrics([0, 0], [0, 1])["sensitivity"])


# ---------- bảng hỏi trong ứng dụng ----------
def _app(research=True, minor=False):
    at = AppTest.from_file(str(SRC / "app.py"), default_timeout=60)
    at.session_state["consent"] = {"minor": minor, "research": research}
    at.run()
    assert not at.exception
    return at


def _say(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception
    return at


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MHAI_TRANSLATOR", "none")
    monkeypatch.setattr(config, "STUDY_DIR", tmp_path / "study")


def _chat_enough(at):
    for text in ["Mình mất ngủ cả tuần nay", "Lo thi quá", "Mệt lắm", "Chẳng có ai để kể", "Ăn uống bỏ bữa"]:
        at = _say(at, text)
    return at


def _fill(at, answers):
    for key, value in answers.items():
        next(r for r in at.radio if r.key == key).set_value(value)


def test_study_panel_is_hidden_without_research_consent_and_for_minors():
    for kw in ({"research": False}, {"research": True, "minor": True}):
        at = _app(**kw)
        assert not any("Tham gia nghiên cứu" in e.label for e in at.expander)


def test_study_panel_asks_for_more_conversation_first():
    at = _say(_app(), "Mình mệt")
    assert any("Hãy trò chuyện thêm" in i.value for i in at.info)
    assert not any(r.key and r.key.startswith("study_") for r in at.radio)


def test_study_form_requires_every_answer_then_saves_a_record_without_text(tmp_path):
    at = _chat_enough(_app())
    assert any(r.key == "study_phq9_0" for r in at.radio) and any("CHƯA được kiểm định" in w.value for w in at.warning)
    submit = next(b for b in at.button if b.label == "Gửi bảng hỏi")
    _fill(at, {"study_phq9_0": "Vài ngày"})
    submit.click().run()
    assert any("trả lời tất cả" in e.value for e in at.error) and study.load_records() == []
    labels = {v: label for v, label, _ in study.OPTIONS}
    # 8 câu đầu "Vài ngày"; câu 9 (ý nghĩ tự hại) = 0 để không kích hoạt luồng an toàn (đã có test riêng)
    _fill(at, {f"study_phq9_{i}": labels[1] for i in range(8)} | {"study_phq9_8": labels[0]}
          | {f"study_gad7_{i}": labels[2] for i in range(7)})
    next(b for b in at.button if b.label == "Gửi bảng hỏi").click().run()
    assert not at.exception
    recs = study.load_records()
    assert len(recs) == 1 and recs[0]["phq9"]["total"] == 8 and recs[0]["gad7"]["total"] == 14
    assert recs[0]["code"].startswith("P-") and recs[0]["bot"]["n_turns"] == 5
    assert "Mình mất ngủ" not in (config.STUDY_DIR / "records.jsonl").read_text(encoding="utf-8")
    assert not at.session_state["last_safety"] == "crisis"


def test_phq9_item9_triggers_the_support_resources_immediately():
    at = _chat_enough(_app())
    labels = {v: label for v, label, _ in study.OPTIONS}
    answers = {f"study_phq9_{i}": labels[0] for i in range(9)} | {f"study_gad7_{i}": labels[0] for i in range(7)}
    answers["study_phq9_8"] = labels[1]                                  # câu 9: "Vài ngày"
    _fill(at, answers)
    next(b for b in at.button if b.label == "Gửi bảng hỏi").click().run()
    assert at.session_state["last_safety"] == "crisis"
    text = "\n".join(m.value for m in at.markdown)
    assert "115" in text and "Cảm ơn bạn đã nói ra" in text
    assert study.load_records()[0]["phq9"]["item9_positive"] is True
