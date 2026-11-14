"""Nhật ký tâm trạng (diary.py, trong app) và bảng điều khiển tổng hợp ẩn danh (dashboard.py)."""
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from streamlit.testing.v1 import AppTest

import analyze_study as an
import config
import dashboard as db
import diary
import study

SRC = Path(__file__).resolve().parents[1] / "src"
D = datetime.date


# ---------- nhật ký ----------
def test_codes_and_validation(tmp_path):
    assert diary.valid_code("P-1A2B3C") and diary.valid_code(" p-1a2b3c ") and not diary.valid_code("P-123") and not diary.valid_code("../x")
    with pytest.raises(ValueError):
        diary.load("../../etc/passwd", tmp_path)
    for bad in ({"mood": 0}, {"mood": 6}, {"mood": 3, "sleep_hours": 30}, {"mood": 3, "stress": 11}):
        with pytest.raises(ValueError):
            diary.add_entry("P-AAAAAA", directory=tmp_path, **bad)


def test_one_entry_per_day_and_deletion(tmp_path):
    diary.add_entry("P-AAAAAA", 2, 5.0, 8, date=D(2026, 11, 1), directory=tmp_path)
    diary.add_entry("P-AAAAAA", 4, 7.0, 3, date=D(2026, 11, 1), directory=tmp_path)             # cùng ngày: ghi đè
    entries = diary.add_entry("P-AAAAAA", 3, date=D(2026, 11, 2), directory=tmp_path)
    assert [(e["date"], e["mood"]) for e in entries] == [("2026-11-01", 4), ("2026-11-02", 3)]
    assert diary.load("p-aaaaaa", tmp_path) == entries                                         # mã không phân biệt hoa thường
    assert diary.delete("P-AAAAAA", tmp_path) and not diary.delete("P-AAAAAA", tmp_path) and diary.load("P-AAAAAA", tmp_path) == []


def test_the_diary_stores_numbers_only(tmp_path):
    diary.add_entry("P-BBBBBB", 2, 4.5, 9, date=D(2026, 11, 1), directory=tmp_path)
    raw = (tmp_path / "P-BBBBBB.jsonl").read_text(encoding="utf-8")
    assert set(json.loads(raw)) == {"date", "mood", "sleep_hours", "stress"}


def test_low_streak_counts_consecutive_days_only():
    mk = lambda days_moods: [{"date": D(2026, 11, d).isoformat(), "mood": m} for d, m in days_moods]
    assert diary.low_streak(mk([(1, 5), (2, 2), (3, 1), (4, 2)])) == 3
    assert diary.low_streak(mk([(1, 2), (2, 2), (4, 2), (5, 2)])) == 2                         # ngày 3 bị thiếu: chuỗi đứt
    assert diary.low_streak(mk([(1, 2), (2, 3)])) == 0 and diary.low_streak([]) == 0


def test_summary_and_alert(tmp_path):
    for i in range(6):
        diary.add_entry("P-CCCCCC", 2 if i >= 1 else 4, 5.0, date=D(2026, 11, 1) + datetime.timedelta(days=i), directory=tmp_path)
    s = diary.summary(diary.load("P-CCCCCC", tmp_path))
    assert s["n"] == 6 and s["low_streak"] == 5 and s["alert"] and s["trend"] < 0 and s["sleep_mean"] == 5.0
    assert "5 ngày liên tiếp" in diary.alert_text(5) and diary.summary([]) == {"n": 0}
    ok = diary.summary([{"date": "2026-11-01", "mood": 4, "sleep_hours": None, "stress": None}])
    assert not ok["alert"] and ok["sleep_mean"] is None


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MHAI_TRANSLATOR", "none")
    monkeypatch.setattr(diary, "DIARY_DIR", tmp_path / "diary")


def _app():
    at = AppTest.from_file(str(SRC / "app.py"), default_timeout=30)
    at.session_state["consent"] = {"minor": False, "research": False}
    at.run()
    assert not at.exception
    return at


def test_diary_in_the_app_create_save_and_delete(app_env):
    at = _app()
    next(b for b in at.button if b.key == "diary_new").click().run()
    code = at.session_state["diary_code"]
    assert diary.valid_code(code) and code in " ".join(m.value for m in at.markdown)
    next(r for r in at.radio if r.key == "diary_mood").set_value("2 · Không ổn")
    next(b for b in at.button if b.key == "diary_save").click().run()
    assert not at.exception and diary.load(code)[0]["mood"] == 2
    next(b for b in at.button if b.key == "diary_delete").click().run()
    assert at.session_state["diary_code"] is None and diary.load(code) == []


def test_diary_in_the_app_rejects_a_malformed_code(app_env):
    at = _app()
    next(t for t in at.text_input if t.key == "diary_enter").set_value("abc").run()
    assert any("không đúng dạng" in e.value for e in at.error) and at.session_state["diary_code"] is None
    next(t for t in at.text_input if t.key == "diary_enter").set_value("p-0a1b2c").run()
    assert at.session_state["diary_code"] == "P-0A1B2C"


# ---------- bảng điều khiển ----------
def test_small_counts_are_suppressed():
    assert db.suppress(0) == 0 and db.suppress(1) == "<5" and db.suppress(4) == "<5" and db.suppress(5) == 5 and db.suppress(40) == 40


def test_access_code_is_required_and_compared_safely(monkeypatch):
    monkeypatch.delenv("MHAI_ADMIN_CODE", raising=False)
    assert not db.authorized("") and not db.authorized("anything")                              # chưa đặt mã: luôn từ chối
    monkeypatch.setenv("MHAI_ADMIN_CODE", "s3cret")
    assert db.authorized("s3cret") and not db.authorized("S3CRET") and not db.authorized("")
    assert db.authorized("x", expected="x") and not db.authorized("x", expected="")


def test_study_aggregate_contains_no_participant_codes_and_no_text():
    recs = an.synthetic_records(40, seed=3)
    agg = db.aggregate_study(recs)
    blob = json.dumps(agg, default=str)
    assert agg["n"] == 40 and "SYN-" not in blob and agg["item9_positive"] >= 0
    assert set(agg["domain_concern"]["sleep"]) == {"concern", "known", "rate"} and 0 <= agg["domain_concern"]["sleep"]["rate"] <= 1
    assert db.aggregate_study([]) == {"n": 0}


def test_weekly_scores_and_drift_flag():
    recs = an.synthetic_records(8, seed=1)
    for i, r in enumerate(recs):
        r["date"] = (D(2026, 10, 5) + datetime.timedelta(weeks=i)).isoformat()
        r["bot"]["score"] = 0.3 if i < 4 else 0.7                                               # bước nhảy lớn giữa chừng
    weekly = db.weekly_scores(recs)
    assert len(weekly) == 8 and all(n == 1 for _, n in weekly.values())
    d = db.drift(weekly)
    assert d["flag"] and d["shift"] == pytest.approx(0.4)
    assert not db.drift({k: (0.5, 3) for k in weekly})["flag"]
    assert not db.drift(dict(list(weekly.items())[:2]))["flag"] and "cần ít nhất" in db.drift({})["note"]


def test_log_aggregate():
    entries = [{"safety_level": "crisis", "reply_source": "safety", "risk_llm": "high", "analyzed": True, "model_ok": False},
               {"safety_level": "none", "reply_source": "rules", "analyzed": True, "model_ok": True},
               {"safety_level": "none", "reply_source": "smalltalk", "analyzed": False}]
    agg = db.aggregate_log(entries)
    assert agg["turns"] == 3 and agg["safety_counts"] == {"crisis": 1, "none": 2} and agg["risk_llm_counts"]["n/a"] == 2
    assert agg["analyzed_rate"] == pytest.approx(2 / 3) and agg["model_scored_rate"] == 0.5 and agg["crisis_rate"] == pytest.approx(1 / 3)
    assert db.aggregate_log([]) == {"turns": 0}


def _dash(monkeypatch, tmp_path, code="s3cret", entered=None, n_records=0):
    monkeypatch.setattr(config, "STUDY_DIR", tmp_path)
    for r in an.synthetic_records(n_records, seed=2):
        study.save_record(r)
    if code is None:
        monkeypatch.delenv("MHAI_ADMIN_CODE", raising=False)
    else:
        monkeypatch.setenv("MHAI_ADMIN_CODE", code)
    at = AppTest.from_file(str(SRC / "dashboard.py"), default_timeout=30)
    at.run()
    if entered is not None:
        at.text_input[0].set_value(entered).run()
    assert not at.exception
    return at


def test_dashboard_refuses_without_a_configured_code(monkeypatch, tmp_path):
    at = _dash(monkeypatch, tmp_path, code=None, n_records=10)
    assert any("MHAI_ADMIN_CODE" in e.value for e in at.error) and not at.metric


def test_dashboard_hides_data_until_the_right_code(monkeypatch, tmp_path):
    at = _dash(monkeypatch, tmp_path, n_records=12, entered="sai")
    assert not at.metric and any("Nhập mã" in i.value for i in at.info)
    at = _dash(monkeypatch, tmp_path, n_records=12, entered="s3cret")
    assert len(at.metric) == 3


def test_dashboard_withholds_aggregates_below_the_minimum_cell(monkeypatch, tmp_path):
    at = _dash(monkeypatch, tmp_path, n_records=3, entered="s3cret")
    assert not at.metric and any("Chưa đủ" in i.value for i in at.info)
    assert "<5" in at.subheader[0].value
