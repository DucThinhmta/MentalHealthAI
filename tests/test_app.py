"""Chạy giao diện Streamlit bằng AppTest với mô hình giả (không cần torch/GPU)."""
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import pytest
from streamlit.testing.v1 import AppTest

import config


def _fake_analyze(raise_missing=False):
    mod = types.ModuleType("analyze")

    def analyze_conversation(msgs, window=5):
        if raise_missing:
            raise FileNotFoundError("bert_stress")
        text = " ".join(msgs[-window:]).lower()
        stressed = "stress" in text or "exam" in text
        return {"stress": config.STRESS_YES if stressed else config.STRESS_NO,
                "stress_confidence": 0.9, "stress_probability": 0.9 if stressed else 0.1,
                "emotions": [("fear", 0.4)] if stressed else [],
                "depression": "severe" if "hopeless" in text else "minimum",
                "depression_confidence": 0.8,
                "depression_probs": {"minimum": 0.7, "mild": 0.1, "moderate": 0.1, "severe": 0.1},
                "n_messages_used": min(window, len(msgs))}
    mod.analyze_conversation = analyze_conversation
    return mod


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MHAI_TRANSLATOR", "none")           # không dùng mô hình dịch đã tải trên máy
    sys.modules["analyze"] = _fake_analyze()
    yield
    sys.modules.pop("analyze", None)


def _app():
    at = AppTest.from_file(str(SRC / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def _say(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception
    return at


def _markdown(at):
    return "\n".join(m.value for m in at.markdown)


def _history(at):
    return at.session_state["sess"].history


def _button(at, key):
    return next(b for b in at.button if b.key == key)


def test_initial_state_has_no_technical_settings():
    at = _app()
    md = _markdown(at)
    assert "Chào bạn" in md and "Hôm nay bạn thấy thế nào?" in md
    assert sum(1 for b in at.button if b.key and b.key.startswith("mood_")) == 5
    assert sum(1 for b in at.button if b.key and b.key.startswith("sugg_")) == 4
    labels = [t.label for t in at.toggle]
    assert "Trò chuyện bằng LLM" not in labels and not at.slider          # cài đặt kỹ thuật đã bỏ
    assert "Chữ lớn, dễ đọc" in labels
    assert "Mình cần nghe thêm" in md                                      # chưa đủ dữ liệu để nhận xét


def test_mood_checkin_records_self_rating_and_mood_domain():
    at = _app()
    _button(at, "mood_2").click().run()
    assert not at.exception
    sess = at.session_state["sess"]
    assert sess.self_moods == [2] and sess.entries[0]["domains"]["mood"] == "concern"
    assert [m["role"] for m in sess.history] == ["user", "assistant"]


def test_topic_starter_starts_conversation():
    at = _app()
    _button(at, "sugg_0").click().run()
    assert not at.exception
    assert "áp lực" in _history(at)[0]["content"] and _history(at)[1]["role"] == "assistant"


def test_normal_turn_logs_and_shows_conversation_panel():
    at = _say(_app(), "I am so stressed about my exam")
    assert [m["role"] for m in _history(at)] == ["user", "assistant"]
    entry = at.session_state["log"][0]
    assert entry["reply_source"] == "rules" and entry["analyzed"] and entry["stress_probability"] == 0.9
    assert entry["conversation"]["evidence"] == "low"
    assert "Các khía cạnh sinh hoạt" in _markdown(at) and "Học tập" in _markdown(at)


def test_quick_reply_answers_the_domain_question():
    at = _app()
    quick = []
    for text in ["Dạo này mình mệt mỏi và áp lực chuyện học quá", "Mình thấy nặng nề lắm", "Ừ"]:
        at = _say(at, text)                                                # bot xen kẽ hỏi chủ đề / hỏi khía cạnh
        quick = [b for b in at.button if b.key and b.key.startswith("quick_")]
        if quick:
            break
    assert quick, [m["content"] for m in _history(at)]                    # bot đã hỏi một khía cạnh
    dom = quick[0].key.split("_")[1]
    quick[-1].click().run()                                                # chọn đáp án "đáng chú ý"
    assert not at.exception
    assert at.session_state["sess"].entries[-1]["domains"][dom] == "concern"


def test_smalltalk_is_answered_but_not_analyzed():
    at = _say(_app(), "hello")
    assert _history(at)[1]["content"].startswith("Hello!")
    assert at.session_state["sess"].entries == []
    assert at.session_state["log"][0]["analyzed"] is False


def test_crisis_banner_and_static_reply():
    at = _say(_app(), "I want to kill myself")
    assert at.session_state["last_safety"] == "crisis"
    assert "115" in _history(at)[1]["content"] and "tự hại" in _markdown(at)
    assert at.session_state["log"][0]["reply_source"] == "safety"


def test_crisis_reply_does_not_need_models():
    sys.modules["analyze"] = _fake_analyze(raise_missing=True)
    at = _say(_app(), "Tôi muốn chết")
    assert at.session_state["last_safety"] == "crisis" and "115" in _history(at)[1]["content"]


def test_vietnamese_without_llm_is_not_scored_by_english_models():
    at = _say(_app(), "Tôi rất căng thẳng vì kỳ thi sắp tới, mất ngủ mấy tuần rồi")
    e = at.session_state["sess"].entries[0]
    assert e["model_ok"] is False                                          # không dùng điểm mô hình cho tiếng Việt chưa dịch
    assert e["domains"]["sleep"] == "concern" and e["domains"]["duration"] == "concern"


def test_summary_and_breathing_dialogs_and_new_chat():
    at = _app()
    for text in ["I am stressed about my exam", "I can't sleep and I feel exhausted",
                 "nobody to talk to", "it has been like this for weeks"]:
        at = _say(at, text)
    _button(at, "open_summary").click().run()
    assert not at.exception and at.session_state["summary_seen"]
    assert "Gợi ý cho bạn" in _markdown(at)
    _button(at, "breathe").click().run()
    assert not at.exception and "Hít vào" in _markdown(at)
    _button(at, "new_chat").click().run()
    assert _history(at) == [] and at.session_state["log"] == []


def test_large_text_toggle():
    at = _app()
    next(t for t in at.toggle if t.label == "Chữ lớn, dễ đọc").set_value(True).run()
    assert not at.exception and "font-size:19px" in _markdown(at)


def test_missing_models_still_chats_and_assesses_from_answers():
    sys.modules["analyze"] = _fake_analyze(raise_missing=True)
    at = _say(_app(), "I can't sleep and I feel exhausted")
    assert [m["role"] for m in _history(at)] == ["user", "assistant"]
    sess = at.session_state["sess"]
    assert sess.models_missing and not sess.entries[0]["model_ok"]
    assert sess.entries[0]["domains"] == {"sleep": "concern", "energy": "concern"}
    assert any("Chưa tải mô hình" in c.value for c in at.caption)
