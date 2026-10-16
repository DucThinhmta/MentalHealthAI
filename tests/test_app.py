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


def test_initial_state():
    at = _app()
    assert any("bắt đầu" in i.value for i in at.info)
    assert at.session_state["messages"] == []


def test_normal_turn_shows_results_and_rule_reply():
    at = _say(_app(), "I am so stressed about my exam")
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["content"]
    assert any("Có dấu hiệu stress" in e.value for e in at.error)
    assert at.session_state["log"][0]["reply_source"] == "rules"
    assert at.session_state["last_safety"] == "none"


def test_crisis_banner_and_static_reply():
    at = _say(_app(), "I want to kill myself")
    assert at.session_state["last_safety"] == "crisis"
    assert "115" in at.session_state["messages"][1]["content"]
    assert any("tự hại" in e.value for e in at.error)
    assert at.session_state["log"][0]["reply_source"] == "safety"


def test_vietnamese_without_llm_warns():
    at = _say(_app(), "Tôi rất căng thẳng vì kỳ thi sắp tới")
    assert any("tiếng Việt" in w.value for w in at.warning)


def test_vietnamese_crisis_detected_without_llm():
    at = _say(_app(), "Tôi muốn chết")
    assert at.session_state["last_safety"] == "crisis"


def test_multi_turn_trend_and_clear():
    at = _say(_app(), "I am fine today")
    at = _say(at, "But the exam is stressing me out")
    assert len(at.session_state["log"]) == 2 and len(at.session_state["analysis_msgs"]) == 2
    at.button[0].click().run()
    assert at.session_state["messages"] == [] and at.session_state["last_result"] is None


def test_missing_model_shows_friendly_error():
    sys.modules["analyze"] = _fake_analyze(raise_missing=True)
    at = _app()
    at.chat_input[0].set_value("hello").run()
    assert any("Chưa tìm thấy mô hình" in e.value for e in at.error)
    assert at.session_state["messages"] == []
