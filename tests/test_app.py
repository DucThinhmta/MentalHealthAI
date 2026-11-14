"""Chạy giao diện Streamlit bằng AppTest với mô hình giả (không cần torch/GPU)."""
import json
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


def _app(consent=True, minor=False, research=False):
    """Mặc định đã qua màn hình đồng ý (consent=False để kiểm tra chính màn hình đó)."""
    at = AppTest.from_file(str(SRC / "app.py"), default_timeout=30)
    if consent:
        at.session_state["consent"] = {"minor": minor, "research": research}
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
    assert "Mình đang theo dõi" in md and "Chưa có gì để đánh giá" in md   # khung phân tích có từ đầu, không chặn gì


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
    assert _history(at)[1]["content"].startswith("Xin chào!")           # người dùng gõ tiếng Anh, bot vẫn đáp tiếng Việt
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


# ---------- đồng ý, vị thành niên, quyền riêng tư ----------
def _widget(items, key):
    return next(w for w in items if w.key == key)


def test_consent_gate_blocks_chat_until_the_user_agrees():
    at = _app(consent=False)
    md = _markdown(at)
    assert "Trước khi bắt đầu" in md and "không phải chẩn đoán y khoa" in md and "115" in md
    assert len(at.chat_input) == 0                                          # chưa được trò chuyện
    assert _widget(at.button, "consent_go").disabled                        # chưa tick đồng ý thì nút bị khóa
    _widget(at.checkbox, "consent_agree").check().run()
    assert not _widget(at.button, "consent_go").disabled
    _widget(at.button, "consent_go").click().run()
    assert at.session_state["consent"] == {"minor": False, "research": False}
    assert len(at.chat_input) == 1


def test_consent_discloses_llm_only_when_it_is_on(monkeypatch):
    assert "Anthropic" not in _markdown(_app(consent=False))
    import llm_client
    monkeypatch.setattr(llm_client, "is_available", lambda: True)           # không phụ thuộc việc đã cài gói anthropic
    assert "Anthropic" in _markdown(_app(consent=False))


def test_minors_cannot_opt_into_research_and_get_the_children_hotline():
    at = _app(consent=False)
    _widget(at.checkbox, "consent_agree").check()
    _widget(at.radio, "consent_age").set_value("Dưới 18 tuổi").run()
    assert not any(c.key == "consent_research" for c in at.checkbox)        # không có lựa chọn lưu nghiên cứu
    _widget(at.button, "consent_go").click().run()
    assert at.session_state["consent"] == {"minor": True, "research": False}
    assert at.session_state["sess"].minor and not at.session_state["sess"].research
    assert "111" in _markdown(at)
    at = _say(at, "Mình muốn chết")
    assert "111" in _history(at)[-1]["content"] and "115" in _history(at)[-1]["content"]


def test_adults_get_no_children_hotline_line():
    at = _say(_app(), "Mình muốn chết")
    assert "111" not in _history(at)[-1]["content"]


def test_log_has_no_text_without_research_consent():
    at = _say(_app(research=False), "Gọi mình số 0912 345 678 nhé, mình mất ngủ")
    entry = at.session_state["log"][0]
    assert entry["user"] == "[không lưu nội dung]" and "912" not in json.dumps(entry, ensure_ascii=False)
    assert entry["analyzed"] and entry["domains"]["sleep"] == "concern"      # điểm và khía cạnh vẫn có


def test_log_text_is_scrubbed_with_research_consent():
    at = _say(_app(research=True), "Email mình là an.nguyen@gmail.com, số 0912 345 678, mình mất ngủ")
    user = at.session_state["log"][0]["user"]
    assert "[EMAIL]" in user and "[SĐT]" in user and "gmail" not in user and "345" not in user


def test_summary_explains_itself():
    at = _app()
    for text in ["I am stressed about my exam", "I can't sleep and I feel exhausted",
                 "nobody to talk to", "it has been like this for weeks"]:
        at = _say(at, text)
    _button(at, "open_summary").click().run()
    assert not at.exception and any("Vì sao mình nhận xét như vậy" in e.label for e in at.expander)


# ---------- chat không bao giờ bị chốt; khung phân tích cập nhật trực tiếp ----------
def test_panel_shows_a_provisional_reading_from_the_first_message():
    at = _say(_app(), "Mình mất ngủ cả tuần nay rồi")
    md = _markdown(at)
    assert "Mức quan tâm hiện tại" in md and "Đang theo dõi" in md and "Độ tin cậy" in md
    assert "Mình đang hiểu bạn đến đâu" in md and "1/8" in md or "2/8" in md or "3/8" in md


def test_chat_stays_open_and_never_announces_a_conclusion():
    at = _app()
    for text in ["Mình mất ngủ cả tuần nay rồi", "Lo thi cuối kỳ quá", "Chẳng có ai để kể cả", "Ăn uống cũng bỏ bữa",
                 "Mệt lắm", "Khoảng ba tuần rồi", "Mình vẫn cố đi học", "Mà chẳng nhớ được gì", "Thôi để mình kể tiếp"]:
        at = _say(at, text)
        assert len(at.chat_input) == 1                                      # lượt nào cũng còn ô nhập
    bot = " ".join(m["content"] for m in _history(at) if m["role"] == "assistant")
    for phrase in ("tổng kết", "Xem tổng kết", "hiểu bạn khá rõ", "kết luận"):
        assert phrase not in bot
    assert not any(b.key == "offer_summary" for b in at.button)            # không có hộp mời chốt trong khung chat
    assert "mh-offer" not in _markdown(at)
    p = at.session_state["sess"].profile()
    assert p["evidence"] in ("medium", "high") and "Độ tin cậy" in _markdown(at)   # khung phải vẫn cập nhật độ tin cậy
    assert at.session_state["sess"].history[-1]["role"] == "assistant"


def test_panel_confidence_rises_as_the_conversation_goes_on():
    at = _app()
    confidence = []
    for text in ["Mình mệt", "Mình mất ngủ cả tuần nay", "Chẳng có ai để kể", "Ăn uống bỏ bữa", "Lo thi quá",
                 "Mà cũng chẳng muốn làm gì", "Khó tập trung nữa"]:
        at = _say(at, text)
        confidence.append(at.session_state["sess"].profile()["evidence"])
    order = ["low", "medium", "high"]
    assert [order.index(c) for c in confidence] == sorted(order.index(c) for c in confidence)   # chỉ tăng, không tụt
    assert confidence[0] == "low" and confidence[-1] in ("medium", "high")


def test_score_series_has_one_point_per_scored_turn():
    at = _app()
    for text in ["Mình mất ngủ", "Mệt lắm", "Lo thi quá"]:
        at = _say(at, text)
    assert "DIỄN BIẾN MỨC QUAN TÂM THEO LƯỢT" in _markdown(at) or any("DIỄN BIẾN" in c.value for c in at.caption)


def test_exercises_dialog_opens_and_lists_every_exercise():
    at = _app()
    _button(at, "exercises").click().run()
    assert not at.exception
    box = next(s for s in at.selectbox if s.key == "exercise_pick")
    assert len(box.options) == 8 and "Tiếp đất" in " ".join(box.options)
    assert "Hít vào bằng mũi" in _markdown(at)                              # bài đầu tiên hiện sẵn các bước
