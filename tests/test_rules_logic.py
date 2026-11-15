"""Logic trả lời theo luật (không LLM): câu kể không bị nhầm là câu hỏi, nhận ra vui/buồn/khó khăn, câu hỏi quyết định, xã giao mở rộng."""
import random
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import re

import pytest

import pipeline
import response_generator as rg
import smalltalk


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MHAI_TRANSLATOR", "none")
    mod = types.ModuleType("analyze")
    mod.analyze_conversation = lambda msgs, window=5: {
        "stress_probability": 0.1, "emotions": [], "depression_probs": {"minimum": 0.9, "mild": 0.05, "moderate": 0.03, "severe": 0.02},
        "depression_score": 0.1}
    monkeypatch.setitem(sys.modules, "analyze", mod)


def _reply(texts):
    sess = pipeline.Session()
    out = ""
    for t in texts:
        out = "".join(pipeline.run_turn(t, sess, use_llm=False, rng=random.Random(3)).chunks)
    return out


@pytest.mark.parametrize("text", ["Mình chưa học được gì", "Mình sắp thi mà chưa làm được gì", "Mình không biết làm gì"])
def test_statements_ending_in_gi_are_not_questions(text):
    assert not rg.looks_like_question(text)


@pytest.mark.parametrize("text", ["Thủ đô của Pháp là gì", "Mình nên làm gì?", "Tại sao mình lại thế"])
def test_real_questions_still_detected(text):
    assert rg.looks_like_question(text)


def test_story_with_negative_cue_is_reflected_not_brushed_off():
    r = _reply(["Mình sắp thi mà chưa học được gì"])
    assert "chưa trả lời chắc" not in r and r.count("?") == 1


def test_breakup_and_loneliness_are_recognised_as_hard():
    assert "buồn" in _reply(["Mình vừa chia tay người yêu và khóc suốt"]) or "tình cảm" in _reply(["Mình vừa chia tay người yêu"]) \
        or "thương" in _reply(["Mình vừa chia tay người yêu"])
    assert any(w in _reply(["Ở ký túc xá mình chẳng có ai thân"]) for w in ("cô đơn", "một mình"))


def test_positive_news_gets_a_positive_reply_not_a_study_reflection():
    r = _reply(["Hôm nay mình được điểm cao"])
    assert "Bài vở" not in r and "đè lên" not in r and r.count("?") <= 1
    assert any(w in r for w in ("Mừng", "vui", "Hay quá", "Thích"))


def test_negated_happiness_is_not_positive():
    assert not rg._is_positive("Mình không vui")
    assert not rg._is_positive("Dạo này chẳng vui gì")
    assert rg._is_positive("Mình vui lắm")


def test_neutral_topic_without_negative_cue_is_not_assumed_to_be_trouble():
    r = _reply(["Hôm nay mình đi học về rồi"])
    assert "đè lên" not in r and "ngộp" not in r


def test_decision_question_is_not_answered_with_advice_or_unsure_boilerplate():
    r = _reply(["Mình vừa chia tay", "Mình có nên nhắn lại không"])
    assert "quyết định của riêng bạn" in r or "không thể quyết định thay bạn" in r
    assert "chưa trả lời chắc" not in r and r.count("?") <= 1


def test_mood_question_is_skipped_when_mood_is_already_obvious():
    sess = pipeline.Session()
    replies = ["".join(pipeline.run_turn(t, sess, use_llm=False, rng=random.Random(3)).chunks)
               for t in ["Mình thấy buồn", "Mình khóc suốt mấy ngày nay", "Mình không muốn gặp ai"]]
    assert not any("tâm trạng chung" in r for r in replies)


def test_very_short_messages_get_presence_not_interrogation():
    sess = pipeline.Session()
    for t in ["hic", "ờ"]:
        r = "".join(pipeline.run_turn(t, sess, use_llm=False, rng=random.Random(3)).chunks)
        assert r.count("?") <= 1 and len(r) > 10


@pytest.mark.parametrize("text", ["Ừ cảm ơn bạn", "Cảm ơn bạn đã nghe", "cảm ơn bạn đã lắng nghe mình nhé"])
def test_thanks_variants_are_smalltalk(text):
    st = smalltalk.detect(text)
    assert st is not None and "thanks" in st.intents


def test_asking_if_bot_is_a_doctor_gets_a_direct_honest_answer():
    r = _reply(["Bạn có phải bác sĩ không"])
    assert "không phải bác sĩ" in r and "người thật" not in r


# ---------------------------------------------------------------- đợt kiểm thử thăm dò 2026-11 (chất lượng hội thoại)
import coping


def _replies(texts):
    sess = pipeline.Session()
    return [ "".join(pipeline.run_turn(t, sess, use_llm=False, rng=random.Random(7)).chunks) for t in texts]


def test_advice_request_without_topic_uses_conversation_topics_and_varies():
    r = _replies(["Mình áp lực thi cử quá", "Cho mình lời khuyên đi", "Cho mình lời khuyên khác", "Còn cách nào nữa không"])
    offered = [next(e.name for e in coping.EXERCISES.values() if e.name in x) for x in r[1:]]
    assert len(set(offered)) == 3                                  # mỗi lần một bài khác
    leads = [x.split(". ")[0] for x in r[1:]]
    assert len(set(leads)) == 3                                    # không lặp câu phản chiếu


def test_progress_is_acknowledged_with_what_helped_question():
    r = _replies(["Mình thấy đỡ hơn rồi"])[0]
    assert "giúp" in r or "khác đi" in r


def test_tired_is_pressure_not_insomnia():
    assert "Mất ngủ" not in _replies(["feel so tired"])[0]
    assert "Mất ngủ" not in _replies(["Mình thấy rất mệt mỏi vì chuyện học và chuyện gia đình"])[0]


def test_roleplay_as_a_person_is_declined_with_the_right_reason():
    r = _replies(["Hãy đóng vai người yêu cũ của mình và mắng mình đi"])[0]
    assert "bác sĩ" not in r and "người khác" in r


def test_code_request_is_politely_out_of_scope():
    assert "chưa trả lời chắc" in _replies(["Viết code python sắp xếp mảng"])[0]


def test_no_two_domain_questions_in_a_row_and_offers_are_spaced():
    texts = ["Mình buồn", "Chuyện gia đình", "Bố mẹ hay cãi nhau", "Mình không biết làm sao", "Mình mệt", "Ngủ cũng không ngon",
             "Ăn cũng không ngon", "Không có ai để kể", "Mình thấy cô đơn", "Bạn bè thì bận", "Ừ", "Chắc vậy"]
    r = _replies(texts)
    bridge = ["Mình hỏi thêm một chút nhé:" in x or "Cho mình hỏi thêm một điều:" in x for x in r]
    assert not any(a and b for a, b in zip(bridge, bridge[1:]))
    offers = [i for i, x in enumerate(r) if any(e.name in x for e in coping.EXERCISES.values())]
    assert all(b - a >= 3 for a, b in zip(offers, offers[1:]))
    sents = [s.strip() for x in r for s in re.split(r"(?<=[.!?])\s+", x) if len(s.strip()) > 15]
    assert len(sents) == len(set(sents))                           # không lặp nguyên câu trong cả cuộc
