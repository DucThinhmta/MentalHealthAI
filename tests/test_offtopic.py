"""Bot phải trả lời đúng câu người dùng hỏi (về bot hoặc ngoài lề), không lờ đi để hỏi sang chuyện khác."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import pipeline
import response_generator as rg
import smalltalk


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MHAI_TRANSLATOR", "none")           # không dùng mô hình dịch đã tải trên máy

    def missing(text):
        raise FileNotFoundError("no models")
    monkeypatch.setattr(pipeline, "_analyze", missing)


@pytest.mark.parametrize("text,intent", [
    ("bạn giới thiệu kỹ bản thân đi", "introduce"),
    ("giới thiệu về bạn đi", "introduce"),
    ("từ đã tên bạn là gì", "ask_name"),
    ("cho mình hỏi bạn tên gì", "ask_name"),
    ("ừ mà bạn là ai vậy", "ask_name"),
    ("thế bạn làm được gì", "capabilities"),
    ("bạn có phải người thật không", "who_made"),
    ("introduce yourself please", "introduce"),
])
def test_questions_about_the_bot_are_recognised(text, intent):
    st = smalltalk.detect(text) or smalltalk.detect_bot_question(text)
    assert st is not None and intent in st.intents


def test_the_conversation_from_the_screenshot():
    sess = pipeline.Session()
    replies = []
    for text in ["Xin chào", "bạn giới thiệu kỹ bản thân đi", "từ đã tên bạn là gì"]:
        replies.append("".join(pipeline.run_turn(text, sess, use_llm=False, rng=random.Random(0)).chunks))
    assert "đề tài nghiên cứu" in replies[1] and "lắng nghe" in replies[1]
    assert replies[2].startswith("Mình tên là MentalHealthAI")
    follow = "Hôm nay bạn cảm thấy thế nào?"
    assert sum(r.count(follow) for r in replies) == 1                      # câu hỏi thăm không lặp lại mỗi lượt
    assert sess.entries == []                                              # câu hỏi về bot không bị đem đi đánh giá


def test_story_with_a_bot_question_is_still_treated_as_a_story():
    sess = pipeline.Session()
    turn = pipeline.run_turn("mình buồn quá, bạn là ai mà hiểu mình", sess, use_llm=False, rng=random.Random(0))
    "".join(turn.chunks)
    assert turn.entry is not None and turn.entry["domains"].get("mood") == "concern"


@pytest.mark.parametrize("text", ["thủ đô của Pháp là gì?", "bạn có biết nấu ăn không", "why is the sky blue"])
def test_unanswerable_question_gets_an_honest_reply_not_a_new_topic(text):
    sess = pipeline.Session()
    reply = "".join(pipeline.run_turn(text, sess, use_llm=False, rng=random.Random(0)).chunks)
    assert any(u in reply for u in rg._UNSURE["vi"] + rg._UNSURE["en"])
    assert not any(q in reply for spec in rg.assessment.DOMAINS.values() for q in spec["ask"]["vi"])


def test_neutral_chat_does_not_trigger_interview_questions():
    sess = pipeline.Session()
    reply = "".join(pipeline.run_turn("hôm nay trời mưa", sess, use_llm=False, rng=random.Random(0)).chunks)
    assert not any(q in reply for spec in rg.assessment.DOMAINS.values() for q in spec["ask"]["vi"])


@pytest.mark.parametrize("text,expected", [
    ("thủ đô của Pháp là gì?", True), ("bạn có biết nấu ăn không", True), ("tai sao vay", True),
    ("hôm nay trời mưa", False), ("Mình mất ngủ", False), ("how do you work", True),
])
def test_looks_like_question(text, expected):
    assert rg.looks_like_question(text) is expected
