import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import smalltalk


@pytest.mark.parametrize("text,intents", [
    ("hello", ("greeting",)),
    ("what your name", ("ask_name",)),
    ("hello, what your name", ("greeting", "ask_name")),
    ("Xin chào, bạn tên gì?", ("greeting", "ask_name")),
    ("ok", ("ack",)), ("lol", ("ack",)), ("I'm good", ("good_status",)), ("I'm fine", ("good_status",)),
    ("tell me a joke", ("joke",)), ("are you a bot?", ("who_made",)), ("I'm bored", ("bored",)),
    ("Bạn bao nhiêu tuổi?", ("how_old",)), ("Kể chuyện cười đi", ("joke",)), ("bạn giỏi quá", ("compliment",)),
    ("sorry", ("sorry",)), ("What are you doing", ("what_doing",)),
])
def test_detects(text, intents):
    st = smalltalk.detect(text)
    assert st is not None and st.intents == intents


@pytest.mark.parametrize("text", ["hello I feel so sad", "I hate my name", "my name is bob", "I feel hopeless", "I'm so tired", "no one cares about me"])
def test_not_smalltalk(text):
    assert smalltalk.detect(text) is None


def test_reply_nonempty_and_language():
    assert "MentalHealthAI" in smalltalk.reply(smalltalk.detect("what is your name"))
    assert smalltalk.detect("Xin chào").lang == "vi"


def test_every_intent_has_replies_in_both_languages():
    intents = {i for i, _, _ in smalltalk._RULES}
    for lang in ("en", "vi"):
        for i in intents:
            assert i in smalltalk._REPLIES[lang], (lang, i)
