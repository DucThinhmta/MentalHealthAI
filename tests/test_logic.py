"""Kiểm thử phần logic KHÔNG cần mô hình/GPU: chạy bằng `pytest tests -q`."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import config
import llm_client
import response_generator as rg
import safety
from textutils import looks_vietnamese, split_windows, strip_accents


# ---------- textutils ----------
def test_strip_accents():
    assert strip_accents("Tôi MUỐN chết, đau lắm") == "toi muon chet, dau lam"


def test_looks_vietnamese():
    assert looks_vietnamese("Tôi đang rất căng thẳng vì kỳ thi")
    assert not looks_vietnamese("I am stressed about my exams")


def test_split_windows_covers_all_words_with_overlap():
    text = " ".join(f"w{i}" for i in range(500))
    wins = split_windows(text, size=170, step=136)
    assert wins[0].split()[0] == "w0" and wins[-1].split()[-1] == "w499"
    assert all(len(w.split()) <= 170 for w in wins)
    covered = set(" ".join(wins).split())
    assert len(covered) == 500                       # không mất từ nào


def test_split_windows_short_and_invalid():
    assert split_windows("hello world", 170, 136) == ["hello world"]
    assert split_windows("", 170, 136) == [""]
    with pytest.raises(ValueError):
        split_windows("a b c", 10, 20)


# ---------- safety ----------
@pytest.mark.parametrize("text", [
    "I want to kill myself", "i don't want to live anymore", "Sometimes I think about suicide",
    "I wish I was dead", "I've been hurting myself again", "there is nothing to live for",
    "Tôi muốn chết", "toi muon chet", "Mình không muốn sống nữa", "Em đang nghĩ đến tự tử",
    "tôi có ý định tự sát", "em muốn rạch tay",
])
def test_detects_crisis(text):
    assert safety.detect_crisis(text)


@pytest.mark.parametrize("text", [
    "This exam is killing me", "I'm killing it at work today", "Tôi đang đi từ từ về nhà",
    "I am stressed about my exams", "Hôm nay tôi rất mệt vì thi", "hello",
])
def test_no_false_crisis(text):
    assert not safety.detect_crisis(text)


def test_assess_levels():
    ok = {"depression": "minimum", "stress": config.STRESS_NO}
    severe = {"depression": "severe", "stress": config.STRESS_YES}
    assert safety.assess_safety("I want to kill myself", ok) == "crisis"
    assert safety.assess_safety("I feel awful", severe) == "elevated"
    assert safety.assess_safety("I feel fine", ok) == "none"


def test_crisis_message_has_resources_and_language():
    vi = safety.crisis_message("Tôi muốn chết")
    assert "115" in vi and "113" in vi and "findahelpline" in vi
    assert "emergency number" not in vi
    assert "emergency number" in safety.crisis_message("I want to die")


# ---------- response_generator ----------
def test_context_uses_word_boundaries():
    ctx = rg.detect_context("I downloaded the latest network update")
    assert not ctx["work"] and not ctx["study"] and not ctx["sad"]
    assert rg.detect_context("My exam is tomorrow")["study"]
    assert rg.detect_context("Mình mất ngủ vì áp lực thi")["sleep"]


def _result(stress=True, depression="minimum", emotions=()):
    return {"stress": config.STRESS_YES if stress else config.STRESS_NO,
            "depression": depression, "emotions": list(emotions)}


def test_crisis_overrides_everything_even_with_llm(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "generate_reply", lambda *a: pytest.fail("LLM không được gọi khi crisis"))
    out = rg.generate_reply(_result(), "I want to kill myself", use_llm=True)
    assert out["source"] == "safety" and out["safety_level"] == "crisis"


def test_llm_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    def boom(*a): raise RuntimeError("network down")
    monkeypatch.setattr(llm_client, "generate_reply", boom)
    out = rg.generate_reply(_result(), "I am stressed about my exam", use_llm=True, rng=random.Random(0))
    assert out["source"] == "rules" and out["reply"]


def test_llm_used_when_available(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "generate_reply", lambda *a: "Xin chào từ LLM")
    out = rg.generate_reply(_result(), "I am stressed", use_llm=True)
    assert out["source"] == "llm" and out["reply"] == "Xin chào từ LLM"


def test_elevated_adds_note_and_topic_follows_latest_message():
    out = rg.generate_reply(_result(depression="severe"), "I can't sleep at night", use_llm=False,
                            rng=random.Random(1))
    assert out["safety_level"] == "elevated" and "chuyên gia" in out["reply"]
    assert "giấc ngủ" in out["reply"] or "ngủ" in out["reply"]


def test_positive_and_neutral_paths():
    pos = rg.generate_reply(_result(False, emotions=[("joy", 0.9)]), "Great day", use_llm=False)
    assert "tích cực" in pos["reply"]
    neu = rg.generate_reply(_result(False), "ok", use_llm=False)
    assert neu["source"] == "rules" and neu["reply"]


def test_legacy_generate_response_returns_string():
    assert isinstance(rg.generate_response(_result(), "I am worried"), str)


# ---------- llm_client ----------
def test_llm_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not llm_client.is_available()


def test_history_normalization():
    hist = [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "a"},
            {"role": "user", "content": "b"}]
    msgs = llm_client._normalize_history(hist, "b")
    assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "user"
    assert all(msgs[i]["role"] != msgs[i + 1]["role"] for i in range(len(msgs) - 1))
    assert llm_client._normalize_history([], "x") == [{"role": "user", "content": "x"}]
