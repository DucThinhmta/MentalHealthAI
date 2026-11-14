"""Kiểm thử tầng hội thoại mới (pipeline, streaming, phản hồi theo luật) và hậu xử lý xác suất. Không cần mô hình."""
import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

import calibration as cal
import config
import llm_client
import pipeline
import response_generator as rg
from textutils import detect_lang


def _result(stress=True, depression="minimum", emotions=()):
    return {"stress": config.STRESS_YES if stress else config.STRESS_NO,
            "depression": depression, "emotions": list(emotions)}


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MHAI_TRANSLATOR", "none")           # không dùng mô hình dịch đã tải trên máy


# ---------- ngôn ngữ ----------
def test_detect_lang():
    assert detect_lang("Mình buồn quá") == "vi"
    assert detect_lang("minh buon qua khong ngu duoc") == "vi"      # gõ không dấu
    assert detect_lang("I am sad and I can't sleep") == "en"


# ---------- phản hồi theo luật ----------
def test_rule_replies_are_vietnamese_even_when_the_user_writes_english():
    vi = rg.generate_reply(_result(), "Mình áp lực vì kỳ thi quá", use_llm=False, rng=random.Random(0))["reply"]
    en_in = rg.generate_reply(_result(), "I am stressed about my exam", use_llm=False, rng=random.Random(0))["reply"]
    assert detect_lang(vi) == "vi" and vi.endswith("?")
    assert detect_lang(en_in) == "vi" and en_in.endswith("?")


def test_auto_mode_still_follows_the_users_language(monkeypatch):
    monkeypatch.setattr(config, "REPLY_LANGUAGE", "auto")
    en = rg.generate_reply(_result(), "I am stressed about my exam", use_llm=False, rng=random.Random(0))["reply"]
    vi = rg.generate_reply(_result(), "Mình áp lực vì kỳ thi quá", use_llm=False, rng=random.Random(0))["reply"]
    assert detect_lang(en) == "en" and detect_lang(vi) == "vi"


def test_english_input_is_analyzed_as_is_but_answered_in_vietnamese(fake_analyze):
    sess = pipeline.Session()
    turn = pipeline.run_turn("I am stressed about my exam", sess, use_llm=False)
    reply = "".join(turn.chunks)
    assert fake_analyze == ["I am stressed about my exam"] and detect_lang(reply) == "vi"      # phân tích bằng tiếng Anh
    assert sess.entries[0]["lang"] == "en" and sess.entries[0]["model_ok"]                     # đầu vào là tiếng Anh, dùng điểm mô hình


def test_rule_replies_do_not_repeat_within_a_conversation():
    history, seen = [], set()
    for _ in range(3):
        history.append({"role": "user", "content": "Mình mất ngủ"})
        reply = rg.generate_reply(_result(), "Mình mất ngủ", history, use_llm=False, rng=random.Random(7))["reply"]
        sentences = {s.strip() for s in reply.replace("?", "?|").replace(".", ".|").split("|") if s.strip()}
        assert not (sentences & seen), reply
        seen |= sentences
        history.append({"role": "assistant", "content": reply})


def test_topic_keyword_without_distress_is_not_treated_as_a_problem():
    out = rg.generate_reply(_result(False, emotions=[("joy", 0.8)]), "I love my school project", use_llm=False)
    assert "swamped" not in out["reply"] and "weighing" not in out["reply"]


# ---------- streaming ----------
def test_stream_reply_uses_llm_chunks(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "stream_reply", lambda *a, **kw: iter(["Ừ, ", "mình nghe đây."]))
    meta, chunks = rg.stream_reply(_result(), "I am stressed")
    assert "".join(chunks) == "Ừ, mình nghe đây." and meta["source"] == "llm"


def test_stream_reply_falls_back_when_llm_fails_before_any_text(monkeypatch):
    def boom(*a):
        raise RuntimeError("network down")
        yield                                    # noqa: biến hàm thành generator
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "stream_reply", boom)
    meta, chunks = rg.stream_reply(_result(), "I am stressed about my exam", rng=random.Random(0))
    assert "".join(chunks) and meta["source"] == "rules"


def test_stream_reply_crisis_never_calls_llm(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "stream_reply", lambda *a, **kw: pytest.fail("LLM không được gọi khi crisis"))
    meta, chunks = rg.stream_reply(_result(), "Tôi muốn chết")
    assert "115" in "".join(chunks) and meta == {"source": "safety", "safety_level": "crisis"}


def test_refusal_returns_empty_text_so_caller_falls_back():
    refused = types.SimpleNamespace(stop_reason="refusal", content=[])
    assert llm_client._text(refused) == ""


# ---------- pipeline ----------
@pytest.fixture
def fake_analyze(monkeypatch):
    calls = []

    def analyze(text):
        calls.append(text)
        stressed = "stress" in text.lower() or "tired" in text.lower()
        return {**_result(stressed), "stress_probability": 0.8 if stressed else 0.1,
                "depression_probs": {"minimum": 0.6, "mild": 0.2, "moderate": 0.1, "severe": 0.1},
                "n_messages_used": 1}
    monkeypatch.setattr(pipeline, "_analyze", analyze)
    return calls


def test_pipeline_smalltalk_skips_analysis(fake_analyze):
    sess = pipeline.Session()
    turn = pipeline.run_turn("Xin chào", sess, use_llm=False)
    assert "".join(turn.chunks).startswith("Xin chào!") and turn.meta["source"] == "smalltalk"
    assert turn.entry is None and sess.entries == [] and fake_analyze == []
    assert [m["role"] for m in sess.history] == ["user", "assistant"]


def test_pipeline_scores_each_message_and_profiles_whole_conversation(fake_analyze):
    sess = pipeline.Session()
    for text in ["I am stressed about my exam", "I had a nice lunch with friends today"]:
        "".join(pipeline.run_turn(text, sess, use_llm=False).chunks)
    assert fake_analyze == ["I am stressed about my exam", "I had a nice lunch with friends today"]
    p = sess.profile()
    assert p["n_scored"] == 2 and 0.1 < p["stress"]["overall"] < 0.8        # gộp cả hai câu, không chỉ câu cuối
    assert p["stress"]["series"] == [0.8, 0.1]


def test_pipeline_records_domains_and_the_question_bot_asked(fake_analyze):
    sess = pipeline.Session()
    turn = pipeline.run_turn("I can't sleep and I feel tired", sess, use_llm=False)
    reply = "".join(turn.chunks)
    assert turn.entry["domains"] == {"sleep": "concern", "energy": "concern"}
    assert turn.ask_domain not in ("sleep", "energy")                         # không hỏi lại điều đã biết
    assert sess.history[-1]["content"] == reply


def test_pipeline_quick_reply_sets_domain(fake_analyze):
    sess = pipeline.Session()
    turn = pipeline.run_turn("Ngủ ổn", sess, use_llm=False, answered=("sleep", "ok"))
    "".join(turn.chunks)
    assert sess.entries[0]["domains"]["sleep"] == "ok"


def test_pipeline_crisis_works_without_models(monkeypatch):
    def missing(text):
        raise FileNotFoundError("bert_stress")
    monkeypatch.setattr(pipeline, "_analyze", missing)
    sess = pipeline.Session()
    turn = pipeline.run_turn("I want to kill myself", sess, use_llm=False)
    assert "115" in "".join(turn.chunks) and turn.meta["safety_level"] == "crisis"
    turn = pipeline.run_turn("I feel tired lately", sess, use_llm=False)      # thiếu mô hình vẫn trò chuyện được
    assert "".join(turn.chunks) and sess.models_missing and not turn.entry["model_ok"]


def test_pipeline_uses_llm_extraction_for_translation_and_domains(fake_analyze, monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "extract", lambda text, last, recent=(): {"english": "I am very tired",
                                                                   "domains": {"energy": "concern"}})
    seen = {}

    def fake_stream(info, text, hist, ask, **kw):
        seen.update(info=info, ask=ask)
        return iter([f"echo:{text}"])
    monkeypatch.setattr(llm_client, "stream_reply", fake_stream)
    sess = pipeline.Session()
    turn = pipeline.run_turn("Mình uể oải quá", sess)
    assert "".join(turn.chunks) == "echo:Mình uể oải quá"                    # LLM trả lời theo câu gốc
    e = sess.entries[0]
    assert fake_analyze == ["I am very tired"] and e["translated"] and e["model_ok"]
    assert e["domains"]["energy"] == "concern" and seen["info"]["evidence"] == "low"
    assert seen["ask"] == turn.ask_domain


def test_pipeline_extraction_failure_falls_back_to_rules(fake_analyze, monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)

    def boom(*a):
        raise RuntimeError("down")
    monkeypatch.setattr(llm_client, "extract", boom)
    monkeypatch.setattr(llm_client, "stream_reply", boom)
    sess = pipeline.Session()
    turn = pipeline.run_turn("Mình mất ngủ", sess)
    assert "".join(turn.chunks) and turn.meta["source"] == "rules"
    assert sess.entries[0]["domains"] == {"sleep": "concern"} and not sess.entries[0]["model_ok"]


def test_summary_without_llm(fake_analyze):
    sess = pipeline.Session()
    profile, chunks = pipeline.summary(sess, use_llm=False)
    assert "chưa muốn vội nhận xét" in "".join(chunks)


# ---------- hiệu chỉnh ----------
def test_temperature_keeps_argmax_and_softens_confidence():
    p = np.array([[0.995, 0.005], [0.02, 0.98]])
    q = cal.apply_temperature(p, 3.0)
    assert (q.argmax(1) == p.argmax(1)).all() and np.allclose(q.sum(1), 1)
    assert q.max() < p.max() and np.allclose(cal.apply_temperature(p, 1.0), p)
    assert cal.apply_temperature(p, None) is not None


def test_fit_temperature_detects_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    correct = rng.random(2000) < 0.8                                 # mô hình đúng 80% nhưng luôn nói 99%
    pred = np.where(correct, y, 1 - y)
    probs = np.where(np.eye(2)[pred] == 1, 0.99, 0.01)
    t = cal.fit_temperature(probs, y)
    assert t > 2
    assert (cal.expected_calibration_error(cal.apply_temperature(probs, t), y)
            < cal.expected_calibration_error(probs, y) / 3)


def test_ordinal_decoding_recovers_middle_class():
    probs = np.array([[0.45, 0.35, 0.15, 0.05]])                     # argmax = minimum, nhưng điểm kỳ vọng = 0,8
    score = cal.expected_level(probs)
    assert abs(score[0] - 0.8) < 1e-9
    assert cal.decode_ordinal(score, [0.5, 1.5, 2.5])[0] == 1        # -> mild
    assert list(cal.decode_ordinal(np.array([0.0, 0.6, 1.6, 2.9]), [0.5, 1.5, 2.5])) == [0, 1, 2, 3]


def test_tune_ordinal_cuts_beats_default_rounding():
    rng = np.random.default_rng(1)
    y = rng.choice(4, 1500, p=[0.7, 0.1, 0.12, 0.08])
    score = np.clip(y * 0.5 + rng.normal(0, 0.25, len(y)), 0, 3)     # điểm bị nén về phía thấp như mô hình thật
    from sklearn.metrics import f1_score
    default = f1_score(y, cal.decode_ordinal(score, [0.5, 1.5, 2.5]), average="macro", zero_division=0)
    cuts, tuned = cal.tune_ordinal_cuts(y, score)
    assert cuts == sorted(cuts) and tuned > default + 0.1


def test_missing_calibration_file_means_no_change(tmp_path):
    assert cal.load(tmp_path / "nope.json") == {}
