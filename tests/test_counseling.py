"""Cấu trúc phiên tư vấn: giai đoạn, cách đồng hành, việc nhỏ, ghi nhớ, viết lại/sửa tin, kết thúc phiên (không cần mô hình)."""
import random
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import pytest

import config
import counseling
import coping
import llm_client
import memory
import pipeline


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MHAI_TRANSLATOR", "none")
    mod = types.ModuleType("analyze")                 # mô hình giả: không cần torch
    mod.analyze_conversation = lambda msgs, window=5: {
        "stress_probability": 0.5, "emotions": [], "depression_probs": {"minimum": 0.7, "mild": 0.1, "moderate": 0.1, "severe": 0.1},
        "depression_score": 0.4}
    monkeypatch.setitem(sys.modules, "analyze", mod)


STORY = ["Dạo này mình áp lực việc học lắm, ngủ cũng không ngon", "Mình lo thi cuối kỳ, đầu óc cứ rối",
         "Hơn một tuần rồi, mình hay thức khuya"]


def _talk(sess, texts, seed=2, **kw):
    out = []
    for t in texts:
        turn = pipeline.run_turn(t, sess, use_llm=False, rng=random.Random(seed), **kw)
        out.append(("".join(turn.chunks), turn))
    return out


# ---------------------------------------------------------------- giai đoạn
def test_stage_progression_and_never_locks():
    assert counseling.stage_for(1, "low") == "opening"
    assert counseling.stage_for(2, "low") == "exploring"
    assert counseling.stage_for(5, "low") == "planning"
    assert counseling.stage_for(2, "medium") == "planning"
    assert counseling.stage_for(3, "low", style="solve") == "planning"
    assert counseling.stage_for(9, "high", style="listen") == "exploring"       # chỉ lắng nghe: không tự chuyển giai đoạn
    assert counseling.stage_for(3, "low", plan_count=1) == "acting"
    assert counseling.stage_for(3, "high", closed=True) == "closing"
    steps = counseling.stage_progress("planning")
    assert [s[3] for s in steps] == ["done", "done", "current", "todo", "todo"]


def test_suggestion_chips_follow_style():
    assert all(counseling.suggestions(s) for s in counseling.STAGE_IDS)
    assert "Cho mình một gợi ý nhỏ" in [c[0] for c in counseling.suggestions("exploring", "gentle")]
    assert "Cho mình một gợi ý nhỏ" not in [c[0] for c in counseling.suggestions("exploring", "listen")]
    assert "Gợi ý khác" not in [c[0] for c in counseling.suggestions("planning", "listen")]


def test_guide_text_mentions_stage_style_plan_memory():
    g = counseling.guide_text("acting", "solve", "người dùng đã nhắc tới học tập", ["Thở hộp"])
    assert "VIỆC NHỎ" in g and "giải quyết" in g and "Thở hộp" in g and "học tập" in g


# ---------------------------------------------------------------- việc nhỏ
def test_plan_add_toggle_remove_no_duplicates():
    plan = []
    ex = next(iter(coping.EXERCISES))
    assert counseling.add_plan(plan, ex) and not counseling.add_plan(plan, ex)
    assert not counseling.add_plan(plan, "khong-co-bai-nay")
    assert counseling.toggle_plan(plan, ex) is True and counseling.toggle_plan(plan, ex) is False
    assert counseling.remove_plan(plan, ex) and plan == []


def test_closing_extras_lists_plan_and_mood_trend():
    plan = []
    counseling.add_plan(plan, next(iter(coping.EXERCISES)))
    counseling.toggle_plan(plan, plan[0]["id"], True)
    text = counseling.closing_extras(plan, [2, 4])
    assert "✅" in text and "2/5" in text and "4/5" in text and "nhỉnh hơn" in text and "quay lại" in text
    assert counseling.mood_delta([3]) is None


# ---------------------------------------------------------------- luồng tư vấn trong pipeline
def test_planning_offers_one_exercise_and_accept_adds_to_plan():
    sess = pipeline.Session()
    sess.style = "solve"
    replies = _talk(sess, STORY)
    assert replies[0][1].stage == "opening" and replies[2][1].stage == "planning"
    assert coping.EXERCISES and any(e.name in replies[2][0] for e in coping.EXERCISES.values())     # có gợi ý một việc nhỏ
    assert replies[2][0].count("?") == 1 and replies[2][0].rstrip().endswith("?")
    accept = _talk(sess, ["Mình muốn thử"])[0]
    assert [p["name"] for p in sess.plan] and sess.plan[0]["name"] in accept[0]
    assert accept[1].stage == "acting" and sess.stage() == "acting"
    tried = _talk(sess, ["Mình đã thử rồi"])[0][0]
    assert "thử" in tried and "?" in tried
    assert len(sess.plan) == 1                                                  # nói "đã thử" không thêm trùng


def test_accept_without_offer_adds_nothing_and_asks_what_to_try():
    sess = pipeline.Session()
    reply = _talk(sess, ["Dạo này mình buồn", "Mình muốn thử"])[1][0]
    assert sess.plan == [] and "thử" in reply


def test_listen_style_never_offers_exercises_unprompted():
    sess = pipeline.Session()
    sess.style = "listen"
    replies = _talk(sess, STORY + ["Mình vẫn còn lo lắm", "Cứ nghĩ tới là mình thấy nặng nề"])
    assert not any(e.name in r for r, _ in replies for e in coping.EXERCISES.values())
    assert sess.stage() == "exploring"


def test_acting_stage_checks_in_on_the_chosen_plan_but_not_every_turn():
    sess = pipeline.Session()
    sess.style = "solve"
    _talk(sess, STORY + ["Mình muốn thử"])
    name = sess.plan[0]["name"]
    after = [r for r, _ in _talk(sess, ["Vẫn còn lo lắm", "Mình cũng không biết nữa", "Hôm nay mình hơi mệt"])]
    asked = [name in r for r in after]
    assert asked[0] and not all(asked)                                           # hỏi thăm nhưng không lặp mỗi lượt


def test_every_reply_is_vietnamese_and_has_at_most_one_question():
    import evaluate_dialogs as ev
    sess = pipeline.Session()
    sess.style = "solve"
    sess.memory_greeting = "Mình nhớ lần trước bạn có nhắc tới học tập."
    for r, _ in _talk(sess, STORY + ["Mình muốn thử", "Mình đã thử rồi", "Vẫn lo lắm"]):
        assert r.count("?") <= 1 and not ev.english_leaks(r)


# ---------------------------------------------------------------- viết lại, sửa tin, kết thúc phiên
def test_regenerate_replaces_last_reply_without_duplicating_evidence():
    sess = pipeline.Session()
    first = _talk(sess, STORY[:2])[1][0]
    n_hist, n_entries, n_asked = len(sess.history), len(sess.entries), len(sess.asked)
    turn = pipeline.regenerate(sess, use_llm=False, rng=random.Random(9))
    second = "".join(turn.chunks)
    assert len(sess.history) == n_hist and len(sess.entries) == n_entries            # không nhân đôi bằng chứng
    assert sess.history[-1]["content"] == second and sess.history[-2]["content"] == STORY[1]
    assert second != first                                                          # tránh lặp lại câu trước
    assert len(sess.asked) in (n_asked - 1, n_asked, n_asked + 1)


def test_undo_last_turn_restores_state_and_returns_text():
    sess = pipeline.Session()
    _talk(sess, STORY[:2], mood=3)
    snapshot = (len(sess.history), len(sess.entries), list(sess.self_moods))
    assert pipeline.undo_last_turn(sess) == STORY[1]
    assert (len(sess.history), len(sess.entries)) == (snapshot[0] - 2, snapshot[1] - 1)
    assert sess.self_moods == snapshot[2][:-1]                                       # tâm trạng đi kèm tin đó cũng được gỡ
    assert pipeline.undo_last_turn(pipeline.Session()) == "" and pipeline.regenerate(pipeline.Session()) is None


def test_undo_accept_turn_removes_the_plan_item():
    sess = pipeline.Session()
    sess.style = "solve"
    _talk(sess, STORY + ["Mình muốn thử"])
    assert sess.plan
    pipeline.undo_last_turn(sess)
    assert sess.plan == []


def test_close_session_keeps_chat_open_and_reopens_on_next_message():
    sess = pipeline.Session()
    sess.style = "solve"
    _talk(sess, STORY + ["Mình muốn thử"], mood=2)
    profile, text = pipeline.close_session(sess, mood=4, use_llm=False)
    assert sess.closed and sess.stage() == "closing" and sess.history[-1]["content"] == text
    assert sess.plan[0]["name"] in text and "2/5" in text and "4/5" in text
    assert not sess.can_regenerate()                                                 # lời kết thúc không viết lại được
    reply = _talk(sess, ["Mình muốn nói thêm một chút"])[0]
    assert not sess.closed and reply[0]                                              # nói tiếp thì phiên mở lại


def test_crisis_still_overrides_everything_in_any_stage():
    sess = pipeline.Session()
    sess.style = "listen"
    sess.memory_greeting = "Mình nhớ lần trước bạn có nhắc tới học tập."
    reply = _talk(sess, ["Mình muốn chết"])[0]
    assert "115" in reply[0] and reply[1].meta["safety_level"] == "crisis" and "Mình nhớ lần trước" not in reply[0]


# ---------------------------------------------------------------- ghi nhớ giữa các cuộc trò chuyện
def _profile_after(texts):
    sess = pipeline.Session()
    _talk(sess, texts)
    return sess


def test_memory_stores_labels_only_and_respects_toggle_and_clear():
    sess = _profile_after(STORY)
    mem = memory.Memory()
    memory.remember(mem, sess.profile(), [])
    assert mem.chats == 1 and mem.topics
    blob = memory.text(mem) + memory.greeting(mem)
    assert not any(t in blob for t in STORY)                                         # không có nguyên văn tin nhắn
    assert memory.greeting(mem).startswith("Mình nhớ") and "?" not in memory.greeting(mem)
    mem.enabled = False
    assert memory.text(mem) == "" and memory.greeting(mem) == ""
    off = memory.Memory(enabled=False)
    memory.remember(off, sess.profile(), [])
    assert off.is_empty() and off.chats == 0
    memory.clear(mem)
    assert mem.is_empty()


def test_memory_skips_chats_that_are_too_short():
    mem = memory.Memory()
    memory.remember(mem, _profile_after(STORY[:1]).profile(), [])
    assert mem.is_empty()


def test_greeting_appears_once_at_start_of_new_chat_and_not_on_smalltalk():
    mem = memory.Memory()
    memory.remember(mem, _profile_after(STORY).profile(), [])
    sess = pipeline.Session()
    sess.memory_text, sess.memory_greeting = memory.text(mem), memory.greeting(mem)
    replies = [r for r, _ in _talk(sess, ["Dạo này mình áp lực việc học lắm", "Mình lo thi cuối kỳ"])]
    assert replies[0].startswith("Mình nhớ lần trước") and "Mình nhớ lần trước" not in replies[1]
    small = pipeline.Session()
    small.memory_greeting = memory.greeting(mem)
    assert "Mình nhớ lần trước" not in _talk(small, ["cảm ơn bạn"])[0][0]


# ---------------------------------------------------------------- LLM nhận định hướng giai đoạn
def test_llm_system_prompt_carries_stage_style_plan_and_avoid_blocks():
    coach = {"stage": "planning", "style": "solve", "plan": ["Thở hộp"], "memory_text": "đã nhắc tới học tập"}
    blocks = llm_client._system({"n_turns": 3}, None, coach, "Câu trả lời cũ của bot")
    text = "\n".join(b["text"] for b in blocks)
    assert "ĐẶT HƯỚNG ĐI" in text and "Thở hộp" in text and "học tập" in text and "Câu trả lời cũ của bot" in text
    base = "\n".join(b["text"] for b in llm_client._system({"n_turns": 3}, None))
    assert "ĐẶT HƯỚNG ĐI" not in base and "Câu trả lời cũ" not in base


def test_pipeline_passes_coach_and_avoid_to_llm(monkeypatch):
    seen = {}

    def fake_stream(info, text, hist, ask, **kw):
        seen.update(kw)
        return iter(["Mình nghe bạn."])
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "extract", lambda *a, **k: {"english": "I am tired", "domains": {}, "risk": None})
    monkeypatch.setattr(llm_client, "stream_reply", fake_stream)
    sess = pipeline.Session()
    sess.style = "solve"
    "".join(pipeline.run_turn("Mình mệt quá", sess, use_llm=True).chunks)
    assert seen["coach"]["style"] == "solve" and seen["coach"]["stage"] == "opening" and seen["avoid"] == ""
    "".join(pipeline.regenerate(sess, use_llm=True).chunks)
    assert seen["avoid"] == "Mình nghe bạn."
