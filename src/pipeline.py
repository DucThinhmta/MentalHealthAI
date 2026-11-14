"""Một lượt hội thoại, dùng chung cho giao diện web (app.py) và terminal (chatbot.py).

Thứ tự: an toàn -> xã giao -> phân tích.
- Tin nhắn có biểu hiện tự hại: trả thông điệp cố định ngay, KHÔNG phụ thuộc vào mô hình.
- Câu xã giao (chào hỏi, cảm ơn...): trả lời trực tiếp, không đưa vào đánh giá.
- Còn lại: mỗi tin nhắn được chấm RIÊNG (stress, mức nặng nề, cảm xúc) + nhận diện chủ đề và khía cạnh sinh hoạt
  (giấc ngủ, năng lượng...); sau đó assessment.build_profile gộp TẤT CẢ các lượt thành đánh giá của cả cuộc
  trò chuyện. Bot dùng đánh giá đó để chọn giọng điệu và chọn khía cạnh nên hỏi tiếp.

Trạng thái hội thoại (`Session`) được giữ bởi bên gọi (st.session_state hoặc biến trong chatbot.py).
"""
from dataclasses import dataclass, field

import assessment
import counseling
import intents
import llm_client
import response_generator as rg
import safety
import smalltalk
import translator
from textutils import detect_lang, looks_vietnamese


@dataclass
class Session:
    history: list = field(default_factory=list)       # [{'role', 'content'}] để LLM nhớ ngữ cảnh
    entries: list = field(default_factory=list)       # bằng chứng từng lượt người dùng (xem assessment.build_profile)
    asked: list = field(default_factory=list)         # các khía cạnh bot đã hỏi
    self_moods: list = field(default_factory=list)    # tâm trạng tự chấm 1..5
    models_missing: bool = False                       # chưa tải 3 mô hình BERT (download_models.py)
    minor: bool = False                                # người dùng dưới 18 tuổi (thêm kênh 111, không lưu nghiên cứu)
    research: bool = False                             # người dùng đồng ý lưu bản ghi ẩn danh cho nghiên cứu
    style: str = counseling.DEFAULT_STYLE              # cách đồng hành người dùng chọn (counseling.STYLES)
    plan: list = field(default_factory=list)           # "Việc nhỏ" đã chọn: [{'id','name','done','added'}]
    closed: bool = False                               # người dùng đã bấm kết thúc phiên (chat vẫn mở, nói tiếp là mở lại)
    turns: list = field(default_factory=list)          # nhật ký từng lượt để hoàn tác/viết lại
    memory_text: str = ""                              # điều đã nhớ từ các cuộc trước (do người dùng cho phép)
    memory_greeting: str = ""                          # lời nhắc nhẹ ở lượt đầu
    feedback: dict = field(default_factory=dict)       # chỉ số tin nhắn bot -> 1 (hữu ích) hoặc 0 (chưa ổn), chỉ trong phiên

    def profile(self):
        return assessment.build_profile(self.entries, self.self_moods)

    def n_user_turns(self):
        return sum(1 for m in self.history if m["role"] == "user")

    def can_regenerate(self):
        """Còn lượt gần nhất (người dùng + bot) để viết lại/sửa; không áp dụng cho lời kết thúc phiên."""
        return bool(self.turns) and not self.turns[-1].get("closing") and len(self.history) >= 2 \
            and self.history[-1]["role"] == "assistant"

    def stage(self):
        ev = self.profile()["evidence"]
        return counseling.stage_for(self.n_user_turns(), ev, len(self.plan), self.closed, self.style)


@dataclass
class Turn:
    chunks: object                       # iterator các đoạn chữ của phản hồi
    meta: dict                           # {'source', 'safety_level'} (source đúng sau khi đọc hết chunks)
    entry: dict | None = None            # bằng chứng của lượt này (None nếu không đánh giá)
    profile: dict | None = None          # đánh giá cả cuộc trò chuyện SAU lượt này
    ask_domain: str | None = None        # khía cạnh gợi ý hỏi trong lượt này
    stage: str | None = None             # giai đoạn của phiên SAU lượt này (counseling.STAGES)


def _analyze(text):
    import sys

    import config
    if "analyze" not in sys.modules:     # thiếu mô hình thì báo ngay, khỏi nạp torch (chậm) cho vô ích
        for name in ("stress", "emotion", "depression"):
            config.find_model(name)
    import analyze                       # nạp mô hình lần đầu có thể mất vài giây
    return analyze.analyze_conversation([text], window=1)


def _models_present():
    """Đã có 3 mô hình BERT chưa (chưa có thì dịch cũng không để làm gì)."""
    import config
    try:
        for name in ("stress", "emotion", "depression"):
            config.find_model(name)
        return True
    except FileNotFoundError:
        return "analyze" in __import__("sys").modules        # môi trường kiểm thử thay analyze bằng bản giả


def _tells_a_story(text):
    """Câu có kể điều đáng lo/chủ đề cá nhân không (khi đó KHÔNG coi là câu hỏi xã giao về bot)."""
    concerns = [v for v in assessment.detect_domains(text).values() if v == "concern"]
    return bool(concerns) or any(rg.detect_context(text).values())


def _understand(user_text, last_bot, llm_ok, use_llm, recent_user=()):
    """(văn bản tiếng Anh cho mô hình, đã dịch?, khía cạnh do LLM trích, mức nguy cơ do LLM đánh giá).

    Lỗi LLM -> dùng luật; khi đó mức nguy cơ là None (chỉ còn lớp luật từ khóa)."""
    if llm_ok and use_llm:
        try:
            data = llm_client.extract(user_text, last_bot, recent_user)
            return data["english"] or user_text, looks_vietnamese(user_text), data["domains"], data.get("risk")
        except Exception:
            pass
    if detect_lang(user_text) == "vi" and _models_present():   # không có LLM: dịch trên máy (nếu đã tải mô hình dịch)
        try:
            english = translator.runtime_vi2en(user_text)
            if english:
                return english, True, {}, None
        except Exception:
            pass
    return user_text, False, {}, None


def _coach(session, profile, first_turn, new_plan=False, with_greeting=True):
    """Thông tin định hướng cho bot: giai đoạn, cách đồng hành, việc nhỏ đã chọn, điều đã nhớ."""
    return {"stage": counseling.stage_for(session.n_user_turns(), profile["evidence"], len(session.plan), False, session.style),
            "style": session.style, "plan": [p["name"] for p in session.plan], "new_plan": new_plan,
            "memory_text": session.memory_text,
            "greeting": session.memory_greeting if first_turn and with_greeting else ""}


def run_turn(user_text, session, *, use_llm=True, rng=None, answered=None, mood=None, avoid=""):
    """Xử lý một tin nhắn và CẬP NHẬT `session` (history, entries, asked, plan...).

    answered=(domain, 'ok'|'concern'): người dùng bấm nút trả lời nhanh cho câu hỏi về khía cạnh đó.
    mood: tâm trạng tự chấm 1..5 đi kèm tin nhắn này (nếu có).
    avoid: câu trả lời trước cần tránh lặp (dùng khi viết lại).
    Nhắn tiếp sau khi đã kết thúc phiên thì phiên mở lại. Chưa có mô hình thì vẫn chạy (models_missing=True).
    """
    kwargs = {"use_llm": use_llm, "minor": session.minor, "avoid": avoid}
    if rng is not None:
        kwargs["rng"] = rng
    llm_ok = llm_client.is_available()
    crisis = safety.detect_crisis(user_text)
    first_turn = session.n_user_turns() == 0
    rec = {"entry": None, "asked": False, "mood": mood, "plan": None, "answered": answered, "text": user_text}
    if mood is not None:
        session.self_moods.append(mood)
    session.closed = False
    session.history.append({"role": "user", "content": user_text})
    session.turns.append(rec)

    small = None if crisis or answered else smalltalk.detect(user_text)
    if small is None and not crisis and not answered and not _tells_a_story(user_text):
        small = smalltalk.detect_bot_question(user_text)     # "từ từ đã, tên bạn là gì?" -> trả lời câu hỏi đó
    if small is not None:
        coach = _coach(session, session.profile(), first_turn, with_greeting=False)
        meta, chunks = rg.stream_reply(None, user_text, session.history, small=small, coach=coach, **kwargs)
        return Turn(_remember(chunks, session, rec), meta, profile=session.profile(), stage=coach["stage"])

    last_bot = next((m["content"] for m in reversed(session.history[:-1]) if m["role"] == "assistant"), "")
    recent_user = [m["content"] for m in session.history[:-1] if m["role"] == "user"][-3:]
    english, translated, llm_domains, risk = _understand(user_text, last_bot, llm_ok, use_llm, recent_user)
    result = None
    try:
        result = _analyze(english)
    except FileNotFoundError:
        session.models_missing = True    # chưa tải mô hình: vẫn trò chuyện và đánh giá qua các khía cạnh sinh hoạt
    except Exception:
        if not crisis:
            raise                        # khủng hoảng: thông điệp an toàn không được phụ thuộc vào mô hình

    domains = assessment.detect_domains(user_text)
    domains.update({d: s for d, s in llm_domains.items() if s in ("ok", "concern")})
    if answered:
        domains[answered[0]] = answered[1]
    entry = {"text": user_text, "english": english, "translated": translated,
             "lang": detect_lang(user_text), "words": len(user_text.split()),
             # mô hình chỉ hiểu tiếng Anh: tin nhắn tiếng Việt chưa dịch thì KHÔNG dùng điểm mô hình
             "model_ok": result is not None and (translated or detect_lang(user_text) == "en"),
             "topics": [t for t, on in rg.detect_context(user_text).items() if on], "domains": domains,
             "risk_llm": risk}
    if result is not None:
        entry.update({"stress_probability": result["stress_probability"],
                      "depression_probs": result["depression_probs"],
                      "depression_score": result.get("depression_score"), "emotions": result["emotions"]})
    session.entries.append(entry)
    rec["entry"] = entry
    intent = None if crisis else intents.detect(user_text)
    new_plan = False
    if intent == "accept":                           # "mình muốn thử": ghi bài tập bot vừa gợi ý vào Việc nhỏ
        ex = counseling.offered_exercise(session.history[:-1])
        if ex and counseling.add_plan(session.plan, ex):
            new_plan = True
            rec["plan"] = ex
    profile = session.profile()
    ask = assessment.next_domain(profile, session.asked)
    coach = _coach(session, profile, first_turn, new_plan)
    meta, chunks = rg.stream_reply(profile, user_text, session.history, latest=result, ask_domain=ask,
                                   risk_hint=risk, intent=intent, coach=coach, **kwargs)
    return Turn(_remember(chunks, session, rec), meta, entry, profile, ask, coach["stage"])


def _pop_last(session):
    """Gỡ lượt gần nhất (tin người dùng + phản hồi) và mọi thứ nó đã ghi; -> (rec, phản hồi trước)."""
    rec = session.turns.pop()
    prev = session.history.pop()["content"] if session.history and session.history[-1]["role"] == "assistant" else ""
    if session.history and session.history[-1]["role"] == "user":
        session.history.pop()
    if rec.get("entry") is not None and session.entries and session.entries[-1] is rec["entry"]:
        session.entries.pop()
    if rec.get("asked") and session.asked:
        session.asked.pop()
    if rec.get("mood") is not None and session.self_moods:
        session.self_moods.pop()
    if rec.get("plan"):
        counseling.remove_plan(session.plan, rec["plan"])
    session.feedback.pop(len(session.history), None)
    return rec, prev


def take_back(session):
    """Gỡ lượt gần nhất để làm lại. -> {'text','answered','mood','avoid'} (avoid = phản hồi vừa gỡ) hoặc None nếu không gỡ được."""
    if not session.can_regenerate():
        return None
    rec, prev = _pop_last(session)
    return {"text": rec["text"], "answered": rec.get("answered"), "mood": rec.get("mood"), "avoid": prev}


def undo_last_turn(session):
    """Gỡ lượt gần nhất để người dùng sửa tin nhắn cuối; trả lại nội dung tin nhắn đó ('' nếu không gỡ được)."""
    back = take_back(session)
    return back["text"] if back else ""


def regenerate(session, *, use_llm=True, rng=None):
    """Viết lại phản hồi cuối: gỡ lượt gần nhất rồi chạy lại cùng tin nhắn, dặn bot tránh lặp câu trước. -> Turn hoặc None."""
    back = take_back(session)
    if back is None:
        return None
    return run_turn(back["text"], session, use_llm=use_llm, rng=rng, answered=back["answered"], mood=back["mood"],
                    avoid=back["avoid"])


def close_session(session, mood=None, use_llm=True):
    """Người dùng chủ động kết thúc phiên: chấm tâm trạng cuối (tuỳ chọn), tạo lời tổng kết + việc nhỏ + lời mời quay lại.

    Chat KHÔNG bị khoá: nhắn tiếp sẽ mở lại phiên. -> (profile, lời kết thúc)."""
    rec = {"closing": True, "mood": None, "entry": None, "asked": False, "plan": None, "text": ""}
    if mood is not None:
        session.self_moods.append(mood)
    profile, chunks = summary(session, use_llm)
    text = "".join(chunks).strip() + "\n\n" + counseling.closing_extras(session.plan, session.self_moods)
    session.closed = True
    session.history.append({"role": "assistant", "content": text})
    session.turns.append(rec)
    return session.profile(), text


def _remember(chunks, session, rec=None):
    """Chuyển tiếp các đoạn chữ; khi xong thì lưu phản hồi vào history và ghi nhận khía cạnh bot đã hỏi."""
    parts = []
    for c in chunks:
        parts.append(c)
        yield c
    reply = "".join(parts)
    session.history.append({"role": "assistant", "content": reply})
    d = assessment.asked_domain(reply)
    if d:
        session.asked.append(d)                       # có thể lặp: dùng để đếm số lần đã hỏi
        if rec is not None:
            rec["asked"] = True


def summary(session, use_llm=True):
    """(profile, iterator đoạn chữ của lời tổng kết). LLM viết lời tổng kết ấm áp; không có thì dùng mẫu."""
    profile = session.profile()

    def chunks():
        if use_llm and llm_client.is_available() and profile["evidence"] != "low":
            try:
                got = False
                for c in llm_client.stream_summary(profile, session.history):
                    got = True
                    yield c
                if got:
                    return
            except Exception:
                pass
        yield rg.summary_fallback(profile)
    return profile, chunks()
