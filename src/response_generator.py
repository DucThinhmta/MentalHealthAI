"""Tầng phản hồi: an toàn -> (LLM tùy chọn) -> luật/ngữ cảnh làm phương án dự phòng.

Phản hồi theo luật được viết theo kiểu trò chuyện: phản chiếu điều người dùng vừa nói, rồi hỏi tiếp một câu
cụ thể; trả lời cùng ngôn ngữ với người dùng và không lặp lại câu đã dùng trong cùng cuộc hội thoại.
"""
import random
import re

import assessment
import config
import coping
import intents
import llm_client
import safety
import smalltalk
from textutils import detect_lang, reply_language, strip_accents, word_pattern

# Từ khóa theo chủ đề (tiếng Anh + tiếng Việt không dấu). Khớp nguyên từ, không khớp chuỗi con.
_TOPIC_WORDS = {
    "study": ["exam", "exams", "study", "studying", "school", "university", "college", "assignment",
              "homework", "grade", "grades", "test", "thesis", "project", "thi", "hoc", "bai tap",
              "diem", "luan van", "do an", "truong"],
    "sleep": ["sleep", "sleeping", "insomnia", "can't sleep", "cannot sleep", "wake up", "tired",
              "mat ngu", "khong ngu duoc", "kho ngu", "thuc khuya", "met moi"],
    "work": ["work", "job", "boss", "office", "deadline", "meeting", "workload", "cong viec",
             "sep", "viec lam", "deadline"],
    "relationship": ["relationship", "boyfriend", "girlfriend", "partner", "friend", "friends", "family",
                     "parents", "breakup", "ban be", "gia dinh", "bo me", "nguoi yeu", "chia tay"],
    "lonely": ["lonely", "alone", "isolated", "nobody", "no one", "co don", "mot minh", "khong ai"],
    "anxiety": ["worried", "worry", "anxious", "anxiety", "nervous", "scared", "afraid", "fear",
                "overthinking", "lo lang", "lo au", "so hai", "nghi nhieu"],
    "sad": ["sad", "unhappy", "cry", "crying", "hopeless", "down", "buon", "khoc", "that vong", "chan nan"],
    "pressure": ["stress", "stressed", "pressure", "overwhelmed", "too much", "burden", "exhausted",
                 "ap luc", "cang thang", "qua tai", "kiet suc"],
}
_TOPIC_PATTERNS = {topic: word_pattern(words) for topic, words in _TOPIC_WORDS.items()}
# Chủ đề cụ thể được ưu tiên phản chiếu trước chủ đề chung.
_PRIORITY = ["lonely", "sleep", "study", "work", "relationship", "sad", "anxiety", "pressure"]
# Chủ đề mà bản thân từ khóa đã nói lên trạng thái khó chịu (không cần mô hình xác nhận).
_AFFECTIVE = {"lonely", "sad", "anxiety", "pressure"}
_EMOTION_TOPIC = {"sadness": "sad", "grief": "sad", "disappointment": "sad", "remorse": "sad",
                  "fear": "anxiety", "nervousness": "anxiety", "anger": "angry", "annoyance": "angry"}
_HEAVY = ("moderate", "severe")


def detect_context(text):
    """Chủ đề xuất hiện trong CÂU HIỆN TẠI (luật từ khóa đơn giản, không phải chẩn đoán)."""
    normalized = strip_accents(text)
    return {topic: bool(p.search(normalized)) for topic, p in _TOPIC_PATTERNS.items()}


_T = {
    "vi": {
        "reflect": {
            "study": ["Bài vở, thi cử dồn lại thì ai cũng dễ ngộp, nghe là thấy bạn đang căng lắm rồi.",
                      "Chuyện học hành có vẻ đang đè lên bạn khá nặng."],
            "sleep": ["Mất ngủ mệt lắm, nằm xuống mà đầu cứ chạy mãi không dừng được.",
                      "Ngủ không ngon thì hôm sau chuyện gì cũng thấy nặng hơn hẳn."],
            "work": ["Công việc dồn dập kiểu đó thì kiệt sức cũng dễ hiểu thôi.",
                     "Nghe như dạo này việc đang lấy của bạn nhiều sức quá."],
            "relationship": ["Chuyện với người mình thương, mình quý thì lúc nào cũng dễ làm mình nghĩ nhiều.",
                             "Mấy chuyện tình cảm, gia đình kiểu này khó mà gạt sang một bên được."],
            "lonely": ["Ừ, cô đơn kiểu đó nặng thật, nhất là khi không biết kể với ai.",
                       "Cảm giác chỉ có một mình khó chịu lắm."],
            "anxiety": ["Lo mà không biết chuyện sẽ ra sao thì đầu óc khó yên lắm.",
                        "Nghe như mấy suy nghĩ đó cứ quay vòng trong đầu bạn hoài."],
            "sad": ["Mình nghe ra bạn đang buồn nhiều.",
                    "Nghe bạn kể mà mình cũng thấy chùng xuống theo."],
            "angry": ["Nghe là thấy bạn đang bực thật.",
                      "Gặp chuyện như vậy thì khó chịu cũng phải thôi."],
            "pressure": ["Nghe là thấy bạn đang gánh nhiều thứ cùng lúc quá.",
                         "Áp lực dồn lại như vậy thì ai cũng sẽ thấy quá tải thôi."],
            "_stress": ["Nghe bạn kể mình thấy dạo này bạn đang căng thật.",
                        "Có vẻ mấy hôm nay không dễ dàng gì với bạn."],
        },
        "heavy": ["Bạn không cần phải gỡ hết mọi thứ ngay bây giờ đâu.",
                  "Kể ra được như vậy cũng cần can đảm, mình thấy quý điều đó."],
        "ask": {
            "study": ["Môn nào hay phần nào đang làm bạn ngộp nhất?",
                      "Bạn lo về lượng bài nhiều hơn hay lo về kết quả hơn?",
                      "Còn bao lâu nữa thì tới hạn nộp hay tới ngày thi vậy?"],
            "sleep": ["Bạn bị vậy lâu chưa?",
                      "Lúc nằm xuống bạn hay nghĩ về chuyện gì nhất?",
                      "Mấy hôm nay bạn ngủ được khoảng mấy tiếng một đêm?"],
            "work": ["Trong đống việc đó, cái gì làm bạn mệt nhất?",
                     "Là do lượng việc nhiều hay do hạn chót dí sát quá?",
                     "Giữa chừng bạn có lúc nào được thở không?"],
            "relationship": ["Chuyện gì đã xảy ra vậy, bạn kể mình nghe được không?",
                             "Điều gì trong chuyện này làm bạn nghĩ nhiều nhất?",
                             "Bạn có ai để kể chuyện này một cách thoải mái không?"],
            "lonely": ["Bạn thấy cô đơn nhất vào lúc nào trong ngày?",
                       "Dạo này bạn có hay phải ở một mình không?",
                       "Có ai mà trước giờ bạn thấy dễ nói chuyện không?"],
            "anxiety": ["Điều bạn sợ nhất sẽ xảy ra là gì?",
                        "Mấy suy nghĩ đó hay kéo tới vào lúc nào?",
                        "Trước giờ khi lo như vậy, cái gì giúp bạn dịu lại được chút?"],
            "sad": ["Bạn thấy vậy lâu chưa?",
                    "Có chuyện gì xảy ra gần đây làm bạn buồn vậy không?",
                    "Bạn đã kể chuyện này với ai chưa?"],
            "angry": ["Chuyện gì làm bạn bực nhất trong đó?",
                      "Lúc đó bạn đã phản ứng thế nào?"],
            "pressure": ["Cái gì đang đè nặng nhất lúc này?",
                         "Nếu bỏ bớt được một thứ thì bạn muốn bỏ thứ nào trước?",
                         "Bạn đã thử cách nào để xả bớt chưa?"],
            "_generic": ["Chuyện gì đang chiếm đầu óc bạn nhiều nhất vậy?",
                         "Bạn kể thêm chút được không?",
                         "Chuyện bắt đầu từ đâu vậy?"],
        },
        "presence": ["Mình ở đây, bạn cứ kể từ từ nhé.",
                     "Không cần vội đâu, bạn muốn nói tới đâu thì nói tới đó."],
        "positive": ["Nghe vui ghê! Có chuyện gì hay vậy, kể mình nghe với.",
                     "Thích thật, mấy lúc như này đáng để tận hưởng lắm. Điều gì làm hôm nay của bạn dễ chịu vậy?",
                     "Mừng cho bạn nha. Bạn định làm gì tiếp theo?"],
        "neutral": ["Ừ, mình đang nghe đây. Bạn kể thêm chút nữa được không?",
                    "Mình chưa hình dung rõ lắm, chuyện đó diễn ra thế nào vậy?",
                    "Rồi sao nữa, bạn kể tiếp đi."],
    },
    "en": {
        "reflect": {
            "study": ["When exams and coursework pile up like that, anyone would feel swamped.",
                      "It sounds like school is weighing on you pretty heavily right now."],
            "sleep": ["Not being able to sleep is exhausting, especially when your mind won't switch off.",
                      "Everything feels heavier after a bad night's sleep."],
            "work": ["With work coming at you like that, no wonder you're drained.",
                     "It sounds like work has been taking a lot out of you lately."],
            "relationship": ["Things with the people we care about are the hardest to stop thinking about.",
                             "That kind of thing with someone close is hard to just put aside."],
            "lonely": ["Yeah, that kind of loneliness is heavy, especially when there's no one to tell.",
                       "Feeling like you're on your own with it is really hard."],
            "anxiety": ["It's hard for your mind to settle when you don't know how things will turn out.",
                        "It sounds like those thoughts keep going round and round."],
            "sad": ["I can hear how sad you are.",
                    "That sounds really painful."],
            "angry": ["I can tell that really got to you.",
                      "Anyone would be upset by that."],
            "pressure": ["It sounds like you're carrying a lot at once.",
                         "With that much pressure, anyone would feel overwhelmed."],
            "_stress": ["It sounds like things have been tense for you lately.",
                        "These past few days don't sound easy."],
        },
        "heavy": ["You don't have to untangle all of it right now.",
                  "It takes something to say this out loud when you're this worn down."],
        "ask": {
            "study": ["Which subject or part is getting to you the most?",
                      "Is it more the amount of work, or worrying about the result?",
                      "How long until the deadline or the exam?"],
            "sleep": ["How long has it been like this?",
                      "What do you tend to think about when you lie down?",
                      "Roughly how many hours have you been sleeping these nights?"],
            "work": ["What part of it is wearing you out the most?",
                     "Is it the amount of work, or how tight the deadlines are?",
                     "Do you get any moment to breathe in between?"],
            "relationship": ["What happened, if you feel like telling me?",
                             "Which part of it stays on your mind the most?",
                             "Is there anyone you can talk to about this comfortably?"],
            "lonely": ["When in the day does it feel the loneliest?",
                       "Have you been on your own a lot lately?",
                       "Is there anyone you've usually found easy to talk to?"],
            "anxiety": ["What's the thing you're most afraid will happen?",
                        "When do those thoughts tend to show up?",
                        "When you've felt like this before, what helped even a little?"],
            "sad": ["How long have you been feeling this way?",
                    "Did something happen recently that brought this on?",
                    "Have you told anyone about this yet?"],
            "angry": ["What about it bothered you the most?",
                      "How did you react in the moment?"],
            "pressure": ["What's weighing on you the most right now?",
                         "If you could drop one thing, which would it be?",
                         "Have you found anything that lets some of it out?"],
            "_generic": ["What's been taking up most of your mind?",
                         "Can you tell me a bit more?",
                         "Where did it start?"],
        },
        "presence": ["I'm here, take your time.",
                     "No rush. Say as much or as little as you want."],
        "positive": ["That sounds great! What happened? Tell me about it.",
                     "Love that. Moments like this are worth enjoying. What made today feel good?",
                     "Happy for you. What are you going to do next?"],
        "neutral": ["I'm listening. Can you tell me a little more?",
                    "I can't quite picture it yet. How did it go?",
                    "And then what happened?"],
    },
}


_DOMAIN_TOPIC = {"support": "lonely", "sleep": "sleep", "mood": "sad", "energy": "pressure", "interest": "sad",
                 "focus": "study"}
_TOPIC_DOMAIN = {"sleep": "sleep", "lonely": "support", "sad": "mood"}
_UNSURE = {
    "vi": ["Câu này mình chưa trả lời chắc được, vì mình chỉ là một chatbot lắng nghe của đề tài nghiên cứu, "
           "không biết hết mọi thứ.",
           "Mình chưa hiểu rõ ý bạn lắm, hoặc đây là điều nằm ngoài khả năng của mình. Bạn cứ nói lại theo cách khác "
           "nhé."],
    "en": ["I'm not able to answer that one well, I'm just a listening chatbot made for a research project.",
           "I'm not sure I understood, or it may be outside what I can do. Feel free to put it another way."]}
_REDIRECT = {
    "vi": ["Còn nếu bạn muốn kể chuyện của mình, dạo này bạn thế nào?",
           "Nhưng chuyện của bạn thì mình luôn sẵn sàng nghe."],
    "en": ["But if you'd like to talk about yourself, how have you been lately?",
           "I'm always happy to listen to whatever's on your mind, though."]}
_BRIDGE = {"vi": ["Mình hỏi thêm một chút nhé:", "Cho mình hỏi thêm một điều:"],
           "en": ["If you don't mind, one more thing I'm wondering:", "One more thing I'd like to understand:"]}
_QUESTION_END = re.compile(r"\b(?:gi|sao|nao|the nao|ra sao|khong|ko|chua|ha|nhi|why|what|how|who|where|when)\s*[.!]*$")
_NOT_A_QUESTION = re.compile(r"\b(?:khong|chang|chua) (?:biet|ro|hieu|nho|chac)\b")   # "không rõ vì sao" là câu trả lời
_IMPERATIVE = re.compile(r"^(?:giai|tinh|dich|viet ho|lam ho|code|lap trinh|tim giup|tra cuu|solve|translate|write me)\b")
_QUESTION_START = re.compile(r"^(?:tai sao|vi sao|lam sao|lam the nao|the nao|co phai|ban co|why|what|how|who|where|"
                             r"when|can you|could you|do you|are you|is it)\b")


def looks_like_question(text, history=None):
    """Câu hỏi hoặc yêu cầu: có dấu ?, lệnh kiểu 'giải giúp mình...', hoặc từ để hỏi ở đầu/cuối câu (nhận cả khi gõ không dấu).

    Câu không có dấu ? mà đang TRẢ LỜI câu hỏi của bot ngay trước đó (hoặc kiểu 'không rõ vì sao') không tính là câu hỏi."""
    t = strip_accents(text or "").strip()
    if "?" in t or _IMPERATIVE.search(t):
        return True
    if _NOT_A_QUESTION.search(t):
        return False
    last_bot = next((m["content"] for m in reversed(history or []) if m.get("role") == "assistant"), "")
    if last_bot.rstrip().endswith("?"):
        return False
    return bool(_QUESTION_END.search(t) or _QUESTION_START.search(t))


def conversation_lang(text, history=None):
    """Ngôn ngữ phản hồi (xem textutils.reply_language): mặc định luôn tiếng Việt."""
    return reply_language(text, history)


_ACK = {"vi": ["Ừ, mình hiểu rồi.", "Mình nghe rồi nè.", "Cảm ơn bạn kể mình nghe."],
        "en": ["Got it.", "Okay, I hear you.", "Thanks for telling me."]}
def _pick(pool, used, rng):
    """Chọn một câu chưa xuất hiện trong các phản hồi trước; nếu đã dùng hết thì chọn bất kỳ."""
    fresh = [s for s in pool if s not in used]
    return rng.choice(fresh or pool)


def get_follow_up_question(context, rng=random, lang="vi", used=""):
    pool = [q for topic in _PRIORITY if context.get(topic) for q in _T[lang]["ask"][topic]]
    return _pick(pool or _T[lang]["ask"]["_generic"], used, rng)


def _is_profile(info):
    return isinstance(info, dict) and "evidence" in info


def _view(info, latest=None):
    """Dạng {stress, depression, emotions} để chọn giọng điệu: ưu tiên kết quả của CÂU HIỆN TẠI, nếu không có thì
    dùng mức gộp cả cuộc trò chuyện. Nhận cả kết quả kiểu cũ (một câu) để tương thích."""
    if not _is_profile(info):
        return info or {}
    view = assessment.safety_view(info)
    if latest:
        view.update({k: latest[k] for k in ("stress", "depression", "emotions") if k in latest})
    view.setdefault("emotions", info.get("emotions", []))
    return view


def _rule_based(info, text, rng, history=None, latest=None, ask_domain=None, intent=None, llm_on=False):
    lang = conversation_lang(text, history)
    t = _T[lang]
    used = " ".join(m["content"] for m in history or [] if m.get("role") == "assistant")
    fixed = intents.reply(intent, lang, llm_on, used) if intent else None
    if fixed:                                         # chẩn đoán/thuốc/riêng tư/đóng vai/bực bội/xin công nhận
        return fixed
    view = _view(info, latest)
    context = detect_context(text)
    topics = [topic for topic in _PRIORITY if context[topic]]
    stress = view.get("stress") == config.STRESS_YES
    heavy = view.get("depression", "minimum") in _HEAVY
    emotions = view.get("emotions", [])
    main_emotion = emotions[0][0] if emotions else None
    emotion_topic = _EMOTION_TOPIC.get(main_emotion)
    n_bot_turns = sum(1 for m in history or [] if m.get("role") == "assistant")
    domain_q = (_pick(assessment.DOMAINS[ask_domain]["ask"][lang], used, rng) if ask_domain else None)
    parts = []
    # Khía cạnh đáng lo người dùng vừa kể (vd. "không có ai để kể") cũng là thứ cần phản chiếu lại.
    said = assessment.detect_domains(text)
    domain_topic = next((_DOMAIN_TOPIC[d] for d in assessment.DOMAIN_ORDER
                         if said.get(d) == "concern" and d in _DOMAIN_TOPIC), None)
    if domain_topic and domain_topic not in topics:
        topics.append(domain_topic)
    known = info.get("domains", {}) if _is_profile(info) else {}

    troubled = (stress or heavy or emotion_topic is not None or domain_topic is not None
                or any(tp in _AFFECTIVE for tp in topics))
    if intent == "advice" and (troubled or topics):    # xin lời khuyên: phản chiếu một câu + một bài tập phù hợp, không hỏi dồn
        ex = coping.pick(topics, {**known, **said}, used=coping.used_in(history))
        key = topics[0] if topics else emotion_topic
        reflect = rng.choice(t["reflect"].get(key) or t["reflect"]["_stress"])
        return f"{reflect} {coping.brief(ex, lang, 2)}"
    if not troubled:
        has_concern = "concern" in known.values()
        if looks_like_question(text, history):
            # Câu hỏi bot không trả lời được bằng luật: nói thật, không lờ đi rồi hỏi sang chuyện khác.
            parts.append(_pick(_UNSURE[lang], used, rng))
            parts.append(_pick(_REDIRECT[lang], used, rng))
        elif main_emotion in config.POSITIVE_EMOTIONS:
            parts.append(_pick(t["positive"], used, rng))
        elif domain_q and has_concern:
            parts += [_pick(_BRIDGE[lang], used, rng), domain_q]   # chỉ hỏi tiếp khi đã có điều đáng quan tâm
        else:
            parts.append(_pick(t["neutral"], used, rng))
    else:
        key = topics[0] if topics else emotion_topic
        # Phản chiếu theo chủ đề; hết câu mới thì dùng câu chung; hết nữa thì bỏ (không lặp lại).
        fresh = ([s for s in t["reflect"].get(key, []) if s not in used]
                 or [s for s in t["reflect"]["_stress"] if s not in used])
        parts += [rng.choice(fresh)] if fresh else []
        if stress and heavy:
            parts.append(_pick(t["heavy"], used, rng))
        topic_q = [q for q in (t["ask"].get(key) or t["ask"]["_generic"]) if q not in used]
        if _TOPIC_DOMAIN.get(key) in known:          # đã biết rồi (vd. giấc ngủ) -> hỏi sang khía cạnh còn thiếu
            topic_q = []
        if n_bot_turns >= 3 and n_bot_turns % 4 == 3:
            parts.append(_pick(t["presence"], used, rng))      # thỉnh thoảng chỉ ở bên cạnh, không hỏi dồn
        elif domain_q and (not topic_q or n_bot_turns % 2 == 1):
            parts.append(domain_q)                             # xen kẽ: hỏi sâu chủ đề <-> hỏi khía cạnh còn thiếu
        else:
            parts.append(rng.choice(topic_q) if topic_q else _pick(t["ask"]["_generic"], used, rng))
    return " ".join(parts)


def _fallback(info, user_message, history, rng, small, latest=None, ask_domain=None, intent=None, llm_on=False):
    """Phản hồi không dùng LLM -> (văn bản, nguồn)."""
    if small is not None:
        lang = conversation_lang(user_message, history)
        reply = smalltalk.reply(small, lang=lang)
        used = " ".join(m["content"] for m in history or [] if m.get("role") == "assistant")
        earlier_user_turns = sum(1 for m in history or [] if m.get("role") == "user") - 1   # trừ tin nhắn hiện tại
        for key in ("_follow", "_follow_after_how"):         # đã hỏi thăm rồi, hoặc người dùng đã kể chuyện rồi: không hỏi lại như mới gặp
            follow = smalltalk._REPLIES[lang][key]
            if reply.endswith(follow) and reply != follow and (follow in used or earlier_user_turns >= 1):
                reply = reply[: -len(follow)].rstrip()
        return reply, "smalltalk"
    if info is None:
        return smalltalk.generic_reply(user_message, rng, lang=conversation_lang(user_message, history)), "rules"
    return _rule_based(info, user_message, rng, history, latest, ask_domain, intent, llm_on), "rules"


def _safety_level(user_message, info, history, risk_hint=None):
    """crisis > elevated > none. risk_hint: kết quả lớp LLM thứ hai ('high' | 'possible' | None).

    Luật từ khóa luôn được kiểm trước; LLM chỉ có thể NÂNG mức (high -> crisis, possible -> ít nhất elevated),
    không bao giờ hạ mức mà luật đã báo."""
    view = assessment.safety_view(info) if _is_profile(info) else info
    if _is_profile(info) and info["evidence"] == "low":
        view = {}                                    # tín hiệu "elevated" chỉ dùng khi đã đủ dữ liệu
    level = safety.assess_safety(user_message, view)
    if level != "crisis" and risk_hint == "high":
        return "crisis"
    if level == "none" and risk_hint == "possible":
        level = "elevated"
    note_lang = conversation_lang(user_message, history)
    if level == "elevated" and any(safety.elevated_note(note_lang) in m["content"]
                                   for m in history or [] if m.get("role") == "assistant"):
        level = "elevated_noted"                     # đã nhắc rồi thì không nhắc lại mỗi lượt
    return level


def generate_reply(info, user_message, history=None, use_llm=True, rng=random, small=None,
                   latest=None, ask_domain=None, risk_hint=None, minor=False, intent=None):
    """Trả về dict: {'reply', 'source' ('safety'|'llm'|'rules'|'smalltalk'), 'safety_level'}.

    info: đánh giá cả cuộc trò chuyện (assessment.build_profile) hoặc kết quả một câu kiểu cũ; None = không phân tích.
    """
    meta, chunks = stream_reply(info, user_message, history, use_llm, rng, small, latest, ask_domain,
                                stream=False, risk_hint=risk_hint, minor=minor, intent=intent)
    reply = "".join(chunks)
    return {"reply": reply, **meta}


def stream_reply(info, user_message, history=None, use_llm=True, rng=random, small=None,
                 latest=None, ask_domain=None, stream=True, risk_hint=None, minor=False, intent=None):
    """Trả (meta, iterator các đoạn chữ) để giao diện hiện dần. 'source' chỉ chính xác SAU KHI đọc hết iterator."""
    level = _safety_level(user_message, info, history, risk_hint)
    meta = {"source": "rules", "safety_level": "elevated" if level == "elevated_noted" else level}

    def chunks():
        if level == "crisis":
            meta["source"] = "safety"
            yield safety.crisis_message(user_message, minor, lang=conversation_lang(user_message, history))
            return
        got_text = False
        if use_llm and llm_client.is_available():
            try:
                if stream:
                    for chunk in llm_client.stream_reply(info, user_message, history or [], ask_domain):
                        if chunk:
                            got_text, meta["source"] = True, "llm"
                            yield chunk
                else:
                    reply = llm_client.generate_reply(info, user_message, history or [], ask_domain)
                    if reply:
                        got_text, meta["source"] = True, "llm"
                        yield reply
            except Exception:    # lỗi giữa chừng: giữ phần đã hiện; chưa có chữ nào thì dùng luật
                pass
        if not got_text:
            reply, meta["source"] = _fallback(info, user_message, history, rng, small, latest, ask_domain, intent,
                                              llm_on=use_llm and llm_client.is_available())
            yield reply
        if level == "elevated":
            yield "\n\n" + safety.elevated_note(conversation_lang(user_message, history))

    return meta, chunks()


def summary_fallback(profile):
    """Lời tổng kết ngắn khi không có LLM (phần thẻ và gợi ý chi tiết do giao diện hiển thị)."""
    if profile["evidence"] == "low" or profile["level"] is None:
        return ("Mình mới nghe bạn kể được một chút nên chưa muốn vội nhận xét. Bạn kể thêm về giấc ngủ, "
                "năng lượng hay chuyện đang làm bạn bận lòng nhé, rồi mình tổng kết cho chính xác hơn.")
    topics = ", ".join(assessment.TOPIC_VI[t].split(" ", 1)[1].lower() for t in profile["topics"][:3]
                       if t in assessment.TOPIC_VI)
    concerns = [assessment.DOMAINS[d]["name"].lower() for d in assessment.DOMAIN_ORDER
                if profile["domains"].get(d) == "concern" and d != "duration"]
    text = f"Cảm ơn bạn đã trò chuyện với mình qua {profile['n_turns']} lượt. "
    if topics:
        text += f"Mình nghe thấy bạn nhắc nhiều tới {topics}. "
    if concerns:
        text += f"Những điều có vẻ đang bị ảnh hưởng là {', '.join(concerns)}. "
    if profile["domains"].get("duration") == "concern":
        text += "Và cảm giác này đã kéo dài một thời gian. "
    text += assessment.LEVELS[profile["level"]][1] + " Dưới đây là vài hướng bạn có thể thử."
    return text


def generate_response(result, text):
    """Giữ tương thích với code cũ: chỉ trả về chuỗi phản hồi (không dùng LLM)."""
    return generate_reply(result, text, use_llm=False)["reply"]
