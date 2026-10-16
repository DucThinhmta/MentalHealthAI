"""Tầng phản hồi: an toàn -> (LLM tùy chọn) -> luật/ngữ cảnh làm phương án dự phòng."""
import random

import config
import llm_client
import safety
from textutils import strip_accents, word_pattern

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


def detect_context(text):
    """Chủ đề xuất hiện trong CÂU HIỆN TẠI (luật từ khóa đơn giản, không phải chẩn đoán)."""
    normalized = strip_accents(text)
    return {topic: bool(p.search(normalized)) for topic, p in _TOPIC_PATTERNS.items()}


_FOLLOW_UPS = {
    "study": ["Điều gì trong việc học hoặc kỳ thi đang khiến bạn áp lực nhất?",
              "Bạn đang lo nhiều hơn về khối lượng bài vở hay về kết quả?",
              "Áp lực học tập này đã ảnh hưởng đến sinh hoạt của bạn như thế nào?"],
    "sleep": ["Tình trạng khó ngủ của bạn bắt đầu từ khi nào?",
              "Bạn thường nghĩ về điều gì khi nằm xuống và cố gắng ngủ?",
              "Giấc ngủ gần đây có ảnh hưởng đến việc học hoặc công việc không?"],
    "work": ["Điều gì trong công việc đang tạo cho bạn nhiều áp lực nhất?",
             "Khối lượng việc hay thời hạn đang khiến bạn mệt hơn?",
             "Giữa những công việc đó, bạn có được nghỉ ngơi không?"],
    "relationship": ["Mối quan hệ này đang khiến bạn cảm thấy thế nào?",
                     "Gần đây có điều gì khiến bạn nghĩ nhiều về mối quan hệ này?",
                     "Bạn có ai để chia sẻ chuyện này một cách thoải mái không?"],
    "lonely": ["Bạn cảm thấy cô đơn nhất vào những lúc nào?",
               "Gần đây bạn có thường phải ở một mình không?",
               "Có ai mà bạn cảm thấy an toàn khi trò chuyện không?"],
    "anxiety": ["Điều gì đang khiến bạn lo lắng nhiều nhất?",
                "Những suy nghĩ này xuất hiện thường xuyên đến mức nào?",
                "Khi lo lắng, điều gì thường giúp bạn bình tĩnh hơn?"],
    "sad": ["Bạn đã cảm thấy buồn như vậy trong bao lâu rồi?",
            "Có sự việc nào gần đây khiến cảm xúc này xuất hiện không?",
            "Bạn có hay chia sẻ cảm xúc này với người thân hoặc bạn bè không?"],
    "pressure": ["Điều gì đang tạo áp lực lớn nhất cho bạn lúc này?",
                 "Áp lực này ảnh hưởng nhiều nhất đến phần nào trong cuộc sống của bạn?",
                 "Bạn đã thử cách nào để giảm bớt áp lực chưa?"],
}
_GENERIC_FOLLOW_UPS = ["Bạn có thể chia sẻ thêm điều gì đang khiến bạn quan tâm không?",
                       "Điều này khiến bạn cảm thấy như thế nào?",
                       "Bạn muốn kể thêm một chút về tình huống này không?"]


def get_follow_up_question(context, rng=random):
    pool = [q for topic, active in context.items() if active for q in _FOLLOW_UPS.get(topic, [])]
    return rng.choice(pool or _GENERIC_FOLLOW_UPS)


_OPENINGS = ["Mình đã đọc những gì bạn chia sẻ.", "Cảm ơn bạn đã chia sẻ điều này.",
             "Mình hiểu điều này có thể đang khiến bạn nghĩ khá nhiều.",
             "Những gì bạn chia sẻ cho thấy đây là điều đang khiến bạn quan tâm."]

_TOPIC_BODY = {  # (khi stress cao) mô tả theo chủ đề; thứ tự = độ ưu tiên
    "sleep": "Mình thấy bạn có dấu hiệu căng thẳng và điều này có vẻ liên quan đến giấc ngủ. "
             "Khi đầu óc cứ nghĩ mãi về một vấn đề, việc nghỉ ngơi sẽ khó hơn.",
    "study": "Áp lực bạn đang gặp có vẻ gắn với việc học hoặc thi cử. "
             "Khi khối lượng việc và kỳ vọng tăng lên, cảm giác quá tải rất dễ xuất hiện.",
    "work": "Mình thấy bạn đang chịu áp lực từ công việc. Phải xử lý nhiều yêu cầu cùng lúc "
            "có thể khiến bạn kiệt sức.",
}
_POSITIVE = {"joy", "gratitude", "love", "optimism", "excitement"}


def _rule_based(result, text, rng):
    context = detect_context(text)
    stress = result.get("stress") == config.STRESS_YES
    depression = result.get("depression", "minimum")
    emotions = result.get("emotions", [])
    main_emotion = emotions[0][0] if emotions else "unknown"
    opening = rng.choice(_OPENINGS)
    question = get_follow_up_question(context, rng)

    if stress and depression in ("moderate", "severe"):
        body = next((_TOPIC_BODY[t] for t in ("sleep", "study", "work") if context[t]),
                    "Nội dung bạn chia sẻ có dấu hiệu căng thẳng và đáng được quan tâm thêm. "
                    "Bạn không nhất thiết phải giải quyết mọi thứ cùng một lúc.")
    elif stress and depression == "mild":
        body = ("Mình thấy bạn có một số dấu hiệu căng thẳng. Có thể sẽ hữu ích nếu xác định cụ thể "
                "điều gì đang tạo áp lực và chia nó thành những phần nhỏ hơn.")
    elif stress:
        if context["study"]:
            body = ("Áp lực học tập có vẻ đang chiếm nhiều sự chú ý của bạn. "
                    "Bạn có thể thử chọn một hoặc hai việc quan trọng nhất để xử lý trước.")
        elif context["sleep"]:
            body = ("Căng thẳng có vẻ đang ảnh hưởng đến việc nghỉ ngơi của bạn. Một khoảng thư giãn "
                    "trước khi ngủ có thể giúp bạn tách khỏi những suy nghĩ về học tập hoặc công việc.")
        elif context["anxiety"]:
            body = ("Bạn có vẻ đang dành nhiều suy nghĩ cho những điều khiến mình lo lắng. "
                    "Thử tập trung vào những việc bạn có thể kiểm soát ngay lúc này.")
        else:
            body = ("Nội dung bạn chia sẻ có dấu hiệu liên quan đến stress. Bạn có thể nghỉ một chút "
                    "và xem điều gì đang tạo áp lực cho mình.")
    elif main_emotion in _POSITIVE:
        return f"{opening} " + rng.choice([
            "Nội dung bạn chia sẻ có xu hướng tích cực. Hãy duy trì những hoạt động và mối quan hệ khiến bạn thoải mái.",
            "Mình thấy một số tín hiệu tích cực trong điều bạn kể. Đó có thể là nguồn động lực tốt.",
        ])
    else:
        body = rng.choice([
            "Mình chưa thấy dấu hiệu stress rõ ràng từ nội dung hiện tại, nhưng một vài câu chưa phản ánh hết trạng thái của một người.",
            "Bạn có thể kể thêm bối cảnh hoặc những gì đang diễn ra gần đây để mình hiểu rõ hơn.",
        ])
    return f"{opening} {body} {question}"


def generate_reply(result, user_message, history=None, use_llm=True, rng=random):
    """Trả về dict: {'reply', 'source' ('safety'|'llm'|'rules'), 'safety_level'}."""
    level = safety.assess_safety(user_message, result)
    if level == "crisis":
        return {"reply": safety.crisis_message(user_message), "source": "safety", "safety_level": level}

    reply, source = None, "rules"
    if use_llm and llm_client.is_available():
        try:
            reply, source = llm_client.generate_reply(result, user_message, history or []), "llm"
        except Exception:        # mất mạng, hết hạn mức... -> dùng phương án dự phòng
            reply = None
    if not reply:
        reply, source = _rule_based(result, user_message, rng), "rules"
    if level == "elevated":
        reply += "\n\n" + safety.elevated_note()
    return {"reply": reply, "source": source, "safety_level": level}


def generate_response(result, text):
    """Giữ tương thích với code cũ: chỉ trả về chuỗi phản hồi (không dùng LLM)."""
    return generate_reply(result, text, use_llm=False)["reply"]
