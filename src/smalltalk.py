"""Lớp xã giao: trả lời trực tiếp các câu chào hỏi/giới thiệu, KHÔNG đưa vào phân tích tâm lý.

Mô hình BERT được huấn luyện trên bài đăng Reddit dài nên các câu như "hello" hay
"what's your name" nằm ngoài miền dữ liệu và cho kết quả vô nghĩa. Luật dưới đây bắt các
câu xã giao ĐẦY ĐỦ (cả câu phải là xã giao). "hello, I feel so sad" KHÔNG phải xã giao
nên vẫn được phân tích bình thường. Thứ tự ưu tiên trong ứng dụng: an toàn -> xã giao -> phân tích.
"""
import re
from dataclasses import dataclass

import config
from textutils import strip_accents

_OPT_PRE = r"(?:(?:hey|hi|hello|ok|okay|so|and|please|well)[ ]+)*"
_OPT_POST = r"(?:[ ]+(?:there|bot|please|today|now|mentalhealthai|everyone|nhe|nha|a|ah|oi|ban|di|nao|nhi))*"

# (intent, ngôn ngữ, regex khớp TOÀN BỘ một mệnh đề đã bỏ dấu, chữ thường)
_RULES = [
    ("greeting", "en", r"(?:hello|hi|hey|hiya|howdy|greetings|yo|good (?:morning|afternoon|evening)|hello hello)"),
    ("greeting", "vi", r"(?:xin chao|chao(?: ban| bot| em| anh| chi)?|alo|chao buoi (?:sang|chieu|toi))"),
    ("how_are_you", "en", r"(?:how are you(?: doing| feeling)?|how are u|how r u|how'?s it going|how is it going|what'?s up|whats up|sup|how do you do)"),
    ("how_are_you", "vi", r"(?:ban (?:co )?khoe khong|ban the nao|dao nay ban the nao)"),
    ("ask_name", "en", r"(?:(?:can|may) i (?:know|ask) )?(?:tell me )?(?:what(?:'s| is|s)? (?:your|ur) name|what(?:'s| is)? you called|what are you called|(?:your|ur) name|who are you|who r u)"),
    ("ask_name", "vi", r"(?:ban ten (?:la )?gi|ten (?:cua )?ban la gi|ten ban|ban la ai)"),
    ("capabilities", "en", r"(?:what can you do|what do you do|what are you for|how can you help(?: me)?|can you help me|how do you work|what are you|what is this|what is this app|help)"),
    ("capabilities", "vi", r"(?:ban lam duoc gi|ban giup duoc gi|ban co the lam gi|ban giup gi duoc|ban hoat dong the nao)"),
    ("thanks", "en", r"(?:thanks|thank you|thank u|thx|ty|thanks a lot|thank you so much|thanks so much|many thanks)"),
    ("thanks", "vi", r"(?:cam on(?: ban| nhieu)?|cam on rat nhieu)"),
    ("bye", "en", r"(?:bye|bye bye|goodbye|see you|see you later|see ya|good night|take care|gotta go)"),
    ("bye", "vi", r"(?:tam biet|hen gap lai|chao nhe|ngu ngon|di nhe)"),
    # --- trò chuyện hằng ngày ---
    ("good_status", "en", r"(?:i(?:'m| am| feel)? (?:good|fine|great|ok|okay|alright|well|happy|doing (?:good|well|great|fine))|(?:good|fine|great|not bad|pretty good|all good|doing well|i'm good))"),
    ("good_status", "vi", r"(?:(?:minh|toi|em|t|tui)? ?(?:khoe|on|vui|tot|binh thuong|khong sao)|khoe|on thoi|binh thuong)"),
    ("ack", "en", r"(?:ok|okay|k|kk|oh|ah|i see|got it|cool|nice|alright|sure|yes|yeah|yep|yup|no|nope|nah|lol|haha+|hehe+|hmm+|wow|really|right|great)"),
    ("ack", "vi", r"(?:u|uh|uk|um|vang|da|oke|ok|o|a|a ra vay|the a|hay|tuyet|haha+|hihi+|kkk+|dung roi|khong)"),
    ("bored", "en", r"(?:i'?m bored|i am bored|so bored|bored|nothing to do|i have nothing to do)"),
    ("bored", "vi", r"(?:chan qua|buon chan|chan ghe|chan the|khong co gi lam)"),
    ("who_made", "en", r"(?:who (?:made|created|built|developed) you|who is your (?:creator|developer|maker)|are you a (?:bot|robot|ai|human|real person|person)|are you human|are you real)"),
    ("who_made", "vi", r"(?:ai (?:tao|lam|viet) ra ban|ban la (?:bot|nguoi|ai that|robot|may)|ban co phai la (?:nguoi|bot|robot))"),
    ("sorry", "en", r"(?:sorry|i'?m sorry|my bad|excuse me|pardon)"),
    ("sorry", "vi", r"(?:xin loi|loi cua minh|sorry)"),
    ("joke", "en", r"(?:tell me a joke|say something funny|make me laugh|joke|another joke)"),
    ("joke", "vi", r"(?:ke chuyen cuoi|noi gi do vui di|ke chuyen vui|ke cho minh nghe chuyen cuoi)"),
    ("weather_chat", "en", r"(?:nice weather|what a nice day|it'?s a nice day|how'?s the weather|what'?s the weather like)"),
    ("weather_chat", "vi", r"(?:troi dep qua|hom nay troi dep|thoi tiet the nao|thoi tiet hom nay the nao)"),
    ("compliment", "en", r"(?:you(?:'re| are) (?:great|nice|kind|smart|funny|cool|awesome|good|helpful|cute)|good bot|nice bot|i like you|love you)"),
    ("compliment", "vi", r"(?:ban gioi (?:qua|that)|ban de thuong|ban tot (?:qua|that)|minh thich ban|ban hay (?:qua|that)|ban thong minh)"),
    ("how_old", "en", r"(?:how old are you|what'?s your age|what is your age|when were you born)"),
    ("how_old", "vi", r"(?:ban bao nhieu tuoi|ban may tuoi|ban sinh nam bao nhieu)"),
    ("what_doing", "en", r"(?:what are you doing|what'?re you doing|what you doing|wyd|what are you up to)"),
    ("what_doing", "vi", r"(?:ban dang lam gi|dang lam gi do|ban dang lam gi the)"),
    ("good_morning", "en", r"(?:good morning|morning|gm)"),
    ("good_morning", "vi", r"(?:chao buoi sang|chuc buoi sang tot lanh)"),
]
_COMPILED = [(i, lang, re.compile(rf"{_OPT_PRE}(?:{p}){_OPT_POST}")) for i, lang, p in _RULES]
_SPLIT = re.compile(r"[,.!?;:\n]+|\s+(?:and|va)\s+")


@dataclass(frozen=True)
class SmallTalk:
    intents: tuple
    lang: str          # 'en' | 'vi'


def _normalize(text: str) -> str:
    t = strip_accents(text)
    t = re.sub(r"[^a-z0-9' ,.!?;:\n]", " ", t)
    return re.sub(r"[ ]+", " ", t).strip()


def detect(text: str):
    """SmallTalk nếu MỌI mệnh đề của câu đều là xã giao; ngược lại None."""
    if not text or not text.strip():
        return None
    clauses = [c.strip() for c in _SPLIT.split(_normalize(text)) if c and c.strip()]
    if not clauses or len(clauses) > 4:
        return None
    intents, langs = [], set()
    for clause in clauses:
        hit = next(((i, lang) for i, lang, rx in _COMPILED if rx.fullmatch(clause)), None)
        if hit is None:
            return None
        if hit[0] not in intents:
            intents.append(hit[0])
        langs.add(hit[1])
    return SmallTalk(tuple(intents), "vi" if "vi" in langs else "en")


_REPLIES = {
    "en": {
        "greeting_short": "Hello!",
        "greeting": "Hello! I'm {bot}, a research chatbot that listens and looks for signs of stress and emotions in what you share.",
        "how_are_you": "I'm doing well, thank you for asking!",
        "ask_name": "My name is {bot}. I'm a research assistant, not a doctor, so I can't diagnose anything.",
        "capabilities": ("You can tell me how you've been feeling or what's on your mind. I'll look for signs of "
                         "stress, emotions and how heavy the text sounds, and reply in a supportive way. "
                         "I'm a research tool and my results can be wrong."),
        "thanks": "You're welcome! I'm here whenever you want to talk.",
        "bye": "Goodbye, take care of yourself! You can come back anytime.",
        "good_status": "That's good to hear! If anything is on your mind, even something small, I'm happy to listen.",
        "ack": "I'm here, take your time. Tell me whatever is on your mind.",
        "bored": "Boredom happens to everyone. Want to chat about your day, or something you've been thinking about?",
        "who_made": "I'm an AI chatbot built for a university research project on spotting signs of stress and emotions in text, so I'm not a human.",
        "sorry": "No need to apologize at all. You can say whatever you like here.",
        "joke": "Why do programmers prefer dark mode? Because light attracts bugs!",
        "weather_chat": "I can't see the weather from here, but I hope your day is going nicely!",
        "compliment": "Thank you, that's kind of you to say! I'm glad to keep you company.",
        "how_old": "I'm a chatbot, so I don't really have an age. I'm quite new, built for a research project.",
        "what_doing": "Right now I'm chatting with you and listening, which is my favorite thing to do.",
        "good_morning": "Good morning! I hope you slept well.",
        "_follow": "How are you feeling today?",
        "_follow_after_how": "How about you, how have you been feeling lately?",
    },
    "vi": {
        "greeting_short": "Xin chào!",
        "greeting": "Xin chào! Mình là {bot}, một chatbot nghiên cứu biết lắng nghe và tìm dấu hiệu căng thẳng, cảm xúc trong những gì bạn chia sẻ.",
        "how_are_you": "Mình vẫn ổn, cảm ơn bạn đã hỏi!",
        "ask_name": "Mình tên là {bot}. Mình là trợ lý nghiên cứu, không phải bác sĩ nên không thể chẩn đoán gì cả.",
        "capabilities": ("Bạn có thể kể mình nghe dạo này bạn cảm thấy thế nào hoặc điều đang khiến bạn bận tâm. "
                         "Mình sẽ tìm dấu hiệu stress, cảm xúc và mức độ nặng nề của nội dung rồi phản hồi nhẹ nhàng. "
                         "Mình là công cụ nghiên cứu nên kết quả có thể sai. Lưu ý: các mô hình phân tích hiểu tiếng Anh tốt hơn."),
        "thanks": "Không có gì! Khi nào muốn trò chuyện, mình luôn ở đây.",
        "good_status": "Nghe vậy mình vui lắm! Nếu có điều gì trong đầu, dù nhỏ thôi, bạn cứ kể mình nghe nhé.",
        "ack": "Mình vẫn ở đây, bạn cứ thong thả. Điều gì đang trong đầu bạn thì kể mình nghe nhé.",
        "bored": "Ai cũng có lúc chán mà. Bạn muốn kể về một ngày của bạn, hay điều gì bạn đang nghĩ không?",
        "who_made": "Mình là chatbot AI được xây dựng cho một đề tài nghiên cứu nhận diện dấu hiệu căng thẳng và cảm xúc trong văn bản, nên mình không phải người thật.",
        "sorry": "Bạn không cần xin lỗi đâu. Ở đây bạn cứ nói điều bạn muốn.",
        "joke": "Vì sao lập trình viên thích chế độ tối? Vì ánh sáng thu hút bọ (bug)!",
        "weather_chat": "Mình không nhìn thấy thời tiết ở chỗ bạn, nhưng mong là ngày hôm nay của bạn dễ chịu!",
        "compliment": "Cảm ơn bạn, bạn tốt quá! Mình rất vui được trò chuyện cùng bạn.",
        "how_old": "Mình là chatbot nên không có tuổi thật sự. Mình còn khá mới, được làm cho một đề tài nghiên cứu.",
        "what_doing": "Giờ mình đang trò chuyện và lắng nghe bạn, đó là việc mình thích nhất.",
        "good_morning": "Chào buổi sáng! Hy vọng bạn ngủ ngon.",
        "bye": "Tạm biệt, bạn nhớ giữ gìn sức khỏe nhé! Bạn có thể quay lại bất cứ lúc nào.",
        "_follow": "Hôm nay bạn cảm thấy thế nào?",
        "_follow_after_how": "Còn bạn, dạo này bạn cảm thấy thế nào?",
    },
}


def reply(st: "SmallTalk", bot_name: str = None) -> str:
    """Ghép câu trả lời cho một hoặc nhiều ý xã giao trong cùng một tin nhắn."""
    bot = bot_name or config.BOT_NAME
    r = _REPLIES[st.lang]
    multi = len(st.intents) > 1
    parts = [r["greeting_short" if (i == "greeting" and multi) else i].format(bot=bot) for i in st.intents]
    ask_feelings = any(i in st.intents for i in ("greeting", "how_are_you", "ask_name", "capabilities",
                                                  "good_morning", "who_made", "how_old", "what_doing"))
    if ask_feelings:
        parts.append(r["_follow_after_how"] if "how_are_you" in st.intents else r["_follow"])
    return " ".join(parts)


_GENERIC = {
    "en": ["I'm glad to chat with you! Tell me a bit more, I'm listening.",
           "That sounds interesting. How has your day been so far?",
           "Thanks for sharing. Is there anything on your mind that you'd like to talk about?"],
    "vi": ["Mình rất vui được trò chuyện với bạn! Bạn kể thêm nhé, mình đang nghe đây.",
           "Nghe thú vị đấy. Hôm nay của bạn thế nào?"],
}


def generic_reply(text: str, rng=None) -> str:
    """Câu đáp chung cho tin nhắn trò chuyện thường ngày mà luật không có mẫu riêng."""
    import random
    from textutils import looks_vietnamese
    lang = "vi" if looks_vietnamese(text) else "en"
    return (rng or random).choice(_GENERIC[lang])
