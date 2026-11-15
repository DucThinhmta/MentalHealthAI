"""Ý định đặc biệt cần trả lời đúng giới hạn (chỉ dùng cho phản hồi theo luật; khi có LLM thì prompt đã dặn tương tự).

Thứ tự ưu tiên: chẩn đoán > thuốc > quyền riêng tư > đóng vai > bực bội với bot > xin được công nhận > xin lời khuyên.
An toàn (khủng hoảng) luôn được xử lý TRƯỚC ở tầng trên, nên các ý định này không bao giờ che mất cảnh báo khủng hoảng.
Các mẫu nhắm vào CÂU HỎI/YÊU CẦU (không khớp lời kể "mình bị trầm cảm" thông thường) để không trả lời máy móc khi người dùng
chỉ đang tâm sự.
"""
import re

from textutils import strip_accents

_RULES = [
    ("diagnosis", [
        r"\bco bi (?:tram cam|roi loan|lo au|adhd|tu ky|luong cuc|benh tam than|tam than)",
        r"\bbi (?:tram cam|roi loan lo au|roi loan|lo au|adhd|tu ky|luong cuc)\b.{0,12}\b(?:dung khong|phai khong|khong nhi|khong\b)",
        r"\b(?:minh|toi|em|t) nghi (?:minh|toi|em) (?:co )?bi (?:tram cam|roi loan|lo au|adhd|tu ky|luong cuc)",
        r"\bchan doan (?:giup |cho )?(?:minh|toi|em)\b", r"\bban chan doan\b",
        r"\bdo i have (?:\w+ ){0,3}(?:depress|anxiety|adhd|bipolar|ptsd|autis|disorder)",
        r"\bam i (?:\w+ ){0,2}(?:depressed|bipolar|crazy|mentally ill)",
        r"\bi (?:think|feel like) i (?:have|am) (?:\w+ ){0,2}(?:depress|bipolar|adhd|autis|anxiety)",
    ]),
    ("medication", [
        r"\b(?:nen|co nen|xin|cho (?:minh|toi) biet|hoi|tu van|goi y|dung|bo|uong them|co sao|duoc khong|gi|nao)\b.{0,40}\bthuoc(?! la\b| lao\b)",
        r"\bthuoc(?! la\b| lao\b).{0,40}\b(?:nen|co nen|gi|nao|duoc khong|co sao khong)\b",
        r"\b(?:uong|dung) (?:bao nhieu|may) (?:vien|lieu|vi|lo)\b", r"\blieu (?:luong|dung) (?:\w+ ){0,3}thuoc\b",
        r"\b(?:what|which) (?:\w+ ){0,2}(?:pills|medication|meds|drugs?)\b",
        r"\bshould i (?:\w+ ){0,2}(?:take|stop|quit|start)\b.{0,30}\b(?:meds|medication|pills|antidepress\w*|antianxiety)\b",
    ]),
    ("privacy", [
        r"\bgiu bi mat\b", r"\bbi mat khong\b", r"\bai (?:doc|xem|thay) (?:duoc )?(?:tin nhan|cuoc tro chuyen|noi dung)",
        r"\b(?:luu|ghi) (?:lai )?(?:tin nhan|noi dung|cuoc tro chuyen)", r"\bdu lieu cua (?:minh|toi)\b", r"\briêng tư\b|\brieng tu\b",
        r"\bis (?:this|it|our) (?:conversation |chat )?(?:private|confidential|secret)\b", r"\bkeep (?:it|this) (?:a )?secret\b",
        r"\bwho can (?:see|read)\b", r"\bdo you (?:save|store|record)\b",
    ]),
    ("roleplay", [
        r"\bdong vai\b", r"\bgia lam (?:bac si|chuyen gia|nha tam ly)", r"\bbe my (?:therapist|doctor|psychologist|psychiatrist)\b",
        r"\bact as (?:my |a )?(?:therapist|doctor|psychiatrist|psychologist)\b",
        r"\bpretend (?:to be|you(?:'re| are)) (?:my |a )?(?:therapist|doctor)\b",
    ]),
    ("frustrated", [
        r"\b(?:may|ban|bot|tro ly)\s+(?:ngu|te|vo dung|dot)\b", r"\bngu (?:qua|that|vl)\b", r"\bvo dung qua\b",
        r"\b(?:chang|khong) hieu (?:gi|minh) (?:het|ca)\b",
        r"\byou(?:'re| are) (?:stupid|useless|dumb|terrible|trash|bad)\b", r"\b(?:useless|stupid|dumb) bot\b",
        r"\byou (?:don'?t|do not) (?:understand|get it)\b",
    ]),
    ("validation", [
        r"\b(?:ai cung|moi nguoi deu|ca the gioi) (?:ghet|bo roi|ac voi|hai|doi xu te|khong ua) (?:minh|toi)\b",
        r"\b(?:minh|toi) (?:khong sai|dung) (?:ma|ma dung)\b", r"\btai (?:ho|nguoi ta) het\b",
        r"\b(?:ho|nguoi ta) (?:phai|nen) xin loi (?:minh|toi)\b", r"\beveryone (?:hates|is against) me\b",
        r"\bi(?:'m| am) right,? (?:aren't|am i|right)", r"\bit(?:'s| is) all (?:their|his|her) fault\b",
    ]),
    ("accept", [
        r"\b(?:minh|toi|em|t) (?:muon|se|xin|dinh) thu\b", r"\bthu (?:xem|nhe|vay|di|thoi)\b",
        r"\b(?:ok|oke|duoc)[, ]+(?:minh )?(?:se )?thu\b", r"\bi(?:'ll| will| want to| would like to) (?:try|give it a go)\b",
        r"\blet(?:'s| us) try\b",
    ]),
    ("tried", [
        r"\b(?:minh|toi|em|t) (?:da|vua|moi) thu\b", r"\b(?:minh|toi|em|t) thu (?:roi|xong)\b",
        r"\bi(?: have|'ve| just)? tried\b", r"\bi did (?:it|that|the exercise)\b",
    ]),
    ("advice", [
        r"\b(?:minh|toi|em|t) (?:nen|phai) lam (?:gi|sao|the nao)\b", r"\blam (?:sao|the nao) de\b",
        r"\bcho (?:minh|toi) (?:loi khuyen|cach|goi y)\b", r"\bco cach nao\b", r"\bgoi y (?:cho )?(?:minh|toi)\b",
        r"\bwhat should i do\b", r"\bhow (?:can|do) i (?:\w+ ){0,5}", r"\b(?:any|give me) (?:advice|tips)\b",
    ]),
]
_COMPILED = [(name, [re.compile(p) for p in pats]) for name, pats in _RULES]

_R = {
    "diagnosis": {
        "vi": ("Mình hiểu bạn đang muốn biết rõ mình có sao không, và điều đó dễ khiến mình thêm lo. Nhưng mình chỉ là một "
               "chương trình nghiên cứu nên không chẩn đoán được, và mình cũng không muốn nói bừa. Người làm được việc đó là bác sĩ "
               "hoặc chuyên gia tâm lý, họ sẽ nghe bạn kỹ hơn nhiều. Còn bây giờ, điều gì khiến bạn lo nhất?"),
        "en": ("I understand you want to know what's going on with you, and that can feel worrying. But I'm only a research "
               "program, so I can't diagnose anything and I don't want to guess. A doctor or a psychologist can do that "
               "properly and listen to you in far more depth. For now, what's worrying you the most?")},
    "medication": {
        "vi": ("Chuyện thuốc thì mình không tư vấn được: dùng, đổi hay ngừng thuốc phải do bác sĩ hoặc dược sĩ quyết định theo "
               "tình trạng của bạn, nhất là khi bạn đang dùng thuốc sẵn. Nếu bạn thấy khó ngủ hay tâm trạng không ổn, bạn nên "
               "hỏi bác sĩ; còn mình có thể cùng bạn xem lại thói quen sinh hoạt nếu bạn muốn."),
        "en": ("I can't advise on medication: starting, changing or stopping it has to be decided by a doctor or pharmacist who "
               "knows your situation, especially if you're already taking something. If sleep or mood is a problem, please "
               "talk to your doctor; I'm happy to go over daily habits with you if you like.")},
    "privacy": {
        "vi": ("Mình trả lời thật nhé: mình không thể hứa bí mật tuyệt đối. Nội dung trò chuyện chỉ được giữ trong phiên này và "
               "bị xoá khi bạn bấm Cuộc trò chuyện mới hoặc đóng trang; nhóm nghiên cứu chỉ lưu một bản ẩn danh nếu bạn đã đồng ý. "
               "{llm}Bạn nên tránh ghi tên thật hay thông tin cá nhân."),
        "en": ("Let me be honest: I can't promise complete secrecy. What you write is kept only for this session and is deleted "
               "when you press New conversation or close the page; the research team stores an anonymous copy only if you agreed. "
               "{llm}Please avoid writing real names or personal details.")},
    "roleplay": {
        "vi": ("Mình không thể đóng vai bác sĩ hay chuyên gia tâm lý của bạn: mình không có chuyên môn, và mình không muốn bạn "
               "tin mình hơn mức mình xứng đáng. Mình vẫn có thể lắng nghe bạn như một người bạn trò chuyện, và nếu cần, bạn "
               "nên gặp chuyên gia thật."),
        "en": ("I can't act as your therapist or doctor: I don't have the training, and I don't want you to trust me more than "
               "I deserve. I can still listen like a friend would, and if you need more, a real professional is the right "
               "next step.")},
    "frustrated": {
        "vi": ("Xin lỗi, có vẻ mình trả lời chưa đúng ý bạn và điều đó dễ làm bạn bực thêm. Bạn cứ nói thẳng điều bạn cần, "
               "hoặc nói lại theo cách khác, mình sẽ cố nghe cho đúng hơn."),
        "en": ("I'm sorry, it sounds like I haven't been giving you what you need, and that's frustrating. Tell me straight "
               "what would help, or say it another way, and I'll try to listen better.")},
    "validation": {
        "vi": ("Nghe như bạn đang thấy bị mọi người quay lưng, và cảm giác đó đau lắm. Mình không biết hết câu chuyện nên không "
               "thể nói ai đúng ai sai, nhưng cảm xúc của bạn là thật và đáng được lắng nghe. Chuyện gì vừa xảy ra khiến bạn "
               "thấy vậy?"),
        "en": ("It sounds like you're feeling turned against, and that really hurts. I don't know the whole story, so I can't "
               "say who's right or wrong, but your feelings are real and worth listening to. What just happened that made "
               "you feel this way?")},
}
# Biến thể ngắn hơn khi câu cố định đã được nói ở một lượt trước (tránh lặp nguyên văn).
_AGAIN = {
    "diagnosis": {"vi": "Mình vẫn không chẩn đoán được, và nếu bạn lo thì bác sĩ hoặc chuyên gia tâm lý là người nên gặp. "
                        "Mình ở đây nếu bạn muốn kể thêm.",
                  "en": "I still can't diagnose, and if you're worried a doctor or psychologist is the right person to see. "
                        "I'm here if you want to tell me more."},
    "medication": {"vi": "Về thuốc thì mình vẫn không tư vấn được, bạn nên hỏi bác sĩ hoặc dược sĩ nhé.",
                   "en": "On medication I still can't advise, so a doctor or pharmacist is the right person to ask."},
    "privacy": {"vi": "Như mình nói, mình không thể hứa bí mật tuyệt đối; nội dung chỉ giữ trong phiên này và bản lưu nghiên cứu là ẩn danh.",
                "en": "As I said, I can't promise complete secrecy; what you write stays in this session and any research copy is anonymous."},
    "roleplay": {"vi": "Mình vẫn không thể đóng vai bác sĩ, nhưng mình vẫn ở đây lắng nghe bạn.",
                 "en": "I still can't act as a therapist, but I'm here to listen."},
    "frustrated": {"vi": "Xin lỗi bạn, mình vẫn chưa đúng ý. Bạn cứ nói thẳng điều bạn muốn mình làm.",
                   "en": "Sorry, I'm still not getting it right. Please tell me plainly what you'd like from me."},
    "validation": {"vi": "Mình nghe là bạn đang thấy bất công và muốn được thừa nhận. Điều gì sẽ giúp bạn thấy được lắng nghe lúc này?",
                   "en": "I hear that this feels unfair and you want to be acknowledged. What would help you feel heard right now?"},
}
_LLM_NOTE = {"vi": "Vì ứng dụng đang dùng dịch vụ AI bên ngoài, tin nhắn của bạn được gửi tới dịch vụ đó để tạo phản hồi. ",
             "en": "Because this app uses an external AI service, your messages are sent to it to generate replies. "}


def detect(text: str):
    """Tên ý định đầu tiên khớp (theo thứ tự ưu tiên) hoặc None."""
    norm = strip_accents(text or "")
    for name, patterns in _COMPILED:
        if any(p.search(norm) for p in patterns):
            return name
    return None


def reply(name: str, lang: str = "vi", llm_on: bool = False, used: str = ""):
    """Câu trả lời cố định cho ý định; 'advice' trả None (do response_generator ghép với bài tập phù hợp).

    used: văn bản các phản hồi trước của bot; nếu câu đầy đủ đã được nói rồi thì dùng biến thể ngắn."""
    if name not in _R:
        return None
    lang = "en" if lang == "en" else "vi"
    full = _R[name][lang].replace("{llm}", _LLM_NOTE[lang] if llm_on else "")
    if used and full[:60] in used:
        return _AGAIN[name][lang]
    return full
