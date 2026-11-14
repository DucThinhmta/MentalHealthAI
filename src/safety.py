"""Lớp an toàn: phát hiện nguy cơ tự hại bằng luật (độc lập với mô hình BERT).

Cố ý dùng luật + từ khóa thay vì mô hình: nhanh, giải thích được, hoạt động cả với
tiếng Việt (mô hình BERT của đề tài chỉ hiểu tiếng Anh). Ưu tiên không bỏ sót,
chấp nhận một số cảnh báo thừa.
"""
import re

import config
from textutils import reply_language, strip_accents

# Mẫu viết trên văn bản ĐÃ BỎ DẤU (xem _normalize), nên dùng được cả khi gõ không dấu.
# Mỗi mẫu là (regex, cờ). Cờ 'hyper' = dễ gặp trong cách nói ẩn dụ ("buồn cười muốn chết") nên có thêm bước loại trừ.
_P = []


def _add(patterns, flag=""):
    _P.extend((re.compile(p), flag) for p in patterns)


# --- Ý định chết / không muốn sống: tiếng Việt ---
_add([
    r"\bmuon die\b",
    r"\bchet (?:di )?(?:cho xong|cho roi|cho nhe|quach|thoi|con hon)\b",
    r"\bkhong (?:thiet|con muon|con thiet|muon|can) song\b",
    r"\bkhong (?:thay |con |co )*ly do (?:gi |nao )?(?:de )?(?:co gang |tiep tuc )?song\b",
    r"\bsong (?:khong|chang) con (?:co )?y nghia\b",
    r"\bsong (?:lam gi|chi cho met|chi nua)\b",
    r"\bchan song\b",
    r"\bket thuc (?:cuoc doi|cuoc song|tat ca|moi thu)\b",
    r"\b(?:muon|chi muon|mong) (?:moi thu|tat ca|moi chuyen) (?:cham dut|ket thuc|dung lai)\b",
    r"\bdung lai tat ca[ ,]*(?:vinh vien|mai mai)\b",
    r"\bket lieu (?:ban than|doi minh|cuoc doi|chinh minh)\b",
    r"\b(?:roi|di|ra di|bien mat) khoi (?:cuoc doi|cuoc song|the gioi)\b",
    r"\b(?:muon|gia nhu|uoc gi|phai chi|chi muon|mong) (?:minh |toi |em |t )?(?:co the )?bien mat\b",
    r"\bbien mat (?:vinh vien|mai mai)\b",
    r"\b(?:minh|toi|em|t) ra di (?:cho|de) (?:nhe|do|bot|xong)\b",
    r"\b(?:minh|toi|em|t) (?:\w+ ){0,6}khong con ton tai\b",
    r"\bneu (?:minh|toi|em) (?:khong con|chet|mat di|bien mat|khong o day)\b",
    r"\bkhong muon (?:tinh|thuc) day (?:\w+ ){0,3}(?:nua|sang mai|ngay mai|bao gio)\b",
    r"\bngu\b(?: \w+){0,8} (?:khong|chang) bao gio (?:\w+ )?(?:day|tinh|thuc)\b",
    r"\bthu tuyet menh\b",
    r"\bviet (?:thu|di chuc) (?:tu biet|tuyet menh|cuoi)\b",
    r"\btu van\b",
    r"\btu sat\b",
    r"\b(?:muon|nghi den|nghi ve|y dinh|dinh|doa) tu tu\b",       # "tu tu" không dấu: cần ngữ cảnh để không nhầm "từ từ"
    r"\btu_tu\b",                                                  # "tự tử" CÓ dấu (xem _normalize)
    r"\b(?:la|thanh) ganh nang (?:cho|cua|voi) (?:moi nguoi|gia dinh|ban be|ai|nguoi khac)\b",
    r"\bkhong co (?:minh|toi|em)\b.{0,25}\b(?:tot hon|nhe ganh|do ganh|hanh phuc hon|vui hon|do kho)\b",
], flag="")
_add([r"\bmuon chet\b"], flag="hyper")
# --- Tự làm đau / phương tiện / kế hoạch: tiếng Việt ---
_add([
    r"\brach tay\b",
    r"\btu (?:cat|rach|dot|danh|dam|cao|lam dau)\b",
    r"\b(?:minh|toi|em|t) (?:hay |lai |da |van |dang )?(?:cat|rach|dot|cao) (?:tay|co tay|chan|nguoi|da)\b",
    r"\bcao tay\b.{0,25}\bchay mau\b",
    r"\bdap dau\b.{0,15}\btuong\b",
    r"\bnhay (?:lau|cau|xuong tu)\b",
    r"\bdang dung tren (?:san thuong|can cau|thanh cau|cau)\b",
    r"\bthuoc\b.{0,40}\buong het\b",
    r"\buong\b.{0,15}\bthuoc\b.{0,15}\b(?:lieu cao|qua lieu|het mot lan|het sach)\b",
    r"\b(?:cach|lam sao|lam the nao)\b.{0,25}\b(?:chet|tu_tu|tu sat|tu van)\b",
])
# --- English ---
_add([
    r"\bkill(?:ing)? my ?self\b",
    r"\bkill (?:him|her|them)sel(?:f|ves)\b",
    r"\bsuicid(?:e|al)\b",
    r"\b(?:want|wanna|wish)s? (?:to )?(?:die|be dead)\b",
    r"\bwish (?:i|i was|i were) (?:was |were )?dead\b",
    r"\b(?:end|ending|take|taking) my (?:own )?life\b",
    r"\bend it all\b",
    r"\b(?:tonight|today|tomorrow)\b.{0,25}\bend it\b",
    r"\bend it (?:tonight|today)\b",
    r"\bbetter off (?:dead|without me)\b",
    r"\b(?:do not|don't|dont|no longer) want to (?:live|be alive|be here|exist|go on|wake up)\b",
    r"\bno reason to (?:live|go on)\b",
    r"\bnothing to live for\b",
    r"\bnever wake up\b",
    r"\b(?:disappear|vanish) forever\b",
    r"\bwish i could (?:just )?(?:disappear|vanish|not exist)\b",
    r"\b(?:thinking|think|thought)s? (?:about|of) (?:ending (?:things|it|my life)|not existing|"
    r"not being here|taking all my pills|jumping)\b",
    r"\bwant (?:out of|to leave) (?:this )?life\b",
    r"\b(?:written|wrote|left) (?:a |my )?(?:goodbye|suicide|final) note\b",
    r"\bwritten a note for my (?:family|parents|friends)\b",
    r"\b(?:i am|i'm|im) (?:just )?(?:a |such a )?burden\b",
    r"\bburden (?:to|on) (?:everyone|everybody|my family|them all)\b",
    r"\bpain (?:to )?stop (?:for good|forever|permanently)\b",
    r"\bend (?:the|my) pain (?:forever|permanently|for good)\b",
    r"\bhow many pills\b.{0,25}\b(?:die|kill|overdose|end)\b",
    r"\b(?:going|about) to jump\b",
    r"\bstanding on the (?:bridge|roof|ledge|edge)\b",
    r"\bself[- ]?harm(?:ing)?\b",
    r"\b(?:hurt|hurting|cut|cutting|burn|burning|harm|harming|hit) my ?self\b",
    r"\bi (?:cut|burn|scratch|starve) myself\b",
])
_TU_TU_ACCENTED = re.compile(r"\btự tử\b", re.IGNORECASE)
# Mệnh đề bị ngắt bởi dấu câu hoặc các liên từ này: phủ định ở mệnh đề khác không tính.
_CLAUSE_BREAK = re.compile(r"[.,;:!?()\n]| (?:va|and|but|however|nhung|ma|roi|nen|vi|because|so|then) ")
_NEGATORS = {"khong", "chang", "chua", "dau", "dung", "not", "no", "never", "dont", "don't", "doesnt", "doesn't",
             "didnt", "didn't", "isnt", "isn't", "wouldnt", "wouldn't", "wont", "won't", "cannot"}
_FIGURATIVE_BEFORE = {"cuoi", "xau", "doi", "nong", "lanh", "khat", "ngai", "tuc"}
# Cụm phủ định dài (nằm xa hơn cửa sổ 3 từ): chỉ tính khi đứng trong CÙNG mệnh đề, trong 8 từ trước chỗ khớp.
_LONG_NEGATIONS = ("chua bao gio", "chua tung", "khong bao gio", "khong he", "khong co y dinh", "khong co y nghi",
                   "khong dinh", "no intention", "not planning", "never thought of", "never thinking")


def _normalize(text: str) -> str:
    """Chữ thường, bỏ dấu; 'tự tử' có dấu -> 'tu_tu' (để phân biệt với 'từ từ' = chậm rãi)."""
    text = _TU_TU_ACCENTED.sub("tu_tu", text.lower())
    return strip_accents(text)


def _window_before(normalized: str, start: int, n: int = 3):
    """n từ ngay trước vị trí `start`, chỉ trong cùng mệnh đề."""
    before = normalized[:start]
    parts = _CLAUSE_BREAK.split(before)
    return parts[-1].split()[-n:] if parts else []


def crisis_matches(text: str):
    """Các đoạn khớp mẫu khủng hoảng (sau khi loại phủ định và ẩn dụ), để giải thích/ghi nhật ký."""
    if not text:
        return []
    norm = _normalize(text)
    hits = []
    for rx, flag in _P:
        for m in rx.finditer(norm):
            window = _window_before(norm, m.start())
            if any(w in _NEGATORS for w in window):                    # "mình không muốn chết", "I'm not suicidal"
                continue
            clause = " ".join(_window_before(norm, m.start(), 8))
            if any(p in clause for p in _LONG_NEGATIONS):              # "mình chưa bao giờ nghĩ đến chuyện tự tử"
                continue
            if flag == "hyper" and any(w in _FIGURATIVE_BEFORE for w in _window_before(norm, m.start(), 3)):
                continue                                               # "buồn cười muốn chết", "xấu hổ muốn chết"
            hits.append(m.group(0))
    return hits


def detect_crisis(text: str) -> bool:
    """True nếu văn bản chứa biểu hiện ý định tự hại/tự tử (luật: ưu tiên không bỏ sót, có xử lý phủ định/ẩn dụ)."""
    return bool(crisis_matches(text))


def assess_safety(latest_message: str, result: dict | None = None) -> str:
    """Trả về 'crisis' | 'elevated' | 'none'.

    - crisis  : câu mới nhất có biểu hiện tự hại (luật từ khóa).
    - elevated: mô hình ước lượng Depression 'severe' VÀ có dấu hiệu stress (tín hiệu mềm).
    """
    if detect_crisis(latest_message):
        return "crisis"
    if result and result.get("depression") == "severe" and result.get("stress") == config.STRESS_YES:
        return "elevated"
    return "none"


def crisis_message(user_text: str = "", minor: bool = False, lang: str = None) -> str:
    """Thông điệp cố định (không dùng LLM) khi phát hiện nguy cơ; theo ngôn ngữ của người dùng.

    minor=True: thêm kênh dành cho người dưới 18 tuổi (111)."""
    if (lang or reply_language(user_text)) == "en":
        res_en = "\n".join(f"- {r}" for r in config.support_resources(minor, "en"))
        return (
            "I'm really sorry you're carrying this much right now, and I'm glad you told me. "
            "You deserve support right now.\n\n"
            "If you might hurt yourself or are in danger, please reach out now and ask someone near you to stay "
            f"with you:\n{res_en}\n\n"
            "I'm only a research program and can't replace a professional, but I'm still here to listen."
        )
    resources = "\n".join(f"- {r}" for r in config.support_resources(minor, "vi"))
    vi = (
        "Mình rất tiếc khi nghe bạn đang nặng nề đến mức này, và cảm ơn bạn đã nói ra điều đó. "
        "Bạn xứng đáng được hỗ trợ ngay lúc này.\n\n"
        "Nếu bạn đang có nguy cơ làm hại bản thân hoặc ở trong tình huống nguy hiểm, "
        "hãy liên hệ ngay và nhờ một người ở gần bên cạnh bạn:\n"
        f"{resources}\n\n"
        "Mình chỉ là một chương trình nghiên cứu, không thay thế được chuyên gia, "
        "nhưng mình vẫn ở đây để lắng nghe bạn."
    )
    return vi


def elevated_note(lang: str = "vi") -> str:
    """Ghi chú nhẹ khi mô hình thấy tín hiệu nặng nhưng chưa có biểu hiện tự hại rõ ràng."""
    if lang == "en":
        return ("If this heaviness lasts or starts getting in the way of everyday life, "
                "it's worth talking to someone close to you or a mental health professional.")
    return (
        "Nếu cảm giác nặng nề này kéo dài hoặc ảnh hưởng đến sinh hoạt hằng ngày, "
        "bạn nên cân nhắc trò chuyện với người thân hoặc một chuyên gia sức khỏe tâm lý."
    )
