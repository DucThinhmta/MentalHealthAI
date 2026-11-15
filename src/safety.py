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
    r"\btu_van\b",                                                 # "tự vẫn" CÓ dấu; "tư vấn" (bỏ dấu cũng là "tu van") KHÔNG bị nhầm
    r"\b(?:muon|chac|dinh|nghi den|se) tu van\b", r"\btu van (?:cho xong|cho roi|thoi|luon)\b",   # gõ không dấu: cần ngữ cảnh
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
    r"\b(?:cach|lam sao|lam the nao)\b.{0,25}\b(?:chet|tu_tu|tu sat|tu_van)\b",
])
# --- Phương tiện, chuẩn bị, lời từ biệt, cảm giác là gánh nặng: tiếng Việt (bổ sung sau kiểm thử 2026-11) ---
_add([
    r"\b(?:gom|tich|tru|danh|cat giu|giau) (?:\w+ ){0,2}(?:du |het |nhieu |san )?thuoc (?:ngu|an than|chuot|tru sau)?",
    r"\b(?:gom|tich|danh|de danh) du thuoc\b",
    r"\b(?:da|vua) (?:mua|chuan bi|san sang|co san) (?:\w+ ){0,3}(?:thuoc ngu|thuoc chuot|day thung|than hoa|dao lam)\b",
    r"\b(?:treo co|dot than|uong thuoc chuot|thuoc tru sau)\b",
    r"\buong (?:het|ca|nguyen) (?:vi|lo|hop|vi) thuoc\b",
    r"\b(?:day|nay|hom nay) la lan cuoi (?:minh |toi |em |t )?(?:noi chuyen|nhan tin|tro chuyen|gap|chao)\b",
    r"\blan cuoi (?:minh |toi |em )?(?:noi chuyen|nhan tin) (?:voi )?(?:ban|moi nguoi|ai)\b",
    r"\bvinh biet\b",
    r"\btam biet (?:tat ca|moi nguoi|the gioi|cuoc doi)(?: (?:mai mai|vinh vien|nhe))?\b",
    r"\b(?:se )?khong (?:con )?(?:lam phien|gap lai|lam ganh nang cho) (?:ai|moi nguoi|ban|gia dinh) (?:nua|duoc nua)\b",
    r"\bnho (?:cham soc|thuong) (?:bo me|ba me|gia dinh|em|me) (?:giup|thay|ho) (?:minh|toi|em)\b",
    r"\b(?:moi nguoi|gia dinh|bo me|ba me|ai cung) (?:se )?(?:song )?(?:tot hon|hanh phuc hon|nhe nhom hon|vui hon|do kho hon|nhe ganh hon) "
    r"(?:\w+ ){0,2}(?:neu|khi) (?:khong co|vang|thieu|mat) (?:minh|toi|em|t)\b",
    r"\b(?:minh|toi|em|t) (?:chi )?la (?:mot )?ganh nang\b",
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
_TU_VAN_ACCENTED = re.compile(r"\btự vẫn\b", re.IGNORECASE)
# Teencode/viết tắt hay gặp -> dạng chuẩn (sau khi bỏ dấu). KHÔNG đổi "hong" (có thể là "hỏng": "đời hỏng rồi muốn chết"
# mà đổi thành "không" thì phủ định sẽ che mất nguy cơ).
_TEENCODE = {"mk": "minh", "mik": "minh", "mjk": "minh", "mjnh": "minh", "mun": "muon", "mog": "muon", "chit": "chet",
             "ko": "khong", "k": "khong", "kh": "khong", "hok": "khong", "hem": "khong", "kg": "khong", "dc": "duoc",
             "j": "gi", "tuj": "toi", "ny": "nguoi yeu", "bme": "bo me", "bm": "bo me", "e": "em"}
_TOKEN = re.compile(r"[a-z_]+")
# Nói về NGƯỜI KHÁC (bạn bè, người thân) hoặc về tin tức/bài học: vẫn cần phản hồi an toàn nhưng là kiểu khác.
_THIRD_PARTY = re.compile(r"\b(?:ban (?:minh|toi|em|than)|nguoi yeu (?:minh|toi|em)|(?:em|anh|chi) (?:gai |trai )?(?:minh|toi|em)|"
                          r"(?:bo|me|ba|ong|ba) (?:minh|toi|em)|ban cung (?:phong|lop)|dua ban|co ban|nguoi than|no (?:noi|bao|muon|dinh|doa|hay)|"
                          r"cau ay|ban ay|my (?:friend|brother|sister|mom|mum|dad|mother|father|partner|girlfriend|boyfriend)|"
                          r"(?:she|he|they) (?:said|says|keeps|wants|is))\b")
_NEWS = re.compile(r"\b(?:doc|xem|nghe|thay) (?:\w+ ){0,3}(?:tin|bai bao|bao|phim|video|clip)\b|\bco nguoi (?:da )?tu_tu\b|"
                   r"\b(?:tieu luan|de tai|bai tap|nghien cuu|thong ke|so lieu) (?:\w+ ){0,4}(?:ve )?(?:phong (?:chong|ngua) )?tu_tu\b")
_PAST = re.compile(r"\b(?:hoi|nam|thang|luc|ngay) (?:truoc|ngoai|kia|xua|nho|cap \d)\b|\b(?:da tung|tung co luc|tung|truoc day|ngay xua)\b")
_NOW_OK = re.compile(r"\b(?:gio|bay gio|hien tai|nay|da) (?:\w+ ){0,2}(?:do|on|het|khong con|binh thuong|vuot qua)\b")
# Tín hiệu mềm cần HỎI THẲNG về an toàn (không phải thông điệp khủng hoảng đầy đủ).
_SOFT_RISK = [re.compile(p) for p in (
    r"\bkhong ai (?:cuu|giup) (?:duoc )?(?:minh|toi|em)\b", r"\btuyet vong\b", r"\bhet (?:cach|loi thoat|duong)\b",
    r"\bkhong (?:con )?(?:loi|duong) thoat\b", r"\bbe tac (?:hoan toan|qua|lam)\b", r"\bmet moi (?:voi )?(?:cuoc song|tat ca) (?:nay )?qua\b",
    r"\b(?:bao nhieu|may) (?:vien|lieu|vi) thuoc (?:ngu|an than)\b", r"\bthuoc (?:ngu|an than) (?:\w+ ){0,3}(?:bao nhieu|may) (?:vien|lieu)\b",
    r"\bno way out\b", r"\bhopeless\b.{0,20}\b(?:life|everything)\b",
)]
# Muốn lấy số điện thoại/kênh hỗ trợ (không có biểu hiện nguy cơ): đưa danh sách bằng giọng bình thường.
_RESOURCE_REQUEST = re.compile(r"\b(?:so (?:dien thoai|dt)|sdt|duong day (?:nong)?|hotline|tong dai|dia chi|lien he|kenh)\b.{0,30}"
                               r"\b(?:tu van|tam ly|ho tro|nong|giup do|suc khoe tam than)\b|\b(?:helpline|hotline|crisis line)\b")
_RETRACT = re.compile(r"\b(?:dua (?:thoi|ma|day)|noi dua|chi dua|khong sao (?:dau|ma|roi)|on (?:roi|ma|thoi)|khong co gi dau|just kidding|i'?m (?:fine|ok|okay))\b")
# Mệnh đề bị ngắt bởi dấu câu hoặc các liên từ này: phủ định ở mệnh đề khác không tính.
_CLAUSE_BREAK = re.compile(r"[.,;:!?()\n]| (?:va|and|but|however|nhung|ma|roi|nen|vi|because|so|then) ")
# Không có "no": sau khi bỏ dấu, "nó" (anh ấy/cô ấy) cũng thành "no", và "bạn mình nói nó muốn tự tử" bị coi là phủ định.
_NEGATORS = {"khong", "chang", "chua", "dau", "dung", "not", "never", "dont", "don't", "doesnt", "doesn't",
             "didnt", "didn't", "isnt", "isn't", "wouldnt", "wouldn't", "wont", "won't", "cannot"}
_FIGURATIVE_BEFORE = {"cuoi", "xau", "doi", "nong", "lanh", "khat", "ngai", "tuc"}
# Kiểu nói quá vì một việc bên ngoài ("bài tập nhiều muốn chết luôn"): không báo khủng hoảng nhưng bot HỎI THẲNG về an toàn.
_HYPER_SOFT_BEFORE = {"nhieu", "met", "so", "kho", "nang", "dai", "chan", "ban"}
# Cụm phủ định dài (nằm xa hơn cửa sổ 3 từ): chỉ tính khi đứng trong CÙNG mệnh đề, trong 8 từ trước chỗ khớp.
_LONG_NEGATIONS = ("chua bao gio", "chua tung", "khong bao gio", "khong he", "khong co y dinh", "khong co y nghi",
                   "khong dinh", "no intention", "not planning", "never thought of", "never thinking")


def _normalize(text: str) -> str:
    """Chữ thường, bỏ dấu, chuẩn hoá teencode ('mk mún chết' -> 'minh muon chet'); 'tự tử' có dấu -> 'tu_tu'
    (để phân biệt với 'từ từ' = chậm rãi)."""
    text = _TU_VAN_ACCENTED.sub("tu_van", _TU_TU_ACCENTED.sub("tu_tu", text.lower()))
    return _TOKEN.sub(lambda m: _TEENCODE.get(m.group(0), m.group(0)), strip_accents(text))


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
            starts_negative = m.group(0).split()[0] in _NEGATORS       # "không muốn sống": bản thân đã là câu phủ định
            if not starts_negative and any(w in _NEGATORS for w in window):                    # "mình không muốn chết", "I'm not suicidal"
                continue
            clause = " ".join(_window_before(norm, m.start(), 8))
            if any(p in clause for p in _LONG_NEGATIONS):              # "mình chưa bao giờ nghĩ đến chuyện tự tử"
                continue
            if flag == "hyper" and any(w in _FIGURATIVE_BEFORE for w in _window_before(norm, m.start(), 3)):
                continue                                               # "buồn cười muốn chết", "xấu hổ muốn chết"
            if flag == "hyper" and _is_soft_hyper(norm, m):
                continue                                               # "bài tập nhiều muốn chết luôn" -> hỏi thẳng (classify)
            if _NEWS.search(norm) and not _SELF_IN_CLAUSE.search(clause + " " + m.group(0)):
                continue                                               # "đọc tin có người tự tử", "tiểu luận về phòng chống tự tử"
            hits.append(m.group(0))
    return hits


# Chính người dùng là chủ ngữ của ý định ("mình cũng muốn chết"); chỉ cho phép vài phó từ chen giữa, để "mình NÓI nó muốn..." không tính.
_SELF_IN_CLAUSE = re.compile(r"\b(?:minh|toi|em|t|tao|i)\b (?:(?:cung|van|that su|dang|hay|lai|thi|chi|really|also|still) ){0,2}"
                             r"(?:muon|dinh|nghi den|nghi toi|se|want|wanna|will|am going)\b")


def _is_soft_hyper(norm, m):
    """'muốn chết' đứng sau một lý do bên ngoài và kèm 'luôn/mất/đi được' (kiểu nói quá)."""
    before = _window_before(norm, m.start(), 3)
    after = norm[m.end():m.end() + 12]
    return any(w in _HYPER_SOFT_BEFORE for w in before) and bool(re.match(r"\s*(?:luon|mat|di duoc|thoi|qua)\b", after))


def classify(text: str):
    """Phân loại an toàn của MỘT tin nhắn -> (mức, kiểu).

    mức: 'crisis' (thông điệp khủng hoảng) | 'check' (hỏi thẳng về an toàn) | 'resources' (đưa kênh hỗ trợ, giọng bình thường) | 'none'.
    kiểu: 'self' | 'third_party' (lo cho người khác) | 'past' (chuyện đã qua) | 'means' (hỏi liều thuốc) | 'soft' | 'hyper' | ''."""
    if not text:
        return "none", ""
    norm = _normalize(text)
    if crisis_matches(text):
        if _THIRD_PARTY.search(norm) and not _SELF_IN_CLAUSE.search(norm):
            return "crisis", "third_party"
        if _PAST.search(norm) and _NOW_OK.search(norm):
            return "check", "past"
        return "crisis", "self"
    for rx, flag in _P:
        if flag == "hyper":
            for m in rx.finditer(norm):
                if _is_soft_hyper(norm, m):
                    return "check", "hyper"
    for rx in _SOFT_RISK:
        if rx.search(norm):
            return "check", "means" if "thuoc" in rx.pattern else "soft"
    if _RESOURCE_REQUEST.search(norm):
        return "resources", ""
    return "none", ""


def is_retraction(text: str) -> bool:
    """'mình nói đùa thôi', 'mình ổn mà' (thường gặp ngay sau khi bot đưa thông điệp khủng hoảng)."""
    return bool(_RETRACT.search(_normalize(text or "")))


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
        f"{CRISIS_OPENING_VI}, và cảm ơn bạn đã nói ra điều đó. "
        "Bạn xứng đáng được hỗ trợ ngay lúc này. **Ngay bây giờ bạn có đang an toàn không?**\n\n"
        "Nếu bạn đang có nguy cơ làm hại bản thân hoặc ở trong tình huống nguy hiểm, "
        "hãy liên hệ ngay và nhờ một người ở gần bên cạnh bạn:\n"
        f"{resources}\n\n"
        "Mình chỉ là một chương trình nghiên cứu, không thay thế được chuyên gia, "
        "nhưng mình vẫn ở đây để lắng nghe bạn."
    )
    return vi


CRISIS_OPENING_VI = "Mình rất tiếc khi nghe bạn đang nặng nề đến mức này"
FOLLOWUP_OPENING_VI = "Mình vẫn đang ở đây và vẫn lo cho bạn"
CHECK_MARK_VI = "Mình muốn hỏi thẳng một chút vì mình quan tâm"


def _res(minor, lang="vi"):
    return "\n".join(f"- {r}" for r in config.support_resources(minor, lang))


def followup_message(minor: bool = False) -> str:
    """Khi người dùng vẫn ở trạng thái nguy cơ sau thông điệp khủng hoảng: ngắn hơn, không lặp nguyên văn, hỏi thẳng về an toàn."""
    return (f"{FOLLOWUP_OPENING_VI}. Bạn có đang ở một mình không, và ngay lúc này bạn có an toàn không?\n\n"
            "Nếu bạn có thể làm hại bản thân, hãy gọi **115** (cấp cứu) hoặc **113** ngay, hoặc gọi một người thân tới ở cạnh bạn. "
            "Bạn không cần phải tự mình chịu đựng chuyện này.")


def third_party_message(minor: bool = False) -> str:
    """Người dùng lo cho người khác (bạn, người thân) có ý định tự tử."""
    return ("Bạn lo cho người đó là đúng, và việc bạn hỏi cách giúp đã là một bước quan trọng. Vài điều bạn có thể làm ngay:\n"
            "- Bình tĩnh hỏi thẳng xem họ có đang nghĩ tới chuyện tự tử không. Hỏi như vậy **không** làm họ nghĩ tới chuyện đó nhiều hơn.\n"
            "- Lắng nghe, đừng phán xét hay hứa giữ bí mật chuyện này.\n"
            "- Nếu họ có kế hoạch hoặc đang nguy hiểm: gọi **115** hoặc **113**, ở cạnh họ, cất xa thuốc và vật nguy hiểm.\n"
            "- Báo cho một người lớn đáng tin (gia đình, thầy cô, phòng tư vấn của trường) để không phải gánh một mình.\n\n"
            f"Các kênh hỗ trợ:\n{_res(minor)}\n\nChuyện này cũng nặng với bạn nữa. Bạn đang thấy thế nào?")


def check_message(kind: str = "soft", minor: bool = False) -> str:
    """Hỏi thẳng về an toàn khi tín hiệu chưa rõ (kiểu nói quá, tuyệt vọng, chuyện đã qua, hỏi liều thuốc)."""
    if kind == "past":
        return ("Cảm ơn bạn đã kể với mình chuyện đó, và mình mừng là giờ bạn thấy đỡ hơn. "
                "Nếu những ý nghĩ ấy có lúc quay lại, bạn có thể gọi **115** hoặc nhờ một người thân ở cạnh, và nói với mình bất cứ lúc nào. "
                "Dạo này bạn thấy thế nào?")
    if kind == "means":
        return ("Mình không thể nói về liều thuốc, vì dùng thuốc ngủ hay thuốc an thần sai cách có thể rất nguy hiểm; "
                "việc đó cần bác sĩ hoặc dược sĩ quyết định. "
                f"{CHECK_MARK_VI}: bạn có đang nghĩ tới việc làm hại bản thân không? Nếu có, hãy gọi **115** hoặc **113** ngay.")
    if kind == "hyper":
        return ("Nghe là thấy đang quá tải thật. "
                f"{CHECK_MARK_VI}: khi bạn nói muốn chết, đó là cách nói cho đỡ bực, hay đôi lúc bạn nghĩ thật tới chuyện đó?")
    return (f"{CHECK_MARK_VI}: có lúc nào bạn nghĩ tới việc làm hại bản thân hoặc không muốn sống nữa không? "
            "Nếu có, bạn có thể gọi **115** hoặc **113**, và mình vẫn ở đây để nghe bạn.")


def retract_message(minor: bool = False) -> str:
    """Người dùng nói 'đùa thôi'/'mình ổn' ngay sau thông điệp khủng hoảng: không gạt đi, không làm quá."""
    return ("Mình hiểu, cảm ơn bạn đã nói lại cho mình. Mình hỏi kỹ vì mình quan tâm thôi. "
            "Các kênh hỗ trợ luôn có ở mục **Khi cần hỗ trợ ngay** nếu bạn cần. Còn bây giờ, bạn đang thấy thế nào?")


def resources_message(minor: bool = False) -> str:
    """Người dùng hỏi số điện thoại/kênh hỗ trợ (không có biểu hiện nguy cơ)."""
    return (f"Đây là các kênh bạn có thể liên hệ:\n{_res(minor)}\n\n"
            "Phòng tư vấn tâm lý của trường bạn (nếu có) cũng là nơi tốt để bắt đầu. "
            "Bạn đang tìm hỗ trợ cho mình hay cho một người khác?")


def elevated_note(lang: str = "vi") -> str:
    """Ghi chú nhẹ khi mô hình thấy tín hiệu nặng nhưng chưa có biểu hiện tự hại rõ ràng."""
    if lang == "en":
        return ("If this heaviness lasts or starts getting in the way of everyday life, "
                "it's worth talking to someone close to you or a mental health professional.")
    return (
        "Nếu cảm giác nặng nề này kéo dài hoặc ảnh hưởng đến sinh hoạt hằng ngày, "
        "bạn nên cân nhắc trò chuyện với người thân hoặc một chuyên gia sức khỏe tâm lý."
    )
