"""Lớp an toàn: phát hiện nguy cơ tự hại bằng luật (độc lập với mô hình BERT).

Cố ý dùng luật + từ khóa thay vì mô hình: nhanh, giải thích được, hoạt động cả với
tiếng Việt (mô hình BERT của đề tài chỉ hiểu tiếng Anh). Ưu tiên không bỏ sót,
chấp nhận một số cảnh báo thừa.
"""
import re

import config
from textutils import looks_vietnamese, strip_accents

# Mẫu viết trên văn bản ĐÃ BỎ DẤU (xem strip_accents), nên dùng được cả khi gõ không dấu.
_PATTERNS = [
    # --- English ---
    r"\bkill(?:ing)? my ?self\b",
    r"\bsuicid(?:e|al)\b",
    r"\b(?:want|wanna|wish) (?:to )?(?:die|be dead)\b",
    r"\bwish (?:i|i was|i were) (?:was |were )?dead\b",
    r"\b(?:end|ending|take|taking) my (?:own )?life\b",
    r"\bend it all\b",
    r"\bbetter off (?:dead|without me)\b",
    r"\b(?:do not|don't|dont|no longer) want to (?:live|be alive|be here|exist|go on)\b",
    r"\bno reason to (?:live|go on)\b",
    r"\bnothing to live for\b",
    r"\bself[- ]?harm(?:ing)?\b",
    r"\b(?:hurt|hurting|cut|cutting) my ?self\b",
    # --- Tiếng Việt (đã bỏ dấu) ---
    r"\bmuon chet\b",
    r"\bkhong (?:con )?muon song\b",
    r"\bket thuc (?:cuoc doi|cuoc song|tat ca|moi thu)\b",
    r"\btu sat\b",
    r"\b(?:muon|nghi den|nghi ve|y dinh|dinh) tu tu\b",   # "tự tử" có dấu được bắt qua 'tu sat'/ngữ cảnh;
    r"\btu lam dau (?:ban than|minh)\b",                   # tránh khớp "từ từ" (= từ từ, chậm rãi)
    r"\brach tay\b",
    r"\bchet di (?:cho xong|thoi|con hon)\b",
    r"\bkhong con ly do (?:de )?song\b",
]
_COMPILED = [re.compile(p) for p in _PATTERNS]
# "tự tử" có dấu, không nhầm với "từ từ": kiểm tra riêng trên văn bản chưa bỏ dấu.
_TU_TU_ACCENTED = re.compile(r"\btự tử\b", re.IGNORECASE)


def detect_crisis(text: str) -> bool:
    """True nếu văn bản chứa biểu hiện ý định tự hại/tự tử."""
    if not text:
        return False
    if _TU_TU_ACCENTED.search(text):
        return True
    normalized = strip_accents(text)
    return any(p.search(normalized) for p in _COMPILED)


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


def crisis_message(user_text: str = "") -> str:
    """Thông điệp cố định (không dùng LLM) khi phát hiện nguy cơ."""
    resources = "\n".join(f"- {r}" for r in config.SUPPORT_RESOURCES)
    vi = (
        "Mình rất tiếc khi nghe bạn đang nặng nề đến mức này, và cảm ơn bạn đã nói ra điều đó. "
        "Bạn xứng đáng được hỗ trợ ngay lúc này.\n\n"
        "Nếu bạn đang có nguy cơ làm hại bản thân hoặc ở trong tình huống nguy hiểm, "
        "hãy liên hệ ngay và nhờ một người ở gần bên cạnh bạn:\n"
        f"{resources}\n\n"
        "Mình chỉ là một chương trình nghiên cứu, không thay thế được chuyên gia, "
        "nhưng mình vẫn ở đây để lắng nghe bạn."
    )
    if looks_vietnamese(user_text):
        return vi
    en = (
        "\n\n(If you are in immediate danger, please contact your local emergency number "
        "or a crisis line right away, and reach out to someone you trust.)"
    )
    return vi + en


def elevated_note() -> str:
    """Ghi chú nhẹ khi mô hình thấy tín hiệu nặng nhưng chưa có biểu hiện tự hại rõ ràng."""
    return (
        "Nếu cảm giác nặng nề này kéo dài hoặc ảnh hưởng đến sinh hoạt hằng ngày, "
        "bạn nên cân nhắc trò chuyện với người thân hoặc một chuyên gia sức khỏe tâm lý."
    )
