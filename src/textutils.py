"""Tiện ích văn bản dùng chung: bỏ dấu tiếng Việt, nhận diện tiếng Việt, chia cửa sổ."""
import re
import unicodedata

_VI_CHARS = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")


def strip_accents(text: str) -> str:
    """Chữ thường, bỏ dấu, đ -> d, chuẩn hóa dấu nháy. Dùng để so khớp từ khóa."""
    text = unicodedata.normalize("NFC", text).lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[’‘`´]", "'", text)


def looks_vietnamese(text: str, min_chars: int = 2) -> bool:
    """Heuristic: có từ 2 ký tự có dấu đặc trưng tiếng Việt trở lên."""
    text = unicodedata.normalize("NFC", text).lower()
    return sum(1 for c in text if c in _VI_CHARS) >= min_chars


# Từ thường gặp khi gõ tiếng Việt KHÔNG dấu (đủ đặc trưng để không lẫn với tiếng Anh).
_VI_PLAIN = re.compile(r"\b(?:minh|toi|tui|khong|nhung|duoc|roi|qua|buon|vui|met|chan|"
                       r"hom nay|cam thay|nhieu|the nao|cam on|xin chao|chao|chac|thi cu|hoc|bai vo|ngu|lo lang|ap luc|"
                       r"co don|ban be|gia dinh|bo me|nguoi yeu|cong viec|that vong|chan nan|mat ngu|kho ngu|nua|nhe|nha|"
                       r"vay|sao|gi|nhi|thoi|dang|bao gio|co ai|khong ai)\b")


def detect_lang(text: str) -> str:
    """'vi' nếu có dấu tiếng Việt hoặc có từ 2 từ tiếng Việt không dấu đặc trưng trở lên; ngược lại 'en'."""
    if looks_vietnamese(text):
        return "vi"
    return "vi" if len(_VI_PLAIN.findall(strip_accents(text))) >= 2 else "en"


_EN_WORDS = {"the", "a", "i", "you", "is", "am", "are", "to", "and", "of", "my", "me", "it", "in", "that", "this", "for",
             "with", "have", "was", "so", "but", "not", "can", "do", "what", "how", "feel", "im"}


def auto_language(text, history=None):
    """Ngôn ngữ của người dùng: theo câu hiện tại; câu ngắn/không dấu khó đoán thì dựa vào các tin trước."""
    if detect_lang(text) == "vi":
        return "vi"
    words = re.findall(r"[a-z']+", (text or "").lower())
    if sum(1 for w in words if w in _EN_WORDS) >= 2:
        return "en"
    users = [m["content"] for m in history or [] if m.get("role") == "user"]
    if users and users[-1] == text:
        users = users[:-1]
    return "vi" if any(detect_lang(m) == "vi" for m in users[-4:]) else "en"


def reply_language(text="", history=None):
    """Ngôn ngữ của câu trả lời của bot: cố định theo config.REPLY_LANGUAGE ('vi' mặc định), hoặc 'auto' = theo người dùng."""
    import config
    mode = getattr(config, "REPLY_LANGUAGE", "vi")
    return mode if mode in ("vi", "en") else auto_language(text, history)


def word_pattern(words):
    """Regex khớp nguyên từ (tránh 'work' khớp 'network', 'test' khớp 'latest')."""
    alternatives = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternatives})\b")


def split_windows(text: str, size: int, step: int):
    """Chia văn bản dài thành các cửa sổ `size` từ, trượt `step` từ."""
    if size <= 0 or step <= 0 or step > size:
        raise ValueError("Cần 0 < step <= size")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    windows, start = [], 0
    while True:
        windows.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            return windows
        start += step
