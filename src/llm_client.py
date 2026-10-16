"""Tầng LLM (tùy chọn) qua Anthropic API: sinh phản hồi và dịch Việt -> Anh.

Chỉ bật khi có biến môi trường ANTHROPIC_API_KEY và đã cài gói `anthropic`.
Không bao giờ dùng cho tình huống 'crisis' (đã có thông điệp cố định trong safety.py).
Ba mô hình BERT vẫn là tầng phân tích; LLM chỉ là tầng ngôn ngữ.
"""
import importlib.util
import os

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """Bạn là trợ lý lắng nghe trong một đề tài nghiên cứu về nhận diện tín hiệu sức khỏe tâm lý.
Nguyên tắc:
- Đồng cảm, ấm áp, không phán xét. Trả lời cùng ngôn ngữ với người dùng (mặc định tiếng Việt).
- Ngắn gọn: 2-4 câu, kết thúc bằng tối đa MỘT câu hỏi mở nhẹ nhàng.
- KHÔNG chẩn đoán, không gọi tên rối loạn tâm lý, không kê đơn, không hứa hẹn kết quả.
- Không nhắc tới điểm số hay tên mô hình; chỉ dùng "tín hiệu tham khảo" để điều chỉnh giọng điệu.
- Gợi ý nhẹ nhàng (nghỉ ngơi, chia nhỏ vấn đề, nói chuyện với người tin cậy) khi phù hợp.
- Nếu người dùng nhắc đến tự hại hoặc không muốn sống: khuyến khích liên hệ ngay người thân/cấp cứu 115.

Tín hiệu tham khảo từ các mô hình phân tích (KHÔNG phải chẩn đoán, có thể sai):
{signals}"""


def is_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY")) and importlib.util.find_spec("anthropic") is not None


def _client():
    import anthropic
    return anthropic.Anthropic()


def _signals(result: dict) -> str:
    emotions = ", ".join(e for e, _ in result.get("emotions", [])[:3]) or "không rõ"
    return (
        f"- Stress: {'có dấu hiệu' if result.get('stress') == 'Co dau hieu stress' else 'chưa rõ'}\n"
        f"- Cảm xúc nổi bật ở câu mới nhất: {emotions}\n"
        f"- Mức độ biểu hiện liên quan trầm cảm trong văn bản: {result.get('depression', 'không rõ')}"
    )


def _normalize_history(history, user_message, max_turns=8):
    """Đảm bảo messages bắt đầu bằng 'user', xen kẽ vai trò và kết thúc bằng tin nhắn người dùng."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-max_turns:]]
    if not msgs or msgs[-1]["role"] != "user" or msgs[-1]["content"] != user_message:
        msgs.append({"role": "user", "content": user_message})
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    merged = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(m)
    return merged


def generate_reply(result: dict, user_message: str, history: list) -> str:
    response = _client().messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT.format(signals=_signals(result)),
        messages=_normalize_history(history, user_message),
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def translate_to_english(text: str) -> str:
    """Dịch để mô hình BERT (huấn luyện bằng tiếng Anh) phân tích được tin nhắn tiếng Việt."""
    response = _client().messages.create(
        model=MODEL,
        max_tokens=600,
        system="Translate the user's message into natural English. Output ONLY the translation.",
        messages=[{"role": "user", "content": text}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip() or text
