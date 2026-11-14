"""Quyền riêng tư: xoá thông tin nhận dạng khỏi nhật ký nghiên cứu, mã người tham gia, nội dung màn hình đồng ý.

Nguyên tắc (xem docs/nghien_cuu/PROTOCOL.md):
- Mặc định KHÔNG lưu nội dung hội thoại vào nhật ký/hồ sơ nghiên cứu; chỉ lưu khi người dùng (từ 18 tuổi trở lên) tự bật.
- Khi lưu, mọi nội dung đi qua scrub() trước. scrub() chỉ bắt các mẫu có cấu trúc (email, số điện thoại, đường dẫn, số giấy
  tờ, tên tài khoản); nó KHÔNG thể nhận ra tên người hay địa chỉ viết tự do, nên màn hình đồng ý nhắc người dùng tránh ghi
  tên thật và nhóm nghiên cứu vẫn nên rà soát thủ công trước khi chia sẻ dữ liệu.
"""
import re
import secrets

_RULES = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("[LINK]", re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)),
    # Số điện thoại Việt Nam: 0xx xxx xxxx, +84 xxx..., có thể ngăn cách bằng dấu cách, chấm, gạch
    ("[SĐT]", re.compile(r"(?<![\d/])(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?![\d/])")),
    ("[SỐ]", re.compile(r"(?<![\d/])\d{9,}(?![\d/])")),                     # CMND/CCCD, số thẻ, mã sinh viên dài
    ("[TÀI KHOẢN]", re.compile(r"(?<![\w.])@\w{3,}")),
]

CONSENT_POINTS = [
    "Đây là **công cụ nghiên cứu** (đề tài nghiên cứu khoa học). Nó **không phải chẩn đoán y khoa**, không thay thế "
    "bác sĩ hay chuyên gia tâm lý, và có thể trả lời sai.",
    "Nếu bạn đang gặp nguy hiểm hoặc nghĩ đến chuyện làm hại bản thân, hãy gọi **115** hoặc **113** ngay "
    "(người dưới 18 tuổi có thể gọi **111**).",
    "Nội dung trò chuyện chỉ giữ trong phiên này; bấm **Cuộc trò chuyện mới** hoặc đóng trang là xoá.",
    "Bạn nên **tránh ghi tên thật, địa chỉ, số điện thoại** của mình hay của người khác.",
]
CONSENT_LLM_POINT = ("Ứng dụng đang bật LLM: tin nhắn của bạn được gửi tới dịch vụ AI của Anthropic để tạo phản hồi và "
                     "dịch thuật.")
RESEARCH_OPTION = ("Mình đồng ý cho nhóm nghiên cứu lưu một bản **ẩn danh** của cuộc trò chuyện (đã xoá số điện thoại, "
                   "email, đường dẫn...) cùng các điểm phân tích. Không bắt buộc, và không ảnh hưởng gì nếu bạn không chọn.")
MINOR_NOTE = ("Bạn dưới 18 tuổi: bản ghi nghiên cứu chỉ dành cho người từ 18 tuổi trở lên, nên mình sẽ không lưu gì. "
              "Hãy cân nhắc chia sẻ với một người lớn bạn tin tưởng.")
NOT_STORED = "[không lưu nội dung]"


def scrub(text: str) -> str:
    """Thay email, link, số điện thoại, số giấy tờ và tên tài khoản bằng nhãn. Không đụng tới số thường (giờ, tuổi, điểm)."""
    out = text or ""
    for label, rx in _RULES:
        out = rx.sub(label, out)
    return out


def scrub_entry(entry: dict, keep_text: bool) -> dict:
    """Bản sao của một dòng nhật ký: có đồng ý nghiên cứu -> giữ nội dung đã scrub; không -> bỏ hẳn nội dung."""
    out = dict(entry)
    for key in ("user", "text", "english", "text_for_models", "original_text"):
        if key in out and isinstance(out[key], str):
            out[key] = scrub(out[key]) if keep_text else NOT_STORED
    return out


def new_participant_code() -> str:
    """Mã ngẫu nhiên không liên hệ với người dùng (không dùng tên, email, thời gian)."""
    return "P-" + secrets.token_hex(3).upper()
