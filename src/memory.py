"""Ghi nhớ giữa các cuộc trò chuyện TRONG CÙNG MỘT PHIÊN TRÌNH DUYỆT, do người dùng kiểm soát.

- Chỉ lưu thứ đã được tóm tắt thành nhãn (chủ đề, khía cạnh đáng chú ý, bài tập đã chọn), KHÔNG lưu nguyên văn tin nhắn.
- Nằm trong bộ nhớ phiên (st.session_state): đóng trang là mất; không ghi ra đĩa.
- Người dùng bật/tắt và xoá được. Tắt thì không ghi thêm và không dùng.
"""
from dataclasses import dataclass, field

import assessment

MAX_TOPICS = 5


@dataclass
class Memory:
    enabled: bool = True
    topics: dict = field(default_factory=dict)          # chủ đề -> số lần nhắc (qua các cuộc trò chuyện)
    concerns: dict = field(default_factory=dict)        # khía cạnh đáng chú ý -> số lần
    exercises: list = field(default_factory=list)       # tên bài tập đã chọn
    chats: int = 0                                      # số cuộc trò chuyện đã được ghi nhớ

    def is_empty(self):
        return not (self.topics or self.concerns or self.exercises)


def remember(mem, profile, plan=()):
    """Ghi nhớ tóm tắt của một cuộc trò chuyện đã xong (khi người dùng chuyển/tạo cuộc mới). Không làm gì nếu đã tắt."""
    if not mem.enabled or profile["n_turns"] < 2:       # cuộc trò chuyện quá ngắn thì không có gì đáng nhớ
        return mem
    for t in profile.get("topics", []):
        mem.topics[t] = mem.topics.get(t, 0) + 1
    for d, v in profile.get("domains", {}).items():
        if v == "concern" and d != "duration":
            mem.concerns[d] = mem.concerns.get(d, 0) + 1
    for p in plan:
        if p["name"] not in mem.exercises:
            mem.exercises.append(p["name"])
    mem.chats += 1
    return mem


def clear(mem):
    mem.topics.clear()
    mem.concerns.clear()
    mem.exercises.clear()
    mem.chats = 0
    return mem


def _names(counts, labels, n=MAX_TOPICS):
    return [labels[k] for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n] if k in labels]


def topic_names(mem):
    return [n.split(" ", 1)[1].lower() for n in _names(mem.topics, assessment.TOPIC_VI)]


def concern_names(mem):
    return [assessment.DOMAINS[d]["name"].lower() for d, _ in sorted(mem.concerns.items(), key=lambda kv: -kv[1])
            if d in assessment.DOMAINS][:MAX_TOPICS]


def text(mem):
    """Một đoạn ngắn mô tả điều đã nhớ (đưa cho LLM và hiển thị cho người dùng); rỗng nếu tắt hoặc chưa có gì."""
    if not mem.enabled or mem.is_empty():
        return ""
    parts = []
    if topic_names(mem):
        parts.append("đã nhắc tới " + ", ".join(topic_names(mem)))
    if concern_names(mem):
        parts.append("khía cạnh đáng chú ý: " + ", ".join(concern_names(mem)))
    if mem.exercises:
        parts.append("đã chọn thử: " + ", ".join(mem.exercises[:3]))
    return "Trong các cuộc trò chuyện trước, người dùng " + "; ".join(parts) + "."


def greeting(mem):
    """Lời nhắc nhẹ ở đầu cuộc trò chuyện mới khi có điều đã nhớ (câu khẳng định, để bot vẫn chỉ hỏi một câu); rỗng nếu không có."""
    names = topic_names(mem)[:2] if mem.enabled else []
    if not names:
        return ""
    return f"Mình nhớ lần trước bạn có nhắc tới {' và '.join(names)}."
