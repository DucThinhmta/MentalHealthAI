"""Cấu trúc của một phiên tư vấn tâm lý: giai đoạn, cách đồng hành, gợi ý trả lời, kết thúc phiên.

Lấy cảm hứng từ cấu trúc của một buổi tư vấn thông thường (đón nhận -> khám phá -> đặt hướng đi -> hành động nhỏ -> kết thúc) và từ cách
các ứng dụng trò chuyện cho người dùng tuỳ chỉnh "cách bạn muốn được hỗ trợ". Đây là KHUNG ĐỊNH HƯỚNG cho bot, không phải liệu pháp; bot không
chẩn đoán và không thay thế chuyên gia.

Nguyên tắc quan trọng: phiên KHÔNG BAO GIỜ bị khoá. Các giai đoạn chỉ quyết định bot nên làm gì; người dùng luôn có thể nói tiếp, đổi cách
đồng hành, hoặc tự bấm kết thúc phiên (và vẫn có thể quay lại nói tiếp sau đó).
"""
import datetime

import coping

STAGES = [("opening", "Mở đầu", "👋"), ("exploring", "Khám phá", "🔍"), ("planning", "Đặt hướng đi", "🧭"),
          ("acting", "Việc nhỏ", "🌱"), ("closing", "Kết thúc", "🏁")]
STAGE_IDS = [s[0] for s in STAGES]
STYLES = {
    "listen": ("Chỉ lắng nghe", "Mình chỉ cần được nghe, đừng vội khuyên."),
    "gentle": ("Lắng nghe và gợi ý nhỏ", "Nghe là chính, thỉnh thoảng gợi ý một việc nhỏ nếu hợp."),
    "solve": ("Cùng tìm hướng giải quyết", "Sau khi hiểu chuyện, cùng mình tìm từng bước cụ thể."),
}
DEFAULT_STYLE = "gentle"
MAX_OFFERS = 3                        # số lần bot chủ động gợi ý bài tập trong một cuộc trò chuyện (không dồn dập)

STAGE_GUIDE = {
    "opening": "Giai đoạn MỞ ĐẦU: đón nhận, tạo cảm giác an toàn. Chưa đưa lời khuyên. Mời người dùng nói điều họ muốn nói, nhẹ nhàng.",
    "exploring": "Giai đoạn KHÁM PHÁ: lắng nghe, phản chiếu, hỏi mở để hiểu chuyện gì xảy ra, từ bao giờ, ảnh hưởng thế nào. Chưa vội giải quyết.",
    "planning": ("Giai đoạn ĐẶT HƯỚNG ĐI: bạn đã hiểu khá rõ. Tóm tắt rất ngắn điều đã nghe rồi đề xuất MỘT việc nhỏ phù hợp để họ chọn "
                 "(không ép); hỏi họ có muốn thử không."),
    "acting": ("Giai đoạn VIỆC NHỎ: họ đã chọn một việc nhỏ. Cùng họ nghĩ cách làm cụ thể (khi nào, ở đâu), dự đoán trở ngại, động viên; "
               "khi họ quay lại hãy hỏi việc đó thế nào."),
    "closing": ("Giai đoạn KẾT THÚC: tóm tắt ngắn những gì đã nói, ghi nhận nỗ lực của họ, nhắc việc nhỏ đã chọn, mời họ chấm lại tâm trạng, "
                "và cho họ biết có thể quay lại bất cứ lúc nào. Họ vẫn có thể nói tiếp nếu muốn."),
}
STYLE_GUIDE = {
    "listen": ("CÁCH ĐỒNG HÀNH: người dùng chỉ muốn được LẮNG NGHE. Không đưa lời khuyên hay bài tập trừ khi họ tự hỏi. "
               "Phản chiếu và ghi nhận nhiều hơn là hỏi; có thể không hỏi gì."),
    "gentle": "CÁCH ĐỒNG HÀNH: lắng nghe là chính; chỉ gợi ý một việc nhỏ khi hợp thời điểm và hỏi họ có muốn thử không.",
    "solve": "CÁCH ĐỒNG HÀNH: người dùng muốn cùng tìm hướng giải quyết. Sau khi hiểu chuyện, chủ động đề xuất từng bước nhỏ, cụ thể.",
}


def stage_for(n_user_turns, evidence, plan_count=0, closed=False, style=DEFAULT_STYLE):
    """Giai đoạn hiện tại của phiên (chỉ để định hướng bot; không bao giờ chặn người dùng)."""
    if closed:
        return "closing"
    if plan_count > 0:                                    # đã có việc nhỏ (kể cả người dùng tự thêm từ đầu) -> hành động
        return "acting"
    if n_user_turns <= 1:
        return "opening"
    if style == "listen":                                 # chỉ muốn được nghe: không tự chuyển sang đặt hướng đi
        return "exploring"
    ready = (evidence != "low" or n_user_turns >= 3) if style == "solve" else (evidence != "low" or n_user_turns >= 5)
    return "planning" if ready else "exploring"


def stage_progress(stage):
    """[(id, nhãn, biểu tượng, 'done'|'current'|'todo')] để vẽ thanh giai đoạn."""
    here = STAGE_IDS.index(stage)
    return [(sid, label, icon, "done" if i < here else "current" if i == here else "todo")
            for i, (sid, label, icon) in enumerate(STAGES)]


def guide_text(stage, style=DEFAULT_STYLE, memory_text="", plan_names=()):
    """Khối hướng dẫn thêm cho LLM: giai đoạn, cách đồng hành, ghi nhớ và việc nhỏ đã chọn."""
    parts = [STAGE_GUIDE[stage], STYLE_GUIDE.get(style, STYLE_GUIDE[DEFAULT_STYLE])]
    if plan_names:
        parts.append("Việc nhỏ người dùng đã chọn: " + "; ".join(plan_names) + ".")
    if memory_text:
        parts.append("Điều bạn nhớ từ các cuộc trò chuyện trước trong phiên này (người dùng cho phép nhớ; chỉ nhắc khi tự nhiên, "
                     "đừng liệt kê): " + memory_text)
    return "\n".join(parts)


_CHIPS = {
    "opening": [("Để mình kể thêm", "Để mình kể thêm một chút nhé"), ("Mình cũng không chắc", "Mình cũng không chắc nữa")],
    "exploring": [("Kể thêm", "Để mình kể thêm một chút"), ("Mình không chắc", "Mình cũng không chắc nữa"),
                  ("Mình hơi mệt khi nói chuyện này", "Mình hơi mệt khi nói chuyện này")],
    "planning": [("Mình muốn thử", "Mình muốn thử việc đó"), ("Gợi ý khác", "Cho mình gợi ý khác nhé"),
                 ("Để mình nghĩ thêm", "Để mình suy nghĩ thêm")],
    "acting": [("Mình đã thử rồi", "Mình đã thử rồi"), ("Khó làm quá", "Mình thấy khó làm quá"), ("Kể thêm", "Để mình kể thêm một chút")],
    "closing": [("Cảm ơn bạn", "Cảm ơn bạn nhé"), ("Mình muốn nói thêm", "Mình muốn nói thêm một chút")],
}


def suggestions(stage, style=DEFAULT_STYLE):
    """Chip gợi ý câu trả lời (người dùng bấm để gửi nhanh), theo giai đoạn và cách đồng hành."""
    chips = list(_CHIPS[stage])
    if style != "listen" and stage == "exploring":
        chips.append(("Cho mình một gợi ý nhỏ", "Cho mình một gợi ý nhỏ nhé"))
    if style == "listen" and stage in ("planning", "acting"):
        chips = [c for c in chips if c[0] != "Gợi ý khác"]
    return chips


# ---------------------------------------------------------------- việc nhỏ (kế hoạch)
def add_plan(plan, exercise_id):
    """Thêm một bài tập vào danh sách việc nhỏ (không trùng). Trả True nếu vừa thêm."""
    if exercise_id not in coping.EXERCISES or any(p["id"] == exercise_id for p in plan):
        return False
    plan.append({"id": exercise_id, "name": coping.EXERCISES[exercise_id].name, "done": False,
                 "added": datetime.date.today().isoformat()})
    return True


def toggle_plan(plan, exercise_id, done=None):
    for p in plan:
        if p["id"] == exercise_id:
            p["done"] = (not p["done"]) if done is None else bool(done)
            return p["done"]
    return None


def remove_plan(plan, exercise_id):
    before = len(plan)
    plan[:] = [p for p in plan if p["id"] != exercise_id]
    return len(plan) < before


def offered_exercise(history):
    """Bài tập mà bot vừa gợi ý ở lượt trả lời gần nhất (id) hoặc None; bài nào được nhắc cuối cùng thì lấy bài đó."""
    last = next((m["content"] for m in reversed(history or []) if m.get("role") == "assistant"), "")
    found = [(last.rfind(e.name), e.id) for e in coping.EXERCISES.values() if e.name in last]
    return max(found)[1] if found else None


def turns_since_offer(history):
    """Số lượt trả lời của bot kể từ lần gần nhất gợi ý bài tập (rất lớn nếu chưa gợi ý lần nào)."""
    bots = [m["content"] for m in history or [] if m.get("role") == "assistant"]
    for i, text in enumerate(reversed(bots)):
        if any(e.name in text for e in coping.EXERCISES.values()):
            return i
    return 10 ** 6


def n_offers(history):
    """Số lượt trả lời của bot đã gợi ý một bài tập trong cuộc trò chuyện này."""
    return sum(1 for m in history or [] if m.get("role") == "assistant" and any(e.name in m["content"] for e in coping.EXERCISES.values()))


# ---------------------------------------------------------------- kết thúc phiên
def mood_delta(moods):
    """(đầu, cuối, chênh lệch) của tâm trạng tự chấm 1-5; None nếu chưa có ít nhất hai lần chấm."""
    return (moods[0], moods[-1], moods[-1] - moods[0]) if len(moods) >= 2 else None


def closing_extras(plan, moods):
    """Phần đuôi của lời kết thúc phiên (Markdown): việc nhỏ đã chọn, tâm trạng trước/sau, và lời mời quay lại."""
    lines = []
    if plan:
        lines.append("**Việc nhỏ của bạn:**")
        lines += [f"- {'✅' if p['done'] else '⬜'} {p['name']}" for p in plan]
    d = mood_delta(moods)
    if d:
        first, last, delta = d
        trend = "nhỉnh hơn" if delta > 0 else "thấp hơn" if delta < 0 else "giữ nguyên"
        lines.append(f"**Tâm trạng:** lúc đầu {first}/5, bây giờ {last}/5 ({trend}).")
    lines.append("Bạn có thể quay lại bất cứ lúc nào, hoặc cứ nói tiếp nếu còn điều muốn chia sẻ. "
                 "Nếu cảm giác nặng nề kéo dài, hãy trò chuyện với người thân hoặc chuyên gia nhé.")
    return "\n\n".join(lines)
