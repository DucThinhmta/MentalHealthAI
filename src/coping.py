"""Thư viện bài tập nhỏ tự giúp mình (kỹ thuật CBT/DBT/hoạt hóa hành vi phổ biến, không phải điều trị).

Mỗi bài có: tên, thời lượng, các bước, và những chủ đề/khía cạnh mà nó hợp. `pick` chọn bài hợp nhất với những gì người
dùng đã kể và tránh gợi ý lại bài đã đưa. Nội dung viết lại bằng lời của nhóm dựa trên các kỹ thuật được mô tả rộng rãi
(thở có nhịp, tiếp đất 5-4-3-2-1, hoạt hóa hành vi, ghi lại suy nghĩ, thư giãn trước giờ ngủ, tự thương mình). Chúng mang
tính hỗ trợ chung; nếu khó chịu kéo dài hoặc nặng lên, người dùng nên gặp chuyên gia.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Exercise:
    id: str
    name: str
    icon: str
    minutes: int
    intro: str
    steps: tuple
    good_for: frozenset = field(default_factory=frozenset)     # chủ đề (assessment.TOPIC_VI) hoặc khía cạnh (assessment.DOMAINS)
    en_name: str = ""


EXERCISES = {e.id: e for e in [
    Exercise("breathing", "Thở hộp 4-4-4-4", "🌬️", 1,
             "Nhịp thở đều giúp cơ thể bình tĩnh lại.",
             ("Hít vào bằng mũi trong 4 giây.", "Giữ hơi 4 giây.", "Thở ra chậm bằng miệng trong 4 giây.",
              "Nghỉ 4 giây rồi lặp lại 4 vòng. Nếu thấy chóng mặt thì dừng và thở bình thường."),
             frozenset({"anxiety", "pressure", "sleep", "energy"}), "Box breathing"),
    Exercise("grounding", "Tiếp đất 5-4-3-2-1", "🖐️", 3,
             "Kéo sự chú ý về hiện tại khi đầu óc đang cuốn đi.",
             ("Nhìn quanh và gọi tên 5 thứ bạn thấy.", "Chạm vào 4 thứ và để ý cảm giác (vải, mặt bàn, nước...).",
              "Lắng nghe 3 âm thanh.", "Tìm 2 mùi bạn ngửi được.", "Nêu 1 vị bạn cảm nhận được, rồi hít một hơi chậm."),
             frozenset({"anxiety", "pressure", "focus"}), "5-4-3-2-1 grounding"),
    Exercise("activation", "Một việc nhỏ trong 10 phút", "🎯", 10,
             "Khi chẳng muốn làm gì, hành động nhỏ thường kéo tâm trạng lên trước cả cảm hứng.",
             ("Chọn MỘT việc rất nhỏ từng làm bạn thấy dễ chịu (nghe một bài hát, đi bộ quanh nhà, gọi một người bạn, "
              "tắm nước ấm).", "Hẹn giờ 10 phút và chỉ cần bắt đầu, không cần làm hoàn hảo.",
              "Xong thì ghi lại một dòng: tâm trạng trước và sau (0-10)."),
             frozenset({"sad", "interest", "energy", "mood"}), "Behavioural activation"),
    Exercise("worry_dump", "Viết ra điều lo", "📝", 10,
             "Đưa nỗi lo ra khỏi đầu để đầu óc đỡ phải ôm.",
             ("Đặt hẹn 10 phút, cầm giấy bút (hoặc ghi chú trên điện thoại).", "Viết hết mọi điều đang lo, không cần sắp xếp.",
              "Khoanh những việc bạn có thể làm gì đó, ghi bước nhỏ đầu tiên cho từng việc.",
              "Gạch những thứ ngoài tầm kiểm soát, và gấp giấy lại: hẹn mai xem tiếp."),
             frozenset({"anxiety", "sleep", "study", "work", "pressure", "focus"}), "Worry time"),
    Exercise("thought_record", "Ghi lại suy nghĩ (3 cột)", "🧩", 10,
             "Nhìn lại một suy nghĩ khiến mình khó chịu, xem còn cách nhìn nào khác không.",
             ("Cột 1: chuyện gì vừa xảy ra, bạn thấy gì, ở đâu.", "Cột 2: suy nghĩ chạy qua đầu và cảm xúc (chấm 0-10).",
              "Cột 3: có bằng chứng nào cho thấy suy nghĩ ấy chưa hoàn toàn đúng? Nếu bạn thân của bạn nghĩ như vậy, bạn sẽ nói "
              "gì với họ?", "Viết lại một cách nhìn cân bằng hơn và chấm lại cảm xúc."),
             frozenset({"anxiety", "sad", "relationship", "pressure", "mood", "study"}), "Thought record"),
    Exercise("wind_down", "Thư giãn trước giờ ngủ", "🌙", 30,
             "Báo cho cơ thể biết sắp đến giờ nghỉ.",
             ("Đặt giờ đi ngủ và giờ dậy gần như cố định, kể cả cuối tuần.", "Cất điện thoại và tắt màn hình 30 phút trước khi ngủ.",
              "Làm việc nhẹ nhàng: tắm nước ấm, giãn cơ, đọc vài trang sách giấy.",
              "Nếu nằm 20 phút chưa ngủ được, ngồi dậy làm việc nhẹ dưới ánh sáng dịu rồi quay lại giường."),
             frozenset({"sleep", "energy"}), "Wind-down routine"),
    Exercise("self_compassion", "Nói với mình như với bạn thân", "🤍", 5,
             "Khi tự trách, thử đổi giọng.",
             ("Nhớ lại điều bạn đang tự trách mình.", "Hình dung một người bạn thân gặp chuyện y hệt: bạn sẽ nói gì với họ?",
              "Nói câu đó với chính mình, thành tiếng hoặc viết ra.", "Thêm một câu: 'Ai gặp chuyện này cũng thấy khó, "
              "mình đang làm hết sức có thể.'"),
             frozenset({"sad", "study", "pressure", "mood", "work"}), "Self-compassion"),
    Exercise("reach_out", "Nhắn cho một người tin cậy", "🤝", 5,
             "Có người cùng gánh thường nhẹ hơn nhiều.",
             ("Chọn MỘT người bạn tin (bạn, người nhà, thầy cô, cố vấn học tập).",
              "Không cần kể hết: chỉ cần 'Dạo này mình hơi mệt, bạn rảnh nói chuyện với mình chút được không?'",
              "Nếu chưa dám nói, thử hẹn đi cà phê hoặc đi bộ cùng, chuyện tự nhiên sẽ đến."),
             frozenset({"lonely", "support", "relationship", "sad"}), "Reach out"),
]}
_ORDER = list(EXERCISES)


def pick(topics=(), domains=None, used=()):
    """Bài hợp nhất với chủ đề/khía cạnh người dùng đã kể (domains: {tên: 'concern'|'ok'}); bỏ các bài trong `used`."""
    wanted = set(topics) | {d for d, v in (domains or {}).items() if v == "concern"}
    candidates = [e for e in EXERCISES.values() if e.id not in used] or list(EXERCISES.values())
    # nhiều chủ đề trùng nhất thắng; hòa thì ưu tiên bài chuyên biệt hơn (ít chủ đề hơn), rồi theo thứ tự khai báo
    best = max(candidates, key=lambda e: (len(e.good_for & wanted), -len(e.good_for), -_ORDER.index(e.id)))
    return best if best.good_for & wanted else (next((e for e in candidates if e.id == "breathing"), candidates[0]))


def used_in(history):
    """Các bài đã được bot nhắc tên trong hội thoại."""
    said = " ".join(m["content"] for m in history or [] if m.get("role") == "assistant")
    return {e.id for e in EXERCISES.values() if e.name in said}


# Hai bước đầu bằng tiếng Anh cho từng bài (để trả lời người dùng tiếng Anh không chen tiếng Việt).
EN_STEPS = {
    "breathing": ("Breathe in through your nose for 4 seconds, hold for 4.", "Breathe out slowly for 4, rest for 4, and repeat 4 rounds."),
    "grounding": ("Name 5 things you can see and touch 4 things.", "Then notice 3 sounds, 2 smells and 1 taste."),
    "activation": ("Pick ONE very small thing that usually feels a bit better (a song, a short walk, a call).",
                   "Set a 10-minute timer and just start; it doesn't need to be perfect."),
    "worry_dump": ("Set 10 minutes and write down everything you're worried about, unsorted.",
                   "Circle what you can act on and write one small first step for each."),
    "thought_record": ("Write what happened and the thought that went through your mind.",
                       "Ask what a good friend would say about that thought, then write a more balanced version."),
    "wind_down": ("Keep bed and wake-up times roughly fixed and put screens away 30 minutes before sleep.",
                  "Do something gentle: a warm shower, stretching, a few pages of a paper book."),
    "self_compassion": ("Picture a close friend in exactly your situation and think of what you'd tell them.",
                        "Say that to yourself, out loud or in writing."),
    "reach_out": ("Pick ONE person you trust (a friend, relative, teacher).",
                  "You don't have to tell everything; \"I've been feeling off lately, can we talk?\" is enough."),
}


def brief(ex: Exercise, lang="vi", max_steps=2):
    """Đoạn ngắn để bot đưa ra trong trò chuyện (đầy đủ hơn nằm ở hộp thoại Bài tập)."""
    if lang == "en":
        steps = " ".join(f"({i}) {s}" for i, s in enumerate(EN_STEPS[ex.id][:max_steps], 1))
        return (f"One small thing you could try is \"{ex.en_name or ex.name}\" (about {ex.minutes} min): {steps} "
                "The Exercises button in the sidebar has the full steps.")
    steps = " ".join(f"({i}) {s.rstrip('.')}." for i, s in enumerate(ex.steps[:max_steps], 1))
    return (f"Một việc nhỏ bạn có thể thử là \"{ex.name}\" (khoảng {ex.minutes} phút): {steps} "
            "Trong mục Bài tập ở thanh bên có hướng dẫn đầy đủ.")
