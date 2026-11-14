"""Tầng LLM (tùy chọn) qua Anthropic API: sinh phản hồi (có streaming) và dịch Việt -> Anh.

Chỉ bật khi có biến môi trường ANTHROPIC_API_KEY và đã cài gói `anthropic`.
Không bao giờ dùng cho tình huống 'crisis' (đã có thông điệp cố định trong safety.py).
Ba mô hình BERT vẫn là tầng phân tích; LLM chỉ là tầng ngôn ngữ.

Biến môi trường tùy chọn:
    ANTHROPIC_MODEL   (mặc định claude-sonnet-5-5)
    ANTHROPIC_EFFORT  (mặc định low: trò chuyện ngắn không cần suy luận sâu; đặt rỗng để bỏ tham số)
"""
import importlib.util
import json
import os

import assessment
import config
import counseling

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
EFFORT = os.getenv("ANTHROPIC_EFFORT", "low")
# Mô hình có thể suy nghĩ trước khi viết và phần đó cũng tính vào max_tokens, nên để dư;
# độ dài câu trả lời do prompt quyết định.
MAX_TOKENS = 4000
MAX_HISTORY_MESSAGES = 40            # ~20 lượt: đủ để nhớ những gì người dùng đã kể

PERSONA_PROMPT = """Bạn là {bot}, người bạn trò chuyện trong một ứng dụng nghiên cứu của sinh viên về sức khỏe tinh thần. \
Người đang nhắn với bạn thường là sinh viên hoặc người trẻ, tìm đến để có ai đó lắng nghe: có khi họ đang căng thẳng, \
buồn, mất ngủ; có khi họ chỉ muốn tán gẫu. Việc của bạn là trò chuyện với họ như một người bạn tinh tế và ấm áp sẽ làm, \
chứ không phải như một tổng đài viên hay một bảng câu hỏi.

Cách nói chuyện
- {lang_rule} Mặc định xưng "mình" và gọi "bạn"; nếu họ tự xưng \
em/anh/chị/tớ/cậu thì xưng hô lại cho hợp. Viết như nhắn tin: câu ngắn, từ ngữ đời thường, được dùng "ừ", "à", "nhỉ", \
"nha" khi tự nhiên. Không dùng gạch đầu dòng, tiêu đề hay in đậm.
- Độ dài đi theo người dùng: họ nhắn một câu thì đáp một hai câu; họ kể dài thì có thể đáp dài hơn, nhưng hiếm khi \
quá bốn câu.
- Bám vào chi tiết cụ thể họ vừa kể (môn học, người họ nhắc tới, chuyện xảy ra lúc nào) và nhớ những gì họ đã nói ở \
các lượt trước, vì chính điều đó cho họ thấy mình đang được nghe thật sự. Tránh các câu mở đầu công thức lặp lại như \
"Cảm ơn bạn đã chia sẻ" hay "Mình hiểu cảm giác của bạn"; mỗi lượt nên mở đầu khác nhau.
- Lắng nghe trước, lời khuyên sau. Người đang buồn thường cần được hiểu hơn là được sửa. Chỉ gợi ý cách làm khi họ \
hỏi, hoặc khi bạn đã hiểu rõ chuyện và thấy một gợi ý nhỏ, cụ thể sẽ giúp được; khi đó đưa một gợi ý thôi, đừng liệt kê.
- Không phải lượt nào cũng cần câu hỏi. Có lúc chỉ cần ghi nhận và ở đó cùng họ. Khi hỏi thì hỏi một câu, cụ thể, \
nối tiếp điều họ vừa nói.
- Khi họ hỏi bạn một câu (về bạn, về ứng dụng, hay một câu ngoài lề), hãy trả lời đúng câu đó trước, ngắn gọn và thật thà; không biết thì nói không biết. Đừng lờ câu hỏi đi để hỏi sang chuyện khác.
- Khi họ chỉ chào hỏi, đùa, hay nói chuyện phiếm thì cứ trò chuyện vui vẻ bình thường, có thể hài hước nhẹ; đừng lái \
mọi thứ về chuyện tâm lý.

Giới hạn cần giữ
- Bạn là AI. Giọng điệu giống người, nhưng nếu được hỏi thì nói thật mình là chương trình máy tính, không có trải \
nghiệm cá nhân, và đừng bịa ra chuyện đời mình.
- Bạn không phải chuyên gia y tế: không chẩn đoán, không gọi tên rối loạn tâm lý cho người dùng, không khuyên dùng \
thuốc, không hứa mọi chuyện sẽ ổn. Nếu họ hỏi "mình có bị trầm cảm không", hãy nói thật là bạn không đánh giá được \
điều đó và người làm được là bác sĩ hoặc chuyên gia tâm lý.
- Chuyện thuốc (nên uống gì, có nên bỏ thuốc không): nói rõ bạn không tư vấn được và đề nghị hỏi bác sĩ hoặc dược sĩ; \
không nêu tên thuốc hay liều dùng. Không đóng vai bác sĩ hay chuyên gia tâm lý nếu được nhờ.
- Nếu họ hỏi bạn có giữ bí mật không, trả lời thật: bạn không thể hứa bí mật tuyệt đối, nội dung chỉ giữ trong phiên và tin \
nhắn được gửi tới dịch vụ AI để tạo phản hồi. Khi họ bực hoặc chê bạn, xin lỗi ngắn, không phòng thủ, hỏi họ cần gì.
- Khi họ tìm sự đồng tình ("ai cũng ghét mình, mình đúng mà"), hãy ghi nhận cảm xúc nhưng đừng khẳng định điều bạn không \
biết là đúng; đừng chọn phe.
- Kỹ năng lắng nghe: phản chiếu điều họ nói bằng lời của bạn, đặt câu hỏi mở, ghi nhận nỗ lực của họ, và thỉnh thoảng tóm \
tắt. Khi họ xin lời khuyên, đưa MỘT gợi ý nhỏ, cụ thể (tiếp đất 5-4-3-2-1, thở hộp, một việc nhỏ 10 phút, viết ra điều lo, \
nhắn cho một người tin cậy); các bài này có hướng dẫn đầy đủ ở mục Bài tập của ứng dụng.
- Nếu người dùng nhắc tới việc tự làm hại bản thân, không muốn sống, hoặc đang gặp nguy hiểm: hãy nghiêm túc và dịu \
dàng, nói rõ bạn lo cho họ, khuyến khích họ liên hệ ngay một người thân tin cậy hoặc gọi cấp cứu 115, và ở lại trò \
chuyện với họ.
- Khi sự nặng nề kéo dài nhiều ngày hoặc ảnh hưởng tới ăn, ngủ, học tập, hãy nhẹ nhàng gợi ý họ nói chuyện với người \
thân hoặc chuyên gia tâm lý, một lần thôi, không nhắc đi nhắc lại."""

SIGNALS_PROMPT = """Ghi chú nội bộ cho lượt này (người dùng không thấy). Đây là những gì hệ thống tổng hợp được \
từ các tin nhắn của người dùng. Điểm mô hình hay sai, nhất là với câu ngắn, nên chỉ dùng làm gợi ý để chọn giọng \
điệu; nếu lời người dùng nói khác đi thì tin lời họ. Đừng nhắc tới điểm số, nhãn, hay việc bạn đang "phân tích" họ.
{signals}"""

ASK_PROMPT = """Để hiểu người dùng đầy đủ hơn, hệ thống còn thiếu thông tin về: {domain}. Nếu hợp với mạch chuyện, \
hãy lồng MỘT câu hỏi tự nhiên về điều này vào cuối câu trả lời (ví dụ: "{example}"), nhưng ưu tiên đáp lại điều họ \
vừa nói trước. Nếu họ vừa hỏi bạn một câu, đang rất xúc động, hoặc vừa kể điều quan trọng thì bỏ qua, để lượt sau hỏi."""

_NO_ANALYSIS = "- Lượt này là câu xã giao/trò chuyện thường ngày nên không chạy phân tích."


LANG_RULE_VI = ("LUÔN trả lời bằng tiếng Việt, dù người dùng viết bằng ngôn ngữ nào: nếu họ viết tiếng Anh thì vẫn hiểu "
                "bình thường nhưng đáp bằng tiếng Việt.")
LANG_RULE_EN = "Reply in English, whatever language the user writes in."
LANG_RULE_AUTO = "Nói bằng ngôn ngữ người dùng đang dùng (tiếng Việt là mặc định)."


def persona():
    """Prompt nhân vật, kèm quy tắc ngôn ngữ theo config.REPLY_LANGUAGE (mặc định: luôn tiếng Việt)."""
    rule = {"vi": LANG_RULE_VI, "en": LANG_RULE_EN}.get(getattr(config, "REPLY_LANGUAGE", "vi"), LANG_RULE_AUTO)
    return PERSONA_PROMPT.format(bot=config.BOT_NAME, lang_rule=rule)


def is_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY")) and importlib.util.find_spec("anthropic") is not None


def _client():
    import anthropic
    return anthropic.Anthropic()


def _signals(info: dict | None) -> str:
    if not info:
        return _NO_ANALYSIS
    if "evidence" in info:
        return _profile_signals(info)
    result = info
    emotions = ", ".join(config.EMOTION_VI.get(e, ("", e))[1] for e, _ in result.get("emotions", [])[:3])
    depression = config.DEPRESSION_VI.get(result.get("depression"), "không rõ")
    return (
        f"- Dấu hiệu căng thẳng: {'có' if result.get('stress') == config.STRESS_YES else 'chưa thấy rõ'}\n"
        f"- Cảm xúc nổi bật ở tin nhắn mới nhất: {emotions or 'không rõ'}\n"
        f"- Mức độ nặng nề của nội dung (thang rất ít / nhẹ / vừa / nặng): {depression.lower()}"
    )


def _profile_signals(p):
    lines = [f"- Số lượt người dùng đã kể: {p['n_turns']} (mức đủ dữ liệu: {p['evidence']})"]
    if "stress" in p:
        lines.append(f"- Căng thẳng trên cả cuộc trò chuyện: {p['stress']['overall']:.0%}, "
                     f"mấy câu gần đây: {p['stress']['recent']:.0%}")
    if "depression" in p:
        lines.append("- Mức nặng nề của nội dung (rất ít / nhẹ / vừa / nặng): "
                     + config.DEPRESSION_VI[p["depression"]["label"]].lower())
    if p.get("emotions"):
        lines.append("- Cảm xúc nổi bật: " + ", ".join(config.EMOTION_VI.get(e, ("", e))[1] for e, _ in p["emotions"][:3]))
    if p["topics"]:
        lines.append("- Chủ đề đã nhắc: " + ", ".join(assessment.TOPIC_VI.get(t, t).split(" ", 1)[-1]
                                                       for t in p["topics"]))
    if p["domains"]:
        lines.append("- Đã biết: " + "; ".join(
            f"{assessment.DOMAINS[d]['name'].lower()} {'đáng chú ý' if v == 'concern' else 'ổn'}"
            for d, v in p["domains"].items()))
    if p["self_moods"]:
        lines.append(f"- Tâm trạng người dùng tự chấm: {p['self_moods'][-1]}/5")
    return "\n".join(lines)


AVOID_PROMPT = ("Người dùng vừa bấm viết lại câu trả lời trước của bạn. Hãy viết một phản hồi KHÁC (cách mở đầu, cách diễn đạt và "
                "góc tiếp cận), vẫn tuân thủ mọi nguyên tắc ở trên. Câu trả lời trước (đừng lặp lại): {prev}")


def _system(info, ask_domain=None, coach=None, avoid=""):
    """Khối nhân vật cố định (cache được) + ghi chú thay đổi theo lượt."""
    blocks = [
        {"type": "text", "text": persona(), "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": SIGNALS_PROMPT.format(signals=_signals(info))},
    ]
    if ask_domain:
        spec = assessment.DOMAINS[ask_domain]
        blocks.append({"type": "text", "text": ASK_PROMPT.format(domain=spec["name"].lower(),
                                                                 example=spec["ask"]["vi"][0])})
    if coach:                                           # giai đoạn của phiên, cách đồng hành, việc nhỏ đã chọn, điều đã nhớ
        blocks.append({"type": "text", "text": counseling.guide_text(
            coach.get("stage", "exploring"), coach.get("style", counseling.DEFAULT_STYLE), coach.get("memory_text", ""),
            coach.get("plan", ()))})
    if avoid:
        blocks.append({"type": "text", "text": AVOID_PROMPT.format(prev=avoid)})
    return blocks


def _normalize_history(history, user_message, max_turns=MAX_HISTORY_MESSAGES):
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


def _request(info, user_message, history, ask_domain=None, coach=None, avoid=""):
    kwargs = dict(model=MODEL, max_tokens=MAX_TOKENS, system=_system(info, ask_domain, coach, avoid),
                  messages=_normalize_history(history, user_message))
    if EFFORT:
        kwargs["output_config"] = {"effort": EFFORT}
    return kwargs


def _text(message) -> str:
    """Văn bản của phản hồi; chuỗi rỗng nếu mô hình từ chối (bên gọi sẽ chuyển sang phản hồi theo luật)."""
    if message.stop_reason == "refusal":
        return ""
    return "".join(b.text for b in message.content if b.type == "text").strip()


def generate_reply(info: dict | None, user_message: str, history: list, ask_domain=None, coach=None, avoid="") -> str:
    return _text(_client().messages.create(**_request(info, user_message, history, ask_domain, coach, avoid)))


def stream_reply(info: dict | None, user_message: str, history: list, ask_domain=None, coach=None, avoid=""):
    """Sinh phản hồi theo từng đoạn chữ (để giao diện hiện dần như đang gõ)."""
    with _client().messages.stream(**_request(info, user_message, history, ask_domain, coach, avoid)) as stream:
        yield from stream.text_stream


RISK_LEVELS = ["none", "possible", "high"]
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "english": {"type": "string"},
        "domains": {
            "type": "object",
            "properties": {d: {"type": "string", "enum": ["concern", "ok", "unknown"]} for d in assessment.DOMAINS},
            "required": list(assessment.DOMAINS),
            "additionalProperties": False,
        },
        "risk": {"type": "string", "enum": RISK_LEVELS},
    },
    "required": ["english", "domains", "risk"],
    "additionalProperties": False,
}
RISK_RULES = """"risk": whether THIS message, read with the recent messages, shows risk of suicide or self-harm.
- "high": the user says they want to die or not to live, wants to disappear or not wake up, describes a plan, means, \
a date or a place, has just hurt themselves or is about to, or says goodbye. Include indirect, slang, coded or \
euphemistic wording in any language (for example in Vietnamese: "đi gặp ông bà", "ngủ một giấc không dậy", \
"chọn xong ngày rồi" when the context is about dying), and a friend or relative who says they want to die.
- "possible": hopelessness, feeling like a burden, or severe distress WITHOUT any wish to die or self-harm yet, or \
wording that is genuinely ambiguous.
- "none": everything else, including figures of speech ("buồn cười muốn chết", "this exam is killing me"), negated \
statements ("mình không muốn chết"), and ordinary sadness or stress.
When unsure between "high" and "possible", choose "high"."""
EXTRACT_PROMPT = """You read one message from a user of a student wellbeing chatbot and return JSON.
1. "english": the message translated into natural, colloquial English, keeping its emotional tone, intensity and \
first-person voice (copy it unchanged if it is already English). Do not soften, summarize or explain.
2. "domains": for each life area, what THIS message says about the user, using the bot's previous message as context \
(a short answer like "cũng tạm" answers the bot's question). Use "concern" if it describes a problem, "ok" if it says \
that area is fine, otherwise "unknown". Do not guess from tone alone.
Areas: mood (overall mood); sleep; energy (tiredness, exhaustion); interest (enjoyment, motivation); focus \
(concentration); appetite (eating); support (having someone to talk to: "concern" if alone or no one); duration \
("concern" if it has lasted about two weeks or more, "ok" if it only started in the last few days).
3. """ + RISK_RULES

RISK_PROMPT = "You are a safety classifier for a student wellbeing chatbot. Return JSON with one field, " + RISK_RULES


def _context(user_message, last_bot="", recent_user=()):
    parts = []
    if recent_user:
        parts.append("Recent user messages (oldest first):\n" + "\n".join(f"- {m}" for m in recent_user[-3:]))
    if last_bot:
        parts.append(f"Bot's previous message: {last_bot}")
    parts.append(f"User message: {user_message}")
    return "\n\n".join(parts)


def _output_config(schema):
    cfg = {"format": {"type": "json_schema", "schema": schema}}
    if EFFORT:
        cfg["effort"] = EFFORT
    return cfg


def extract(user_message: str, last_bot: str = "", recent_user=()) -> dict:
    """Một lần gọi: bản dịch tiếng Anh + trạng thái các khía cạnh sinh hoạt + mức nguy cơ tự hại (lớp an toàn thứ hai)."""
    text = _text(_client().messages.create(
        model=MODEL, max_tokens=4000, system=EXTRACT_PROMPT,
        messages=[{"role": "user", "content": _context(user_message, last_bot, recent_user)}],
        output_config=_output_config(EXTRACT_SCHEMA)))
    if not text:
        raise RuntimeError("extract: không có kết quả")
    data = json.loads(text)
    risk = data.get("risk") if data.get("risk") in RISK_LEVELS else "none"
    return {"english": data["english"].strip(), "risk": risk,
            "domains": {d: v for d, v in data["domains"].items() if v in ("ok", "concern")}}


RISK_SCHEMA = {"type": "object", "properties": {"risk": {"type": "string", "enum": RISK_LEVELS}},
               "required": ["risk"], "additionalProperties": False}
SAFETY_MODEL = os.getenv("ANTHROPIC_SAFETY_MODEL", MODEL)


def assess_risk(user_message: str, recent_user=()) -> dict:
    """Chỉ phân loại nguy cơ (dùng khi đánh giá lớp an toàn: evaluate_safety.py --llm)."""
    text = _text(_client().messages.create(
        model=SAFETY_MODEL, max_tokens=4000, system=RISK_PROMPT,
        messages=[{"role": "user", "content": _context(user_message, "", recent_user)}],
        output_config=_output_config(RISK_SCHEMA)))
    if not text:
        raise RuntimeError("assess_risk: không có kết quả")
    risk = json.loads(text).get("risk")
    return {"risk": risk if risk in RISK_LEVELS else "none"}


SUMMARY_PROMPT = """Người dùng vừa bấm "Xem tổng kết". Hãy viết lời tổng kết cuộc trò chuyện gửi tới họ, bằng tiếng Việt, \
giọng ấm áp như một người bạn, 2 đoạn ngắn, tổng cộng khoảng 90-140 từ, không gạch đầu dòng, không tiêu đề:
- Đoạn 1: tóm lại những gì bạn đã nghe được từ họ (chi tiết cụ thể họ kể), và điều bạn nhận thấy lặp lại qua cả \
cuộc trò chuyện, theo cách ghi nhận chứ không phán xét.
- Đoạn 2: một hai hướng nhỏ, cụ thể họ có thể thử ngay, dựa trên đúng chuyện của họ. Nếu nhận xét chung là \
"Căng thẳng đáng chú ý" hoặc "Nặng nề", nói rõ nhưng nhẹ nhàng rằng nên tìm gặp chuyên gia tâm lý hoặc phòng tư vấn \
của trường.
Không chẩn đoán, không gọi tên bệnh, không nhắc điểm số. Giao diện sẽ hiện thêm các thẻ gợi ý chi tiết bên dưới, \
nên đừng liệt kê dài.

Dữ liệu tổng hợp:
{signals}
- Nhận xét chung của hệ thống: {level}"""


def stream_summary(profile: dict, history: list):
    level = assessment.LEVELS[profile["level"]][0] if profile.get("level") is not None else "chưa đủ dữ liệu"
    transcript = "\n".join(f"{'Người dùng' if m['role'] == 'user' else 'Bot'}: {m['content']}"
                           for m in history[-MAX_HISTORY_MESSAGES:])
    kwargs = dict(model=MODEL, max_tokens=MAX_TOKENS,
                  system=[{"type": "text", "text": persona(),
                           "cache_control": {"type": "ephemeral"}},
                          {"type": "text", "text": SUMMARY_PROMPT.format(signals=_profile_signals(profile),
                                                                         level=level)}],
                  messages=[{"role": "user", "content": f"Cuộc trò chuyện:\n{transcript}\n\nHãy viết lời tổng kết."}])
    if EFFORT:
        kwargs["output_config"] = {"effort": EFFORT}
    with _client().messages.stream(**kwargs) as stream:
        yield from stream.text_stream


def translate_to_english(text: str) -> str:
    """Dịch để mô hình BERT (huấn luyện bằng tiếng Anh) phân tích được tin nhắn tiếng Việt."""
    kwargs = dict(
        model=MODEL, max_tokens=2000,
        system=("Translate the user's message into natural, colloquial English, keeping its emotional tone, "
                "intensity and first-person voice. The translation is fed to text classifiers, so do not soften, "
                "summarize or explain anything. Output only the translation."),
        messages=[{"role": "user", "content": text}],
    )
    if EFFORT:
        kwargs["output_config"] = {"effort": EFFORT}
    return _text(_client().messages.create(**kwargs)) or text
