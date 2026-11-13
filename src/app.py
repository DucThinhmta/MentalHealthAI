"""Giao diện Streamlit.   streamlit run src/app.py   (chạy từ thư mục gốc để nhận .streamlit/config.toml)

Người dùng chỉ thấy phần trò chuyện và "bức tranh chung"; các thiết lập kỹ thuật được tự bật khi có thể
(LLM + dịch khi có ANTHROPIC_API_KEY) và phân tích luôn dùng cả cuộc trò chuyện. Số liệu chi tiết cho nhóm
nghiên cứu nằm trong mục "Dành cho nhóm nghiên cứu" ở thanh bên.
"""
import json
import time

import pandas as pd
import streamlit as st

import assessment as A
import config
import llm_client
import pipeline
from textutils import detect_lang

st.set_page_config(page_title=config.BOT_NAME, page_icon="🌿", layout="wide")

LLM_OK = llm_client.is_available()
BOT_AVATAR = "🌿"
TOPIC_STARTERS = [
    ("📚", "Dạo này mình áp lực chuyện học quá"),
    ("🌙", "Mấy hôm nay mình khó ngủ"),
    ("💭", "Mình thấy hơi cô đơn"),
    ("☀️", "Hôm nay mình có chuyện vui muốn kể"),
]
MOOD_TEXT = {1: "Hôm nay mình thấy rất tệ", 2: "Hôm nay mình thấy không ổn lắm", 3: "Hôm nay mình thấy bình thường",
             4: "Hôm nay mình thấy khá ổn", 5: "Hôm nay mình thấy rất vui"}

# ------------------------------------------------------------------ trạng thái
S = st.session_state
for key, factory in {"sess": pipeline.Session, "log": list, "last_safety": lambda: "none",
                     "pending": lambda: None, "summary_cache": dict, "summary_seen": lambda: False,
                     "large_text": lambda: False, "details": lambda: False}.items():
    if key not in S:
        S[key] = factory()


def reset_conversation():
    S.sess, S.log, S.last_safety, S.pending = pipeline.Session(), [], "none", None
    S.summary_cache, S.summary_seen = {}, False


# ------------------------------------------------------------------ giao diện chung (CSS)
BASE = 19 if S.large_text else 16
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=Lora:wght@600;700&display=swap');
:root {{
  --mh-primary:#2F6F5E; --mh-primary-soft:#E3F0EB; --mh-lav:#EEEAF8; --mh-lav-strong:#6D5BA8;
  --mh-text:#1F2A37; --mh-muted:#4B5563; --mh-border:#E3DFEC; --mh-card:#FFFFFF;
  --mh-ok-bg:#E3F1EA; --mh-ok:#1F5A47; --mh-warn-bg:#FBEDE6; --mh-warn:#8A3B1F; --mh-unk-bg:#EEF0F3; --mh-unk:#46525A;
}}
html, body, .stApp, .stMarkdown, p, li, label, button, input, textarea,
h1, h2, h3, h4, div[data-testid="stMarkdownContainer"] {{font-family:'Be Vietnam Pro', system-ui, sans-serif;}}
.stApp p, .stApp li, .stApp label, .stApp button p {{font-size:{BASE}px;}}
.block-container {{padding-top:3.4rem; padding-bottom:6rem; max-width:1200px;}}
#MainMenu, footer {{visibility:hidden;}}
.stButton button {{min-height:44px; border-radius:12px; transition:background-color .2s, border-color .2s;}}
.stButton button p {{white-space:normal; text-align:left;}}
[class*="st-key-mood_"] button {{min-height:72px; padding:.4rem .2rem;}}
[class*="st-key-mood_"] button p {{text-align:center; white-space:normal; line-height:1.3; font-size:{BASE - 2}px;}}
[class*="st-key-mood_"] button p:first-child {{font-size:{BASE + 8}px;}}
.stButton button:focus-visible {{outline:3px solid var(--mh-lav-strong); outline-offset:2px;}}

.mh-brand {{display:flex; align-items:center; gap:.75rem;}}
.mh-logo {{width:46px; height:46px; border-radius:14px; background:var(--mh-primary); display:grid; place-items:center;
          font-size:1.4rem;}}
.mh-title {{font-family:'Lora', serif; font-size:{BASE + 10}px; font-weight:700; color:var(--mh-text); line-height:1.15;}}
.mh-sub {{color:var(--mh-muted); font-size:{BASE - 1}px;}}
.mh-note {{color:var(--mh-muted); font-size:{BASE - 3}px; margin:.5rem 0 1rem;}}
.mh-crisis {{background:#FBE9E7; border:1px solid #E9B7AE; color:#7A2A1E; border-radius:14px; padding:.85rem 1rem;
            margin-bottom:1rem; font-size:{BASE - 1}px;}}
.mh-welcome {{background:var(--mh-card); border:1px solid var(--mh-border); border-radius:20px; padding:1.5rem 1.5rem 1.2rem;
             margin:.3rem 0 .8rem;}}
.mh-welcome h3 {{font-family:'Lora', serif; margin:0 0 .35rem; font-size:{BASE + 5}px; color:var(--mh-text);}}
.mh-welcome p {{margin:0; color:var(--mh-muted); line-height:1.6;}}
.mh-label {{font-size:{BASE - 3}px; letter-spacing:.05em; text-transform:uppercase; color:var(--mh-muted); font-weight:600;
           margin:.9rem 0 .4rem;}}

[data-testid="stChatMessage"] {{background:var(--mh-card); border:1px solid var(--mh-border); border-radius:18px;
                               padding:.8rem 1rem; margin-bottom:.55rem; max-width:88%;}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
    background:var(--mh-primary); border-color:var(--mh-primary); margin-left:auto; flex-direction:row-reverse;}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {{color:#FFFFFF;}}
[data-testid="stChatMessage"] p {{line-height:1.65;}}
.mh-offer {{background:var(--mh-lav); border:1px solid #D9D1F0; border-radius:14px; padding:.75rem 1rem; color:#3B2F6B;
           margin:.4rem 0;}}

.mh-card {{background:var(--mh-card); border:1px solid var(--mh-border); border-radius:16px; padding:1rem 1.1rem;
          margin-bottom:.75rem; color:var(--mh-text);}}
.mh-card h4 {{margin:0 0 .6rem; font-size:{BASE - 3}px; letter-spacing:.05em; text-transform:uppercase; color:var(--mh-muted);
             font-weight:600;}}
.mh-big {{font-size:{BASE + 2}px; font-weight:650; margin-bottom:.3rem;}}
.mh-small {{font-size:{BASE - 3}px; color:var(--mh-muted); margin-top:.45rem; line-height:1.5;}}
.mh-meter {{height:10px; border-radius:99px; background:#ECE9F2; overflow:hidden; margin-top:.5rem;}}
.mh-meter > span {{display:block; height:100%; border-radius:99px;}}
.mh-dots {{display:flex; gap:.35rem; margin:.15rem 0 .3rem;}}
.mh-dot {{width:22px; height:8px; border-radius:99px; background:#E4E0EE;}}
.mh-dot.on {{background:var(--mh-primary);}}
.mh-chips {{display:flex; flex-wrap:wrap; gap:.4rem;}}
.mh-chip {{border-radius:99px; padding:.3rem .75rem; font-size:{BASE - 2}px; border:1px solid transparent;}}
.mh-chip.pos {{background:var(--mh-ok-bg); color:var(--mh-ok); border-color:#C5E2D5;}}
.mh-chip.neg {{background:var(--mh-warn-bg); color:var(--mh-warn); border-color:#F1D3C4;}}
.mh-chip.neu {{background:var(--mh-unk-bg); color:var(--mh-unk); border-color:#DDE1E5;}}
.mh-row {{display:flex; justify-content:space-between; align-items:center; padding:.42rem 0; border-bottom:1px dashed #ECE9F2;
         font-size:{BASE - 1}px;}}
.mh-row:last-child {{border-bottom:none;}}
.mh-badge {{border-radius:99px; padding:.12rem .6rem; font-size:{BASE - 3}px; font-weight:600;}}
.mh-badge.ok {{background:var(--mh-ok-bg); color:var(--mh-ok);}}
.mh-badge.concern {{background:var(--mh-warn-bg); color:var(--mh-warn);}}
.mh-badge.unknown {{background:var(--mh-unk-bg); color:var(--mh-unk);}}
.mh-tip {{border-left:4px solid var(--mh-primary); background:#FAFAFD; border-radius:10px; padding:.7rem .9rem; margin-bottom:.6rem;}}
.mh-tip b {{display:block; margin-bottom:.2rem;}}

.mh-breath-wrap {{display:grid; place-items:center; padding:1.2rem 0 .6rem;}}
.mh-breath {{width:190px; height:190px; border-radius:50%; background:radial-gradient(circle, #CFE6DD 0%, #9CCBB9 70%);
            display:grid; place-items:center; animation:mh-box 16s ease-in-out infinite; position:relative;}}
.mh-breath span {{position:absolute; font-weight:600; color:#1F4D40; opacity:0; font-size:{BASE + 1}px;}}
.mh-breath span:nth-child(1) {{animation:mh-t1 16s infinite;}}
.mh-breath span:nth-child(2) {{animation:mh-t2 16s infinite;}}
.mh-breath span:nth-child(3) {{animation:mh-t3 16s infinite;}}
.mh-breath span:nth-child(4) {{animation:mh-t4 16s infinite;}}
@keyframes mh-box {{0%{{transform:scale(.62)}} 25%{{transform:scale(1)}} 50%{{transform:scale(1)}} 75%{{transform:scale(.62)}} 100%{{transform:scale(.62)}}}}
@keyframes mh-t1 {{0%,24%{{opacity:1}} 25%,100%{{opacity:0}}}}
@keyframes mh-t2 {{0%,24%{{opacity:0}} 25%,49%{{opacity:1}} 50%,100%{{opacity:0}}}}
@keyframes mh-t3 {{0%,49%{{opacity:0}} 50%,74%{{opacity:1}} 75%,100%{{opacity:0}}}}
@keyframes mh-t4 {{0%,74%{{opacity:0}} 75%,99%{{opacity:1}} 100%{{opacity:0}}}}
@media (prefers-reduced-motion: reduce) {{
  .mh-breath, .mh-breath span {{animation:none;}} .mh-breath span:nth-child(1) {{opacity:1;}}
}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ tiện ích hiển thị
def typewriter(chunks):
    """Hiện dần từng từ. Đoạn dài (phản hồi theo luật/thông điệp cố định) được chia nhỏ; đoạn LLM đã nhỏ sẵn."""
    for chunk in chunks:
        if len(chunk) <= 24:
            yield chunk
            continue
        for i, word in enumerate(chunk.split(" ")):
            yield word if i == 0 else " " + word
            time.sleep(0.012)


def meter(value, color):
    return f'<div class="mh-meter"><span style="width:{max(3, round(value * 100))}%;background:{color}"></span></div>'



# ------------------------------------------------------------------ hộp thoại
@st.dialog("Bài thở hộp 4-4-4-4", width="small")
def breathing_dialog():
    st.markdown('<div class="mh-breath-wrap"><div class="mh-breath"><span>Hít vào</span><span>Giữ hơi</span>'
                '<span>Thở ra</span><span>Nghỉ</span></div></div>', unsafe_allow_html=True)
    st.markdown("Nhìn vòng tròn và thở theo: **hít vào 4 giây** khi vòng to dần, **giữ 4 giây**, **thở ra 4 giây** "
                "khi vòng nhỏ lại, **nghỉ 4 giây**. Làm 4 vòng (khoảng 1 phút). Nếu thấy chóng mặt thì dừng lại "
                "và thở bình thường.")


@st.dialog("Tổng kết cuộc trò chuyện", width="large")
def summary_dialog():
    S.summary_seen = True
    key = len(S.sess.history)                    # chỉ viết lại lời tổng kết khi cuộc trò chuyện có thêm lượt mới
    if key in S.summary_cache:
        profile, narrative = S.summary_cache[key]
        st.markdown(narrative)
    else:
        profile, chunks = pipeline.summary(S.sess, use_llm=LLM_OK)
        narrative = st.write_stream(typewriter(chunks))
        narrative = narrative if isinstance(narrative, str) else "".join(map(str, narrative))
        S.summary_cache = {key: (profile, narrative)}
    render_summary(profile, narrative)


def render_summary(profile, narrative):
    if profile["evidence"] != "low" and profile["level"] is not None:
        name, desc, color = A.LEVELS[profile["level"]]
        st.markdown(f'<div class="mh-card"><h4>Nhận xét chung</h4><div class="mh-big" style="color:{color}">{name}</div>'
                    f'{meter(profile["score"], color)}<div class="mh-small">{desc} Dựa trên {profile["n_turns"]} lượt '
                    f'trò chuyện và {len(profile["domains"])}/{len(A.DOMAINS)} khía cạnh đã biết.</div></div>',
                    unsafe_allow_html=True)
    else:
        missing = [A.DOMAINS[d]["name"].lower() for d in A.DOMAIN_ORDER if d not in profile["domains"]][:4]
        st.info("Mình cần nghe thêm để nhận xét khách quan. Bạn có thể kể thêm về: " + ", ".join(missing) + ".")
    st.markdown('<div class="mh-label">Gợi ý cho bạn</div>', unsafe_allow_html=True)
    st.markdown("".join(f'<div class="mh-tip"><b>{i} {t}</b>{x}</div>' for i, t, x in A.suggestions(profile)),
                unsafe_allow_html=True)
    st.caption("Đây là kết quả của một công cụ nghiên cứu, không phải chẩn đoán y khoa. Nếu bạn có ý nghĩ làm hại "
               "bản thân, hãy gọi 115 hoặc nói ngay với người thân.")
    st.download_button("⬇️ Tải bản tổng kết", A.summary_markdown(profile, narrative), file_name="tong_ket.md",
                       mime="text/markdown", use_container_width=True)


# ------------------------------------------------------------------ thanh bên
profile_now = S.sess.profile()
with st.sidebar:
    st.button("💬  Cuộc trò chuyện mới", on_click=reset_conversation, use_container_width=True, key="new_chat")
    if st.button("📋  Xem tổng kết", use_container_width=True, key="open_summary",
                 disabled=profile_now["n_turns"] == 0,
                 help="Tóm lại cuộc trò chuyện và gợi ý vài hướng. Trò chuyện càng nhiều, nhận xét càng chính xác."):
        summary_dialog()
    if st.button("🌬️  Thở cùng mình 1 phút", use_container_width=True, key="breathe"):
        breathing_dialog()
    st.toggle("Chữ lớn, dễ đọc", key="large_text")
    st.divider()
    with st.expander("📞 Khi cần hỗ trợ ngay", expanded=S.last_safety == "crisis"):
        for r in config.SUPPORT_RESOURCES:
            st.markdown(f"- {r}")
    with st.expander("🔬 Dành cho nhóm nghiên cứu"):
        st.toggle("Hiện số liệu chi tiết", key="details")
        st.caption(("LLM đang bật: trò chuyện bằng LLM, tin nhắn tiếng Việt được dịch trước khi phân tích."
                    if LLM_OK else "Chưa có ANTHROPIC_API_KEY: bot trả lời theo luật; điểm mô hình chỉ tính trên tin "
                    "nhắn tiếng Anh, tin nhắn tiếng Việt vẫn được dùng qua các khía cạnh (giấc ngủ, năng lượng...)."))
        if S.log:
            st.download_button("⬇️ Nhật ký phân tích (JSON)", json.dumps(S.log, ensure_ascii=False, indent=2),
                               file_name="analysis_log.json", mime="application/json", use_container_width=True)
            st.caption("Nhật ký chứa nội dung hội thoại: hãy ẩn danh trước khi chia sẻ.")

# ------------------------------------------------------------------ đầu trang
st.markdown(f"""
<div class="mh-brand"><div class="mh-logo">🌿</div>
<div><div class="mh-title">{config.BOT_NAME}</div>
<div class="mh-sub">Một nơi để bạn nói ra điều đang nghĩ. Mình lắng nghe.</div></div></div>
<div class="mh-note">Công cụ nghiên cứu, không phải chẩn đoán y khoa và không thay thế chuyên gia.</div>
""", unsafe_allow_html=True)
if S.last_safety == "crisis":
    st.markdown('<div class="mh-crisis"><b>Bạn không phải ở một mình với chuyện này.</b> Nội dung vừa rồi có thể liên quan '
                'đến nguy cơ tự hại. Nếu bạn hoặc ai đó đang gặp nguy hiểm, hãy gọi <b>115</b> hoặc <b>113</b> và xem '
                'mục <b>Khi cần hỗ trợ ngay</b> ở thanh bên.</div>', unsafe_allow_html=True)

typed = st.chat_input("Bạn đang nghĩ gì, cứ viết ra nhé...")
pending = S.pending
S.pending = None
if typed:
    pending = {"text": typed}

chat_col, panel_col = st.columns([1.7, 1], gap="large")


def handle(item):
    """Chạy một lượt: hiện tin nhắn người dùng, hiện dần phản hồi, cập nhật nhật ký."""
    text, answered = item["text"], item.get("answered")
    if item.get("mood"):
        S.sess.self_moods.append(item["mood"])
    with st.chat_message("user"):
        st.write(text)
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Đang nghe bạn..."):
            turn = pipeline.run_turn(text, S.sess, use_llm=LLM_OK, answered=answered)
        st.write_stream(typewriter(turn.chunks))
    S.last_safety = turn.meta["safety_level"]
    entry = {"turn": len(S.log) + 1, "user": text, "analyzed": turn.entry is not None,
             "safety_level": turn.meta["safety_level"], "reply_source": turn.meta["source"]}
    if turn.entry:
        e = turn.entry
        entry.update({k: e.get(k) for k in ("english", "translated", "model_ok", "stress_probability",
                                            "depression_score", "emotions", "topics", "domains")})
    if turn.profile:
        entry["conversation"] = {k: turn.profile.get(k) for k in ("evidence", "score", "level", "coverage")}
    S.log.append(entry)
    st.rerun()


# ------------------------------------------------------------------ cột trò chuyện
with chat_col:
    history = S.sess.history
    if not history and not pending:
        st.markdown("""<div class="mh-welcome"><h3>Chào bạn 👋</h3>
<p>Mình ở đây để lắng nghe, không phán xét. Bạn có thể kể bất cứ điều gì: chuyện học, giấc ngủ, chuyện buồn vui
trong ngày. Trò chuyện càng nhiều, mình càng hiểu bạn và gợi ý được sát hơn.</p></div>""", unsafe_allow_html=True)
        st.markdown('<div class="mh-label">Hôm nay bạn thấy thế nào?</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for col, (v, icon, label) in zip(cols, A.MOOD_SCALE):
            if col.button(f"{icon}\n\n{label}", key=f"mood_{v}", use_container_width=True):
                answered = ("mood", "ok") if v >= 4 else ("mood", "concern") if v <= 2 else None
                S.pending = {"text": MOOD_TEXT[v], "answered": answered, "mood": v}
                st.rerun()
        st.markdown('<div class="mh-label">Hoặc bắt đầu bằng một chủ đề</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (icon, text) in enumerate(TOPIC_STARTERS):
            if cols[i % 2].button(f"{icon}  {text}", key=f"sugg_{i}", use_container_width=True):
                S.pending = {"text": text}
                st.rerun()
    for m in history:
        with st.chat_message(m["role"], avatar=BOT_AVATAR if m["role"] == "assistant" else None):
            st.write(m["content"])
    if pending:
        handle(pending)
    # Nút trả lời nhanh cho câu hỏi bot vừa hỏi (giúp người ngại gõ, và cho dữ liệu rõ ràng hơn)
    if history and history[-1]["role"] == "assistant":
        dom = A.asked_domain(history[-1]["content"])
        if dom and dom not in profile_now["domains"]:
            lang = detect_lang(next((m["content"] for m in reversed(history) if m["role"] == "user"), ""))
            options = A.DOMAINS[dom]["quick"].get(lang, A.DOMAINS[dom]["quick"]["vi"])
            cols = st.columns(len(options))
            for i, (label, pol) in enumerate(options):
                if cols[i].button(label, key=f"quick_{dom}_{i}", use_container_width=True):
                    S.pending = {"text": label, "answered": (dom, pol)}
                    st.rerun()
        if profile_now["evidence"] == "high" and not S.summary_seen:
            st.markdown('<div class="mh-offer">Mình đã hiểu bạn khá rõ rồi. Bạn có muốn xem tổng kết và vài gợi ý '
                        'không?</div>', unsafe_allow_html=True)
            if st.button("📋  Xem tổng kết", key="offer_summary"):
                summary_dialog()


# ------------------------------------------------------------------ cột bức tranh chung
def understanding_card(p):
    known = len(p["domains"])
    dots = "".join(f'<span class="mh-dot{" on" if i < known else ""}"></span>' for i in range(len(A.DOMAINS)))
    msg = {"low": "Mình cần nghe thêm vài lượt nữa trước khi nhận xét, để không vội kết luận từ một câu nói.",
           "medium": "Mình đã bắt đầu hiểu bạn. Kể thêm một chút nữa để nhận xét chính xác hơn nhé.",
           "high": "Mình đã có đủ thông tin để tổng kết khách quan."}[p["evidence"]]
    return (f'<div class="mh-card"><h4>Mình đã hiểu bạn</h4><div class="mh-dots" aria-label="{known} trên '
            f'{len(A.DOMAINS)} khía cạnh">{dots}</div><div class="mh-small" style="margin-top:.2rem">{known}/{len(A.DOMAINS)}'
            f' khía cạnh · {p["n_turns"]} lượt trò chuyện. {msg}</div></div>')


def overall_card(p):
    if p["evidence"] == "low" or p["level"] is None:
        return ""
    name, desc, color = A.LEVELS[p["level"]]
    return (f'<div class="mh-card"><h4>Bức tranh chung</h4><div class="mh-big" style="color:{color}">{name}</div>'
            f'{meter(p["score"], color)}<div class="mh-small">{desc} Tổng hợp từ cả cuộc trò chuyện, không phải '
            'chẩn đoán.</div></div>')


def emotion_card(p):
    if not p.get("emotions"):
        return ""
    chips = []
    for emo, _ in p["emotions"][:5]:
        icon, name = config.EMOTION_VI.get(emo, ("", emo))
        tone = "pos" if emo in config.POSITIVE_EMOTIONS else "neg" if emo in config.NEGATIVE_EMOTIONS else "neu"
        chips.append(f'<span class="mh-chip {tone}">{icon} {name}</span>')
    return f'<div class="mh-card"><h4>Cảm xúc nổi bật qua cuộc trò chuyện</h4><div class="mh-chips">{"".join(chips)}</div></div>'


def topic_card(p):
    if not p["topics"]:
        return ""
    chips = "".join(f'<span class="mh-chip neu">{A.TOPIC_VI.get(t, t)}</span>' for t in p["topics"][:6])
    return f'<div class="mh-card"><h4>Điều bạn nhắc tới</h4><div class="mh-chips">{chips}</div></div>'


def domain_card(p):
    rows = []
    for d in A.DOMAIN_ORDER:
        v = p["domains"].get(d, "unknown")
        if d == "duration":
            text = {"concern": "Đã kéo dài", "ok": "Mới gần đây", "unknown": "Chưa rõ"}[v]
        else:
            text = {"concern": "Đáng chú ý", "ok": "Ổn", "unknown": "Chưa rõ"}[v]
        rows.append(f'<div class="mh-row"><span>{A.DOMAINS[d]["icon"]} {A.DOMAINS[d]["name"]}</span>'
                    f'<span class="mh-badge {v}">{text}</span></div>')
    return f'<div class="mh-card"><h4>Các khía cạnh sinh hoạt</h4>{"".join(rows)}</div>'


with panel_col:
    p = profile_now
    if S.sess.models_missing:
        st.caption("Chưa tải mô hình phân tích (chạy `python src/download_models.py`). Hiện đánh giá dựa trên "
                   "những gì bạn kể về giấc ngủ, năng lượng, tâm trạng... và tâm trạng bạn tự chấm.")
    st.markdown(understanding_card(p) + overall_card(p) + emotion_card(p) + topic_card(p) + domain_card(p),
                unsafe_allow_html=True)
    if S.details and "stress" in p:
        with st.container(border=True):
            st.caption("SỐ LIỆU CHI TIẾT (NGHIÊN CỨU)")
            st.markdown(f"- Stress cả cuộc trò chuyện: **{p['stress']['overall']:.0%}** · gần đây: "
                        f"**{p['stress']['recent']:.0%}** · tỷ lệ câu có stress: **{p['stress']['share']:.0%}**\n"
                        + (f"- Mức nặng nề: **{config.DEPRESSION_VI[p['depression']['label']]}** "
                           f"(điểm {p['depression']['score']:.2f}/3)\n" if "depression" in p else "")
                        + f"- Thành phần điểm: `{p['components']}` → **{p['score']:.2f}**")
            trend = pd.DataFrame({"Căng thẳng": p["stress"]["series"]})
            if "depression" in p and len(p["depression"]["series"]) == len(trend):
                trend["Mức nặng nề (0-1)"] = [s / 3 for s in p["depression"]["series"]]
            trend.index = range(1, len(trend) + 1)
            st.line_chart(trend, height=170, color=["#D9694A", "#2F6F5E"][:trend.shape[1]])
