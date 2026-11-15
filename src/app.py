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
import coping
import counseling
import diary
import llm_client
import memory
import pipeline
import privacy
import study
from textutils import detect_lang, reply_language

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
# Biến của MỘT cuộc trò chuyện (được cất đi/khôi phục khi chuyển cuộc trò chuyện). Mọi thứ chỉ nằm trong phiên trình duyệt.
CHAT_VARS = {"sess": pipeline.Session, "log": list, "last_safety": lambda: "none", "summary_cache": dict,
             "summary_seen": lambda: False, "study_done": lambda: None, "participant": lambda: None}
for key, factory in {**CHAT_VARS, "pending": lambda: None, "large_text": lambda: False, "details": lambda: False,
                     "consent": lambda: None, "diary_code": lambda: None, "chats": dict, "chat_seq": lambda: 1,
                     "active": lambda: "c1", "title": lambda: None, "memory": memory.Memory, "remembered": set,
                     "editing": lambda: False}.items():
    if key not in S:
        S[key] = factory()
# Trạng thái đồng ý (nhóm tuổi, cho phép lưu nghiên cứu) áp vào phiên hội thoại ở MỖI lần chạy lại.
S.sess.minor = bool(S.consent and S.consent.get("minor"))
S.sess.research = bool(S.consent and S.consent.get("research") and not S.sess.minor)
# Điều đã nhớ từ các cuộc trước (rỗng nếu người dùng tắt hoặc đã xoá).
S.sess.memory_text = memory.text(S.memory)
S.sess.memory_greeting = memory.greeting(S.memory)


def chat_title(sess, custom=None):
    first = next((m["content"] for m in sess.history if m["role"] == "user"), "")
    return custom or ((first[:28] + "…") if len(first) > 28 else first) or "Cuộc trò chuyện mới"


def remember_current():
    """Ghi nhớ tóm tắt (chỉ nhãn) của cuộc trò chuyện đang mở, một lần cho mỗi cuộc, khi người dùng rời nó."""
    if S.active not in S.remembered and S.sess.profile()["n_turns"] >= 2 and S.memory.enabled:
        memory.remember(S.memory, S.sess.profile(), S.sess.plan)
        S.remembered.add(S.active)


def _fresh_chat():
    S.chat_seq += 1
    S.active = f"c{S.chat_seq}"
    for k, factory in CHAT_VARS.items():
        S[k] = factory()
    S.title, S.pending, S.editing = None, None, False


def _park():
    S.chats[S.active] = {**{k: S[k] for k in CHAT_VARS}, "title": S.title}


def _load(cid):
    snap = S.chats.pop(cid)
    for k in CHAT_VARS:
        S[k] = snap[k]
    S.title, S.active, S.pending, S.editing = snap["title"], cid, None, False


def new_chat():
    if not S.sess.history:                  # cuộc đang mở còn trống thì dùng luôn
        return
    remember_current()
    _park()
    _fresh_chat()


def switch_chat(cid):
    if cid == S.active or cid not in S.chats:
        return
    remember_current()
    if S.sess.history:
        _park()
    _load(cid)


def delete_chat(cid):
    """Xoá hẳn một cuộc trò chuyện (không được ghi nhớ nữa)."""
    if cid != S.active:
        S.chats.pop(cid, None)
        return
    S.remembered.discard(cid)
    if S.chats:
        _load(next(reversed(S.chats)))
    else:
        _fresh_chat()


def transcript_markdown():
    lines = [f"# {chat_title(S.sess, S.title)}", ""]
    for m in S.sess.history:
        lines += [f"**{'Bạn' if m['role'] == 'user' else config.BOT_NAME}:** {m['content']}", ""]
    return "\n".join(lines)


def take_back_app():
    """Gỡ lượt cuối (cả phân tích và nhật ký) để viết lại hoặc sửa."""
    back = pipeline.take_back(S.sess)
    if back and S.log:
        S.log.pop()
    return back


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
.mh-stages {{display:flex; flex-wrap:wrap; gap:.35rem; margin-bottom:.3rem;}}
.mh-stage {{border-radius:99px; padding:.25rem .65rem; font-size:{BASE - 3}px; border:1px solid var(--mh-border);
           color:var(--mh-muted); background:#FAFAFD;}}
.mh-stage.done {{background:var(--mh-ok-bg); color:var(--mh-ok); border-color:#C5E2D5;}}
.mh-stage.current {{background:var(--mh-primary); color:#FFFFFF; border-color:var(--mh-primary); font-weight:600;}}
.mh-mobilebar {{display:none; align-items:center; justify-content:space-between; gap:.6rem; flex-wrap:wrap;
               background:var(--mh-card); border:1px solid var(--mh-border); border-radius:14px; padding:.5rem .8rem;
               margin:0 0 .7rem; font-size:{BASE - 2}px; color:var(--mh-text);}}
.mh-mobilebar b {{font-weight:650;}}
.st-key-actions [data-testid="stHorizontalBlock"], .st-key-chips [data-testid="stHorizontalBlock"] {{flex-wrap:wrap; gap:.4rem;}}
.st-key-actions [data-testid="stColumn"], .st-key-chips [data-testid="stColumn"] {{flex:0 0 auto; width:auto !important; min-width:0;}}
.st-key-actions button, .st-key-chips button {{min-height:38px; padding:.2rem .8rem; border-radius:99px;}}
.st-key-actions button p, .st-key-chips button p {{white-space:nowrap; font-size:{BASE - 3}px;}}
.st-key-chips button p {{white-space:normal;}}
@media (max-width: 900px) {{
  .st-key-livepanel {{display:none;}}
  .mh-mobilebar {{display:flex;}}
  .block-container {{padding-left:1rem; padding-right:1rem;}}
  [data-testid="stChatMessage"] {{max-width:96%;}}
}}
.mh-curchat {{background:var(--mh-primary-soft); border-radius:12px; padding:.5rem .8rem; color:var(--mh-text);
             font-size:{BASE - 1}px; font-weight:600; margin:.2rem 0 .3rem;}}

.mh-status {{display:flex; align-items:center; gap:.55rem; background:var(--mh-lav); border-radius:12px;
            padding:.5rem .85rem; color:#3B2F6B; font-size:{BASE - 2}px; margin-bottom:.7rem;}}
.mh-pulse {{width:9px; height:9px; border-radius:50%; background:var(--mh-lav-strong); animation:mh-pulse 1.2s ease-in-out infinite;}}
@keyframes mh-pulse {{0%,100%{{opacity:.35; transform:scale(.8)}} 50%{{opacity:1; transform:scale(1.1)}}}}
@media (prefers-reduced-motion: reduce) {{ .mh-pulse {{animation:none;}} }}
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


def diary_panel():
    """Nhật ký tâm trạng nhiều ngày: chỉ lưu điểm số, khóa là mã ngẫu nhiên người dùng tự giữ."""
    st.caption("Chỉ lưu điểm số (không có chữ), gắn với một mã ngẫu nhiên bạn tự giữ. Ai biết mã đều xem được các điểm này.")
    if not S.diary_code:
        if st.button("Tạo mã mới", key="diary_new", use_container_width=True):
            S.diary_code = privacy.new_participant_code()
            st.rerun()
        code = st.text_input("Hoặc nhập mã cũ của bạn (dạng P-XXXXXX)", key="diary_enter")
        if code.strip():
            if diary.valid_code(code):
                S.diary_code = code.strip().upper()
                st.rerun()
            else:
                st.error("Mã không đúng dạng P-XXXXXX.")
        return
    st.markdown(f"Mã nhật ký của bạn: **{S.diary_code}** (hãy lưu lại để quay lại sau)")
    labels = [f"{v} · {text}" for v, text in diary.MOOD_LABELS.items()]
    pick = st.radio("Hôm nay tâm trạng của bạn?", labels, index=2, key="diary_mood")
    sleep = st.number_input("Số giờ ngủ đêm qua", min_value=0.0, max_value=24.0, value=7.0, step=0.5, key="diary_sleep")
    stress = st.slider("Mức căng thẳng hôm nay (0-10)", 0, 10, 3, key="diary_stress")
    if st.button("Lưu hôm nay", key="diary_save", use_container_width=True):
        diary.add_entry(S.diary_code, int(pick[0]), float(sleep), int(stress))
        st.rerun()
    entries = diary.load(S.diary_code)
    summ = diary.summary(entries)
    if summ["n"]:
        st.line_chart(pd.DataFrame({"Tâm trạng (1-5)": [e["mood"] for e in entries[-30:]]},
                                   index=[e["date"][5:] for e in entries[-30:]]), height=140, color="#2F6F5E")
        st.caption(f"{summ['n']} ngày gần đây · tâm trạng trung bình {summ['mood_mean']:.1f}/5"
                   + (f" · ngủ trung bình {summ['sleep_mean']:.1f} giờ" if summ["sleep_mean"] is not None else ""))
        if summ["alert"]:
            st.warning(diary.alert_text(summ["low_streak"]))
    if st.button("Xoá nhật ký này", key="diary_delete"):
        diary.delete(S.diary_code)
        S.diary_code = None
        st.rerun()


def study_panel():
    """Bảng hỏi PHQ-9/GAD-7 cho người tham gia nghiên cứu (đã đồng ý, từ 18 tuổi). Hồ sơ lưu KHÔNG kèm nội dung hội thoại."""
    profile = S.sess.profile()
    if S.study_done:
        phq, gad = S.study_done
        for line in study.feedback(phq, gad):
            st.markdown(f"- {line}")
        if study.item9_positive(phq):
            for r in config.support_resources(False):
                st.markdown(f"- {r}")
        st.caption("Cảm ơn bạn đã tham gia. Hồ sơ chỉ chứa mã ngẫu nhiên và điểm, không có nội dung trò chuyện.")
        return
    if profile["n_turns"] < study.MIN_TURNS:
        st.info(f"Hãy trò chuyện thêm một chút (mới {profile['n_turns']}/{study.MIN_TURNS} lượt) rồi quay lại điền bảng hỏi.")
        return
    st.caption("Bảng hỏi giúp nhóm nghiên cứu kiểm chứng công cụ. Không bắt buộc, bạn có thể bỏ qua.")
    if study.INSTRUMENT_VI_STATUS == "DRAFT_NOT_VALIDATED":
        st.warning(study.VI_WARNING)
    labels = [o[1] for o in study.OPTIONS]
    with st.form("study_form"):
        answers = {}
        for inst_id in ("phq9", "gad7"):
            inst = study.INSTRUMENTS[inst_id]
            st.markdown(f"**{inst['name']}**: {inst['stem'][0]}")
            answers[inst_id] = [st.radio(f"{i + 1}. {vi}", labels, index=None, horizontal=True, key=f"study_{inst_id}_{i}")
                                for i, (vi, _en) in enumerate(inst["items"])]
        submitted = st.form_submit_button("Gửi bảng hỏi")
    if submitted:
        if any(v is None for vals in answers.values() for v in vals):
            st.error("Bạn hãy trả lời tất cả các câu.")
            return
        to_int = {label: value for value, label, _ in study.OPTIONS}
        phq, gad = ([to_int[v] for v in answers[i]] for i in ("phq9", "gad7"))
        S.participant = S.participant or privacy.new_participant_code()
        study.save_record(study.build_record(S.participant, profile, phq, gad))
        S.study_done = (phq, gad)
        if study.item9_positive(phq):
            S.last_safety = "crisis"                                    # hiện banner và mở mục hỗ trợ ngay
        st.rerun()


@st.dialog("Bài tập nhỏ", width="large")
def exercises_dialog():
    by_label = {f"{e.icon} {e.name} · {e.minutes} phút": e for e in coping.EXERCISES.values()}
    ex = by_label[st.selectbox("Chọn một bài", list(by_label), key="exercise_pick")]
    st.markdown(f"**{ex.intro}**")
    for i, step in enumerate(ex.steps, 1):
        st.markdown(f"{i}. {step}")
    if any(p_["id"] == ex.id for p_ in S.sess.plan):
        st.caption("✅ Bài này đã nằm trong mục Việc nhỏ của bạn.")
    elif st.button("🌱 Thêm vào Việc nhỏ của mình", key="plan_add_dialog"):
        counseling.add_plan(S.sess.plan, ex.id)
        st.rerun()
    st.caption("Đây là kỹ thuật tự giúp mình phổ biến, không phải điều trị. Nếu khó chịu kéo dài hoặc nặng lên, "
               "hãy trò chuyện với người thân hoặc chuyên gia.")


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
    with st.expander("Vì sao mình nhận xét như vậy?"):
        for line in A.explain(profile):
            st.markdown(f"- {line}")
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
    st.button("💬  Cuộc trò chuyện mới", on_click=new_chat, use_container_width=True, key="new_chat")
    st.markdown(f'<div class="mh-curchat">▸ {chat_title(S.sess, S.title)}</div>', unsafe_allow_html=True)
    for cid, snap in reversed(list(S.chats.items())[-8:]):
        st.button("💬 " + chat_title(snap["sess"], snap["title"]), key=f"chat_{cid}", on_click=switch_chat, args=(cid,),
                  use_container_width=True)
    with st.expander("⚙️ Quản lý cuộc trò chuyện này"):
        new_name = st.text_input("Đổi tên", value=chat_title(S.sess, S.title), key=f"rename_{S.active}", max_chars=40)
        if new_name.strip() and new_name.strip() != chat_title(S.sess, S.title):
            S.title = new_name.strip()
            st.rerun()
        st.download_button("⬇️ Xuất cuộc trò chuyện (.md)", transcript_markdown(), file_name="tro_chuyen.md",
                           mime="text/markdown", use_container_width=True, disabled=not S.sess.history)
        st.button("🗑️ Xoá cuộc trò chuyện này", on_click=delete_chat, args=(S.active,), use_container_width=True,
                  key="delete_chat", help="Xoá hẳn khỏi phiên trình duyệt này; điều trong đó cũng sẽ không được ghi nhớ.")
    st.divider()
    _style_keys = list(counseling.STYLES)
    _style_labels = [counseling.STYLES[k][0] for k in _style_keys]
    _pick = st.radio("Bạn muốn mình đồng hành thế nào?", _style_labels, index=_style_keys.index(S.sess.style),
                     key=f"style_{S.active}")
    S.sess.style = _style_keys[_style_labels.index(_pick)]
    st.caption(counseling.STYLES[S.sess.style][1])
    if st.button("📋  Xem tổng kết", use_container_width=True, key="open_summary",
                 disabled=profile_now["n_turns"] == 0,
                 help="Tóm lại cuộc trò chuyện và gợi ý vài hướng. Trò chuyện càng nhiều, nhận xét càng chính xác."):
        summary_dialog()
    if st.button("🌬️  Thở cùng mình 1 phút", use_container_width=True, key="breathe"):
        breathing_dialog()
    if st.button("🧰  Bài tập nhỏ", use_container_width=True, key="exercises",
                 help="Tiếp đất, viết ra điều lo, một việc nhỏ trong 10 phút..."):
        exercises_dialog()
    st.toggle("Chữ lớn, dễ đọc", key="large_text")
    st.divider()
    n_done = sum(1 for p_ in S.sess.plan if p_["done"])
    with st.expander(f"🌱 Việc nhỏ của mình ({n_done}/{len(S.sess.plan)})" if S.sess.plan else "🌱 Việc nhỏ của mình"):
        if not S.sess.plan:
            st.caption("Chưa có việc nào. Khi bot gợi ý một việc nhỏ và bạn muốn thử, nó sẽ được ghi vào đây; bạn cũng "
                       "có thể tự thêm từ danh sách dưới.")
        for p_ in list(S.sess.plan):
            c1, c2 = st.columns([5, 1])
            ticked = c1.checkbox(p_["name"], value=p_["done"], key=f"plan_{S.active}_{p_['id']}")
            if c2.button("✕", key=f"plan_rm_{S.active}_{p_['id']}", help="Bỏ khỏi danh sách"):
                counseling.remove_plan(S.sess.plan, p_["id"])
                st.rerun()
            if ticked != p_["done"]:
                counseling.toggle_plan(S.sess.plan, p_["id"], ticked)
                st.rerun()
        _labels = {f"{e.icon} {e.name}": e.id for e in coping.EXERCISES.values()}
        _choice = st.selectbox("Thêm một việc", list(_labels), key="plan_pick")
        if st.button("➕ Thêm vào danh sách", key="plan_add", use_container_width=True):
            counseling.add_plan(S.sess.plan, _labels[_choice])
            st.rerun()
    with st.expander("🏁 Kết thúc phiên"):
        st.caption("Chấm lại tâm trạng và nhận lời tổng kết. Bạn vẫn nhắn tiếp được sau đó, phiên sẽ mở lại.")
        _moods = [f"{icon} {label}" for _v, icon, label in A.MOOD_SCALE]
        _mood_pick = st.radio("Bây giờ bạn thấy thế nào?", _moods, index=2, key=f"endmood_{S.active}")
        if st.button("Kết thúc phiên", key="close_session", use_container_width=True,
                     disabled=S.sess.n_user_turns() == 0):
            pipeline.close_session(S.sess, mood=A.MOOD_SCALE[_moods.index(_mood_pick)][0], use_llm=LLM_OK)
            S.summary_cache = {}
            st.rerun()
    with st.expander("🧠 Mình nhớ gì về bạn"):
        S.memory.enabled = st.toggle("Nhớ giữa các cuộc trò chuyện trong phiên này", value=S.memory.enabled, key="memory_on")
        st.caption("Chỉ lưu nhãn tóm tắt (chủ đề, khía cạnh), không lưu nguyên văn; mất khi đóng trang. Bạn xoá được bất cứ lúc nào.")
        _mem = memory.text(S.memory)
        st.markdown(_mem if _mem else "Chưa nhớ gì.")
        if st.button("Xoá những điều đã nhớ", key="memory_clear", disabled=S.memory.is_empty()):
            memory.clear(S.memory)
            S.remembered.clear()
            st.rerun()
    with st.expander("📞 Khi cần hỗ trợ ngay", expanded=S.last_safety == "crisis"):
        for r in config.support_resources(S.sess.minor):
            st.markdown(f"- {r}")
    with st.expander("📔 Nhật ký tâm trạng"):
        diary_panel()
    if S.sess.research:
        with st.expander("📝 Tham gia nghiên cứu (bảng hỏi)"):
            study_panel()
    with st.expander("🔬 Dành cho nhóm nghiên cứu"):
        st.toggle("Hiện số liệu chi tiết", key="details")
        st.caption(("LLM đang bật: trò chuyện bằng LLM, tin nhắn tiếng Việt được dịch trước khi phân tích."
                    if LLM_OK else "Chưa bật LLM (ANTHROPIC_API_KEY hoặc MHAI_LLM=ollama): bot trả lời theo luật; điểm mô hình chỉ tính trên tin "
                    "nhắn tiếng Anh, tin nhắn tiếng Việt vẫn được dùng qua các khía cạnh (giấc ngủ, năng lượng...)."))
        if S.log:
            st.download_button("⬇️ Nhật ký phân tích (JSON)", json.dumps(S.log, ensure_ascii=False, indent=2),
                               file_name="analysis_log.json", mime="application/json", use_container_width=True)
            st.caption("Nội dung chỉ có trong nhật ký khi người dùng đồng ý lưu bản ghi ẩn danh; số điện thoại, email, "
                       "đường dẫn đã được xoá, nhưng tên người và địa chỉ thì không tự xoá được: hãy rà soát trước khi chia sẻ.")

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

def render_consent():
    """Màn hình đồng ý: phải xác nhận mới trò chuyện được. Dưới 18 tuổi: không có lựa chọn lưu nghiên cứu."""
    st.markdown("### Trước khi bắt đầu")
    for point in privacy.CONSENT_POINTS + ([privacy.llm_consent_point(llm_client.backend())] if LLM_OK else []):
        st.markdown(f"- {point}")
    agree = st.checkbox("Mình đã đọc và hiểu những điều trên", key="consent_agree")
    age = st.radio("Bạn bao nhiêu tuổi?", ["Từ 18 tuổi trở lên", "Dưới 18 tuổi"], key="consent_age", horizontal=True)
    minor = age == "Dưới 18 tuổi"
    research = False
    if minor:
        st.info(privacy.MINOR_NOTE)
    else:
        research = st.checkbox(privacy.RESEARCH_OPTION, key="consent_research", value=False)
    if st.button("Bắt đầu trò chuyện", key="consent_go", disabled=not agree, type="primary"):
        S.consent = {"minor": minor, "research": research}
        st.rerun()


if S.consent is None:
    # Chưa đồng ý thì ẩn thanh bên (chưa có gì để dùng) cho màn hình đồng ý gọn hơn
    st.markdown('<style>[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display:none;}</style>',
                unsafe_allow_html=True)
    render_consent()
    st.stop()
if S.sess.minor:
    st.info(privacy.MINOR_NOTE.split(". ")[-1].rstrip(".") + ". Tổng đài 111 (miễn phí, 24/7) luôn sẵn sàng.")

typed = st.chat_input("Bạn đang nghĩ gì, cứ viết ra nhé...")
pending = S.pending
S.pending = None
if typed:
    pending = {"text": typed}

# ------------------------------------------------------------------ khung phân tích (cập nhật trực tiếp)
CONF_VI = {"low": "Thấp", "medium": "Vừa", "high": "Cao"}


def understanding_card(p):
    known = len(p["domains"])
    dots = "".join(f'<span class="mh-dot{" on" if i < known else ""}"></span>' for i in range(len(A.DOMAINS)))
    msg = {"low": "Mình đang theo dõi và cập nhật theo từng tin nhắn. Độ tin cậy còn thấp nên các chỉ số bên dưới chỉ là sơ bộ.",
           "medium": "Độ tin cậy ở mức vừa. Càng trò chuyện, chỉ số càng sát với bạn.",
           "high": "Độ tin cậy cao: chỉ số đã ổn định hơn qua nhiều lượt. Bạn cứ trò chuyện tiếp bình thường."}[p["evidence"]]
    return (f'<div class="mh-card"><h4>Mình đang hiểu bạn đến đâu</h4><div class="mh-dots" aria-label="{known} trên '
            f'{len(A.DOMAINS)} khía cạnh">{dots}</div><div class="mh-small" style="margin-top:.2rem">{known}/{len(A.DOMAINS)}'
            f' khía cạnh · {p["n_turns"]} lượt trò chuyện · độ tin cậy: <b>{CONF_VI[p["evidence"]]}</b>. {msg}</div></div>')


def level_title(p):
    """(tiêu đề, màu) của mức quan tâm hiện tại, ghi rõ khi còn sơ bộ; None nếu chưa có gì để đánh giá."""
    if p["score"] is None:
        return None
    name, _desc, color = A.LEVELS[p["level"]]
    if p["evidence"] == "low":
        return "Đang theo dõi", "#6B7A8F"
    return name + (" (sơ bộ)" if p["evidence"] == "medium" else ""), color


def mobile_bar():
    """Thanh trạng thái gọn trên điện thoại (khung phân tích đầy đủ chỉ hiện trên màn hình rộng)."""
    p, cur = S.sess.profile(), S.sess.stage()
    icon, label = next((i, l) for sid, l, i in counseling.STAGES if sid == cur)
    lv = level_title(p)
    state = f'<span style="color:{lv[1]}"><b>{lv[0]}</b></span> · tin cậy {CONF_VI[p["evidence"]].lower()}' if lv else "Chưa có đánh giá"
    return (f'<div class="mh-mobilebar"><span>{icon} <b>{label}</b></span><span>{state}</span>'
            f'<span class="mh-small" style="margin:0;width:100%">Xem chi tiết: mở menu ☰ → Xem tổng kết.</span></div>')


def live_card(p):
    """Mức quan tâm hiện tại: hiện ngay từ tin nhắn đầu, ghi rõ độ tin cậy; chưa đủ dữ liệu thì chỉ nói 'đang theo dõi'."""
    if p["score"] is None:
        return ('<div class="mh-card"><h4>Mức quan tâm hiện tại</h4><div class="mh-small" style="margin:0">'
                'Chưa có gì để đánh giá. Chỉ số sẽ hiện ngay khi bạn gửi tin nhắn đầu tiên.</div></div>')
    _name, desc, _c = A.LEVELS[p["level"]]
    title, color = level_title(p)
    note = "Chỉ số sơ bộ, có thể đổi nhiều khi bạn kể thêm." if p["evidence"] == "low" else desc
    return (f'<div class="mh-card"><h4>Mức quan tâm hiện tại</h4><div class="mh-big" style="color:{color}">{title}</div>'
            f'{meter(p["score"], color)}<div class="mh-small">{note} Độ tin cậy: <b>{CONF_VI[p["evidence"]]}</b>. '
            'Cập nhật theo từng tin nhắn, không phải chẩn đoán.</div></div>')


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


def score_series():
    """Điểm 'mức quan tâm' sau từng lượt (tính lại tiến dần trên các lượt đã có)."""
    ents, moods = S.sess.entries, S.sess.self_moods
    out = []
    for i in range(1, len(ents) + 1):
        sc = A.build_profile(ents[:i], moods)["score"]
        if sc is not None:
            out.append(round(sc, 3))
    return out


STAGE_NOTE = {
    "opening": "Mình đang làm quen và lắng nghe bạn.",
    "exploring": "Mình đang cố hiểu chuyện gì đang xảy ra với bạn.",
    "planning": "Mình đã hiểu khá rõ, có thể cùng bạn chọn một việc nhỏ (nếu bạn muốn).",
    "acting": "Bạn đã có việc nhỏ để thử; mình sẽ hỏi thăm khi bạn quay lại.",
    "closing": "Phiên đã kết thúc. Bạn vẫn có thể nhắn tiếp bất cứ lúc nào."}


def stage_card():
    cur = S.sess.stage()
    parts = "".join(f'<span class="mh-stage {state}">{icon} {label}</span>'
                    for _sid, label, icon, state in counseling.stage_progress(cur))
    return (f'<div class="mh-card"><h4>Phiên trò chuyện</h4><div class="mh-stages">{parts}</div>'
            f'<div class="mh-small">{STAGE_NOTE[cur]} Các giai đoạn chỉ để định hướng, bạn không bị khoá ở bước nào.</div></div>')


def render_panel(p, status=None):
    """Vẽ toàn bộ khung bên phải. KHÔNG tạo widget (hàm này được gọi nhiều lần trong một lượt chạy để cập nhật trực tiếp)."""
    if status:
        st.markdown(f'<div class="mh-status"><span class="mh-pulse"></span>{status}</div>', unsafe_allow_html=True)
    if S.sess.models_missing:
        st.caption("Chưa tải mô hình phân tích (chạy `python src/download_models.py`). Hiện đánh giá dựa trên "
                   "những gì bạn kể về giấc ngủ, năng lượng, tâm trạng... và tâm trạng bạn tự chấm.")
    st.markdown(stage_card() + live_card(p) + understanding_card(p) + emotion_card(p) + topic_card(p) + domain_card(p),
                unsafe_allow_html=True)
    series = score_series()
    if len(series) >= 2:
        st.caption("DIỄN BIẾN MỨC QUAN TÂM THEO LƯỢT")
        st.line_chart(pd.DataFrame({"Mức quan tâm (0-1)": series}, index=pd.RangeIndex(1, len(series) + 1)),
                      height=150, color="#2F6F5E")
    if S.details and "stress" in p:
        with st.container(border=True):
            st.caption("SỐ LIỆU CHI TIẾT (NGHIÊN CỨU)")
            st.markdown(f"- Stress cả cuộc trò chuyện: **{p['stress']['overall']:.0%}** · gần đây: "
                        f"**{p['stress']['recent']:.0%}** · tỷ lệ câu có stress: **{p['stress']['share']:.0%}**\n"
                        + (f"- Mức nặng nề: **{config.DEPRESSION_VI[p['depression']['label']]}** "
                           f"(điểm {p['depression']['score']:.2f}/3)\n" if "depression" in p else "")
                        + f"- Thành phần điểm: `{p['components']}` → **{p['score']:.2f}**")


chat_col, panel_col = st.columns([1.7, 1], gap="large")
panel_slot = panel_col.container(key="livepanel").empty()      # nơi vẽ lại khung phân tích trực tiếp


def handle(item):
    """Chạy một lượt: hiện tin nhắn người dùng, hiện dần phản hồi, cập nhật nhật ký."""
    text, answered = item["text"], item.get("answered")
    with st.chat_message("user"):
        st.write(text)
    with panel_slot.container():
        render_panel(S.sess.profile(), status="Đang phân tích tin nhắn của bạn…")
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Đang nghe bạn..."):
            turn = pipeline.run_turn(text, S.sess, use_llm=LLM_OK, answered=answered, mood=item.get("mood"),
                                     avoid=item.get("avoid", ""))
        with panel_slot.container():                      # phân tích xong: cập nhật ngay, trước khi bot viết xong
            render_panel(turn.profile or S.sess.profile(), status="Đã cập nhật · bot đang soạn phản hồi…")
        st.write_stream(typewriter(turn.chunks))
    with panel_slot.container():
        render_panel(turn.profile or S.sess.profile())
    S.last_safety = turn.meta["safety_level"]
    entry = {"turn": len(S.log) + 1, "user": text, "analyzed": turn.entry is not None,
             "safety_level": turn.meta["safety_level"], "reply_source": turn.meta["source"]}
    if turn.entry:
        entry["risk_llm"] = turn.entry.get("risk_llm")
    if turn.entry:
        e = turn.entry
        entry.update({k: e.get(k) for k in ("english", "translated", "model_ok", "stress_probability",
                                            "depression_score", "emotions", "topics", "domains")})
    if turn.profile:
        entry["conversation"] = {k: turn.profile.get(k) for k in ("evidence", "score", "level", "coverage")}
    S.log.append(privacy.scrub_entry(entry, keep_text=S.sess.research))     # mặc định KHÔNG lưu nội dung
    st.rerun()


def render_actions():
    """Hành động trên tin nhắn cuối của bot: viết lại, sửa tin cuối của mình, đánh giá, sao chép."""
    idx = len(S.sess.history) - 1
    if S.editing:
        new = st.text_area("Sửa tin nhắn cuối của bạn", value=S.sess.history[-2]["content"], key=f"edit_{S.active}_{idx}")
        c1, c2, _ = st.columns([1, 1, 2])
        if c1.button("Gửi lại", key="edit_send", type="primary", disabled=not new.strip()):
            back = take_back_app()
            S.editing = False
            S.pending = {"text": new.strip(), "mood": back["mood"] if back else None}
            st.rerun()
        if c2.button("Huỷ", key="edit_cancel"):
            S.editing = False
            st.rerun()
        return
    c1, c2, c3, c4, _ = st.columns([1.25, 1.4, 1, 1, 2.2])
    if c1.button("🔄 Viết lại", key="act_regen", help="Cho mình một cách trả lời khác"):
        back = take_back_app()
        if back:
            S.pending = back
        st.rerun()
    if c2.button("✏️ Sửa tin cuối", key="act_edit", help="Sửa tin nhắn cuối của bạn rồi gửi lại"):
        S.editing = True
        st.rerun()
    with c3:
        fb = st.feedback("thumbs", key=f"fb_{S.active}_{idx}")
    if fb is not None:
        S.sess.feedback[idx] = fb
    with c4.popover("📋 Chép"):
        st.code(S.sess.history[-1]["content"], language=None)


# ------------------------------------------------------------------ cột trò chuyện
with chat_col:
    history = S.sess.history
    if history:
        st.markdown(mobile_bar(), unsafe_allow_html=True)
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
    if S.sess.closed:
        st.caption("🏁 Phiên đã kết thúc. Bạn vẫn có thể nhắn tiếp bất cứ lúc nào, phiên sẽ mở lại.")
    if S.sess.can_regenerate():
        with st.container(key="actions"):
            render_actions()
    # Nút trả lời nhanh cho câu hỏi bot vừa hỏi (giúp người ngại gõ, và cho dữ liệu rõ ràng hơn)
    quick_shown = False
    if history and history[-1]["role"] == "assistant" and not S.editing:
        dom = A.asked_domain(history[-1]["content"])
        if dom and dom not in profile_now["domains"]:
            lang = reply_language(next((m["content"] for m in reversed(history) if m["role"] == "user"), ""), history)
            options = A.DOMAINS[dom]["quick"].get(lang, A.DOMAINS[dom]["quick"]["vi"])
            quick_shown = True
            cols = st.columns(len(options))
            for i, (label, pol) in enumerate(options):
                if cols[i].button(label, key=f"quick_{dom}_{i}", use_container_width=True):
                    S.pending = {"text": label, "answered": (dom, pol)}
                    st.rerun()
    if history and history[-1]["role"] == "assistant" and not quick_shown and not S.editing:
        chips = counseling.suggestions(S.sess.stage(), S.sess.style)[:3]
        with st.container(key="chips"):
            cols = st.columns(len(chips))
            for i, (label, text) in enumerate(chips):
                if cols[i].button(label, key=f"chip_{i}"):
                    S.pending = {"text": text}
                    st.rerun()


with panel_slot.container():
    render_panel(S.sess.profile())
