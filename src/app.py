"""Giao diện Streamlit.   streamlit run src/app.py"""
import json

import pandas as pd
import streamlit as st

import config
import llm_client
import response_generator as rg
import safety
from textutils import looks_vietnamese

st.set_page_config(page_title="MentalHealthAI", page_icon="🧠", layout="wide")

# ------------------------------------------------------------------ session
for key, default in {"messages": [], "analysis_msgs": [], "log": [], "last_result": None,
                     "last_safety": "none", "translated_last": False}.items():
    st.session_state.setdefault(key, default)

LLM_OK = llm_client.is_available()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    use_llm = st.toggle("Sinh phản hồi bằng LLM (API)", value=LLM_OK, disabled=not LLM_OK,
                        help="Cần ANTHROPIC_API_KEY. Tắt thì dùng phản hồi theo luật.")
    translate = st.toggle("Dịch tiếng Việt → Anh trước khi phân tích", value=LLM_OK, disabled=not LLM_OK,
                          help="Ba mô hình BERT chỉ hiểu tiếng Anh.")
    window = st.slider("Số tin nhắn gần nhất dùng để phân tích", 1, 10, 5)
    if not LLM_OK:
        st.caption("LLM đang tắt (chưa có ANTHROPIC_API_KEY): phản hồi theo luật, "
                   "tin nhắn tiếng Việt sẽ KHÔNG được phân tích chính xác.")
    st.divider()
    with st.expander("📞 Nguồn hỗ trợ"):
        for r in config.SUPPORT_RESOURCES:
            st.markdown(f"- {r}")
    if st.session_state.log:
        st.download_button("⬇️ Tải nhật ký phân tích (JSON)",
                           json.dumps(st.session_state.log, ensure_ascii=False, indent=2),
                           file_name="analysis_log.json", mime="application/json")
        st.caption("Nhật ký chứa nội dung hội thoại: chỉ dùng cho nghiên cứu, hãy ẩn danh trước khi chia sẻ.")

# ------------------------------------------------------------------ header
st.title("🧠 MentalHealthAI")
st.caption("Phân tích hội thoại để nhận diện tín hiệu stress, cảm xúc và mức độ biểu hiện liên quan đến trầm cảm.")
st.warning("⚠️ Đây là công cụ nghiên cứu. Kết quả KHÔNG phải chẩn đoán y khoa và có thể sai.")

if st.session_state.last_safety == "crisis":
    st.error("Hệ thống phát hiện nội dung có thể liên quan đến nguy cơ tự hại. "
             "Nếu bạn hoặc ai đó đang gặp nguy hiểm, hãy xem mục **Nguồn hỗ trợ** ở thanh bên và liên hệ ngay.")

chat_col, result_col = st.columns([1.5, 1])

# ------------------------------------------------------------------ chat
with chat_col:
    st.subheader("💬 Hội thoại")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_input = st.chat_input("Nhập nội dung bạn muốn chia sẻ...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        try:
            with st.spinner("Đang phân tích..."):
                import analyze  # nạp mô hình lần đầu có thể mất vài giây
                text_for_models, translated = user_input, False
                if translate and LLM_OK and looks_vietnamese(user_input):
                    try:
                        text_for_models, translated = llm_client.translate_to_english(user_input), True
                    except Exception:
                        pass
                st.session_state.analysis_msgs.append(text_for_models)
                result = analyze.analyze_conversation(st.session_state.analysis_msgs, window=window)
                out = rg.generate_reply(result, user_input, st.session_state.messages, use_llm=use_llm)
        except FileNotFoundError as e:
            st.session_state.messages.pop()
            st.error(f"Chưa tìm thấy mô hình: {e}")
            st.stop()

        st.session_state.last_result = result
        st.session_state.last_safety = out["safety_level"]
        st.session_state.translated_last = translated
        st.session_state.messages.append({"role": "assistant", "content": out["reply"]})
        st.session_state.log.append({
            "turn": len(st.session_state.analysis_msgs), "user": user_input,
            "text_for_models": text_for_models, "translated": translated,
            "stress_probability": round(result["stress_probability"], 4),
            "depression": result["depression"], "emotions": result["emotions"],
            "safety_level": out["safety_level"], "reply_source": out["source"],
        })
        st.rerun()

# ------------------------------------------------------------------ kết quả
with result_col:
    st.subheader("📊 Kết quả phân tích")
    r = st.session_state.last_result
    if r is None:
        st.info("Hãy nhập nội dung ở bên trái để bắt đầu.")
    else:
        last_user = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else ""
        if looks_vietnamese(last_user) and not st.session_state.translated_last:
            st.warning("Tin nhắn là tiếng Việt nhưng chưa được dịch: các mô hình được huấn luyện bằng "
                       "tiếng Anh nên kết quả bên dưới có thể không đáng tin.")
        st.caption(f"Phân tích dựa trên {r['n_messages_used']} tin nhắn gần nhất của bạn.")

        st.markdown("### 😣 Stress")
        (st.error if r["stress"] == config.STRESS_YES else st.success)(
            "Có dấu hiệu stress" if r["stress"] == config.STRESS_YES else "Chưa thấy dấu hiệu stress")
        st.progress(float(r["stress_probability"]), text=f"Xác suất có stress: {r['stress_probability']:.0%}")

        st.divider()
        st.markdown("### 😊 Cảm xúc (tin nhắn mới nhất)")
        if r["emotions"]:
            for emo, p in r["emotions"][:5]:
                st.progress(float(min(p, 1.0)), text=f"{emo}: {p:.0%}")
        else:
            st.info("Không có cảm xúc nào vượt ngưỡng.")

        st.divider()
        st.markdown("### 🧠 Mức độ biểu hiện liên quan trầm cảm")
        level_fn = {"minimum": st.success, "mild": st.warning, "moderate": st.warning, "severe": st.error}
        level_fn.get(r["depression"], st.info)(r["depression"].capitalize())
        st.bar_chart(pd.Series(r["depression_probs"], name="Xác suất"), height=160)
        st.caption("Mô hình còn yếu ở mức mild và severe; chỉ mang tính tham khảo nghiên cứu.")

        if len(st.session_state.log) > 1:
            st.divider()
            st.markdown("### 📈 Xu hướng stress theo lượt")
            st.line_chart(pd.Series([e["stress_probability"] for e in st.session_state.log],
                                    index=[e["turn"] for e in st.session_state.log], name="Xác suất stress"),
                          height=160)

st.divider()
if st.button("🗑️ Xóa cuộc hội thoại", use_container_width=True):
    for k in ("messages", "analysis_msgs", "log"):
        st.session_state[k] = []
    st.session_state.last_result, st.session_state.last_safety = None, "none"
    st.rerun()
