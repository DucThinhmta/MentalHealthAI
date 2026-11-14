"""Giao diện gán nhãn tin nhắn tiếng Việt.   streamlit run src/annotate_app.py

Mỗi người gán chạy riêng, nhập tên/mã của mình và gán ĐỘC LẬP (không bàn bạc, không xem nhãn của người kia). Giao diện CỐ Ý
không hiện dự đoán của bot để không làm lệch người gán. Có thể ngừng và quay lại bất cứ lúc nào: mục đã gán được lưu ngay.
Hướng dẫn gán nhãn: docs/nghien_cuu/ANNOTATION_GUIDE.md
"""
import pandas as pd
import streamlit as st

import annotation
import assessment
import config

st.set_page_config(page_title="Gán nhãn tiếng Việt", page_icon="🏷️", layout="centered")

STRESS_LABELS = {"Có dấu hiệu căng thẳng": "co", "Không có": "khong", "Không rõ": "khong_ro"}
LEVEL_LABELS = {"(không rõ)": "", "Rất ít": "0", "Nhẹ": "1", "Vừa": "2", "Nặng": "3"}
DOMAIN_LABELS = {"Không nhắc tới": "", "Ổn": "ok", "Đáng chú ý": "concern"}
EMOTION_CHOICES = {"(không rõ)": "khong_ro"} | {f"{config.EMOTION_VI[e][1]} ({e})": e for e in config.EMOTION_LABELS}

st.title("🏷️ Gán nhãn tin nhắn tiếng Việt")
name = st.sidebar.text_input("Tên hoặc mã của bạn (viết liền, không dấu)", key="annotator")
pool_path = annotation.ANNOTATION_DIR / "pool.csv"
if not pool_path.exists():
    st.error("Chưa có data/annotation/pool.csv. Hãy chạy: python src/sample_for_annotation.py --n 600")
    st.stop()
if not name.strip():
    st.info("Nhập tên hoặc mã của bạn ở thanh bên để bắt đầu. Hai người gán phải dùng hai tên khác nhau.")
    st.stop()

pool = pd.read_csv(pool_path, dtype=str, keep_default_na=False)
done = annotation.load_annotations(name)
todo = pool[~pool["id"].isin(set(done["id"]))]
st.sidebar.progress(len(done) / max(len(pool), 1), text=f"Đã gán {len(done)}/{len(pool)}")
if todo.empty:
    st.success("Bạn đã gán xong toàn bộ. Cảm ơn bạn!")
    st.stop()

item = todo.iloc[0]
st.markdown("**Tin nhắn cần gán nhãn:**")
st.info(item["text"])
st.caption("Gán dựa trên điều người viết THỂ HIỆN trong tin nhắn này, không suy đoán thêm. Nếu không chắc, chọn 'Không rõ'.")

with st.form(key=f"form_{item['id']}"):
    stress = st.radio("Người viết có dấu hiệu căng thẳng (stress)?", list(STRESS_LABELS), horizontal=True)
    emotion = st.selectbox("Cảm xúc chính", list(EMOTION_CHOICES))
    level = st.radio("Mức nặng nề của nội dung", list(LEVEL_LABELS), horizontal=True)
    st.markdown("**Các khía cạnh sinh hoạt được nhắc tới**")
    cols = st.columns(2)
    domains = {}
    for i, d in enumerate(assessment.DOMAIN_ORDER):
        domains[d] = cols[i % 2].selectbox(f"{assessment.DOMAINS[d]['icon']} {assessment.DOMAINS[d]['name']}",
                                           list(DOMAIN_LABELS), key=f"d_{d}_{item['id']}")
    crisis = st.checkbox("Có dấu hiệu nguy cơ tự hại / không muốn sống (kể cả nói gián tiếp)")
    note = st.text_input("Ghi chú (không bắt buộc)")
    submitted = st.form_submit_button("Lưu và sang mục tiếp theo", type="primary")

if submitted:
    row = {"id": item["id"], "text": item["text"], "source": item.get("source", ""), "annotator": name.strip(),
           "stress": STRESS_LABELS[stress], "emotion": EMOTION_CHOICES[emotion], "level": LEVEL_LABELS[level],
           "crisis": "1" if crisis else "0", "note": note}
    row.update({f"dom_{d}": DOMAIN_LABELS[v] for d, v in domains.items()})
    annotation.save_annotation(name, row)
    st.rerun()
