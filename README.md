# MentalHealthAI — chạy nhanh

> Công cụ nghiên cứu phân tích hội thoại (stress / cảm xúc / mức độ biểu hiện trầm cảm). **Không phải chẩn đoán y khoa.**
> Mô hình chỉ hiểu **tiếng Anh**.

## Yêu cầu
- Windows/Linux/macOS, **Python 3.11**, Git, kết nối mạng (lần đầu).

## Cài đặt
```bash
git clone https://github.com/<TEN_GITHUB>/MentalHealthAI.git
cd MentalHealthAI
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
hf auth login                    # chỉ cần nếu repo mô hình ở chế độ private
python src/download_models.py    # tải 3 mô hình về models/
streamlit run src/app.py         # mở http://localhost:8501
```

## (Tùy chọn) Bật LLM
`pip install anthropic`, đặt biến môi trường `ANTHROPIC_API_KEY`. Không có khóa thì ứng dụng dùng phản hồi theo luật.

## Kiểm thử logic (không cần mô hình)
```bash
pytest tests -q
```

---


Ba mô hình BERT (Stress / Emotion / Depression severity) + lớp an toàn + tầng phản hồi (LLM tùy chọn)
+ giao diện Streamlit. **Công cụ nghiên cứu, không phải chẩn đoán y khoa.**

## Kiến trúc
```
tin nhắn người dùng ─► safety.py (luật, chạy trước tiên, hiểu cả tiếng Việt)
        │                       └─ crisis ─► thông điệp cố định + nguồn hỗ trợ (KHÔNG dùng LLM)
        ▼
 (tùy chọn) llm_client.translate_to_english  ─►  analyze.py
        │        Stress + Depression: N tin gần nhất (cửa sổ trượt, không bị cắt)
        │        Emotion: tin mới nhất, ngưỡng riêng từng cảm xúc
        ▼
 response_generator.py ─► LLM (nếu có API key) ─► dự phòng: phản hồi theo luật
        ▼
 app.py (Streamlit)  /  chatbot.py (terminal)
```

## Cấu trúc thư mục
| Tệp | Vai trò |
|---|---|
| `config.py` | đường dẫn, nhãn, tham số, nguồn hỗ trợ (sửa ở đây) |
| `modeling.py` | nạp mô hình, suy luận theo lô, cửa sổ trượt cho văn bản dài |
| `analyze.py` | `analyze_conversation(user_messages, window)` |
| `safety.py` | phát hiện nguy cơ tự hại (Anh + Việt), thông điệp an toàn |
| `llm_client.py` | sinh phản hồi + dịch Việt→Anh qua Anthropic API (tùy chọn) |
| `response_generator.py` | an toàn → LLM → luật (từ khóa khớp nguyên từ) |
| `clean_data.py` | bỏ trùng/mâu thuẫn, chống rò rỉ train↔test, tách dev |
| `train_classifier.py` | huấn luyện Stress / Depression (dev, chọn epoch tốt nhất, class weight) |
| `train_emotion.py` | huấn luyện GoEmotions multi-label |
| `find_thresholds.py` | chọn ngưỡng Emotion trên **dev** |
| `evaluate_all.py` | đánh giá test: per-class, ma trận nhầm lẫn, QWK/MAE cho Depression |
| `baseline_tfidf.py` | baseline TF-IDF + LogReg/SVM |
| `metrics.py`, `datasets_io.py`, `textutils.py` | tiện ích |

## Cách chạy
```bash
pip install -r requirements.txt
python src/clean_data.py                       # 1. làm sạch + tách dev  -> data/clean/
python src/baseline_tfidf.py                   # 2. baseline (không cần GPU)

# 3. Huấn luyện — nên dùng GPU (Google Colab), CPU rất chậm
python src/train_classifier.py --task stress --epochs 3
python src/train_classifier.py --task depression --epochs 4
python src/train_emotion.py --epochs 3
python src/find_thresholds.py                  # 4. ngưỡng Emotion (dev)

python src/evaluate_all.py --task all          # 5. số liệu + biểu đồ -> reports/
streamlit run src/app.py                       # 6. giao diện
pytest tests -q                                # kiểm thử logic (không cần mô hình)
```
Muốn dùng LLM: `pip install anthropic`, đặt `ANTHROPIC_API_KEY` (và tùy chọn `ANTHROPIC_MODEL`).
Không có khóa thì ứng dụng vẫn chạy bằng phản hồi theo luật.

`config.find_model` ưu tiên `bert_*_v2` (do script train mới tạo), sau đó là mô hình cũ
(`bert_stress_final`, `bert_emotion/checkpoint-5427`, `bert_depression_final`) ở thư mục trong hoặc ngoài.

## Lưu ý 
- Mô hình chỉ hiểu **tiếng Anh**; tiếng Việt cần bật dịch (cần API) — nêu rõ trong hạn chế.
- Nhãn Depression là nhãn bài đăng Reddit, không phải đánh giá lâm sàng.
- Báo cáo **macro-F1 + per-class**, không chỉ accuracy (Depression lệch 73% `minimum`).
- Chỉ chọn epoch/ngưỡng trên **dev**; test chỉ chạm một lần.
- Kiểm tra lại `SUPPORT_RESOURCES` trong `config.py` trước khi demo.
