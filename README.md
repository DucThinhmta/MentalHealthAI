# MentalHealthAI 

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
| `assessment.py`, `intents.py`, `coping.py`, `diary.py` | đánh giá cả cuộc trò chuyện; ý định đặc biệt; bài tập CBT; nhật ký tâm trạng |
| `privacy.py`, `study.py`, `analyze_study.py`, `dashboard.py` | quyền riêng tư; bảng hỏi PHQ-9/GAD-7; phân tích kiểm chứng; bảng điều khiển ẩn danh |
| `annotation.py`, `annotate_app.py`, `agreement.py`, `evaluate_gold.py` | gán nhãn tiếng Việt, kappa, tập vàng |
| `evaluate_safety.py`, `evaluate_dialogs.py`, `run_seeds.py` | đo lớp an toàn, đo hội thoại, nhiều seed |
| `pipeline.py` | một lượt hội thoại dùng chung cho web và terminal: an toàn → xã giao → phân tích |
| `smalltalk.py` | nhận diện câu xã giao (chào hỏi, cảm ơn...) để trả lời trực tiếp, không đưa vào phân tích |
| `calibration.py`, `calibrate.py` | hiệu chỉnh xác suất (temperature) và ngưỡng thứ bậc cho Depression, chọn trên **dev** |
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

## Cải thiện mô hình (cần chạy lại trên máy có dữ liệu/GPU)
Các bước dưới đây đã có mã nhưng **chưa có số liệu**: repo không chứa dữ liệu và mô hình, nên phải tự chạy rồi mới biết có cải thiện hay không.

```bash
# A. Không cần huấn luyện lại: hiệu chỉnh xác suất + ngưỡng thứ bậc cho Depression (vài phút trên CPU)
python src/calibrate.py                        # -> models/calibration.json, reports/calibration.json

# B. Huấn luyện lại với mô hình nền và hàm mất mát khác (GPU)
python src/train_classifier.py --task depression --base-model roberta-base --loss focal --weight-power 0.5 --label-smoothing 0.05 --metric qwk --epochs 6 --patience 2
python src/train_classifier.py --task stress --base-model roberta-base --epochs 4 --patience 2
python src/train_emotion.py --base-model roberta-base --pos-weight-cap 10 --epochs 4 --patience 2
python src/find_thresholds.py && python src/calibrate.py && python src/evaluate_all.py --task all
```

- `calibrate.py` xử lý hai điểm yếu thấy rõ trong `reports/`: Stress quá tự tin (gần như chỉ ra 0,99 hoặc 0,01) và Depression không bao giờ chọn `mild` (F1 = 0). Nó in so sánh trước/sau trên tập test; nếu ngưỡng thứ bậc không giúp, xóa khóa `depression_cuts` trong `models/calibration.json`.
- `--base-model xlm-roberta-base` cho mô hình đa ngữ, phân tích trực tiếp tiếng Việt mà không cần bước dịch. Đây là chuyển giao chéo ngôn ngữ từ dữ liệu tiếng Anh, nên cần một tập tiếng Việt có nhãn (dù nhỏ) để đánh giá trước khi kết luận.
- Checkpoint mới ghi `mhai_meta.json` để lúc suy luận dùng đúng tokenizer; checkpoint cũ vẫn chạy như trước.

## Đánh giá trên cả cuộc trò chuyện
Nhận xét không dựa vào một câu mà gom bằng chứng qua nhiều lượt (`assessment.py`):
- điểm mô hình của TỪNG tin nhắn (stress, mức nặng nề, cảm xúc), gộp có trọng số theo độ dài; tin nhắn tiếng Việt chưa dịch không được tính điểm mô hình;
- 8 khía cạnh sinh hoạt bot chủ động hỏi qua trò chuyện (tâm trạng, giấc ngủ, năng lượng, hứng thú, tập trung, ăn uống, người để chia sẻ, thời gian kéo dài), có nút trả lời nhanh;
- tâm trạng người dùng tự chấm khi bắt đầu.
Chỉ khi đủ dữ liệu (số lượt, số từ, số khía cạnh đã biết) mới đưa ra nhận xét chung và hướng giải quyết (nút "Xem tổng kết"). Trọng số trong `config.CONCERN_WEIGHTS` là heuristic, cần kiểm chứng (vd. so với đánh giá của chuyên gia) trước khi dùng để kết luận. Chưa tải mô hình thì ứng dụng vẫn chạy và đánh giá qua các khía cạnh.

## Dữ liệu cảm xúc bổ sung
```bash
python src/fetch_datasets.py                                   # UIT-VSMEC (tiếng Việt), dair-ai/emotion, EmpatheticDialogues
python src/train_emotion.py --extra all --base-model xlm-roberta-base --pos-weight-cap 10 --epochs 4 --patience 2
python src/find_thresholds.py --extra all && python src/evaluate_all.py --task emotion   # báo cáo riêng từng bộ
```
Các bộ này chỉ dùng cho nghiên cứu, không thương mại (xem đầu `fetch_datasets.py`). Chưa tìm được bộ stress/trầm cảm tiếng Việt công khai phù hợp.

## Dữ liệu song ngữ Anh - Việt
Mọi bộ dữ liệu được dịch sang ngôn ngữ còn lại (Anh → Việt, Việt → Anh), giữ nguyên nhãn, để huấn luyện mô hình hiểu trực tiếp tiếng Việt:
```bash
python src/translate_datasets.py --list --estimate        # số câu cần dịch + chi phí nếu dùng Claude
# Cách 1, miễn phí, chạy trên máy:
python src/translator.py --download opus                  # Opus-MT, ~290 MB mỗi chiều (hoặc: --download vinai, ~1,7 GB, dịch tốt hơn)
python src/translate_datasets.py --engine local
# Cách 2, chất lượng cao nhất, có phí (Message Batches, giảm 50%):
python src/translate_datasets.py --engine claude --submit
python src/translate_datasets.py --engine claude --collect --wait
# Huấn luyện song ngữ:
python src/train_classifier.py --task stress --base-model xlm-roberta-base --bilingual --epochs 4 --patience 2
python src/train_emotion.py --base-model xlm-roberta-base --extra all --bilingual --epochs 4 --patience 2
python src/evaluate_all.py --task all                     # có thêm kết quả trên tập test dịch máy sang tiếng Việt
```
Khi chạy ứng dụng mà không có API key, nếu đã tải mô hình dịch (`translator.py --download`), tin nhắn tiếng Việt được dịch sang tiếng Anh trên máy trước khi phân tích. Tập test dịch máy chỉ là thước đo gần đúng cho tiếng Việt; nên kiểm tra tay một mẫu bản dịch và ghi rõ trong báo cáo.

## Hội thoại
- Trải nghiệm như một cuộc trò chuyện bình thường: **không bao giờ bị chốt**. Bot không "kết luận" và ô nhập luôn mở; **khung bên phải cập nhật trực tiếp**
  sau mỗi tin nhắn (kể cả trong lúc bot đang soạn phản hồi), hiện chỉ số ngay từ tin đầu kèm **độ tin cậy** (thấp/vừa/cao) và diễn biến theo lượt. Nút *Xem tổng kết*
  luôn có ở thanh bên nếu người dùng muốn.
- Có `ANTHROPIC_API_KEY`: LLM trả lời theo kiểu trò chuyện tự nhiên, hiện dần từng chữ, nhớ khoảng 20 lượt gần nhất. Tùy chọn `ANTHROPIC_MODEL`, `ANTHROPIC_EFFORT` (mặc định `low`).
- Không có khóa: phản hồi theo luật, cùng ngôn ngữ với người dùng, không lặp câu, trả lời thật thà các câu hỏi về bot, hỏi chẩn đoán, hỏi thuốc, hỏi về bí mật (`intents.py`).
- Bài tập nhỏ có hướng dẫn (`coping.py`: thở hộp, tiếp đất 5-4-3-2-1, một việc nhỏ 10 phút, viết ra điều lo, ghi lại suy nghĩ...), nhật ký tâm trạng nhiều ngày (`diary.py`).

## An toàn và quyền riêng tư
- **Hai tầng phát hiện nguy cơ tự hại:** luật có xử lý phủ định/ẩn dụ (`safety.py`) + phân loại bằng LLM gộp trong lời gọi trích thông tin (chỉ nâng mức, không bao giờ hạ). Khi có nguy cơ luôn trả thông điệp cố định (không dùng LLM), kể cả khi chưa tải mô hình.
- Đo bằng bộ test hồi quy: `python src/evaluate_safety.py --show-errors` (luật: recall 0,978, trước đó 0,451, **trên bộ do nhóm tự viết nên lạc quan**), `--llm` để thêm lớp LLM.
- **Màn hình đồng ý** bắt buộc; hỏi nhóm tuổi (dưới 18: thêm kênh 111, không lưu nghiên cứu); mặc định không lưu nội dung hội thoại; khi lưu thì xoá số điện thoại/email/đường dẫn (`privacy.py`).
- Kênh hỗ trợ: `config.py` (danh sách đã đối chiếu và danh sách cần xác minh); điền liên hệ của trường vào `resources/local_resources.json`.

## Đánh giá hội thoại
```bash
python eval/build_dialog_scenarios.py                 # 100 kịch bản: khủng hoảng, hỏi chẩn đoán/thuốc, nịnh, bị chửi, đa ngôn ngữ...
python src/evaluate_dialogs.py --show-fails           # chế độ luật: 100/100 kiểm tra tự động (trước đó 67/100)
python src/evaluate_dialogs.py --llm --judge          # LLM thật + bộ chấm LLM sơ bộ (tốn API); người phải kiểm lại
```
Kiểm tra tự động chỉ bắt lỗi rõ ràng; **đồng cảm và tự nhiên cần người chấm** (cột `rating_*` trong `reports/dialog_eval.csv`).

## Kiểm chứng khoa học
Chưa có bằng chứng rằng "mức quan tâm" của bot phản ánh tình trạng thật. Quy trình kiểm chứng:
```bash
python src/analyze_study.py --sample-size             # cỡ mẫu cần thu (ρ=0,3: 85 người)
python src/analyze_study.py --demo                    # chạy thử mã trên dữ liệu GIẢ LẬP (không phải kết quả)
# Thu dữ liệu: người dùng từ 18 tuổi bật "lưu bản ghi ẩn danh" -> điền PHQ-9/GAD-7 trong mục "Tham gia nghiên cứu" của app
python src/analyze_study.py --min-evidence medium     # tương quan, AUC, kappa, kiểm chứng từng triệu chứng + khoảng tin cậy
python src/run_seeds.py --task depression --tag focal --seeds 42 43 44 -- --loss focal --weight-power 0.5   # nhiều seed, trung bình ± độ lệch
```
Tài liệu: [`docs/nghien_cuu/PROTOCOL.md`](docs/nghien_cuu/PROTOCOL.md) (đề cương), [`CONSENT_FORM.md`](docs/nghien_cuu/CONSENT_FORM.md) (mẫu phiếu đồng ý),
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) (hiệu năng, hạn chế, đạo đức). **Bản tiếng Việt PHQ-9/GAD-7 trong app là bản nháp chưa kiểm định**: phải thay trước khi thu dữ liệu thật.

## Tập vàng tiếng Việt
```bash
python src/sample_for_annotation.py --n 600           # tạo pool.csv (đã xoá thông tin nhận dạng)
streamlit run src/annotate_app.py                     # MỖI người gán chạy riêng, độc lập, không thấy dự đoán của bot
python src/agreement.py an linh                       # kappa từng trường + danh sách bất đồng; sau khi điền final_*: thêm --build-gold
python src/evaluate_gold.py [--translate]             # recall nguy cơ tự hại, khía cạnh, stress trên tập vàng
```
Hướng dẫn gán nhãn: [`docs/nghien_cuu/ANNOTATION_GUIDE.md`](docs/nghien_cuu/ANNOTATION_GUIDE.md).

## Công cụ cho nhóm nghiên cứu
`MHAI_ADMIN_CODE=... streamlit run src/dashboard.py`: số liệu gộp ẩn danh (che ô nhỏ hơn 5), theo dõi trôi dạt điểm theo tuần; từ chối hiển thị nếu chưa đặt mã.
CI: `.github/workflows/tests.yml` chạy bộ test (không cần torch/anthropic).

## Lưu ý 
- Mô hình chỉ hiểu **tiếng Anh**; tiếng Việt cần bật dịch (cần API) — nêu rõ trong hạn chế.
- Nhãn Depression là nhãn bài đăng Reddit, không phải đánh giá lâm sàng.
