# Model card: MentalHealthAI

Tài liệu này mô tả hệ thống ở dạng mà người đọc báo cáo, hội đồng đạo đức và người dùng cần biết. Mọi con số lấy từ `reports/`
(chạy lại các lệnh trong README để cập nhật). **Đây là công cụ nghiên cứu, không phải thiết bị hay dịch vụ y tế.**

## Mục đích sử dụng
- **Dự kiến:** nghiên cứu nhận diện tín hiệu căng thẳng, cảm xúc và mức nặng nề của nội dung trong hội thoại; cho người dùng
  (từ 18 tuổi, đã đồng ý) một nơi để nói chuyện, nhận gợi ý tự giúp mình, và tìm tới hỗ trợ thật khi cần.
- **Không dùng để:** chẩn đoán, sàng lọc lâm sàng, ra quyết định về một cá nhân (học tập, việc làm, bảo hiểm), thay thế bác sĩ hay
  chuyên gia tâm lý, hoặc là kênh duy nhất cho người đang khủng hoảng.

## Các thành phần
| Thành phần | Vai trò | Ghi chú |
|---|---|---|
| 3 mô hình BERT (Stress, Emotion, Depression-severity) | điểm cho từng tin nhắn | chỉ hiểu **tiếng Anh**; tiếng Việt phải qua dịch |
| Dịch Việt→Anh (LLM hoặc Opus-MT trên máy) | cầu nối cho tiếng Việt | lỗi dịch cộng dồn vào lỗi mô hình |
| `assessment.py` | gộp điểm từng tin + 8 khía cạnh sinh hoạt + tâm trạng tự chấm thành đánh giá cả cuộc trò chuyện | **trọng số là heuristic, chưa được kiểm chứng** (xem Hạn chế) |
| `safety.py` + lớp LLM | phát hiện nguy cơ tự hại; luôn trả thông điệp cố định, không dùng LLM để trả lời | luật + phân loại LLM; LLM chỉ nâng mức |
| LLM hội thoại (tùy chọn) | giọng trò chuyện tự nhiên | không dùng khi có nguy cơ; bị ràng buộc: không chẩn đoán, không tư vấn thuốc |

## Dữ liệu huấn luyện
| Nhánh | Dữ liệu | Bản chất | Ghi chú giấy phép |
|---|---|---|---|
| Stress | Dreaddit (Turcan & McKeown, 2019) | bài đăng Reddit | xem điều khoản gốc |
| Emotion | GoEmotions (Demszky et al., 2020) | bình luận Reddit, 28 nhãn | xem điều khoản gốc |
| Depression | Depression Severity Dataset (Naseem et al., 2022) | bài đăng Reddit, nhãn 4 mức | nhãn **không phải đánh giá lâm sàng** |
| Bổ sung (tùy chọn) | UIT-VSMEC (Ho et al., 2019), dair-ai/emotion (Saravia et al., 2018), EmpatheticDialogues (Rashkin et al., 2019) | cảm xúc, một phần tiếng Việt | CC BY-NC(-SA) hoặc chỉ dùng nghiên cứu: **không thương mại** |

Hệ quả: dữ liệu là văn bản dài trên mạng xã hội Reddit, còn người dùng chatbot viết câu ngắn, ngôi thứ nhất, nhiều khi bằng tiếng Việt.
Hiệu năng trên hội thoại thật có thể thấp hơn các số dưới đây (lệch miền dữ liệu).

## Kết quả (tập test của bộ dữ liệu gốc, tiếng Anh)
| Nhánh | Chỉ số | Giá trị | Đối chiếu baseline TF-IDF |
|---|---|---|---|
| Stress | accuracy / macro-F1 | 0,794 / 0,793 | LinearSVC 0,733 / 0,733 |
| Emotion (ngưỡng riêng từng nhãn) | micro-F1 / macro-F1 | 0,589 / 0,433 | không có |
| Depression | accuracy / macro-F1 / QWK | 0,714 / 0,365 / 0,456 | LogReg: 0,665 / **0,412** / 0,376 |

Đọc kết quả cho đúng:
- **Depression yếu:** macro-F1 của BERT (0,365) **thấp hơn** baseline TF-IDF + LogReg (0,412); BERT chỉ hơn về QWK. Lớp `mild` có F1 = 0
  và `severe` chỉ 0,198. Đừng trình bày đây là mô hình phát hiện trầm cảm.
- **Emotion:** các nhãn hiếm (grief, pride, relief, embarrassment) có F1 = 0 hoặc rất thấp.
- Các cải tiến trong `calibrate.py` (hiệu chỉnh xác suất, ngưỡng thứ bậc) và các tùy chọn huấn luyện mới (focal loss, `xlm-roberta-base`,
  `--bilingual`, đa seed) **đã có mã nhưng chưa có số liệu**: cần dữ liệu và GPU, xem `src/run_seeds.py`.

## Lớp an toàn
- **Bộ test hồi quy** (`eval/crisis_cases.json`, n = 136, do nhóm phần mềm viết): luật cho recall 0,978 (89/91), precision 1,000.
  Trước khi nâng cấp: recall 0,451. **Đây là số trên bộ do chính người sửa luật viết, nên lạc quan hơn thực tế**; chưa có thẩm định độc lập.
- Bỏ sót đã biết: các cách nói cần ngữ cảnh ("đi gặp ông bà", "đã chọn xong ngày") được giao cho lớp LLM, **chưa đo được** khi chưa có khóa API.
- Để biết độ tin cậy thật cần: (1) chuyên gia gán nhãn một tập tin nhắn thật, (2) `src/evaluate_gold.py` trên tập vàng tiếng Việt.

## Hội thoại
- 100 kịch bản (`eval/dialog_scenarios.json`), chế độ luật: 100/100 qua các kiểm tra tự động (trước khi sửa: 67/100). Các kiểm tra chỉ bắt lỗi rõ ràng
  (chẩn đoán, khuyên thuốc, >1 câu hỏi, lặp câu, sai ngôn ngữ, thiếu số khẩn cấp); **không đo được sự đồng cảm hay tự nhiên**, và bot đã được
  chỉnh trong lúc nhìn các kịch bản này. Phần chất lượng phải do người chấm (`reports/dialog_eval.csv`).
- Chế độ LLM chưa được đánh giá.

## Hạn chế và rủi ro
1. **Đánh giá cả cuộc trò chuyện chưa được kiểm chứng.** Điểm là trung bình có trọng số do nhóm đặt (`config.CONCERN_WEIGHTS`). Quy trình kiểm chứng
   với PHQ-9/GAD-7 đã có (`src/study.py`, `src/analyze_study.py`, `docs/nghien_cuu/PROTOCOL.md`) nhưng **chưa thu dữ liệu**.
2. **Tiếng Việt:** mô hình chỉ hiểu tiếng Anh; bản dịch máy làm sai sắc thái. Tập test dịch máy chỉ là thước đo gần đúng.
3. **Nhãn Reddit ≠ lâm sàng.** Không suy ra chẩn đoán từ bất kỳ đầu ra nào.
4. **Lệch miền:** văn bản dài trên mạng xã hội vs câu ngắn trong chat.
5. **Nhóm người dùng:** chưa đánh giá theo tuổi, giới, vùng miền; người dùng dưới 18 tuổi được bảo vệ thêm (không lưu nghiên cứu, kênh 111) nhưng chưa được nghiên cứu riêng.
6. **LLM** có thể trả lời sai hoặc quá tự tin dù đã có ràng buộc; tin nhắn gửi tới dịch vụ bên ngoài khi bật LLM (đã nêu trong màn hình đồng ý).
7. Kênh hỗ trợ hiển thị đã đối chiếu ngày 2026-11-14 qua nguồn báo chí; **chưa đối chiếu trang chính thức** (xem `config.RESOURCES_TO_VERIFY`).

## Quyền riêng tư và đạo đức
Màn hình đồng ý bắt buộc; dưới 18 tuổi không lưu nghiên cứu; mặc định không lưu nội dung hội thoại; khi lưu thì xoá số điện thoại, email, đường dẫn
(`privacy.py`; không xoá được tên người/địa chỉ viết tự do); bảng hỏi nghiên cứu không lưu nội dung; có cơ chế xoá. Việc dùng với người thật cần
**phê duyệt của hội đồng đạo đức** và quy trình xử lý khi người tham gia có nguy cơ (xem PROTOCOL).

## Trích dẫn
Demszky et al., 2020 (GoEmotions); Turcan & McKeown, 2019 (Dreaddit); Naseem et al., 2022 (Depression Severity); Ho et al., 2019 (UIT-VSMEC);
Saravia et al., 2018 (CARER, dair-ai/emotion); Rashkin et al., 2019 (EmpatheticDialogues); Kroenke et al., 2001 (PHQ-9); Spitzer et al., 2006 (GAD-7);
Guo et al., 2017 (hiệu chỉnh nhiệt độ); Lin et al., 2017 (focal loss); Landis & Koch, 1977 (diễn giải kappa); Hanley & McNeil, 1982 (sai số AUC).
