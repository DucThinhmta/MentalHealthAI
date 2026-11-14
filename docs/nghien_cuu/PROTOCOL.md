# Đề cương kiểm chứng: điểm của chatbot có tương ứng với thang đo chuẩn không?

> **MẪU để nhóm hoàn thiện.** Phải được hội đồng đạo đức của cơ sở phê duyệt trước khi thu dữ liệu từ người thật. Những chỗ đánh dấu
> `[ĐIỀN]` là thông tin nhóm cần tự bổ sung. Công cụ phần mềm đi kèm: `src/study.py`, `src/analyze_study.py`.

## 1. Câu hỏi và giả thuyết
Công cụ gộp tín hiệu từ nhiều tin nhắn thành một "mức quan tâm" (`assessment.py`). Chưa có bằng chứng rằng mức này phản ánh tình trạng thật của người dùng.

- **Câu hỏi chính:** điểm của bot có tương quan với tổng điểm PHQ-9 và GAD-7 của cùng người không?
- **H1 (chính):** Spearman ρ giữa điểm bot và PHQ-9 ≥ 0,30.
- **H2:** AUC của điểm bot khi phân biệt PHQ-9 ≥ 10 đạt ≥ 0,70 (cận dưới khoảng tin cậy 95% > 0,60).
- **H3:** các khía cạnh bot hỏi (giấc ngủ, năng lượng, ăn uống, hứng thú, tâm trạng, tập trung) khớp với câu PHQ-9 tương ứng (κ ≥ 0,40).
- Kết quả **không ủng hộ** giả thuyết vẫn là kết quả đáng báo cáo (lúc đó không được dùng điểm bot như một thước đo).

## 2. Thiết kế
Cắt ngang, một phiên. Người tham gia trò chuyện tự do với bot (tối thiểu 5 lượt), sau đó điền PHQ-9 và GAD-7. Bot **không** hiện điểm của thang đo
trước khi người dùng trả lời, và người điền thang không biết điểm bot.

## 3. Người tham gia
- Sinh viên/người trưởng thành **từ 18 tuổi** `[ĐIỀN: cơ sở, cách tuyển, tiêu chí loại trừ]`. Người dưới 18 tuổi **không** tham gia (ứng dụng đã chặn lưu hồ sơ).
- Mẫu thuận tiện: nêu rõ hạn chế về tính đại diện.
- **Cỡ mẫu:** `python src/analyze_study.py --sample-size`: ρ = 0,30 cần n = 85; AUC kỳ vọng 0,75 với khoảng tin cậy ±0,10 và 30% dương tính cần n ≈ 125;
  ±0,07 cần n ≈ 218. Nên tuyển nhiều hơn 15-20% để bù người bỏ dở. `[ĐIỀN: cỡ mẫu chọn]`

## 4. Công cụ đo
- **PHQ-9** (Kroenke et al., 2001) và **GAD-7** (Spitzer et al., 2006): thang công khai. **Bản tiếng Việt trong phần mềm là bản nháp tự dịch, CHƯA kiểm định**:
  trước khi thu dữ liệu phải thay bằng bản tiếng Việt đã kiểm định (nhóm tự tra cứu, trích dẫn nguồn, rồi đổi `study.INSTRUMENT_VI_STATUS`). `[ĐIỀN: nguồn bản dịch]`
- PSS-10 (Cohen) không đưa vào vì cần xin phép sử dụng; có thể bổ sung sau khi được cấp phép.
- Điểm cắt: PHQ-9 ≥ 10 và GAD-7 ≥ 10 là ngưỡng sàng lọc thường dùng; **không phải chẩn đoán**.

## 5. Quy trình an toàn (bắt buộc, không thể bỏ qua)
1. Câu 9 PHQ-9 (ý nghĩ chết/tự hại) khác 0: ứng dụng hiện ngay các kênh hỗ trợ (`study_panel`). Nhóm phải có **người thật chịu trách nhiệm** `[ĐIỀN: tên, liên hệ, giờ trực]`
   và quy trình liên hệ/chuyển tuyến; bot không đủ.
2. Người tham gia được biết trước (phiếu đồng ý) rằng bảng hỏi có câu về ý nghĩ tự hại và có kênh hỗ trợ.
3. Danh sách kênh hỗ trợ phải được xác minh bằng cách gọi thử/đối chiếu trang chính thức trước khi bắt đầu (`config.RESOURCES_CHECKED`, `config.RESOURCES_TO_VERIFY`);
   thêm liên hệ của trường vào `resources/local_resources.json`.
4. Nếu có người tham gia đang nguy hiểm tức thời, ưu tiên an toàn cá nhân hơn mọi quy trình thu dữ liệu.

## 6. Dữ liệu và quyền riêng tư
Hồ sơ gồm mã ngẫu nhiên, điểm bot, câu trả lời thang đo; **không** có nội dung hội thoại (`study.build_record`). Chỉ người từ 18 tuổi đã bật lựa chọn lưu mới có hồ sơ.
Lưu trữ `[ĐIỀN: nơi, ai được truy cập, thời hạn]`; không đưa lên Git (`data/` đã bị bỏ qua). Người tham gia có quyền rút lui và yêu cầu xoá theo mã.
Bảng điều khiển (`src/dashboard.py`) chỉ hiện số liệu gộp và che ô nhỏ hơn 5.

## 7. Kế hoạch phân tích (đăng ký trước khi nhìn dữ liệu)
- **Chính:** Spearman ρ (bot, PHQ-9) và khoảng tin cậy bootstrap 95%. Quyết định H1 dựa trên khoảng tin cậy, không chỉ giá trị p.
- **Phụ (báo cáo ước lượng và khoảng tin cậy, không suy diễn p khi so sánh nhiều lần):** ρ với GAD-7; AUC cho PHQ-9 ≥ 10 và GAD-7 ≥ 10; kappa có trọng số giữa mức bot và mức thang;
  độ nhạy/độ đặc hiệu theo từng triệu chứng.
- Phân tích tập con theo độ tin cậy của bot (`--min-evidence`): người trò chuyện ít sẽ có điểm kém tin cậy; nêu rõ tiêu chí đã chọn **trước**.
- Chạy thử quy trình bằng `python src/analyze_study.py --demo` (dữ liệu **giả lập**, chỉ để kiểm tra mã).
- Không tinh chỉnh trọng số `CONCERN_WEIGHTS` trên chính dữ liệu dùng để báo cáo; nếu muốn điều chỉnh, chia dữ liệu thành tập chỉnh và tập kiểm.

## 8. Hạn chế cần nêu trong báo cáo
Mẫu thuận tiện; tự báo cáo; thang tiếng Việt `[ĐIỀN]`; bot và thang đo cùng đo trong một phiên; ảnh hưởng của dịch máy và của LLM lên điểm; sinh viên không đại diện dân số.

## 9. Đạo đức
Phiếu đồng ý (`CONSENT_FORM.md`); phê duyệt của hội đồng đạo đức `[ĐIỀN: số quyết định]`; người tham gia tự nguyện, được rút lui bất cứ lúc nào; không có lợi ích kinh tế
gắn với điểm số; thông báo kết quả cho người tham gia (điểm thang kèm lời nhắc đây không phải chẩn đoán, đã có trong ứng dụng).

## 10. Tiến độ `[ĐIỀN]`
Chuẩn bị thang đã kiểm định và phê duyệt đạo đức → thử nghiệm nhỏ 10-15 người (kiểm tra luồng, thời gian, an toàn) → thu dữ liệu → phân tích → báo cáo.
