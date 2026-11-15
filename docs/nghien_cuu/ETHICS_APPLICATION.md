# Đơn đề nghị xét duyệt đạo đức nghiên cứu

> **BẢN NHÁP để nhóm hoàn thiện và nộp.** Mỗi mục `[ĐIỀN]` cần thông tin thật. Hãy đối chiếu với **mẫu đơn riêng của hội đồng đạo đức/đơn vị chủ quản**
> (nếu có) và chuyển nội dung sang mẫu đó. Tài liệu này chỉ mô tả hệ thống đúng như mã nguồn hiện tại; mọi điều chưa làm đều được ghi rõ là chưa làm.
> Đi kèm: `PROTOCOL.md` (đề cương), `PARTICIPANT_INFO_SHEET.md`, `CONSENT_FORM.md`, `SAFETY_SOP.md`, `DATA_MANAGEMENT_PLAN.md`, `SUBMISSION_CHECKLIST.md`.

## 1. Thông tin chung
| Mục | Nội dung |
|---|---|
| Tên đề tài | `[ĐIỀN]` (gợi ý: Kiểm chứng điểm "mức quan tâm" của chatbot lắng nghe so với thang PHQ-9/GAD-7 ở sinh viên) |
| Đơn vị chủ quản / cơ quan chủ trì | `[ĐIỀN]` |
| Chủ nhiệm đề tài, giảng viên hướng dẫn | `[ĐIỀN: họ tên, chức danh, email, điện thoại]` |
| Thành viên nhóm và vai trò | `[ĐIỀN]` |
| Người chịu trách nhiệm an toàn người tham gia (phải là người thật, có mặt trong thời gian thu dữ liệu) | `[ĐIỀN: họ tên, liên hệ, giờ trực]` |
| Chuyên gia tâm lý/sức khỏe tinh thần cố vấn | `[ĐIỀN]` (nên có; xem `SUBMISSION_CHECKLIST.md`) |
| Thời gian dự kiến, địa điểm | `[ĐIỀN]` |
| Nguồn kinh phí, xung đột lợi ích | `[ĐIỀN: ghi rõ nếu không có]` |

## 2. Tóm tắt (dành cho người không chuyên)
Nhóm xây dựng một chatbot cho người dùng trò chuyện về căng thẳng, giấc ngủ, cảm xúc và nhận gợi ý tự giúp mình. Chatbot ước lượng một "mức quan tâm"
từ nội dung và từ các khía cạnh sinh hoạt người dùng kể. **Câu hỏi nghiên cứu:** mức này có tương ứng với kết quả của hai thang tự báo cáo đã được dùng
rộng rãi (PHQ-9, GAD-7) hay không? Người tham gia trò chuyện khoảng 10 phút rồi điền hai bảng hỏi. Đây **không phải thử nghiệm điều trị** và chatbot **không chẩn đoán**.

## 3. Mục tiêu khoa học và giá trị
- Mục tiêu chính và giả thuyết: xem `PROTOCOL.md` mục 1 (H1: Spearman ρ ≥ 0,30 với PHQ-9; H2: AUC ≥ 0,70 cho PHQ-9 ≥ 10; H3: khớp các câu PHQ-9 tương ứng, κ ≥ 0,40).
- Giá trị: chưa có bằng chứng rằng điểm của chatbot phản ánh tình trạng thật. Nếu kết quả **không ủng hộ**, nhóm vẫn công bố và **không** dùng điểm này như một thước đo.
- Vì sao cần người thật: các mô hình được huấn luyện trên bài đăng Reddit tiếng Anh, còn người dùng nhắn tin ngắn bằng tiếng Việt; chỉ dữ liệu thật mới trả lời được.

## 4. Thiết kế và quy trình
1. Tuyển người từ 18 tuổi, đọc `PARTICIPANT_INFO_SHEET.md`, ký `CONSENT_FORM.md` `[ĐIỀN: ký giấy hay xác nhận điện tử]`.
2. Trong ứng dụng: màn hình đồng ý, chọn nhóm tuổi, bật lựa chọn lưu hồ sơ nghiên cứu (tắt mặc định).
3. Trò chuyện tự do ≥ 5 lượt (tối đa khoảng 10 phút). Chatbot không hiện điểm thang đo trước khi người tham gia trả lời.
4. Điền PHQ-9 và GAD-7 (16 câu). Ứng dụng hiện kết quả kèm lời nhắc đây không phải chẩn đoán.
5. Hồ sơ ẩn danh được lưu theo mã ngẫu nhiên (xem mục 7). Người tham gia có thể dừng bất cứ lúc nào.
6. **Thí điểm 10-15 người trước** để kiểm tra luồng, thời gian và quy trình an toàn; chỉnh sửa rồi mới thu chính thức.

## 5. Người tham gia
- Tiêu chí vào: từ 18 tuổi, đọc hiểu tiếng Việt, đồng ý. Tiêu chí loại `[ĐIỀN: ví dụ đang điều trị nội trú tâm thần, đang trong khủng hoảng cấp tính]`.
- Số lượng `[ĐIỀN]`: theo `python src/analyze_study.py --sample-size`, tối thiểu khoảng 85 người cho H1 và khoảng 125-218 người cho H2; nên tuyển thêm 15-20%.
- Cách tuyển `[ĐIỀN]`. Tuyển không gây áp lực: **không** tuyển trực tiếp trong lớp do người chấm điểm các em thực hiện; không dùng điểm học tập làm khuyến khích.
- **Người dưới 18 tuổi không tham gia nghiên cứu.** Ứng dụng chặn lưu hồ sơ nghiên cứu cho nhóm này và chỉ hiện kênh hỗ trợ (kể cả tổng đài 111).
- Nhóm dễ tổn thương: chủ đề là sức khỏe tinh thần nên toàn bộ người tham gia được coi là có thể dễ tổn thương; xem mục 6.

## 6. Rủi ro và biện pháp giảm thiểu
| Rủi ro | Mức | Biện pháp (đã có trong phần mềm / cần nhóm làm) |
|---|---|---|
| Khó chịu khi nói về cảm xúc, trả lời bảng hỏi | Thấp-vừa | Tự nguyện, dừng bất cứ lúc nào, bỏ qua câu hỏi; chatbot không ép trả lời. **Phần mềm:** cuộc trò chuyện không bị khoá, người dùng đổi cách đồng hành (chỉ lắng nghe) và kết thúc phiên bất cứ lúc nào. |
| Người tham gia có ý nghĩ tự hại | Cao nếu xảy ra | **Phần mềm:** hai tầng phát hiện (luật có xử lý phủ định/ẩn dụ + LLM chỉ nâng mức), thông điệp cố định, không dùng LLM trả lời; câu 9 PHQ-9 khác 0 hiện ngay kênh hỗ trợ. **Nhóm:** `SAFETY_SOP.md` (người trực, chuyển tuyến). **Giới hạn đã biết:** recall 0,980 đo trên bộ ca do chính nhóm viết nên lạc quan (đợt kiểm thử với câu mới trước khi sửa chỉ bắt 5/8); chưa đo trên người thật. Phát hiện tự động **không thay thế** người trực. |
| Hiểu nhầm kết quả là chẩn đoán | Vừa | Lời nhắc ở màn hình đồng ý, khi xem tổng kết và khi hiện điểm thang; chatbot từ chối chẩn đoán và tư vấn thuốc (đã kiểm bằng bộ kịch bản hội thoại). |
| Lộ thông tin cá nhân | Vừa | Mặc định không lưu nội dung; hồ sơ nghiên cứu **không chứa nội dung hội thoại** (chỉ mã ngẫu nhiên, điểm, câu trả lời thang); số điện thoại/email/đường dẫn bị xoá khi lưu; bảng quản trị che ô nhỏ hơn 5. Tên người, địa chỉ viết tự do **không** tự xoá được, nên không lưu nội dung. |
| Dữ liệu gửi cho bên thứ ba (LLM) | Vừa | Nếu bật LLM, tin nhắn đi tới dịch vụ AI bên ngoài (Anthropic) để trả lời và dịch. Phải nêu trong phiếu thông tin. **Quyết định cần nhóm đưa ra:** thí điểm có dùng LLM hay chạy bằng luật `[ĐIỀN]`. Nếu dùng LLM cần xem xét chuyển dữ liệu ra nước ngoài theo quy định hiện hành. |
| Thang đo tiếng Việt sai | Vừa | Bản trong phần mềm là **bản nháp tự dịch chưa kiểm định**: phải thay bằng bản đã kiểm định trước khi thu (xem `SUBMISSION_CHECKLIST.md`). |
| Áp lực tham gia | Thấp-vừa | Mục 5 và "Tự nguyện" trong phiếu đồng ý; không ảnh hưởng điểm học tập hay quan hệ với giảng viên. |

**Cân bằng rủi ro-lợi ích:** người tham gia không có lợi ích trực tiếp đảm bảo; rủi ro ở mức tối thiểu đến thấp nếu quy trình an toàn được thực hiện.
Đề nghị hội đồng xét theo loại hình nghiên cứu phù hợp `[ĐIỀN: hội đồng quyết định hình thức xét]`.

## 7. Dữ liệu và quyền riêng tư
Chi tiết: `DATA_MANAGEMENT_PLAN.md`. Tóm tắt:
- **Thu:** mã ngẫu nhiên, ngày, điểm và độ tin cậy của chatbot, số lượt/số từ, nhãn các khía cạnh sinh hoạt, điểm tâm trạng tự chấm, câu trả lời PHQ-9/GAD-7. **Không thu** họ tên, email, số điện thoại, nội dung hội thoại.
- Họ tên trên phiếu đồng ý (nếu có) lưu **tách riêng** khỏi dữ liệu và không nối được với mã.
- Lưu `[ĐIỀN: nơi]`, truy cập `[ĐIỀN: ai]`, thời hạn `[ĐIỀN]`; không đưa lên Git. Rút lui và xoá theo mã.
- Công bố chỉ ở dạng tổng hợp; ô nhỏ hơn 5 bị che.
- Sức khỏe tinh thần là dữ liệu nhạy cảm: nhóm cần đối chiếu **Nghị định 13/2023/NĐ-CP** về bảo vệ dữ liệu cá nhân và văn bản hiện hành mà đơn vị áp dụng `[ĐIỀN: đã đối chiếu]`.

## 8. Đồng ý có thông tin
Phiếu thông tin (`PARTICIPANT_INFO_SHEET.md`) và phiếu đồng ý (`CONSENT_FORM.md`) nêu: mục đích, việc phải làm, rủi ro (kể cả câu hỏi về ý nghĩ tự hại), lợi ích, dữ liệu,
việc gửi tin nhắn tới dịch vụ AI (nếu bật), quyền rút lui và xoá, người liên hệ. Chỉ người từ 18 tuổi.

## 9. Kế hoạch phân tích
Đã đăng ký trước trong `PROTOCOL.md` mục 7 (Spearman + bootstrap KTC 95%; AUC; kappa có trọng số). Quy trình phần mềm đã chạy thử bằng dữ liệu **giả lập**
(`analyze_study.py --demo`): đó là kiểm tra mã, **không phải kết quả**.

## 10. Hạn chế đã biết (nêu trung thực với hội đồng)
Mẫu thuận tiện, tự báo cáo; thang tiếng Việt chưa kiểm định (cho đến khi thay); mô hình huấn luyện trên dữ liệu Reddit tiếng Anh, tiếng Việt qua dịch máy;
mô hình Depression hiện chưa vượt baseline đơn giản (macro-F1 0,365 so với 0,412 một lần chạy; đang được kiểm tra nhiều seed); chế độ LLM chưa được kiểm thử với người thật;
trọng số tổng hợp là heuristic.

## 11. Cam kết
Nhóm cam kết: dừng thu dữ liệu và báo hội đồng nếu có sự cố an toàn; không dùng dữ liệu ngoài mục đích đã nêu; báo cáo mọi thay đổi quy trình;
thông báo kết quả cho người tham gia khi họ yêu cầu.

Chủ nhiệm đề tài: ________________  Giảng viên hướng dẫn: ________________  Ngày: ___/___/______
