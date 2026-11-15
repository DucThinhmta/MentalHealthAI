# Kế hoạch quản lý dữ liệu

> **MẪU để nhóm hoàn thiện.** Mô tả đúng hành vi của phần mềm hiện tại; các mục `[ĐIỀN]` là quyết định của nhóm/đơn vị.

## 1. Dữ liệu nào được thu
| Dữ liệu | Có thu? | Ghi chú |
|---|---|---|
| Mã ngẫu nhiên dạng `P-XXXXXX` | Có | `privacy.new_participant_code`; không suy ra được danh tính |
| Ngày thu, ngôn ngữ | Có | |
| Điểm "mức quan tâm" của chatbot, mức, độ tin cậy, số lượt, số từ | Có | `study.build_record` |
| Nhãn 8 khía cạnh sinh hoạt (ổn/đáng chú ý/chưa rõ), tâm trạng tự chấm | Có | |
| Điểm thành phần (stress, nặng nề) | Có | số liệu, không phải văn bản |
| Câu trả lời PHQ-9 / GAD-7, tổng điểm, câu 9 dương tính | Có | |
| **Nội dung cuộc trò chuyện** | **Không** trong hồ sơ nghiên cứu | xem mục 2 về nhật ký phân tích |
| Họ tên, email, số điện thoại, địa chỉ IP | Không | ứng dụng không hỏi và không ghi |
| Họ tên trên phiếu đồng ý (nếu ký giấy) | Có, **lưu tách riêng** | không nối được với mã; `[ĐIỀN: nơi cất]` |

## 2. Điểm cần nhóm quyết định (đừng để mơ hồ)
1. **Nhật ký phân tích (`analysis_log.json`)** là tệp người dùng/nhóm có thể tải từ mục "Dành cho nhóm nghiên cứu". Khi người dùng bật lựa chọn lưu bản ẩn danh, nhật ký này
   **giữ nội dung đã xoá số điện thoại/email/đường dẫn nhưng KHÔNG xoá được tên người, địa chỉ viết tự do.** Phiếu đồng ý và hồ sơ nghiên cứu hứa "không lưu nội dung".
   → Đề xuất cho nghiên cứu kiểm chứng: **không bật lưu nội dung**, chỉ dùng hồ sơ `study.build_record`. Nếu muốn lưu nội dung (để cải thiện mô hình) thì phải thêm điều khoản riêng
   trong phiếu đồng ý, tách khỏi phần kiểm chứng, và rà soát thủ công trước khi chia sẻ. `[ĐIỀN: quyết định]`
2. **LLM bên ngoài:** nếu bật, tin nhắn đi tới Anthropic (có thể xử lý ở nước ngoài) trong lúc trò chuyện. `[ĐIỀN: bật/không bật trong nghiên cứu; nếu bật, đã ghi trong phiếu thông tin]`.
3. **Ứng dụng chạy ở đâu:** máy của nhóm, máy chủ của trường, hay dịch vụ đám mây `[ĐIỀN]`. Nơi chạy quyết định ai có thể thấy nội dung đi qua phiên.

## 3. Lưu trữ và truy cập
- Hồ sơ nằm ở `data/study/` (`config.STUDY_DIR`), **đã bị `.gitignore`: không đưa lên Git/GitHub.**
- Nơi lưu thật `[ĐIỀN: ổ đĩa/thư mục của trường, có mã hoá]`; sao lưu `[ĐIỀN]`.
- Người được truy cập `[ĐIỀN: tên/vai trò, tối đa bao nhiêu người]`; ghi lại ai đã tải dữ liệu.
- Bảng điều khiển (`src/dashboard.py`) cần mã quản trị (`MHAI_ADMIN_CODE`), chỉ hiện số liệu gộp và che ô nhỏ hơn 5. Đặt mã dài, không chia sẻ chung.

## 4. Thời hạn và xoá
- Giữ đến `[ĐIỀN: ví dụ 12 tháng sau khi công bố]` rồi xoá hoặc ẩn danh hoàn toàn.
- **Quyền xoá:** người tham gia đưa mã ngẫu nhiên → nhóm xoá dòng tương ứng trong `data/study/`. Sau khi dữ liệu đã được gộp để công bố, không thể rút từng cá nhân; phải nói rõ điều này trong phiếu thông tin
  `[ĐIỀN: mốc thời gian sau đó không rút được]`.
- Nhật ký tâm trạng (`diary.py`) là chức năng riêng của người dùng, không thuộc nghiên cứu; khoá là mã người dùng tự giữ.

## 5. Chia sẻ và công bố
- Chỉ công bố số liệu tổng hợp; ô nhỏ hơn 5 bị che; không công bố dòng cá nhân.
- Nếu chia sẻ dữ liệu để người khác tái lập: chỉ chia sẻ hồ sơ đã ẩn danh, sau khi hội đồng đồng ý và kiểm tra khả năng nhận diện lại (mẫu nhỏ, đặc điểm hiếm).
- Mã nguồn công khai trên GitHub **không chứa dữ liệu người tham gia**.

## 6. Bảo mật
- Tối thiểu: máy có mật khẩu và mã hoá ổ đĩa; không gửi dữ liệu qua tin nhắn/email cá nhân; khoá API (`ANTHROPIC_API_KEY`) chỉ ở biến môi trường, không đưa vào mã.
- Ghi sự cố lộ dữ liệu (nếu có) và báo hội đồng đạo đức `[ĐIỀN: thời hạn]`.

## 7. Văn bản pháp lý cần đối chiếu
Dữ liệu sức khỏe tinh thần là dữ liệu cá nhân nhạy cảm. Nhóm đối chiếu **Nghị định 13/2023/NĐ-CP** và các văn bản hiện hành mà đơn vị áp dụng, và quy định riêng của trường về nghiên cứu có người tham gia. `[ĐIỀN: đã đối chiếu, ghi người đối chiếu và ngày]`
