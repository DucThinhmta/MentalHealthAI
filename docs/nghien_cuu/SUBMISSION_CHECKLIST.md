# Danh sách việc trước khi nộp hồ sơ đạo đức và trước khi thu dữ liệu

Đánh dấu khi xong. Mục **[BLOCKER]** nghĩa là chưa làm thì **không được thu dữ liệu từ người thật**.

## A. Trước khi nộp hồ sơ cho hội đồng
- [ ] Hỏi **đơn vị chủ quản/khoa** về hội đồng đạo đức nào xét, mẫu đơn, thời hạn xét, và hình thức xét phù hợp (quy trình khác nhau giữa các trường). Việc này quyết định thời gian chờ nên làm đầu tiên.
- [ ] Điền mọi `[ĐIỀN]` trong `ETHICS_APPLICATION.md`, `PARTICIPANT_INFO_SHEET.md`, `CONSENT_FORM.md`, `SAFETY_SOP.md`, `DATA_MANAGEMENT_PLAN.md`, `PROTOCOL.md`. (Tìm `ĐIỀN` trong thư mục để không sót.)
- [ ] Chuyển nội dung sang **mẫu đơn của hội đồng** nếu họ có mẫu riêng.
- [ ] Giảng viên hướng dẫn xem và ký bảo trợ.
- [ ] Quyết định 2 điểm của `DATA_MANAGEMENT_PLAN.md` mục 2: có lưu nội dung không, có bật LLM bên ngoài không; rồi chỉnh phiếu thông tin cho khớp.
- [ ] Đính kèm: đề cương, phiếu thông tin, phiếu đồng ý, bản thang đo sẽ dùng, SOP an toàn, kế hoạch dữ liệu, lý lịch ngắn của nhóm, `docs/MODEL_CARD.md`, ảnh chụp màn hình các màn hình người tham gia thấy.

## B. [BLOCKER] Trước khi thu dữ liệu
- [ ] **Hội đồng đạo đức đã duyệt** (ghi số quyết định vào phiếu đồng ý và đề cương).
- [ ] **Thay bản thang đo nháp bằng bản tiếng Việt đã kiểm định** của PHQ-9 và GAD-7 (nhóm tra cứu, trích dẫn nguồn, xin phép nếu cần), rồi đổi `study.INSTRUMENT_VI_STATUS` và nội dung `study.py`. Bản hiện tại là bản tự dịch chưa kiểm định.
- [ ] **Người phụ trách an toàn** đã điền tên, liên hệ, giờ trực (`SAFETY_SOP.md` mục 1) và đã đọc SOP.
- [ ] **Kênh hỗ trợ đã tự gọi thử**; điền phòng tư vấn của trường vào `resources/local_resources.json`; kiểm tra lại `config.RESOURCES_CHECKED`. Xác minh (hoặc bỏ) `1900 1267`, `(024) 3576 5344`.
- [ ] Đã diễn tập SOP (mục 5 của SOP).
- [ ] Đã thí điểm 10-15 người **không thuộc mẫu chính** và sửa lỗi quy trình.
- [ ] Nơi lưu dữ liệu và người được truy cập đã chốt; `MHAI_ADMIN_CODE` đã đặt.
- [ ] Ứng dụng chạy ở nơi quyết định (mục 2 của kế hoạch dữ liệu), thử đủ luồng: đồng ý, trò chuyện, bảng hỏi, câu 9 dương tính, rút lui.

## C. Khuyến nghị để nghiên cứu đáng tin hơn
- [ ] Mời **chuyên gia tâm lý** cố vấn (xem các câu trả lời mẫu, quy trình an toàn).
- [ ] Mở rộng bộ kiểm tra an toàn bằng ca do chuyên gia viết (hiện recall 0,978 đo trên bộ tự viết nên lạc quan).
- [ ] Hoàn thành **Bước 1-2 mô hình** (`notebooks/train_and_evaluate_colab.ipynb`) và ghi số liệu nhiều seed vào `MODEL_CARD.md` trước khi viết phần kết quả.
- [ ] Đăng ký trước kế hoạch phân tích (đã có ở `PROTOCOL.md` mục 7; đừng sửa sau khi nhìn dữ liệu).
- [ ] Tập vàng tiếng Việt 300-500 tin nhắn do ít nhất 2 người gán nhãn (`ANNOTATION_GUIDE.md`).

## D. Trong khi thu dữ liệu
- [ ] Người trực có mặt mọi buổi; ghi mọi sự cố.
- [ ] Mỗi tuần xem số liệu gộp (không xem dòng cá nhân khi không cần).
- [ ] Báo hội đồng khi có sự cố an toàn hoặc cần đổi quy trình.
