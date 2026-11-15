# Quy trình an toàn khi người tham gia có dấu hiệu nguy hiểm (SOP)

> **MẪU để nhóm hoàn thiện cùng chuyên gia tâm lý.** Chatbot **không đủ** để bảo đảm an toàn: phải có người thật chịu trách nhiệm trong suốt thời gian thu dữ liệu.
> Nhóm không được tự đảm nhận vai trò điều trị. Mục tiêu duy nhất: người đó được kết nối với hỗ trợ thật nhanh và nhẹ nhàng.

## 1. Vai trò (điền trước khi thu dữ liệu)
| Vai trò | Họ tên | Liên hệ | Giờ trực |
|---|---|---|---|
| Người phụ trách an toàn (có mặt khi có người tham gia) | `[ĐIỀN]` | `[ĐIỀN]` | `[ĐIỀN]` |
| Người thay thế | `[ĐIỀN]` | `[ĐIỀN]` | `[ĐIỀN]` |
| Chuyên gia tâm lý/tư vấn cố vấn (hoặc phòng tư vấn của trường) | `[ĐIỀN]` | `[ĐIỀN]` | `[ĐIỀN]` |

Quy tắc: **không thu dữ liệu khi không có người phụ trách an toàn trực.**

## 2. Khi nào quy trình được kích hoạt
| Tín hiệu | Nguồn | Mức |
|---|---|---|
| Ứng dụng hiện thông điệp khủng hoảng (có biểu hiện tự hại trong tin nhắn) | `safety.py` (luật) hoặc lớp LLM nâng mức | **Cao** |
| Câu 9 PHQ-9 khác 0 | `study.item9_positive` | **Vừa-cao** (cần người hỏi thăm trong cùng buổi) |
| Người tham gia nói trực tiếp với nhân sự nhóm rằng họ đang nguy hiểm | người trực | **Cao** |
| Người tham gia tỏ ra rất đau khổ, khóc, muốn dừng | người trực | **Thấp-vừa** (dừng, hỏi thăm, hỗ trợ) |

Lưu ý: bộ phát hiện tự động có thể **bỏ sót** hoặc **báo nhầm**. Đã đo trên bộ ca do chính nhóm viết (recall 0,978, lạc quan); chưa đo trên người thật.
Vì vậy người trực phải để ý người tham gia, không chỉ chờ cảnh báo.

## 3. Các bước
**Bước 1: dừng thu dữ liệu (ngay).** Người trực đến ngồi cạnh người tham gia (nếu thu tại chỗ) hoặc liên hệ ngay (nếu từ xa, theo cách đã thỏa thuận trong phiếu đồng ý).

**Bước 2: hỏi thăm nhẹ nhàng, không phán xét.** Gợi ý lời nói: "Mình thấy bạn đang rất nặng lòng. Bạn có muốn nói thêm với mình không?"
Không hứa giữ bí mật tuyệt đối; không tranh luận; không để người đó ở một mình khi nghi ngờ nguy hiểm tức thời.

**Bước 3: đánh giá nhanh mức nguy hiểm** (cần người được tập huấn hoặc chuyên gia; nhóm sinh viên không tự đánh giá lâm sàng):
- Có ý định hoặc kế hoạch làm hại bản thân ngay bây giờ, hoặc có phương tiện trong tầm tay?  → **Bước 4A**
- Có ý nghĩ nhưng chưa có kế hoạch/ý định tức thời?  → **Bước 4B**
- Chỉ đau khổ nhiều, không có ý nghĩ tự hại?  → **Bước 4C**

**Bước 4A (nguy hiểm tức thời).** Gọi **115** (cấp cứu) hoặc **113**; ở bên cạnh người đó cho đến khi có người đến; báo người thân/đơn vị theo quy định của trường `[ĐIỀN: ai, số nào]`.

**Bước 4B (có ý nghĩ, chưa tức thời).** Kết nối cùng buổi với chuyên gia/phòng tư vấn của trường `[ĐIỀN]`; đề nghị người đó cho phép báo người thân tin cậy; đưa danh sách kênh hỗ trợ; hẹn hỏi thăm lại trong `[ĐIỀN: 24 giờ]`.

**Bước 4C (đau khổ nhưng không có ý nghĩ tự hại).** Cho nghỉ, đưa thông tin hỗ trợ tâm lý của trường, hỏi người đó có muốn dừng hoặc tiếp tục; không ép.

**Bước 5: ghi lại và báo cáo.** Ghi (không kèm danh tính nếu không cần): thời điểm, tín hiệu, bước đã làm, kết quả. Báo chủ nhiệm đề tài **và hội đồng đạo đức** theo thời hạn họ quy định `[ĐIỀN]`.
Dừng thu dữ liệu cho đến khi chủ nhiệm đồng ý tiếp tục.

## 4. Kênh hỗ trợ (phải kiểm tra trước khi bắt đầu)
- Đã đối chiếu (hiển thị trong ứng dụng): 115, 113; trang tổng hợp https://findahelpline.com/countries/vn/topics/suicidal-thoughts; tổng đài bảo vệ trẻ em 111 cho người dưới 18 tuổi
  (xác minh chủ yếu qua tin tức, **chưa** đối chiếu trang chính thức của Cục Trẻ em).
- **Chưa xác minh, KHÔNG hiển thị:** `1900 1267` (Bệnh viện Tâm thần TP.HCM), `(024) 3576 5344` (Viện Sức khỏe Tâm thần Quốc gia). Nhóm tự gọi thử, hỏi vai trò và giờ làm việc rồi mới đưa vào
  (`config.RESOURCES_TO_VERIFY`).
- Phòng tư vấn của trường: điền vào `resources/local_resources.json` sau khi gọi thử. `[ĐIỀN]`
- Ngày kiểm tra gần nhất: `config.RESOURCES_CHECKED`. Kiểm tra lại ngay trước khi thu dữ liệu.

## 5. Thử diễn tập trước khi thu thật
- [ ] Mỗi thành viên đọc SOP; người trực biết số gọi và việc phải làm.
- [ ] Diễn tập 1 tình huống giả định (mức 4B) trong thí điểm, không dùng người tham gia thật có nguy cơ.
- [ ] Gọi thử các số trong mục 4 và ghi kết quả.
- [ ] Kiểm tra ứng dụng: nhập tin nhắn kiểm thử có nguy cơ, thấy thông điệp cố định và danh sách kênh hỗ trợ đúng; đặt câu 9 PHQ-9 > 0 thấy kênh hỗ trợ.
