# Hướng dẫn gán nhãn tin nhắn tiếng Việt

Mục tiêu: tạo **tập vàng** để đo thật độ chính xác của hệ thống trên tiếng Việt (`src/evaluate_gold.py`). Chất lượng tập vàng quyết định giá trị mọi con số sau đó,
nên cần **hai người gán độc lập** và đo độ đồng thuận (kappa).

## Quy trình
1. Tạo mẫu: `python src/sample_for_annotation.py --n 600` (đã xoá số điện thoại/email/đường dẫn). Ưu tiên nhật ký chat thật của người đã đồng ý (`--logs`).
2. **Thử nghiệm nhỏ:** hai người cùng gán 50 mục đầu (độc lập), chạy `python src/agreement.py <tên1> <tên2>`, đọc các mục bất đồng, sửa hướng dẫn nếu cần, **rồi mới gán hàng loạt**.
3. Mỗi người chạy `streamlit run src/annotate_app.py` trên máy riêng, nhập tên/mã của mình. Không bàn bạc, không xem nhãn của nhau, **không xem dự đoán của bot** (giao diện không hiện).
4. Tính kappa; hai người họp xử lý các mục bất đồng trong `data/annotation/adjudication.csv` (điền cột `final_*`; dùng `-` nếu thống nhất để trống), rồi
   `python src/agreement.py <tên1> <tên2> --build-gold` → `data/gold/vi_gold.csv`. Mục chưa thống nhất bị loại.
5. **Mục tiêu kappa ≥ 0,60** cho stress và nguy cơ tự hại (Landis & Koch: "khá tốt"). Thấp hơn: đừng dùng làm chuẩn, hãy làm rõ hướng dẫn và gán lại một mẫu.

## Nhãn
Gán theo điều người viết **thể hiện trong chính tin nhắn đó**, không suy đoán thêm. Không chắc thì chọn "Không rõ".

| Trường | Giá trị | Cách gán |
|---|---|---|
| Stress | có / không / không rõ | Người viết thể hiện căng thẳng, áp lực, quá tải, lo âu. "Hôm nay thi xong, mệt quá nhưng nhẹ nhõm" → không (mệt không bằng căng thẳng). |
| Cảm xúc chính | 28 nhãn GoEmotions hoặc không rõ | Một nhãn nổi bật nhất. Nhiều cảm xúc: chọn cái người viết nhấn mạnh. Trung tính: `neutral`. |
| Mức nặng nề | 0 rất ít · 1 nhẹ · 2 vừa · 3 nặng · trống | Đánh giá **nội dung** (không phải bạn đoán người viết bị bệnh gì). 0: bình thường/vui; 1: buồn/lo thoáng qua; 2: kéo dài, ảnh hưởng sinh hoạt; 3: tuyệt vọng, mất hết ý nghĩa, ý nghĩ tự hại. |
| Nguy cơ tự hại | 0 / 1 | **1 nếu có bất kỳ dấu hiệu nào**, kể cả gián tiếp, tiếng lóng, nói về người khác ("bạn mình muốn tự tử"). Thà gán thừa còn hơn bỏ sót. Nói ẩn dụ rõ ràng ("buồn cười muốn chết") hoặc phủ định ("mình không muốn chết") → 0. |
| 8 khía cạnh | trống / ok / concern | Chỉ gán khi tin nhắn **nhắc tới** khía cạnh đó. `concern`: có vấn đề (mất ngủ, mệt, bỏ bữa, mất hứng thú, khó tập trung, cô đơn, buồn, kéo dài nhiều tuần). `ok`: nói là ổn. Không nhắc: để trống. |

### Các tình huống dễ nhầm
- **Châm biếm, đùa:** gán theo nghĩa thật mà người viết ngụ ý; nếu không thể biết, "không rõ".
- **Tin nhắn quá ngắn** ("ừ", "không"): thường "không rõ" hoặc trống ở mọi trường.
- **Nói về người khác:** stress/cảm xúc gán cho người viết; riêng nguy cơ tự hại gán 1 nếu nhắc tới ai đó có ý định tự hại.
- **Tiếng lóng, viết tắt, không dấu, lẫn tiếng Anh:** cố hiểu theo ngữ cảnh; không hiểu thì "không rõ".
- **Nội dung gây khó chịu:** bạn được nghỉ bất cứ lúc nào; nếu bản thân thấy nặng nề khi gán nhãn, hãy dừng và nói với người phụ trách.

## Đạo đức khi gán nhãn
Dữ liệu đã được xoá thông tin nhận dạng có cấu trúc nhưng **không xoá được tên người hay địa chỉ viết tự do**: nếu thấy thông tin nhận dạng, đánh dấu ở ô ghi chú và báo người phụ trách,
không sao chép ra ngoài. Không chia sẻ tệp gán nhãn qua kênh công khai.
