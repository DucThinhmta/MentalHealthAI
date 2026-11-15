# Chạy dự án trên Google Colab (không cần tải gì về máy)

Mọi thứ nặng (thư viện, mô hình BERT, mô hình dịch, huấn luyện) chạy trên máy chủ Google. Máy của bạn chỉ cần trình duyệt và tài khoản Google.

| Việc | Notebook | Cần GPU? |
|---|---|---|
| Dùng thử / trình bày ứng dụng chatbot | `notebooks/run_app_colab.ipynb` | Không bắt buộc (GPU nhanh hơn) |
| Đo chatbot ở chế độ LLM **miễn phí** (Qwen2.5 qua Ollama) | `notebooks/evaluate_llm_colab.ipynb` | **Có (T4)** |
| Đo lại mô hình, huấn luyện Depression/Stress nhiều seed | `notebooks/train_and_evaluate_colab.ipynb` | **Có (T4)** |

Mở bằng nút "Mở trong Colab" ở đầu mỗi notebook (sau khi nhánh `main` có notebook), hoặc: Colab > File > Open notebook > GitHub > dán `DucThinhmta/MentalHealthAI`.

## Thiết lập một lần cho cả nhóm
1. **Cất khoá bằng Colab Secrets** (biểu tượng chìa khoá bên trái, bật "Notebook access"), mỗi người tự làm trong tài khoản của mình:
   `HF_TOKEN` (nếu mô hình trên Hugging Face ở chế độ private), `GITHUB_TOKEN` (nếu repo private), `ANTHROPIC_API_KEY` (tuỳ chọn, có phí). Không có khoá: bật `USE_FREE_LLM` trong notebook ứng dụng để dùng LLM miễn phí chạy trên Colab.
   Không dán khoá vào ô mã; không chia sẻ notebook đã chứa khoá.
2. **Dữ liệu huấn luyện** (chỉ khi chạy notebook huấn luyện): một người giữ dữ liệu tạo thư mục Drive `MentalHealthAI_colab/data/`, đưa 7 tệp vào (tên tệp trong notebook và `data/README.md`),
   chia sẻ cả thư mục `MentalHealthAI_colab` cho thành viên (quyền Editor, **không** để "anyone with the link" vì các bộ dữ liệu có điều khoản riêng). Mỗi thành viên **Add shortcut to Drive** cho thư mục đó.

## Chạy ứng dụng
1. Mở `run_app_colab.ipynb`, **Runtime > Run all**.
2. Mở liên kết `https://....trycloudflare.com` in ở ô cuối. Giữ tab notebook mở; xong thì **Runtime > Disconnect and delete runtime**.

Lưu ý quan trọng:
- Liên kết là **địa chỉ công khai tạm thời**: ai có liên kết đều dùng được. Chỉ gửi cho người trong nhóm.
- Trò chuyện không được lưu (phiên Colab xoá khi đóng), nhưng nội dung vẫn đi qua máy chủ Colab. **Không dùng cách này để thu dữ liệu nghiên cứu từ người thật** khi chưa có phê duyệt đạo đức và quy trình an toàn (`docs/nghien_cuu/`).
- Có `ANTHROPIC_API_KEY` thì tin nhắn còn đi tới dịch vụ AI của Anthropic. Không có thì bot trả lời theo luật.
- Muốn chạy công cụ khác (bảng điều khiển, công cụ gán nhãn): trong một ô mới, `ch.start_app(8502, app="src/dashboard.py")` rồi `ch.start_tunnel(8502)`
  (bảng điều khiển cần `MHAI_ADMIN_CODE` trong `env`).

## Huấn luyện
1. Chọn **GPU (T4)**: Runtime > Change runtime type.
2. Chạy từ trên xuống. Sau mỗi giai đoạn kết quả được sao lưu vào `MyDrive/MentalHealthAI_colab/`; khi phiên bị ngắt, mở lại notebook và chạy lại: kết quả cũ tự được khôi phục, các seed đã xong được bỏ qua (`--skip-done`).
3. Kết quả (`reports/`) nằm chung trên Drive cho cả nhóm. Trọng số mô hình nặng không được sao lưu mặc định.
4. Hai người không nên cùng chạy một cấu hình/seed cùng lúc vào cùng thư mục Drive (sẽ ghi đè nhau): chia nhau theo `--tag` hoặc `--seeds`.

## Giới hạn của Colab miễn phí
GPU không đảm bảo luôn có, phiên tự ngắt sau một thời gian dài hoặc khi bỏ không, và giới hạn giờ GPU mỗi ngày. Vì vậy huấn luyện được chia thành từng seed có sao lưu. Nếu hết hạn mức, đợi hoặc dùng Colab Pro.
Colab chỉ chạy được khi có mạng; không dùng cho dữ liệu mà quy định không cho phép đưa lên đám mây.
