"""Chatbot dòng lệnh (dùng cùng pipeline với giao diện web).   python src/chatbot.py
Gõ 'tongket' để xem tổng kết, 'thoat' để kết thúc."""
import assessment
import llm_client
import pipeline


def main():
    print("=== CHATBOT LẮNG NGHE (công cụ nghiên cứu, không phải chẩn đoán y khoa) ===")
    print("Gõ 'tongket' để xem tổng kết, 'thoat' để kết thúc. LLM:",
          "BẬT" if llm_client.is_available() else "TẮT (dùng luật)")
    sess = pipeline.Session()
    while True:
        text = input("\nBạn: ").strip()
        if text.lower() == "thoat":
            print("Bot: Tạm biệt, bạn nhớ giữ gìn sức khỏe nhé!")
            return
        if text.lower() == "tongket":
            profile, chunks = pipeline.summary(sess)
            print("\n" + assessment.summary_markdown(profile, "".join(chunks)))
            continue
        if not text:
            continue
        turn = pipeline.run_turn(text, sess)
        p = turn.profile
        if p and "stress" in p:
            print(f"[Cả cuộc trò chuyện: stress {p['stress']['overall']:.0%} | dữ liệu: {p['evidence']} | "
                  f"khía cạnh đã biết: {len(p['domains'])}/{len(assessment.DOMAINS)}]")
        print("Bot: ", end="", flush=True)
        for chunk in turn.chunks:
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
