"""Chatbot dòng lệnh (dùng cùng pipeline với giao diện web).   python src/chatbot.py"""
import analyze
import llm_client
import response_generator as rg


def main():
    print("=== CHATBOT PHÂN TÍCH TÂM LÝ (công cụ nghiên cứu, không phải chẩn đoán y khoa) ===")
    print("Nhập 'thoat' để kết thúc. LLM:", "BẬT" if llm_client.is_available() else "TẮT (dùng luật)")
    user_msgs, history = [], []
    while True:
        text = input("\nBạn: ").strip()
        if text.lower() == "thoat":
            print("Bot: Tạm biệt, bạn nhớ giữ gìn sức khỏe nhé!")
            return
        if not text:
            continue
        user_msgs.append(text); history.append({"role": "user", "content": text})
        result = analyze.analyze_conversation(user_msgs)
        print(f"[Stress {result['stress_probability']:.0%} | Depression: {result['depression']} | "
              f"Cảm xúc: {', '.join(e for e, _ in result['emotions'][:3]) or '-'}]")
        out = rg.generate_reply(result, text, history)
        history.append({"role": "assistant", "content": out["reply"]})
        print("Bot:", out["reply"])


if __name__ == "__main__":
    main()
