import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# =========================
# 1. Đường dẫn model
# =========================

MODEL_PATH = "./models/bert_emotion/checkpoint-5427"


# =========================
# 2. Load tokenizer + model
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


# =========================
# 3. Danh sách 28 cảm xúc
# =========================

emotion_labels = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral"
]


# =========================
# 4. Hàm dự đoán cảm xúc
# =========================
THRESHOLD = 0.25

def predict_emotion(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.sigmoid(
        outputs.logits
    )[0]

    results = []

    for i, probability in enumerate(probabilities):

        if probability >= THRESHOLD:
            results.append(
            (
                emotion_labels[i],
                probability.item()
            )
        )

    results.sort(
         key=lambda x: x[1],
         reverse=True
)

    return results


# =========================
# 5. Test
# =========================

text = "I am really worried about my exams and I can't sleep."

results = predict_emotion(text)

print("\n===== KẾT QUẢ CẢM XÚC =====")
print("Threshold:", THRESHOLD)

if len(results) == 0:

    print("Không phát hiện cảm xúc nào.")

else:

    for emotion, confidence in results:

        print(
            emotion,
            "-",
            f"{confidence * 100:.2f}%"
        )