import os
import sys
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ==================================================
# 1. Đường dẫn
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
STRESS_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "bert_stress_final"
)

EMOTION_MODEL_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "models",
        "bert_emotion",
        "checkpoint-5427"
    )
)


# ==================================================
# 2. Load STRESS MODEL
# ==================================================

stress_tokenizer = AutoTokenizer.from_pretrained(
    STRESS_MODEL_PATH
)

stress_model = AutoModelForSequenceClassification.from_pretrained(
    STRESS_MODEL_PATH
)

stress_model.eval()


# ==================================================
# 3. Load EMOTION MODEL
# ==================================================

emotion_tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

emotion_model = AutoModelForSequenceClassification.from_pretrained(
    EMOTION_MODEL_PATH
)

emotion_model.eval()


# ==================================================
# 4. Danh sách cảm xúc
# ==================================================

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


# ==================================================
# 5. Phân tích STRESS
# ==================================================

def predict_stress(text):

    inputs = stress_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = stress_model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    prediction = torch.argmax(
        probabilities
    ).item()

    confidence = probabilities[
        prediction
    ].item()

    if prediction == 0:
        label = "Khong stress"
    else:
        label = "Co dau hieu stress"

    return label, confidence


# ==================================================
# 6. Phân tích EMOTION
# ==================================================

def predict_emotion(text):

    inputs = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = emotion_model(
            **inputs
        )

    probabilities = torch.sigmoid(
        outputs.logits
    )[0]

    THRESHOLD = 0.25

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


# ==================================================
# 7. Phân tích tổng hợp
# ==================================================

def analyze(text):

    stress_label, stress_confidence = predict_stress(
        text
    )

    emotions = predict_emotion(
        text
    )

    return {
        "stress": stress_label,
        "stress_confidence": stress_confidence,
        "emotions": emotions
    }


# ==================================================
# 8. Test
# ==================================================

if __name__ == "__main__":

    text = input("Nhap cau can phan tich: ")

    result = analyze(text)

    print("\n===== PHÂN TÍCH HỘI THOẠI =====")

    print(
        "Stress:",
        result["stress"]
    )

    print(
        "Độ tin cậy:",
        f"{result['stress_confidence'] * 100:.2f}%"
    )

    print("\nCảm xúc:")

    if len(result["emotions"]) == 0:

        print("Không phát hiện cảm xúc vượt threshold.")

    else:

        for emotion, confidence in result["emotions"]:

            print(
                f"- {emotion}: "
                f"{confidence * 100:.2f}%"
            )