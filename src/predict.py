import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Đường dẫn đến model BERT đã train
MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "bert_stress_final"
    )
)

# =========================
# LOAD MODEL
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


# =========================
# HÀM DỰ ĐOÁN
# =========================

def predict_stress(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    prediction = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[0][prediction].item()

    if prediction == 0:
        label = "Khong stress"
    else:
        label = "Co dau hieu stress"

    return label, confidence
print("Load model thanh cong")

label, confidence = predict_stress(
    "I feel very stressed and overwhelmed lately."
)

print("Ket qua:", label)
print("Do tin cay:", f"{confidence * 100:.2f}%")