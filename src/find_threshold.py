import pandas as pd
import ast
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.metrics import f1_score


# =========================
# 1. Đọc tập DEV
# =========================

dev = pd.read_csv(
    "./MentalHealthAI/data/dev_emotion.csv"
)

dev["labels"] = dev["label_vector"].apply(
    lambda x: [float(v) for v in ast.literal_eval(x)]
)

dev = dev[["text", "labels"]]

dev_dataset = Dataset.from_pandas(dev)


# =========================
# 2. Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


dev_tokenized = dev_dataset.map(
    tokenize_function,
    batched=True
)


# =========================
# 3. Load model
# =========================

MODEL_PATH = "./models/bert_emotion/checkpoint-5427"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


# =========================
# 4. Dự đoán toàn bộ DEV
# =========================

trainer = Trainer(
    model=model
)

predictions = trainer.predict(dev_tokenized)

logits = predictions.predictions
labels = np.array(dev["labels"].tolist())


# =========================
# 5. Chuyển logits → xác suất
# =========================

probabilities = 1 / (1 + np.exp(-logits))


# =========================
# 6. Thử nhiều threshold
# =========================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

print("\n===== SO SÁNH THRESHOLD =====")

best_threshold = 0
best_f1 = 0

for threshold in thresholds:

    pred = (
        probabilities >= threshold
    ).astype(int)

    f1_micro = f1_score(
        labels,
        pred,
        average="micro",
        zero_division=0
    )

    f1_macro = f1_score(
        labels,
        pred,
        average="macro",
        zero_division=0
    )

    print(
        f"Threshold {threshold:.2f}"
        f" | F1 Micro: {f1_micro:.4f}"
        f" | F1 Macro: {f1_macro:.4f}"
    )

    if f1_micro > best_f1:
        best_f1 = f1_micro
        best_threshold = threshold


print("\n===== THRESHOLD TỐT NHẤT =====")
print("Threshold:", best_threshold)
print("F1 Micro:", best_f1)