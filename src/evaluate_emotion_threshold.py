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
# 1. Đọc TEST
# =========================

test = pd.read_csv(
    "./MentalHealthAI/data/test_emotion.csv"
)

test["labels"] = test["label_vector"].apply(
    lambda x: [float(v) for v in ast.literal_eval(x)]
)

test = test[["text", "labels"]]

test_dataset = Dataset.from_pandas(test)


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


test_tokenized = test_dataset.map(
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
# 4. Dự đoán TEST
# =========================

trainer = Trainer(
    model=model
)

predictions = trainer.predict(test_tokenized)

logits = predictions.predictions

labels = np.array(
    test["labels"].tolist()
)


# =========================
# 5. Sigmoid
# =========================

probabilities = 1 / (1 + np.exp(-logits))


# =========================
# 6. Threshold = 0.25
# =========================

THRESHOLD = 0.25

pred = (
    probabilities >= THRESHOLD
).astype(int)


# =========================
# 7. Tính F1
# =========================

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


# =========================
# 8. Kết quả
# =========================

print("\n===== TEST VỚI THRESHOLD 0.25 =====")

print(
    "F1 Micro:",
    f"{f1_micro:.4f}"
)

print(
    "F1 Macro:",
    f"{f1_macro:.4f}"
)