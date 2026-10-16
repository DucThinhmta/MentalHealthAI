import pandas as pd
import torch
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ==========================================
# 1. Đường dẫn model
# ==========================================

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "models",
        "bert_depression_final"
    )
)

print("MODEL_PATH =", MODEL_PATH)

DATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "depression_test.csv"
    )
)


# ==========================================
# 2. Load model + tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.eval()


# ==========================================
# 3. Đọc Test
# ==========================================

df = pd.read_csv(DATA_PATH)

texts = df["text"].tolist()
true_labels = df["label"].tolist()


# ==========================================
# 4. Dự đoán
# ==========================================

predicted_labels = []

print("===================================")
print("DANG DU DOAN TEST DATASET")
print("===================================")

for i, text in enumerate(texts):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=-1
        ).item()

    predicted_labels.append(prediction)

    if (i + 1) % 100 == 0:
        print(f"Da xu ly: {i + 1}/{len(texts)}")


# ==========================================
# 5. Classification Report
# ==========================================

class_names = [
    "minimum",
    "mild",
    "moderate",
    "severe"
]

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ==========================================
# 6. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print(cm)


# ==========================================
# 7. Vẽ Confusion Matrix
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()

plt.title(
    "Confusion Matrix - BERT Depression"
)

plt.tight_layout()

plt.show()