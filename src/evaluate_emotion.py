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
# 1. Đọc test
# =========================

test = pd.read_csv(
    "./MentalHealthAI/data/test_emotion.csv"
)


# =========================
# 2. Chuyển labels
# =========================

test["labels"] = test["label_vector"].apply(
    lambda x: [float(v) for v in ast.literal_eval(x)]
)


test = test[["text", "labels"]]


# =========================
# 3. Chuyển sang Dataset
# =========================

test_dataset = Dataset.from_pandas(test)


# =========================
# 4. Tokenizer
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
# 5. Load model đã train
# =========================

MODEL_PATH = "./models/bert_emotion/checkpoint-5427"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)


# =========================
# 6. Hàm tính F1
# =========================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    probabilities = 1 / (1 + np.exp(-logits))

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    f1_micro = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0
    )

    f1_macro = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro
    }


# =========================
# 7. Trainer
# =========================

trainer = Trainer(
    model=model,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics
)


# =========================
# 8. Đánh giá Test
# =========================

print("\n===== ĐÁNH GIÁ TEST =====")

results = trainer.evaluate()

print("\nKẾT QUẢ:")
print(results)