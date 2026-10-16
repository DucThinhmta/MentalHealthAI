import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


print("===================================")
print("HUAN LUYEN BERT - DEPRESSION")
print("===================================")


# ==========================================
# 1. Đọc dữ liệu
# ==========================================

train = pd.read_csv("./MentalHealthAI/data/depression_train.csv")
test = pd.read_csv("./MentalHealthAI/data/depression_test.csv")

print("\nTrain:", train.shape)
print("Test :", test.shape)


# ==========================================
# 2. Chuyển sang Hugging Face Dataset
# ==========================================

train_dataset = Dataset.from_pandas(
    train[["text", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test[["text", "label"]],
    preserve_index=False
)


# ==========================================
# 3. Tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# ==========================================
# 4. Đổi label -> labels
# ==========================================

train_dataset = train_dataset.rename_column(
    "label",
    "labels"
)

test_dataset = test_dataset.rename_column(
    "label",
    "labels"
)


# ==========================================
# 5. Load BERT 4 lớp
# ==========================================

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)


# ==========================================
# 6. Class weights
# ==========================================

class_weights = torch.tensor(
    [0.3434, 3.0625, 2.2556, 3.1438],
    dtype=torch.float
)


# ==========================================
# 7. Custom Trainer có weighted loss
# ==========================================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device)
        )

        loss = loss_function(
            logits,
            labels
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# ==========================================
# 8. Hàm đánh giá
# ==========================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1
    }


# ==========================================
# 9. Training Arguments
# ==========================================

training_args = TrainingArguments(
    output_dir="./models/bert_depression",
    num_train_epochs=1,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    learning_rate=2e-5,

    eval_strategy="epoch",
    save_strategy="epoch",

    logging_steps=50,

    report_to="none",

    use_cpu=True
)


# ==========================================
# 10. Trainer
# ==========================================

trainer = WeightedTrainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=test_dataset,

    compute_metrics=compute_metrics
)


# ==========================================
# 11. Bắt đầu huấn luyện
# ==========================================

print("\n===================================")
print("BAT DAU HUAN LUYEN...")
print("===================================")

trainer.train()


# ==========================================
# 12. Đánh giá
# ==========================================

print("\n===================================")
print("DANH GIA MO HINH")
print("===================================")

results = trainer.evaluate()

print(results)


# ==========================================
# 13. Lưu model
# ==========================================

trainer.save_model(
    "./models/bert_depression_final"
)

tokenizer.save_pretrained(
    "./models/bert_depression_final"
)

print("\n===================================")
print("DA LUU MODEL")
print("===================================")

print("./models/bert_depression_final")