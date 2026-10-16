import pandas as pd
import ast
from datasets import Dataset


# =========================
# 1. Đọc dữ liệu
# =========================

train = pd.read_csv(
    "./MentalHealthAI/data/train_emotion.csv"
)

dev = pd.read_csv(
    "./MentalHealthAI/data/dev_emotion.csv"
)

test = pd.read_csv(
    "./MentalHealthAI/data/test_emotion.csv"
)


# =========================
# 2. Chuyển label_vector
#    từ chuỗi -> list
# =========================

train["labels"] = train["label_vector"].apply(
    lambda x: [float(v) for v in ast.literal_eval(x)]
)

dev["labels"] = dev["label_vector"].apply(
    lambda x: [float(v) for v in ast.literal_eval(x)]
)

test["labels"] = test["label_vector"].apply(
    lambda x: [float(v) for v in ast.literal_eval(x)]
)

# =========================
# 3. Chỉ giữ text + labels
# =========================

train = train[["text", "labels"]]
dev = dev[["text", "labels"]]
test = test[["text", "labels"]]


# =========================
# 4. Chuyển sang Hugging Face Dataset
# =========================

train_dataset = Dataset.from_pandas(train)
dev_dataset = Dataset.from_pandas(dev)
test_dataset = Dataset.from_pandas(test)


# =========================
# 5. Kiểm tra
# =========================

print("TRAIN:", train_dataset)
print("DEV:", dev_dataset)
print("TEST:", test_dataset)

print("\nMẫu đầu tiên:")
print(train_dataset[0])
from transformers import AutoTokenizer


# =========================
# 6. Tải tokenizer BERT
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)


# =========================
# 7. Hàm tokenize
# =========================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


# ========================= 
# 8. Tokenize 3 tập dữ liệu
# =========================

train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True
)

dev_tokenized = dev_dataset.map(
    tokenize_function,
    batched=True
)

test_tokenized = test_dataset.map(
    tokenize_function,
    batched=True
)


# =========================
# 9. Kiểm tra
# =========================

print("\nTokenize hoàn tất!")

print("Train:", train_tokenized)
print("Dev:", dev_tokenized)
print("Test:", test_tokenized)

print("\nMột mẫu sau tokenize:")
print(train_tokenized[0].keys())

print("\nĐộ dài input_ids:",
      len(train_tokenized[0]["input_ids"]))

print("Độ dài attention_mask:",
      len(train_tokenized[0]["attention_mask"]))

print("Số nhãn:",
      len(train_tokenized[0]["labels"]))
print("\n===== KIỂM TRA LABEL =====")

print("Labels:", train_tokenized[0]["labels"])
print("Số phần tử:", len(train_tokenized[0]["labels"]))
print("Kiểu phần tử:", type(train_tokenized[0]["labels"][0]))
from transformers import AutoModelForSequenceClassification


# =========================
# 10. Tạo mô hình BERT
# =========================

emotion_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=28,
    problem_type="multi_label_classification"
)

print("\n===== MODEL =====")
print("Số nhãn:", emotion_model.config.num_labels)
print("Problem type:", emotion_model.config.problem_type)
import numpy as np
from sklearn.metrics import f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Chuyển logits thành xác suất
    probabilities = 1 / (1 + np.exp(-logits))

    # Ngưỡng 0.5:
    # >= 0.5 -> có cảm xúc
    # < 0.5  -> không có cảm xúc
    predictions = (probabilities >= 0.5).astype(int)

    # F1 trung bình
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


print("\n===== METRICS =====")
print("Đã tạo hàm đánh giá F1 multi-label")
from transformers import TrainingArguments, Trainer


training_args = TrainingArguments(
    output_dir="./models/bert_emotion",
    
    num_train_epochs=1,
    
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    
    learning_rate=2e-5,
    
    eval_strategy="epoch",
    save_strategy="epoch",
    
    logging_steps=100,
    
    report_to="none",
    
    use_cpu=True
)


trainer = Trainer(
    model=emotion_model,
    args=training_args,
    
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    
    compute_metrics=compute_metrics
)


print("\n===== TRAINER =====")
print("Trainer đã được cấu hình.")
print("\n===== BAT DAU TRAIN =====")

trainer.train()