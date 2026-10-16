import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

print("===================================")
print("TOKENIZE DEPRESSION DATASET")
print("===================================")

# ==========================================
# 1. Đọc dữ liệu
# ==========================================

train = pd.read_csv("./MentalHealthAI/data/depression_train.csv")
test = pd.read_csv("./MentalHealthAI/data/depression_test.csv")

print("\nTrain:", train.shape)
print("Test :", test.shape)

# ==========================================
# 2. Chuyển Pandas -> Hugging Face Dataset
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
# 3. Load BERT tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

# ==========================================
# 4. Hàm tokenize
# ==========================================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

# ==========================================
# 5. Tokenize Train và Test
# ==========================================

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

# ==========================================
# 6. Đổi tên label -> labels
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
# 7. Chỉ giữ các cột BERT cần
# ==========================================

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# ==========================================
# 8. Kiểm tra
# ==========================================

print("\n===================================")
print("KET QUA")
print("===================================")

print("Train:", train_dataset)
print("Test :", test_dataset)

print("\nMau dau tien:")
print(train_dataset[0])

print("\nLabels Train:")
print(train_dataset["labels"][:10])

print("\nHoan tat tokenize!")