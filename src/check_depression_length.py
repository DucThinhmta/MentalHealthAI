import pandas as pd
import os
from transformers import AutoTokenizer

# ==========================================
# 1. Đường dẫn dữ liệu
# ==========================================

DATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "depression_train.csv"
    )
)

# ==========================================
# 2. Đọc dữ liệu
# ==========================================

df = pd.read_csv(DATA_PATH)

print("===================================")
print("KIEM TRA DO DAI DEPRESSION DATASET")
print("===================================")

# ==========================================
# 3. Load tokenizer BERT
# ==========================================

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ==========================================
# 4. Tokenize không cắt ngắn
# ==========================================

token_lengths = []

for text in df["text"]:
    tokens = tokenizer(
        text,
        truncation=False,
        add_special_tokens=True
    )

    token_lengths.append(len(tokens["input_ids"]))

# ==========================================
# 5. Thống kê
# ==========================================

df["token_length"] = token_lengths

print("\nSo luong mau:", len(df))

print("Do dai ngan nhat:", df["token_length"].min())
print("Do dai dai nhat :", df["token_length"].max())
print("Do dai trung binh:", df["token_length"].mean())

# ==========================================
# 6. Kiểm tra số mẫu vượt quá 128 và 256
# ==========================================

over_128 = (df["token_length"] > 128).sum()
over_256 = (df["token_length"] > 256).sum()

print("\nSo mau > 128 tokens:", over_128)
print("Ty le > 128:", f"{over_128 / len(df) * 100:.2f}%")

print("\nSo mau > 256 tokens:", over_256)
print("Ty le > 256:", f"{over_256 / len(df) * 100:.2f}%")