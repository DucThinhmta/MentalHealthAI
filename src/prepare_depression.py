import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Đường dẫn dataset
# ==========================================

DATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "Reddit_depression_dataset.csv"
    )
)

# ==========================================
# 2. Đọc dữ liệu
# ==========================================

df = pd.read_csv(DATA_PATH)

print("===================================")
print("CHUAN BI DU LIEU DEPRESSION")
print("===================================")

print("\nKich thuoc ban dau:")
print(df.shape)

# ==========================================
# 3. Làm sạch text
# ==========================================

df["text"] = (
    df["text"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# ==========================================
# 4. Chuyển nhãn thành số
# ==========================================

label_mapping = {
    "minimum": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3
}

df["label"] = df["label"].map(label_mapping)

# ==========================================
# 5. Kiểm tra dữ liệu
# ==========================================

print("\nNhan sau khi ma hoa:")
print(df["label"].value_counts().sort_index())

print("\nSo luong gia tri thieu:")
print(df.isnull().sum())

# ==========================================
# 6. Chia Train / Test
# ==========================================

train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ==========================================
# 7. Chỉ giữ 2 cột cần thiết
# ==========================================

train = train[["text", "label"]]
test = test[["text", "label"]]

# ==========================================
# 8. Lưu dữ liệu
# ==========================================

PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")

train.to_csv(
    os.path.join(DATA_DIR, "depression_train.csv"),
    index=False,
    encoding="utf-8-sig"
)

test.to_csv(
    os.path.join(DATA_DIR, "depression_test.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 9. Hiển thị kết quả
# ==========================================

print("\n===================================")
print("KET QUA CHIA DU LIEU")
print("===================================")

print("Train:", train.shape)
print("Test :", test.shape)

print("\nPhan bo nhan Train:")
print(train["label"].value_counts().sort_index())

print("\nPhan bo nhan Test:")
print(test["label"].value_counts().sort_index())

print("\nDa luu:")
print("data/depression_train.csv")
print("data/depression_test.csv")