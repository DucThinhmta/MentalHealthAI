import pandas as pd
import os

# Đường dẫn tới file dataset
DATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "Reddit_depression_dataset.csv"
    )
)

# Đọc dữ liệu
df = pd.read_csv(DATA_PATH)

print("===================================")
print("KIEM TRA DEPRESSION DATASET")
print("===================================")

# 1. Kích thước dataset
print("\nKich thuoc dataset:")
print(df.shape)

# 2. Tên các cột
print("\nTen cac cot:")
print(df.columns.tolist())

# 3. 5 dòng đầu tiên
print("\n5 dong dau tien:")
print(df.head())

# 4. Số lượng giá trị của từng nhãn
print("\nSo luong tung nhan:")
print(df["label"].value_counts())

# 5. Kiểm tra dữ liệu thiếu
print("\nSo luong gia tri thieu:")
print(df.isnull().sum())