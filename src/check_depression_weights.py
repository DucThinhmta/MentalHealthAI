import pandas as pd
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ==========================================
# 1. Đọc dữ liệu Train
# ==========================================

DATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "depression_train.csv"
    )
)

train = pd.read_csv(DATA_PATH)

# ==========================================
# 2. Tính class weight
# ==========================================

classes = np.array([0, 1, 2, 3])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train["label"]
)

# ==========================================
# 3. Hiển thị
# ==========================================

print("===================================")
print("CLASS WEIGHT DEPRESSION")
print("===================================")

for label, weight in zip(classes, weights):
    print(f"Class {label}: {weight:.4f}")

print("\nNhan:")
print("0 = minimum")
print("1 = mild")
print("2 = moderate")
print("3 = severe")