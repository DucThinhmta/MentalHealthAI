import pandas as pd
import matplotlib.pyplot as plt
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# 1. ĐỌC DỮ LIỆU
# =========================

train = pd.read_csv("./data/train_clean.csv")
test = pd.read_csv("./data/test_clean.csv")


# =========================
# 2. TÁCH TEXT VÀ LABEL
# =========================

X_train_text = train["text_clean"]
y_train = train["label"]

X_test_text = test["text_clean"]
y_test = test["label"]


# =========================
# 3. TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# =========================
# 4. LOGISTIC REGRESSION
# =========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# =========================
# 5. DỰ ĐOÁN
# =========================

y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
from sklearn.svm import LinearSVC

print("\n===================================")
print("SVM - TF-IDF")
print("===================================")

svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("Accuracy:", svm_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        svm_pred,
        target_names=[
            "Khong stress",
            "Co dau hieu stress"
        ]
    )
)
# Confusion Matrix cho SVM
svm_cm = confusion_matrix(y_test, svm_pred)

print("\nConfusion Matrix - SVM:")
print(svm_cm)

plt.figure(figsize=(6, 5))
plt.imshow(svm_cm)

plt.title("Confusion Matrix - SVM")
plt.xlabel("Du doan")
plt.ylabel("Thuc te")

plt.xticks([0, 1], ["Khong stress", "Co stress"])
plt.yticks([0, 1], ["Khong stress", "Co stress"])

for i in range(2):
    for j in range(2):
        plt.text(
            j, i,
            svm_cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


# =========================
# 6. ĐÁNH GIÁ
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("===================================")
print("LOGISTIC REGRESSION - BASELINE")
print("===================================")

print("Accuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Khong stress",
            "Co dau hieu stress"
        ]
    )
)
print("\n===================================")
print("KIEM TRA MAX LENGTH = 256")
print("===================================")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

lengths = []

for text in train["text_clean"]:
    tokens = tokenizer(
        text,
        truncation=False
    )
    lengths.append(len(tokens["input_ids"]))

so_mau_cat = sum(x > 256 for x in lengths)

print("So mau > 256 token:", so_mau_cat)
print("Ty le bi cat:", so_mau_cat / len(lengths) * 100, "%")
print("\n===================================")
print("CHUAN BI DU LIEU CHO BERT")
print("===================================")

train_dataset = Dataset.from_pandas(
    train[["text_clean", "label"]]
)

test_dataset = Dataset.from_pandas(
    test[["text_clean", "label"]]
)

print("Train:", train_dataset)
print("Test:", test_dataset)
print("\n===================================")
print("TOKENIZE TOAN BO DU LIEU")
print("===================================")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text_clean"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True
)

test_tokenized = test_dataset.map(
    tokenize_function,
    batched=True
)

print("Train tokenized:", train_tokenized)
print("Test tokenized:", test_tokenized)
print("\n===================================")
print("KIEM TRA DU LIEU TOKENIZED")
print("===================================")

print("Do dai input_ids:",
      len(train_tokenized[0]["input_ids"]))

print("Do dai attention_mask:",
      len(train_tokenized[0]["attention_mask"]))

print("Label mau dau tien:",
      train_tokenized[0]["label"])
print("\n===================================")
print("TAO MO HINH BERT")
print("===================================")

bert_model = AutoModelForSequenceClassification.from_pretrained(
    "./models/bert_stress/checkpoint-1419"
)

print("Da load BERT da train thanh cong")

print("Da tao mo hinh BERT thanh cong")
print("\n===================================")
print("CAU HINH TRAINER")
print("===================================")

training_args = TrainingArguments(
    output_dir="./models/bert_stress",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="none",
    use_cpu=True
)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized
)

print("Da cau hinh Trainer thanh cong")
#print("\n===================================")
#print("BAT DAU TRAIN BERT")
#print("===================================")

#trainer.train()

#print("TRAIN BERT HOAN TAT")
print("\n===================================")
print("DANH GIA BERT")
print("===================================")

results = trainer.evaluate()

print(results)
print("\n===================================")
print("DU DOAN TAP TEST BANG BERT")
print("===================================")

predictions = trainer.predict(test_tokenized)

y_pred_bert = predictions.predictions.argmax(axis=-1)
y_true_bert = predictions.label_ids

print("Da du doan xong!")
print("\n===================================")
print("KET QUA BERT")
print("===================================")

print("Accuracy:", accuracy_score(y_true_bert, y_pred_bert))

print("\nClassification Report:")
print(
    classification_report(
        y_true_bert,
        y_pred_bert,
        target_names=[
            "Khong stress",
            "Co dau hieu stress"
        ]
    )
)
print("\n===================================")
print("CONFUSION MATRIX - BERT")
print("===================================")

bert_cm = confusion_matrix(y_true_bert, y_pred_bert)

print(bert_cm)
plt.figure(figsize=(6, 5))
plt.imshow(bert_cm)

plt.title("Confusion Matrix - BERT")
plt.xlabel("Du doan")
plt.ylabel("Thuc te")

plt.xticks(
    [0, 1],
    ["Khong stress", "Co stress"]
)

plt.yticks(
    [0, 1],
    ["Khong stress", "Co stress"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            bert_cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()
# =========================
# SO SANH 3 MO HINH
# =========================

print("\n===================================")
print("SO SANH 3 MO HINH")
print("===================================")

models = [
    "Logistic Regression",
    "SVM",
    "BERT"
]

accuracies = [
    accuracy,
    svm_accuracy,
    accuracy_score(y_true_bert, y_pred_bert)
]

for name, acc in zip(models, accuracies):
    print(name, ":", acc)

plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.title("So sanh Accuracy cua 3 mo hinh")
plt.xlabel("Mo hinh")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

for i, acc in enumerate(accuracies):
    plt.text(
        i,
        acc + 0.02,
        f"{acc * 100:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()
# =========================
# LUU MODEL BERT
# =========================

print("\n===================================")
print("LUU MODEL BERT")
print("===================================")

trainer.save_model("./models/bert_stress_final")
tokenizer.save_pretrained("./models/bert_stress_final")

print("Da luu model BERT thanh cong!")
# =========================
# DU DOAN CAU MOI BANG BERT
# =========================

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict_stress(text):
    # Load model BERT đã train
    model = AutoModelForSequenceClassification.from_pretrained(
        "./models/bert_stress_final"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "./models/bert_stress_final"
    )

    # Tokenize câu nhập vào
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    # Không tính gradient vì chỉ dự đoán
    with torch.no_grad():
        outputs = model(**inputs)

    # Lấy xác suất
    probabilities = torch.softmax(outputs.logits, dim=-1)

    # Lấy class có xác suất cao nhất
    prediction = torch.argmax(probabilities, dim=-1).item()

    confidence = probabilities[0][prediction].item()

    if prediction == 0:
        label = "Khong stress"
    else:
        label = "Co dau hieu stress"

    return label, confidence
print("\n===================================")
print("THU DU DOAN CAU MOI")
print("===================================")

text = input("Nhap cau: ")

label, confidence = predict_stress(text)

print("Ket qua:", label)
print("Do tin cay:", f"{confidence * 100:.2f}%")