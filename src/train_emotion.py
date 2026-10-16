"""Huấn luyện BERT multi-label cho GoEmotions (28 nhãn) — thay cho prepare_emotion.py cũ
(file đó vừa chuẩn bị dữ liệu vừa train ngay khi import, khó tái sử dụng).

    python src/train_emotion.py --epochs 3          # CPU ~4h/epoch; nên dùng GPU (Colab)
    python src/train_emotion.py --max-train 2000    # chạy thử nhanh
Sau đó: python src/find_thresholds.py
"""
import argparse

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

import config
from datasets_io import load_split


def compute_metrics(eval_pred, threshold=0.3):
    logits, labels = eval_pred
    pred = (1 / (1 + np.exp(-logits)) >= threshold).astype(int)
    return {"f1_micro": f1_score(labels, pred, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, pred, average="macro", zero_division=0)}


def to_dataset(df, tokenizer, max_length):
    ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": df["labels"].tolist()})
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
                  batched=True, remove_columns=["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(config.PROJECT_DIR / "models" / "bert_emotion_v2"))
    a = ap.parse_args()
    set_seed(a.seed)

    train, dev = load_split("emotion", "train"), load_split("emotion", "dev")
    if a.max_train:
        train = train.sample(a.max_train, random_state=a.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL, num_labels=len(config.EMOTION_LABELS), problem_type="multi_label_classification")
    args = TrainingArguments(
        output_dir=a.out + "_ckpt", num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size * 2,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        warmup_ratio=0.1, weight_decay=0.01, fp16=torch.cuda.is_available(),
        logging_steps=100, report_to="none", seed=a.seed,
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=to_dataset(train, tokenizer, config.MAX_LEN["emotion"]),
                      eval_dataset=to_dataset(dev, tokenizer, config.MAX_LEN["emotion"]),
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    trainer.train()
    print("DEV:", trainer.evaluate())
    trainer.save_model(a.out)
    tokenizer.save_pretrained(a.out)
    print("Đã lưu:", a.out)


if __name__ == "__main__":
    main()
