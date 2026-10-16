"""Huấn luyện BERT cho Stress (2 lớp) hoặc Depression (4 lớp) với tập dev và chọn epoch tốt nhất.

    python src/clean_data.py
    python src/train_classifier.py --task depression --epochs 4      # nên chạy trên GPU (Colab)
    python src/train_classifier.py --task stress --epochs 3

Cải tiến so với bản cũ: nhiều epoch, tập dev riêng (không dùng test để chọn mô hình),
class weight tự tính, padding động (nhanh hơn), tự dùng GPU nếu có, lưu cả báo cáo huấn luyện.
"""
import argparse
import json

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

import config
from datasets_io import load_split

NUM_LABELS = {"stress": 2, "depression": 4}


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = None if self.class_weights is None else self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    p, r, f, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, pred), "precision_macro": p, "recall_macro": r, "f1_macro": f}


def to_dataset(df, tokenizer, max_length):
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    ds = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
                batched=True, remove_columns=["text"])
    return ds.rename_column("label", "labels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=NUM_LABELS, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--out", default=None, help="mặc định: models/bert_<task>_v2")
    a = ap.parse_args()
    set_seed(a.seed)

    max_length = config.MAX_LEN[a.task]
    out_dir = a.out or str(config.PROJECT_DIR / "models" / f"bert_{a.task}_v2")
    train, dev = load_split(a.task, "train"), load_split(a.task, "dev")
    print(f"train={len(train)} dev={len(dev)} | GPU: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.BASE_MODEL, num_labels=NUM_LABELS[a.task])

    weights = None
    if a.task == "depression" and not a.no_class_weights:
        w = compute_class_weight("balanced", classes=np.arange(4), y=train["label"])
        weights = torch.tensor(w, dtype=torch.float)
        print("class weights:", np.round(w, 4))

    args = TrainingArguments(
        output_dir=out_dir + "_ckpt", num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size * 2,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        warmup_ratio=0.1, weight_decay=0.01, fp16=torch.cuda.is_available(),
        logging_steps=50, report_to="none", seed=a.seed,
    )
    trainer = WeightedTrainer(
        model=model, args=args, class_weights=weights,
        train_dataset=to_dataset(train, tokenizer, max_length),
        eval_dataset=to_dataset(dev, tokenizer, max_length),
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
    )
    trainer.train()
    dev_metrics = trainer.evaluate()
    print("DEV:", dev_metrics)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    (config.PROJECT_DIR / "reports").mkdir(exist_ok=True)
    with open(config.PROJECT_DIR / "reports" / f"train_{a.task}.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "dev": dev_metrics, "log": trainer.state.log_history}, f, indent=2, default=float)
    print("Đã lưu:", out_dir, "\nĐánh giá trên test: python src/evaluate_all.py --task", a.task)


if __name__ == "__main__":
    main()
