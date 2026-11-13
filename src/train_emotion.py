"""Huấn luyện BERT multi-label cho GoEmotions (28 nhãn) — thay cho prepare_emotion.py cũ
(file đó vừa chuẩn bị dữ liệu vừa train ngay khi import, khó tái sử dụng).

    python src/train_emotion.py --epochs 3          # CPU ~4h/epoch; nên dùng GPU (Colab)
    python src/train_emotion.py --max-train 2000    # chạy thử nhanh
Sau đó: python src/find_thresholds.py

Tùy chọn mới (mặc định giữ nguyên hành vi cũ):
    --base-model roberta-base     đổi mô hình nền (xem config.py)
    --pos-weight-cap 10           bù mất cân bằng nhãn: trọng số dương = (số âm / số dương), chặn trên ở mức này.
                                  Nhắm vào các nhãn hiếm đang có F1 = 0 (grief, pride, relief, embarrassment).
    --patience 2                  dừng sớm
    --bilingual                   thêm bản dịch máy (Anh<->Việt) của GoEmotions và các bộ trong --extra
    --extra all                   thêm các bộ cảm xúc bổ sung (fetch_datasets.py): uit_vsmec (tiếng Việt),
                                  dair_emotion, empathetic. Nên dùng cùng --base-model xlm-roberta-base để mô hình
                                  hiểu trực tiếp tiếng Việt (khỏi cần bước dịch).
"""
import argparse

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

import config
import modeling
from datasets_io import load_extra, load_split, load_translated, resolve_extra


class PosWeightTrainer(Trainer):
    """BCE có pos_weight theo từng nhãn (nhãn hiếm bị phạt nặng hơn khi bỏ sót)."""

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs.logits, labels.float(), pos_weight=self.pos_weight.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def pos_weights(label_matrix, cap):
    """(số mẫu âm / số mẫu dương) cho từng nhãn, chặn trong [1, cap]."""
    y = np.asarray(label_matrix, dtype=float)
    pos = y.sum(axis=0)
    return np.clip((len(y) - pos) / np.maximum(pos, 1.0), 1.0, cap)


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
    ap.add_argument("--base-model", default=config.BASE_MODEL)
    ap.add_argument("--pos-weight-cap", type=float, default=0.0, help="0 = tắt (BCE thường)")
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--bilingual", action="store_true", help="thêm bản dịch máy Anh<->Việt")
    ap.add_argument("--extra", default="", help="'all' hoặc danh sách tên, vd. uit_vsmec,empathetic")
    ap.add_argument("--out", default=str(config.PROJECT_DIR / "models" / "bert_emotion_v2"))
    a = ap.parse_args()
    set_seed(a.seed)

    train, dev = load_split("emotion", "train"), load_split("emotion", "dev")
    if a.max_train:
        train = train.sample(a.max_train, random_state=a.seed)
    extra = resolve_extra(a.extra)
    if extra:
        import pandas as pd
        train = pd.concat([train, load_extra("emotion", "train", extra)], ignore_index=True)
        dev = pd.concat([dev, load_extra("emotion", "dev", extra)], ignore_index=True)
        print(f"+ dữ liệu bổ sung {extra}: train={len(train)} dev={len(dev)}")
    if a.bilingual:
        import pandas as pd
        names = ["core"] + extra
        tr, dv = load_translated("emotion", "train", names), load_translated("emotion", "dev", names)
        if tr is None:
            raise SystemExit("Chưa có bản dịch. Chạy: python src/translate_datasets.py --engine local (hoặc claude)")
        train = pd.concat([train, tr], ignore_index=True)
        if dv is not None:
            dev = pd.concat([dev, dv], ignore_index=True)
        print(f"+ bản dịch máy: train={len(train)} dev={len(dev)}")
    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.base_model, num_labels=len(config.EMOTION_LABELS), problem_type="multi_label_classification")
    args = TrainingArguments(
        output_dir=a.out + "_ckpt", num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size * 2,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        warmup_ratio=0.1, weight_decay=0.01, fp16=torch.cuda.is_available(),
        logging_steps=100, report_to="none", seed=a.seed,
    )
    extra = {}
    if a.pos_weight_cap > 0:
        w = pos_weights(train["labels"].tolist(), a.pos_weight_cap)
        extra["pos_weight"] = torch.tensor(w, dtype=torch.float)
        print("pos_weight:", dict(zip(config.EMOTION_LABELS, np.round(w, 2))))
    if a.patience:
        from transformers import EarlyStoppingCallback
        extra["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=a.patience)]
    trainer_cls = PosWeightTrainer if a.pos_weight_cap > 0 else Trainer
    trainer = trainer_cls(model=model, args=args, **extra,
                      train_dataset=to_dataset(train, tokenizer, config.MAX_LEN["emotion"]),
                      eval_dataset=to_dataset(dev, tokenizer, config.MAX_LEN["emotion"]),
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    trainer.train()
    print("DEV:", trainer.evaluate())
    trainer.save_model(a.out)
    tokenizer.save_pretrained(a.out)
    modeling.write_meta(a.out, a.base_model, task="emotion", args=vars(a), extra_datasets=extra)
    print("Đã lưu:", a.out)


if __name__ == "__main__":
    main()
