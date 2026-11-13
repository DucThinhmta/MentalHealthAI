"""Huấn luyện BERT cho Stress (2 lớp) hoặc Depression (4 lớp) với tập dev và chọn epoch tốt nhất.

    python src/clean_data.py
    python src/train_classifier.py --task depression --epochs 4      # nên chạy trên GPU (Colab)
    python src/train_classifier.py --task stress --epochs 3

Cải tiến so với bản cũ: nhiều epoch, tập dev riêng (không dùng test để chọn mô hình),
class weight tự tính, padding động (nhanh hơn), tự dùng GPU nếu có, lưu cả báo cáo huấn luyện.

Tùy chọn mới (mặc định giữ nguyên hành vi cũ, để kết quả cũ vẫn tái lập được):
    --base-model roberta-base     đổi mô hình nền (xem gợi ý trong config.py; xlm-roberta-base cho tiếng Việt)
    --loss focal --gamma 2        focal loss: tập trung vào mẫu khó, giúp lớp hiếm (mild, severe)
    --weight-power 0.5            làm dịu class weight (1 = balanced, 0.5 = căn bậc hai, 0 = tắt)
    --label-smoothing 0.05        giảm quá tự tin (nhãn Reddit vốn nhiễu)
    --metric qwk                  chọn epoch theo Quadratic Weighted Kappa (hợp với nhãn có thứ bậc)
    --patience 2                  dừng sớm khi chỉ số dev không tăng
    --bilingual                   thêm bản dịch tiếng Việt của train/dev (translate_datasets.py); nên dùng với
                                  --base-model xlm-roberta-base để mô hình hiểu trực tiếp tiếng Việt
Ví dụ cho Depression (một dòng):
    python src/train_classifier.py --task depression --base-model roberta-base --loss focal --weight-power 0.5
        --label-smoothing 0.05 --metric qwk --epochs 6 --patience 2
Mỗi cấu hình nên chạy với vài --seed khác nhau và báo cáo trung bình ± độ lệch chuẩn.
"""
import argparse
import json

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

import config
import modeling
from datasets_io import load_split, load_translated

NUM_LABELS = {"stress": 2, "depression": 4}


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, loss="ce", gamma=2.0, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss, self.gamma, self.label_smoothing = loss, gamma, label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = None if self.class_weights is None else self.class_weights.to(outputs.logits.device)
        loss = classification_loss(outputs.logits, labels, weight, self.loss, self.gamma, self.label_smoothing)
        return (loss, outputs) if return_outputs else loss


def classification_loss(logits, labels, weight=None, kind="ce", gamma=2.0, label_smoothing=0.0):
    """Cross-entropy (có trọng số lớp, label smoothing) hoặc focal loss (Lin et al., 2017)."""
    if kind == "ce":
        return torch.nn.functional.cross_entropy(logits, labels, weight=weight, label_smoothing=label_smoothing)
    per_sample = torch.nn.functional.cross_entropy(logits, labels, weight=weight, reduction="none",
                                                   label_smoothing=label_smoothing)
    p_true = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    return ((1 - p_true) ** gamma * per_sample).mean()


def class_weights(labels, n_classes, power=1.0):
    """Trọng số 'balanced' nâng lên lũy thừa `power` (power=1 giống hệt bản cũ)."""
    return compute_class_weight("balanced", classes=np.arange(n_classes), y=labels) ** power


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    p, r, f, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, pred), "precision_macro": p, "recall_macro": r, "f1_macro": f,
            "qwk": cohen_kappa_score(labels, pred, weights="quadratic")}


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
    ap.add_argument("--base-model", default=config.BASE_MODEL)
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--gamma", type=float, default=2.0, help="chỉ dùng với --loss focal")
    ap.add_argument("--weight-power", type=float, default=1.0, help="1 = balanced, 0.5 = căn bậc hai, 0 = tắt")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--metric", choices=["f1_macro", "qwk", "accuracy"], default="f1_macro")
    ap.add_argument("--patience", type=int, default=0, help="dừng sớm sau N epoch không cải thiện (0 = tắt)")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--bilingual", action="store_true", help="thêm bản dịch máy tiếng Việt của train/dev")
    ap.add_argument("--out", default=None, help="mặc định: models/bert_<task>_v2")
    a = ap.parse_args()
    set_seed(a.seed)

    max_length = a.max_length or config.MAX_LEN[a.task]
    out_dir = a.out or str(config.PROJECT_DIR / "models" / f"bert_{a.task}_v2")
    train, dev = load_split(a.task, "train"), load_split(a.task, "dev")
    if a.bilingual:
        import pandas as pd
        tr, dv = load_translated(a.task, "train", lang="vi"), load_translated(a.task, "dev", lang="vi")
        if tr is None:
            raise SystemExit("Chưa có bản dịch. Chạy: python src/translate_datasets.py --engine local (hoặc claude)")
        train = pd.concat([train, tr[["text", "label"]]], ignore_index=True)
        if dv is not None:
            dev = pd.concat([dev, dv[["text", "label"]]], ignore_index=True)
        print(f"+ bản dịch tiếng Việt: train={len(train)} dev={len(dev)}")
    print(f"train={len(train)} dev={len(dev)} | GPU: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(a.base_model, num_labels=NUM_LABELS[a.task])

    weights = None
    if a.task == "depression" and not a.no_class_weights and a.weight_power > 0:
        w = class_weights(train["label"], 4, a.weight_power)
        weights = torch.tensor(w, dtype=torch.float)
        print("class weights:", np.round(w, 4))

    args = TrainingArguments(
        output_dir=out_dir + "_ckpt", num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size * 2,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model=a.metric, greater_is_better=True,
        warmup_ratio=0.1, weight_decay=0.01, fp16=torch.cuda.is_available(),
        logging_steps=50, report_to="none", seed=a.seed,
    )
    trainer = WeightedTrainer(
        model=model, args=args, class_weights=weights, loss=a.loss, gamma=a.gamma,
        label_smoothing=a.label_smoothing,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=a.patience)] if a.patience else None,
        train_dataset=to_dataset(train, tokenizer, max_length),
        eval_dataset=to_dataset(dev, tokenizer, max_length),
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
    )
    trainer.train()
    dev_metrics = trainer.evaluate()
    print("DEV:", dev_metrics)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    modeling.write_meta(out_dir, a.base_model, task=a.task, args=vars(a))
    (config.PROJECT_DIR / "reports").mkdir(exist_ok=True)
    with open(config.PROJECT_DIR / "reports" / f"train_{a.task}.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "dev": dev_metrics, "log": trainer.state.log_history}, f, indent=2, default=float)
    print("Đã lưu:", out_dir, "\nĐánh giá trên test: python src/evaluate_all.py --task", a.task)


if __name__ == "__main__":
    main()
