"""Nạp mô hình và suy luận theo lô, có xử lý văn bản dài bằng cửa sổ trượt."""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from textutils import split_windows


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def base_model_of(path) -> str:
    """Mô hình nền của một checkpoint: đọc mhai_meta.json (script train mới ghi), không có thì là BERT cũ."""
    meta = Path(path) / config.MODEL_META_FILE
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get("base_model", config.LEGACY_BASE_MODEL)
    return config.LEGACY_BASE_MODEL


def write_meta(path, base_model, **extra):
    """Ghi lại mô hình nền cạnh checkpoint để lúc suy luận chọn đúng tokenizer."""
    (Path(path) / config.MODEL_META_FILE).write_text(
        json.dumps({"base_model": base_model, **extra}, indent=2, default=str), encoding="utf-8")


def load(path):
    """Nạp (tokenizer, model).

    Luôn dùng tokenizer GỐC của mô hình nền (bert-base-uncased với checkpoint cũ, hoặc mô hình ghi trong
    mhai_meta.json). Tokenizer lưu trong checkpoint Emotion cũ tách từ sai, làm mô hình cho cùng một đầu ra
    với mọi câu (micro F1 0,30 so với 0,58). Chỉ khi không nạp được tokenizer gốc
    (máy offline, chưa tải) mới dùng tokenizer trong checkpoint.
    """
    path = str(path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_of(path))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    return tokenizer, model.to(get_device()).eval()


def predict_probs(tokenizer, model, texts, max_length, multilabel=False, batch_size=32):
    """Xác suất cho danh sách văn bản -> mảng (n, số_nhãn). softmax hoặc sigmoid (multi-label)."""
    device = get_device()
    outputs = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = [t if t else " " for t in texts[i:i + batch_size]]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                            padding=True, max_length=max_length).to(device)
            logits = model(**enc).logits
            probs = torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=-1)
            outputs.append(probs.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def predict_text(tokenizer, model, text, max_length, window_words,
                 multilabel=False, aggregate="mean"):
    """Dự đoán cho MỘT văn bản bất kỳ độ dài: chia cửa sổ rồi gộp kết quả.

    aggregate='mean' (stress, depression) hoặc 'max' (emotion: có ở bất kỳ đoạn nào là tính).
    """
    step = max(1, int(window_words * config.WINDOW_STEP_RATIO))
    windows = split_windows(text, window_words, step)
    probs = predict_probs(tokenizer, model, windows, max_length, multilabel)
    return probs.max(axis=0) if aggregate == "max" else probs.mean(axis=0)