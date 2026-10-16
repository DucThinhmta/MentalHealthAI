"""Pipeline phân tích hội thoại: Stress + Emotion + Depression severity."""
import json

import numpy as np

import config
import modeling

_cache = {}


def _get(name):
    """Nạp mô hình một lần duy nhất (lazy) để import nhanh và dùng lại giữa các lượt."""
    if name not in _cache:
        _cache[name] = modeling.load(config.find_model(name))
    return _cache[name]


# ---------------------------------------------------------------- Stress
def stress_probs(text):
    tok, model = _get("stress")
    return modeling.predict_text(tok, model, text, config.MAX_LEN["stress"],
                                 config.WINDOW_WORDS["stress"])


def predict_stress(text):
    """-> (nhãn, độ tin cậy của lớp được chọn)."""
    p = stress_probs(text)
    k = int(p.argmax())
    return (config.STRESS_YES if k == 1 else config.STRESS_NO), float(p[k])


# ---------------------------------------------------------------- Emotion
def load_emotion_thresholds():
    """Ngưỡng theo từng cảm xúc nếu đã chạy find_thresholds.py, ngược lại dùng ngưỡng chung."""
    default = np.full(len(config.EMOTION_LABELS), config.DEFAULT_EMOTION_THRESHOLD)
    path = config.EMOTION_THRESHOLD_FILE
    if not path.exists():
        return default
    data = json.loads(path.read_text(encoding="utf-8"))
    glob = data.get("global", config.DEFAULT_EMOTION_THRESHOLD)
    per_class = data.get("per_class", {})
    return np.array([per_class.get(label, glob) for label in config.EMOTION_LABELS])


def predict_emotion(text, thresholds=None):
    """-> [(cảm xúc, điểm sigmoid)] vượt ngưỡng, sắp giảm dần."""
    tok, model = _get("emotion")
    probs = modeling.predict_text(tok, model, text, config.MAX_LEN["emotion"],
                                  config.WINDOW_WORDS["emotion"], multilabel=True, aggregate="max")
    thr = load_emotion_thresholds() if thresholds is None else thresholds
    found = [(config.EMOTION_LABELS[i], float(p)) for i, p in enumerate(probs) if p >= thr[i]]
    return sorted(found, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------- Depression
def depression_probs(text):
    tok, model = _get("depression")
    return modeling.predict_text(tok, model, text, config.MAX_LEN["depression"],
                                 config.WINDOW_WORDS["depression"])


def predict_depression(text):
    p = depression_probs(text)
    k = int(p.argmax())
    return config.DEPRESSION_LABELS[k], float(p[k])


# ---------------------------------------------------------------- Tổng hợp
def analyze_conversation(user_messages, window=5):
    """Phân tích hội thoại nhiều lượt.

    - Stress + Depression: dùng `window` tin nhắn gần nhất (văn bản dài được chia cửa sổ,
      nên không còn bị cắt mất phần cuối như khi ghép toàn bộ lịch sử).
    - Emotion: chỉ tin nhắn mới nhất (cảm xúc thay đổi theo từng câu).
    """
    messages = [m.strip() for m in user_messages if m and m.strip()]
    if not messages:
        raise ValueError("Cần ít nhất một tin nhắn không rỗng.")
    context = " ".join(messages[-window:])

    sp = stress_probs(context)
    dp = depression_probs(context)
    k_s, k_d = int(sp.argmax()), int(dp.argmax())
    return {
        "stress": config.STRESS_YES if k_s == 1 else config.STRESS_NO,
        "stress_confidence": float(sp[k_s]),
        "stress_probability": float(sp[1]),          # xác suất CÓ stress (dễ đọc trên giao diện)
        "emotions": predict_emotion(messages[-1]),
        "depression": config.DEPRESSION_LABELS[k_d],
        "depression_confidence": float(dp[k_d]),
        "depression_probs": {l: float(p) for l, p in zip(config.DEPRESSION_LABELS, dp)},
        "n_messages_used": min(window, len(messages)),
    }


def analyze(text):
    """Giữ tương thích với code cũ: phân tích một đoạn văn bản."""
    return analyze_conversation([text], window=1)


if __name__ == "__main__":
    out = analyze(input("Nhap cau can phan tich (tieng Anh): "))
    print(json.dumps(out, indent=2, ensure_ascii=False))
