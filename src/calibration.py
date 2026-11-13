"""Hậu xử lý xác suất (chỉ cần numpy): hiệu chỉnh nhiệt độ và giải mã thứ bậc cho Depression.

Hai vấn đề của mô hình hiện tại mà phần này xử lý KHÔNG cần huấn luyện lại:
1. Quá tự tin: Stress gần như chỉ cho ra 0,99 hoặc 0,01 dù độ chính xác ~79%. Temperature scaling
   (Guo et al., 2017) chia log-xác suất cho T > 1 để con số hiển thị phản ánh đúng độ chắc chắn hơn;
   nó KHÔNG đổi lớp được chọn (argmax) nên không làm thay đổi accuracy/F1.
2. Depression là nhãn có thứ bậc (minimum < mild < moderate < severe) nhưng argmax coi 4 lớp như không liên
   quan và gần như không bao giờ chọn 'mild' (F1 = 0). Thay vào đó tính điểm kỳ vọng s = Σ k·p_k ∈ [0, 3]
   rồi cắt bằng 3 ngưỡng chọn trên tập dev.

Các tham số được calibrate.py ước lượng trên tập DEV và lưu vào models/calibration.json.
"""
import json

import numpy as np

import config


def apply_temperature(probs, temperature):
    """softmax(log p / T). T = 1 giữ nguyên; T > 1 làm phân phối bớt cực đoan."""
    probs = np.asarray(probs, dtype=float)
    if temperature is None or abs(temperature - 1.0) < 1e-9:
        return probs
    logp = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logp -= logp.max(axis=-1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=-1, keepdims=True)


def nll(probs, y):
    probs = np.asarray(probs, dtype=float)
    return float(-np.log(np.clip(probs[np.arange(len(y)), np.asarray(y)], 1e-12, 1.0)).mean())


def fit_temperature(probs, y, grid=None):
    """T tối thiểu hóa negative log-likelihood trên tập dev (tìm trên lưới log, đủ cho bài toán 1 tham số)."""
    grid = np.exp(np.linspace(np.log(0.5), np.log(20.0), 120)) if grid is None else grid
    return float(min(grid, key=lambda t: nll(apply_temperature(probs, t), y)))


def expected_calibration_error(probs, y, n_bins=10):
    """ECE: chênh lệch trung bình giữa độ tự tin và độ chính xác thực tế (càng thấp càng tốt)."""
    probs, y = np.asarray(probs, dtype=float), np.asarray(y)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean())
    return float(ece)


def expected_level(probs):
    """Điểm thứ bậc kỳ vọng Σ k·p_k (0 = minimum ... K-1 = severe)."""
    probs = np.asarray(probs, dtype=float)
    return probs @ np.arange(probs.shape[-1])


def decode_ordinal(score, cuts):
    """Điểm kỳ vọng -> chỉ số lớp theo các ngưỡng tăng dần."""
    return np.searchsorted(np.asarray(cuts, dtype=float), score, side="right")


def tune_ordinal_cuts(y, score, n_classes=4, grid=None, rounds=3):
    """Chọn K-1 ngưỡng tối đa hóa macro-F1 trên dev bằng tìm kiếm tọa độ (mỗi lần tối ưu một ngưỡng)."""
    from sklearn.metrics import f1_score
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    grid = np.round(np.arange(0.05, n_classes - 1, 0.05), 3) if grid is None else grid
    labels = list(range(n_classes))

    def f1(cuts):
        return f1_score(y, decode_ordinal(score, cuts), labels=labels, average="macro", zero_division=0)

    cuts = [k + 0.5 for k in range(n_classes - 1)]            # khởi tạo: làm tròn điểm kỳ vọng
    best = f1(cuts)
    for _ in range(rounds):
        for i in range(len(cuts)):
            lo = cuts[i - 1] if i > 0 else -np.inf
            hi = cuts[i + 1] if i + 1 < len(cuts) else np.inf
            for c in grid:
                if not lo < c < hi:
                    continue
                trial = cuts[:i] + [float(c)] + cuts[i + 1:]
                value = f1(trial)
                if value > best + 1e-12:
                    best, cuts = value, trial
    return [float(c) for c in cuts], float(best)


def load(path=None):
    """Đọc models/calibration.json; trả {} nếu chưa chạy calibrate.py (khi đó hành vi giống bản cũ)."""
    path = config.CALIBRATION_FILE if path is None else path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
