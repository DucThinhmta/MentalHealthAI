"""Cấu hình tập trung: đường dẫn, nhãn, tham số. Sửa ở đây thay vì rải khắp các file."""
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent                 # .../MentalHealthAI (thư mục trong)
DATA_DIR = PROJECT_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"               # dữ liệu sau clean_data.py
REPORT_DIR = PROJECT_DIR / "reports"

# Mô hình có thể nằm trong models/ của thư mục trong HOẶC thư mục ngoài (cấu trúc hiện tại của bạn).
MODEL_SEARCH_DIRS = [PROJECT_DIR / "models", PROJECT_DIR.parent / "models"]

# Mỗi nhánh liệt kê các ứng viên theo thứ tự ưu tiên (bản *_v2 do script train mới tạo ra).
MODEL_CANDIDATES = {
    "stress": [("bert_stress_v2",), ("bert_stress_final",)],
    "emotion": [("bert_emotion_v2",), ("bert_emotion", "checkpoint-5427")],
    "depression": [("bert_depression_v2",), ("bert_depression_final",)],
}

BASE_MODEL = "bert-base-uncased"
MAX_LEN = {"stress": 256, "emotion": 128, "depression": 256}
# Ước lượng ~1,25 token/từ => cửa sổ từ nhỏ hơn max_length để không bị cắt.
WINDOW_WORDS = {"stress": 170, "emotion": 85, "depression": 170}
WINDOW_STEP_RATIO = 0.8                      # bước trượt = 80% cửa sổ (chồng lấn 20%)

STRESS_NO = "Khong stress"
STRESS_YES = "Co dau hieu stress"
DEPRESSION_LABELS = ["minimum", "mild", "moderate", "severe"]
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
# Các cảm xúc liên quan nhất tới sức khỏe tâm lý (dùng khi vẽ biểu đồ báo cáo).
MENTAL_HEALTH_EMOTIONS = ["fear", "nervousness", "sadness", "grief", "disappointment", "remorse", "anger"]

DEFAULT_EMOTION_THRESHOLD = 0.25
EMOTION_THRESHOLD_FILE = PROJECT_DIR / "models" / "emotion_thresholds.json"

# Nguồn lực hỗ trợ hiển thị khi phát hiện nguy cơ. HÃY XÁC MINH LẠI trước khi demo/nộp.
SUPPORT_RESOURCES = [
    "Nguy hiểm trực tiếp: gọi cấp cứu y tế 115 hoặc công an 113.",
    "Danh sách đường dây hỗ trợ tâm lý/phòng chống tự tử tại Việt Nam (cập nhật): "
    "https://findahelpline.com/countries/vn/topics/suicidal-thoughts",
    "Nhắn hoặc gọi ngay cho một người thân, bạn bè hay thầy cô mà bạn tin tưởng.",
]


def find_model(name: str) -> Path:
    """Tìm thư mục mô hình theo thứ tự ưu tiên; báo lỗi rõ ràng nếu không thấy."""
    tried = []
    for candidate in MODEL_CANDIDATES[name]:
        for base in MODEL_SEARCH_DIRS:
            path = base.joinpath(*candidate)
            tried.append(str(path))
            if path.exists():
                return path
    raise FileNotFoundError(f"Không tìm thấy mô hình '{name}'. Đã thử:\n  " + "\n  ".join(tried))
