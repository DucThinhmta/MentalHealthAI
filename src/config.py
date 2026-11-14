"""Cấu hình tập trung: đường dẫn, nhãn, tham số. Sửa ở đây thay vì rải khắp các file."""
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent                 # .../MentalHealthAI (thư mục trong)
DATA_DIR = PROJECT_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"               # dữ liệu sau clean_data.py
STUDY_DIR = DATA_DIR / "study"               # hồ sơ nghiên cứu ẩn danh (study.py), nằm ngoài git
EXTRA_DIR = DATA_DIR / "extra"               # bộ dữ liệu bổ sung do fetch_datasets.py tải về
REPORT_DIR = PROJECT_DIR / "reports"

# Mô hình có thể nằm trong models/ của thư mục trong HOẶC thư mục ngoài (cấu trúc hiện tại của bạn).
MODEL_SEARCH_DIRS = [PROJECT_DIR / "models", PROJECT_DIR.parent / "models"]

# Mỗi nhánh liệt kê các ứng viên theo thứ tự ưu tiên (bản *_v2 do script train mới tạo ra).
MODEL_CANDIDATES = {
    "stress": [("bert_stress_v2",), ("bert_stress_final",)],
    "emotion": [("bert_emotion_v2",), ("bert_emotion", "checkpoint-5427")],
    "depression": [("bert_depression_v2",), ("bert_depression_final",)],
    "chatgen": [("chatgen_v2",)],               # tùy chọn: mô hình SINH câu trả lời (DialoGPT tinh chỉnh)
    "chitchat": [("bert_chitchat_v2",)],        # tùy chọn: bộ phân loại "trò chuyện thường ngày"
}

# Mô hình nền khi HUẤN LUYỆN. Đổi bằng biến môi trường MHAI_BASE_MODEL hoặc tham số --base-model, ví dụ:
#   roberta-base                     (tiếng Anh, thường nhỉnh hơn BERT)
#   mental/mental-roberta-base       (tiền huấn luyện trên văn bản sức khỏe tâm lý; cần xin quyền trên Hugging Face)
#   xlm-roberta-base                 (đa ngữ: phân tích trực tiếp tiếng Việt, không cần bước dịch)
# Checkpoint mới ghi lại mô hình nền trong mhai_meta.json để lúc suy luận dùng đúng tokenizer.
BASE_MODEL = os.getenv("MHAI_BASE_MODEL", "bert-base-uncased")
LEGACY_BASE_MODEL = "bert-base-uncased"      # các checkpoint cũ (không có mhai_meta.json)
MODEL_META_FILE = "mhai_meta.json"
MAX_LEN = {"stress": 256, "emotion": 128, "depression": 256, "chitchat": 64}
# Ước lượng ~1,25 token/từ => cửa sổ từ nhỏ hơn max_length để không bị cắt.
WINDOW_WORDS = {"stress": 170, "emotion": 85, "depression": 170, "chitchat": 40}
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

BOT_NAME = "MentalHealthAI"
# Ngôn ngữ của MỌI câu bot nói ra: "vi" (luôn tiếng Việt, mặc định) | "en" | "auto" (theo ngôn ngữ người dùng).
# Phần PHÂN TÍCH không phụ thuộc cài đặt này: tin nhắn tiếng Việt vẫn được dịch sang tiếng Anh để chạy các mô hình.
REPLY_LANGUAGE = os.getenv("MHAI_REPLY_LANG", "vi")
# Mức "quên" khi gộp nhiều câu: câu mới nhất có trọng số 1, câu liền trước = decay, trước nữa = decay^2...
RECENCY_DECAY = 0.4

# Đánh giá cả cuộc trò chuyện (assessment.py). Trọng số HEURISTIC, cần kiểm chứng trước khi dùng để kết luận.
CONCERN_WEIGHTS = {"stress": 0.25, "depression": 0.25, "emotion": 0.15, "domains": 0.25, "self_mood": 0.10}
CONCERN_THRESHOLDS = [0.3, 0.5, 0.7]          # -> 4 mức: ổn định / chút căng thẳng / đáng chú ý / nặng nề
DURATION_BONUS = 0.08

# Bộ phân loại trò chuyện thường ngày: chỉ bỏ qua phân tích khi rất chắc chắn (tránh bỏ sót người cần giúp).
#3CHITCHAT_THRESHOLD = 0.9

# Mô hình sinh hội thoại (tiếng Anh). Chỉ dùng cho tin nhắn thường ngày, không bao giờ dùng khi có tín hiệu khủng hoảng.
#CHATGEN_BASE = "microsoft/DialoGPT-small"
#CHATGEN_CONTEXT_TURNS = 4            # số lượt hội thoại gần nhất đưa vào làm ngữ cảnh
#CHATGEN_MAX_NEW_TOKENS = 60

DEFAULT_EMOTION_THRESHOLD = 0.25
EMOTION_THRESHOLD_FILE = PROJECT_DIR / "models" / "emotion_thresholds.json"
# Hiệu chỉnh xác suất (temperature) + ngưỡng thứ bậc cho Depression, do calibrate.py tạo từ tập dev.
CALIBRATION_FILE = PROJECT_DIR / "models" / "calibration.json"

# Tên hiển thị tiếng Việt: nhãn -> (biểu tượng, tên).
EMOTION_VI = {
    "admiration": ("🤩", "ngưỡng mộ"), "amusement": ("😄", "thích thú"), "anger": ("😠", "tức giận"),
    "annoyance": ("😤", "khó chịu"), "approval": ("👍", "tán thành"), "caring": ("🤗", "quan tâm"),
    "confusion": ("😕", "bối rối"), "curiosity": ("🧐", "tò mò"), "desire": ("✨", "mong muốn"),
    "disappointment": ("😞", "thất vọng"), "disapproval": ("👎", "không đồng tình"), "disgust": ("🤢", "chán ghét"),
    "embarrassment": ("😳", "ngượng ngùng"), "excitement": ("🎉", "hào hứng"), "fear": ("😨", "sợ hãi"),
    "gratitude": ("🙏", "biết ơn"), "grief": ("🖤", "đau buồn"), "joy": ("😊", "vui vẻ"), "love": ("❤️", "yêu thương"),
    "nervousness": ("😰", "lo lắng"), "optimism": ("🌤️", "lạc quan"), "pride": ("😌", "tự hào"),
    "realization": ("💡", "nhận ra"), "relief": ("😮‍💨", "nhẹ nhõm"), "remorse": ("😔", "hối hận"),
    "sadness": ("😢", "buồn bã"), "surprise": ("😮", "ngạc nhiên"), "neutral": ("😐", "trung tính"),
}
POSITIVE_EMOTIONS = {"admiration", "amusement", "approval", "caring", "excitement", "gratitude", "joy", "love",
                     "optimism", "pride", "relief"}
NEGATIVE_EMOTIONS = {"anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear",
                     "grief", "nervousness", "remorse", "sadness"}
DEPRESSION_VI = {"minimum": "Rất ít", "mild": "Nhẹ", "moderate": "Vừa", "severe": "Nặng"}

# --- Nguồn lực hỗ trợ hiển thị khi phát hiện nguy cơ ---
# Chỉ đưa vào danh sách HIỂN THỊ những kênh đã xác minh; mọi con số chưa xác minh nằm ở RESOURCES_TO_VERIFY
# (KHÔNG hiển thị cho người dùng). Hãy tự gọi thử/đối chiếu nguồn chính thức rồi mới chuyển sang danh sách hiển thị.
RESOURCES_CHECKED = "2026-11-14"
SUPPORT_RESOURCES = [
    "Nguy hiểm trực tiếp: gọi cấp cứu y tế 115 hoặc công an 113.",
    "Danh sách đường dây hỗ trợ tâm lý/phòng chống tự tử tại Việt Nam (cập nhật): "
    "https://findahelpline.com/countries/vn/topics/suicidal-thoughts",
    "Nhắn hoặc gọi ngay cho một người thân, bạn bè hay thầy cô mà bạn tin tưởng.",
]
SUPPORT_RESOURCES_EN = [
    "Immediate danger: in Vietnam call 115 (ambulance) or 113 (police), or your local emergency number.",
    "Helplines in Vietnam: https://findahelpline.com/countries/vn/topics/suicidal-thoughts",
    "Reach out right now to someone you trust: a relative, friend or teacher.",
]
# Dành cho người dưới 18 tuổi. 111: Tổng đài quốc gia bảo vệ trẻ em, miễn phí, 24/7 (đối chiếu qua nhiều báo chính thống
# ngày RESOURCES_CHECKED; CHƯA đối chiếu trang chính thức của Cục Trẻ em, nhóm nên làm bước này trước khi công bố).
SUPPORT_RESOURCES_MINOR = ["Nếu bạn dưới 18 tuổi: Tổng đài quốc gia bảo vệ trẻ em 111 (miễn phí, 24/7). "
                           "Bạn cũng nên nói với một người lớn bạn tin tưởng."]
SUPPORT_RESOURCES_MINOR_EN = ["If you are under 18: Vietnam's national child protection hotline 111 (free, 24/7). "
                              "Please also tell an adult you trust."]
# Số chỉ thấy trên các trang tổng hợp, chưa có nguồn chính thức => KHÔNG hiển thị cho người dùng.
RESOURCES_TO_VERIFY = {
    "1900 1267": "Bệnh viện Tâm thần TP.HCM (một số trang ghi là 'cấp cứu trầm cảm'): chưa xác minh vai trò, giờ làm việc",
    "(024) 3576 5344 / 0984 104 115": "Viện Sức khỏe Tâm thần Quốc gia: chưa xác minh với trang chính thức",
}
LOCAL_RESOURCES_FILE = PROJECT_DIR / "resources" / "local_resources.json"


def load_local_resources(path=None):
    """Danh sách hỗ trợ riêng của trường/địa phương do nhóm điền vào resources/local_resources.json.

    Chỉ nhận mục có 'name' và 'contact' không rỗng; thiếu tệp hoặc tệp hỏng -> danh sách rỗng."""
    import json
    try:
        data = json.loads(Path(path or LOCAL_RESOURCES_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = data.get("resources", []) if isinstance(data, dict) else []
    return [r for r in items if isinstance(r, dict) and str(r.get("name", "")).strip()
            and str(r.get("contact", "")).strip()]


def support_resources(minor=False, lang="vi"):
    """Các dòng hỗ trợ hiển thị: kênh chung + (nếu dưới 18 tuổi) 111 + liên hệ của trường/địa phương."""
    en = lang == "en"
    lines = list(SUPPORT_RESOURCES_EN if en else SUPPORT_RESOURCES)
    if minor:
        lines.insert(1, (SUPPORT_RESOURCES_MINOR_EN if en else SUPPORT_RESOURCES_MINOR)[0])
    for r in load_local_resources():
        extra = f" ({r['hours']})" if str(r.get("hours", "")).strip() else ""
        lines.append(f"{r['name']}: {r['contact']}{extra}")
    return lines


def archive_existing(path):
    """Không ghi đè: nếu `path` (tệp hoặc thư mục) đã có, chuyển nó vào `_archive/` cạnh đó kèm dấu thời gian.

    Trả về đường dẫn mới của bản cũ, hoặc None nếu chưa có gì để lưu trữ. Gọi hàm này TRƯỚC khi ghi tệp mới.
    """
    import shutil
    import time
    path = Path(path)
    if not path.exists():
        return None
    arch = path.parent / "_archive"
    arch.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = arch / (f"{path.stem}.{stamp}{path.suffix}" if path.is_file() else f"{path.name}.{stamp}")
    n = 1
    while dest.exists():                         # hai lần chạy trong cùng một giây
        dest = dest.with_name(dest.name + f".{n}"); n += 1
    shutil.move(str(path), str(dest))
    return dest


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
