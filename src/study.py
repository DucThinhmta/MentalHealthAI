"""Chế độ nghiên cứu: thang đo chuẩn (PHQ-9, GAD-7), chấm điểm, và hồ sơ ẩn danh để đối chiếu với điểm của bot.

Mục đích: trả lời câu hỏi khoa học "điểm của bot có đo đúng thứ nó nói là đo không?" bằng cách cho người tham gia (từ 18 tuổi,
đã đồng ý) trò chuyện với bot rồi điền thang đo đã được kiểm định, sau đó so sánh (xem analyze_study.py và
docs/nghien_cuu/PROTOCOL.md).

LƯU Ý QUAN TRỌNG
- Nội dung tiếng Anh của PHQ-9 và GAD-7 là các thang công khai, được phép dùng miễn phí. Bản tiếng Việt ở đây (cột 'vi') là
  BẢN NHÁP DO NHÓM TỰ DỊCH, CHƯA KIỂM ĐỊNH: phải thay bằng bản tiếng Việt đã được kiểm định (và ghi nguồn) trước khi thu dữ liệu
  thật. Ứng dụng hiển thị cảnh báo này khi dùng bản nháp.
- Câu 9 của PHQ-9 hỏi về ý nghĩ tự hại: nếu khác 0 thì PHẢI hiện ngay kênh hỗ trợ (item9_positive).
- Hồ sơ KHÔNG chứa nội dung hội thoại; chỉ chứa mã ngẫu nhiên, điểm của bot, và câu trả lời thang đo.
"""
import datetime
import json
from pathlib import Path

import config

SCHEMA_VERSION = 1
INSTRUMENT_VI_STATUS = "DRAFT_NOT_VALIDATED"          # đổi thành nguồn bản đã kiểm định (vd. 'Nguyen et al., 20xx') khi thay bản dịch
VI_WARNING = ("Bản tiếng Việt của bảng hỏi này là bản nháp do nhóm tự dịch, CHƯA được kiểm định. Chỉ dùng để thử giao diện; "
              "cần thay bằng bản đã kiểm định trước khi thu dữ liệu thật.")
OPTIONS = [(0, "Hoàn toàn không", "Not at all"), (1, "Vài ngày", "Several days"),
           (2, "Hơn một nửa số ngày", "More than half the days"), (3, "Gần như mỗi ngày", "Nearly every day")]

INSTRUMENTS = {
    "phq9": {
        "name": "PHQ-9", "max": 27, "cutoff": 10,
        "stem": ("Trong 2 tuần qua, bạn có thường bị làm phiền bởi những vấn đề sau không?",
                 "Over the last 2 weeks, how often have you been bothered by any of the following problems?"),
        "bands": [(0, 4, "Tối thiểu"), (5, 9, "Nhẹ"), (10, 14, "Vừa"), (15, 19, "Khá nặng"), (20, 27, "Nặng")],
        "items": [
            ("Ít hứng thú hay niềm vui khi làm mọi việc", "Little interest or pleasure in doing things"),
            ("Cảm thấy buồn, chán nản hoặc tuyệt vọng", "Feeling down, depressed, or hopeless"),
            ("Khó ngủ, ngủ không yên hoặc ngủ quá nhiều", "Trouble falling or staying asleep, or sleeping too much"),
            ("Cảm thấy mệt hoặc ít năng lượng", "Feeling tired or having little energy"),
            ("Ăn không ngon hoặc ăn quá nhiều", "Poor appetite or overeating"),
            ("Cảm thấy tệ về bản thân, thấy mình thất bại hoặc làm bản thân/gia đình thất vọng",
             "Feeling bad about yourself, or that you are a failure or have let yourself or your family down"),
            ("Khó tập trung vào việc như đọc báo hay xem tivi",
             "Trouble concentrating on things, such as reading the newspaper or watching television"),
            ("Cử động hay nói chậm đến mức người khác nhận ra, hoặc ngược lại bồn chồn, đứng ngồi không yên hơn thường lệ",
             "Moving or speaking so slowly that other people could have noticed, or the opposite: being so fidgety or "
             "restless that you have been moving around a lot more than usual"),
            ("Có ý nghĩ rằng chết đi thì tốt hơn, hoặc muốn làm đau bản thân theo cách nào đó",
             "Thoughts that you would be better off dead, or of hurting yourself in some way"),
        ],
    },
    "gad7": {
        "name": "GAD-7", "max": 21, "cutoff": 10,
        "stem": ("Trong 2 tuần qua, bạn có thường bị làm phiền bởi những vấn đề sau không?",
                 "Over the last 2 weeks, how often have you been bothered by the following problems?"),
        "bands": [(0, 4, "Tối thiểu"), (5, 9, "Nhẹ"), (10, 14, "Vừa"), (15, 21, "Nặng")],
        "items": [
            ("Cảm thấy lo lắng, bồn chồn hoặc căng thẳng", "Feeling nervous, anxious or on edge"),
            ("Không thể ngừng hoặc kiểm soát được việc lo lắng", "Not being able to stop or control worrying"),
            ("Lo lắng quá nhiều về nhiều việc khác nhau", "Worrying too much about different things"),
            ("Khó thư giãn", "Trouble relaxing"),
            ("Bồn chồn đến mức khó ngồi yên", "Being so restless that it is hard to sit still"),
            ("Dễ bực mình hoặc cáu kỉnh", "Becoming easily annoyed or irritable"),
            ("Cảm thấy sợ hãi như thể có điều gì tồi tệ sắp xảy ra", "Feeling afraid, as if something awful might happen"),
        ],
    },
}
PHQ9_SUICIDE_ITEM = 8                                   # chỉ số (0-based) của câu 9
# Khía cạnh sinh hoạt của bot <-> câu PHQ-9 tương ứng (chỉ số 0-based), dùng để kiểm chứng ở mức triệu chứng.
DOMAIN_TO_PHQ9_ITEM = {"interest": 0, "mood": 1, "sleep": 2, "energy": 3, "appetite": 4, "focus": 6}
ITEM_POSITIVE = 2                                       # "hơn một nửa số ngày" trở lên tính là có triệu chứng
MIN_TURNS = 5                                           # số lượt tối thiểu trước khi mời điền thang đo


def _inst(instrument):
    if instrument not in INSTRUMENTS:
        raise ValueError(f"thang đo không có: {instrument}")
    return INSTRUMENTS[instrument]


def score(instrument, answers):
    """(tổng điểm, chỉ số mức độ, tên mức). answers: danh sách số nguyên 0..3 đủ số câu."""
    inst = _inst(instrument)
    if len(answers) != len(inst["items"]) or any(a not in (0, 1, 2, 3) for a in answers):
        raise ValueError(f"{inst['name']} cần {len(inst['items'])} câu trả lời, mỗi câu 0-3")
    total = sum(answers)
    for i, (lo, hi, name) in enumerate(inst["bands"]):
        if lo <= total <= hi:
            return total, i, name
    raise ValueError("điểm ngoài thang")                # không thể xảy ra với dữ liệu hợp lệ


def item9_positive(phq9_answers):
    """True nếu câu 9 (ý nghĩ chết/tự hại) khác 0: cần hiện kênh hỗ trợ ngay."""
    return len(phq9_answers) > PHQ9_SUICIDE_ITEM and phq9_answers[PHQ9_SUICIDE_ITEM] > 0


def feedback(phq9_answers, gad7_answers, lang="vi"):
    """Lời phản hồi thận trọng cho người tham gia (không chẩn đoán)."""
    p_total, _, p_band = score("phq9", phq9_answers)
    g_total, _, g_band = score("gad7", gad7_answers)
    lines = [f"PHQ-9: {p_total}/27 ({p_band}). GAD-7: {g_total}/21 ({g_band}).",
             "Đây là bảng sàng lọc, không phải chẩn đoán. Điểm cao không có nghĩa là bạn có bệnh, và điểm thấp cũng không "
             "loại trừ khó khăn của bạn."]
    if p_total >= INSTRUMENTS["phq9"]["cutoff"] or g_total >= INSTRUMENTS["gad7"]["cutoff"]:
        lines.append("Với điểm này, bạn nên cân nhắc trò chuyện với bác sĩ hoặc chuyên gia tâm lý.")
    if item9_positive(phq9_answers):
        lines.append("Bạn đã chọn có ý nghĩ chết hoặc làm đau bản thân. Cảm ơn bạn đã nói ra. Hãy liên hệ ngay một người "
                     "thân tin cậy hoặc các kênh hỗ trợ bên dưới.")
    return lines


def build_record(code, profile, phq9_answers, gad7_answers, *, lang="vi", n_words=None):
    """Hồ sơ ẩn danh: KHÔNG có nội dung hội thoại. profile là assessment.build_profile()."""
    p_total, p_i, p_band = score("phq9", phq9_answers)
    g_total, g_i, g_band = score("gad7", gad7_answers)
    rec = {
        "schema": SCHEMA_VERSION, "code": code, "date": datetime.date.today().isoformat(), "lang": lang,
        "instrument_vi_status": INSTRUMENT_VI_STATUS if lang == "vi" else "original_en",
        "bot": {"score": profile.get("score"), "level": profile.get("level"), "evidence": profile.get("evidence"),
                "n_turns": profile.get("n_turns"), "n_words": n_words if n_words is not None else profile.get("n_words"),
                "domains": dict(profile.get("domains", {})), "self_moods": list(profile.get("self_moods", [])),
                "level_capped": profile.get("level_capped", False),
                "components": dict(profile.get("components", {}))},
        "phq9": {"items": list(phq9_answers), "total": p_total, "band": p_i, "band_name": p_band,
                 "item9_positive": item9_positive(phq9_answers)},
        "gad7": {"items": list(gad7_answers), "total": g_total, "band": g_i, "band_name": g_band},
    }
    if "stress" in profile:
        rec["bot"]["stress"] = {k: profile["stress"][k] for k in ("overall", "recent", "share")}
    if "depression" in profile:
        rec["bot"]["depression"] = {"score": profile["depression"]["score"], "label": profile["depression"]["label"]}
    return rec


def save_record(rec, path=None):
    path = Path(path or config.STUDY_DIR / "records.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def load_records(path=None):
    path = Path(path or config.STUDY_DIR / "records.jsonl")
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                                # bỏ dòng hỏng, không làm hỏng cả phân tích
    return out
