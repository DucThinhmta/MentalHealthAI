"""Nhật ký tâm trạng nhiều ngày: đánh giá qua nhiều ngày đáng tin hơn một lần ngồi chat.

Thiết kế vì quyền riêng tư:
- CHỈ lưu điểm số (tâm trạng 1-5, số giờ ngủ, mức căng thẳng 0-10) và ngày; KHÔNG có ô nhập chữ, nên không dính tên/địa chỉ.
- Khóa là một MÃ NGẪU NHIÊN (xem privacy.new_participant_code) do người dùng tự giữ; không gắn với tên hay email.
  Ai biết mã đều đọc được nhật ký đó (chỉ gồm điểm số), nên mã nên được coi như mật khẩu nhẹ.
- Mỗi ngày một dòng: lưu lại trong cùng ngày sẽ ghi đè.
"""
import datetime
import json
import re
from pathlib import Path

import config

MOOD_LABELS = {1: "Rất tệ", 2: "Không ổn", 3: "Bình thường", 4: "Khá ổn", 5: "Rất tốt"}
LOW_STREAK = 5                      # số ngày liên tiếp tâm trạng <= 2 thì gợi ý nói chuyện với người tin cậy
DIARY_DIR = config.DATA_DIR / "diary"
_CODE = re.compile(r"^P-[0-9A-F]{6}$")


def valid_code(code):
    return bool(_CODE.match((code or "").strip().upper()))


def _path(code, directory=None):
    code = code.strip().upper()
    if not valid_code(code):
        raise ValueError("Mã nhật ký không hợp lệ (dạng P-XXXXXX)")
    return Path(directory or DIARY_DIR) / f"{code}.jsonl"


def load(code, directory=None):
    path = _path(code, directory)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return sorted(out, key=lambda e: e["date"])


def add_entry(code, mood, sleep_hours=None, stress=None, date=None, directory=None):
    """Thêm/ghi đè mục của ngày `date` (mặc định hôm nay). Trả danh sách mục sau khi lưu."""
    if mood not in MOOD_LABELS:
        raise ValueError("Tâm trạng phải là 1-5")
    if sleep_hours is not None and not 0 <= sleep_hours <= 24:
        raise ValueError("Số giờ ngủ phải trong 0-24")
    if stress is not None and not 0 <= stress <= 10:
        raise ValueError("Mức căng thẳng phải trong 0-10")
    date = (date or datetime.date.today()).isoformat()
    entries = [e for e in load(code, directory) if e["date"] != date]
    entries.append({"date": date, "mood": int(mood), "sleep_hours": sleep_hours, "stress": stress})
    entries.sort(key=lambda e: e["date"])
    path = _path(code, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return entries


def delete(code, directory=None):
    """Xoá toàn bộ nhật ký của mã này. Trả True nếu có tệp để xoá."""
    path = _path(code, directory)
    if path.exists():
        path.unlink()
        return True
    return False


def low_streak(entries, threshold=2):
    """Số ngày LIÊN TIẾP (tính từ ngày gần nhất, các ngày phải kề nhau) có tâm trạng <= threshold."""
    streak, prev = 0, None
    for e in sorted(entries, key=lambda e: e["date"], reverse=True):
        day = datetime.date.fromisoformat(e["date"])
        if prev is not None and (prev - day).days != 1:
            break
        if e["mood"] > threshold:
            break
        streak += 1
        prev = day
    return streak


def summary(entries, days=14):
    """Tóm tắt N ngày gần nhất: trung bình tâm trạng/giấc ngủ, xu hướng (nửa sau trừ nửa đầu), và cảnh báo nếu tâm trạng thấp kéo dài."""
    if not entries:
        return {"n": 0}
    last = datetime.date.fromisoformat(max(e["date"] for e in entries))
    recent = [e for e in entries if (last - datetime.date.fromisoformat(e["date"])).days < days]
    moods = [e["mood"] for e in recent]
    sleeps = [e["sleep_hours"] for e in recent if e.get("sleep_hours") is not None]
    half = len(moods) // 2
    out = {"n": len(recent), "mood_mean": sum(moods) / len(moods),
           "sleep_mean": sum(sleeps) / len(sleeps) if sleeps else None,
           "trend": (sum(moods[half:]) / len(moods[half:]) - sum(moods[:half]) / len(moods[:half])) if half else 0.0,
           "low_streak": low_streak(recent)}
    out["alert"] = out["low_streak"] >= LOW_STREAK
    return out


def alert_text(streak):
    return (f"Tâm trạng của bạn ở mức thấp {streak} ngày liên tiếp. Nếu cảm giác này kéo dài, bạn nên trò chuyện với một người "
            "thân tin cậy hoặc chuyên gia tâm lý; mục 'Khi cần hỗ trợ ngay' ở thanh bên có các kênh hỗ trợ.")
