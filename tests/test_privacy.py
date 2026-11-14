"""Xoá thông tin nhận dạng và cấu hình hỗ trợ (privacy.py, config.support_resources)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import config
import privacy
import safety


@pytest.mark.parametrize("raw,must_not_contain", [
    ("mình là an, email an.nguyen@gmail.com nhé", "gmail"),
    ("gọi mình 0912 345 678", "345"),
    ("sđt +84 912345678 nha", "912345678"),
    ("số của mình 091.234.5678", "5678"),
    ("zalo 0912345678", "0912345678"),
    ("facebook.com/an.nguyen.123 hoặc https://fb.com/xyz", "fb.com"),
    ("cccd của mình 012345678901", "012345678901"),
    ("add instagram @an_nguyen_99 đi", "an_nguyen"),
])
def test_scrub_removes_identifiers(raw, must_not_contain):
    assert must_not_contain not in privacy.scrub(raw)


@pytest.mark.parametrize("text", [
    "mình ngủ 4 tiếng mỗi đêm", "đã 3 tuần rồi", "lúc 10:30 tối", "ngày 12/11/2026 mình thi", "điểm 7.5 thôi",
    "mình 19 tuổi", "mất 2 tháng nay", "khoảng 150000 đồng",
])
def test_scrub_leaves_ordinary_numbers_alone(text):
    assert privacy.scrub(text) == text


def test_scrub_entry_drops_or_keeps_text():
    entry = {"turn": 1, "user": "gọi 0912 345 678", "english": "call 0912 345 678", "stress_probability": 0.4}
    kept = privacy.scrub_entry(entry, keep_text=True)
    assert kept["user"] == "gọi [SĐT]" and kept["english"] == "call [SĐT]" and kept["stress_probability"] == 0.4
    dropped = privacy.scrub_entry(entry, keep_text=False)
    assert dropped["user"] == privacy.NOT_STORED and dropped["english"] == privacy.NOT_STORED
    assert entry["user"] == "gọi 0912 345 678"                       # không sửa bản gốc


def test_participant_codes_are_random_and_unlinked():
    codes = {privacy.new_participant_code() for _ in range(50)}
    assert len(codes) == 50 and all(c.startswith("P-") and len(c) == 8 for c in codes)


# ---------- tài nguyên hỗ trợ ----------
def test_support_resources_minor_and_language():
    base = config.support_resources()
    assert any("115" in r and "113" in r for r in base) and not any("111" in r for r in base)
    minor = config.support_resources(minor=True)
    assert any("111" in r for r in minor) and len(minor) == len(base) + 1
    assert any("Immediate danger" in r for r in config.support_resources(lang="en"))


def test_unverified_numbers_are_never_shown_to_users():
    shown = " ".join(config.support_resources(True) + config.support_resources(True, "en"))
    shown += safety.crisis_message("Tôi muốn chết", True) + safety.crisis_message("I want to die", True)
    for number in ("1900", "3576", "0984"):
        assert number not in shown
    assert set(config.RESOURCES_TO_VERIFY) and config.RESOURCES_CHECKED


def test_local_resources_file(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"resources": [
        {"name": "Phòng tư vấn A", "contact": "tuvan@truong.edu.vn", "hours": "T2-T6"},
        {"name": "Chưa điền", "contact": ""}, {"contact": "x"}, "rác"]}), encoding="utf-8")
    assert [r["name"] for r in config.load_local_resources(f)] == ["Phòng tư vấn A"]
    assert config.load_local_resources(tmp_path / "missing.json") == []
    f.write_text("{hỏng", encoding="utf-8")
    assert config.load_local_resources(f) == []
    assert config.load_local_resources() == []                       # tệp mẫu trong repo chưa điền: không hiển thị gì


def test_local_resources_appear_in_the_crisis_message(monkeypatch):
    monkeypatch.setattr(config, "load_local_resources",
                        lambda path=None: [{"name": "Phòng tư vấn A", "contact": "tuvan@truong.edu.vn", "hours": "T2-T6"}])
    assert "Phòng tư vấn A: tuvan@truong.edu.vn (T2-T6)" in safety.crisis_message("Tôi muốn chết")


def test_crisis_message_is_always_vietnamese_by_default_and_follows_the_user_in_auto_mode(monkeypatch):
    vi, en_in = safety.crisis_message("Tôi muốn chết"), safety.crisis_message("I want to die")
    assert "Mình rất tiếc" in vi and "emergency number" not in vi
    assert en_in == safety.crisis_message("Tôi muốn chết")                          # cùng một thông điệp tiếng Việt
    assert "Mình rất tiếc" in safety.crisis_message("toi muon chet")
    monkeypatch.setattr(config, "REPLY_LANGUAGE", "auto")
    en = safety.crisis_message("I want to die")
    assert "I'm really sorry" in en and "Immediate danger" in en and "Mình" not in en
    assert "Mình rất tiếc" in safety.crisis_message("Tôi muốn chết")
