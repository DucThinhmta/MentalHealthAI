"""Kiểm thử đánh giá cả cuộc trò chuyện (assessment.py) và nạp dữ liệu bổ sung. Không cần mô hình."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

import assessment as A
import config
import datasets_io
import fetch_datasets


# ---------- nhận diện khía cạnh ----------
@pytest.mark.parametrize("text,expected", [
    ("Mình mất ngủ cả tuần nay, mệt lắm", {"sleep": "concern", "energy": "concern", "duration": "concern"}),
    ("Mình không mất ngủ, ngủ ngon", {"sleep": "ok"}),
    ("dạo này không ổn lắm", {"mood": "concern"}),
    ("chẳng có ai để kể cả, cô đơn quá", {"support": "concern"}),
    ("ăn uống bình thường", {"appetite": "ok"}),
    ("minh van on ma", {"mood": "ok"}),                                  # gõ không dấu
    ("I can't focus and I'm not eating, for weeks now", {"focus": "concern", "appetite": "concern",
                                                          "duration": "concern"}),
    ("Hôm nay mình ôn thi", {}),                                         # 'ôn' không phải 'ổn'
])
def test_detect_domains(text, expected):
    assert A.detect_domains(text) == expected


def test_asked_domain_reads_the_last_question():
    assert A.asked_domain("Mất ngủ mệt thật. Dạo này bạn ngủ thế nào?") == "sleep"
    assert A.asked_domain("Bạn đã kể chuyện này với ai chưa?") == "support"
    assert A.asked_domain("Mình nghe rồi.") is None


def test_every_domain_question_is_recognised_as_asking_that_domain():
    for d, spec in A.DOMAINS.items():
        for lang in ("vi", "en"):
            for q in spec["ask"][lang]:
                assert A.asked_domain(q) == d, (d, q)
            assert spec["quick"][lang]


# ---------- tổng hợp ----------
def _entry(stress, words=10, dep=0.5, emotions=(), domains=None, topics=()):
    return {"model_ok": True, "stress_probability": stress, "words": words, "depression_score": dep,
            "emotions": list(emotions), "domains": domains or {}, "topics": list(topics)}


def test_one_message_is_not_enough_to_judge():
    p = A.build_profile([_entry(0.95, words=12, dep=2.5)])
    assert p["evidence"] == "low"


def test_profile_uses_whole_conversation_not_only_last_message():
    entries = [_entry(0.9), _entry(0.9), _entry(0.9), _entry(0.1)]
    p = A.build_profile(entries)
    assert p["stress"]["overall"] > 0.6                                  # một câu vui cuối không xóa cả cuộc trò chuyện
    assert p["stress"]["recent"] < p["stress"]["overall"]               # nhưng xu hướng gần đây vẫn được ghi nhận
    assert p["stress"]["share"] == 0.75


def test_longer_messages_weigh_more():
    p = A.build_profile([_entry(0.9, words=100), _entry(0.1, words=4)])
    assert p["stress"]["overall"] > 0.6


def test_untranslated_messages_do_not_contribute_model_scores():
    e = _entry(0.99)
    e["model_ok"] = False
    p = A.build_profile([e, _entry(0.1)])
    assert p["n_scored"] == 1 and p["stress"]["overall"] == pytest.approx(0.1)


def test_later_answer_updates_domain_and_evidence_grows():
    entries = [_entry(0.8, words=15, domains={"sleep": "concern"}), _entry(0.7, words=15, domains={"sleep": "ok"})]
    assert A.build_profile(entries)["domains"]["sleep"] == "ok"
    many = [_entry(0.8, words=15, domains={d: "concern"}) for d in A.DOMAIN_ORDER[:6]]
    assert A.build_profile(many)["evidence"] == "high"


def test_levels_order_from_calm_to_heavy():
    calm = A.build_profile([_entry(0.05, dep=0.1, emotions=[("joy", 0.8)], domains={"sleep": "ok"})] * 4, [5])
    heavy = A.build_profile([_entry(0.95, dep=2.8, emotions=[("sadness", 0.8)],
                                    domains={"sleep": "concern", "support": "concern", "duration": "concern"})] * 4, [1])
    assert calm["level"] == 0 and heavy["level"] == 3


def test_profile_works_without_any_model_scores():
    e = {"words": 12, "model_ok": False, "domains": {"sleep": "concern", "energy": "concern"}, "topics": []}
    p = A.build_profile([e, e, e])
    assert "stress" not in p and p["components"] == {"domains": 1.0} and p["level"] == 3


def test_next_domain_prefers_duration_once_something_is_wrong_then_reasks_ignored():
    p = A.build_profile([_entry(0.8, domains={"sleep": "concern"})])
    assert A.next_domain(p) == "duration"
    all_asked = [d for d in A.DOMAIN_ORDER if d != "sleep"]
    assert A.next_domain(p, all_asked) in all_asked                     # bị lờ đi thì hỏi lại một lần
    assert A.next_domain(p, all_asked * 2) is None


def test_suggestions_follow_concerns_and_recommend_professional_when_heavy():
    heavy = A.build_profile([_entry(0.95, dep=2.8, domains={"sleep": "concern"}, topics=["study"])] * 4, [1])
    titles = [t for _, t, _ in A.suggestions(heavy)]
    assert titles[0] == "Nói chuyện với chuyên gia" and "Chăm lại giấc ngủ" in titles
    md = A.summary_markdown(heavy, "Lời tổng kết")
    assert "Lời tổng kết" in md and "không phải chẩn đoán" in md.lower()


def test_concern_weights_cover_all_components():
    assert set(config.CONCERN_WEIGHTS) == {"stress", "depression", "emotion", "domains", "self_mood"}


# ---------- dữ liệu bổ sung ----------
def test_fetch_convert_maps_labels_and_drops_unmapped():
    spec = fetch_datasets.SOURCES["empathetic"]
    df = pd.DataFrame({"situation": ["I lost my dog", "I miss old times", "  "],
                       "emotion": ["devastated", "nostalgic", "sad"]})
    out = fetch_datasets.convert(df, spec)
    assert list(out["goemotion"]) == ["grief"]                          # 'nostalgic' bỏ, câu rỗng bỏ
    vec = eval(out["label_vector"].iloc[0])
    assert len(vec) == 28 and vec[config.EMOTION_LABELS.index("grief")] == 1.0 and sum(vec) == 1.0


def test_vsmec_mapping_covers_all_labels():
    assert set(fetch_datasets.SOURCES["uit_vsmec"]["map"].values()) <= set(config.EMOTION_LABELS)
    for spec in fetch_datasets.SOURCES.values():
        assert set(spec["map"].values()) <= set(config.EMOTION_LABELS)


def test_load_extra_and_resolve(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXTRA_DIR", tmp_path)
    for split in ("train", "dev", "test"):
        pd.DataFrame({"text": ["xin chào"], "label_vector": [str([0.0] * 27 + [1.0])], "lang": ["vi"]}) \
            .to_csv(tmp_path / f"emotion_demo_set_{split}.csv", index=False)
    assert datasets_io.extra_names() == ["demo_set"] and datasets_io.resolve_extra("all") == ["demo_set"]
    df = datasets_io.load_extra("emotion", "train", ["demo_set"])
    assert df["source"].iloc[0] == "demo_set" and df["labels"].iloc[0][-1] == 1.0
    assert datasets_io.resolve_extra("") == []
    with pytest.raises(FileNotFoundError):
        datasets_io.resolve_extra("missing")
