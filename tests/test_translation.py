"""Kiểm thử phần dịch dữ liệu hai chiều và dịch Việt->Anh lúc chạy (dùng bộ dịch giả, không tải mô hình)."""
import json
import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

import config
import datasets_io
import pipeline
import translate_datasets as td
import translator


def test_split_for_translation_keeps_every_word():
    text = " ".join(f"Câu số {i} khá dài một chút." for i in range(60))
    chunks = translator.split_for_translation(text, max_words=30)
    assert all(len(c.split()) <= 30 for c in chunks) and " ".join(chunks) == text
    long_sentence = " ".join(["từ"] * 75)
    assert [len(c.split()) for c in translator.split_for_translation(long_sentence, 30)] == [30, 30, 15]
    assert translator.split_for_translation("  ") == []


def test_available_needs_downloaded_model(tmp_path, monkeypatch):
    monkeypatch.setattr(translator, "TRANSLATOR_DIR", tmp_path)
    monkeypatch.delenv("MHAI_TRANSLATOR", raising=False)
    assert translator.available("vi-en") is None and translator.runtime_vi2en("xin chào") is None
    d = translator.local_path("opus", "vi-en")
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    assert translator.available("vi-en") == "opus"


def test_group_respects_limits():
    groups = td.group(["a" * 100] * 45, max_items=20, max_chars=1500)
    assert [len(g) for g in groups] == [15, 15, 15]
    assert sum(len(g) for g in td.group(["x"] * 41)) == 41


def test_parse_items_aligns_by_id_and_skips_bad_entries():
    originals = ["one", "two", "three"]
    text = json.dumps({"items": [{"id": 2, "text": "ba"}, {"id": 0, "text": "một"}, {"id": 7, "text": "?"},
                                 {"id": 1, "text": "  "}]})
    assert td.parse_items(text, originals) == [("three", "ba"), ("one", "một")]


def test_estimate_is_positive_and_haiku_cheaper():
    todo = {("en", "vi"): ["I feel stressed about my exams"] * 100, ("vi", "en"): ["mình buồn quá"] * 50}
    s, h = td.estimate(todo, "claude-sonnet-5-5"), td.estimate(todo, "claude-haiku-5-5")
    assert s["requests"] == h["requests"] > 0 and 0 < h["usd"] < s["usd"]


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CLEAN_DIR", tmp_path / "clean")
    monkeypatch.setattr(config, "EXTRA_DIR", tmp_path / "extra")
    monkeypatch.setattr(td, "OUT_DIR", tmp_path / "translated")
    monkeypatch.setattr(td, "CACHE_DIR", tmp_path / "translated" / "cache")
    (tmp_path / "clean").mkdir()
    (tmp_path / "extra").mkdir()
    pd.DataFrame({"text": ["I am so stressed", "Nice day today"], "label": [1, 0]}) \
        .to_csv(tmp_path / "clean" / "stress_train.csv", index=False)
    vec = str([0.0] * 17 + [1.0] + [0.0] * 10)
    pd.DataFrame({"text": ["vui quá"], "label_vector": [vec], "lang": ["vi"], "goemotion": ["joy"]}) \
        .to_csv(tmp_path / "extra" / "emotion_demo_train.csv", index=False)
    for split in ("dev", "test"):
        pd.DataFrame({"text": [], "label_vector": [], "lang": [], "goemotion": []}) \
            .to_csv(tmp_path / "extra" / f"emotion_demo_{split}.csv", index=False)
    return tmp_path


def test_translate_roundtrip_both_directions_keeps_labels(data_dirs):
    found = td.sources()
    assert {(t, n, s, lg) for t, n, s, lg, _ in found} >= {("stress", "core", "train", "en"),
                                                           ("emotion", "demo", "train", "vi")}
    todo = td.pending(found)
    assert sorted(todo[("en", "vi")]) == ["I am so stressed", "Nice day today"] and todo[("vi", "en")] == ["vui quá"]
    fake = {"I am so stressed": "Mình căng thẳng quá", "Nice day today": "Hôm nay đẹp trời", "vui quá": "so happy"}
    for (src, tgt), texts in todo.items():
        td.append_cache(src, tgt, [(t, fake[t]) for t in texts], "test")
    assert td.pending(found) == {}                                         # chạy lại không dịch lại
    assert td.write_outputs(found) == 0
    vi = datasets_io.load_translated("stress", "train", lang="vi")
    assert list(vi["text"]) == ["Mình căng thẳng quá", "Hôm nay đẹp trời"] and list(vi["label"]) == [1, 0]
    assert list(vi["original_text"]) == ["I am so stressed", "Nice day today"]
    en = datasets_io.load_translated("emotion", "train", ["demo"], lang="en")
    assert en["text"].iloc[0] == "so happy" and en["labels"].iloc[0][17] == 1.0 and en["lang"].iloc[0] == "en"


def test_missing_translations_are_reported(data_dirs):
    found = td.sources()
    assert td.write_outputs(found) == 3
    assert datasets_io.load_translated("stress", "train") is None


# ---------- dịch lúc chạy ----------
def _fake_analyze(monkeypatch):
    seen = []

    def analyze(text):
        seen.append(text)
        return {"stress": config.STRESS_YES, "stress_probability": 0.8, "emotions": [], "depression": "mild",
                "depression_probs": {"minimum": .2, "mild": .5, "moderate": .2, "severe": .1}, "n_messages_used": 1}
    monkeypatch.setattr(pipeline, "_analyze", analyze)
    return seen


def test_runtime_local_translation_makes_vietnamese_analyzable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    seen = _fake_analyze(monkeypatch)
    monkeypatch.setattr(translator, "runtime_vi2en", lambda t: "I have not slept all week")
    monkeypatch.setattr(pipeline, "_models_present", lambda: True)
    sess = pipeline.Session()
    "".join(pipeline.run_turn("Mình mất ngủ cả tuần nay", sess, use_llm=False, rng=random.Random(0)).chunks)
    e = sess.entries[0]
    assert seen == ["I have not slept all week"] and e["translated"] and e["model_ok"]
    assert e["domains"]["sleep"] == "concern"                              # khía cạnh vẫn đọc từ câu tiếng Việt gốc


def test_runtime_without_translator_falls_back_to_untranslated(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    seen = _fake_analyze(monkeypatch)
    monkeypatch.setattr(translator, "runtime_vi2en", lambda t: None)
    sess = pipeline.Session()
    "".join(pipeline.run_turn("minh met moi qua", sess, use_llm=False, rng=random.Random(0)).chunks)
    assert not sess.entries[0]["model_ok"] and seen == ["minh met moi qua"]  # tiếng Việt không dấu cũng được nhận ra
