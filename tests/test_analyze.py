"""Kiểm thử logic pipeline analyze.py với `modeling` giả (không cần torch)."""
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

import config
import llm_client


@pytest.fixture
def analyze_mod(monkeypatch):
    calls = []
    fake = types.ModuleType("modeling")
    fake.load = lambda path: ("tok", Path(path).name)   # "model" = tên nhánh, để fake biết đang gọi nhánh nào

    def predict_text(tok, model, text, max_length, window_words, multilabel=False, aggregate="mean"):
        calls.append({"text": text, "multilabel": multilabel, "aggregate": aggregate})
        if model == "emotion":
            p = np.full(28, 0.05); p[14] = 0.4; p[25] = 0.2        # fear 0.4, sadness 0.2
            return p
        return np.array([0.2, 0.8]) if model == "stress" else np.array([0.1, 0.2, 0.6, 0.1])
    fake.predict_text = predict_text
    monkeypatch.setitem(sys.modules, "modeling", fake)
    monkeypatch.setattr(config, "find_model", lambda name: Path("/fake") / name)
    sys.modules.pop("analyze", None)
    import analyze
    analyze._cache.clear()
    analyze._calls = calls
    yield analyze
    sys.modules.pop("analyze", None)


def test_window_scores_each_message_and_latest_only_emotion(analyze_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMOTION_THRESHOLD_FILE", tmp_path / "none.json")
    msgs = [f"message {i}" for i in range(8)]
    out = analyze_mod.analyze_conversation(msgs, window=3)
    stress_texts = [c["text"] for c in analyze_mod._calls if not c["multilabel"]]
    assert "message 5" in stress_texts and "message 7" in stress_texts and "message 4" not in stress_texts
    emo_call = [c for c in analyze_mod._calls if c["multilabel"]][0]
    assert emo_call["text"] == "message 7" and emo_call["aggregate"] == "max"
    assert out["stress"] == config.STRESS_YES and abs(out["stress_probability"] - 0.8) < 1e-9
    assert out["depression"] == "moderate" and out["n_messages_used"] == 3
    assert [e for e, _ in out["emotions"]] == ["fear"]


def test_recency_weights():
    w = analyze_weights()
    assert abs(w.sum() - 1) < 1e-9 and w[-1] > w[0]
    assert abs(analyze_weights(decay=1.0)[0] - 1 / 3) < 1e-9
    with pytest.raises(ValueError):
        analyze_weights(decay=0)


def analyze_weights(n=3, decay=0.4):
    had = "modeling" in sys.modules
    sys.modules.setdefault("modeling", types.ModuleType("modeling"))
    try:
        sys.modules.pop("analyze", None)
        import analyze
        return analyze.recency_weights(n, decay)
    finally:
        sys.modules.pop("analyze", None)
        if not had:
            sys.modules.pop("modeling", None)


def test_positive_after_negative_recovers(analyze_mod, monkeypatch):
    """Câu tích cực sau câu tiêu cực phải kéo kết quả xuống, không bị dính."""
    probs = {"I feel hopeless and stressed": 0.95, "I am happy today": 0.05}
    analyze_mod._msg_cache.clear()
    monkeypatch.setattr(analyze_mod, "stress_probs", lambda t: np.array([1 - probs[t], probs[t]]))
    out = analyze_mod.analyze_conversation(list(probs), window=5, decay=0.4)
    assert out["latest_stress_probability"] == 0.05
    assert out["stress_probability"] < 0.5 and out["stress"] == config.STRESS_NO


def test_default_threshold_filters(analyze_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMOTION_THRESHOLD_FILE", tmp_path / "none.json")
    out = analyze_mod.analyze_conversation(["hi"])
    assert [e for e, _ in out["emotions"]] == ["fear"]              # sadness 0.2 < 0.25


def test_per_class_thresholds_file(analyze_mod, tmp_path, monkeypatch):
    f = tmp_path / "thr.json"
    f.write_text(json.dumps({"global": 0.5, "per_class": {"sadness": 0.15}}))
    monkeypatch.setattr(config, "EMOTION_THRESHOLD_FILE", f)
    out = analyze_mod.analyze_conversation(["hi"])
    assert {e for e, _ in out["emotions"]} == {"sadness"}           # fear 0.4 < global 0.5; sadness 0.2 >= 0.15


def test_empty_messages_raise_and_legacy_analyze(analyze_mod):
    with pytest.raises(ValueError):
        analyze_mod.analyze_conversation(["", "   "])
    assert "stress_probability" in analyze_mod.analyze("hello there")


def test_llm_generate_reply_builds_request(monkeypatch):
    sent = {}

    class FakeMessages:
        def create(self, **kw):
            sent.update(kw)
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=" Xin chào ")])

    monkeypatch.setattr(llm_client, "_client", lambda: types.SimpleNamespace(messages=FakeMessages()))
    result = {"stress": config.STRESS_YES, "emotions": [("fear", 0.4)], "depression": "mild"}
    hist = [{"role": "user", "content": "I'm worried"}]
    assert llm_client.generate_reply(result, "I'm worried", hist) == "Xin chào"
    assert "fear" in sent["system"] and "KHÔNG phải chẩn đoán" in sent["system"]
    assert sent["messages"][-1] == {"role": "user", "content": "I'm worried"} and sent["max_tokens"] == 400
    assert llm_client.translate_to_english("xin chào") == "Xin chào"
