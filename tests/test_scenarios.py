import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import config
import evaluate_scenarios as es


def _fake_analyze():
    m = types.ModuleType("analyze")
    neg = ("exam", "panic", "overwhelmed", "afraid", "sick", "crying", "worried")

    def p(text):
        t = text.lower()
        hit = sum(w in t for w in neg)
        return 0.9 if hit else 0.1

    m.stress_probs = lambda text: np.array([1 - p(text), p(text)])

    def analyze_conversation(msgs, window=5, decay=None):
        d = decay or config.RECENCY_DECAY
        ps = np.array([p(x) for x in msgs[-window:]])
        w = np.array([d ** (len(ps) - 1 - i) for i in range(len(ps))]); w /= w.sum()
        return {"stress_probability": float((ps * w).sum())}
    m.analyze_conversation = analyze_conversation
    return m


def test_scenario_file_is_well_formed():
    sc = es.load_scenarios()
    assert len(sc) == 24
    assert {s["category"] for s in sc} == {"neg_to_pos", "pos_to_neg", "neg_neg", "pos_pos"}
    assert all(s["expect"] in ("stress", "no_stress") and len(s["messages"]) >= 2 for s in sc)
    assert len({s["id"] for s in sc}) == 24


def test_decay_beats_concat_on_recovery():
    sc = [{"id": "x", "category": "neg_to_pos", "expect": "no_stress",
           "messages": ["I panic about my exam", "Now it is all fine and I feel good"]}]
    df = es.run(sc, _fake_analyze())
    r = df.set_index("strategy")["correct"]
    assert r["concat"] == False and r["per_msg_decay"] == True and r["latest_only"] == True
    overall, by_cat = es.summarize(df)
    assert overall["per_msg_decay"] == 1.0 and "neg_to_pos" in by_cat.index
