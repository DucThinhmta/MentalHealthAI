import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compare_depression as cd


def _agg(mean, lo, hi, n=3):
    return {"macro_f1": {"mean": mean, "std": 0.01, "ci95": [lo, hi], "n": n, "values": [mean] * n}}


def test_verdict_rules():
    assert cd.verdict(_agg(0.45, 0.43, 0.47), 0.412) == "BERT vượt baseline"
    assert cd.verdict(_agg(0.36, 0.34, 0.38), 0.412) == "baseline vượt BERT"
    assert cd.verdict(_agg(0.40, 0.37, 0.43), 0.412) == "chưa phân biệt được"
    assert "chưa đủ seed" in cd.verdict(_agg(0.5, 0.5, 0.5, n=1), 0.412)


def test_recommendation_is_honest_about_each_outcome():
    assert "ghi KTC 95%" in cd.recommendation([("focal", "BERT vượt baseline")])
    assert "TF-IDF" in cd.recommendation([("focal", "baseline vượt BERT"), ("base", "baseline vượt BERT")])
    assert "Chưa phân biệt" in cd.recommendation([("focal", "chưa phân biệt được")])
    assert "run_seeds" in cd.recommendation([])


def test_report_uses_best_baseline_and_lists_each_config(tmp_path):
    csv = tmp_path / "b.csv"
    csv.write_text("task,model,accuracy,macro_f1,f1_per_class,qwk\n"
                   "depression,TF-IDF + LogReg,0.66,0.412,,0.376\ndepression,TF-IDF + LinearSVC,0.70,0.352,,0.273\n"
                   "stress,TF-IDF + LogReg,0.72,0.72,,\n", encoding="utf-8")
    text, verdicts = cd.build_report({"focal": _agg(0.365, 0.35, 0.38)}, cd.load_baselines(csv))
    assert "TF-IDF + LogReg" in text and "'focal'" in text and verdicts == [("focal", "baseline vượt BERT")]
