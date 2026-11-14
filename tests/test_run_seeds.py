"""Tổng hợp nhiều seed (run_seeds.py): phần thống kê và dựng lệnh; không huấn luyện thật."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import config
import run_seeds as rs


def test_aggregate_matches_hand_calculation():
    agg = rs.aggregate([{"macro_f1": 0.5}, {"macro_f1": 0.6}, {"macro_f1": 0.7}])["macro_f1"]
    assert agg["mean"] == pytest.approx(0.6) and agg["std"] == pytest.approx(0.1) and agg["n"] == 3
    half = 4.302653 * 0.1 / math.sqrt(3)                       # t(0,975; 2 bậc tự do) = 4,3027
    assert agg["ci95"] == pytest.approx([0.6 - half, 0.6 + half], abs=1e-5)


def test_aggregate_handles_one_seed_and_missing_metrics():
    one = rs.aggregate([{"a": 0.5, "b": 1.0}])
    assert one["a"]["mean"] == 0.5 and math.isnan(one["a"]["std"]) and math.isnan(one["a"]["ci95"][0])
    mixed = rs.aggregate([{"a": 1.0, "b": 2.0}, {"a": 3.0}])
    assert mixed["a"]["n"] == 2 and mixed["b"]["n"] == 1


def test_compare_detects_a_clear_difference_and_refuses_with_one_seed():
    clear = rs.compare([0.62, 0.63, 0.61, 0.64], [0.50, 0.51, 0.49, 0.52])
    assert clear["diff"] == pytest.approx(0.12) and clear["p"] < 0.001 and clear["cohens_d"] > 5
    noisy = rs.compare([0.50, 0.60, 0.55], [0.52, 0.58, 0.54])
    assert noisy["p"] > 0.5
    assert math.isnan(rs.compare([0.5], [0.4, 0.45])["p"])


def test_commands_use_a_separate_model_dir_per_seed_and_pass_extra_args():
    train, evaluate = rs.build_commands("depression", "focal", 43, ["--loss", "focal", "--epochs", "4"], python="py")
    assert train[0] == "py" and train[2:6] == ["--task", "depression", "--seed", "43"]
    assert "--seed" in train and train[train.index("--seed") + 1] == "43"
    assert train[-4:] == ["--loss", "focal", "--epochs", "4"]
    assert evaluate[evaluate.index("--model-dir") + 1] == train[train.index("--out") + 1]
    assert evaluate[evaluate.index("--tag") + 1] == "focal_s43"
    assert rs.model_dir("stress", "a", 1) != rs.model_dir("stress", "a", 2)


def test_load_seed_results_reads_what_evaluate_all_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path)
    for s, f1 in ((1, 0.30), (2, 0.40)):
        (tmp_path / f"t_s{s}_results.json").write_text(json.dumps({"depression": {"macro_f1": f1, "accuracy": 0.7}}))
    rows = rs.load_seed_results("depression", "t", [1, 2, 3])          # seed 3 chưa có
    assert [r["macro_f1"] for r in rows] == [0.30, 0.40] and "thiếu t_s3_results.json" in capsys.readouterr().out
    table = rs.format_table("depression", rs.aggregate(rows))
    assert "macro_f1" in table and "0.3500" in table
