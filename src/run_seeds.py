"""Huấn luyện + đánh giá một cấu hình với NHIỀU seed rồi tổng hợp trung bình ± độ lệch chuẩn (và khoảng tin cậy 95%).

Một lần chạy đơn lẻ có thể may hoặc rủi (đặc biệt với lớp hiếm như 'mild'); báo cáo nên dùng số liệu nhiều seed.

    # Huấn luyện 3 seed (cần GPU/dữ liệu) rồi đánh giá trên tập test và tổng hợp:
    python src/run_seeds.py --task depression --tag focal --seeds 42 43 44 -- --loss focal --weight-power 0.5 --epochs 4
    python src/run_seeds.py --task depression --tag base  --seeds 42 43 44 -- --epochs 4          # cấu hình đối chứng
    python src/run_seeds.py --compare focal base --task depression                                # so sánh hai cấu hình
    python src/run_seeds.py --task depression --tag focal --seeds 42 43 44 --aggregate-only      # chỉ tổng hợp kết quả có sẵn
    python src/run_seeds.py --task depression --tag x --seeds 1 2 --dry-run -- --epochs 1         # xem lệnh sẽ chạy

Kết quả của mỗi seed nằm ở reports/<tag>_s<seed>_results.json (do evaluate_all.py ghi); tổng hợp ở reports/seeds_<tag>.json.
Chỉ hỗ trợ stress và depression (evaluate_all.py chỉ nhận --model-dir cho hai nhánh này).
"""
import argparse
import json
import subprocess
import sys

import numpy as np
from scipy import stats

import config

SRC = config.SRC_DIR
KEY_METRICS = {"stress": ["accuracy", "macro_f1", "macro_precision", "macro_recall"],
               "depression": ["accuracy", "macro_f1", "weighted_f1", "mae_levels", "within_one_level",
                              "quadratic_weighted_kappa"]}


def model_dir(task, tag, seed):
    return config.PROJECT_DIR / "models" / "seeds" / f"{task}_{tag}_s{seed}"


def build_commands(task, tag, seed, train_args, python=None):
    """(lệnh huấn luyện, lệnh đánh giá) cho một seed."""
    py = python or sys.executable
    out = str(model_dir(task, tag, seed))
    train = [py, str(SRC / "train_classifier.py"), "--task", task, "--seed", str(seed), "--out", out, *train_args]
    evaluate = [py, str(SRC / "evaluate_all.py"), "--task", task, "--model-dir", out, "--tag", f"{tag}_s{seed}"]
    return train, evaluate


def is_done(tag, seed):
    """Seed này đã được huấn luyện và đánh giá xong chưa (có tệp kết quả)."""
    return (config.REPORT_DIR / f"{tag}_s{seed}_results.json").exists()


def aggregate(per_seed):
    """per_seed: danh sách dict {metric: giá trị}. Trả {metric: {mean, std, ci95, n, values}} (std với ddof=1)."""
    out = {}
    for metric in sorted({m for d in per_seed for m in d}):
        vals = np.array([d[metric] for d in per_seed if metric in d], dtype=float)
        n = len(vals)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else float("nan")
        half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else float("nan")
        out[metric] = {"mean": mean, "std": std, "ci95": [mean - half, mean + half], "n": n,
                       "values": [float(v) for v in vals]}
    return out


def compare(a_values, b_values):
    """So sánh hai cấu hình theo từng seed (kiểm định t Welch, d Cohen). Chênh lệch = A - B. Với n nhỏ chỉ mang tính tham khảo."""
    a, b = np.asarray(a_values, dtype=float), np.asarray(b_values, dtype=float)
    diff = float(a.mean() - b.mean())
    if len(a) < 2 or len(b) < 2:
        return {"diff": diff, "p": float("nan"), "cohens_d": float("nan"), "note": "cần >= 2 seed mỗi cấu hình"}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return {"diff": diff, "t": float(t), "p": float(p), "cohens_d": float(diff / pooled) if pooled > 0 else float("nan")}


def load_seed_results(task, tag, seeds):
    rows = []
    for s in seeds:
        path = config.REPORT_DIR / f"{tag}_s{s}_results.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")).get(task, {}))
        else:
            print(f"[cảnh báo] thiếu {path.name}")
    return [r for r in rows if r]


def format_table(task, agg):
    lines = [f"{'chỉ số':<28}{'trung bình':>11}{'độ lệch':>9}   KTC 95%          n"]
    for m in KEY_METRICS[task]:
        if m in agg:
            v = agg[m]
            lines.append(f"{m:<28}{v['mean']:>11.4f}{v['std']:>9.4f}   [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]  {v['n']}")
    return "\n".join(lines)


def run_command(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["stress", "depression"], required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--skip-done", action="store_true",
                    help="bỏ qua seed đã có reports/<tag>_s<seed>_results.json (chạy tiếp sau khi Colab bị ngắt)")
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="đặt sau '--': tham số truyền cho train_classifier.py")
    a = ap.parse_args()
    extra = a.train_args[1:] if a.train_args[:1] == ["--"] else a.train_args

    if a.compare:
        ta, tb = a.compare
        agg = {t: aggregate(load_seed_results(a.task, t, a.seeds)) for t in (ta, tb)}
        for m in KEY_METRICS[a.task]:
            if m in agg[ta] and m in agg[tb]:
                c = compare(agg[ta][m]["values"], agg[tb][m]["values"])
                print(f"{m:<28} {ta} - {tb} = {c['diff']:+.4f}   p = {c['p']:.3f}   d = {c['cohens_d']:.2f}")
        return
    if not a.tag:
        raise SystemExit("Cần --tag (tên cấu hình, vd. focal)")
    if not a.aggregate_only:
        for seed in a.seeds:
            if a.skip_done and is_done(a.tag, seed):
                print(f"[bỏ qua] seed {seed} đã có kết quả")
                continue
            train, evaluate = build_commands(a.task, a.tag, seed, extra)
            if a.dry_run:
                print(" ".join(train), "\n" + " ".join(evaluate))
            else:
                run_command(train)
                run_command(evaluate)
    if a.dry_run:
        return
    agg = aggregate(load_seed_results(a.task, a.tag, a.seeds))
    config.REPORT_DIR.mkdir(exist_ok=True)
    (config.REPORT_DIR / f"seeds_{a.tag}.json").write_text(
        json.dumps({"task": a.task, "tag": a.tag, "seeds": a.seeds, "train_args": extra, "metrics": agg}, indent=2), encoding="utf-8")
    print(format_table(a.task, agg))


if __name__ == "__main__":
    main()
