"""Hiệu chỉnh xác suất và ngưỡng thứ bậc TRÊN TẬP DEV, rồi báo cáo tác động trên tập TEST.

    python src/clean_data.py        # cần data/clean/{stress,depression}_dev.csv
    python src/calibrate.py         # ghi models/calibration.json + reports/calibration.json

Không huấn luyện lại mô hình; analyze.py tự đọc models/calibration.json nếu có.
Xem giải thích phương pháp trong calibration.py.
"""
import json

import numpy as np

import calibration as cal
import config
import metrics
import modeling
from datasets_io import load_split

NAMES = {"stress": ["no_stress", "stress"], "depression": config.DEPRESSION_LABELS}


def _probs(task, split):
    df = load_split(task, split)
    tok, model = modeling.load(config.find_model(task))
    probs = modeling.predict_probs(tok, model, df["text"].tolist(), config.MAX_LEN[task])
    return probs, df["label"].to_numpy()


def main():
    spec, report = {"temperature": {}}, {}
    for task in ("stress", "depression"):
        dev_p, dev_y = _probs(task, "dev")
        test_p, test_y = _probs(task, "test")
        t = cal.fit_temperature(dev_p, dev_y)
        spec["temperature"][task] = t
        test_cal = cal.apply_temperature(test_p, t)
        report[task] = {
            "temperature": t,
            "test_ece_before": cal.expected_calibration_error(test_p, test_y),
            "test_ece_after": cal.expected_calibration_error(test_cal, test_y),
            "test_nll_before": cal.nll(test_p, test_y), "test_nll_after": cal.nll(test_cal, test_y),
        }
        if task == "depression":
            cuts, dev_f1 = cal.tune_ordinal_cuts(dev_y, cal.expected_level(cal.apply_temperature(dev_p, t)))
            spec["depression_cuts"] = cuts
            before, per_before, _ = metrics.single_label_report(test_y, test_p.argmax(1), NAMES[task], ordinal=True)
            pred = cal.decode_ordinal(cal.expected_level(test_cal), cuts)
            after, per_after, cm = metrics.single_label_report(test_y, pred, NAMES[task], ordinal=True)
            report[task].update({"cuts": cuts, "dev_macro_f1_with_cuts": dev_f1,
                                 "test_argmax": before, "test_ordinal_cuts": after})
            config.REPORT_DIR.mkdir(exist_ok=True)
            per_after.to_csv(config.REPORT_DIR / "depression_ordinal_per_class.csv", index=False)
            metrics.plot_confusion(cm, NAMES[task], "Ma trận nhầm lẫn - depression (ngưỡng thứ bậc)",
                                   config.REPORT_DIR / "depression_ordinal_confusion.png")
            print("\nDepression theo lớp, argmax:\n", per_before.round(4).to_string(index=False))
            print("\nDepression theo lớp, ngưỡng thứ bậc:\n", per_after.round(4).to_string(index=False))

    config.CALIBRATION_FILE.parent.mkdir(exist_ok=True)
    config.CALIBRATION_FILE.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (config.REPORT_DIR / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nĐã lưu", config.CALIBRATION_FILE)
    print("Lưu ý: ngưỡng chọn trên dev; nếu macro-F1/QWK trên TEST không tăng thì xóa khóa 'depression_cuts' "
          "trong tệp này để quay về argmax, và ghi nhận điều đó trong báo cáo.")


if __name__ == "__main__":
    main()
