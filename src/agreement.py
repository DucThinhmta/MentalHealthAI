"""Độ đồng thuận giữa hai người gán nhãn và tạo tập vàng.

    python src/agreement.py an linh                  # kappa theo từng trường; ghi data/annotation/adjudication.csv (các mục bất đồng)
    python src/agreement.py an linh --build-gold     # sau khi hai người điền cột final_* ('-' = thống nhất để trống)

Mục tiêu tham khảo: kappa >= 0,6 cho stress/nguy cơ tự hại thì tập nhãn đủ tin cậy để làm chuẩn; nếu thấp hơn, hãy xem lại hướng dẫn
gán nhãn (docs/nghien_cuu/ANNOTATION_GUIDE.md) và gán lại một mẫu nhỏ trước khi gán hàng loạt.
"""
import argparse

import pandas as pd

import annotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotator_a")
    ap.add_argument("annotator_b")
    ap.add_argument("--build-gold", action="store_true")
    a = ap.parse_args()
    df_a, df_b = annotation.load_annotations(a.annotator_a), annotation.load_annotations(a.annotator_b)
    common = len(set(df_a["id"]) & set(df_b["id"]))
    print(f"{a.annotator_a}: {len(df_a)} mục; {a.annotator_b}: {len(df_b)} mục; cả hai cùng gán: {common}")
    if not common:
        raise SystemExit("Hai người chưa có mục nào trùng nhau.")
    adj_path = annotation.ANNOTATION_DIR / "adjudication.csv"
    if a.build_gold:
        adjudicated = pd.read_csv(adj_path, dtype=str, keep_default_na=False) if adj_path.exists() else None
        gold, dropped = annotation.build_gold(df_a, df_b, adjudicated)
        annotation.GOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
        gold.to_csv(annotation.GOLD_FILE, index=False, encoding="utf-8-sig")
        print(f"Tập vàng: {len(gold)} mục -> {annotation.GOLD_FILE}; loại {dropped} mục chưa thống nhất.")
        return
    table = annotation.pair_agreement(df_a, df_b)
    print(table.round(3).to_string(index=False))
    dis = annotation.disagreements(df_a, df_b)
    dis.to_csv(adj_path, index=False, encoding="utf-8-sig")
    print(f"\n{len(dis)} mục bất đồng -> {adj_path} (điền các cột final_*, dùng '-' nếu thống nhất để trống)")


if __name__ == "__main__":
    main()
