"""Tạo data/annotation/pool.csv: tập tin nhắn tiếng Việt (đã xoá thông tin nhận dạng) để nhiều người gán nhãn độc lập.

    python src/sample_for_annotation.py --n 600
    python src/sample_for_annotation.py --n 400 --logs analysis_log.json --extra-csv my_texts.csv

Nguồn:
  - vsmec : bình luận mạng xã hội tiếng Việt thật (UIT-VSMEC, cần chạy fetch_datasets.py trước). Tiện nhưng KHÔNG giống hội
            thoại với chatbot: chỉ dùng khi chưa có dữ liệu chat thật.
  - logs  : nhật ký ứng dụng (analysis_log.json) của người đã đồng ý lưu bản ghi ẩn danh; ƯU TIÊN nguồn này vì đúng miền dữ liệu.
  - extra : CSV bất kỳ có cột 'text'.
Lấy mẫu cân bằng giữa các nguồn đang có, bỏ trùng, xoá số điện thoại/email/đường dẫn, trộn ngẫu nhiên (seed cố định để tái lập).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import annotation
import config
import datasets_io
import privacy


def vsmec_texts():
    names = datasets_io.extra_names()
    if "uit_vsmec" not in names:
        return []
    frames = [datasets_io.load_extra("emotion", sp, ["uit_vsmec"]) for sp in ("dev", "test")]
    return pd.concat(frames)["text"].tolist()


def log_texts(paths):
    out = []
    for path in paths:
        for entry in json.loads(open(path, encoding="utf-8").read()):
            text = entry.get("user", "")
            if text and text != privacy.NOT_STORED:
                out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--extra-csv", default=None)
    ap.add_argument("--out", default=str(annotation.ANNOTATION_DIR / "pool.csv"))
    a = ap.parse_args()
    sources = {"vsmec": vsmec_texts(), "logs": log_texts(a.logs)}
    if a.extra_csv:
        sources["extra"] = pd.read_csv(a.extra_csv)["text"].astype(str).tolist()
    sources = {k: v for k, v in sources.items() if v}
    if not sources:
        raise SystemExit("Không có nguồn nào. Chạy 'python src/fetch_datasets.py' hoặc truyền --logs / --extra-csv.")
    pool = annotation.sample_pool(sources, a.n, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {len(pool)} mục -> {out}; nguồn: {pool['source'].value_counts().to_dict()}")
    if set(pool["source"]) == {"vsmec"}:
        print("Lưu ý: chỉ có bình luận mạng xã hội; hãy bổ sung nhật ký chat thật khi có để tập vàng đúng miền dữ liệu.")


if __name__ == "__main__":
    main()
