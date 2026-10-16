"""TẢI 3 mô hình từ Hugging Face về thư mục models/ (người dùng khác chạy bước này).

    python src/download_models.py
Nếu repo mô hình ở chế độ private: chạy `hf auth login` trước.
"""
from huggingface_hub import snapshot_download

import config

HF_USER = "NguyenDucThinh06" 

TARGETS = {
    f"{HF_USER}/mhai-stress": "bert_stress_final",
    f"{HF_USER}/mhai-depression": "bert_depression_final",
    f"{HF_USER}/mhai-emotion": "bert_emotion/checkpoint-5427",
}


def main():
    if HF_USER.startswith("SUA_THANH"):
        raise SystemExit("Hãy mở src/download_models.py và sửa HF_USER thành tên tài khoản Hugging Face.")
    for repo, subdir in TARGETS.items():
        dest = config.PROJECT_DIR / "models" / subdir
        if dest.exists() and any(dest.iterdir()):
            print("Đã có, bỏ qua:", dest)
            continue
        print("Đang tải", repo, "->", dest)
        snapshot_download(repo_id=repo, local_dir=str(dest))
    print("Xong. Chạy: streamlit run src/app.py")


if __name__ == "__main__":
    main()
