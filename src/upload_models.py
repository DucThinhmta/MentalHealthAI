"""ĐẨY 3 mô hình lên Hugging Face (chạy MỘT LẦN trên máy của bạn).

    python src/upload_models.py --user <ten_tai_khoan_HF>            # repo private
    python src/upload_models.py --user <ten_tai_khoan_HF> --public   # repo public

Cần đăng nhập trước:  hf auth login   (token quyền Write)
Chỉ đẩy file cần để chạy (cấu hình, trọng số, tokenizer); KHÔNG đẩy optimizer.pt, scheduler.pt...
"""
import argparse

from huggingface_hub import HfApi

import config

KEEP = ["config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json",
        "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]

# (tên repo trên HF, tên nhánh trong config.find_model)
TARGETS = [("mhai-stress", "stress"), ("mhai-emotion", "emotion"), ("mhai-depression", "depression")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="tên tài khoản Hugging Face")
    ap.add_argument("--public", action="store_true", help="mặc định là private")
    a = ap.parse_args()
    api = HfApi()
    for repo_name, task in TARGETS:
        folder = config.find_model(task)            # tự tìm đúng thư mục mô hình của bạn
        repo_id = f"{a.user}/{repo_name}"
        print(f"\n[{task}] {folder}\n  -> {repo_id}")
        api.create_repo(repo_id, private=not a.public, exist_ok=True)
        api.upload_folder(folder_path=str(folder), repo_id=repo_id, allow_patterns=KEEP,
                          commit_message=f"Upload {task} model")
        print("  xong.")
    print("\nHoàn tất. Nhớ sửa HF_USER trong src/download_models.py thành:", a.user)


if __name__ == "__main__":
    main()
