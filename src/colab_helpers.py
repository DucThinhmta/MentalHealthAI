"""Tiện ích để chạy cả dự án trên Google Colab (đám mây), để thành viên nhóm không phải tải mô hình/dữ liệu về máy.

Dùng trong notebook (xem notebooks/run_app_colab.ipynb và train_and_evaluate_colab.ipynb):
    import colab_helpers as ch
    ch.sync_dirs([("reports", DRIVE / "reports")], "to_drive")      # sao lưu kết quả lên Drive (Colab có thể ngắt bất cứ lúc nào)
    url, proc = ch.start_tunnel(8501)                               # liên kết web tạm thời tới ứng dụng đang chạy

Không phụ thuộc Colab nên có thể kiểm thử trên máy thường.
"""
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
_TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
HEAVY_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")          # trọng số mô hình: nặng, mặc định không sao lưu


def sync_dirs(pairs, direction="to_drive", skip_suffixes=HEAVY_SUFFIXES, only_newer=True):
    """Sao chép giữa thư mục trong Colab và Drive. pairs = [(thư mục cục bộ, thư mục Drive), ...].

    direction='to_drive': cục bộ -> Drive (sao lưu); 'from_drive': Drive -> cục bộ (khôi phục khi phiên Colab mới bắt đầu).
    Chỉ chép tệp mới hơn hoặc chưa có. Bỏ qua trọng số mô hình (mặc định) để Drive không đầy; truyền skip_suffixes=() để chép cả.
    Trả số tệp đã chép."""
    if direction not in ("to_drive", "from_drive"):
        raise ValueError("direction phải là 'to_drive' hoặc 'from_drive'")
    copied = 0
    for local, drive in pairs:
        src, dst = (Path(local), Path(drive)) if direction == "to_drive" else (Path(drive), Path(local))
        if not src.exists():
            continue
        for path in src.rglob("*"):
            if not path.is_file() or path.suffix.lower() in skip_suffixes:
                continue
            target = dst / path.relative_to(src)
            if only_newer and target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    return copied


def extract_tunnel_url(text):
    """Địa chỉ https://....trycloudflare.com trong một đoạn nhật ký của cloudflared (None nếu chưa có)."""
    m = _TUNNEL_RE.search(text or "")
    return m.group(0) if m else None


def ensure_cloudflared(path="/tmp/cloudflared"):
    """Tải cloudflared (một tệp, vài chục MB, nằm trên đĩa của Colab chứ không phải máy bạn) nếu chưa có."""
    p = Path(path)
    if not p.exists():
        urllib.request.urlretrieve(CLOUDFLARED_URL, p)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def wait_for_port(port, timeout=90):
    """Chờ ứng dụng nghe cổng `port`. True nếu sẵn sàng."""
    import socket
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    return False


def start_app(port=8501, app="src/app.py", env=None):
    """Chạy Streamlit nền trong Colab. Trả tiến trình."""
    cmd = [sys.executable, "-m", "streamlit", "run", app, "--server.port", str(port), "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    return subprocess.Popen(cmd, env={**os.environ, **(env or {})}, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def start_tunnel(port=8501, timeout=60):
    """Mở liên kết web tạm thời tới ứng dụng (cloudflared, không cần tài khoản). -> (url, tiến trình).

    LƯU Ý QUYỀN RIÊNG TƯ: ai có liên kết đều mở được ứng dụng và nội dung trò chuyện đi qua máy chủ Colab. Chỉ chia sẻ liên kết cho
    người đã đồng ý tham gia, tắt khi xong, và không dùng liên kết này để thu dữ liệu nghiên cứu thật khi chưa có phê duyệt đạo đức."""
    exe = ensure_cloudflared()
    proc = subprocess.Popen([exe, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    end = time.time() + timeout
    while time.time() < end:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        url = extract_tunnel_url(line)
        if url:
            return url, proc
    proc.terminate()
    raise RuntimeError("Không mở được liên kết web (cloudflared). Chạy lại ô này, hoặc dùng cách khác trong docs/COLAB.md.")


def secret(name, default=None):
    """Đọc khoá từ Colab Secrets (biểu tượng chìa khoá bên trái) hoặc biến môi trường; không bao giờ in ra."""
    try:
        from google.colab import userdata
        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)
