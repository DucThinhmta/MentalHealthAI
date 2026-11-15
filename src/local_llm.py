"""LLM MIỄN PHÍ chạy trên máy (hoặc trên Colab) qua Ollama, thay cho Anthropic API khi không có khóa trả phí.

    # 1. Cài Ollama (https://ollama.com), tải mô hình (một lần):  ollama pull qwen2.5:7b
    # 2. Bật cho dự án:
    set MHAI_LLM=ollama                 (Windows)   |   export MHAI_LLM=ollama   (Linux/Colab)
    # tuỳ chọn: MHAI_OLLAMA_MODEL=qwen2.5:7b   MHAI_OLLAMA_URL=http://localhost:11434

Vì sao Qwen2.5: mô hình mở, viết tiếng Việt khá, bản 7B lượng tử hoá 4-bit (~4,7 GB) chạy được trên GPU 6 GB (vd. RTX 4050) hoặc GPU T4 của Colab;
máy yếu hơn dùng qwen2.5:3b. Chất lượng THẤP HƠN Claude: mọi kết quả đo bằng mô hình này phải ghi rõ tên mô hình.

Lớp này giả lập đúng phần giao diện `client.messages.create(...)` / `client.messages.stream(...)` mà llm_client.py dùng,
nên phần còn lại của dự án không cần đổi. Không thêm thư viện nào (dùng urllib).
"""
import json
import os
import urllib.request
from types import SimpleNamespace

URL = os.getenv("MHAI_OLLAMA_URL", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("MHAI_OLLAMA_MODEL", "qwen2.5:7b")
TIMEOUT = float(os.getenv("MHAI_OLLAMA_TIMEOUT", "180"))


def enabled() -> bool:
    """Chỉ bật khi người dùng chọn rõ ràng (MHAI_LLM=ollama), để kiểm thử và máy khác không tự dưng gọi LLM."""
    return os.getenv("MHAI_LLM", "").lower() == "ollama"


def reachable(timeout=1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _system_text(system):
    """system của Anthropic có thể là chuỗi hoặc danh sách khối {'type':'text','text':...}."""
    if isinstance(system, str):
        return system
    return "\n\n".join(b.get("text", "") for b in system or [] if isinstance(b, dict))


def _content_text(content):
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text")


def build_payload(model, max_tokens, system=None, messages=(), output_config=None, stream=False, **_ignored):
    """Đổi tham số kiểu Anthropic sang yêu cầu /api/chat của Ollama."""
    msgs = []
    sys_text = _system_text(system)
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})
    msgs += [{"role": m["role"], "content": _content_text(m["content"])} for m in messages]
    payload = {"model": _local_model(model), "messages": msgs, "stream": stream,
               "options": {"num_predict": min(int(max_tokens or 1024), 2048), "temperature": 0.6}}
    fmt = (output_config or {}).get("format")
    if fmt and fmt.get("type") == "json_schema":
        payload["format"] = fmt["schema"]                   # Ollama ép đầu ra theo JSON schema
        payload["options"]["temperature"] = 0
    return payload


def _local_model(model):
    """Tên mô hình Claude (mặc định của llm_client) không có trên Ollama: dùng mô hình cục bộ đã chọn."""
    return DEFAULT_MODEL if not model or model.startswith("claude") else model


def _post(payload):
    req = urllib.request.Request(f"{URL}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def _as_message(text):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


class _Stream:
    def __init__(self, payload):
        self._payload = {**payload, "stream": True}
        self._resp = None

    def __enter__(self):
        self._resp = _post(self._payload)
        return self

    def __exit__(self, *exc):
        if self._resp is not None:
            self._resp.close()
        return False

    @property
    def text_stream(self):
        for line in self._resp:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break


class _Messages:
    def create(self, **kw):
        with _post(build_payload(**kw)) as r:
            data = json.loads(r.read().decode("utf-8"))
        return _as_message(data.get("message", {}).get("content", "").strip())

    def stream(self, **kw):
        return _Stream(build_payload(**kw))


class OllamaClient:
    """Đối tượng thay cho anthropic.Anthropic() (chỉ phần messages.create / messages.stream)."""
    def __init__(self):
        self.messages = _Messages()
