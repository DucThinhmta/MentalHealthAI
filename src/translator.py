"""Dịch máy Việt <-> Anh chạy trên máy (không cần API), dùng cho:
  1. translate_datasets.py: dịch toàn bộ bộ dữ liệu sang ngôn ngữ còn lại (engine 'local').
  2. Ứng dụng: khi KHÔNG có ANTHROPIC_API_KEY, tin nhắn tiếng Việt được dịch sang tiếng Anh bằng mô hình này
     trước khi đưa vào ba mô hình BERT (trước đây tin nhắn tiếng Việt khi đó không được phân tích).

Hai họ mô hình (tải một lần về models/translator/):
    opus   Helsinki-NLP/opus-mt-{en-vi,vi-en}           ~290 MB mỗi chiều, Apache-2.0, nhanh, chất lượng khá
    vinai  vinai/vinai-translate-{en2vi,vi2en}-v2      ~1,7 GB mỗi chiều, AGPL-3.0, dịch tiếng Việt tốt hơn, chậm hơn

    python src/translator.py --download opus            # tải cả hai chiều
    python src/translator.py --download vinai --direction vi-en
    python src/translator.py "Mình mất ngủ cả tuần nay"  # thử dịch
Ứng dụng tự dùng mô hình đã tải (ưu tiên theo biến môi trường MHAI_TRANSLATOR, mặc định: vinai rồi opus).
"""
import argparse
import os
import re

import config

FAMILIES = {
    "opus": {"en-vi": "Helsinki-NLP/opus-mt-en-vi", "vi-en": "Helsinki-NLP/opus-mt-vi-en"},
    "vinai": {"en-vi": "vinai/vinai-translate-en2vi-v2", "vi-en": "vinai/vinai-translate-vi2en-v2"},
}
_VINAI_CODES = {"en": "en_XX", "vi": "vi_VN"}
TRANSLATOR_DIR = config.PROJECT_DIR / "models" / "translator"
MAX_CHUNK_WORDS = 80                    # câu dài được cắt theo câu rồi gom lại để không vượt giới hạn của mô hình dịch


def local_path(family, direction):
    return TRANSLATOR_DIR / FAMILIES[family][direction].split("/")[-1]


def available(direction="vi-en"):
    """Họ mô hình đã tải cho chiều này (theo thứ tự ưu tiên), hoặc None."""
    order = [os.getenv("MHAI_TRANSLATOR")] if os.getenv("MHAI_TRANSLATOR") else ["vinai", "opus"]
    for fam in order:
        if fam in FAMILIES and (local_path(fam, direction) / "config.json").exists():
            return fam
    return None


def split_for_translation(text, max_words=MAX_CHUNK_WORDS):
    """Cắt văn bản dài thành các đoạn <= max_words từ, ưu tiên ranh giới câu. Ghép các đoạn lại = văn bản gốc."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    chunks, cur = [], []
    for sent in sentences:
        words = sent.split()
        while len(words) > max_words:                     # một câu quá dài: cắt cứng theo số từ
            if cur:
                chunks.append(" ".join(cur)); cur = []
            chunks.append(" ".join(words[:max_words])); words = words[max_words:]
        if cur and len(cur) + len(words) > max_words:
            chunks.append(" ".join(cur)); cur = []
        cur += words
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def normalize_source(text, src):
    """Đưa câu mạng xã hội về dạng mô hình dịch quen hơn: 'i' -> 'I', viết hoa chữ đầu, có dấu kết câu.
    (dair-ai/emotion toàn chữ thường, không dấu chấm, làm Opus-MT dịch hỏng, kiểu '♪ ... ♪')."""
    t = re.sub(r"\s+([.,!?])", r"\1", re.sub(r"\s+", " ", text or "")).strip()
    if not t:
        return t
    if src == "en":
        t = re.sub(r"\bi\b", "I", t)
        t = re.sub(r"\b(im|ive|ill|id)\b", lambda m: {"im": "I'm", "ive": "I've", "ill": "I'll", "id": "I'd"}[m.group(1)], t)
        # dair-ai/emotion tách sẵn "can t", "don t", "it s": ghép lại để mô hình không làm mất phủ định
        t = re.sub(r"\b(can|don|doesn|didn|isn|wasn|aren|weren|won|wouldn|couldn|shouldn|haven|hasn|hadn|ain) t\b",
                   r"\1't", t, flags=re.I)
        t = re.sub(r"\b(I|you|we|they|he|she|it|that|there|what|who) (ll|ve|re|d|m|s)\b", r"\1'\2", t, flags=re.I)
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?…)\"'" and not re.search(r"[\U0001F300-\U0001FAFF:;]\W*$", t):
        t += "."
    return t


def quality_ok(source, output):
    """Loại bản dịch hỏng hay gặp của dịch máy: ký hiệu phụ đề, lặp từ, độ dài vô lý, rỗng."""
    out = (output or "").strip()
    if not out or "♪" in out:
        return False
    words = re.findall(r"\w+", out.lower())
    if any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2)):
        return False
    if len(words) >= 6 and len(set(words)) / len(words) < 0.35:
        return False
    n_src = max(len(re.findall(r"\w+", source or "")), 1)
    return 0.4 <= len(words) / n_src <= 2.6 or n_src < 4


class LocalTranslator:
    """Bọc mô hình seq2seq của Hugging Face; dịch theo lô, tự cắt văn bản dài."""

    def __init__(self, family="opus", direction="vi-en", device=None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.family, self.direction = family, direction
        src, tgt = direction.split("-")
        path = local_path(family, direction)
        name = str(path) if path.exists() else FAMILIES[family][direction]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"src_lang": _VINAI_CODES[src]} if family == "vinai" else {}
        self.tok = AutoTokenizer.from_pretrained(name, **kw)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to(self.device).eval()
        self.gen_kw = ({"decoder_start_token_id": self.tok.convert_tokens_to_ids(_VINAI_CODES[tgt])}
                       if family == "vinai" else {})
        self._torch = torch

    def _translate_flat(self, pieces, batch_size, beams):
        # Xếp theo độ dài trước khi chia lô: ít phần đệm thừa -> nhanh hơn nhiều trên CPU.
        order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
        out = [""] * len(pieces)
        with self._torch.inference_mode():
            for i in range(0, len(order), batch_size):
                idx = order[i:i + batch_size]
                enc = self.tok([pieces[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                               max_length=256).to(self.device)
                gen = self.model.generate(**enc, num_beams=beams, max_new_tokens=320, no_repeat_ngram_size=3,
                                          repetition_penalty=1.2, **self.gen_kw)
                for j, t in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True)):
                    out[j] = t.strip()
        return out

    def translate(self, texts, batch_size=16, beams=4):
        """Danh sách bản dịch; chuỗi rỗng = bản dịch bị loại vì hỏng (xem quality_ok)."""
        src = self.direction.split("-")[0]
        chunks = [[normalize_source(c, src) for c in split_for_translation(t)] for t in texts]
        flat = [c for cs in chunks for c in cs]
        done = iter(self._translate_flat(flat, batch_size, beams)) if flat else iter([])
        out = []
        for cs in chunks:
            parts = [(c, next(done)) for c in cs]
            out.append(" ".join(p for _, p in parts) if all(quality_ok(c, p) for c, p in parts) else "")
        return out


_runtime = {}


def runtime_vi2en(text):
    """Dịch một tin nhắn Việt -> Anh bằng mô hình ĐÃ TẢI (không tự tải khi đang chạy). None nếu chưa có mô hình."""
    fam = available("vi-en")
    if fam is None:
        return None
    if fam not in _runtime:
        _runtime[fam] = LocalTranslator(fam, "vi-en")
    return _runtime[fam].translate([text], batch_size=8)[0] or None


def download(family, directions):
    from huggingface_hub import snapshot_download
    for d in directions:
        dest = local_path(family, d)
        print("Đang tải", FAMILIES[family][d], "->", dest)
        snapshot_download(repo_id=FAMILIES[family][d], local_dir=str(dest),
                          ignore_patterns=["*.h5", "*.msgpack", "*.ot", "*.onnx", "rust_model*"])   # chỉ cần bản PyTorch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--download", choices=list(FAMILIES))
    ap.add_argument("--direction", choices=["en-vi", "vi-en", "both"], default="both")
    ap.add_argument("--family", choices=list(FAMILIES), default=None)
    a = ap.parse_args()
    dirs = ["en-vi", "vi-en"] if a.direction == "both" else [a.direction]
    if a.download:
        download(a.download, dirs)
    if a.text:
        d = "vi-en" if a.direction in ("both", "vi-en") else "en-vi"
        fam = a.family or available(d) or "opus"
        print(LocalTranslator(fam, d).translate([a.text])[0])


if __name__ == "__main__":
    main()
