"""Dịch TẤT CẢ bộ dữ liệu sang ngôn ngữ còn lại (Anh -> Việt, Việt -> Anh), giữ nguyên nhãn.

Mục đích: huấn luyện mô hình song ngữ (vd. xlm-roberta-base với --bilingual) để tin nhắn tiếng Việt được phân tích
trực tiếp và chính xác hơn, thay vì chỉ dựa vào bước dịch lúc chạy.

    python src/translate_datasets.py --list                       # xem có những bộ nào, bao nhiêu câu
    python src/translate_datasets.py --estimate                   # ước tính chi phí nếu dịch bằng Claude
    python src/translate_datasets.py --engine local               # dịch trên máy (cần: python src/translator.py --download opus)
    python src/translate_datasets.py --engine local --limit 50    # thử nhanh 50 câu mỗi tệp
    python src/translate_datasets.py --engine claude --submit     # gửi Claude Message Batches (giảm 50% giá, cần API key)
    python src/translate_datasets.py --engine claude --collect    # lấy kết quả khi batch xong (chạy lại được nhiều lần)

Kết quả: data/translated/<task>_<bộ>_<split>.<ngôn ngữ đích>.csv (cột text = bản dịch, original_text = câu gốc).
Bản dịch được lưu đệm theo từng câu (data/translated/cache/), chạy lại chỉ dịch phần còn thiếu.

Lưu ý phương pháp (nên ghi trong báo cáo): tập TEST dịch máy chỉ là thước đo gần đúng cho tiếng Việt; nhãn được
giữ nguyên từ câu gốc nên có thể sai khi bản dịch làm lệch sắc thái. Nên kiểm tra tay một mẫu ngẫu nhiên.
"""
import argparse
import hashlib
import json
import time

import pandas as pd

import config
import datasets_io

OUT_DIR = config.DATA_DIR / "translated"
CACHE_DIR = OUT_DIR / "cache"
BATCH_STATE = OUT_DIR / "claude_batches.json"
REJECTED = [0]                          # số câu có bản dịch bị loại (rỗng) ở lần write_outputs gần nhất
TASKS = ("stress", "depression", "emotion")
SPLITS = ("train", "dev", "test")
# Giá Claude API (USD / 1 triệu token, chưa giảm giá batch), xem bảng giá chính thức trước khi chạy.
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00), "claude-haiku-5-5": (0.10, 0.50)}
SYSTEM = """You translate text for a research dataset on stress, emotion and mental health. Translate each item from \
{src} to {tgt}. Keep the meaning, emotional tone, intensity, slang, profanity, emoji and first-person voice exactly; do \
not soften, censor, summarize, explain or add anything. Translate naturally, the way a native speaker would write a \
social media post or chat message. Return every item with its id."""
SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
        "required": ["id", "text"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False,
}
LANG_NAME = {"en": "English", "vi": "Vietnamese"}


# --------------------------------------------------------------------------- nguồn dữ liệu
def sources():
    """[(task, tên bộ, split, ngôn ngữ gốc, DataFrame)] cho mọi tệp đang có trên máy."""
    found = []
    for task in TASKS:
        for split in SPLITS:
            try:
                df = datasets_io.load_split(task, split, required=False)
            except FileNotFoundError:
                df = None
            if df is not None:
                found.append((task, "core", split, "en", df))
        for name in datasets_io.extra_names(task):
            for split in SPLITS:
                df = datasets_io.load_extra(task, split, [name])
                lang = df["lang"].iloc[0] if "lang" in df and len(df) else "en"
                found.append((task, name, split, lang, df))
    return found


def target(lang):
    return "vi" if lang == "en" else "en"


def _key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- bộ nhớ đệm
def cache_path(src, tgt):
    return CACHE_DIR / f"{src}-{tgt}.jsonl"


def load_cache(src, tgt):
    path, cache = cache_path(src, tgt), {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["h"]] = row["t"]
    return cache


def append_cache(src, tgt, pairs, engine):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path(src, tgt), "a", encoding="utf-8") as f:
        for text, tr in pairs:
            f.write(json.dumps({"h": _key(text), "t": tr, "engine": engine}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- ghi kết quả
def write_outputs(found, limit=None):
    """Ghi các tệp đã dịch từ bộ nhớ đệm; trả về số câu còn thiếu bản dịch."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing_total = 0
    REJECTED[0] = 0
    caches = {}
    for task, name, split, lang, df in found:
        tgt = target(lang)
        cache = caches.setdefault((lang, tgt), load_cache(lang, tgt))
        df = df.head(limit) if limit else df
        tr = df["text"].map(lambda t: cache.get(_key(t)))
        missing_total += int(tr.isna().sum())
        usable = tr.notna() & (tr.fillna("").str.strip() != "")          # "" = bản dịch bị bộ lọc chất lượng loại
        REJECTED[0] += int((tr.notna() & ~usable).sum())
        out = df.loc[usable].copy()
        if out.empty:
            continue
        out["original_text"], out["text"] = out["text"], tr[usable]
        out["lang"], out["translated_from"] = tgt, lang
        cols = [c for c in out.columns if c != "labels"]                 # 'labels' là cột tính lại khi đọc
        out[cols].to_csv(OUT_DIR / f"{task}_{name}_{split}.{tgt}.csv", index=False, encoding="utf-8")
    return missing_total


def pending(found, limit=None):
    """{(src, tgt): [câu chưa dịch, không trùng]}"""
    todo, caches = {}, {}
    for task, name, split, lang, df in found:
        tgt = target(lang)
        cache = caches.setdefault((lang, tgt), load_cache(lang, tgt))
        texts = (df.head(limit) if limit else df)["text"].tolist()
        bucket = todo.setdefault((lang, tgt), {})
        for t in texts:
            if t and _key(t) not in cache:
                bucket[_key(t)] = t
    return {k: list(v.values()) for k, v in todo.items() if v}


# --------------------------------------------------------------------------- engine: local
def run_local(todo, family, batch_size, beams=1, save_every=256):
    import translator
    for (src, tgt), texts in todo.items():
        direction = f"{src}-{tgt}"
        fam = family or translator.available(direction) or "opus"
        print(f"[local/{fam}] {direction}: {len(texts)} câu")
        model = translator.LocalTranslator(fam, direction)
        texts = sorted(texts, key=len)                    # câu dài gần nhau -> chia lô hiệu quả hơn
        t0 = time.time()
        for i in range(0, len(texts), save_every):
            part = texts[i:i + save_every]
            append_cache(src, tgt, zip(part, model.translate(part, batch_size, beams)), f"local:{fam}")
            done = i + len(part)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(texts)}  ~{rate:.1f} câu/giây, còn ~{(len(texts) - done) / rate / 60:.0f} phút")


# --------------------------------------------------------------------------- engine: claude (Message Batches)
def group(texts, max_items=20, max_chars=6000):
    """Gom nhiều câu vào một request để giảm chi phí lặp lại system prompt."""
    groups, cur, size = [], [], 0
    for t in texts:
        if cur and (len(cur) >= max_items or size + len(t) > max_chars):
            groups.append(cur); cur, size = [], 0
        cur.append(t); size += len(t)
    if cur:
        groups.append(cur)
    return groups


def estimate(todo, model):
    """Ước tính token và chi phí (USD) khi dịch bằng Claude Message Batches (đã giảm 50%)."""
    n_req = n_in = n_out = 0
    for (src, tgt), texts in todo.items():
        per_tok_src = 3.6 if src == "en" else 2.4                     # số ký tự trung bình / token (ước lượng)
        per_tok_tgt = 2.4 if tgt == "vi" else 3.6
        for g in group(texts):
            chars = sum(len(t) for t in g)
            n_req += 1
            n_in += 230 + chars / per_tok_src + 12 * len(g)            # system + nội dung + khung JSON
            n_out += (chars / per_tok_tgt) * 1.15 + 15 * len(g)        # bản dịch + khung JSON + chút suy nghĩ
    pin, pout = PRICES.get(model, PRICES["claude-sonnet-5-5"])
    cost = (n_in * pin + n_out * pout) / 1e6 * 0.5
    return {"requests": n_req, "input_tokens": int(n_in), "output_tokens": int(n_out), "usd": round(cost, 4)}


def submit_claude(todo, model):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    state = json.loads(BATCH_STATE.read_text(encoding="utf-8")) if BATCH_STATE.exists() else {}
    for (src, tgt), texts in todo.items():
        groups = group(texts)
        for start in range(0, len(groups), 50_000):                   # giới hạn 100k request/batch
            chunk = groups[start:start + 50_000]
            reqs, meta = [], {}
            for gi, g in enumerate(chunk):
                cid = f"{src}-{tgt}-{start + gi}"
                meta[cid] = g
                payload = json.dumps([{"id": i, "text": t} for i, t in enumerate(g)], ensure_ascii=False)
                reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
                    model=model, max_tokens=16000,
                    system=SYSTEM.format(src=LANG_NAME[src], tgt=LANG_NAME[tgt]),
                    messages=[{"role": "user", "content": payload}],
                    output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})))
            batch = client.messages.batches.create(requests=reqs)
            state[batch.id] = {"src": src, "tgt": tgt, "model": model, "groups": meta, "collected": False}
            print(f"Đã gửi batch {batch.id}: {len(reqs)} request ({src}->{tgt})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_STATE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def parse_items(text, originals):
    """JSON {'items': [{'id', 'text'}]} -> [(câu gốc, bản dịch)] cho các id hợp lệ."""
    items = json.loads(text)["items"]
    out = []
    for it in items:
        i = it.get("id")
        if isinstance(i, int) and 0 <= i < len(originals) and str(it.get("text", "")).strip():
            out.append((originals[i], it["text"].strip()))
    return out


def collect_claude(wait):
    import anthropic
    client = anthropic.Anthropic()
    state = json.loads(BATCH_STATE.read_text(encoding="utf-8"))
    for bid, info in state.items():
        if info["collected"]:
            continue
        batch = client.messages.batches.retrieve(bid)
        while wait and batch.processing_status != "ended":
            print(f"{bid}: {batch.processing_status}, chờ 60 giây...")
            time.sleep(60)
            batch = client.messages.batches.retrieve(bid)
        if batch.processing_status != "ended":
            print(f"{bid}: chưa xong ({batch.processing_status})")
            continue
        ok = bad = 0
        for res in client.messages.batches.results(bid):
            g = info["groups"].get(res.custom_id, [])
            msg = res.result.message if res.result.type == "succeeded" else None
            if msg is None or msg.stop_reason != "end_turn":
                bad += 1
                continue
            text = next((b.text for b in msg.content if b.type == "text"), "")
            try:
                pairs = parse_items(text, g)
            except (ValueError, KeyError):
                bad += 1
                continue
            append_cache(info["src"], info["tgt"], pairs, f"claude:{info['model']}")
            ok += 1
        info["collected"] = True
        print(f"{bid}: {ok} request thành công, {bad} lỗi (chạy lại --submit để dịch phần còn thiếu)")
    BATCH_STATE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["local", "claude"])
    ap.add_argument("--family", choices=["opus", "vinai"], default=None, help="engine local: họ mô hình dịch")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--beams", type=int, default=1, help="engine local: 1 = nhanh (mặc định), 2-4 = kỹ hơn nhưng chậm")
    ap.add_argument("--model", default=None, help="engine claude: mặc định theo ANTHROPIC_MODEL")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--wait", action="store_true", help="--collect: chờ tới khi batch xong")
    ap.add_argument("--limit", type=int, default=None, help="chỉ dịch N câu đầu mỗi tệp (để thử)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args()

    found = sources()
    if not found:
        raise SystemExit("Không thấy bộ dữ liệu nào. Đặt dữ liệu vào data/ (xem data/README.md) hoặc chạy fetch_datasets.py.")
    todo = pending(found, a.limit)
    if a.list or not (a.engine or a.estimate):
        for task, name, split, lang, df in found:
            print(f"{task:<10} {name:<13} {split:<5} {lang}->{target(lang)}  {len(df):>6} câu")
        print("Chưa dịch:", {f"{s}->{t}": len(v) for (s, t), v in todo.items()} or "không còn")
    if a.estimate:
        import llm_client
        for m in ([a.model] if a.model else [llm_client.MODEL, "claude-haiku-5-5"]):
            print(f"Ước tính với {m} (Message Batches, -50%):", estimate(todo, m))
    if a.engine == "local" and todo:
        run_local(todo, a.family, a.batch_size, a.beams)
    if a.engine == "claude":
        import llm_client
        if a.collect:
            collect_claude(a.wait)
        elif a.submit and todo:
            if not llm_client.is_available():
                raise SystemExit("Cần ANTHROPIC_API_KEY để dịch bằng Claude.")
            model = a.model or llm_client.MODEL
            print("Chi phí ước tính:", estimate(todo, model))
            submit_claude(todo, model)
            print("Khi batch xong (thường dưới 1 giờ): python src/translate_datasets.py --engine claude --collect")
    if a.engine or a.collect:
        left = write_outputs(found, a.limit)
        print(f"Đã ghi vào {OUT_DIR}. Còn {left} câu chưa có bản dịch; {REJECTED[0]} câu bị loại vì bản dịch hỏng.")


if __name__ == "__main__":
    main()
