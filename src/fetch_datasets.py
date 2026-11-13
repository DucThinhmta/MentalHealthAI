"""Tải các bộ dữ liệu CẢM XÚC bổ sung từ Hugging Face và đổi về nhãn GoEmotions (28 nhãn) của đề tài.

    python src/fetch_datasets.py                    # tải cả ba -> data/extra/emotion_<tên>_{train,dev,test}.csv
    python src/fetch_datasets.py --only uit_vsmec

Vì sao cần: GoEmotions là bình luận Reddit, còn người dùng chatbot viết câu ngắn, ngôi thứ nhất, kể chuyện của mình,
và nhiều người viết tiếng Việt. Ba bộ dưới đây gần với cách người dùng nhắn tin hơn:

| Tên            | Nguồn (Hugging Face)              | Ngôn ngữ | Kiểu văn bản                         | Giấy phép        |
|----------------|-----------------------------------|----------|--------------------------------------|------------------|
| uit_vsmec      | ura-hcmut/UIT-VSMEC               | Việt     | bình luận mạng xã hội, 7 cảm xúc     | CC BY-NC-SA 4.0  |
| dair_emotion   | dair-ai/emotion                   | Anh      | câu ngắn "i feel ...", 6 cảm xúc     | chỉ nghiên cứu   |
| empathetic     | Estwld/empathetic_dialogues_llm   | Anh      | người dùng tự kể một tình huống, 32  | CC BY-NC 4.0     |
|                | (bản parquet của facebook/empathetic_dialogues)                                       |

Trích dẫn: Ho et al. 2019 (UIT-VSMEC, PACLING); Saravia et al. 2018 (CARER, EMNLP); Rashkin et al. 2019
(EmpatheticDialogues, ACL). Đọc điều khoản của từng bộ trước khi chia sẻ lại; tất cả đều KHÔNG dùng thương mại.

Mỗi câu được gán đúng một nhãn GoEmotions (one-hot trong vector 28 chiều). Nhãn không tương ứng rõ (vd. 'nostalgic',
'prepared' của EmpatheticDialogues) bị bỏ, thay vì ép vào một nhãn sai.
Sau đó huấn luyện: python src/train_emotion.py --extra all --base-model xlm-roberta-base   (xem README)
"""
import argparse
import json
import urllib.request

import pandas as pd

import config

E = config.EMOTION_LABELS
SOURCES = {
    "uit_vsmec": {
        "repo": "ura-hcmut/UIT-VSMEC", "config": "default", "lang": "vi",
        "splits": {"train": "train", "dev": "validation", "test": "test"},
        "text": "Sentence", "label": "Emotion",
        "map": {"Enjoyment": "joy", "Sadness": "sadness", "Fear": "fear", "Anger": "anger", "Disgust": "disgust",
                "Surprise": "surprise", "Other": "neutral"},
    },
    "dair_emotion": {
        "repo": "dair-ai/emotion", "config": "split", "lang": "en",
        "splits": {"train": "train", "dev": "validation", "test": "test"},
        "text": "text", "label": "label",
        "map": {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"},
    },
    "empathetic": {
        "repo": "Estwld/empathetic_dialogues_llm", "config": "default", "lang": "en",
        "splits": {"train": "train", "dev": "valid", "test": "test"},
        "text": "situation", "label": "emotion",
        "map": {"sad": "sadness", "lonely": "sadness", "devastated": "grief", "afraid": "fear", "terrified": "fear",
                "anxious": "nervousness", "apprehensive": "nervousness", "angry": "anger", "furious": "anger",
                "annoyed": "annoyance", "disgusted": "disgust", "embarrassed": "embarrassment",
                "ashamed": "embarrassment", "guilty": "remorse", "disappointed": "disappointment",
                "grateful": "gratitude", "proud": "pride", "joyful": "joy", "content": "joy",
                "excited": "excitement", "hopeful": "optimism", "caring": "caring", "surprised": "surprise",
                "impressed": "admiration"},
    },
}


def _parquet_urls(repo):
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/parquet", timeout=60) as r:
        return json.load(r)


def convert(df, spec):
    """DataFrame gốc -> (text, label_vector, lang, goemotion) với nhãn đã ánh xạ; bỏ nhãn không ánh xạ được."""
    texts = df[spec["text"]].astype(str)
    labels = df[spec["label"]].map(spec["map"])
    out = pd.DataFrame({"text": texts.str.replace(r"\s+", " ", regex=True).str.strip(), "goemotion": labels})
    out = out[out["goemotion"].notna() & (out["text"].str.len() > 0)].drop_duplicates("text")
    out["label_vector"] = out["goemotion"].map(lambda l: str([1.0 if e == l else 0.0 for e in E]))
    out["lang"] = spec["lang"]
    return out[["text", "label_vector", "lang", "goemotion"]]


def fetch(name, out_dir):
    spec = SOURCES[name]
    urls = _parquet_urls(spec["repo"])[spec["config"]]
    frames = {split: convert(pd.concat(pd.read_parquet(u) for u in urls[src]), spec)
              for split, src in spec["splits"].items()}
    held_out = set(frames["dev"]["text"]) | set(frames["test"]["text"])
    frames["train"] = frames["train"][~frames["train"]["text"].isin(held_out)]      # chống rò rỉ train -> dev/test
    for split, df in frames.items():
        df.to_csv(out_dir / f"emotion_{name}_{split}.csv", index=False, encoding="utf-8")
    print(f"[{name}] " + " ".join(f"{s}={len(d)}" for s, d in frames.items()),
          "| nhãn:", frames["train"]["goemotion"].value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(SOURCES), default=list(SOURCES))
    a = ap.parse_args()
    out_dir = config.EXTRA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.only:
        fetch(name, out_dir)
    print("Đã ghi vào", out_dir)


if __name__ == "__main__":
    main()
