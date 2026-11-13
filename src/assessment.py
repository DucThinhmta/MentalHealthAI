"""Đánh giá ở mức CẢ CUỘC TRÒ CHUYỆN (thay vì từng câu) + các khía cạnh bot hỏi thêm để thu thập thông tin.

Ý tưởng: một câu nói không đủ để nhận xét về một người. Hệ thống gom bằng chứng từ ba nguồn qua nhiều lượt:
  1. Tín hiệu mô hình (stress, mức độ nặng nề, cảm xúc) của TỪNG tin nhắn, gộp lại có trọng số theo độ dài.
  2. Các khía cạnh sinh hoạt hỏi được qua trò chuyện (giấc ngủ, năng lượng, hứng thú, ăn uống, tập trung,
     người để chia sẻ, tâm trạng, thời gian kéo dài). Danh sách lấy cảm hứng từ các nhóm triệu chứng mà bảng
     sàng lọc PHQ-9 / GAD-7 hỏi tới, nhưng ở đây chỉ là trò chuyện, KHÔNG phải bảng hỏi đã chuẩn hóa.
  3. Tâm trạng người dùng tự chấm (1-5) nếu có.
Chỉ đưa ra nhận xét khi đủ dữ liệu (evidence), và luôn ghi rõ đây không phải chẩn đoán.

Các trọng số trong CONCERN_WEIGHTS là HEURISTIC do nhóm đặt, cần được kiểm chứng (vd. so với đánh giá của
chuyên gia trên một tập hội thoại có nhãn) trước khi dùng để kết luận trong báo cáo.
Module chỉ cần numpy, không cần mô hình -> kiểm thử được độc lập.
"""
import re
from collections import Counter

import numpy as np

import calibration
import config
from textutils import strip_accents

# --------------------------------------------------------------------------- khía cạnh (domain)
# patterns: regex trên văn bản ĐÃ BỎ DẤU, chữ thường. 'concern' = đáng chú ý, 'ok' = ổn.
# Với 'duration': concern = kéo dài (nhiều tuần), ok = mới xảy ra gần đây.
DOMAINS = {
    "mood": {
        "name": "Tâm trạng", "icon": "🌦️",
        "concern": r"buon|chan nan|that vong|tuyet vong|vo vong|khoc|nang ne|trong rong|u uat|suy sup|tieu cuc|"
                   r"khong (?:on|vui|tot)|te (?:qua|lam)|sad|down|depressed|hopeless|empty|miserable|crying|awful|"
                   r"not (?:ok|okay|good|fine|great)",
        "ok": r"(?:minh|toi|em|to|van|cung|kha|deu|thay) on\b(?! (?:thi|bai|tap|lai|luyen))|on (?:ma|roi|thoi|lam)|vui ve|thoai mai|"
              r"(?:tam trang|thay|van) binh thuong|"
              r"yeu doi|hanh phuc|khong buon|i'?m (?:good|fine|okay|ok|great|happy)|feel(?:ing)? (?:good|fine|great|happy)",
        "ask": {"vi": ["Mấy hôm nay tâm trạng chung của bạn thế nào?",
                       "Nếu chấm tâm trạng tuần này của bạn thì bạn thấy nó nghiêng về vui hay buồn hơn?"],
                "en": ["How has your mood been overall these days?",
                       "If you had to rate your mood this week, would it lean more good or bad?"]},
        "quick": {"vi": [("Khá ổn", "ok"), ("Lúc lên lúc xuống", "concern"), ("Buồn nhiều", "concern")],
                  "en": [("Pretty okay", "ok"), ("Up and down", "concern"), ("Mostly low", "concern")]},
        "probe": r"tam trang|cam thay the nao|mood|feeling overall",
    },
    "sleep": {
        "name": "Giấc ngủ", "icon": "🌙",
        "concern": r"mat ngu|kho ngu|khong ngu duoc|ngu khong (?:ngon|duoc)|thuc trang|thuc khuya|tinh giac|ngu it|"
                   r"it ngu|ac mong|trang dem|ngu (?:nhieu|ca ngay|li bi)|insomnia|can'?t sleep|cannot sleep|"
                   r"couldn'?t sleep|not sleeping|trouble sleeping|nightmares?|barely slept|sleep(?:ing)? badly",
        "ok": r"ngu (?:ngon|du|duoc|tot|binh thuong|van on)|khong (?:bi )?mat ngu|slept well|sleep(?:ing)? "
              r"(?:well|fine|ok|okay)",
        "ask": {"vi": ["Dạo này bạn ngủ thế nào?", "Mấy đêm gần đây bạn có ngủ được không?"],
                "en": ["How have you been sleeping lately?", "Have you been able to sleep these past few nights?"]},
        "quick": {"vi": [("Ngủ ổn", "ok"), ("Khó ngủ", "concern"), ("Hay thức giấc giữa đêm", "concern")],
                  "en": [("Sleeping fine", "ok"), ("Hard to fall asleep", "concern"), ("Waking up at night", "concern")]},
        "probe": r"\bngu\b|sleep",
    },
    "energy": {
        "name": "Năng lượng", "icon": "🔋",
        "concern": r"met moi|\bmet\b|kiet suc|ue oai|duoi suc|khong (?:co|con) suc|uot suc|tired|exhausted|drained|"
                   r"no energy|worn out|fatigue|burn(?:ed|t)? out",
        "ok": r"khong met|van khoe|(?:nhieu|tran day|day) nang luong|kha khoe|not tired|energetic|full of energy",
        "ask": {"vi": ["Mấy hôm nay bạn thấy người có còn sức không, hay hay mệt?",
                       "Năng lượng của bạn dạo này thế nào?"],
                "en": ["Have you had much energy these days, or are you often tired?",
                       "How's your energy been lately?"]},
        "quick": {"vi": [("Vẫn khỏe", "ok"), ("Hơi mệt", "concern"), ("Kiệt sức", "concern")],
                  "en": [("I'm fine", "ok"), ("A bit tired", "concern"), ("Exhausted", "concern")]},
        "probe": r"\bmet\b|nang luong|con suc|energy|tired",
    },
    "interest": {
        "name": "Hứng thú", "icon": "🎨",
        "concern": r"khong (?:con )?(?:hung thu|thich gi|muon lam gi)|mat hung thu|chang muon lam gi|"
                   r"chan (?:het|moi thu|tat ca)|khong thiet|buong xuoi|lost interest|no interest|don'?t enjoy|"
                   r"nothing (?:is )?fun|don'?t feel like doing anything|no motivation|unmotivated",
        "ok": r"van (?:thich|choi|hung thu|di choi)|co hung thu|hao hung|dam me|still enjoy|looking forward|"
              r"excited about",
        "ask": {"vi": ["Những việc bạn hay thích làm, dạo này bạn còn thấy vui khi làm không?",
                       "Gần đây có việc gì bạn vẫn thấy thích làm không?"],
                "en": ["The things you usually enjoy, do they still feel good lately?",
                       "Is there anything you still enjoy doing these days?"]},
        "quick": {"vi": [("Vẫn thích như thường", "ok"), ("Ít hứng thú hơn", "concern"), ("Chẳng muốn làm gì", "concern")],
                  "en": [("Still enjoy them", "ok"), ("Less than before", "concern"), ("Don't feel like anything", "concern")]},
        "probe": r"hung thu|thich lam|enjoy|interest",
    },
    "focus": {
        "name": "Tập trung", "icon": "🎯",
        "concern": r"(?:khong|kho|mat|chang) tap trung|hay quen|dau oc (?:trong rong|lu mu|roi)|khong vao dau|"
                   r"can'?t (?:focus|concentrate)|trouble (?:focusing|concentrating)|distracted|brain fog",
        "ok": r"tap trung (?:tot|duoc|on)|van (?:hoc|lam viec) (?:duoc|binh thuong)|(?:can|able to) focus|focused",
        "ask": {"vi": ["Khi học hay làm việc, bạn có tập trung được không?",
                       "Dạo này đầu óc bạn có hay bị rối, khó tập trung không?"],
                "en": ["When you study or work, are you able to focus?",
                       "Has it been hard to concentrate lately?"]},
        "quick": {"vi": [("Tập trung được", "ok"), ("Hơi khó tập trung", "concern"), ("Không vào đầu chữ nào", "concern")],
                  "en": [("I can focus", "ok"), ("It's a bit hard", "concern"), ("Can't focus at all", "concern")]},
        "probe": r"tap trung|focus|concentrat",
    },
    "appetite": {
        "name": "Ăn uống", "icon": "🍚",
        "concern": r"(?:khong|chang) muon an|bieng an|chan an|an khong (?:ngon|vo|duoc)|bo bua|"
                   r"an (?:it|nhieu) (?:hon|qua|lam)|an uong that thuong|sut can|not eating|no appetite|"
                   r"lost (?:my )?appetite|skipping meals|eating (?:too much|a lot more)|lost weight",
        "ok": r"an (?:uong )?(?:ngon|binh thuong|deu|du|duoc)|eating (?:well|normally|fine)|appetite is fine",
        "ask": {"vi": ["Chuyện ăn uống của bạn dạo này thế nào, vẫn ăn đều không?",
                       "Mấy hôm nay bạn ăn có ngon miệng không?"],
                "en": ["How's your eating been, still having regular meals?",
                       "Have you had much of an appetite lately?"]},
        "quick": {"vi": [("Vẫn ăn bình thường", "ok"), ("Ăn ít đi", "concern"), ("Ăn nhiều hơn hẳn", "concern")],
                  "en": [("Eating normally", "ok"), ("Eating less", "concern"), ("Eating a lot more", "concern")]},
        "probe": r"an uong|\ban\b.*\?|bua|eat|appetite|meal",
    },
    "support": {
        "name": "Người để chia sẻ", "icon": "🤝",
        "concern": r"khong (?:co )?ai|co don|mot minh|khong biet (?:ke|tam su|chia se|noi) voi ai|khong ai hieu|"
                   r"bi bo roi|lac long|no one|nobody|alone|lonely|no friends|isolated",
        "ok": r"(?:co|ke voi|tam su voi|noi chuyen voi|chia se voi|goi cho) (?:ban|me|bo|ba|gia dinh|nguoi yeu|anh|"
              r"chi|em|thay|co giao|ban be|ban than)|duoc (?:dong vien|ung ho)|(?:talked|talk|spoke) to (?:my )?"
              r"(?:friend|mom|dad|parents|family|partner)|supportive|my friends? (?:help|support)",
        "ask": {"vi": ["Bên cạnh bạn có ai mà bạn có thể kể chuyện này không?",
                       "Bạn đã nói chuyện này với ai chưa, bạn bè hay người nhà chẳng hạn?"],
                "en": ["Is there someone around you that you can talk to about this?",
                       "Have you told anyone about this, like a friend or family?"]},
        "quick": {"vi": [("Có, mình có người để kể", "ok"), ("Có nhưng ngại nói", "concern"), ("Không có ai", "concern")],
                  "en": [("Yes, I have someone", "ok"), ("Yes, but I hold back", "concern"), ("No one really", "concern")]},
        "probe": r"\bai\b|ke voi|chia se|tam su|someone|anyone|talk to",
    },
    "duration": {
        "name": "Thời gian kéo dài", "icon": "⏳",
        "concern": r"(?:may|vai|hai|ba|nhieu|2|3) (?:tuan|thang|nam)|(?:ca|suot) tuan(?: nay)?|mot thoi gian (?:dai|roi)|lau (?:lam )?roi|"
                   r"tu (?:dau|hoi) (?:nam|ky|hoc ky|thang)|ca thang|suot (?:thang|nam|may tuan)|for (?:weeks|months|"
                   r"years|a long time)|(?:two|three|several|few|2|3) (?:weeks|months)|all (?:month|year)",
        "ok": r"(?:chi|moi|vua) (?:moi )?(?:hom nay|hom qua|vai hom|mot hai hom|tuan nay|day)|tu hom qua|vua moi|"
              r"(?:only|just) (?:today|now|recently|this week)|since yesterday|(?:a couple of|few) days",
        "ask": {"vi": ["Cảm giác này kéo dài bao lâu rồi bạn?", "Bạn thấy như vậy từ khi nào?"],
                "en": ["How long have you been feeling like this?", "Since when has it been like this?"]},
        "quick": {"vi": [("Mới vài hôm", "ok"), ("Khoảng 1-2 tuần", "concern"), ("Hơn một tháng", "concern")],
                  "en": [("Just a few days", "ok"), ("About 1-2 weeks", "concern"), ("Over a month", "concern")]},
        "probe": r"bao lau|lau chua|tu khi nao|how long|since when",
    },
}
DOMAIN_ORDER = ["mood", "sleep", "energy", "support", "duration", "interest", "focus", "appetite"]
_COMPILED = {d: {k: re.compile(rf"\b(?:{spec[k]})", re.I) for k in ("concern", "ok", "probe")}
             for d, spec in DOMAINS.items()}

TOPIC_VI = {"study": "📚 Học tập", "sleep": "🌙 Giấc ngủ", "work": "💼 Công việc", "relationship": "💬 Các mối quan hệ",
            "lonely": "🫂 Cô đơn", "anxiety": "😟 Lo âu", "sad": "🌧️ Nỗi buồn", "pressure": "🏋️ Áp lực"}
MOOD_SCALE = [(1, "😞", "Rất tệ"), (2, "🙁", "Không ổn"), (3, "😐", "Bình thường"), (4, "🙂", "Khá ổn"), (5, "😄", "Rất tốt")]
LEVELS = [
    ("Ổn định", "Chưa thấy dấu hiệu căng thẳng đáng kể trong những gì bạn chia sẻ.", "#4E9C82"),
    ("Có chút căng thẳng", "Có một vài dấu hiệu căng thẳng, thường gặp khi đang có nhiều việc.", "#C9A23A"),
    ("Căng thẳng đáng chú ý", "Nhiều dấu hiệu căng thẳng lặp lại qua cuộc trò chuyện, nên được quan tâm.", "#DE8A3C"),
    ("Nặng nề, nên tìm hỗ trợ", "Nhiều tín hiệu nặng nề và kéo dài, nên nói chuyện với người tin cậy hoặc chuyên gia.", "#D9694A"),
]


def detect_domains(text: str) -> dict:
    """{domain: 'concern' | 'ok'} cho các khía cạnh được nhắc tới trong câu. Khớp dài hơn thắng khi chồng lấn
    (vd. 'không mất ngủ' -> ok, dù chứa 'mất ngủ'); nếu cùng lúc có cả hai thì ưu tiên 'concern'."""
    norm = strip_accents(text or "")
    found = {}
    for d, rx in _COMPILED.items():
        matches = [(m.start(), m.end(), pol) for pol in ("concern", "ok") for m in rx[pol].finditer(norm)]
        matches.sort(key=lambda m: m[1] - m[0], reverse=True)
        taken, pols = [], set()
        for s, e, pol in matches:
            if all(e <= ts or s >= te for ts, te in taken):
                taken.append((s, e))
                pols.add(pol)
        if pols:
            found[d] = "concern" if "concern" in pols else "ok"
    return found


def asked_domain(bot_text: str):
    """Khía cạnh mà câu HỎI cuối trong phản hồi của bot đang hỏi tới (để hiện nút trả lời nhanh), hoặc None."""
    questions = [q for q in re.split(r"(?<=[.!?])\s+", bot_text or "") if q.strip().endswith("?")]
    if not questions:
        return None
    q = strip_accents(questions[-1])
    for d in DOMAIN_ORDER:
        if _COMPILED[d]["probe"].search(q):
            return d
    return None


# --------------------------------------------------------------------------- tổng hợp
def _weights_by_length(words):
    w = np.sqrt(np.maximum(np.asarray(words, dtype=float), 1.0))   # câu dài có nhiều thông tin hơn, nhưng không áp đảo
    return w / w.sum()


def _recent(values, decay=None):
    decay = config.RECENCY_DECAY if decay is None else decay
    n = len(values)
    w = np.array([decay ** (n - 1 - i) for i in range(n)])
    return float(np.dot(w / w.sum(), values))


def _dep_score(entry):
    if entry.get("depression_score") is not None:
        return float(entry["depression_score"])
    probs = entry.get("depression_probs")
    if probs:
        return float(calibration.expected_level([probs[l] for l in config.DEPRESSION_LABELS]))
    return None


def _dep_label(score):
    cuts = calibration.load().get("depression_cuts") or [0.5, 1.5, 2.5]
    return config.DEPRESSION_LABELS[int(calibration.decode_ordinal(np.array([score]), cuts)[0])]


def build_profile(entries, self_moods=()):
    """Gộp bằng chứng của cả cuộc trò chuyện.

    entries: các lượt người dùng (dict) với khóa tùy chọn: words, model_ok, stress_probability, depression_score
             hoặc depression_probs, emotions [(nhãn, điểm)], topics [..], domains {d: 'concern'|'ok'}.
    self_moods: điểm tâm trạng tự chấm 1..5.
    """
    entries = list(entries)
    scored = [e for e in entries if e.get("model_ok") and e.get("stress_probability") is not None]
    p = {"n_turns": len(entries), "n_scored": len(scored),
         "n_words": int(sum(e.get("words", 0) for e in entries)), "self_moods": list(self_moods)}

    if scored:
        w = _weights_by_length([e.get("words", 1) for e in scored])
        stress = np.array([e["stress_probability"] for e in scored])
        p["stress"] = {"overall": float(np.dot(w, stress)), "recent": _recent(stress),
                       "share": float((stress >= 0.5).mean()),
                       "trend": float(stress[-3:].mean() - stress[:-3].mean()) if len(stress) > 3 else 0.0,
                       "series": stress.round(4).tolist()}
        deps = [(wi, s) for wi, s in zip(w, map(_dep_score, scored)) if s is not None]
        if deps:
            dw = np.array([d[0] for d in deps]); dw = dw / dw.sum()
            score = float(np.dot(dw, [d[1] for d in deps]))
            p["depression"] = {"score": score, "label": _dep_label(score),
                               "series": [round(d[1], 3) for d in deps]}
        mass = Counter()
        for wi, e in zip(w, scored):
            for emo, sc in e.get("emotions") or []:
                mass[emo] += wi * float(sc)
        neg = sum(v for k, v in mass.items() if k in config.NEGATIVE_EMOTIONS)
        pos = sum(v for k, v in mass.items() if k in config.POSITIVE_EMOTIONS)
        p["emotions"] = [(k, round(v, 4)) for k, v in mass.most_common(6) if k != "neutral"]
        p["negative_share"] = float(neg / (neg + pos)) if neg + pos > 0 else None

    topics = Counter(t for e in entries for t in e.get("topics") or [])
    p["topics"] = [t for t, _ in topics.most_common()]
    status = {}
    for e in entries:                                     # câu sau cập nhật câu trước (người dùng có thể đính chính)
        status.update(e.get("domains") or {})
    p["domains"] = status
    p["coverage"] = len(status) / len(DOMAINS)
    p["evidence"] = _evidence(p)
    p["score"], p["components"] = _concern_score(p)
    p["level"] = None if p["score"] is None else int(np.digitize(p["score"], config.CONCERN_THRESHOLDS))
    return p


def _evidence(p):
    """'low' | 'medium' | 'high': đã đủ dữ liệu để nhận xét chưa (số lượt, số từ, số khía cạnh đã biết)."""
    turns, words, cov = p["n_turns"], p["n_words"], p["coverage"]
    if turns >= 6 and words >= 80 and cov >= 0.6:
        return "high"
    if turns >= 3 and words >= 25 and (cov >= 0.25 or p["n_scored"] >= 3):
        return "medium"
    return "low"


def _concern_score(p):
    """Điểm 0..1 = trung bình có trọng số của các thành phần CÓ DỮ LIỆU (thiếu thành phần nào thì bỏ, chia lại)."""
    comp = {}
    if "stress" in p:
        comp["stress"] = p["stress"]["overall"]
    if "depression" in p:
        comp["depression"] = p["depression"]["score"] / 3
    if p.get("negative_share") is not None:
        comp["emotion"] = p["negative_share"]
    known = [v for d, v in p["domains"].items() if d != "duration"]
    if known:
        comp["domains"] = sum(v == "concern" for v in known) / len(known)
    if p["self_moods"]:
        comp["self_mood"] = (5 - float(np.mean(p["self_moods"]))) / 4
    if not comp:
        return None, {}
    wts = {k: config.CONCERN_WEIGHTS[k] for k in comp}
    score = sum(comp[k] * wts[k] for k in comp) / sum(wts.values())
    if p["domains"].get("duration") == "concern" and score >= 0.35:
        score = min(1.0, score + config.DURATION_BONUS)       # kéo dài nhiều tuần làm tín hiệu đáng tin hơn
    return float(score), {k: round(v, 4) for k, v in comp.items()}


def next_domain(profile, asked=()):
    """Khía cạnh nên hỏi tiếp: chưa biết, ưu tiên chưa hỏi. Ưu tiên 'duration' ngay khi đã thấy có điều đáng lo."""
    asked = list(asked)
    unknown = [d for d in DOMAIN_ORDER if d not in profile["domains"] and d not in asked]
    if not unknown:                                   # người dùng lờ câu hỏi đi: hỏi lại tối đa một lần nữa
        unknown = [d for d in DOMAIN_ORDER if d not in profile["domains"] and asked.count(d) < 2]
    if not unknown:
        return None
    if "duration" in unknown and "concern" in profile["domains"].values():
        return "duration"
    return unknown[0]


def safety_view(profile):
    """Dạng kết quả mà safety.assess_safety hiểu, lấy từ mức gộp cả cuộc trò chuyện."""
    view = {}
    if "stress" in profile:
        view["stress"] = config.STRESS_YES if profile["stress"]["overall"] >= 0.5 else config.STRESS_NO
    if "depression" in profile:
        view["depression"] = profile["depression"]["label"]
    return view


# --------------------------------------------------------------------------- hướng giải quyết
_TIPS = {
    "sleep": ("🌙", "Chăm lại giấc ngủ",
              "Thử giữ giờ ngủ và giờ dậy cố định kể cả cuối tuần, cất điện thoại 30 phút trước khi ngủ. "
              "Nếu nằm 20 phút không ngủ được, dậy làm việc gì nhẹ nhàng dưới ánh sáng dịu rồi quay lại giường."),
    "energy": ("🔋", "Nạp lại năng lượng",
               "Chia ngày thành những khoảng làm việc ngắn xen nghỉ. Đi bộ 10-15 phút ngoài trời, uống đủ nước, "
               "và đừng tự trách khi hôm nay làm được ít hơn bình thường."),
    "interest": ("🎨", "Một việc nhỏ dễ chịu mỗi ngày",
                 "Khi chẳng muốn làm gì, đừng đợi có hứng rồi mới làm: chọn một việc nhỏ từng làm bạn vui "
                 "(nghe một bài nhạc, gọi một người bạn, ra ngoài 10 phút) và làm thử, cảm hứng thường đến sau."),
    "focus": ("🎯", "Học/làm theo từng khối ngắn",
              "Thử 25 phút tập trung một việc duy nhất rồi nghỉ 5 phút. Viết ra 1-3 việc quan trọng nhất trong ngày "
              "và bắt đầu từ việc nhỏ nhất để lấy đà."),
    "appetite": ("🍚", "Giữ bữa ăn đều đặn",
                 "Kể cả khi không thấy đói, cố ăn những bữa nhỏ đúng giờ. Ăn uống thất thường kéo dài là điều "
                 "nên kể với người thân hoặc nhân viên y tế."),
    "support": ("🤝", "Đừng giữ một mình",
                "Chọn một người bạn tin (bạn bè, người nhà, thầy cô, cố vấn học tập) và kể một phần nhỏ thôi cũng được. "
                "Nhiều trường có phòng tư vấn tâm lý miễn phí cho sinh viên."),
    "mood": ("📝", "Viết ra điều đang nghĩ",
             "Dành 10 phút viết tự do những gì trong đầu, không cần hay. Cuối cùng ghi thêm một điều nhỏ hôm nay "
             "đã ổn, dù rất nhỏ."),
    "study": ("📚", "Gỡ áp lực học tập",
              "Liệt kê các hạn nộp, xếp theo thứ tự gấp, chia mỗi việc thành bước nhỏ làm được trong một buổi. "
              "Nếu quá tải, trao đổi sớm với giảng viên hoặc cố vấn học tập thường hiệu quả hơn bạn nghĩ."),
    "work": ("💼", "Đặt ranh giới với công việc",
             "Chọn ra việc thật sự phải xong hôm nay, phần còn lại ghi xuống để não không phải giữ. "
             "Thử nói chuyện với người quản lý về khối lượng việc nếu nó kéo dài."),
    "relationship": ("💬", "Khi chuyện các mối quan hệ làm bạn nặng lòng",
                     "Cho bản thân thời gian bình tĩnh trước khi nói chuyện lại; khi nói, kể cảm xúc của mình "
                     "(\"mình thấy buồn khi...\") thay vì trách người kia."),
    "anxiety": ("🌬️", "Khi lo âu dâng lên",
                "Thở hộp 4-4-4-4 (hít 4 giây, giữ 4, thở ra 4, giữ 4) vài vòng, hoặc gọi tên 5 thứ bạn đang nhìn thấy "
                "để kéo mình về hiện tại. Có nút bài thở ở thanh bên."),
    "pressure": ("🧩", "Chia nhỏ áp lực",
                 "Viết hết mọi việc đang đè lên ra giấy, gạch những việc không thuộc quyền kiểm soát của bạn, "
                 "và chỉ chọn một việc nhỏ để làm tiếp theo."),
}
_PRO = ("🩺", "Nói chuyện với chuyên gia",
        "Những gì bạn kể đã kéo dài và ảnh hưởng tới sinh hoạt, nên gặp chuyên gia tâm lý hoặc bác sĩ là bước hợp lý. "
        "Đó không phải dấu hiệu yếu đuối mà là cách chăm sóc bản thân.")


def suggestions(profile, limit=4):
    """Danh sách (biểu tượng, tiêu đề, nội dung), ưu tiên khía cạnh đáng chú ý rồi tới chủ đề người dùng nhắc."""
    keys = [d for d in DOMAIN_ORDER if profile["domains"].get(d) == "concern" and d in _TIPS]
    keys += [t for t in profile["topics"] if t in _TIPS]
    if "lonely" in profile["topics"]:
        keys.append("support")
    if "sad" in profile["topics"]:
        keys.append("mood")
    out, seen = [], set()
    if profile.get("level") is not None and profile["level"] >= 2:
        out.append(_PRO)
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(_TIPS[k])
    if not out:
        out.append(_TIPS["mood"])
    return out[:limit]


def summary_markdown(profile, narrative=""):
    """Bản tổng kết dạng Markdown (để tải về hoặc hiển thị khi không có LLM)."""
    lines = ["# Tổng kết cuộc trò chuyện", ""]
    if narrative:
        lines += [narrative.strip(), ""]
    if profile["evidence"] == "low" or profile["level"] is None:
        lines += ["Chưa đủ thông tin để nhận xét. Hãy trò chuyện thêm vài lượt nhé.", ""]
    else:
        name, desc, _ = LEVELS[profile["level"]]
        lines += [f"**Nhận xét chung:** {name}. {desc}", ""]
    if profile["topics"]:
        lines.append("**Điều bạn nhắc tới:** " + ", ".join(TOPIC_VI.get(t, t) for t in profile["topics"]))
    if profile["domains"]:
        lines += ["", "**Các khía cạnh đã trò chuyện:**"]
        for d in DOMAIN_ORDER:
            if d in profile["domains"]:
                mark = "đáng chú ý" if profile["domains"][d] == "concern" else "ổn"
                if d == "duration":
                    mark = "đã kéo dài" if profile["domains"][d] == "concern" else "mới gần đây"
                lines.append(f"- {DOMAINS[d]['icon']} {DOMAINS[d]['name']}: {mark}")
    lines += ["", "## Gợi ý cho bạn"]
    for icon, title, text in suggestions(profile):
        lines += [f"**{icon} {title}.** {text}", ""]
    lines += ["---", "Đây là kết quả của một công cụ nghiên cứu, KHÔNG phải chẩn đoán y khoa. "
              "Nếu bạn có ý nghĩ làm hại bản thân, hãy gọi 115 hoặc nói ngay với người thân."]
    return "\n".join(lines)
