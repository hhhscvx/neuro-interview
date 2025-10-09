from pathlib import Path
import json

def load_segments(tagged_json_path: str):
    segs = json.loads(Path(tagged_json_path).read_text(encoding="utf-8"))
    segs.sort(key=lambda x: x["start"])
    return segs

def merge_consecutive(segs, gap=0.6):
    out = []
    cur = None
    for s in segs:
        if cur and s["speaker"] == cur["speaker"] and s["start"] <= cur["end"] + gap:
            cur["end"] = s["end"]
            cur["text"] += " " + s["text"]
        else:
            if cur: out.append(cur)
            cur = dict(s)
    if cur: out.append(cur)
    return out

def as_plain_text(segs):
    # Формат удобный для LLM: [00:01:08–00:01:14] SPEAKER_01: текст
    def f(sec):
        h = int(sec//3600); m = int((sec%3600)//60); s = int(sec%60)
        return f"{h:02}:{m:02}:{s:02}"
    lines = []
    for s in segs:
        lines.append(f"[{f(s['start'])}–{f(s['end'])}] {s['speaker']}: {s['text'].strip()}")
    return "\n".join(lines)

def chunk_text(text, max_chars=15000, overlap=800):
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        # стараться резать по переводу строки/точке
        cut = text.rfind("\n", i+int(0.7*max_chars), j)
        if cut == -1: cut = text.rfind(". ", i+int(0.7*max_chars), j) + 1
        if cut <= i: cut = j
        chunks.append(text[i:cut].strip())
        i = max(cut - overlap, cut)  # overlap только вперёд
    return [c for c in chunks if c]
