import json
from pathlib import Path


def save_whisperx_result(base: Path, segments) -> Path:
    out_json = base.with_suffix(".tagged.json")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    return out_json
