import os
import json
import argparse
from pathlib import Path

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline


def srt_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def save_json_srt(base: Path, segments):
    out_json = base.with_suffix(".tagged.json")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    out_srt = base.with_suffix(".tagged.srt")
    with out_srt.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            speaker = seg.get("speaker") or "Speaker?"
            f.write(
                f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{speaker}: {seg['text'].strip()}\n\n"
            )
    print(f"Saved:\n  {out_json}\n  {out_srt}")


def main():
    ap = argparse.ArgumentParser(
        description="Диааризация whisper-сегментов через WhisperX"
    )
    ap.add_argument("audio", help="путь к исходному аудио/видео (моно ок)")
    ap.add_argument(
        "whisper_json", help="путь к JSON от whisper: [{'start','end','text'},...]"
    )
    ap.add_argument("--lang", default="ru", help="код языка для align (ru/en/...)")
    ap.add_argument(
        "--device", default=None, help="cuda|cpu (M1: оставь пустым -> cpu)"
    )
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # исходные сегменты whisper
    with open(args.whisper_json, encoding="utf-8") as f:
        segs = json.load(f)
    result = {"segments": segs, "language": args.lang}

    diar = DiarizationPipeline(
        use_auth_token=os.getenv("HF_TOKEN"), device=device,
        min_speakers=2,
        max_speakers=3
    )
    diar_segments = diar(args.audio)

    # 4) назначаем спикеров сегментам/словам и упрощаем имена
    res_spk = whisperx.assign_word_speakers(diar_segments, result)
    segments = res_spk["segments"]
    for seg in segments:
        spk = seg.get("speaker")
        if spk and spk.startswith("SPEAKER_"):
            try:
                n = int(spk.split("_")[-1])
                seg["speaker"] = f"Speaker{n+1}"
            except Exception:
                pass

    base = Path(args.whisper_json).with_suffix("")
    save_json_srt(base, segments)


if __name__ == "__main__":
    main()
