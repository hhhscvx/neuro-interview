import os
import json
import argparse
from pathlib import Path
from time import perf_counter

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from utils import (settings, logger, split_audio_to_parts,
                  normalize_diar_segments)


def srt_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def save_result(base: Path, segments):
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
    logger.success(f"Диаризация завершена. Сохранено: {out_json} | {out_srt}")


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    ap = argparse.ArgumentParser()
    ap.add_argument("filename", help="название файла без расширения (например: Собес)")
    ap.add_argument("--lang", default="ru")
    args = ap.parse_args()

    filename = Path(args.filename)
    base_name = filename.stem
    
    audio_path = None
    for ext in ['.wav', '.mp3', '.m4a', '.flac']:
        potential_path = Path(settings.SCRAPED_FFMPEG_PATH) / f"{base_name}{ext}"
        if potential_path.exists():
            audio_path = potential_path
            logger.success(f"Путь к аудио найден: {audio_path}")
            break
    
    if not audio_path:
        logger.error(f"Аудио файл не найден для {base_name} в {settings.SCRAPED_FFMPEG_PATH}")
        return

    whisper_json_path = Path(settings.SCRAPED_WHISPER_PATH) / f"{base_name}.json"
    if not whisper_json_path.exists():
        logger.error(f"JSON файл не найден: {whisper_json_path}")
        return

    with open(whisper_json_path, encoding="utf-8") as f:
        segs = json.load(f)
    result = {"segments": segs, "language": args.lang}

    logger.info("Подготовка аудио (возможна нарезка на части)…")
    t_split0 = perf_counter()
    parts, total_sec = split_audio_to_parts(audio_path)
    logger.info(f"Готово: {len(parts)} ч.(частей), длительность: {round(total_sec/60,1)} мин | "
                f"заняло {round(perf_counter()-t_split0,2)} с")

    diar = DiarizationPipeline(use_auth_token=settings.HF_TOKEN, device="cpu")
    
    all_diar_segments: list[dict] = []
    for idx, (part_path, t0, t1) in enumerate(parts, start=1):
        logger.info(f"[{idx}/{len(parts)}] Диаризация куска {part_path.name} ({round(t0,1)}–{round(t1,1)} с)…")
        t0_run = perf_counter()
        diar_raw = diar(str(part_path))
        diar_segments = normalize_diar_segments(diar_raw)

        for seg in diar_segments:
            logger.info(f"Type seg: {type(seg)} | seg: {seg}")
            start_g = seg["start"] + t0
            end_g = seg["end"] + t0
            if idx > 1 and end_g <= (t0 + settings.AUDIO_OVERLAP_SEC * 0.6):
                continue
            all_diar_segments.append({"start": start_g, "end": end_g, "speaker": seg["speaker"]})

        logger.success(f"[{idx}/{len(parts)}] Готово: сегментов={len(diar_segments)}, накоплено={len(all_diar_segments)}, {round(perf_counter()-t0_run,2)} с")

    logger.info("Назначаем спикеров сегментам Whisper…")
    res_spk = whisperx.assign_word_speakers(all_diar_segments, result)
    segments = res_spk["segments"]

    # for seg in segments:
    #     spk = seg.get("speaker")
    #     if spk and spk.startswith("SPEAKER_"):
    #         try:
    #             n = int(spk.split("_")[-1])
    #             seg["speaker"] = f"Speaker{n+1}"
    #         except Exception:
    #             pass

    base = Path(settings.SCRAPED_RESULT_PATH) / base_name
    save_result(base, segments)


if __name__ == "__main__":
    if not os.path.exists(settings.SCRAPED_RESULT_PATH):
        os.mkdir(settings.SCRAPED_RESULT_PATH)
    main()
