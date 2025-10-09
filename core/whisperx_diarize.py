import os
import json
from pathlib import Path
from time import perf_counter

import torch
import pandas as pd
import whisperx
from whisperx.diarize import DiarizationPipeline

from core.utils import (
    settings,
    logger,
    split_audio_to_parts,
    normalize_diar_segments,
    save_whisperx_result,
)


def whisperx_diarize(filename: str, lang: str = "ru") -> bool:
    torch.set_num_threads(os.cpu_count() or 4)

    filename = Path(filename)
    base_name = filename.stem

    audio_path = None
    for ext in [".wav", ".mp3", ".m4a", ".flac"]:
        potential_path = Path(settings.SCRAPED_FFMPEG_PATH) / f"{base_name}{ext}"
        if potential_path.exists():
            audio_path = potential_path
            logger.success(f"[whisperx] Путь к аудио найден: {audio_path}")
            break

    if not audio_path:
        logger.error(
            f"[whisperx] Аудио файл не найден для {base_name} в {settings.SCRAPED_FFMPEG_PATH}"
        )
        return

    whisper_json_path = Path(settings.SCRAPED_WHISPER_PATH) / f"{base_name}.json"
    if not whisper_json_path.exists():
        logger.error(f"[whisperx] JSON файл не найден: {whisper_json_path}")
        return

    with open(whisper_json_path, encoding="utf-8") as f:
        segs = json.load(f)
    result = {"segments": segs, "language": lang}

    logger.info("[whisperx] Подготовка аудио (возможна нарезка на части)…")
    parts, total_sec = split_audio_to_parts(audio_path)
    logger.info(
        f"[whisperx] Готово: частей={len(parts)}шт. длительность: {round(total_sec/60,1)} мин | "
    )

    try:
        diar = DiarizationPipeline(use_auth_token=settings.HF_TOKEN, device="cpu")

        all_diar_segments: list[dict] = []
        for idx, (part_path, t0, t1) in enumerate(parts, start=1):
            logger.info(
                f"[{idx}/{len(parts)}] Диаризация куска {part_path.name} ({round(t0,1)}–{round(t1,1)} с)…"
            )
            t0_run = perf_counter()
            diar_raw = diar(str(part_path))
            diar_segments = normalize_diar_segments(diar_raw)

            for seg in diar_segments:
                start_g = seg["start"] + t0
                end_g = seg["end"] + t0
                if idx > 1 and end_g <= (t0 + settings.AUDIO_OVERLAP_SEC * 0.6):
                    continue
                all_diar_segments.append(
                    {"start": start_g, "end": end_g, "speaker": seg["speaker"]}
                )

            logger.success(
                f"[whisperx] [{idx}/{len(parts)}] Готово: сегментов={len(diar_segments)},"
                f" накоплено={len(all_diar_segments)}, {round(perf_counter()-t0_run,2)} с"
            )

        logger.info("Назначаем спикеров сегментам Whisper…")
        diar_df = pd.DataFrame(all_diar_segments, columns=["start", "end", "speaker"])
        diar_df = diar_df.sort_values("start").reset_index(drop=True)
        diar_df["start"] = diar_df["start"].astype(float)
        diar_df["end"] = diar_df["end"].astype(float)
        diar_df["speaker"] = diar_df["speaker"].astype(str)

        res_spk = whisperx.assign_word_speakers(diar_df, result)
        segments = res_spk["segments"]

        base = Path(settings.SCRAPED_RESULT_PATH) / base_name
        out_json = save_whisperx_result(base, segments)
        logger.success(f"Диаризация завершена. Сохранено: {out_json}")
        return True

    except Exception as error:
        logger.error(f"[whisperx] Ошибка при диаризации: {error}")
        return False
