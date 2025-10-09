import json
import os
from time import perf_counter

from faster_whisper import WhisperModel

from core.utils import settings, logger


def whisper_scribe(
    path: str, lang: str = "ru", prompt: str | None = None, model: str = "medium"
) -> bool:
    try:
        model = WhisperModel(
            "medium",
            compute_type="int8",
            device="cpu",
            cpu_threads=os.cpu_count(),
            num_workers=1,
        )

        logger.info("[whisper] Транскрибирование...")
        t0 = perf_counter()
        segments, info = model.transcribe(
            audio=f"{settings.SCRAPED_FFMPEG_PATH}/{path}",
            language=lang,
            vad_filter=True,
            temperature=0,
            initial_prompt=prompt,
            condition_on_previous_text=False,
            vad_parameters={
                "threshold": 0.6,
                "min_speech_duration_ms": 500,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        )

        if not os.path.exists(settings.SCRAPED_WHISPER_PATH):
            os.mkdir(settings.SCRAPED_WHISPER_PATH)

        scraped_whisper_path = f"{settings.SCRAPED_WHISPER_PATH}/{path.split('.')[0]}"

        segments_list = []
        full_text = ""

        for segment in segments:
            segment_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            }
            logger.info(segment_dict)
            segments_list.append(segment_dict)
            full_text += segment.text.strip() + " "

        logger.success(f"[whisper] Транскрибация заняла {round(perf_counter()-t0,2)} с")

        with open(scraped_whisper_path + ".json", "w", encoding="utf-8") as f:
            json.dump(segments_list, f, ensure_ascii=False, indent=2)
        logger.success("[whisper] Успешно сохранено")
        return True

    except Exception as error:
        logger.error(f"Ошибка при транскрибации whisper: {error}")
        return False
