import argparse
import json
import os
from time import perf_counter

from faster_whisper import WhisperModel

from utils import settings, logger


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--lang", type=str, default="ru")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--model", default="medium", help="small|medium|large-v3")


def main():
    args = parser.parse_args()

    model = WhisperModel(
        "medium",
        compute_type="int8",
        device="cpu",
        cpu_threads=os.cpu_count(),
        num_workers=1,
    )

    logger.info("Транскрибирование...")
    t0 = perf_counter()
    segments, info = model.transcribe(
        audio=f"{settings.SCRAPED_FFMPEG_PATH}/{args.path}",
        language=args.lang,
        vad_filter=True,
        temperature=0,
        initial_prompt=args.prompt,
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

    scraped_whisper_path = f"{settings.SCRAPED_WHISPER_PATH}/{args.path.split('.')[0]}"

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

    logger.success(f"Транскрибация заняла {round(perf_counter()-t0,2)} с")

    with open(scraped_whisper_path + ".json", "w", encoding="utf-8") as f:
        json.dump(segments_list, f, ensure_ascii=False, indent=2)
    logger.success("Сохранено:", scraped_whisper_path + ".json / .txt")


if __name__ == "__main__":
    main()
