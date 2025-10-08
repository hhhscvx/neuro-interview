import argparse
from pathlib import Path

from utils import settings, logger
from utils.ensure_dirs import ensure_dirs
from ffmpeg_scribe import ffmpeg_scribe
from whisper_scribe import whisper_scribe
from whisperx_diarize import whisperx_diarize


VIDEO_EXTS = {".mp4", ".mkv", ".mov"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac"}
ALL_EXTS = VIDEO_EXTS | AUDIO_EXTS


def already_done(base: str) -> bool:
    return (Path(settings.SCRAPED_RESULT_PATH) / f"{base}.tagged.json").exists()


def main():
    parser = argparse.ArgumentParser(
        description="Запуск полного пайплайна: ffmpeg → whisper → whisperx"
    )
    parser.add_argument(
        "--skip-ffmpeg", action="store_true", help="Пропустить шаг ffmpeg"
    )
    parser.add_argument(
        "--skip-whisper", action="store_true", help="Пропустить шаг whisper"
    )
    parser.add_argument(
        "--skip-whisperx", action="store_true", help="Пропустить шаг whisperx"
    )
    args = parser.parse_args()
    ensure_dirs()
    interviews = Path(settings.INTERVIEWS_PATH)

    items = sorted(
        [
            p
            for p in interviews.iterdir()
            if p.is_file() and p.suffix.lower() in ALL_EXTS
        ]
    )
    if not items:
        logger.info(f"Пусто в {settings.INTERVIEWS_PATH}")
        return

    for src in items:
        base = src.stem
        if already_done(base):
            logger.info(f"[skip] Уже готово: {base}")
            continue

        logger.info(f"=== ▶ {base} ===")

        # ЗАПУСК FFMPEG
        if not args.skip_ffmpeg:
            if ffmpeg_scribe(src.name) is not True:
                return
        else:
            logger.info("[skip] Пропуск ffmpeg")

        # ЗАПУСК WHISPER
        if not args.skip_whisper:
            if not whisper_scribe(f"{base}.wav", lang="ru"):
                return
        else:
            logger.info("[skip] Пропуск whisper")

        # ЗАПУСК WHISPERX
        if not args.skip_whisperx:
            if not whisperx_diarize(base, lang="ru"):
                return
        else:
            logger.info("[skip] Пропуск whisperx")

        logger.success(f"Пайплайн завершен: {base}")


if __name__ == "__main__":
    main()
