import argparse
from pathlib import Path

from core.utils import settings, logger
from core.chunk_transcribed import chunk_transcribed
from core.utils.ensure_dirs import ensure_dirs

from core.ffmpeg_scribe import ffmpeg_scribe
from core.whisper_scribe import whisper_scribe
from core.whisperx_diarize import whisperx_diarize


VIDEO_EXTS = {".mp4", ".mkv", ".mov"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac"}
ALL_EXTS = VIDEO_EXTS | AUDIO_EXTS


def already_done(base: str) -> bool:
    return (Path(settings.SCRAPED_RESULT_PATH) / f"{base}.tagged.json").exists()


def ffmpeg_done(base: str) -> bool:
    """Проверяет, есть ли аудио файл после ffmpeg"""
    return (
        (Path(settings.SCRAPED_FFMPEG_PATH) / f"{base}.wav").exists()
        or (Path(settings.SCRAPED_FFMPEG_PATH) / f"{base}.mp3").exists()
        or (Path(settings.SCRAPED_FFMPEG_PATH) / f"{base}.m4a").exists()
        or (Path(settings.SCRAPED_FFMPEG_PATH) / f"{base}.flac").exists()
    )


def whisper_done(base: str) -> bool:
    """Проверяет, есть ли JSON транскрибация от whisper"""
    return (Path(settings.SCRAPED_WHISPER_PATH) / f"{base}.json").exists()


def whisperx_done(base: str) -> bool:
    """Проверяет, есть ли финальный результат от whisperx"""
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
        try:
            if not args.skip_ffmpeg:
                if ffmpeg_done(base):
                    logger.info(f"[skip] FFmpeg уже готов: {base}.wav")
                else:
                    if ffmpeg_scribe(src.name) is not True:
                        raise Exception("FFmpeg не смог обработать файл")
            else:
                logger.info("[skip] Пропуск ffmpeg")

            if not args.skip_whisper:
                if whisper_done(base):
                    logger.info(f"[skip] Whisper уже готов: {base}.json")
                else:
                    if not whisper_scribe(f"{base}.wav", lang="ru"):
                        raise Exception("Ошибка при Whisper")
            else:
                logger.info("[skip] Пропуск whisper")

            if not args.skip_whisperx:
                if not whisperx_diarize(base, lang="ru"):
                    raise Exception("Ошибка при WhisperX")
            else:
                logger.info("[skip] Пропуск whisperx")

            try:
                if whisperx_done(base):
                    chunk_transcribed(base)
                else:
                    logger.warning("Пропуск нарезки: нет финального результата whisperx")
            except Exception as e:
                logger.error(f"Ошибка при нарезке чанков для {base}: {e}")

            logger.success(f"Пайплайн завершен: {base}")
        except Exception as e:
            logger.error(f"Ошибка при выполнении пайплайна для {base}: {e}")
            logger.info(f"Пропуск {base}")


if __name__ == "__main__":
    main()
