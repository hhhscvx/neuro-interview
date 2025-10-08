from pathlib import Path

from utils.config import settings


def ensure_dirs():
    Path(settings.INTERVIEWS_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.SCRAPED_FFMPEG_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.SCRAPED_WHISPER_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.SCRAPED_RESULT_PATH).mkdir(parents=True, exist_ok=True)