from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    HF_TOKEN: str

    # директория, откуда ffmpeg будет брать видосы. По умолчанию из корня
    INTERVIEWS_PATH: str = ""
    # директория, в которой будут находиться wav файлы
    SCRAPED_FFMPEG_PATH: str = ""
    # директория, в которой будут находиться json-ы транскрибации whisper
    SCRAPED_WHISPER_PATH: str = ""
    # директория после окончательного этапа транскрибации whisperx
    SCRAPED_RESULT_PATH: str = ""
    RESULT_CHUNKS_PATH: str = "result_chunks"

    AUDIO_PARTS_PATH: str = "audio_parts"
    AUDIO_PARTS_MINUTES: int = 15
    AUDIO_OVERLAP_SEC: float = 2.0


settings = Settings()
