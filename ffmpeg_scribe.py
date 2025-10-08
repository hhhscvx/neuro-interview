import os
import ffmpeg

from utils import settings, logger


def process_audio(input_path, output_path):
    """Обработка аудио с удалением пауз"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stream = ffmpeg.input(input_path)
    stream = ffmpeg.output(
        stream,
        output_path,
        acodec="pcm_s16le",
        ac=1,
        ar=16000,
        af=(
            "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-35dB:detection=peak"
        ),
        vn=None,
        threads=os.cpu_count() or 4,
    )

    ffmpeg.run(stream, overwrite_output=True)


def ffmpeg_scribe(input_file: str) -> bool:
    output_file = input_file.split(".")[0]

    input_path = f"{settings.INTERVIEWS_PATH}/{input_file}"
    output_path = f"{settings.SCRAPED_FFMPEG_PATH}/{output_file}.wav"

    try:
        process_audio(input_path, output_path)
        logger.success(f"[ffmpeg] Обработка завершена: {output_path}")
        return True
    except Exception as e:
        logger.error(f"[ffmpeg] Ошибка при преобразовании в аудио: {e}")
        return False
