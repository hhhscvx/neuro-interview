from pathlib import Path

import numpy as np
import soundfile as sf

from core.utils import settings


def split_audio_to_parts(
    audio_path: Path,
    out_dir: Path = Path(settings.AUDIO_PARTS_PATH),
    part_minutes: int = settings.AUDIO_PARTS_MINUTES,
    overlap_sec: float = settings.AUDIO_OVERLAP_SEC,
) -> list[tuple[Path, float, float]]:
    """
    Режем на куски длиной part_minutes c перекрытием overlap_sec.
    Возвращаем список (path, t_start, t_end) в секундах глобального времени.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    total_sec = len(audio) / sr

    chunk_len = part_minutes * 60.0
    hop = max(chunk_len - overlap_sec, 1.0)

    parts = []
    i = 0
    t0 = 0.0
    while t0 < total_sec:
        t1 = min(t0 + chunk_len, total_sec)
        i += 1
        beg = int(t0 * sr)
        end = int(t1 * sr)
        piece = audio[beg:end]
        part_path = out_dir / f"{audio_path.stem}_part{i}.wav"
        sf.write(str(part_path), piece, sr, subtype="PCM_16")
        parts.append((part_path, t0, t1))
        if t1 >= total_sec:
            break
        t0 += hop

    return parts, total_sec
