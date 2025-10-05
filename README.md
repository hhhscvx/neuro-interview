### Видео —> Выжимка
За основу брал [эту статью](https://habr.com/ru/companies/alfa/articles/909498/), но вместо NeMo использую whisperx (проще ставится) и в конце скармливаю в GPT для выжимки полученного

- `poetry install`

---

1. ffmpeg:
   - Сначала ставим ffmpeg системно
   - Затем `python ffmpeg_scribe.py input.mp4` (берется из директории interviews)

2. whisper:
   - `python whisper_scribe.py input.wav` (берется из директории scraped_ffmpeg)

3. whisperx (диаризация):
   - Зарегаться и получить **Read токен** на [huggingface](https://huggingface.co/settings/tokens) и указать в .env. Обязательно получить доступ на репо [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) и [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - `python whisperx_diarize.py input.wav input.json --lang ru`
