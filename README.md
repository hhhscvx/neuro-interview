### Видео —> Выжимка
За основу брал [эту статью](https://habr.com/ru/companies/alfa/articles/909498/)

- `poetry install`
- Для nemo_env: `pip install -r nemo_requirements.txt` (в другое окружение)

---

1. ffmpeg:
   - Сначала ставим ffmpeg системно
   - Затем `python ffmpeg_scribe.py input.mp4` (берется из директории interviews)

2. whisper:
   - `python whisper_scribe.py input.wav` (берется из директории scraped_ffmpeg)

3. nemo:
   - Нужно зарегаться в nemo, получить токен и указать в .env как NEMO_HF_TOKEN
   - Лучше это запускать на linux по идее
   - pip install --upgrade pip setuptools wheel
   - pip install Cython
   - `python nemo_scribe.py input.wav` (берется из директории scraped_ffmpeg)
