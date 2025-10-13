### Видео —> Выжимка
За основу брал [эту статью](https://habr.com/ru/companies/alfa/articles/909498/), но вместо NeMo использую whisperx (проще ставится) и в конце скармливаю в GPT для выжимки полученного

### Этапы транскрибации:
1. ffmpeg для преобразования видео в .wav аудио, чтоб whisper-у было проще и лучше работать
2. whisper — делает транскрибацию из аудио в речь
3. whisperx — делает диаризация, разбивает речь по спикерам

### Этапы установки:
1. `poetry install`
2. Поставить ffmpeg системно
3. Для whisperx: 
   - Зарегаться и получить **Read токен** на [huggingface](https://huggingface.co/settings/tokens) и указать в .env. Обязательно получить доступ на репо [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) и [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Установить прочие переменные окружения в .env: опираться на .env.example
5. Загрузить видео для транскрибации в директорию settings.INTERVIEWS_PATH

### Запуск:
- `poetry run python main.py` — начинает пайплайн для всех интервью, у которых еще нет готовой транскрибации (их нет в директории settings.SCRAPED_RESULT_PATH)
- Есть флаги `--skip-ffmpeg`, `--skip-whisper` и `--skip-whisperx`, если нужно пропустить какой-то из этапов

#### Финал:
Скормить полученное любой LLM (например GPT) и попросить сделать **выжимку**. Лучше перед этим разбить по удобным чанкам:
   - `python chunk_result.py <filename>` - *filename без расширения, он берется из директории settings.SCRAPED_RESULT_PATH*
- Гайд на промпт для выжимки: [prompt-guide.txt](prompt-guide.txt)

---

### Или вызов по отдельности:

1. ffmpeg:
   - Затем `python -m core.ffmpeg_scribe input.mp4` *(файл берется из директории settings.INTERVIEWS_PATH)*

2. whisper:
   - `python -m core.whisper_scribe input.wav` *(файл берется из директории settings.SCRAPED_FFMPEG_PATH)*

3. whisperx (диаризация):
   - `python -m core.whisperx_diarize <filename>` *filename должен быть без расширения, подразуемевается, что есть filename.wav (после ffmpeg) и filename.json (после whisper)*
