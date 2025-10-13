### Video —> Summary
Based on [this article](https://habr.com/ru/companies/alfa/articles/909498/), but instead of NeMo I use whisperx (easier to install) and feed the result to GPT for summarization

### Transcription stages:
1. ffmpeg to convert video to .wav audio for better whisper performance
2. whisper — transcribes audio to speech
3. whisperx — performs diarization, splits speech by speakers

### Installation stages:
1. `poetry install`
2. Install ffmpeg system-wide
3. For whisperx:
   - Register and get **Read token** on [huggingface](https://huggingface.co/settings/tokens) and specify in .env. Make sure to get access to [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) repos
4. Set other environment variables in .env: refer to .env.example
5. Upload videos for transcribation to settings.INTERVIEWS_PATH directory

### Running:
- `poetry run python main.py` — starts pipeline for all interviews that don't have ready transcription yet (not in settings.SCRAPED_RESULT_PATH directory)
- Flags `--skip-ffmpeg`, `--skip-whisper` and `--skip-whisperx` available to skip specific stages

#### Final step:
Feed the result to any LLM (e.g. GPT) and ask for **summary**. Better to split into convenient chunks first:
   - `python chunk_result.py <filename>` - *filename without extension, taken from settings.SCRAPED_RESULT_PATH directory*
- Prompt guide for summary: [prompt-guide.txt](prompt-guide.txt)
---

### Or run separately:

1. ffmpeg:
   - Then `python -m core.ffmpeg_scribe input.mp4` *(file taken from settings.INTERVIEWS_PATH directory)*

2. whisper:
   - `python -m core.whisper_scribe input.wav` *(file taken from settings.SCRAPED_FFMPEG_PATH directory)*

3. whisperx (diarization):
   - `python -m core.whisperx_diarize <filename>` *filename should be without extension, assumes filename.wav (after ffmpeg) and filename.json (after whisper) exist*
