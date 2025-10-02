import argparse
import json
import os
from time import perf_counter

from faster_whisper import WhisperModel
import torch


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--lang", type=str, default="ru")
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--model", default="medium", help="small|medium|large-v3")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # MPS (Metal) для Mac M1/M2
    return "cpu"


print("Устройство:", get_device().upper())


def main():
    args = parser.parse_args()

    model = WhisperModel("medium", device=get_device(), compute_type="int8")

    print("Транскрибирование...")
    t0 = perf_counter()
    segments, info = model.transcribe(
        audio=f"scraped_ffmpeg/{args.path}",
        language=args.lang,
        vad_filter=True,
        temperature=args.temperature,
        beam_size=args.beam_size,
        initial_prompt=args.prompt,
    )

    if not os.path.exists("scraped_whisper"):
        os.mkdir("scraped_whisper")

    scraped_whisper_path = f"scraped_whisper/{args.path.split('.')[0]}"
    
    segments_list = []
    full_text = ""
    
    for segment in segments:
        segment_dict = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        }
        print(segment_dict)
        segments_list.append(segment_dict)
        full_text += segment.text.strip() + " "

    print(f"Транскрибация заняла {round(perf_counter()-t0,2)} с")
    
    with open(scraped_whisper_path + ".json", "w", encoding="utf-8") as f:
        json.dump(segments_list, f, ensure_ascii=False, indent=2)
    with open(scraped_whisper_path + ".txt", "w", encoding="utf-8") as f:
        f.write(full_text.strip())
    print("Сохранено:", scraped_whisper_path + ".json / .txt")


if __name__ == "__main__":
    main()
