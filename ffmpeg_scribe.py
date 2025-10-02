import os
import argparse
import ffmpeg


def process_audio(input_path, output_path):
    """Обработка аудио с удалением пауз"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Используем ffmpeg-python
    stream = ffmpeg.input(input_path)
    stream = ffmpeg.output(
        stream,
        output_path,
        ac=1,
        ar=16000,
        af="silenceremove=start_periods=1:start_silence=0.3:start_threshold=-35dB:detection=peak"
    )
    
    # Запускаем ffmpeg
    ffmpeg.run(stream, overwrite_output=True)
    return True

def main():
    parser = argparse.ArgumentParser(description='Обработка аудио с удалением пауз')
    parser.add_argument('input_file', help='Имя входного файла (без пути)')
    
    args = parser.parse_args()
    output_file = args.input_file.split(".")[0]
    
    input_path = f"interviews/{args.input_file}"
    output_path = f"scraped_ffmpeg/{output_file}.wav"
    
    try:
        process_audio(input_path, output_path)
        print(f"Обработка завершена: {output_path}")
    except Exception as e:
        print(f"Ошибка: {e}")
        raise e

if __name__ == "__main__":
    exit(main())
