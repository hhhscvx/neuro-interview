import signal

if not hasattr(signal, "SIGKILL"):
    signal.SIGKILL = signal.SIGTERM
import sys
import os
import json
from pathlib import Path
import numpy as np
import librosa
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from nemo.collections.asr.models import EncDecSpeakerLabelModel


def extract_embeddings(
    wav: np.ndarray,
    sr: int,
    model: EncDecSpeakerLabelModel,
    win_s: float = 3.0,
    step_s: float = 1.5,
):
    import soundfile as sf, tempfile, os as _os

    embs, stamps = [], []
    t = 0.0
    total_dur = len(wav) / sr
    while t + win_s <= total_dur:
        segment = wav[int(t * sr) : int((t + win_s) * sr)]

        # создаём временный wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, segment, sr)
            tmp_path = tmp.name
        try:
            with torch.no_grad():
                emb = model.get_embedding(tmp_path).cpu().numpy().squeeze()
            embs.append(emb / np.linalg.norm(emb))
            stamps.append((t, t + win_s))
        finally:
            _os.remove(tmp_path)
        t += step_s
    return np.stack(embs), stamps


def auto_cluster(embs: np.ndarray, max_k: int = 10):
    """Перебирает кластеризации 2..max_k, выбирает лучшую по silhouette"""
    best_lbl, best_sc = None, -1
    # ограничим число соседей
    n_samples = len(embs)
    n_neighbors = max(2, min(10, n_samples - 1))
    for k in range(2, min(max_k, n_samples) + 1):
        lbl = SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors", n_neighbors=n_neighbors,
            assign_labels="kmeans", random_state=42
        ).fit_predict(embs)
        try:
            sc = silhouette_score(embs, lbl)
        except Exception:
            sc = -1
        if sc > best_sc:
            best_lbl, best_sc = lbl, sc
    return best_lbl


def merge_segments(stamps, labels, gap: float = 0.35):
    """Склеивает соседние окна одного спикера, если пауза < gap"""
    merged = []
    cur = {"spk": int(labels[0]), "s": stamps[0][0], "e": stamps[0][1]}
    for (s, e), lab in zip(stamps[1:], labels[1:]):
        lab = int(lab)
        if lab == cur["spk"] and s <= cur["e"] + gap:
            cur["e"] = e
        else:
            merged.append(cur)
            cur = {"spk": lab, "s": s, "e": e}
    merged.append(cur)
    return merged


def main():
    if len(sys.argv) != 4:
        print(
            "Usage:  python diarize_nemo_auto.py <audio.wav> <whisper.json> <max_speakers>"
        )
        sys.exit(1)
    wav_path, whisper_json, max_k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if not Path(wav_path).is_file() or not Path(whisper_json).is_file():
        print("File not found.")
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading NeMo speaker‑embedding model …")

    def _load_model():
        # Пытаемся вызвать from_pretrained() с токеном, затем без.
        repo = "nvidia/speakerverification_en_titanet_large"
        token = os.getenv("NEMO_HF_TOKEN")
        try:
            if token:
                return EncDecSpeakerLabelModel.from_pretrained(repo, token=token)
            raise TypeError
        except TypeError:
            return EncDecSpeakerLabelModel.from_pretrained(repo)

    model = _load_model().to(device).eval()
    # Читаем аудио
    print("Reading audio …")
    wav, sr = librosa.load(wav_path, sr=16000, mono=True)
    # Извлекаем эмбеддинги
    print("Extracting embeddings …")
    embs, stamps = extract_embeddings(wav, sr, model)
    # Кластеризуем
    print(f"Auto‑clustering 2..{max_k} speakers …")
    labels = auto_cluster(embs, max_k=max_k)
    spk_cnt = len(set(labels))
    print(f"   → selected {spk_cnt} speakers.")
    diar = merge_segments(stamps, labels)
    # Мержим c Whisper
    print("Merging with Whisper transcript …")
    with open(whisper_json, encoding="utf-8") as f:
        whisper_segs = json.load(f)
    tagged = []
    for seg in whisper_segs:
        spk = next(
            (
                f"Speaker{d['spk'] + 1}"
                for d in diar
                if not (seg["end"] <= d["s"] or seg["start"] >= d["e"])
            ),
            "Unknown",
        )
        tagged.append({**seg, "speaker": spk})
    out_path = Path(whisper_json).with_stem(Path(whisper_json).stem + "_tagged")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tagged, f, ensure_ascii=False, indent=2)
    print(f"Tagged transcript saved → {out_path}")


if __name__ == "__main__":
    main()
