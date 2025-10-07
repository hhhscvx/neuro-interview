
def normalize_diar_segments(obj):
    """
    Приводит вывод диаризации к списку словарей [{start, end, speaker}]
    """
    # уже нужный формат
    if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
        return obj

    # словарь-обёртка с ключом 'segments'
    if isinstance(obj, dict) and "segments" in obj and isinstance(obj["segments"], list):
        return obj["segments"]

    # pyannote Annotation
    if hasattr(obj, "itertracks"):
        out = []
        for seg, _, label in obj.itertracks(yield_label=True):
            out.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(label)})
        return out

    if isinstance(obj, str):
        out = []
        for line in obj.splitlines():
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            try:
                start = float(parts[3]); dur = float(parts[4]); spk = parts[7]
                out.append({"start": start, "end": start + dur, "speaker": spk})
            except Exception:
                pass
        if out:
            return out

    raise TypeError(f"Unknown diarization output type: {type(obj)}")
