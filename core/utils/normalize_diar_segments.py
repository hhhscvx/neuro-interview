import pandas as pd


def normalize_diar_segments(obj):
    """
    Приводит вывод диаризации к списку словарей [{start, end, speaker}]
    """
    # уже нужный формат
    if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
        return obj

    # словарь-обёртка с ключом 'segments'
    if (
        isinstance(obj, dict)
        and "segments" in obj
        and isinstance(obj["segments"], list)
    ):
        return obj["segments"]

    # pyannote Annotation
    if hasattr(obj, "itertracks"):
        out = []
        for seg, _, label in obj.itertracks(yield_label=True):
            out.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "speaker": str(label),
                }
            )
        return out

    if isinstance(obj, str):
        out = []
        for line in obj.splitlines():
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            try:
                start = float(parts[3])
                dur = float(parts[4])
                spk = parts[7]
                out.append({"start": start, "end": start + dur, "speaker": spk})
            except Exception:
                pass
        if out:
            return out

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

        # Нормализуем частые варианты наименований колонок
        cols = {c.lower(): c for c in df.columns}
        if {"start", "end", "speaker"}.issubset({c.lower() for c in df.columns}):
            return [
                {
                    "start": float(row[cols["start"]]),
                    "end": float(row[cols["end"]]),
                    "speaker": str(row[cols["speaker"]]),
                }
                for _, row in df.iterrows()
            ]
        if {"start", "end", "label"}.issubset({c.lower() for c in df.columns}):
            return [
                {
                    "start": float(row[cols["start"]]),
                    "end": float(row[cols["end"]]),
                    "speaker": str(row[cols["label"]]),
                }
                for _, row in df.iterrows()
            ]
        if "segment" in {c.lower() for c in df.columns}:
            seg_col = cols["segment"]
            spk_col = cols["speaker"] if "speaker" in cols else cols.get("label")
            if spk_col is None:
                raise TypeError(
                    "DataFrame имеет column 'segment', но нет 'speaker'/'label'"
                )
            out = []
            for _, row in df.iterrows():
                seg = row[seg_col]
                if hasattr(seg, "start") and hasattr(seg, "end"):
                    start, end = float(seg.start), float(seg.end)
                elif isinstance(seg, (tuple, list)) and len(seg) >= 2:
                    start, end = float(seg[0]), float(seg[1])
                else:
                    raise TypeError(f"Неизвестный тип segment: {type(seg)}")
                out.append({"start": start, "end": end, "speaker": str(row[spk_col])})
            return out

    raise TypeError(f"Unknown diarization output type: {type(obj)}")
