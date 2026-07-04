#!/usr/bin/env python3
from pathlib import Path
import json, sys
import numpy as np
import librosa

BREAKS_DIR = Path("breaks")
OUT = Path("dataset/grooves")
SR = 44100
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}

def find_break(name):
    files = sorted(p for p in BREAKS_DIR.rglob("*")
                   if p.suffix.lower() in AUDIO_EXTS and not p.name.endswith(".asd"))
    matches = [p for p in files if name.lower() in p.name.lower()]
    if not matches:
        print("Break introuvable:", name)
        sys.exit(1)
    return matches[0]

def features(y, sr):
    rms = float(np.sqrt(np.mean(y*y))) if len(y) else 0
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) if len(y) > 64 else 0
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y))) if len(y) > 64 else 0
    return rms, centroid, zcr

def classify(rms, centroid, dur):
    if rms < 0.018:
        return "ghost"
    if centroid < 1350 and rms > 0.035:
        return "kick"
    if centroid > 4200 and dur < 0.25:
        return "hat"
    if 1300 <= centroid <= 5200 and rms > 0.028:
        return "snare"
    if centroid > 3800:
        return "hat"
    return "perc"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--break-name", default="Camo")
    args = ap.parse_args()

    path = find_break(args.break_name)
    print("Analyse:", path)

    y, sr = librosa.load(path, sr=SR, mono=True)

    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, units="samples", backtrack=True, delta=0.12, wait=1
    )

    points = sorted(set([0] + [int(x) for x in onsets] + [len(y)]))

    events = []
    for i in range(len(points)-1):
        start, end = points[i], points[i+1]
        if end-start < int(sr*0.025):
            continue

        chunk = y[start:end]
        dur = (end-start)/sr
        rms, centroid, zcr = features(chunk, sr)
        label = classify(rms, centroid, dur)

        events.append({
            "index": len(events),
            "start_sample": int(start),
            "end_sample": int(end),
            "start_sec": start/sr,
            "duration_sec": dur,
            "gap_to_next_sec": 0,
            "rms": rms,
            "centroid": centroid,
            "zcr": zcr,
            "label": label
        })

    for i in range(len(events)-1):
        events[i]["gap_to_next_sec"] = events[i+1]["start_sec"] - events[i]["start_sec"]

    safe = path.stem.replace(" ", "_").replace("'", "")
    OUT.mkdir(parents=True, exist_ok=True)

    groove = {
        "source": str(path),
        "sr": sr,
        "duration_sec": len(y)/sr,
        "event_count": len(events),
        "events": events
    }

    out = OUT / f"{safe}_groove.json"
    out.write_text(json.dumps(groove, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Groove appris:", out)
    print("Events:", len(events))
    print("Labels:")
    from collections import Counter
    print(Counter(e["label"] for e in events))

if __name__ == "__main__":
    main()

