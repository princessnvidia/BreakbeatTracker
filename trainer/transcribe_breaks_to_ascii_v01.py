#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BreakBrain - transcribe_breaks_to_ascii_v01.py

But :
Retranscrire tous tes breaks WAV en partitions ASCII 32 pas.

Entrée :
    breaks/*.wav

Sorties :
    dataset/ascii_transcriptions/ascii_transcriptions_v01.txt
    dataset/ascii_transcriptions/ascii_transcriptions_v01.json

Format :
    KICK : K...K...|....K...|K.......|...K....
    SNARE: ....S...|........|....S...|........
    GHOST: ..g.....|...g....|..g.g...|....g...
    HAT  : H.H.H.H.|H.H.H.H.|H.H.H.H.|H.H.H.H.
    FULL : K.H.S.H.|...

Notes :
Cette v01 utilise une classification automatique approximative.
Elle sert à créer un corpus d'apprentissage propre.
On pourra ensuite corriger les règles de détection.
"""

from pathlib import Path
import argparse
import json
import sys
from collections import Counter

import librosa
import numpy as np

BREAKS_DIR = Path("breaks")
OUT_DIR = Path("dataset/ascii_transcriptions")
OUT_TXT = OUT_DIR / "ascii_transcriptions_v01.txt"
OUT_JSON = OUT_DIR / "ascii_transcriptions_v01.json"

SR = 44100
STEPS = 32

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}


def normalize_audio(y):
    m = np.max(np.abs(y)) if len(y) else 0
    return y if m <= 0 else y / m


def find_breaks():
    return sorted(
        p for p in BREAKS_DIR.rglob("*")
        if p.suffix.lower() in AUDIO_EXTS and not p.name.endswith(".asd")
    )


def classify_slice(chunk, sr):
    """
    Classification simple :
    - kick : grave + énergie
    - snare : medium + énergie
    - ghost : faible énergie / petit coup
    - hat : aigu
    - perc : reste
    """

    if len(chunk) < 64:
        return "perc"

    rms = float(np.sqrt(np.mean(chunk * chunk)))

    centroid = float(np.mean(
        librosa.feature.spectral_centroid(y=chunk, sr=sr)
    ))

    zcr = float(np.mean(
        librosa.feature.zero_crossing_rate(chunk)
    ))

    dur = len(chunk) / sr

    if rms < 0.012:
        return "ghost"

    if centroid < 1200 and rms > 0.028:
        return "kick"

    if 1200 <= centroid <= 5200 and rms > 0.030:
        return "snare"

    if centroid > 4200 and dur < 0.32:
        return "hat"

    if centroid > 3600:
        return "hat"

    return "perc"


def quantize_step(sample_pos, total_len):
    if total_len <= 0:
        return 0

    pos = sample_pos / total_len
    step = int(round(pos * (STEPS - 1)))
    return max(0, min(STEPS - 1, step))


def priority_symbol(old, new):
    priority = {
        ".": 0,
        "p": 1,
        "H": 2,
        "g": 3,
        "K": 4,
        "S": 5,
    }
    return new if priority[new] >= priority[old] else old


def label_to_symbol(label):
    if label == "kick":
        return "K"
    if label == "snare":
        return "S"
    if label == "ghost":
        return "g"
    if label == "hat":
        return "H"
    return "p"


def split32(row):
    return "|".join(row[i:i+8] for i in range(0, STEPS, 8))


def transcribe_one(path, delta=0.16, wait=1):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y = normalize_audio(y)

    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="samples",
        backtrack=True,
        delta=delta,
        wait=wait,
    )

    points = sorted(set([0] + [int(x) for x in onsets] + [len(y)]))

    layers = {
        "kick": ["."] * STEPS,
        "snare": ["."] * STEPS,
        "ghost": ["."] * STEPS,
        "hat": ["."] * STEPS,
        "perc": ["."] * STEPS,
    }

    full = ["."] * STEPS

    events = []
    label_counter = Counter()

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        if end - start < int(sr * 0.030):
            continue

        # On limite l'analyse du timbre au début de la slice.
        analysis_end = min(end, start + int(sr * 0.30))
        chunk = y[start:analysis_end]

        label = classify_slice(chunk, sr)
        symbol = label_to_symbol(label)
        step = quantize_step(start, len(y))

        if label == "kick":
            layers["kick"][step] = "K"
        elif label == "snare":
            layers["snare"][step] = "S"
        elif label == "ghost":
            layers["ghost"][step] = "g"
        elif label == "hat":
            layers["hat"][step] = "H"
        else:
            layers["perc"][step] = "p"

        full[step] = priority_symbol(full[step], symbol)

        label_counter[label] += 1

        events.append({
            "step": step,
            "label": label,
            "symbol": symbol,
            "start_sample": int(start),
            "end_sample": int(end),
            "start_sec": float(start / sr),
            "duration_sec": float((end - start) / sr),
        })

    result = {
        "source": str(path),
        "name": path.name,
        "steps": STEPS,
        "duration_sec": float(len(y) / sr),
        "event_count": len(events),
        "label_count": dict(label_counter),
        "layers": {
            "kick": "".join(layers["kick"]),
            "snare": "".join(layers["snare"]),
            "ghost": "".join(layers["ghost"]),
            "hat": "".join(layers["hat"]),
            "perc": "".join(layers["perc"]),
        },
        "full": "".join(full),
        "events": events,
    }

    return result


def render_text(transcriptions):
    lines = []

    for idx, t in enumerate(transcriptions, start=1):
        lines.append(f"BREAK {idx:04d} - {t['name']}")
        lines.append("12345678|12345678|12345678|12345678")

        layers = t["layers"]

        lines.append("KICK : " + split32(layers["kick"]))
        lines.append("SNARE: " + split32(layers["snare"]))
        lines.append("GHOST: " + split32(layers["ghost"]))
        lines.append("HAT  : " + split32(layers["hat"]))
        lines.append("PERC : " + split32(layers["perc"]))
        lines.append("FULL : " + split32(t["full"]))

        counts = t["label_count"]
        lines.append(
            "COUNT: "
            + " ".join(f"{k}={counts.get(k, 0)}" for k in ["kick", "snare", "ghost", "hat", "perc"])
        )

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limiter le nombre de breaks pour tester.")
    parser.add_argument("--delta", type=float, default=0.16,
                        help="Sensibilité onset. Plus bas = plus de coups détectés.")
    parser.add_argument("--wait", type=int, default=1,
                        help="Attente onset. Plus haut = moins de découpes proches.")
    args = parser.parse_args()

    files = find_breaks()

    if args.limit:
        files = files[:args.limit]

    if not files:
        print("Aucun break trouvé dans :", BREAKS_DIR)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    transcriptions = []

    print(f"{len(files)} breaks à transcrire.")

    global_counts = Counter()

    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path.name}")

        try:
            t = transcribe_one(path, delta=args.delta, wait=args.wait)
        except Exception as e:
            print("  ERREUR :", e)
            continue

        transcriptions.append(t)
        global_counts.update(t["label_count"])

    payload = {
        "version": "ascii_transcriptions_v01",
        "steps": STEPS,
        "count": len(transcriptions),
        "global_label_count": dict(global_counts),
        "transcriptions": transcriptions,
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    OUT_TXT.write_text(
        render_text(transcriptions),
        encoding="utf-8"
    )

    print("")
    print("Terminé.")
    print("TXT :", OUT_TXT)
    print("JSON:", OUT_JSON)
    print("Stats globales:", dict(global_counts))


if __name__ == "__main__":
    main()
