#!/usr/bin/env python3
"""
BreakBrain - generate_learned_groove_v02.py

Génère des variations à partir d'UN break uniquement.
Contrairement à la v01, cette version :
- garde l'ordre global du groove
- ajoute davantage de syncopes
- crée des ghost notes
- décale légèrement certaines slices
- ajoute des stutters principalement sur hats/ghosts
"""

from pathlib import Path
import argparse
import json
import random
import sys

import librosa
import numpy as np
import soundfile as sf

GROOVES = Path("dataset/grooves")
EXPORTS = Path("exports")
SR = 44100


def normalize(y):
    m = np.max(np.abs(y)) if len(y) else 0
    return y if m == 0 else y / m * 0.95


def find_groove(name):
    files = sorted(GROOVES.glob("*_groove.json"))
    for f in files:
        if name.lower() in f.name.lower():
            return f
    print("Groove introuvable :", name)
    sys.exit(1)


def load_chunk(audio, start, end):
    return audio[start:end].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groove-name", default="Camo")
    ap.add_argument("--variations", type=int, default=8)
    ap.add_argument("--strength", type=float, default=0.20)
    args = ap.parse_args()

    groove_path = find_groove(args.groove_name)
    groove = json.loads(groove_path.read_text(encoding="utf-8"))

    source = Path(groove["source"])
    audio, sr = librosa.load(source, sr=SR, mono=True)

    outdir = EXPORTS / f"learned_{source.stem.replace(' ','_')}"
    outdir.mkdir(parents=True, exist_ok=True)

    events = groove["events"]

    for v in range(1, args.variations + 1):
        out = np.zeros(len(audio) + SR, dtype=np.float32)

        for e in events:

            chunk = load_chunk(audio, e["start_sample"], e["end_sample"])

            pos = e["start_sample"]

            label = e["label"]

            # micro décalages
            if random.random() < args.strength:
                ms = random.choice(
                    [-45, -30, -15, 15, 30, 45]
                )
                pos += int(ms / 1000 * SR)

            # volume
            gain = 1.0

            if label == "ghost":
                gain *= random.uniform(0.25, 0.8)

            elif label == "hat":
                gain *= random.uniform(0.6, 1.1)

            elif random.random() < args.strength:
                gain *= random.uniform(0.8, 1.2)

            # stutter
            if label in ("hat", "ghost", "perc"):
                if random.random() < args.strength * 1.5:
                    reps = random.choice([2, 2, 3, 4])
                    part = chunk[: max(1, len(chunk)//reps)]

                    for r in range(reps):
                        p = pos + r * len(part)
                        end = min(len(out), p + len(part))
                        if p >= 0:
                            out[p:end] += part[:end-p] * gain * 0.9
                    continue

            # ghost supplémentaire
            if random.random() < args.strength * 0.5:
                ghost = chunk[: max(1, int(len(chunk)*0.4))]
                off = random.randint(-3500, 3500)
                p = pos + off
                if p >= 0:
                    end = min(len(out), p + len(ghost))
                    out[p:end] += ghost[:end-p] * 0.22

            if pos >= 0:
                end = min(len(out), pos + len(chunk))
                out[pos:end] += chunk[:end-pos] * gain

        out = normalize(out)
        outfile = outdir / f"{source.stem}_v02_{v:03d}.wav"
        sf.write(outfile, out, SR)
        print("Export :", outfile)

    print("Terminé :", outdir)


if __name__ == "__main__":
    main()
