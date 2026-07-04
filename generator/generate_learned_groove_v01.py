#!/usr/bin/env python3
from pathlib import Path
import argparse, json, random, sys
import numpy as np
import librosa
import soundfile as sf

GROOVES = Path("dataset/grooves")
EXPORTS = Path("exports")
SR = 44100

def norm(y):
    m = np.max(np.abs(y)) if len(y) else 0
    return y / m * 0.95 if m > 0 else y

def find_groove(name):
    files = sorted(GROOVES.glob("*_groove.json"))
    matches = [p for p in files if name.lower() in p.name.lower()]
    if not matches:
        print("Groove introuvable:", name)
        sys.exit(1)
    return matches[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groove-name", default="Camo")
    ap.add_argument("--variations", type=int, default=8)
    ap.add_argument("--strength", type=float, default=0.05)
    args = ap.parse_args()

    groove_path = find_groove(args.groove_name)
    groove = json.load(open(groove_path, encoding="utf-8"))

    source = Path(groove["source"])
    print("Groove:", groove_path)
    print("Source:", source)

    y, sr = librosa.load(source, sr=SR, mono=True)
    events = groove["events"]

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = EXPORTS / f"learned_{safe}"
    outdir.mkdir(parents=True, exist_ok=True)

    for v in range(1, args.variations + 1):
        out = np.zeros(len(y) + SR, dtype=np.float32)

        for e in events:
            start = e["start_sample"]
            end = e["end_sample"]
            chunk = y[start:end].copy()

            # mutation timing très légère
            shift = 0
            if random.random() < args.strength:
                shift = int(SR * random.uniform(-0.012, 0.012))

            # mutation volume selon type
            vol = 1.0
            if e["label"] == "ghost":
                vol *= random.uniform(0.45, 0.9)
            elif e["label"] == "hat":
                vol *= random.uniform(0.65, 1.05)
            elif random.random() < args.strength:
                vol *= random.uniform(0.75, 1.15)

            # très petit stutter uniquement sur hats/ghosts
            repeats = 1
            if e["label"] in ["hat", "ghost"] and random.random() < args.strength * 0.6:
                repeats = 2
                chunk = chunk[:max(1, len(chunk)//2)]

            pos = start + shift

            for r in range(repeats):
                p = pos + r * len(chunk)
                if p < 0:
                    continue
                endpos = min(p + len(chunk), len(out))
                out[p:endpos] += chunk[:endpos-p] * vol

        wav = outdir / f"{safe}_learned_mutation_{v:03d}.wav"
        sf.write(wav, norm(out), SR)
        print("Export:", wav)

    print("Terminé:", outdir)

if __name__ == "__main__":
    main()

