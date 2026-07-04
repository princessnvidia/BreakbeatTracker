#!/usr/bin/env python3
from pathlib import Path
import argparse
import random
import sys
import numpy as np
import librosa
import soundfile as sf

BREAKS_DIR = Path("breaks")
EXPORTS_DIR = Path("exports")
SR = 44100

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}

def norm(y):
    m = np.max(np.abs(y)) if len(y) else 0
    return y / m * 0.95 if m > 0 else y

def find_break(name):
    files = sorted(
        p for p in BREAKS_DIR.rglob("*")
        if p.suffix.lower() in AUDIO_EXTS and not p.name.endswith(".asd")
    )

    matches = [p for p in files if name.lower() in p.name.lower()]

    if not matches:
        print(f"Aucun break trouvé avec : {name}")
        sys.exit(1)

    return matches[0]

def detect_slices(y, sr):
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="samples",
        backtrack=True,
        delta=0.12,
        wait=1
    )

    points = sorted(set([0] + [int(x) for x in onsets] + [len(y)]))

    slices = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        if end - start < int(sr * 0.025):
            continue

        chunk = y[start:end].copy()

        fade = min(int(sr * 0.003), len(chunk) // 4)
        if fade > 1:
            ramp = np.linspace(0, 1, fade)
            chunk[:fade] *= ramp
            chunk[-fade:] *= ramp[::-1]

        slices.append({
            "start": start,
            "end": end,
            "audio": chunk,
            "length": end - start,
        })

    return slices

def reconstruct_original(slices, total_len):
    out = np.zeros(total_len + SR, dtype=np.float32)

    for s in slices:
        start = s["start"]
        audio = s["audio"]
        end = min(start + len(audio), len(out))
        out[start:end] += audio[:end-start]

    return out

def mutate_slices(slices, total_len, strength=0.08):
    out = np.zeros(total_len + SR, dtype=np.float32)

    for i, s in enumerate(slices):
        audio = s["audio"].copy()
        start = s["start"]

        # Mutation légère de timing : quelques ms seulement
        if random.random() < strength:
            shift_ms = random.uniform(-18, 18)
            start += int(SR * shift_ms / 1000)

        # Quelques slices ghost sont baissées
        vol = 1.0
        if random.random() < strength * 1.5:
            vol *= random.uniform(0.35, 0.75)

        # Quelques répétitions rapides façon breakbeat
        repeats = 1
        if random.random() < strength * 0.55 and len(audio) > int(SR * 0.04):
            repeats = random.choice([2, 2, 3])
            audio = audio[:max(1, len(audio)//repeats)]

        for r in range(repeats):
            pos = start + r * len(audio)
            if pos < 0:
                continue
            end = min(pos + len(audio), len(out))
            if end > pos:
                out[pos:end] += audio[:end-pos] * vol

        # Petit ghost copié juste avant/après
        if random.random() < strength * 0.35:
            offset = random.choice([-1, 1]) * random.randint(800, 2600)
            pos = start + offset
            ghost = audio[:min(len(audio), int(SR * 0.07))] * 0.22
            if pos >= 0:
                end = min(pos + len(ghost), len(out))
                out[pos:end] += ghost[:end-pos]

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--break-name", default="Camo")
    parser.add_argument("--variations", type=int, default=8)
    parser.add_argument("--strength", type=float, default=0.08)
    args = parser.parse_args()

    path = find_break(args.break_name)

    print(f"Break source : {path}")

    y, sr = librosa.load(path, sr=SR, mono=True)
    y = norm(y)

    slices = detect_slices(y, sr)

    print(f"Slices détectées : {len(slices)}")

    safe = path.stem.replace(" ", "_").replace("'", "")
    outdir = EXPORTS_DIR / f"mutate_{safe}"
    outdir.mkdir(parents=True, exist_ok=True)

    original = reconstruct_original(slices, len(y))
    sf.write(outdir / f"{safe}_reconstruction.wav", norm(original), SR)

    print(f"Reconstruction : {outdir}/{safe}_reconstruction.wav")

    for i in range(1, args.variations + 1):
        mutated = mutate_slices(slices, len(y), strength=args.strength)
        out = outdir / f"{safe}_mutation_{i:03d}.wav"
        sf.write(out, norm(mutated), SR)
        print(f"Mutation {i:03d} : {out}")

    print("")
    print("Terminé.")
    print(f"Dossier : {outdir}")

if __name__ == "__main__":
    main()
