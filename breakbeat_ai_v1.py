#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Breakbeat AI v1
---------------
Prend un break WAV, le découpe automatiquement, classe grossièrement les slices,
puis génère des variations breakbeat syncopées à 150 BPM.

Usage :
    python3 breakbeat_ai_v1.py input.wav

Options :
    python3 breakbeat_ai_v1.py input.wav --bpm 150 --variations 8
    python3 breakbeat_ai_v1.py input.wav --steps 32
    python3 breakbeat_ai_v1.py input.wav --out exports_breaks
"""

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


def require_librosa():
    try:
        import librosa
        import soundfile as sf
        return librosa, sf
    except ImportError:
        print("\nIl manque des dépendances.")
        print("Installe-les avec :")
        print("  python3 -m pip install librosa soundfile numpy")
        sys.exit(1)


def load_audio(path, sr=44100):
    librosa, _ = require_librosa()
    y, sr = librosa.load(path, sr=sr, mono=True)
    if len(y) == 0:
        raise ValueError("Audio vide.")
    return y.astype(np.float32), sr


def normalize(y, peak=0.95):
    maxv = float(np.max(np.abs(y))) if len(y) else 0.0
    if maxv < 1e-9:
        return y
    return (y / maxv * peak).astype(np.float32)


def estimate_bpm(y, sr):
    librosa, _ = require_librosa()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    try:
        tempo = float(tempo)
    except Exception:
        tempo = 0.0
    if tempo <= 0:
        tempo = 150.0
    return tempo


def detect_onsets(y, sr, min_gap_ms=45):
    librosa, _ = require_librosa()

    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="frames",
        backtrack=True,
        pre_max=3,
        post_max=3,
        pre_avg=8,
        post_avg=8,
        delta=0.18,
        wait=1,
    )

    samples = librosa.frames_to_samples(onset_frames)

    # Toujours inclure début et fin.
    samples = sorted(set([0] + [int(s) for s in samples] + [len(y)]))

    min_gap = int(sr * min_gap_ms / 1000)
    cleaned = []
    last = -10**12
    for s in samples:
        if s - last >= min_gap:
            cleaned.append(s)
            last = s

    if cleaned[-1] != len(y):
        cleaned.append(len(y))

    return cleaned


def slice_audio(y, points, sr, min_len_ms=35, max_len_ms=700):
    min_len = int(sr * min_len_ms / 1000)
    max_len = int(sr * max_len_ms / 1000)

    slices = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        if end - start < min_len:
            continue

        end = min(end, start + max_len)
        chunk = y[start:end].copy()

        # petit fade pour éviter les clics
        fade = min(int(sr * 0.004), len(chunk) // 4)
        if fade > 1:
            ramp = np.linspace(0, 1, fade)
            chunk[:fade] *= ramp
            chunk[-fade:] *= ramp[::-1]

        slices.append({
            "index": len(slices),
            "start": int(start),
            "end": int(end),
            "audio": chunk,
        })

    return slices


def spectral_features(chunk, sr):
    librosa, _ = require_librosa()

    if len(chunk) < 64:
        return {
            "rms": 0.0,
            "centroid": 0.0,
            "zcr": 0.0,
            "duration": len(chunk) / sr,
        }

    rms = float(np.sqrt(np.mean(chunk * chunk)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))
    duration = float(len(chunk) / sr)

    return {
        "rms": rms,
        "centroid": centroid,
        "zcr": zcr,
        "duration": duration,
    }


def classify_slice(chunk, sr):
    """
    Classification volontairement simple :
    - kick : grave / centroid bas / énergie forte
    - snare : énergie forte et medium/aigu
    - hat : très aigu / court
    - ghost : faible énergie, souvent snare/perc
    - perc : reste
    """
    f = spectral_features(chunk, sr)
    rms = f["rms"]
    centroid = f["centroid"]
    zcr = f["zcr"]
    duration = f["duration"]

    if rms < 0.018:
        return "ghost", f

    if centroid < 1350 and rms > 0.035:
        return "kick", f

    if centroid > 4200 and duration < 0.24:
        return "hat", f

    if 1300 <= centroid <= 5200 and rms > 0.028:
        return "snare", f

    if centroid > 3800:
        return "hat", f

    return "perc", f


def save_wav(path, y, sr):
    _, sf = require_librosa()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, normalize(y), sr)


def export_slices(slices, sr, out_dir):
    by_type = {"kick": [], "snare": [], "hat": [], "ghost": [], "perc": []}
    metadata = []

    for s in slices:
        label, features = classify_slice(s["audio"], sr)
        by_type[label].append(s)

        filename = f"{label}_{len(by_type[label]):03d}.wav"
        folder = out_dir / "slices" / label
        save_wav(folder / filename, s["audio"], sr)

        metadata.append({
            "index": s["index"],
            "label": label,
            "filename": str(Path("slices") / label / filename),
            "start_sample": s["start"],
            "end_sample": s["end"],
            "features": features,
        })

    with open(out_dir / "slices_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return by_type, metadata


def choose(pool, fallback_pools):
    if pool:
        return random.choice(pool)
    for p in fallback_pools:
        if p:
            return random.choice(p)
    return None


def make_pattern(steps=32, snare_density=0.55, hat_density=0.72, chaos=0.25):
    """
    Pattern breakbeat syncopé.
    steps=32 = 2 mesures en double-croches si on pense 16 steps par mesure,
    ou 4 groupes de 8 façon tracker.
    """
    kick = ["."] * steps
    snare = ["."] * steps
    hat = ["."] * steps

    # snares principales : backbeat
    for p in [8, 24]:
        if p < steps:
            snare[p] = "S"

    # kicks syncopés, inspirés downtempo / breakbeat
    base_kicks = [0, 6, 10, 16, 20, 27]
    for p in base_kicks:
        if p < steps and random.random() < 0.86:
            kick[p] = "K"

    # possibilités de kicks supplémentaires
    extra_kicks = [3, 11, 14, 18, 23, 26, 29, 30]
    for p in extra_kicks:
        if p < steps and random.random() < chaos * 0.55:
            kick[p] = "K"

    # ghost snares autour du backbeat + roulements
    ghost_candidates = [4, 6, 7, 11, 12, 13, 14, 18, 21, 22, 23, 26, 27, 28, 30, 31]
    for p in ghost_candidates:
        if p < steps and snare[p] == "." and random.random() < snare_density:
            snare[p] = "g"

    # hats : croches + petits paquets
    for p in range(steps):
        if p % 2 == 0:
            if random.random() < 0.95:
                hat[p] = "H"
        else:
            if random.random() < hat_density:
                hat[p] = "H"

    # fills en fin de boucle
    if steps >= 32 and random.random() < 0.7:
        for p in [28, 30, 31]:
            if random.random() < 0.65:
                snare[p] = "g" if snare[p] == "." else snare[p]
                hat[p] = "H"

    return {"kick": kick, "snare": snare, "hat": hat}


def pattern_to_ascii(pattern):
    def group(row):
        s = "".join(row)
        return "|".join(s[i:i+8] for i in range(0, len(s), 8))

    nums = "12345678|" * (len(pattern["kick"]) // 8)
    nums = nums.rstrip("|")

    return "\n".join([
        nums,
        f"KICK : {group(pattern['kick'])}",
        f"SNARE: {group(pattern['snare'])}",
        f"HAT  : {group(pattern['hat'])}",
    ])


def render_pattern(pattern, pools, bpm, sr, out_path):
    steps = len(pattern["kick"])
    seconds_per_step = 60.0 / bpm / 4.0  # double-croche
    step_samples = int(sr * seconds_per_step)
    total = step_samples * steps

    out = np.zeros(total + sr, dtype=np.float32)

    all_pools = pools.get("kick", []) + pools.get("snare", []) + pools.get("hat", []) + pools.get("ghost", []) + pools.get("perc", [])

    for i in range(steps):
        events = []

        if pattern["kick"][i] == "K":
            s = choose(pools.get("kick", []), [pools.get("perc", []), all_pools])
            if s:
                events.append((s["audio"], 0.95))

        if pattern["snare"][i] == "S":
            s = choose(pools.get("snare", []), [pools.get("perc", []), all_pools])
            if s:
                events.append((s["audio"], 0.9))

        if pattern["snare"][i] == "g":
            s = choose(pools.get("ghost", []), [pools.get("snare", []), pools.get("perc", []), all_pools])
            if s:
                events.append((s["audio"], 0.32))

        if pattern["hat"][i] == "H":
            s = choose(pools.get("hat", []), [pools.get("ghost", []), pools.get("perc", []), all_pools])
            if s:
                events.append((s["audio"], 0.36))

        start = i * step_samples
        for audio, vol in events:
            max_len = min(len(audio), int(step_samples * 2.2))
            chunk = audio[:max_len] * vol
            end = min(start + len(chunk), len(out))
            out[start:end] += chunk[:end-start]

    save_wav(out_path, out, sr)


def main():
    parser = argparse.ArgumentParser(description="IA v1 pour découper un break et générer des breakbeats syncopés.")
    parser.add_argument("input", help="Fichier WAV/AIFF/MP3 source")
    parser.add_argument("--bpm", type=float, default=150.0, help="BPM de sortie")
    parser.add_argument("--steps", type=int, default=32, help="Nombre de pas")
    parser.add_argument("--variations", type=int, default=8, help="Nombre de breaks générés")
    parser.add_argument("--out", default="breakbeat_ai_output", help="Dossier de sortie")
    parser.add_argument("--seed", type=int, default=None, help="Seed aléatoire")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    in_path = Path(args.input).expanduser()
    if not in_path.exists():
        print(f"Fichier introuvable : {in_path}")
        sys.exit(1)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Chargement audio...")
    y, sr = load_audio(in_path)

    estimated = estimate_bpm(y, sr)
    print(f"Tempo estimé du fichier : {estimated:.1f} BPM")
    print(f"Tempo de sortie : {args.bpm:.1f} BPM")

    print("Détection des attaques...")
    points = detect_onsets(y, sr)
    print(f"Points détectés : {len(points)}")

    print("Découpe des slices...")
    slices = slice_audio(y, points, sr)
    print(f"Slices gardées : {len(slices)}")

    print("Export + classification des slices...")
    pools, metadata = export_slices(slices, sr, out_dir)

    print("Répartition :")
    for label in ["kick", "snare", "hat", "ghost", "perc"]:
        print(f"  {label:6}: {len(pools.get(label, []))}")

    patterns_dir = out_dir / "patterns"
    exports_dir = out_dir / "exports"
    patterns_dir.mkdir(exist_ok=True)
    exports_dir.mkdir(exist_ok=True)

    print("\nGénération des variations...\n")
    all_ascii = []

    for n in range(1, args.variations + 1):
        pattern = make_pattern(steps=args.steps)
        ascii_grid = pattern_to_ascii(pattern)

        wav_path = exports_dir / f"break_{args.bpm:.0f}bpm_{n:03d}.wav"
        json_path = patterns_dir / f"pattern_{n:03d}.json"
        txt_path = patterns_dir / f"pattern_{n:03d}.txt"

        render_pattern(pattern, pools, args.bpm, sr, wav_path)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pattern, f, indent=2, ensure_ascii=False)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(ascii_grid + "\n")

        all_ascii.append(f"VARIATION {n:03d}\n{ascii_grid}\n")
        print(f"Variation {n:03d}")
        print(ascii_grid)
        print()

    with open(out_dir / "all_patterns.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_ascii))

    print(f"Terminé. Dossier : {out_dir.resolve()}")
    print(f"Écoute les WAV dans : {(exports_dir).resolve()}")


if __name__ == "__main__":
    main()
