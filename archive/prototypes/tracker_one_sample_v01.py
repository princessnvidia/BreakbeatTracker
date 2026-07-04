#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracker One Sample v01

Un tracker automatique pour breakbeat/jungle.

Principe :
- charge UN SEUL break source depuis breaks/
- découpe le break en slices successives
- crée un pattern original : 00, 01, 02, 03...
- génère des variations par mutations :
    swap
    repeat
    skip
    mini-fill
    reverse-fill
    jump vers slice proche
- recolle les slices une par une
- AUCUNE superposition
- AUCUN sample d'un autre break

Usage :
    python tracker_one_sample_v01.py --source "Camo"
    python tracker_one_sample_v01.py --source "Camo" --count 32 --mutation 0.25
    python tracker_one_sample_v01.py --source "Camo" --bpm 150 --steps 32
"""

from pathlib import Path
import argparse
import json
import random
import sys
import warnings

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    print("librosa manquant : pip install librosa soundfile numpy")
    sys.exit(1)


BREAKS_DIR = Path("breaks")
OUT_DIR = Path("exports/tracker_one_sample_v01")
SR = 44100
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}


def find_source(name):
    files = sorted(
        p for p in BREAKS_DIR.rglob("*")
        if p.suffix.lower() in AUDIO_EXTS and not p.name.endswith(".asd")
    )
    matches = [p for p in files if name.lower() in p.name.lower()]
    if not matches:
        print(f"Aucun break trouvé pour : {name}")
        sys.exit(1)
    return matches[0]


def normalize(y, peak=0.95):
    m = np.max(np.abs(y)) if len(y) else 0
    if m <= 1e-9:
        return y
    return y / m * peak


def fade(y, sr, ms=2):
    if len(y) < 16:
        return y

    n = min(int(sr * ms / 1000), len(y) // 4)
    if n <= 1:
        return y

    y = y.copy()
    ramp = np.linspace(0, 1, n)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return normalize(y), sr


def detect_slices(y, sr, delta=0.08, min_ms=35, max_ms=520):
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="samples",
        backtrack=True,
        delta=delta,
        wait=1,
    )

    points = sorted(set([0] + [int(x) for x in onsets] + [len(y)]))

    min_len = int(sr * min_ms / 1000)
    max_len = int(sr * max_ms / 1000)

    slices = []

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        if end - start < min_len:
            continue

        end = min(end, start + max_len)

        audio = y[start:end].copy()
        audio = fade(audio, sr, ms=2)

        slices.append({
            "index": len(slices),
            "start": int(start),
            "end": int(end),
            "duration": float((end - start) / sr),
            "audio": audio,
        })

    return slices


def make_original_pattern(slices, steps):
    if not slices:
        return []

    # Pattern tracker : une suite d'indices.
    # Si on demande plus de steps que de slices, on boucle.
    return [i % len(slices) for i in range(steps)]


def mutate_pattern(pattern, mutation=0.22, fill_chance=0.25):
    p = pattern[:]
    n = len(p)

    for i in range(n):
        if random.random() > mutation:
            continue

        action = random.choice([
            "repeat_prev",
            "repeat_next",
            "swap_local",
            "jump_near",
            "skip_forward",
            "keep",
        ])

        if action == "repeat_prev" and i > 0:
            p[i] = p[i - 1]

        elif action == "repeat_next" and i < n - 1:
            p[i] = p[i + 1]

        elif action == "swap_local":
            j = max(0, min(n - 1, i + random.choice([-2, -1, 1, 2])))
            p[i], p[j] = p[j], p[i]

        elif action == "jump_near":
            p[i] = max(0, p[i] + random.choice([-3, -2, -1, 1, 2, 3]))

        elif action == "skip_forward":
            p[i] = p[(i + random.choice([1, 2, 3])) % n]

    # Mini fills en fin de boucle : très tracker/jungle
    if n >= 8 and random.random() < fill_chance:
        fill_start = n - random.choice([4, 6, 8])
        fill_start = max(0, fill_start)

        mode = random.choice(["repeat", "chop", "reverse"])

        if mode == "repeat":
            src = p[fill_start]
            for i in range(fill_start, n):
                if random.random() < 0.65:
                    p[i] = src

        elif mode == "chop":
            local = p[max(0, fill_start-4):fill_start+1]
            if local:
                for i in range(fill_start, n):
                    p[i] = random.choice(local)

        elif mode == "reverse":
            segment = p[fill_start:n]
            p[fill_start:n] = list(reversed(segment))

    return p


def render_pattern(slices, pattern, bpm=150, step_mode="slice", steps_per_beat=4):
    """
    step_mode:
    - slice : chaque slice garde sa durée d'origine
    - grid  : chaque slice est recoupée sur une durée fixe de double-croche
    """
    chunks = []

    if step_mode == "grid":
        step_samples = int(SR * (60.0 / bpm / steps_per_beat))

        for idx in pattern:
            audio = slices[idx % len(slices)]["audio"].copy()
            audio = audio[:step_samples]

            if len(audio) < step_samples:
                pad = np.zeros(step_samples - len(audio), dtype=np.float32)
                audio = np.concatenate([audio, pad])

            chunks.append(fade(audio, SR, ms=2))

    else:
        for idx in pattern:
            chunks.append(slices[idx % len(slices)]["audio"])

    if not chunks:
        return np.zeros(1, dtype=np.float32)

    out = np.concatenate(chunks)
    return normalize(out)


def pattern_text(pattern, slices):
    lines = []
    lines.append("STEP | SLICE | DUR")
    lines.append("------------------")

    for i, idx in enumerate(pattern, start=1):
        s = slices[idx % len(slices)]
        lines.append(f"{i:04d} | {idx:05d} | {s['duration']:.3f}s")

    return "\n".join(lines)


def pattern_compact(pattern):
    return " ".join(f"{x:02d}" for x in pattern)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Camo")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--mutation", type=float, default=0.22)
    parser.add_argument("--fill-chance", type=float, default=0.30)
    parser.add_argument("--bpm", type=float, default=150.0)
    parser.add_argument("--step-mode", choices=["slice", "grid"], default="slice")
    parser.add_argument("--delta", type=float, default=0.08)
    parser.add_argument("--min-ms", type=int, default=35)
    parser.add_argument("--max-ms", type=int, default=520)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    source = find_source(args.source)
    print("Source unique :", source)

    y, sr = load_audio(source)
    slices = detect_slices(
        y,
        sr,
        delta=args.delta,
        min_ms=args.min_ms,
        max_ms=args.max_ms,
    )

    if not slices:
        print("Aucune slice détectée.")
        sys.exit(1)

    print(f"Slices détectées : {len(slices)}")

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)

    original = make_original_pattern(slices, args.steps)

    # export reconstruction tracker originale
    original_audio = render_pattern(slices, original, bpm=args.bpm, step_mode=args.step_mode)
    original_wav = outdir / f"{safe}_tracker_original_{args.step_mode}.wav"
    sf.write(original_wav, original_audio, SR)

    all_report = []
    all_report.append(f"SOURCE: {source}")
    all_report.append(f"SLICES: {len(slices)}")
    all_report.append(f"STEP_MODE: {args.step_mode}")
    all_report.append("")
    all_report.append("ORIGINAL PATTERN")
    all_report.append(pattern_compact(original))
    all_report.append("")

    renders = []

    for n in range(1, args.count + 1):
        pattern = mutate_pattern(
            original,
            mutation=args.mutation,
            fill_chance=args.fill_chance,
        )

        audio = render_pattern(
            slices,
            pattern,
            bpm=args.bpm,
            step_mode=args.step_mode,
        )

        wav = outdir / f"{safe}_tracker_mutation_{n:03d}_{args.step_mode}.wav"
        txt = outdir / f"{safe}_tracker_mutation_{n:03d}_{args.step_mode}.txt"

        sf.write(wav, audio, SR)
        txt.write_text(pattern_text(pattern, slices), encoding="utf-8")

        renders.append({
            "index": n,
            "wav": str(wav),
            "pattern": pattern,
        })

        all_report.append(f"VARIATION {n:03d} -> {wav}")
        all_report.append(pattern_compact(pattern))
        all_report.append("")

        print("Export :", wav)

    meta = {
        "version": "tracker_one_sample_v01",
        "source": str(source),
        "source_only": True,
        "slice_count": len(slices),
        "step_mode": args.step_mode,
        "bpm": args.bpm,
        "steps": args.steps,
        "mutation": args.mutation,
        "fill_chance": args.fill_chance,
        "original_wav": str(original_wav),
        "renders": renders,
        "slices": [
            {
                "index": s["index"],
                "start": s["start"],
                "end": s["end"],
                "duration": s["duration"],
            }
            for s in slices
        ],
    }

    (outdir / "tracker_report_v01.txt").write_text("\n".join(all_report), encoding="utf-8")
    (outdir / "tracker_metadata_v01.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("")
    print("Original :", original_wav)
    print("Dossier  :", outdir)


if __name__ == "__main__":
    main()
