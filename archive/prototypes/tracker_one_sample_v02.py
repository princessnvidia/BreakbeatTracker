#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracker One Sample v02

Tracker automatique one-sample avec transitions apprises.

Différence avec v01 :
- v01 mutait la suite de slices un peu au hasard
- v02 apprend les transitions naturelles du break :
      slice 00 -> slice 01
      slice 01 -> slice 02
      ...
  puis génère des variations qui restent proches de cette logique.

Toujours :
- UN SEUL break source
- AUCUN sample externe
- AUCUNE superposition
- audio = succession de slices

Usage :
    python tracker_one_sample_v02.py --source "London"
    python tracker_one_sample_v02.py --source "Camo" --count 32 --mutation 0.25
    python tracker_one_sample_v02.py --source "Stepper" --mode markov
"""

from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    print("librosa manquant : pip install librosa soundfile numpy")
    sys.exit(1)


BREAKS_DIR = Path("breaks")
OUT_DIR = Path("exports/tracker_one_sample_v02")
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
        audio = fade(y[start:end].copy(), sr, ms=2)

        slices.append({
            "index": len(slices),
            "start": int(start),
            "end": int(end),
            "duration": float((end - start) / sr),
            "audio": audio,
        })

    return slices


def original_pattern(slices, steps):
    return [i % len(slices) for i in range(steps)]


def build_transition_memory(pattern):
    """
    Apprend les transitions naturelles.
    Pour le break original :
        0 -> 1
        1 -> 2
        ...
    On ajoute aussi quelques transitions locales :
        i -> i+2
        i -> i-1
    avec poids plus faible.
    """
    transitions = {}

    n = len(pattern)

    for i, current in enumerate(pattern):
        nxt = pattern[(i + 1) % n]
        transitions.setdefault(current, {})
        transitions[current][nxt] = transitions[current].get(nxt, 0) + 10

        # transitions locales utiles pour variations crédibles
        for offset, weight in [(-2, 1), (-1, 2), (2, 3), (3, 1)]:
            j = (i + offset) % n
            alt = pattern[j]
            transitions[current][alt] = transitions[current].get(alt, 0) + weight

    return transitions


def weighted_choice(weight_map):
    items = list(weight_map.items())
    total = sum(w for _, w in items)
    if total <= 0:
        return random.choice(items)[0]
    r = random.random() * total
    acc = 0
    for item, weight in items:
        acc += weight
        if r <= acc:
            return item
    return items[-1][0]


def markov_generate(original, transitions, steps, mutation=0.22):
    """
    Génère une nouvelle suite.
    La plupart du temps : suit les transitions apprises.
    Parfois : revient au pattern original pour garder la structure.
    """
    if not original:
        return []

    current = original[0]
    out = [current]

    for step in range(1, steps):
        if random.random() < mutation and current in transitions:
            nxt = weighted_choice(transitions[current])
        else:
            # structure de base
            nxt = original[step % len(original)]

        out.append(nxt)
        current = nxt

    return out


def mutate_pattern(original, transitions, mutation=0.22, fill_chance=0.30, mode="hybrid"):
    n = len(original)

    if mode == "markov":
        p = markov_generate(original, transitions, n, mutation=mutation)
    elif mode == "local":
        p = original[:]
    else:
        p = markov_generate(original, transitions, n, mutation=mutation * 0.65)

    # Mutations locales façon tracker
    for i in range(n):
        if random.random() > mutation:
            continue

        action = random.choice([
            "repeat_prev",
            "repeat_next",
            "transition",
            "swap_neighbor",
            "micro_jump",
            "keep",
        ])

        if action == "repeat_prev" and i > 0:
            p[i] = p[i - 1]

        elif action == "repeat_next" and i < n - 1:
            p[i] = p[i + 1]

        elif action == "transition":
            current = p[i - 1] if i > 0 else p[i]
            if current in transitions:
                p[i] = weighted_choice(transitions[current])

        elif action == "swap_neighbor":
            j = max(0, min(n - 1, i + random.choice([-1, 1])))
            p[i], p[j] = p[j], p[i]

        elif action == "micro_jump":
            # saute vers une slice proche dans l'index source
            p[i] = max(0, min(max(original), p[i] + random.choice([-2, -1, 1, 2])))

    # Fills finaux
    if n >= 8 and random.random() < fill_chance:
        fill_start = n - random.choice([4, 6, 8])
        fill_start = max(0, fill_start)

        mode_fill = random.choice(["repeat", "chop", "transition_roll", "reverse"])

        if mode_fill == "repeat":
            src = p[fill_start]
            for i in range(fill_start, n):
                if random.random() < 0.72:
                    p[i] = src

        elif mode_fill == "chop":
            pool = p[max(0, fill_start - 4):fill_start + 1]
            for i in range(fill_start, n):
                p[i] = random.choice(pool)

        elif mode_fill == "transition_roll":
            current = p[fill_start]
            for i in range(fill_start, n):
                if current in transitions:
                    current = weighted_choice(transitions[current])
                p[i] = current

        elif mode_fill == "reverse":
            p[fill_start:n] = list(reversed(p[fill_start:n]))

    return p


def render_pattern(slices, pattern, bpm=150, step_mode="slice", steps_per_beat=4):
    chunks = []

    if step_mode == "grid":
        step_samples = int(SR * (60.0 / bpm / steps_per_beat))
        for idx in pattern:
            audio = slices[idx % len(slices)]["audio"].copy()
            audio = audio[:step_samples]
            if len(audio) < step_samples:
                audio = np.concatenate([audio, np.zeros(step_samples - len(audio), dtype=np.float32)])
            chunks.append(fade(audio, SR, ms=2))
    else:
        for idx in pattern:
            chunks.append(slices[idx % len(slices)]["audio"])

    if not chunks:
        return np.zeros(1, dtype=np.float32)

    return normalize(np.concatenate(chunks))


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
    parser.add_argument("--source", default="London")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--mutation", type=float, default=0.22)
    parser.add_argument("--fill-chance", type=float, default=0.30)
    parser.add_argument("--bpm", type=float, default=150.0)
    parser.add_argument("--step-mode", choices=["slice", "grid"], default="slice")
    parser.add_argument("--mode", choices=["hybrid", "markov", "local"], default="hybrid")
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
    slices = detect_slices(y, sr, delta=args.delta, min_ms=args.min_ms, max_ms=args.max_ms)

    if not slices:
        print("Aucune slice détectée.")
        sys.exit(1)

    print(f"Slices détectées : {len(slices)}")

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)

    original = original_pattern(slices, args.steps)
    transitions = build_transition_memory(original)

    original_audio = render_pattern(slices, original, bpm=args.bpm, step_mode=args.step_mode)
    original_wav = outdir / f"{safe}_tracker_original_v02_{args.step_mode}.wav"
    sf.write(original_wav, original_audio, SR)

    report = []
    report.append(f"SOURCE: {source}")
    report.append(f"SLICES: {len(slices)}")
    report.append(f"MODE: {args.mode}")
    report.append(f"STEP_MODE: {args.step_mode}")
    report.append("")
    report.append("ORIGINAL")
    report.append(pattern_compact(original))
    report.append("")

    renders = []

    for n in range(1, args.count + 1):
        pattern = mutate_pattern(
            original,
            transitions,
            mutation=args.mutation,
            fill_chance=args.fill_chance,
            mode=args.mode,
        )

        audio = render_pattern(slices, pattern, bpm=args.bpm, step_mode=args.step_mode)

        wav = outdir / f"{safe}_tracker_v02_{args.mode}_{n:03d}_{args.step_mode}.wav"
        txt = outdir / f"{safe}_tracker_v02_{args.mode}_{n:03d}_{args.step_mode}.txt"

        sf.write(wav, audio, SR)
        txt.write_text(pattern_text(pattern, slices), encoding="utf-8")

        renders.append({
            "index": n,
            "wav": str(wav),
            "pattern": pattern,
        })

        report.append(f"VARIATION {n:03d} -> {wav}")
        report.append(pattern_compact(pattern))
        report.append("")

        print("Export :", wav)

    meta = {
        "version": "tracker_one_sample_v02",
        "source": str(source),
        "source_only": True,
        "slice_count": len(slices),
        "mode": args.mode,
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
        "transitions": {
            str(k): {str(kk): vv for kk, vv in v.items()}
            for k, v in transitions.items()
        },
    }

    (outdir / "tracker_report_v02.txt").write_text("\n".join(report), encoding="utf-8")
    (outdir / "tracker_metadata_v02.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("")
    print("Original :", original_wav)
    print("Dossier  :", outdir)


if __name__ == "__main__":
    main()
