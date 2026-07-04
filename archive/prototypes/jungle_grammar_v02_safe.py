#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jungle Grammar v02 SAFE

Correction :
- v01 était trop fouillis
- v02 revient à une génération sobre proche de la base Amen
- pas de stutter agressif par défaut
- pas de chop aléatoire destructeur
- mutations deux par deux uniquement
- variations très proches de la reconstruction originale
- WAV seulement + metadata global

Usage recommandé :
    python jungle_grammar_v02_safe.py --source "Amen" --count 64

Un peu plus de variation :
    python jungle_grammar_v02_safe.py --source "Amen" --mutation 0.18 --count 64

Très sage :
    python jungle_grammar_v02_safe.py --source "Amen" --mutation 0.08 --count 64
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
OUT_DIR = Path("exports/jungle_grammar_v02_safe")
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
    return y if m <= 1e-9 else y / m * peak


def fade(y, ms=2):
    if len(y) < 16:
        return y
    n = min(int(SR * ms / 1000), len(y) // 4)
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


def clean_cuts(cuts, total_len, min_samples):
    cuts = sorted(set(int(c) for c in cuts if 0 <= int(c) < total_len))
    if not cuts or cuts[0] != 0:
        cuts = [0] + cuts
    if cuts[-1] != total_len:
        cuts.append(total_len)

    cleaned = [cuts[0]]
    for c in cuts[1:]:
        if c - cleaned[-1] >= min_samples or c == total_len:
            cleaned.append(c)

    if cleaned[-1] != total_len:
        cleaned.append(total_len)

    return cleaned


def cuts_to_slices(y, cuts):
    slices = []
    for i in range(len(cuts) - 1):
        start = cuts[i]
        end = cuts[i + 1]
        if end <= start:
            continue
        audio = fade(y[start:end].copy(), ms=2)
        slices.append({
            "index": i,
            "start": int(start),
            "end": int(end),
            "start_ms": float(start / SR * 1000),
            "end_ms": float(end / SR * 1000),
            "duration": float((end - start) / SR),
            "audio": audio,
        })
    return slices


def onset_strong_cuts(y, target_cuts, min_ms):
    env = librosa.onset.onset_strength(y=y, sr=SR)
    samples = librosa.frames_to_samples(np.arange(len(env)))

    peaks = librosa.util.peak_pick(
        env,
        pre_max=3,
        post_max=3,
        pre_avg=8,
        post_avg=8,
        delta=0.08,
        wait=1,
    )

    ranked = sorted(peaks, key=lambda i: env[i], reverse=True)
    chosen = sorted(samples[i] for i in ranked[:max(1, target_cuts - 1)])
    return clean_cuts([0] + chosen, len(y), int(SR * min_ms / 1000))


def best_grid_offset(y, target_cuts, search_ms=250):
    slice_len = len(y) / target_cuts
    env = librosa.onset.onset_strength(y=y, sr=SR)
    env_samples = librosa.frames_to_samples(np.arange(len(env)))

    best = None
    step = int(SR * 5 / 1000)
    max_offset = int(SR * search_ms / 1000)

    for offset in range(0, max_offset + 1, step):
        score = 0.0
        for i in range(target_cuts):
            c = int(offset + i * slice_len)
            win = int(SR * 0.035)
            idx = np.where((env_samples >= c - win) & (env_samples <= c + win))[0]
            if len(idx):
                score += float(np.max(env[idx]))

        if best is None or score > best[0]:
            best = (score, offset)

    return best[1] if best else 0


def hybrid_cuts(y, target_cuts, min_ms):
    offset = best_grid_offset(y, target_cuts)
    usable = max(1, len(y) - offset)
    slice_len = usable / target_cuts

    onsets = librosa.onset.onset_detect(
        y=y,
        sr=SR,
        units="samples",
        backtrack=True,
        delta=0.06,
        wait=1,
    )

    cuts = []
    snap_window = int(SR * 0.045)

    for i in range(target_cuts):
        grid = int(offset + i * slice_len)
        nearby = [o for o in onsets if abs(o - grid) <= snap_window]
        cut = min(nearby, key=lambda o: abs(o - grid)) if nearby else grid
        cuts.append(cut)

    return clean_cuts([0] + cuts, len(y), int(SR * min_ms / 1000))


def get_cuts(y, method, target_cuts, min_ms):
    if method == "onset_strong":
        return onset_strong_cuts(y, target_cuts, min_ms)
    return hybrid_cuts(y, target_cuts, min_ms)


def render_sequence(slices, seq):
    chunks = [slices[i % len(slices)]["audio"] for i in seq]
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return normalize(np.concatenate(chunks))


def base_sequence(slices):
    return list(range(len(slices)))


def split_pairs(seq):
    return [seq[i:i+2] for i in range(0, len(seq), 2)]


def flatten(pairs):
    return [x for p in pairs for x in p]


def mutate_safe_pairs(base, mutation=0.12, fill_chance=0.12):
    """
    Mutations très sobres :
    - la plupart des blocs restent originaux
    - parfois un bloc répète le précédent/suivant
    - parfois swap interne d'un bloc
    - fill final léger
    """
    pairs = split_pairs(base)

    for i in range(len(pairs)):
        if random.random() > mutation:
            continue

        action = random.choice(["repeat_prev", "repeat_next", "swap_inside", "keep", "keep"])

        if action == "repeat_prev" and i > 0:
            pairs[i] = pairs[i - 1][:]

        elif action == "repeat_next" and i < len(pairs) - 1:
            pairs[i] = pairs[i + 1][:]

        elif action == "swap_inside" and len(pairs[i]) == 2:
            pairs[i] = [pairs[i][1], pairs[i][0]]

    # fill final très léger uniquement sur le dernier bloc
    if len(pairs) >= 2 and random.random() < fill_chance:
        action = random.choice(["repeat_prev", "swap_last"])
        if action == "repeat_prev":
            pairs[-1] = pairs[-2][:]
        elif action == "swap_last" and len(pairs[-1]) == 2:
            pairs[-1] = [pairs[-1][1], pairs[-1][0]]

    return flatten(pairs)[:len(base)]


def export_preview(slices, outdir):
    p = outdir / "slices_preview"
    p.mkdir(parents=True, exist_ok=True)
    for s in slices:
        wav = p / f"slice_{s['index']:03d}_{int(s['start_ms'])}ms.wav"
        sf.write(wav, normalize(s["audio"]), SR)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Amen")
    parser.add_argument("--method", choices=["hybrid", "onset_strong"], default="hybrid")
    parser.add_argument("--target-cuts", type=int, default=16)
    parser.add_argument("--min-ms", type=float, default=45.0)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--mutation", type=float, default=0.12)
    parser.add_argument("--fill-chance", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    source = find_source(args.source)
    print("Source :", source)

    y, sr = load_audio(source)
    cuts = get_cuts(y, args.method, args.target_cuts, args.min_ms)
    slices = cuts_to_slices(y, cuts)

    if len(slices) < 4:
        print("Pas assez de slices.")
        sys.exit(1)

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe / f"{args.method}_{args.target_cuts}cuts_mut{args.mutation}"
    outdir.mkdir(parents=True, exist_ok=True)

    preview_dir = export_preview(slices, outdir)

    base = base_sequence(slices)
    base_audio = render_sequence(slices, base)
    base_wav = outdir / f"{safe}_safe_base.wav"
    sf.write(base_wav, base_audio, SR)

    print(f"Slices : {len(slices)}")
    print("Base :", base_wav)

    renders = []

    for i in range(1, args.count + 1):
        seq = mutate_safe_pairs(
            base,
            mutation=args.mutation,
            fill_chance=args.fill_chance,
        )

        audio = render_sequence(slices, seq)

        wav = outdir / f"{safe}_safe_variation_{i:03d}.wav"
        sf.write(wav, audio, SR)

        renders.append({
            "index": i,
            "wav": str(wav),
            "sequence": [int(x) for x in seq],
        })

        print("Export :", wav)

    metadata = {
        "version": "jungle_grammar_v02_safe",
        "source": str(source),
        "source_only": True,
        "no_overlap": True,
        "no_txt_variations": True,
        "method": args.method,
        "target_cuts": args.target_cuts,
        "min_ms": args.min_ms,
        "mutation": args.mutation,
        "fill_chance": args.fill_chance,
        "base_wav": str(base_wav),
        "preview_dir": str(preview_dir),
        "slices": [
            {
                "index": s["index"],
                "start_ms": s["start_ms"],
                "end_ms": s["end_ms"],
                "duration": s["duration"],
            }
            for s in slices
        ],
        "renders": renders,
    }

    (outdir / "metadata_jungle_grammar_v02_safe.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)


if __name__ == "__main__":
    main()
