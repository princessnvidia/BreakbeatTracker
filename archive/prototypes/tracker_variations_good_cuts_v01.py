#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracker Variations From Good Cuts v01

But :
Utiliser la bonne découpe trouvée par find_good_cuts_v01.py,
puis générer des variations tracker.

Principe :
- 1 seul break source
- on reprend une méthode de cut : hybrid / onset_strong / grid_snap / onset_backtrack
- on découpe une fois
- les variations réutilisent ces mêmes slices
- aucune superposition
- pas de samples externes

Usage :
    python tracker_variations_good_cuts_v01.py --source "Camo" --method hybrid

Autres méthodes :
    python tracker_variations_good_cuts_v01.py --source "Camo" --method onset_strong
    python tracker_variations_good_cuts_v01.py --source "Camo" --method grid_snap
    python tracker_variations_good_cuts_v01.py --source "Camo" --method onset_backtrack

Si tu avais préféré target-cuts 8 :
    python tracker_variations_good_cuts_v01.py --source "Camo" --target-cuts 8 --method hybrid

Plus ou moins de mutation :
    python tracker_variations_good_cuts_v01.py --source "Camo" --mutation 0.18
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
OUT_DIR = Path("exports/tracker_variations_good_cuts_v01")
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


def onset_backtrack_cuts(y, min_ms):
    onset_samples = librosa.onset.onset_detect(
        y=y,
        sr=SR,
        units="samples",
        backtrack=True,
        delta=0.07,
        wait=1,
        pre_max=3,
        post_max=3,
        pre_avg=8,
        post_avg=8,
    )
    return clean_cuts([0] + list(onset_samples), len(y), int(SR * min_ms / 1000))


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


def grid_snap_cuts(y, target_cuts):
    offset = best_grid_offset(y, target_cuts)
    usable = max(1, len(y) - offset)
    slice_len = usable / target_cuts
    cuts = [int(offset + i * slice_len) for i in range(target_cuts)]
    return clean_cuts([0] + cuts, len(y), 1)


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
        if nearby:
            cut = min(nearby, key=lambda o: abs(o - grid))
        else:
            cut = grid
        cuts.append(cut)

    return clean_cuts([0] + cuts, len(y), int(SR * min_ms / 1000))


def get_cuts(y, method, target_cuts, min_ms):
    if method == "onset_backtrack":
        return onset_backtrack_cuts(y, min_ms)
    if method == "onset_strong":
        return onset_strong_cuts(y, target_cuts, min_ms)
    if method == "grid_snap":
        return grid_snap_cuts(y, target_cuts)
    if method == "hybrid":
        return hybrid_cuts(y, target_cuts, min_ms)
    raise ValueError(method)


def render_sequence(slices, seq):
    chunks = [slices[i % len(slices)]["audio"] for i in seq]
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return normalize(np.concatenate(chunks))


def base_sequence(slices, steps=None):
    n = len(slices)
    if steps is None:
        return list(range(n))
    return [i % n for i in range(steps)]


def mutate_sequence(base, mutation=0.18, fill_chance=0.25):
    seq = base[:]
    n = len(seq)

    for i in range(n):
        if random.random() > mutation:
            continue

        action = random.choice([
            "repeat_prev",
            "repeat_next",
            "swap_neighbor",
            "local_jump",
            "keep",
        ])

        if action == "repeat_prev" and i > 0:
            seq[i] = seq[i - 1]

        elif action == "repeat_next" and i < n - 1:
            seq[i] = seq[i + 1]

        elif action == "swap_neighbor":
            j = max(0, min(n - 1, i + random.choice([-1, 1])))
            seq[i], seq[j] = seq[j], seq[i]

        elif action == "local_jump":
            seq[i] = max(0, min(max(base), seq[i] + random.choice([-2, -1, 1, 2])))

    # fill de fin de boucle
    if n >= 8 and random.random() < fill_chance:
        start = n - random.choice([4, 6, 8])
        start = max(0, start)
        mode = random.choice(["repeat", "chop", "reverse"])

        if mode == "repeat":
            src = seq[start]
            for i in range(start, n):
                if random.random() < 0.70:
                    seq[i] = src

        elif mode == "chop":
            pool = seq[max(0, start - 4):start + 1]
            for i in range(start, n):
                seq[i] = random.choice(pool)

        elif mode == "reverse":
            seq[start:n] = list(reversed(seq[start:n]))

    return seq


def mutate_pairs(base, mutation=0.18, fill_chance=0.25):
    """
    Même idée, mais par blocs de 2 slices.
    """
    pairs = [base[i:i+2] for i in range(0, len(base), 2)]

    for i in range(len(pairs)):
        if random.random() > mutation:
            continue

        action = random.choice(["repeat_prev", "repeat_next", "swap_inside", "keep"])

        if action == "repeat_prev" and i > 0:
            pairs[i] = pairs[i - 1][:]

        elif action == "repeat_next" and i < len(pairs) - 1:
            pairs[i] = pairs[i + 1][:]

        elif action == "swap_inside" and len(pairs[i]) == 2:
            pairs[i] = [pairs[i][1], pairs[i][0]]

    if len(pairs) >= 4 and random.random() < fill_chance:
        mode = random.choice(["repeat_last", "reverse_last"])
        if mode == "repeat_last":
            pairs[-1] = pairs[-2][:]
        elif mode == "reverse_last":
            pairs[-2:] = list(reversed(pairs[-2:]))

    return [x for pair in pairs for x in pair]


def pattern_text(seq, slices):
    lines = []
    lines.append("STEP | SLICE | START_MS | END_MS | DUR")
    lines.append("---------------------------------------")
    for step, idx in enumerate(seq, start=1):
        s = slices[idx % len(slices)]
        lines.append(
            f"{step:04d} | {idx:05d} | {s['start_ms']:8.1f} | "
            f"{s['end_ms']:8.1f} | {s['duration']:.3f}s"
        )
    lines.append("")
    lines.append("SEQUENCE:")
    lines.append(" ".join(f"{i:02d}" for i in seq))
    return "\n".join(lines)


def export_slices_preview(slices, outdir):
    p = outdir / "slices_preview"
    p.mkdir(parents=True, exist_ok=True)

    lines = ["IDX | START_MS | END_MS | DUR | FILE", "-----------------------------------"]

    for s in slices:
        wav = p / f"slice_{s['index']:03d}_{int(s['start_ms'])}ms.wav"
        sf.write(wav, normalize(s["audio"]), SR)
        lines.append(
            f"{s['index']:03d} | {s['start_ms']:8.1f} | {s['end_ms']:8.1f} | "
            f"{s['duration']:.3f}s | {wav}"
        )

    (p / "slices_preview.txt").write_text("\n".join(lines), encoding="utf-8")
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Camo")
    parser.add_argument("--method", choices=["hybrid", "onset_strong", "grid_snap", "onset_backtrack"], default="hybrid")
    parser.add_argument("--target-cuts", type=int, default=16)
    parser.add_argument("--min-ms", type=float, default=45.0)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--mutation", type=float, default=0.18)
    parser.add_argument("--fill-chance", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--pair-mode", action="store_true", help="Mute les variations deux par deux.")
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

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe / f"{args.method}_{args.target_cuts}cuts"
    if args.pair_mode:
        outdir = OUT_DIR / safe / f"{args.method}_{args.target_cuts}cuts_pairmode"
    outdir.mkdir(parents=True, exist_ok=True)

    preview_dir = export_slices_preview(slices, outdir)

    base = base_sequence(slices, steps=args.steps)
    base_audio = render_sequence(slices, base)
    base_wav = outdir / f"{safe}_{args.method}_base.wav"
    sf.write(base_wav, base_audio, SR)
    (outdir / "base_pattern.txt").write_text(pattern_text(base, slices), encoding="utf-8")

    print(f"Slices : {len(slices)}")
    print("Base :", base_wav)

    renders = []

    for i in range(1, args.count + 1):
        if args.pair_mode:
            seq = mutate_pairs(base, mutation=args.mutation, fill_chance=args.fill_chance)
        else:
            seq = mutate_sequence(base, mutation=args.mutation, fill_chance=args.fill_chance)

        audio = render_sequence(slices, seq)

        wav = outdir / f"{safe}_{args.method}_variation_{i:03d}.wav"
        txt = outdir / f"{safe}_{args.method}_variation_{i:03d}.txt"

        sf.write(wav, audio, SR)
        txt.write_text(pattern_text(seq, slices), encoding="utf-8")

        renders.append({
            "index": i,
            "wav": str(wav),
            "sequence": [int(x) for x in seq],
        })

        print("Export :", wav)

    metadata = {
        "version": "tracker_variations_good_cuts_v01",
        "source": str(source),
        "source_only": True,
        "no_overlap": True,
        "method": args.method,
        "target_cuts": args.target_cuts,
        "min_ms": args.min_ms,
        "pair_mode": bool(args.pair_mode),
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

    (outdir / "metadata_variations_good_cuts.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("Preview :", preview_dir)


if __name__ == "__main__":
    main()
