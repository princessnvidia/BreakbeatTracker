#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jungle Grammar v01

Phase 2 de BreakbeatAI :
- découpe un break avec la bonne logique de cuts
- apprend une petite grammaire jungle symbolique
- génère des variations par motifs :
    repeat
    chop
    stutter
    reverse fill
    amen switch
    two-by-two mutation
- ne génère plus de .txt par variation
- exporte uniquement des WAV + un metadata JSON global

Usage :
    python jungle_grammar_v01.py --source "Amen" --count 64
    python jungle_grammar_v01.py --source "Camo" --count 64
    python jungle_grammar_v01.py --source "London" --method onset_strong --count 64

Modes :
    --grammar balanced
    --grammar busy
    --grammar minimal
    --grammar fills
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
OUT_DIR = Path("exports/jungle_grammar_v01")
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


def chunks(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def flatten(blocks):
    return [x for block in blocks for x in block]


def op_repeat_prev(blocks):
    if len(blocks) < 2:
        return blocks
    i = random.randrange(1, len(blocks))
    blocks[i] = blocks[i - 1][:]
    return blocks


def op_repeat_next(blocks):
    if len(blocks) < 2:
        return blocks
    i = random.randrange(0, len(blocks) - 1)
    blocks[i] = blocks[i + 1][:]
    return blocks


def op_swap_neighbors(blocks):
    if len(blocks) < 2:
        return blocks
    i = random.randrange(0, len(blocks) - 1)
    blocks[i], blocks[i + 1] = blocks[i + 1], blocks[i]
    return blocks


def op_reverse_fill(blocks):
    if len(blocks) < 2:
        return blocks
    n = random.choice([2, 3, 4])
    n = min(n, len(blocks))
    blocks[-n:] = list(reversed(blocks[-n:]))
    return blocks


def op_stutter(blocks):
    if not blocks:
        return blocks
    i = random.randrange(len(blocks))
    block = blocks[i]
    if not block:
        return blocks
    src = random.choice(block)
    blocks[i] = [src for _ in block]
    return blocks


def op_chop_fill(blocks):
    if not blocks:
        return blocks
    start = max(0, len(blocks) - random.choice([2, 3, 4]))
    pool = flatten(blocks[max(0, start - 2):start + 1])
    if not pool:
        return blocks
    for i in range(start, len(blocks)):
        blocks[i] = [random.choice(pool) for _ in blocks[i]]
    return blocks


def op_amen_switch(blocks):
    """
    Motif jungle classique : petites permutations locales,
    mais on garde la longueur et la structure.
    """
    if len(blocks) < 4:
        return blocks
    i = random.randrange(0, len(blocks) - 3)
    pattern = random.choice([
        [0, 1, 1, 3],
        [0, 2, 1, 3],
        [0, 1, 3, 2],
        [1, 0, 2, 3],
    ])
    src = [b[:] for b in blocks[i:i + 4]]
    for j, p in enumerate(pattern):
        blocks[i + j] = src[p][:]
    return blocks


def grammar_weights(grammar):
    if grammar == "minimal":
        return [
            ("repeat_prev", 2),
            ("repeat_next", 1),
            ("swap_neighbors", 1),
        ]
    if grammar == "busy":
        return [
            ("repeat_prev", 2),
            ("repeat_next", 2),
            ("swap_neighbors", 2),
            ("reverse_fill", 2),
            ("stutter", 3),
            ("chop_fill", 3),
            ("amen_switch", 2),
        ]
    if grammar == "fills":
        return [
            ("reverse_fill", 3),
            ("stutter", 3),
            ("chop_fill", 4),
            ("amen_switch", 2),
        ]
    return [
        ("repeat_prev", 2),
        ("repeat_next", 1),
        ("swap_neighbors", 2),
        ("reverse_fill", 2),
        ("stutter", 2),
        ("chop_fill", 2),
        ("amen_switch", 2),
    ]


def choose_operation(grammar):
    weighted = grammar_weights(grammar)
    total = sum(w for _, w in weighted)
    r = random.random() * total
    acc = 0
    for name, weight in weighted:
        acc += weight
        if r <= acc:
            return name
    return weighted[-1][0]


def apply_operation(name, blocks):
    blocks = [b[:] for b in blocks]
    if name == "repeat_prev":
        return op_repeat_prev(blocks)
    if name == "repeat_next":
        return op_repeat_next(blocks)
    if name == "swap_neighbors":
        return op_swap_neighbors(blocks)
    if name == "reverse_fill":
        return op_reverse_fill(blocks)
    if name == "stutter":
        return op_stutter(blocks)
    if name == "chop_fill":
        return op_chop_fill(blocks)
    if name == "amen_switch":
        return op_amen_switch(blocks)
    return blocks


def generate_jungle_sequence(base, grammar="balanced", block_size=2, operations=3):
    blocks = chunks(base, block_size)
    ops_used = []

    for _ in range(operations):
        op = choose_operation(grammar)
        blocks = apply_operation(op, blocks)
        ops_used.append(op)

    seq = flatten(blocks)

    # sécurité : même longueur que base
    if len(seq) < len(base):
        seq += base[len(seq):]
    seq = seq[:len(base)]

    return seq, ops_used


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
    parser.add_argument("--method", choices=["hybrid", "onset_strong", "grid_snap", "onset_backtrack"], default="hybrid")
    parser.add_argument("--target-cuts", type=int, default=16)
    parser.add_argument("--min-ms", type=float, default=45.0)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--grammar", choices=["balanced", "busy", "minimal", "fills"], default="balanced")
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--operations", type=int, default=3)
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
    outdir = OUT_DIR / safe / f"{args.method}_{args.target_cuts}cuts_{args.grammar}"
    outdir.mkdir(parents=True, exist_ok=True)

    preview_dir = export_preview(slices, outdir)

    base = base_sequence(slices, steps=args.steps)
    base_audio = render_sequence(slices, base)
    base_wav = outdir / f"{safe}_jungle_grammar_base.wav"
    sf.write(base_wav, base_audio, SR)

    print(f"Slices : {len(slices)}")
    print("Base :", base_wav)

    renders = []

    for i in range(1, args.count + 1):
        seq, ops_used = generate_jungle_sequence(
            base,
            grammar=args.grammar,
            block_size=args.block_size,
            operations=args.operations,
        )

        audio = render_sequence(slices, seq)

        wav = outdir / f"{safe}_jungle_grammar_{i:03d}.wav"
        sf.write(wav, audio, SR)

        renders.append({
            "index": i,
            "wav": str(wav),
            "sequence": [int(x) for x in seq],
            "operations": ops_used,
        })

        print("Export :", wav)

    metadata = {
        "version": "jungle_grammar_v01",
        "phase": 2,
        "source": str(source),
        "source_only": True,
        "no_overlap": True,
        "no_txt_variations": True,
        "method": args.method,
        "target_cuts": args.target_cuts,
        "min_ms": args.min_ms,
        "grammar": args.grammar,
        "block_size": args.block_size,
        "operations": args.operations,
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

    (outdir / "metadata_jungle_grammar_v01.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("Preview :", preview_dir)


if __name__ == "__main__":
    main()
