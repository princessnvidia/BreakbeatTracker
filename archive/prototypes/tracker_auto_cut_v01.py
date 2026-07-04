#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracker Auto Cut v01

But :
Découper automatiquement un break comme un tracker :
- 1 seul break source
- 8 découpes de 2 cases
- grille de 16 cases
- le script cherche automatiquement :
    * le meilleur offset de départ
    * la meilleure durée de slice
- pas besoin d'entrer les cuts à la main
- pas de superposition
- pas de samples externes

Usage :
    python tracker_auto_cut_v01.py --source "Camo" --image camobreak.png

Plus de variations :
    python tracker_auto_cut_v01.py --source "Camo" --image camobreak.png --count 64

Si besoin :
    python tracker_auto_cut_v01.py --source "Camo" --image camobreak.png --min-slice-ms 140 --max-slice-ms 320
"""

from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import soundfile as sf
from PIL import Image

try:
    import librosa
except ImportError:
    print("librosa manquant : pip install librosa soundfile numpy pillow")
    sys.exit(1)


BREAKS_DIR = Path("breaks")
OUT_DIR = Path("exports/tracker_auto_cut_v01")
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


def pink_mask(img):
    arr = np.array(img.convert("RGB"))
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    return (
        (r > 170)
        & (g > 65)
        & (g < 190)
        & (b > 85)
        & (b < 220)
        & (r > g + 30)
    )


def group_contiguous(indices, min_len=1):
    if len(indices) == 0:
        return []
    groups = []
    start = indices[0]
    prev = indices[0]
    for x in indices[1:]:
        if x == prev + 1:
            prev = x
        else:
            if prev - start + 1 >= min_len:
                groups.append((start, prev))
            start = x
            prev = x
    if prev - start + 1 >= min_len:
        groups.append((start, prev))
    return groups


def detect_lanes(mask):
    row_strength = mask.sum(axis=1)
    ys = np.where(row_strength > 5)[0]
    bands = group_contiguous(list(ys), min_len=3)
    scored = []
    for y1, y2 in bands:
        scored.append((int(row_strength[y1:y2 + 1].sum()), y1, y2))
    scored.sort(reverse=True)
    selected = sorted([(y1, y2) for _, y1, y2 in scored[:3]], key=lambda x: x[0])
    if len(selected) < 3:
        print("Impossible de détecter 3 lignes dans l'image.")
        sys.exit(1)
    return {"hat": selected[0], "kick": selected[1], "snare": selected[2]}


def detect_events_from_image(image_path):
    img = Image.open(Path(image_path).expanduser())
    mask = pink_mask(img)
    lanes = detect_lanes(mask)

    events = []
    for role, (y1, y2) in lanes.items():
        lane_mask = mask[y1:y2 + 1, :]
        col_strength = lane_mask.sum(axis=0)
        xs = np.where(col_strength > 2)[0]
        intervals = group_contiguous(list(xs), min_len=3)

        for x1, x2 in intervals:
            events.append({
                "role": role,
                "x1": int(x1),
                "x2": int(x2),
                "xc": float((x1 + x2) / 2),
            })

    events.sort(key=lambda e: e["xc"])
    if not events:
        print("Aucun event détecté dans l'image.")
        sys.exit(1)

    x_min = min(e["x1"] for e in events)
    x_max = max(e["x2"] for e in events)
    span = max(1, x_max - x_min)

    for i, e in enumerate(events):
        pos = (e["xc"] - x_min) / span
        grid16 = int(round(pos * 15))
        grid16 = max(0, min(15, grid16))
        e["index"] = i
        e["grid16"] = grid16
        e["pair"] = grid16 // 2

    return events


def onset_strength_curve(y):
    env = librosa.onset.onset_strength(y=y, sr=SR)
    times = librosa.frames_to_samples(np.arange(len(env)))
    return env, times


def score_cut_grid(y, offset_samples, slice_samples, env, env_samples):
    """
    Score un découpage 8 slices.
    Un bon cut doit tomber proche d'un onset / transient.
    """
    score = 0.0
    cuts = [offset_samples + i * slice_samples for i in range(8)]

    for c in cuts:
        if c < 0 or c >= len(y):
            score -= 5.0
            continue

        # Cherche l'énergie onset proche du cut.
        window = int(SR * 0.035)
        lo = c - window
        hi = c + window

        idx = np.where((env_samples >= lo) & (env_samples <= hi))[0]
        if len(idx):
            score += float(np.max(env[idx]))

        # Bonus si le signal démarre fort au cut.
        local = y[c:min(len(y), c + int(SR * 0.030))]
        if len(local):
            score += float(np.sqrt(np.mean(local * local))) * 8.0

    # pénalité si la grille dépasse trop
    end = offset_samples + 8 * slice_samples
    if end > len(y):
        score -= (end - len(y)) / SR * 10.0

    return score


def auto_find_cuts(y, min_slice_ms=140, max_slice_ms=320, offset_search_ms=160, step_ms=5):
    env, env_samples = onset_strength_curve(y)

    best = None

    min_slice = int(SR * min_slice_ms / 1000)
    max_slice = int(SR * max_slice_ms / 1000)
    step = int(SR * step_ms / 1000)
    max_offset = int(SR * offset_search_ms / 1000)

    for slice_samples in range(min_slice, max_slice + 1, step):
        for offset_samples in range(0, max_offset + 1, step):
            score = score_cut_grid(y, offset_samples, slice_samples, env, env_samples)

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "offset_samples": offset_samples,
                    "slice_samples": slice_samples,
                    "offset_ms": offset_samples / SR * 1000,
                    "slice_ms": slice_samples / SR * 1000,
                }

    return best


def make_pair_slices(y, offset_samples, slice_samples):
    slices = []
    for pair in range(8):
        start = offset_samples + pair * slice_samples
        end = start + slice_samples

        if start >= len(y):
            chunk = np.zeros(slice_samples, dtype=np.float32)
        else:
            chunk = y[start:min(end, len(y))].copy()
            if len(chunk) < slice_samples:
                chunk = np.concatenate([chunk, np.zeros(slice_samples - len(chunk), dtype=np.float32)])

        chunk = fade(chunk, ms=2)

        slices.append({
            "pair": pair,
            "start_sample": int(start),
            "end_sample": int(end),
            "start_ms": start / SR * 1000,
            "end_ms": end / SR * 1000,
            "audio": chunk,
        })
    return slices


def build_base_sequence(events):
    return [
        {
            "role": e["role"],
            "grid16": int(e["grid16"]),
            "pair": int(e["pair"]),
            "slice_pair": int(e["pair"]),
        }
        for e in events
    ]


def mutate_sequence(base_seq, mutation=0.15):
    seq = [dict(e) for e in base_seq]
    by_role = {"hat": [], "kick": [], "snare": []}

    for e in base_seq:
        by_role[e["role"]].append(e)

    for e in seq:
        if random.random() > mutation:
            continue

        action = random.choice(["same_role_pair", "neighbor_pair", "keep"])

        if action == "same_role_pair":
            candidates = by_role.get(e["role"], [])
            if candidates:
                repl = random.choice(candidates)
                e["slice_pair"] = int(repl["pair"])

        elif action == "neighbor_pair":
            e["slice_pair"] = max(0, min(7, int(e["pair"]) + random.choice([-1, 1])))

    # fill final uniquement sur dernière paire
    if random.random() < mutation:
        for e in seq:
            if e["pair"] == 7 and random.random() < 0.55:
                e["slice_pair"] = random.choice([6, 7])

    return seq


def render_sequence(pair_slices, seq):
    chunks = []
    for e in seq:
        chunks.append(pair_slices[int(e["slice_pair"])]["audio"])
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return normalize(np.concatenate(chunks))


def role_letter(role):
    return {"hat": "H", "kick": "K", "snare": "S"}.get(role, "?")


def sequence_text(seq):
    rows = {"hat": ["."] * 16, "kick": ["."] * 16, "snare": ["."] * 16}

    lines = []
    lines.append("EVENT | ROLE  | GRID16 | PAIR | SLICE_PAIR")
    lines.append("-------------------------------------------")

    for i, e in enumerate(seq):
        rows[e["role"]][int(e["grid16"])] = role_letter(e["role"])
        lines.append(
            f"{i:05d} | {e['role']:5} | {int(e['grid16']):06d} | "
            f"{int(e['pair']):04d} | {int(e['slice_pair']):010d}"
        )

    lines.append("")
    lines.append("GRID16:")
    lines.append("PAIR : 11|22|33|44|55|66|77|88")
    lines.append("STEP : 12|34|56|78|90|12|34|56")
    lines.append("HAT  : " + "|".join("".join(rows["hat"][i:i+2]) for i in range(0, 16, 2)))
    lines.append("KICK : " + "|".join("".join(rows["kick"][i:i+2]) for i in range(0, 16, 2)))
    lines.append("SNARE: " + "|".join("".join(rows["snare"][i:i+2]) for i in range(0, 16, 2)))
    lines.append("")
    lines.append("SLICE PAIRS:")
    lines.append(" ".join(str(e["slice_pair"]) for e in seq))

    return "\n".join(lines)


def export_preview(pair_slices, outdir):
    p = outdir / "preview_auto_cuts"
    p.mkdir(parents=True, exist_ok=True)

    lines = ["PAIR | START_MS | END_MS | FILE", "-----------------------------"]

    for s in pair_slices:
        wav = p / f"pair_{s['pair']:02d}_{int(s['start_ms'])}ms.wav"
        sf.write(wav, normalize(s["audio"]), SR)
        lines.append(f"{s['pair']:04d} | {s['start_ms']:8.1f} | {s['end_ms']:8.1f} | {wav}")

    (p / "preview_auto_cuts.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Camo")
    parser.add_argument("--image", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--mutation", type=float, default=0.15)
    parser.add_argument("--min-slice-ms", type=float, default=140)
    parser.add_argument("--max-slice-ms", type=float, default=320)
    parser.add_argument("--offset-search-ms", type=float, default=180)
    parser.add_argument("--step-ms", type=float, default=5)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    source = find_source(args.source)
    print("Source unique :", source)

    y, sr = load_audio(source)
    events = detect_events_from_image(args.image)

    best = auto_find_cuts(
        y,
        min_slice_ms=args.min_slice_ms,
        max_slice_ms=args.max_slice_ms,
        offset_search_ms=args.offset_search_ms,
        step_ms=args.step_ms,
    )

    print("")
    print("Auto-cut trouvé :")
    print(f"  offset : {best['offset_ms']:.1f} ms")
    print(f"  slice  : {best['slice_ms']:.1f} ms")
    print(f"  score  : {best['score']:.3f}")

    pair_slices = make_pair_slices(y, best["offset_samples"], best["slice_samples"])

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe / f"auto_offset_{int(best['offset_ms'])}ms_slice_{int(best['slice_ms'])}ms"
    outdir.mkdir(parents=True, exist_ok=True)

    export_preview(pair_slices, outdir)

    if args.preview:
        print("Preview :", outdir / "preview_auto_cuts")
        return

    base_seq = build_base_sequence(events)
    base_audio = render_sequence(pair_slices, base_seq)
    base_wav = outdir / f"{safe}_auto_base.wav"
    sf.write(base_wav, base_audio, SR)

    (outdir / "base_auto_pattern.txt").write_text(sequence_text(base_seq), encoding="utf-8")

    print("Base :", base_wav)

    renders = []

    for i in range(1, args.count + 1):
        seq = mutate_sequence(base_seq, mutation=args.mutation)
        audio = render_sequence(pair_slices, seq)

        wav = outdir / f"{safe}_auto_variation_{i:03d}.wav"
        txt = outdir / f"{safe}_auto_variation_{i:03d}.txt"

        sf.write(wav, audio, SR)
        txt.write_text(sequence_text(seq), encoding="utf-8")

        renders.append({
            "index": i,
            "wav": str(wav),
            "sequence": [
                {
                    "role": e["role"],
                    "grid16": int(e["grid16"]),
                    "pair": int(e["pair"]),
                    "slice_pair": int(e["slice_pair"]),
                }
                for e in seq
            ],
        })

        print("Export :", wav)

    metadata = {
        "version": "tracker_auto_cut_v01",
        "source": str(source),
        "image": str(args.image),
        "source_only": True,
        "no_overlap": True,
        "rule": "auto-find offset and 8 pair-slices of 2 grid cases",
        "auto_cut": {
            "offset_ms": float(best["offset_ms"]),
            "slice_ms": float(best["slice_ms"]),
            "score": float(best["score"]),
        },
        "event_count": int(len(events)),
        "base_wav": str(base_wav),
        "pair_slices": [
            {
                "pair": int(s["pair"]),
                "start_ms": float(s["start_ms"]),
                "end_ms": float(s["end_ms"]),
            }
            for s in pair_slices
        ],
        "renders": renders,
    }

    (outdir / "metadata_auto_cut.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("Preview cuts :", outdir / "preview_auto_cuts")


if __name__ == "__main__":
    main()
