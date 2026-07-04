#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracker From Image v03 - pair slices fixed

Correction v03 :
- chaque découpe audio fait TOUJOURS 2 cases sur une grille de 16
- donc il y a 8 grosses découpes/paires :
      pair 0 = cases 01-02
      pair 1 = cases 03-04
      ...
      pair 7 = cases 15-16
- l'image sert à lire la partition tracker :
      ligne haut   = hat
      ligne milieu = kick
      ligne bas    = snare
- MAIS l'audio est découpé par paires fixes, pas par rectangle
- aucune superposition
- une seule source audio

Usage :
    python tracker_from_image_v03.py --source "Camo" --image camobreak.png

Options :
    python tracker_from_image_v03.py --source "Camo" --image camobreak.png --count 32 --mutation 0.15
    python tracker_from_image_v03.py --source "Camo" --image camobreak.png --offset-ms 20
    python tracker_from_image_v03.py --source "Camo" --image camobreak.png --slice-ms 220

Important :
Si les cuts sont décalés, ajuste --offset-ms.
Si les slices sont trop longues/courtes, ajuste --slice-ms.
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
OUT_DIR = Path("exports/tracker_from_image_v03")

SR = 44100
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}


def expand_user(path):
    return Path(str(path)).expanduser()


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

    if len(ys) == 0:
        print("Aucun bloc rose détecté.")
        sys.exit(1)

    bands = group_contiguous(list(ys), min_len=3)

    scored = []
    for y1, y2 in bands:
        scored.append((int(row_strength[y1:y2 + 1].sum()), y1, y2))

    scored.sort(reverse=True)
    selected = sorted([(y1, y2) for _, y1, y2 in scored[:3]], key=lambda x: x[0])

    if len(selected) < 3:
        print("Je n'ai pas trouvé 3 lignes/pistes.")
        print("Bandes :", selected)
        sys.exit(1)

    return {
        "hat": selected[0],
        "kick": selected[1],
        "snare": selected[2],
    }


def detect_events(image_path):
    img = Image.open(image_path)
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
                "y1": int(y1),
                "y2": int(y2),
            })

    events.sort(key=lambda e: (e["x1"], e["xc"]))

    if not events:
        print("Aucun event détecté.")
        sys.exit(1)

    x_min = min(e["x1"] for e in events)
    x_max = max(e["x2"] for e in events)
    span = max(1, x_max - x_min)

    for i, e in enumerate(events):
        pos = (e["xc"] - x_min) / span
        grid16 = int(round(pos * 15))
        grid16 = max(0, min(15, grid16))
        pair = grid16 // 2

        e["index"] = i
        e["grid16"] = int(grid16)
        e["pair"] = int(pair)
        e["x_norm"] = float(pos)

    return events, {
        "x_min": int(x_min),
        "x_max": int(x_max),
        "lanes": lanes,
        "image_size": img.size,
    }


def make_pair_slices(y, slice_samples, offset_samples):
    """
    Crée exactement 8 slices de 2 cases.
    Chaque slice correspond à une paire de la grille 16.
    """
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
            "pair": int(pair),
            "start_sample": int(start),
            "end_sample": int(end),
            "start_ms": float(start / SR * 1000),
            "end_ms": float(end / SR * 1000),
            "duration": float(slice_samples / SR),
            "audio": chunk,
        })

    return slices


def build_base_sequence(events):
    """
    La séquence base est l'ordre des événements de gauche à droite.
    Chaque event pointe vers la slice de sa paire.
    Donc toutes les notes de la paire 0 jouent la slice pair 0, etc.
    """
    seq = []
    for e in events:
        seq.append({
            "role": e["role"],
            "grid16": int(e["grid16"]),
            "pair": int(e["pair"]),
            "slice_pair": int(e["pair"]),
        })
    return seq


def mutate_sequence_by_pair(base_seq, mutation=0.15):
    """
    Mutations conservatrices :
    - chaque event reste sur une paire de 2 cases
    - pas de découpe hors pair
    - on peut remplacer un event par un autre event du même rôle
      MAIS toujours sur une paire entière
    """
    seq = [dict(e) for e in base_seq]

    by_role = {"hat": [], "kick": [], "snare": []}
    for e in base_seq:
        by_role[e["role"]].append(e)

    # Mutations très contrôlées
    for i, e in enumerate(seq):
        if random.random() > mutation:
            continue

        action = random.choice(["same_role_pair", "neighbor_pair", "keep"])

        if action == "same_role_pair":
            candidates = by_role.get(e["role"], [])
            if candidates:
                repl = random.choice(candidates)
                # On garde la position grid16 de l'event, mais on emprunte la slice d'une autre paire.
                seq[i]["slice_pair"] = int(repl["pair"])

        elif action == "neighbor_pair":
            delta = random.choice([-1, 1])
            seq[i]["slice_pair"] = max(0, min(7, e["pair"] + delta))

    # Fill final par bloc de 2 cases seulement : dernière paire
    if random.random() < mutation:
        last_pair = 7
        for e in seq:
            if e["pair"] == last_pair and random.random() < 0.55:
                # répète la paire précédente ou garde dernière
                e["slice_pair"] = random.choice([6, 7])

    return seq


def render_sequence(pair_slices, seq):
    """
    Timeline tracker : une succession de notes.
    Pas de superposition.
    Chaque événement joue la slice entière de sa paire.
    """
    chunks = []
    for e in seq:
        pair = int(e["slice_pair"])
        chunks.append(pair_slices[pair]["audio"])
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return normalize(np.concatenate(chunks))


def render_pair_preview(pair_slices, outdir):
    preview = outdir / "preview_pair_slices"
    preview.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("PAIR | START_MS | END_MS | FILE")
    lines.append("--------------------------------")

    for s in pair_slices:
        wav = preview / f"pair_{s['pair']:02d}_{int(s['start_ms'])}ms.wav"
        sf.write(wav, normalize(s["audio"]), SR)
        lines.append(f"{s['pair']:04d} | {s['start_ms']:8.1f} | {s['end_ms']:8.1f} | {wav}")

    (preview / "preview_pair_slices.txt").write_text("\n".join(lines), encoding="utf-8")


def role_letter(role):
    return {"hat": "H", "kick": "K", "snare": "S"}.get(role, "?")


def sequence_text(seq):
    lines = []
    lines.append("EVENT | ROLE  | GRID16 | PAIR | SLICE_PAIR")
    lines.append("-------------------------------------------")

    lane_rows = {
        "hat": ["."] * 16,
        "kick": ["."] * 16,
        "snare": ["."] * 16,
    }

    for i, e in enumerate(seq):
        role = e["role"]
        g = int(e["grid16"])
        lane_rows[role][g] = role_letter(role)

        lines.append(
            f"{i:05d} | {role:5} | {g:06d} | {int(e['pair']):04d} | {int(e['slice_pair']):010d}"
        )

    lines.append("")
    lines.append("GRID16 / PAIRS:")
    lines.append("PAIR : 11|22|33|44|55|66|77|88")
    lines.append("STEP : 12|34|56|78|90|12|34|56")
    lines.append("HAT  : " + "|".join("".join(lane_rows["hat"][i:i+2]) for i in range(0, 16, 2)))
    lines.append("KICK : " + "|".join("".join(lane_rows["kick"][i:i+2]) for i in range(0, 16, 2)))
    lines.append("SNARE: " + "|".join("".join(lane_rows["snare"][i:i+2]) for i in range(0, 16, 2)))

    lines.append("")
    lines.append("ROLE SEQUENCE:")
    lines.append(" ".join(role_letter(e["role"]) for e in seq))

    lines.append("")
    lines.append("SLICE PAIRS:")
    lines.append(" ".join(str(e["slice_pair"]) for e in seq))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Camo")
    parser.add_argument("--image", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--mutation", type=float, default=0.15)
    parser.add_argument("--slice-ms", type=float, default=None,
                        help="Durée d'une découpe de 2 cases. Par défaut: durée_audio / 8.")
    parser.add_argument("--offset-ms", type=float, default=0.0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    source = find_source(args.source)
    image_path = expand_user(args.image)

    if not image_path.exists():
        print("Image introuvable :", image_path)
        sys.exit(1)

    print("Source unique :", source)
    print("Image tracker :", image_path)

    y, sr = load_audio(source)
    events, image_meta = detect_events(image_path)

    if args.slice_ms is None:
        slice_samples = int(len(y) / 8)
        slice_ms = slice_samples / SR * 1000
    else:
        slice_ms = args.slice_ms
        slice_samples = int(SR * slice_ms / 1000)

    offset_samples = int(SR * args.offset_ms / 1000)

    print(f"Events détectés : {len(events)}")
    print(f"Découpe fixe : 8 slices de 2 cases")
    print(f"Slice de 2 cases : {slice_ms:.1f} ms")
    print(f"Offset : {args.offset_ms:.1f} ms")

    pair_slices = make_pair_slices(y, slice_samples, offset_samples)

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe / f"offset_{int(args.offset_ms)}ms_slice_{int(slice_ms)}ms"
    outdir.mkdir(parents=True, exist_ok=True)

    render_pair_preview(pair_slices, outdir)

    if args.preview:
        print("Preview exportée.")
        print("Dossier :", outdir / "preview_pair_slices")
        return

    base_seq = build_base_sequence(events)
    base_audio = render_sequence(pair_slices, base_seq)
    base_wav = outdir / f"{safe}_base_pair_slices.wav"
    sf.write(base_wav, base_audio, SR)

    (outdir / "base_pair_slices.txt").write_text(sequence_text(base_seq), encoding="utf-8")

    print("Base :", base_wav)

    renders = []

    for i in range(1, args.count + 1):
        seq = mutate_sequence_by_pair(base_seq, mutation=args.mutation)
        audio = render_sequence(pair_slices, seq)

        wav = outdir / f"{safe}_pair_variation_{i:03d}.wav"
        txt = outdir / f"{safe}_pair_variation_{i:03d}.txt"

        sf.write(wav, audio, SR)
        txt.write_text(sequence_text(seq), encoding="utf-8")

        renders.append({
            "index": int(i),
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
        "version": "tracker_from_image_v03_pair_slices",
        "source": str(source),
        "image": str(image_path),
        "source_only": True,
        "no_overlap": True,
        "grid": "16 cases",
        "rule": "every audio slice is exactly 2 cases; 8 fixed pair slices",
        "slice_ms": float(slice_ms),
        "offset_ms": float(args.offset_ms),
        "event_count": int(len(events)),
        "base_wav": str(base_wav),
        "image_meta": {
            "x_min": int(image_meta["x_min"]),
            "x_max": int(image_meta["x_max"]),
            "lanes": {k: [int(v[0]), int(v[1])] for k, v in image_meta["lanes"].items()},
            "image_size": [int(image_meta["image_size"][0]), int(image_meta["image_size"][1])],
        },
        "pair_slices": [
            {
                "pair": int(s["pair"]),
                "start_sample": int(s["start_sample"]),
                "end_sample": int(s["end_sample"]),
                "start_ms": float(s["start_ms"]),
                "end_ms": float(s["end_ms"]),
                "duration": float(s["duration"]),
            }
            for s in pair_slices
        ],
        "events": [
            {
                "index": int(e["index"]),
                "role": e["role"],
                "grid16": int(e["grid16"]),
                "pair": int(e["pair"]),
                "x1": int(e["x1"]),
                "x2": int(e["x2"]),
            }
            for e in events
        ],
        "renders": renders,
    }

    (outdir / "metadata_pair_slices.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("À écouter d'abord :", base_wav)


if __name__ == "__main__":
    main()
