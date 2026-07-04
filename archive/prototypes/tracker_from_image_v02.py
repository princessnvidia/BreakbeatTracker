#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracker From Image v02

Correction :
- la première version reconstruisait bien la base
- mais les variations mutaient événement par événement
- cette v02 force les variations à rester calées sur une grille de 16
- mutations par BLOCS DE 2 CASES

Donc :
    16 cases = 8 blocs de 2
    bloc 01 = cases 01-02
    bloc 02 = cases 03-04
    etc.

Les variations gardent la structure :
    [bloc 1] [bloc 2] [bloc 3] [bloc 4] ...

Pas de superposition.
Une seule source.
Les slices viennent uniquement du break source.
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
OUT_DIR = Path("exports/tracker_from_image_v02")

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
        & (g > 70)
        & (g < 185)
        & (b > 90)
        & (b < 215)
        & (r > g + 35)
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
        print("Je n'ai pas trouvé 3 pistes.")
        print(selected)
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

    for i, e in enumerate(events):
        e["index"] = i
        e["x_norm"] = (e["x1"] - x_min) / max(1, x_max - x_min)

    return events, {
        "x_min": x_min,
        "x_max": x_max,
        "lanes": lanes,
        "image_size": img.size,
    }


def assign_to_16_grid(events, x_min, x_max):
    """
    Chaque événement reçoit une case 0..15.
    Plusieurs événements peuvent tomber dans la même case.
    La séquence globale reste celle de gauche à droite.
    """
    span = max(1, x_max - x_min)

    for e in events:
        pos = (e["xc"] - x_min) / span
        step = int(round(pos * 15))
        step = max(0, min(15, step))
        e["grid16"] = step
        e["pair"] = step // 2

    return events


def extract_slices(y, events, x_min, x_max, slice_end="next", min_ms=35, tail_ms=15):
    duration_samples = len(y)
    span = max(1, x_max - x_min)

    starts = []
    rect_ends = []

    for e in events:
        start = int((e["x1"] - x_min) / span * duration_samples)
        end = int((e["x2"] - x_min) / span * duration_samples)
        starts.append(max(0, min(duration_samples - 1, start)))
        rect_ends.append(max(0, min(duration_samples, end)))

    slices = []

    for i, e in enumerate(events):
        start = starts[i]

        if slice_end == "rect":
            end = rect_ends[i] + int(SR * tail_ms / 1000)
        else:
            if i < len(events) - 1:
                end = starts[i + 1]
            else:
                end = rect_ends[i] + int(SR * tail_ms / 1000)

        min_len = int(SR * min_ms / 1000)
        if end <= start + min_len:
            end = start + min_len

        end = max(start + 1, min(duration_samples, end))

        slices.append({
            "index": i,
            "role": e["role"],
            "grid16": int(e["grid16"]),
            "pair": int(e["pair"]),
            "x1": int(e["x1"]),
            "x2": int(e["x2"]),
            "start_sample": int(start),
            "end_sample": int(end),
            "duration": float((end - start) / SR),
            "audio": fade(y[start:end].copy(), ms=2),
        })

    return slices


def base_sequence(slices):
    return list(range(len(slices)))


def split_into_pairs(seq, slices):
    """
    Crée 8 blocs, chacun contient les slices dont grid16 // 2 == pair.
    L'ordre interne de chaque paire reste l'ordre d'origine.
    """
    pairs = [[] for _ in range(8)]

    for idx in seq:
        pair = slices[idx]["pair"]
        pair = max(0, min(7, pair))
        pairs[pair].append(idx)

    return pairs


def flatten_pairs(pairs):
    out = []
    for block in pairs:
        out.extend(block)
    return out


def mutate_pairs(base_pairs, slices, mutation=0.18):
    """
    Mutations respectant les blocs de 2 cases :
    - ne casse pas la grille 16
    - chaque bloc reste à sa place
    - on mute seulement à l'intérieur d'un bloc
    - parfois on remplace un bloc par un bloc du même rôle dominant
    """
    pairs = [block[:] for block in base_pairs]

    for i, block in enumerate(pairs):
        if not block:
            continue

        if random.random() > mutation:
            continue

        action = random.choice([
            "repeat_inside",
            "swap_inside",
            "same_pair_role",
            "light_reverse",
            "keep",
        ])

        if action == "repeat_inside" and len(block) >= 1:
            src = random.choice(block)
            if len(block) >= 2:
                j = random.randrange(len(block))
                block[j] = src

        elif action == "swap_inside" and len(block) >= 2:
            a, b = random.sample(range(len(block)), 2)
            block[a], block[b] = block[b], block[a]

        elif action == "same_pair_role":
            # Remplace un event par un autre event du même rôle, mais dans le même bloc de 2 cases si possible.
            j = random.randrange(len(block))
            old = block[j]
            role = slices[old]["role"]
            candidates = [
                s["index"] for s in slices
                if s["role"] == role and s["pair"] == i
            ]
            if candidates:
                block[j] = random.choice(candidates)

        elif action == "light_reverse" and len(block) >= 2:
            block.reverse()

        pairs[i] = block

    # Fill de fin : uniquement sur le dernier bloc de 2 cases
    if random.random() < mutation:
        i = 7
        block = pairs[i]
        if block:
            mode = random.choice(["repeat", "reverse"])
            if mode == "repeat":
                src = random.choice(block)
                pairs[i] = [src for _ in block]
            elif mode == "reverse":
                pairs[i] = list(reversed(block))

    return pairs


def render_sequence(slices, seq):
    chunks = [slices[idx]["audio"] for idx in seq]
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return normalize(np.concatenate(chunks))


def role_letter(role):
    return {"hat": "H", "kick": "K", "snare": "S"}.get(role, "?")


def sequence_text(seq, slices):
    lines = []
    lines.append("STEP | SLICE | ROLE  | GRID16 | PAIR | DUR")
    lines.append("--------------------------------------------")

    role_grid = [["." for _ in range(16)] for _ in range(3)]
    lane_index = {"hat": 0, "kick": 1, "snare": 2}

    for step, idx in enumerate(seq, start=1):
        s = slices[idx]
        role = s["role"]
        g = s["grid16"]
        if role in lane_index:
            role_grid[lane_index[role]][g] = role_letter(role)

        lines.append(
            f"{step:04d} | {idx:05d} | {role:5} | "
            f"{g:06d} | {s['pair']:04d} | {s['duration']:.3f}s"
        )

    lines.append("")
    lines.append("GRID 16:")
    lines.append("PAIR : 11|22|33|44|55|66|77|88")
    lines.append("STEP : 12|34|56|78|90|12|34|56")
    lines.append("HAT  : " + "|".join("".join(role_grid[0][i:i+2]) for i in range(0, 16, 2)))
    lines.append("KICK : " + "|".join("".join(role_grid[1][i:i+2]) for i in range(0, 16, 2)))
    lines.append("SNARE: " + "|".join("".join(role_grid[2][i:i+2]) for i in range(0, 16, 2)))

    lines.append("")
    lines.append("ROLE SEQUENCE:")
    lines.append(" ".join(role_letter(slices[idx]["role"]) for idx in seq))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Camo")
    parser.add_argument("--image", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--mutation", type=float, default=0.18)
    parser.add_argument("--slice-end", choices=["next", "rect"], default="next")
    parser.add_argument("--min-ms", type=int, default=35)
    parser.add_argument("--tail-ms", type=int, default=15)
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

    events, meta = detect_events(image_path)
    events = assign_to_16_grid(events, meta["x_min"], meta["x_max"])
    slices = extract_slices(
        y,
        events,
        meta["x_min"],
        meta["x_max"],
        slice_end=args.slice_end,
        min_ms=args.min_ms,
        tail_ms=args.tail_ms,
    )

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)

    base_seq = base_sequence(slices)
    base_pairs = split_into_pairs(base_seq, slices)

    base_audio = render_sequence(slices, base_seq)
    base_wav = outdir / f"{safe}_base_grid16_pairs.wav"
    sf.write(base_wav, base_audio, SR)

    (outdir / "base_grid16_pairs.txt").write_text(sequence_text(base_seq, slices), encoding="utf-8")

    print(f"Events détectés : {len(events)}")
    print("Base :", base_wav)

    renders = []

    for i in range(1, args.count + 1):
        mutated_pairs = mutate_pairs(base_pairs, slices, mutation=args.mutation)
        seq = flatten_pairs(mutated_pairs)

        audio = render_sequence(slices, seq)

        wav = outdir / f"{safe}_grid16_pair_variation_{i:03d}.wav"
        txt = outdir / f"{safe}_grid16_pair_variation_{i:03d}.txt"

        sf.write(wav, audio, SR)
        txt.write_text(sequence_text(seq, slices), encoding="utf-8")

        renders.append({
            "index": i,
            "wav": str(wav),
            "sequence": [int(x) for x in seq],
        })

        print("Export :", wav)

    json_safe = {
        "version": "tracker_from_image_v02_grid16_pairs",
        "source": str(source),
        "image": str(image_path),
        "source_only": True,
        "no_overlap": True,
        "grid": "16 steps, mutations by pairs of 2 steps",
        "slice_end": args.slice_end,
        "event_count": len(events),
        "base_wav": str(base_wav),
        "image_meta": {
            "x_min": int(meta["x_min"]),
            "x_max": int(meta["x_max"]),
            "lanes": {k: [int(v[0]), int(v[1])] for k, v in meta["lanes"].items()},
            "image_size": [int(meta["image_size"][0]), int(meta["image_size"][1])],
        },
        "slices": [
            {
                "index": int(s["index"]),
                "role": s["role"],
                "grid16": int(s["grid16"]),
                "pair": int(s["pair"]),
                "x1": int(s["x1"]),
                "x2": int(s["x2"]),
                "start_sample": int(s["start_sample"]),
                "end_sample": int(s["end_sample"]),
                "duration": float(s["duration"]),
            }
            for s in slices
        ],
        "renders": renders,
    }

    (outdir / "metadata_grid16_pairs.json").write_text(json.dumps(json_safe, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("À écouter d'abord :", base_wav)


if __name__ == "__main__":
    main()
