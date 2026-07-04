#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracker From Image v01

But :
Utiliser une image de découpe tracker comme celle que tu as montrée :
- ligne du haut  = hats
- ligne du milieu = kicks
- ligne du bas = snares
- blocs roses = événements successifs
- aucun sample ne se superpose
- on ordonne les blocs de gauche à droite
- on extrait les slices depuis UN SEUL break source
- on génère des variations en réordonnant ces slices

Entrées :
    breaks/Camo Break - 3A.wav
    image de découpe, ex: camobreak.png

Sorties :
    exports/tracker_from_image_v01/<source>/
        *_base_from_image.wav
        *_variation_001.wav
        pattern_from_image.txt
        metadata_from_image.json

Usage :
    python tracker_from_image_v01.py --source "Camo" --image camobreak.png

Si ton image est dans Téléchargements :
    python tracker_from_image_v01.py --source "Camo" --image ~/Téléchargements/camobreak.png

Variations :
    python tracker_from_image_v01.py --source "Camo" --image camobreak.png --count 32 --mutation 0.20

Mode de fin de slice :
    --slice-end next   = slice jusqu'au prochain bloc
    --slice-end rect   = slice seulement sur la largeur du rectangle
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
OUT_DIR = Path("exports/tracker_from_image_v01")
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

    # blocs roses Ableton-like
    mask = (
        (r > 170)
        & (g > 70)
        & (g < 180)
        & (b > 90)
        & (b < 210)
        & (r > g + 35)
    )

    return mask


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

    # On garde les 3 bandes les plus importantes, triées verticalement.
    band_scores = []
    for y1, y2 in bands:
        score = int(row_strength[y1:y2 + 1].sum())
        band_scores.append((score, y1, y2))

    band_scores.sort(reverse=True)
    selected = sorted([(y1, y2) for _, y1, y2 in band_scores[:3]], key=lambda x: x[0])

    if len(selected) < 3:
        print("Je n'ai pas trouvé 3 pistes dans l'image.")
        print("Bandes trouvées :", selected)
        sys.exit(1)

    return {
        "hat": selected[0],
        "kick": selected[1],
        "snare": selected[2],
    }


def detect_events_from_image(image_path):
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

    # timeline visuelle utile
    x_min = min(e["x1"] for e in events)
    x_max = max(e["x2"] for e in events)

    for i, e in enumerate(events):
        e["index"] = i
        e["x_norm"] = (e["x1"] - x_min) / max(1, x_max - x_min)

    return events, {"x_min": x_min, "x_max": x_max, "lanes": lanes, "image_size": img.size}


def extract_slices_from_events(y, events, x_min, x_max, slice_end="next", min_ms=35, tail_ms=20):
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

        audio = y[start:end].copy()
        audio = fade(audio, ms=2)

        slices.append({
            "index": i,
            "role": e["role"],
            "x1": e["x1"],
            "x2": e["x2"],
            "start_sample": int(start),
            "end_sample": int(end),
            "duration": float((end - start) / SR),
            "audio": audio,
        })

    return slices


def render_sequence(slices, sequence):
    chunks = []
    for idx in sequence:
        chunks.append(slices[idx % len(slices)]["audio"])
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return normalize(np.concatenate(chunks))


def original_sequence(slices):
    return list(range(len(slices)))


def mutate_sequence(seq, slices, mutation=0.18, fill_chance=0.25):
    out = seq[:]
    n = len(out)

    for i in range(n):
        if random.random() > mutation:
            continue

        action = random.choice([
            "repeat_prev",
            "repeat_next",
            "swap_neighbor",
            "jump_same_role",
            "local_jump",
            "keep",
        ])

        if action == "repeat_prev" and i > 0:
            out[i] = out[i - 1]

        elif action == "repeat_next" and i < n - 1:
            out[i] = out[i + 1]

        elif action == "swap_neighbor":
            j = max(0, min(n - 1, i + random.choice([-1, 1])))
            out[i], out[j] = out[j], out[i]

        elif action == "jump_same_role":
            role = slices[out[i]]["role"]
            candidates = [s["index"] for s in slices if s["role"] == role]
            if candidates:
                out[i] = random.choice(candidates)

        elif action == "local_jump":
            out[i] = max(0, min(n - 1, out[i] + random.choice([-3, -2, -1, 1, 2, 3])))

    # fill fin de boucle façon tracker
    if n >= 8 and random.random() < fill_chance:
        start = n - random.choice([4, 6, 8])
        start = max(0, start)
        mode = random.choice(["repeat", "same_role_chop", "reverse"])

        if mode == "repeat":
            src = out[start]
            for i in range(start, n):
                if random.random() < 0.70:
                    out[i] = src

        elif mode == "same_role_chop":
            roles = [slices[out[i]]["role"] for i in range(start, n)]
            for i in range(start, n):
                role = random.choice(roles)
                candidates = [s["index"] for s in slices if s["role"] == role]
                if candidates:
                    out[i] = random.choice(candidates)

        elif mode == "reverse":
            out[start:n] = list(reversed(out[start:n]))

    return out


def role_letter(role):
    return {"hat": "H", "kick": "K", "snare": "S"}.get(role, "?")


def sequence_text(seq, slices):
    lines = []
    lines.append("STEP | SLICE | ROLE  | DUR")
    lines.append("----------------------------")

    compact = []

    for step, idx in enumerate(seq, start=1):
        s = slices[idx % len(slices)]
        compact.append(role_letter(s["role"]))
        lines.append(f"{step:04d} | {idx:05d} | {s['role']:5} | {s['duration']:.3f}s")

    lines.append("")
    lines.append("ROLE SEQUENCE:")
    lines.append(" ".join(compact))

    return "\n".join(lines)


def report_events(events):
    lines = []
    lines.append("DETECTED EVENTS FROM IMAGE")
    lines.append("IDX | ROLE  | X1-X2")
    lines.append("-------------------")
    for e in events:
        lines.append(f"{e['index']:03d} | {e['role']:5} | {e['x1']:04d}-{e['x2']:04d}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Camo")
    parser.add_argument("--image", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--mutation", type=float, default=0.18)
    parser.add_argument("--fill-chance", type=float, default=0.25)
    parser.add_argument("--slice-end", choices=["next", "rect"], default="next")
    parser.add_argument("--min-ms", type=int, default=35)
    parser.add_argument("--tail-ms", type=int, default=20)
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

    events, image_meta = detect_events_from_image(image_path)
    slices = extract_slices_from_events(
        y,
        events,
        image_meta["x_min"],
        image_meta["x_max"],
        slice_end=args.slice_end,
        min_ms=args.min_ms,
        tail_ms=args.tail_ms,
    )

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)

    base_seq = original_sequence(slices)
    base_audio = render_sequence(slices, base_seq)

    base_wav = outdir / f"{safe}_base_from_image.wav"
    sf.write(base_wav, base_audio, SR)

    (outdir / "pattern_from_image.txt").write_text(sequence_text(base_seq, slices), encoding="utf-8")
    (outdir / "events_from_image.txt").write_text(report_events(events), encoding="utf-8")

    print(f"Events détectés : {len(events)}")
    print("Base :", base_wav)

    renders = []

    for i in range(1, args.count + 1):
        seq = mutate_sequence(
            base_seq,
            slices,
            mutation=args.mutation,
            fill_chance=args.fill_chance,
        )

        audio = render_sequence(slices, seq)

        wav = outdir / f"{safe}_tracker_image_variation_{i:03d}.wav"
        txt = outdir / f"{safe}_tracker_image_variation_{i:03d}.txt"

        sf.write(wav, audio, SR)
        txt.write_text(sequence_text(seq, slices), encoding="utf-8")

        renders.append({
            "index": i,
            "wav": str(wav),
            "sequence": seq,
        })

        print("Export :", wav)

    metadata = {
        "version": "tracker_from_image_v01",
        "source": str(source),
        "image": str(image_path),
        "source_only": True,
        "no_overlap": True,
        "logic": "events detected from tracker image; audio slices extracted from same source and concatenated",
        "slice_end": args.slice_end,
        "event_count": len(events),
        "image_meta": {
            "x_min": image_meta["x_min"],
            "x_max": image_meta["x_max"],
            "lanes": {k: list(v) for k, v in image_meta["lanes"].items()},
            "image_size": list(image_meta["image_size"]),
        },
        "base_wav": str(base_wav),
        "events": events,
        "slices": [
            {
                "index": s["index"],
                "role": s["role"],
                "x1": s["x1"],
                "x2": s["x2"],
                "start_sample": s["start_sample"],
                "end_sample": s["end_sample"],
                "duration": s["duration"],
            }
            for s in slices
        ],
        "renders": renders,
    }

    (outdir / "metadata_from_image.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("À écouter d'abord :", base_wav)


if __name__ == "__main__":
    main()
