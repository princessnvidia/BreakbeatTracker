#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracker From Image v04 - image locked

Correction :
- l'image redevient la référence absolue
- le pattern vient de l'image, pas d'un auto-cut
- les events restent à leur place sur la grille 16
- les blocs de 2 cases restent respectés
- les variations ne changent PAS la structure du pattern
- elles remplacent seulement une slice par une autre slice du même rôle,
  idéalement dans le même bloc de 2 cases
- aucune superposition
- une seule source audio

Découpe :
- chaque rectangle rose de l'image devient une slice
- x1/x2 du rectangle sont mappés vers le WAV source
- si la découpe audio est décalée, ajuste :
      --audio-start-ms
      --audio-end-ms

Usage :
    python tracker_from_image_v04.py --source "Camo" --image camobreak.png

Si le début audio est trop tôt :
    python tracker_from_image_v04.py --source "Camo" --image camobreak.png --audio-start-ms 40

Si la zone utile du break finit avant la fin du fichier :
    python tracker_from_image_v04.py --source "Camo" --image camobreak.png --audio-start-ms 20 --audio-end-ms 1800
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
OUT_DIR = Path("exports/tracker_from_image_v04")

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
        & (g > 60)
        & (g < 195)
        & (b > 80)
        & (b < 225)
        & (r > g + 25)
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
        e["index"] = int(i)
        e["grid16"] = int(grid16)
        e["pair"] = int(grid16 // 2)
        e["x_norm"] = float(pos)

    return events, {
        "x_min": int(x_min),
        "x_max": int(x_max),
        "lanes": lanes,
        "image_size": img.size,
    }


def extract_slices_image_locked(y, events, x_min, x_max, audio_start_ms=0.0, audio_end_ms=None, min_ms=35, tail_ms=10):
    """
    Découpe verrouillée sur l'image :
    - les rectangles déterminent le start/end
    - aucun auto-cut
    - audio_start_ms/audio_end_ms permettent de caler la zone audio utile
    """
    audio_start = int(SR * audio_start_ms / 1000.0)

    if audio_end_ms is None:
        audio_end = len(y)
    else:
        audio_end = int(SR * audio_end_ms / 1000.0)

    audio_start = max(0, min(len(y) - 1, audio_start))
    audio_end = max(audio_start + 1, min(len(y), audio_end))

    audio_span = audio_end - audio_start
    image_span = max(1, x_max - x_min)

    slices = []

    for e in events:
        rel_start = (e["x1"] - x_min) / image_span
        rel_end = (e["x2"] - x_min) / image_span

        start = audio_start + int(rel_start * audio_span)
        end = audio_start + int(rel_end * audio_span)

        end += int(SR * tail_ms / 1000.0)

        min_len = int(SR * min_ms / 1000.0)
        if end < start + min_len:
            end = start + min_len

        start = max(0, min(len(y) - 1, start))
        end = max(start + 1, min(len(y), end))

        audio = fade(y[start:end].copy(), ms=2)

        slices.append({
            "index": int(e["index"]),
            "role": e["role"],
            "grid16": int(e["grid16"]),
            "pair": int(e["pair"]),
            "x1": int(e["x1"]),
            "x2": int(e["x2"]),
            "start_sample": int(start),
            "end_sample": int(end),
            "start_ms": float(start / SR * 1000),
            "end_ms": float(end / SR * 1000),
            "duration": float((end - start) / SR),
            "audio": audio,
        })

    return slices


def base_sequence(slices):
    return list(range(len(slices)))


def build_candidates(slices):
    by_role = {"hat": [], "kick": [], "snare": []}
    by_role_pair = {}

    for s in slices:
        by_role[s["role"]].append(s["index"])
        key = (s["role"], s["pair"])
        by_role_pair.setdefault(key, []).append(s["index"])

    return by_role, by_role_pair


def mutate_image_locked(base_seq, slices, mutation=0.12):
    """
    Mutations qui ne cassent pas l'image :
    - même nombre d'events
    - même rôle affiché à la même position
    - même grid16
    - remplacement de slice uniquement par même rôle
    - priorité au même bloc de 2 cases
    """
    by_role, by_role_pair = build_candidates(slices)

    seq = base_seq[:]

    for pos, idx in enumerate(seq):
        if random.random() > mutation:
            continue

        s = slices[idx]
        role = s["role"]
        pair = s["pair"]

        same_pair = by_role_pair.get((role, pair), [])
        same_role = by_role.get(role, [])

        if same_pair and random.random() < 0.80:
            seq[pos] = random.choice(same_pair)
        elif same_role:
            seq[pos] = random.choice(same_role)

    # Fill final très léger, mais toujours même rôle
    if random.random() < mutation:
        for pos in range(max(0, len(seq) - 6), len(seq)):
            idx = seq[pos]
            s = slices[idx]
            role = s["role"]
            pair = s["pair"]
            candidates = by_role_pair.get((role, pair), [])
            if candidates and random.random() < 0.50:
                seq[pos] = random.choice(candidates)

    return seq


def render_sequence(slices, seq):
    chunks = [slices[idx]["audio"] for idx in seq]
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return normalize(np.concatenate(chunks))


def role_letter(role):
    return {"hat": "H", "kick": "K", "snare": "S"}.get(role, "?")


def sequence_text(seq, slices):
    lines = []
    lines.append("EVENT | SLICE | ROLE  | GRID16 | PAIR | START_MS | END_MS | DUR")
    lines.append("---------------------------------------------------------------")

    rows = {
        "hat": ["."] * 16,
        "kick": ["."] * 16,
        "snare": ["."] * 16,
    }

    for event_pos, idx in enumerate(seq):
        s = slices[idx]
        rows[s["role"]][s["grid16"]] = role_letter(s["role"])

        lines.append(
            f"{event_pos:05d} | {idx:05d} | {s['role']:5} | "
            f"{s['grid16']:06d} | {s['pair']:04d} | "
            f"{s['start_ms']:8.1f} | {s['end_ms']:7.1f} | {s['duration']:.3f}s"
        )

    lines.append("")
    lines.append("IMAGE-LOCKED GRID16:")
    lines.append("PAIR : 11|22|33|44|55|66|77|88")
    lines.append("STEP : 12|34|56|78|90|12|34|56")
    lines.append("HAT  : " + "|".join("".join(rows["hat"][i:i+2]) for i in range(0, 16, 2)))
    lines.append("KICK : " + "|".join("".join(rows["kick"][i:i+2]) for i in range(0, 16, 2)))
    lines.append("SNARE: " + "|".join("".join(rows["snare"][i:i+2]) for i in range(0, 16, 2)))
    lines.append("")
    lines.append("ROLE SEQUENCE:")
    lines.append(" ".join(role_letter(slices[idx]["role"]) for idx in seq))

    return "\n".join(lines)


def export_preview(slices, outdir):
    p = outdir / "preview_image_locked_slices"
    p.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("IDX | ROLE  | GRID16 | PAIR | START_MS | END_MS | FILE")
    lines.append("-------------------------------------------------------")

    for s in slices:
        wav = p / f"slice_{s['index']:03d}_{s['role']}_g{s['grid16']:02d}_p{s['pair']}.wav"
        sf.write(wav, normalize(s["audio"]), SR)

        lines.append(
            f"{s['index']:03d} | {s['role']:5} | {s['grid16']:06d} | {s['pair']:04d} | "
            f"{s['start_ms']:8.1f} | {s['end_ms']:7.1f} | {wav}"
        )

    (p / "preview_image_locked_slices.txt").write_text("\n".join(lines), encoding="utf-8")
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Camo")
    parser.add_argument("--image", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--mutation", type=float, default=0.12)
    parser.add_argument("--audio-start-ms", type=float, default=0.0)
    parser.add_argument("--audio-end-ms", type=float, default=None)
    parser.add_argument("--min-ms", type=float, default=35.0)
    parser.add_argument("--tail-ms", type=float, default=10.0)
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

    events, meta = detect_events(image_path)
    slices = extract_slices_image_locked(
        y,
        events,
        meta["x_min"],
        meta["x_max"],
        audio_start_ms=args.audio_start_ms,
        audio_end_ms=args.audio_end_ms,
        min_ms=args.min_ms,
        tail_ms=args.tail_ms,
    )

    safe = source.stem.replace(" ", "_").replace("'", "")
    suffix = f"start_{int(args.audio_start_ms)}ms"
    if args.audio_end_ms is not None:
        suffix += f"_end_{int(args.audio_end_ms)}ms"

    outdir = OUT_DIR / safe / suffix
    outdir.mkdir(parents=True, exist_ok=True)

    preview_dir = export_preview(slices, outdir)

    if args.preview:
        print("Preview :", preview_dir)
        return

    base_seq = base_sequence(slices)
    base_audio = render_sequence(slices, base_seq)
    base_wav = outdir / f"{safe}_image_locked_base.wav"
    sf.write(base_wav, base_audio, SR)

    (outdir / "image_locked_base.txt").write_text(sequence_text(base_seq, slices), encoding="utf-8")

    print(f"Events détectés : {len(events)}")
    print("Base :", base_wav)

    renders = []

    for i in range(1, args.count + 1):
        seq = mutate_image_locked(base_seq, slices, mutation=args.mutation)
        audio = render_sequence(slices, seq)

        wav = outdir / f"{safe}_image_locked_variation_{i:03d}.wav"
        txt = outdir / f"{safe}_image_locked_variation_{i:03d}.txt"

        sf.write(wav, audio, SR)
        txt.write_text(sequence_text(seq, slices), encoding="utf-8")

        renders.append({
            "index": int(i),
            "wav": str(wav),
            "sequence": [int(x) for x in seq],
        })

        print("Export :", wav)

    metadata = {
        "version": "tracker_from_image_v04_image_locked",
        "source": str(source),
        "image": str(image_path),
        "source_only": True,
        "no_overlap": True,
        "image_is_reference": True,
        "grid": "16 cases, 8 pairs of 2 cases",
        "mutation_rule": "same role, same pair preferred, no structural move",
        "audio_start_ms": float(args.audio_start_ms),
        "audio_end_ms": None if args.audio_end_ms is None else float(args.audio_end_ms),
        "min_ms": float(args.min_ms),
        "tail_ms": float(args.tail_ms),
        "event_count": int(len(events)),
        "base_wav": str(base_wav),
        "preview_dir": str(preview_dir),
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
                "start_ms": float(s["start_ms"]),
                "end_ms": float(s["end_ms"]),
                "duration": float(s["duration"]),
            }
            for s in slices
        ],
        "renders": renders,
    }

    (outdir / "metadata_image_locked.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("Preview :", preview_dir)
    print("À écouter d'abord :", base_wav)


if __name__ == "__main__":
    main()
