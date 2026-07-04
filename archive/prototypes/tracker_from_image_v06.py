#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracker From Image v06 - pair chunks mutations

Correction importante :
- La base image-locked est reconstruite.
- Ensuite elle est découpée en 8 BLOCS audio fixes de 2 cases.
- Les variations ne touchent PLUS aux events individuels.
- Les variations réordonnent / répètent / inversent uniquement ces blocs 2 par 2.

Donc :
    base = pair0 + pair1 + pair2 + ... + pair7
    variation = pair0 + pair1 + pair1 + pair3 + pair4 + pair5 + pair6 + pair7

La découpe reste la même sur tous les WAV.
Aucune superposition.
Une seule source audio.
L'image reste la référence.

Usage :
    python tracker_from_image_v06.py --source "Camo" --image camobreak.png

Plus de variations :
    python tracker_from_image_v06.py --source "Camo" --image camobreak.png --count 64 --mutation 0.25

Très stable :
    python tracker_from_image_v06.py --source "Camo" --image camobreak.png --mutation 0.10
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
OUT_DIR = Path("exports/tracker_from_image_v06")
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
        sys.exit(1)

    return {"hat": selected[0], "kick": selected[1], "snare": selected[2]}


def detect_events(image_path):
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


def extract_base_slices(y, events, x_min, x_max, audio_start_ms=0.0, audio_end_ms=None, min_ms=35, tail_ms=10):
    audio_start = int(SR * audio_start_ms / 1000.0)
    audio_end = len(y) if audio_end_ms is None else int(SR * audio_end_ms / 1000.0)

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
            "start_ms": float(start / SR * 1000),
            "end_ms": float(end / SR * 1000),
            "duration": float((end - start) / SR),
            "audio": audio,
        })

    return slices


def base_sequence(slices):
    return list(range(len(slices)))


def render_sequence(slices, seq):
    chunks = [slices[idx]["audio"] for idx in seq]
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return normalize(np.concatenate(chunks))


def build_pair_chunks_from_base(slices, base_seq):
    """
    Rend la base par paires, puis stocke 8 chunks audio.
    Les variations utiliseront ces chunks.
    """
    pair_chunks = []

    for pair in range(8):
        indices = [idx for idx in base_seq if slices[idx]["pair"] == pair]

        if not indices:
            # silence très court si paire vide
            audio = np.zeros(int(SR * 0.05), dtype=np.float32)
            roles = []
        else:
            audio = render_sequence(slices, indices)
            roles = [slices[idx]["role"] for idx in indices]

        pair_chunks.append({
            "pair": pair,
            "indices": indices,
            "roles": roles,
            "audio": fade(audio, ms=2),
            "duration": float(len(audio) / SR),
        })

    return pair_chunks


def render_pairs(pair_chunks, pair_order):
    chunks = [pair_chunks[pair]["audio"] for pair in pair_order]
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return normalize(np.concatenate(chunks))


def mutate_pair_order(mutation=0.18, fill_chance=0.20):
    """
    La timeline est toujours 8 blocs.
    On mute les numéros de blocs, pas les events.
    """
    order = list(range(8))

    for i in range(8):
        if random.random() > mutation:
            continue

        action = random.choice([
            "repeat_prev",
            "repeat_next",
            "neighbor",
            "same",
        ])

        if action == "repeat_prev" and i > 0:
            order[i] = order[i - 1]

        elif action == "repeat_next" and i < 7:
            order[i] = order[i + 1]

        elif action == "neighbor":
            order[i] = max(0, min(7, order[i] + random.choice([-1, 1])))

    # Fill final uniquement sur les deux derniers blocs
    if random.random() < fill_chance:
        mode = random.choice(["repeat_6", "repeat_7", "swap_67", "reverse_last4"])

        if mode == "repeat_6":
            order[7] = order[6]
        elif mode == "repeat_7":
            order[6] = order[7]
        elif mode == "swap_67":
            order[6], order[7] = order[7], order[6]
        elif mode == "reverse_last4":
            order[4:8] = list(reversed(order[4:8]))

    return order


def role_letter(role):
    return {"hat": "H", "kick": "K", "snare": "S"}.get(role, "?")


def pair_chunks_text(pair_chunks):
    lines = []
    lines.append("PAIR CHUNKS FROM BASE")
    lines.append("PAIR | DUR | SLICE INDICES | ROLES")
    lines.append("-----------------------------------")

    for c in pair_chunks:
        lines.append(
            f"{c['pair']:04d} | {c['duration']:.3f}s | "
            f"{' '.join(str(x) for x in c['indices'])} | "
            f"{' '.join(c['roles'])}"
        )

    lines.append("")
    lines.append("GRID 16 PAIRS:")
    lines.append("PAIR : 11|22|33|44|55|66|77|88")

    return "\n".join(lines)


def order_text(order, pair_chunks):
    lines = []
    lines.append("PAIR ORDER:")
    lines.append(" ".join(str(x) for x in order))
    lines.append("")

    lines.append("BLOCK DETAILS:")
    for pos, pair in enumerate(order):
        c = pair_chunks[pair]
        lines.append(
            f"POS {pos:02d} -> PAIR {pair} | dur={c['duration']:.3f}s | "
            f"indices={' '.join(str(x) for x in c['indices'])} | roles={' '.join(c['roles'])}"
        )

    return "\n".join(lines)


def export_pair_preview(pair_chunks, outdir):
    p = outdir / "preview_pair_chunks"
    p.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("PAIR | DUR | FILE")
    lines.append("-----------------")

    for c in pair_chunks:
        wav = p / f"pair_chunk_{c['pair']:02d}.wav"
        sf.write(wav, normalize(c["audio"]), SR)
        lines.append(f"{c['pair']:04d} | {c['duration']:.3f}s | {wav}")

    (p / "preview_pair_chunks.txt").write_text("\n".join(lines), encoding="utf-8")
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Camo")
    parser.add_argument("--image", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--mutation", type=float, default=0.18)
    parser.add_argument("--fill-chance", type=float, default=0.20)
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
    image_path = Path(args.image).expanduser()

    if not image_path.exists():
        print("Image introuvable :", image_path)
        sys.exit(1)

    print("Source unique :", source)
    print("Image tracker :", image_path)

    y, sr = load_audio(source)
    events, meta = detect_events(image_path)
    slices = extract_base_slices(
        y,
        events,
        meta["x_min"],
        meta["x_max"],
        audio_start_ms=args.audio_start_ms,
        audio_end_ms=args.audio_end_ms,
        min_ms=args.min_ms,
        tail_ms=args.tail_ms,
    )

    base_seq = base_sequence(slices)
    base_audio = render_sequence(slices, base_seq)
    pair_chunks = build_pair_chunks_from_base(slices, base_seq)

    safe = source.stem.replace(" ", "_").replace("'", "")
    suffix = f"start_{int(args.audio_start_ms)}ms"
    if args.audio_end_ms is not None:
        suffix += f"_end_{int(args.audio_end_ms)}ms"

    outdir = OUT_DIR / safe / suffix
    outdir.mkdir(parents=True, exist_ok=True)

    base_wav = outdir / f"{safe}_pairchunks_base.wav"
    sf.write(base_wav, base_audio, SR)

    pair_preview = export_pair_preview(pair_chunks, outdir)

    (outdir / "pair_chunks_from_base.txt").write_text(pair_chunks_text(pair_chunks), encoding="utf-8")

    if args.preview:
        print("Base :", base_wav)
        print("Preview pair chunks :", pair_preview)
        return

    print(f"Events détectés : {len(events)}")
    print("Base :", base_wav)
    print("Pair chunks :", pair_preview)

    renders = []

    for i in range(1, args.count + 1):
        order = mutate_pair_order(
            mutation=args.mutation,
            fill_chance=args.fill_chance,
        )

        audio = render_pairs(pair_chunks, order)

        wav = outdir / f"{safe}_pairchunks_variation_{i:03d}.wav"
        txt = outdir / f"{safe}_pairchunks_variation_{i:03d}.txt"

        sf.write(wav, audio, SR)
        txt.write_text(order_text(order, pair_chunks), encoding="utf-8")

        renders.append({
            "index": int(i),
            "wav": str(wav),
            "pair_order": [int(x) for x in order],
        })

        print("Export :", wav)

    metadata = {
        "version": "tracker_from_image_v06_pair_chunks",
        "source": str(source),
        "image": str(image_path),
        "source_only": True,
        "same_cuts_as_base": True,
        "no_overlap": True,
        "variation_unit": "2-case pair chunks rendered from base",
        "mutation_rule": "reorder/repeat pair chunks only; no individual event slicing",
        "audio_start_ms": float(args.audio_start_ms),
        "audio_end_ms": None if args.audio_end_ms is None else float(args.audio_end_ms),
        "event_count": int(len(events)),
        "base_wav": str(base_wav),
        "pair_preview": str(pair_preview),
        "pair_chunks": [
            {
                "pair": int(c["pair"]),
                "indices": [int(x) for x in c["indices"]],
                "roles": list(c["roles"]),
                "duration": float(c["duration"]),
            }
            for c in pair_chunks
        ],
        "renders": renders,
    }

    (outdir / "metadata_pairchunks.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("À écouter d'abord :", base_wav)


if __name__ == "__main__":
    main()
